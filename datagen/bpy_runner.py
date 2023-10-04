
import os, sys, bpy, time, glob, random
import numpy as np

sys.path.append('/home/beans/bespoke')
sys.path.append('/home/beans/bespoke/datagen')
from datagen.episode import *
from bpy_handler import *
from constants import *

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--" ['example', 'args', '123']
dataloader_id = argv[0]
dataloader_root = f"{BLENDER_MEMBANK_ROOT}/dataloader_{dataloader_id}"

FAILED_TO_GET_MAP_F = f"{dataloader_root}/failed_to_get_map.npy"
RUN_COUNTER_F = f"{dataloader_root}/run_counter.npy" # TODO bring this out into constant

import tracemalloc

import linecache, os

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def indicate_failed():
    # Deal w failed map/route
    print("wtf couldn't get good map? Killing runner")
    np.save(FAILED_TO_GET_MAP_F, np.array(1))
    set_should_stop(True) # if one runner dies, shutdown all runners to make it obvious. Perhaps in the future we can tolerate this, but not now

TRACE_MALLOC = False # finding memleak
if __name__ == "__main__":
    if TRACE_MALLOC: tracemalloc.start()

    print(f"Launching runner {dataloader_id}")
    
    reset_npc_objects(bpy) # Delete existing npc copies and make new ones. To ensure up to date w any changes made in master.

    # If picking up from where we left off, start from last run not from zero
    # incrememnted only after successful run
    if os.path.exists(RUN_COUNTER_F):
        run_counter = np.load(RUN_COUNTER_F)[0]
        print(f"Resuming from run {run_counter}")
    else:
        run_counter = 0

    failed_counter = 0

    need_map = True
    start_left = True

    # bpy.context.window.scene = bpy.data.scenes['Scene'] #TODO remove prob

    while True:

        # Initial setup
        timer = Timer("*** gather run total ***")
        t0 = time.time()
        run_root = f"{dataloader_root}/run_{run_counter}"
        for s in ["aux", "targets", "maps"]: os.makedirs(f"{run_root}/{s}", exist_ok=True)

        print(f"Making new map: {need_map}. Starting from left: {start_left}.")
        print(f"Dataloader {dataloader_id}, run {run_counter}")

        # Get map. We can do multiple routes through each map
        if need_map:
            # Make episode map
            episode_info = make_map(timer) # updates dg w new rd network

            # Randomize appearance -- doesn't alter targets
            randomize_appearance(timer, episode_info, run_counter) # updates dg w new materials

            # Retrieve map
            wp_df, coarse_map_df, success = get_map_data(bpy, episode_info, timer) # dg evaluated here to retrieve rd network wps
            if not success:
                failed_counter += 1
                if failed_counter == 20:
                    indicate_failed()
                    break
                continue

        # Get route. Currently two routes for each map (one in each direction)
        ego_route = get_ego_route(wp_df, episode_info, start_left) # can fail here
        if ego_route is None:
            failed_counter += 1
            if failed_counter == 20:
                indicate_failed()
                break
            need_map = True
            start_left = True
            continue
            
        start_left = not start_left
        need_map = not need_map

        # Toggle bev and semseg
        save_depth(bpy, True)
        toggle_semseg(bpy, False) # updates dg w new materials
        toggle_bev(bpy, False, pitch_perturbation=episode_info.pitch)
        # toggle_semseg(bpy, True) # updates dg w new materials
        # toggle_bev(bpy, True)
        # bpy.data.scenes["Scene"].node_tree.nodes["Switch"].check = False # rgb, not depth

        get_node("semseg_out", bpy.data.scenes["Scene"].node_tree.nodes).base_path = f"{run_root}/imgs_semseg/"
        get_node("depth_out", bpy.data.scenes["Scene"].node_tree.nodes).base_path = f"{run_root}/imgs_depth/"
        # get_node("normals_out", bpy.data.scenes["Scene"].node_tree.nodes).base_path = f"{run_root}/imgs_normals/"

        # We've successfully gotten map and route. May continue
        failed_counter = 0

        # Create AP and TM  
        ap, tm = create_ap_tm(bpy, wp_df, coarse_map_df, ego_route, episode_info, timer, run_root=run_root,
                              dataloader_id=dataloader_id, run_id=run_counter) # updates dg w new ego pos

        # Reset scene
        reset_scene(bpy, ap, tm, timer=timer, save_data=True, render_filepath=f"{run_root}/imgs/") # dg not evaluated here, will be eval on first frame of render prob
        timer.log("reset scene")

        init_time = time.time() - t0
        report_runner_metric(dataloader_root, init_time, INIT_TIME_F)
        _t0 = time.time()
        ##################
        # Render

        # redirect output to log file. From https://blender.stackexchange.com/questions/44560/how-to-supress-bpy-render-messages-in-terminal-output
        # Render too verbose, makes it impossible to debug anything in the terminal
        logfile = f'{run_root}/render.log'
        if os.path.exists(logfile): os.remove(logfile)
        open(logfile, 'a').close()
        old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        fd = os.open(logfile, os.O_WRONLY)
        timer.log("redirect stdout")

        bpy.ops.render.render(animation=True)
        timer.log("render")

        # render_bev = True
        # if render_bev:
        #     save_depth(bpy, False)
        #     toggle_bev(bpy, True, pitch_perturbation=episode_info.pitch)
        #     toggle_semseg(bpy, True) # updates dg w new materials
        #     reset_ap_tm(bpy, ap, tm)
        #     reset_scene(bpy, ap, tm, timer=timer, save_data=False, render_filepath=f"{run_root}/imgs_bev/")
        #     bpy.ops.render.render(animation=True)
        #     timer.log("render bev")

        render_persp_semseg = True
        if render_persp_semseg:
            save_depth(bpy, False)
            toggle_bev(bpy, False, pitch_perturbation=episode_info.pitch)
            toggle_semseg(bpy, True) # updates dg w new materials
            #toggle_frame_sz(bpy, "SMALL")
            reset_ap_tm(bpy, ap, tm)
            reset_scene(bpy, ap, tm, timer=timer, save_data=False, render_filepath=f"{run_root}/imgs_semseg/")
            bpy.ops.render.render(animation=True)
            timer.log("render perspective semseg")

        

        # render_depth = True
        # if render_depth:
        #     toggle_bev(bpy, False)
        #     toggle_semseg(bpy, True) # updates dg w new materials
        #     toggle_frame_sz(bpy, "SMALL") #TODO consolidate these toggles, refactor a bit. The whole bit might be better saved manually each time
        #     bpy.data.scenes["Scene"].node_tree.nodes["Switch"].check = True # True for depth, False for normal rgb
        #     reset_ap_tm(bpy, ap, tm)
        #     reset_scene(bpy, ap, tm, timer=timer, save_data=False, render_filepath=f"{run_root}/imgs_depth/")
        #     bpy.ops.render.render(animation=True)
        #     bpy.data.scenes["Scene"].node_tree.nodes["Switch"].check = False
        #     timer.log("render depth")


        # bpy.app.handlers.frame_change_post.clear()

        # disable output redirection
        os.close(fd)
        os.dup(old)
        os.close(old)

        obs_per_sec = EPISODE_LEN/(time.time()-t0)
        report_runner_metric(dataloader_root, obs_per_sec, OBS_PER_SEC_F)
        print(f"Dataloader performance: {obs_per_sec} obs per second")

        render_time = time.time() - _t0
        report_runner_metric(dataloader_root, render_time, RENDER_TIME_F)

        if TRACE_MALLOC:
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)

        # Increment run counter
        run_counter += 1
        run_counter = run_counter % RUNS_TO_STORE_PER_PROCESS # Overwrite from beginning once reach membank limit
        np.save(RUN_COUNTER_F, np.array([run_counter]))

        pretty_print(timer.finish())

        print("\n\n\n\n\n\n********************************************\n\n\n\n\n\n")

        if get_should_stop(): 
            print("Interupting datagen bc received should_stop flag")
            break

"""
# Install new packages into blender's python

import subprocess
package_name = "pandas"
python_exe = "/home/beans/blender_3.3/3.3/python/bin/python3.10"
subprocess.call([python_exe, "-m", "pip", "install", package_name])
"""