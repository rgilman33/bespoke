
import os, sys, bpy, time, glob, random
import numpy as np

sys.path.append('/home/beans/bespoke')
sys.path.append('/home/beans/bespoke/datagen')
from datagen.episode import make_episode
from bpy_handler import *
from constants import *

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--" ['example', 'args', '123']
dataloader_id = argv[0]
dataloader_root = f"{BLENDER_MEMBANK_ROOT}/dataloader_{dataloader_id}"

FAILED_TO_GET_MAP_F = f"{dataloader_root}/failed_to_get_map.npy"
RUN_COUNTER_F = f"{dataloader_root}/run_counter.npy"

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

TRACE_MALLOC = False # finding memleak
if __name__ == "__main__":
    if TRACE_MALLOC:
        tracemalloc.start()

    print(f"Launching runner {dataloader_id}")
    # Delete existing npc copies and make new ones. To ensure up to date w any changes made in master.
    reset_npc_objects(bpy)

    # Clear failed message
    if os.path.exists(FAILED_TO_GET_MAP_F):
        os.remove(RUN_COUNTER_F)

    # If picking up from where we left off, start from last run not from zero
    if os.path.exists(RUN_COUNTER_F):
        last_run_counter = np.load(RUN_COUNTER_F)[0]
        print(f"Resuming from run {last_run_counter}")
    else:
        last_run_counter = 0

    for i in range(last_run_counter, 10_000_000):
        timer = Timer("*** gather run total ***")
        overall_frame_counter = 0
        run_counter = i % RUNS_TO_STORE_PER_PROCESS # Overwrite from beginning once reach membank limit
        t0 = time.time()
        np.save(RUN_COUNTER_F, np.array([run_counter]))

        run_root = f"{dataloader_root}/run_{run_counter}"
        os.makedirs(run_root, exist_ok=True)
        os.makedirs(f"{run_root}/aux", exist_ok=True)
        os.makedirs(f"{run_root}/targets", exist_ok=True)
        os.makedirs(f"{run_root}/maps", exist_ok=True)
        timer.log("prep dirs")

        # route of insufficient len causes fail. Also now when have overlap, which is more often
        successful = False
        failed_counter = -1
        while not successful:

            # Deal w failed map/route
            if failed_counter==20:
                print("wtf couldn't get good map? Killing runner")
                np.save(FAILED_TO_GET_MAP_F, np.array(1))
                set_should_stop(True) # if one runner dies, shutdown all runners to make it obvious. Perhaps in the future we can tolerate this, but not now
                break
    
            # Setup episode map
            episode_info = make_episode(timer)

            # Retrieve map
            wp_df, coarse_map_df = get_map_data(bpy, episode_info, timer)
            if wp_df is None: # when map has overlapping rds
                failed_counter += 1
                continue

            # Get route
            ego_route = get_ego_route(wp_df, episode_info)
            timer.log("get route")
            if ego_route is None:
                failed_counter += 1
                continue
            
            # We've got a good map and good route, set up handler
            set_frame_change_post_handler(bpy, wp_df, coarse_map_df, ego_route, episode_info, timer, save_data=True, run_root=run_root)
            successful = True

        if not successful: break

        init_time = time.time() - t0
        report_runner_metric(dataloader_root, init_time, INIT_TIME_F)
        _t0 = time.time()
        ##################
        # Render

        # redirect output to log file. From https://blender.stackexchange.com/questions/44560/how-to-supress-bpy-render-messages-in-terminal-output
        # Render too verbose, makes it impossible to debug anything in the terminal
        logfile = f'{dataloader_root}/blender_render.log'
        if os.path.exists(logfile): os.remove(logfile)
        open(logfile, 'a').close()
        old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        fd = os.open(logfile, os.O_WRONLY)
        timer.log("redirect stdout")

        bpy.ops.render.render(animation=True)
        bpy.app.handlers.frame_change_post.clear()
        timer.log("render")

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