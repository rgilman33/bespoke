
import os, sys, bpy, time, glob, random
import numpy as np

sys.path.append('/home/beans/bespoke')
sys.path.append('/home/beans/bespoke/datagen')
from datagen.episode import make_episode
from bpy_handler import set_frame_change_post_handler, reset_npc_objects
from constants import *

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--" ['example', 'args', '123']
dataloader_id = argv[0]
dataloader_root = f"{BLENDER_MEMBANK_ROOT}/dataloader_{dataloader_id}"

FAILED_TO_GET_MAP_F = f"{dataloader_root}/failed_to_get_map.npy"

if __name__ == "__main__":

    print(f"Launching runner {dataloader_id}")
    # Delete existing npc copies and make new ones. To ensure up to date w any changes made in master.
    reset_npc_objects(bpy)

    # Clear failed message
    if os.path.exists(FAILED_TO_GET_MAP_F):
        os.remove(FAILED_TO_GET_MAP_F)

    for i in range(10_000_000):
        overall_frame_counter = 0
        run_counter = i % RUNS_TO_STORE_PER_PROCESS # Overwrite from beginning once reach membank limit
        t0 = time.time()

        run_root = f"{dataloader_root}/run_{run_counter}"
        os.makedirs(run_root, exist_ok=True)

        # Just removing in advance one at a time. Removing the first one before we start
        next_targets_path = f"{run_root}/targets_{SEQ_LEN-1}.npy"
        if os.path.exists(next_targets_path):
            os.remove(next_targets_path)

        # route of insufficient len causes fail. Also now when have overlap, which is more often
        successful = False
        failed_counter = -1
        while not successful:
            episode_info = make_episode()
            successful = set_frame_change_post_handler(bpy, episode_info, save_data=True, run_root=run_root)
            failed_counter += 1
            if failed_counter==20:
                print("wtf couldn't get good map? Killing runner")
                np.save(FAILED_TO_GET_MAP_F, np.array(1))
                set_should_stop(True) # if one runner dies, shutdown all runners to make it obvious. Perhaps in the future we can tolerate this, but not now
                break
        if not successful: break

        bpy.data.scenes["Scene"].render.image_settings.file_format = 'JPEG' #"AVI_JPEG"
        bpy.data.scenes["Scene"].render.image_settings.quality = 100 #random.randint(50, 100) # zero to 100. Default 50. Going to 30 didn't speed up anything, but we're prob io bound now so test again later when using ramdisk
        # Render samples slows datagen down linearly.
        # Too low and get aliasing around edges, harsh looking. More is softer. We're keeping low samples, trying to make up for it 
        # in data aug w blur and other distractors
        bpy.data.scenes["Scene"].eevee.taa_render_samples = random.randint(2, 5) 
 
        bpy.data.scenes["Scene"].render.filepath = f"{run_root}/imgs/" #f"{run_root}/imgs.avi"
        bpy.data.scenes["Scene"].frame_end = EPISODE_LEN
        bpy.data.scenes["Scene"].render.fps = 20
        # bpy.data.scenes["Scene"].render.resolution_x = 1440 # hardcoded in the blendfile
        # bpy.data.scenes["Scene"].render.resolution_y = 360

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

        bpy.ops.render.render(animation=True)

        # disable output redirection
        os.close(fd)
        os.dup(old)
        os.close(old)

        obs_per_sec = EPISODE_LEN/(time.time()-t0)
        report_obs_per_sec(dataloader_root, obs_per_sec)
        print(f"Dataloader performance: {obs_per_sec} obs per second")
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