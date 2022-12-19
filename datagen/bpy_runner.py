
import os, sys, bpy, time, glob, random
import numpy as np

sys.path.append('/home/beans/bespoke')
sys.path.append('/home/beans/bespoke/datagen')
from material_updater import setup_map
from bpy_handler import set_frame_change_post_handler, reset_npc_objects
from constants import *

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--" ['example', 'args', '123']
dataloader_id = argv[0]
dataloader_root = f"{BLENDER_MEMBANK_ROOT}/dataloader_{dataloader_id}"

def random_uniform(_min, _max):
    return random.uniform(_min, _max)

bpy.app.driver_namespace['random_uniform'] = random_uniform

if __name__ == "__main__":

    print(f"Launching runner {dataloader_id}")
    # Delete existing npc copies and make new ones. To ensure up to date w any changes made in master.
    reset_npc_objects(bpy)

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

        # currently only route of insufficient len causes fail. 
        successful = False
        failed_counter = -1
        while not successful:
            is_highway, is_lined, pitch_perturbation, yaw_perturbation, has_npcs, is_single_rd = setup_map()

            successful = set_frame_change_post_handler(bpy, save_data=True, run_root=run_root, _is_highway=is_highway, _is_lined=is_lined, 
                                            _pitch_perturbation=pitch_perturbation, _yaw_perturbation=yaw_perturbation, has_npcs=has_npcs, 
                                            _is_single_rd=is_single_rd)
            failed_counter += 1
            if failed_counter==3:
                print("wtf couldn't get good map? Killing runner")
                np.save(f"{run_root}/failed_to_get_map.npy", np.array(1))
                break
        if not successful: break

        bpy.data.scenes["Scene"].render.image_settings.file_format = 'JPEG' #"AVI_JPEG"
        bpy.data.scenes["Scene"].render.image_settings.quality = 100 #random.randint(50, 100) # zero to 100. Default 50. Going to 30 didn't speed up anything, but we're prob io bound now so test again later when using ramdisk
        bpy.data.scenes["Scene"].eevee.taa_render_samples = random.randint(2, 5)
 
        bpy.data.scenes["Scene"].render.filepath = f"{run_root}/imgs/" #f"{run_root}/imgs.avi"
        bpy.data.scenes["Scene"].frame_end = EPISODE_LEN
        bpy.data.scenes["Scene"].render.fps = 20
        # bpy.data.scenes["Scene"].render.resolution_x = 640 # just hardcoded in the blendfile
        # bpy.data.scenes["Scene"].render.resolution_y = 120
        # bpy.data.scenes["Scene"].render.resolution_x = 1440
        # bpy.data.scenes["Scene"].render.resolution_y = 360

        bpy.ops.render.render(animation=True)
        #bpy.ops.render.opengl(animation=True) # can't do this headless

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