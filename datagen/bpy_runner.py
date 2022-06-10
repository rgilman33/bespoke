
import os, sys, bpy, time, glob, random
import numpy as np

sys.path.append("/media/beans/ssd/bespoke")
sys.path.append("/media/beans/ssd/bespoke/datagen")
from material_updater import setup_map
from bpy_handler import set_frame_change_post_handler
from constants import *

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--" ['example', 'args', '123']
dataloader_id = argv[0]
dataloader_root = f"{BLENDER_MEMBANK_ROOT}/dataloader_{dataloader_id}"

if __name__ == "__main__":
    for i in range(10_000_000):
        overall_frame_counter = 0
        run_counter = i % RUNS_TO_STORE_PER_PROCESS # Overwrite from beginning once reach membank limit
        t0 = time.time()

        run_root = f"{dataloader_root}/run_{run_counter}"
        os.makedirs(run_root, exist_ok=True)
        # # Just removing the targets files, as we'll use these to grab for the dataloader. The imgs will just get written over,
        # # but we'll know that the imgs are fresh bc the imgs keep up w the target files
        # existing_targets_npy = glob.glob(f"{run_root}/targets_*.npy")
        # for f in existing_targets_npy:
        #     os.remove(f)
        # Just removing in advance one at a time. Removing the first one before we start
        next_targets_path = f"{run_root}/targets_{SEQ_LEN-1}.npy"
        if os.path.exists(next_targets_path):
            os.remove(next_targets_path)

        set_frame_change_post_handler(bpy, save_data=True, run_root=run_root)

        setup_map()

        bpy.data.scenes["Scene"].render.image_settings.file_format = 'JPEG' #"AVI_JPEG"
        bpy.data.scenes["Scene"].render.image_settings.quality = random.randint(40, 80) # zero to 100. Default 50. Going to 30 didn't speed up anything, but we're prob io bound now so test again later when using ramdisk
        bpy.data.scenes["Scene"].eevee.taa_render_samples = random.randint(2, 5)
 
        bpy.data.scenes["Scene"].render.filepath = f"{run_root}/imgs/" #f"{run_root}/imgs.avi"
        bpy.data.scenes["Scene"].frame_end = EPISODE_LEN
        bpy.data.scenes["Scene"].render.fps = 20

        bpy.ops.render.render(animation=True)
        #bpy.ops.render.opengl(animation=True) # can't do this headless

        obs_per_sec = EPISODE_LEN/(time.time()-t0)
        report_obs_per_sec(dataloader_root, obs_per_sec)
        print(f"Dataloader performance: {obs_per_sec} obs per second")
        if get_should_stop(): 
            print("Interupting datagen bc received should_stop flag")
            break

