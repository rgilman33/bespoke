import os
import sys, bpy
import numpy as np
import importlib

sys.path.append('/home/beans/bespoke')
sys.path.append('/home/beans/bespoke/datagen')

import constants
import material_updater
import bpy_handler
import traj_utils
import map_utils

importlib.reload(constants)
importlib.reload(traj_utils)
importlib.reload(material_updater)
importlib.reload(bpy_handler)
importlib.reload(map_utils)

is_highway, is_lined, pitch_perturbation, yaw_perturbation = material_updater.setup_map() 
bpy_handler.set_frame_change_post_handler(bpy, save_data=False, _is_highway=is_highway, _is_lined=is_lined)