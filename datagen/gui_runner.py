import os
import sys, bpy
import numpy as np
import importlib

sys.path.append("/media/beans/ssd/bespoke/datagen")

import material_updater
import bpy_handler

importlib.reload(material_updater)
importlib.reload(bpy_handler)

bpy_handler.set_frame_change_post_handler(bpy, save_data=False)
material_updater.setup_map()