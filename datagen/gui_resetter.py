import os
import sys, bpy
import numpy as np
import importlib, random

sys.path.append('/home/beans/bespoke')
sys.path.append('/home/beans/bespoke/datagen')

import constants
import datagen.episode as episode 
import bpy_handler
import traj_utils
import map_utils
import autopilot
  
importlib.reload(constants)
importlib.reload(traj_utils)
importlib.reload(map_utils)
importlib.reload(episode)
importlib.reload(autopilot)
importlib.reload(bpy_handler)

 
bpy_handler.toggle_bev(bpy, True)
bpy_handler.toggle_semseg(bpy, True)
