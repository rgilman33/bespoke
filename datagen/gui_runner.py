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

 
timer = constants.Timer("nothing")

bpy_handler.reset_npc_objects(bpy)

# Make episode map
episode_info = episode.make_episode(timer)

# Retrieve map
wp_df, coarse_map_df, success = bpy_handler.get_map_data(bpy, episode_info, timer) # can fail here

# Get route
start_left = random.random()<.5
ego_route = bpy_handler.get_ego_route(wp_df, episode_info, start_left) # can fail here

bpy_handler.set_frame_change_post_handler(bpy, wp_df, coarse_map_df, ego_route, episode_info, timer, save_data=False)
