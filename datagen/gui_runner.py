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
episode_info = episode.make_map(timer)

# Retrieve map
wp_df, coarse_map_df, success = bpy_handler.get_map_data(bpy, episode_info, timer) # can fail here

run_counter = 0
episode.randomize_appearance(timer, episode_info, run_counter)
 
# Get route
start_left = random.random()<.5
ego_route = bpy_handler.get_ego_route(wp_df, episode_info, start_left) # can fail here

# Create AP and TM  
ap, tm = bpy_handler.create_ap_tm(bpy, wp_df, coarse_map_df, ego_route, episode_info, timer, run_root=None)

# Reset scene
bpy_handler.reset_scene(bpy, ap, tm, save_data=False, render_filepath=None)

bpy_handler.toggle_bev(bpy, False)
bpy_handler.toggle_semseg(bpy, False)
