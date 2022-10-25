import numpy as np
import os, random, sys, bpy, time, glob
import pandas as pd

sys.path.append("/home/beans/bespoke")
from constants import *
from traj_utils import *
from map_utils import *


def linear_to_cos(p):
    # p is linear from 0 to 1. Outputs smooth values from 0 to 1 to back to zero
    return (np.cos(p*np.pi*2)*-1 + 1) / 2

def linear_to_sin_decay(p):
    # p is linear from 0 to 1. Outputs smooth values from 1 to 0
    return np.sin(p*np.pi+np.pi/2) / 2 + .5

def reset_drive_style():
    global wp_m_offset, speed_limit, lateral_kP, long_kP, curve_speed_mult, turn_slowdown_sec_before, max_accel
    global is_highway

    wp_m_offset = -30 #-12 # telling to always be right on top of traj, this will always be the closest wp
    rr = random.random() 
    if is_highway:
        speed_limit = random.uniform(16, 19) if rr<.05 else random.uniform(19, 25) if rr < .5 else random.uniform(25, 27)
    else:
        speed_limit = random.uniform(8, 12) if rr < .05 else random.uniform(12, 18) if rr < .1 else random.uniform(18, 26) # mps

    lateral_kP = 1.0 #.95 #.85 #random.uniform(.75, .95)
    long_kP = random.uniform(.02, .05)
    curve_speed_mult = random.uniform(.7, 1.2)
    turn_slowdown_sec_before = random.uniform(.25, .75)
    max_accel = random.uniform(1.0, 2.0)

import time

#########################################
"""
We're currently getting some of our wps by shifting the white lines inwards, getting others by creating a new sigmoid
curve. The point of the latter is to be able to smooth the curve, don't want to always just follow the lines, humans smooth them
out. 
"""
ROUTE_LEN_M = 2000
WP_SPACING = .1 # TODO plug this into value in blendfile so always up to date
ROUTE_LEN = ROUTE_LEN_M // WP_SPACING
TRAJ_WP_IXS = np.round(np.array(TRAJ_WP_DISTS) / WP_SPACING).astype('int')

TRAJ_WP_DISTS_NP = np.array(TRAJ_WP_DISTS, dtype='float32')

def get_next_possible_wps(prev_wp, df):
    df = df.copy()
    threshold = 1.0 # m
    all_start_wps = df[df.ix_in_curve==0]
    dist_to_prev_wp = ((all_start_wps.pos_x - prev_wp.pos_x)**2 + (all_start_wps.pos_y - prev_wp.pos_y)**2)**.5
    close_start_wps = all_start_wps[dist_to_prev_wp < threshold]
    return close_start_wps

def get_route(df):
    segments = []
    start_wp = df.sample(1).iloc[0]
    intersection_id = start_wp.intersection_id
    segment_id = start_wp.segment_id
    n_wps = 0
    while n_wps < ROUTE_LEN:
        
        segment_df = df[(df.segment_id == segment_id) & (df.intersection_id==intersection_id)]
        segment_df = segment_df.sort_values("ix_in_curve")
        segments.append(segment_df)
        n_wps += len(segment_df)

        last_wp = segment_df[segment_df.ix_in_curve==segment_df.ix_in_curve.max()]
        last_wp = last_wp.iloc[0]
        close_start_wps = get_next_possible_wps(last_wp, df)
        if len(close_start_wps)==0: break
        next_wp = close_start_wps.sample(1).iloc[0]
        intersection_id = next_wp.intersection_id
        segment_id = next_wp.segment_id
    
    route_wps = pd.concat(segments)
    return route_wps

def get_wp_df(wps_holder_object):
    t0 = time.time()

    df = pd.DataFrame()
    df['is_wp_curve'] = [d.value for d in wps_holder_object.data.attributes["is_wp_curve"].data]
    t0 = time.time()
    pos = [d.vector for d in wps_holder_object.data.attributes["curve_position"].data] # .2 sec
    print("getting pos", time.time() - t0)
    t0 = time.time()
    df["pos_x"] = [d[0] for d in pos] # adding to pandas .3 s
    df["pos_y"] = [d[1] for d in pos]
    df["pos_z"] = [d[2] for d in pos]
    print("setting pos in df", time.time() - t0)

    normals = [d.vector for d in wps_holder_object.data.attributes["curve_normal"].data]
    df["normal_x"] = [d[0] for d in normals]
    df["normal_y"] = [d[1] for d in normals]
    df["normal_z"] = [d[2] for d in normals]

    rotations = [d.vector for d in wps_holder_object.data.attributes["curve_rotation"].data]
    df["curve_pitch"] = [d[0] for d in rotations]
    df["curve_roll"] = [d[1] for d in rotations]
    df["curve_heading"] = [d[2] for d in rotations]

    df["intersection_id"] = intersection_id = [d.value for d in wps_holder_object.data.attributes["intersection_id"].data] # .06 sec

    df["segment_id"] = [d.value for d in wps_holder_object.data.attributes["segment_id"].data]
    df["ix_in_curve"] = [d.value for d in wps_holder_object.data.attributes["ix_in_curve"].data]

    df["is_turnoff"] = [d.value for d in wps_holder_object.data.attributes["is_turnoff"].data]
    df.is_turnoff = df.is_turnoff.astype(int)

    df["way_id"] = (df.intersection_id + df.is_turnoff/10)*10

    return df
####################################
    

def set_frame_change_post_handler(bpy, save_data=False, run_root=None, _is_highway=False, _is_lined=False,
                                    _pitch_perturbation=0, _yaw_perturbation=0):
    global current_speed_mps, counter, targets_container, overall_frame_counter
    global wp_m_offset, speed_limit, lateral_kP, long_kP, curve_speed_mult 
    global current_tire_angle
    global is_highway, is_lined, pitch_perturbation, yaw_perturbation
    is_highway, is_lined, pitch_perturbation, yaw_perturbation = _is_highway, _is_lined, _pitch_perturbation, _yaw_perturbation 

    current_tire_angle = 0

    DRIVE_STYLE_CHANGE_IX = random.randint(200, 600)
    reset_drive_style()

    make_vehicle_nodes = bpy.data.node_groups['MakeVehicle'].nodes 

    targets_container = np.zeros((SEQ_LEN, N_WPS*3), dtype=np.float32)
    aux_container = np.zeros((SEQ_LEN, N_AUX), dtype=np.float32)
    maps_container = np.zeros((SEQ_LEN, MAP_HEIGHT, MAP_WIDTH, 3), dtype='uint8')

    counter = 0
    overall_frame_counter = 0
    current_speed_mps = speed_limit / 2 # starting a bit slower in case in curve

    # dagger
    global is_doing_dagger, dagger_counter, shift_x, shift_y, normal_shift, DAGGER_FREQ
    is_doing_dagger = False
    dagger_counter = 0
    shift_x, shift_y = 0, 0
    NORMAL_SHIFT_MAX = .8
    normal_shift = NORMAL_SHIFT_MAX if random.random()<.5 else -NORMAL_SHIFT_MAX
    DAGGER_FREQ = 200 # TODO UNDO random.randint(400, 1000) #random.randint(200, 400)

    global roll_noise # TODO this may be affecting our wp_angles, that may be a problem especially at higher values of roll. Actually no, i don't think it does
    num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity. can go even lower here, eg to mimic bumpy gravel rd
    ROLL_MAX_DEG = 5
    do_roll = random.random() < .2
    roll_noise_mult = random.uniform(.001, np.radians(ROLL_MAX_DEG)) if do_roll else 0
    roll_noise = get_random_roll_noise(num_passes=num_passes) * roll_noise_mult

    gps_bad = random.random() < .05
    # noise for the map, heading
    num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity
    maps_noise_mult = np.radians(30) if gps_bad else random.uniform(.001, np.radians(5)) # radians
    maps_noise = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult

    # noise for the map, position
    num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity
    maps_noise_mult = 50 if gps_bad else random.uniform(.001, 10) # meters
    maps_noise_position = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult



    t0 = time.time()
    dg = bpy.context.evaluated_depsgraph_get()
    #######################
    wps_holder_object = bpy.data.objects["wps_holder"].evaluated_get(dg)
    wp_df = get_wp_df(wps_holder_object)
    print("get wp df", time.time() - t0)

    route_len = 0
    while route_len < ROUTE_LEN:
        route = get_route(wp_df)
        route_len = len(route)
    print("get route", time.time() - t0)

    #######################

    global lats, lons, way_ids

    coarse_map_df = wp_df[wp_df.is_wp_curve==False]

    way_ids = coarse_map_df.way_id.to_numpy()
    lats, lons = coarse_map_df.pos_x.to_numpy(), coarse_map_df.pos_y.to_numpy()
    # lats, lons, way_ids = add_noise_rds_to_map(lats, lons, way_ids)

    lats = np.array(lats, dtype='float64')
    lons = np.array(lons, dtype='float64')
    way_ids = np.array(way_ids, dtype='int')

    refresh_nav_map_freq = random.choice([3,4,5]) # alternatively, can use vehicle speed and heading to interpolate ala kalman TODO

    # global gps_tracker
    # gps_tracker = GPSTracker()

    waypoints = np.empty((len(route), 3), dtype="float64")
    smooth_amount = 20

    waypoints[:,0] = moving_average(route.pos_x.to_numpy(), smooth_amount)
    waypoints[:,1] = moving_average(route.pos_y.to_numpy(), smooth_amount)
    waypoints[:,2] = moving_average(route.pos_z.to_numpy(), 20)

    wp_normals = np.empty((len(route), 3), dtype="float64")
    wp_normals[:, 0] = moving_average(route.normal_x.to_numpy(), smooth_amount) 
    wp_normals[:, 1] = moving_average(route.normal_y.to_numpy(), smooth_amount) 
    wp_normals[:, 2] = moving_average(route.normal_z.to_numpy(), smooth_amount)

    wp_rotations = np.empty((len(route), 3), dtype="float64") # pitch, roll, yaw
    wp_rotations[:, 0] = moving_average(route.curve_pitch.to_numpy(), 40) ## pitch
    wp_rotations[:, 1] = route.curve_roll.to_numpy() ## roll
    wp_rotations[:, 2] = moving_average(route.curve_heading.to_numpy(), smooth_amount) # yaw



    global current_pos, current_heading, distance_along_loop
    START_IX = (smooth_amount // 2) + 1
    current_pos = waypoints[START_IX]
    current_heading = wp_rotations[START_IX, 2]

    get_node("pitch", make_vehicle_nodes).outputs["Value"].default_value = wp_rotations[START_IX, 0]
    get_node("heading", make_vehicle_nodes).outputs["Value"].default_value = current_heading
    get_node("roll", make_vehicle_nodes).outputs["Value"].default_value = wp_rotations[START_IX, 1]

    get_node("pos_x", make_vehicle_nodes).outputs["Value"].default_value = waypoints[START_IX][0]
    get_node("pos_y", make_vehicle_nodes).outputs["Value"].default_value = waypoints[START_IX][1]
    get_node("pos_z", make_vehicle_nodes).outputs["Value"].default_value = waypoints[START_IX][2]

    distance_along_loop = START_IX * WP_SPACING

    print(f"Preparing the nav map took {round(time.time()-t0, 3)} seconds. Map has {len(route)} nodes")

    def frame_change_post(scene, dg):
        global current_speed_mps, counter, targets_container, overall_frame_counter
        global shift_x, shift_y
        global wp_m_offset, speed_limit, lateral_kP, long_kP, curve_speed_mult, turn_slowdown_sec_before, max_accel
        global current_tire_angle, is_lined
        global current_pos, current_heading, distance_along_loop

        cube = bpy.data.objects["Cube"].evaluated_get(dg)

        frame_ix = scene.frame_current
        
        sec_to_undagger = 3 # TODO shift should prob be variable, in which case this should also be variable as a fn of shift
        meters_to_undagger = current_speed_mps * sec_to_undagger

        current_wp_ix = int(round(distance_along_loop / WP_SPACING, 0))
        wp_ixs = current_wp_ix + TRAJ_WP_IXS
        current_wps = waypoints[wp_ixs]
        cam_normal = wp_normals[current_wp_ix]


        # getting targets
            
        deltas = current_wps - current_pos
        xs, ys = deltas[:, 0], deltas[:, 1]

        if abs(shift_x) > 0 or abs(shift_y) > 0:
            perc_into_undaggering = TRAJ_WP_DISTS_NP / meters_to_undagger 
            p = np.clip(linear_to_sin_decay(perc_into_undaggering), 0, 1)
            xs += shift_x*p
            ys += shift_y*p

        angles_to_wps = get_angles_to(xs, ys, -current_heading)
        wp_dists_actual = np.sqrt(xs**2 + ys**2)
        
        targets_container[counter, :N_WPS] = angles_to_wps
        targets_container[counter, N_WPS:N_WPS*2] = wp_dists_actual
        targets_container[counter, N_WPS*2:] = deltas[:,2]


        # Navmap
        global lats, lons, way_ids, small_map
        # gps doesn't refresh as fast as do frames
        if overall_frame_counter % refresh_nav_map_freq == 0: # This always has to be called on first frame otherwise small_map is none
            close_buffer = CLOSE_RADIUS # TODO is this correct?
            current_lat, current_lon = current_pos[0], current_pos[1]
            #heading = gps_tracker.step(current_lat, current_lon, current_speed_mps)

            small_map = get_map(lats, lons, way_ids, 
                                current_lat + maps_noise_position[overall_frame_counter], 
                                current_lon + maps_noise_position[overall_frame_counter], 
                                current_heading+(np.pi)+maps_noise[overall_frame_counter], 
                                close_buffer)

        maps_container[counter,:,:,:] = small_map

        ############
        # save data
        aux_container[counter, 2] = mps_to_kph(current_speed_mps)
        aux_container[counter, 4] = current_tire_angle
        aux_container[counter, 0] = pitch_perturbation
        aux_container[counter, 1] = yaw_perturbation

        if (counter+1) == SEQ_LEN:
            if save_data:
                np.save(f"{run_root}/aux_{overall_frame_counter}.npy", aux_container)  
                np.save(f"{run_root}/targets_{overall_frame_counter}.npy", targets_container)
                np.save(f"{run_root}/maps_{overall_frame_counter}.npy", maps_container)  

                next_seq_path = f"{run_root}/targets_{overall_frame_counter+SEQ_LEN}.npy"
                if os.path.exists(next_seq_path):
                    os.remove(next_seq_path)

            targets_container[:,:] = 0.0 # TODO shouldn't need this, but it's for safety?
            aux_container[:,:] = 0.0
            counter = 0
        else:
            counter += 1
        
        overall_frame_counter += 1

        #############################################################################

        # target wp close to vehicle, used for steering AP to keep it tight on traj
        # This just follows the cos dagger traj directly. This is the wp we aim towards.
        CLOSE_WP_DIST = 3.0
        wp = waypoints[current_wp_ix + int(round(CLOSE_WP_DIST/WP_SPACING))]
        if abs(shift_x)>0 or abs(shift_y)>0:
            wp = [wp[0] + shift_x, wp[1] + shift_y, wp[2]]

        deltas = wp - current_pos
        xs, ys = deltas[0:1], deltas[1:2]
        angle_to_target_wp_ap = get_angles_to(xs, ys, -current_heading)[0] # blender counterclockwise is positive

        # DAGGER
        global shift_x_max, shift_y_max, DAGGER_FREQ, is_doing_dagger, dagger_counter, normal_shift

        DAGGER_DURATION = sec_to_undagger*2*FPS # frames. TODO this doesn't need to be tied w sec_to_undagger

        shift_x_max = cam_normal[0]*normal_shift
        shift_y_max = cam_normal[1]*normal_shift

        if frame_ix % DAGGER_FREQ == 0:
            is_doing_dagger = True
        if is_doing_dagger:
            dagger_counter += 1
            r = linear_to_cos(dagger_counter/DAGGER_DURATION)
            
            shift_x = r * shift_x_max
            shift_y = r * shift_y_max
            
        if dagger_counter == DAGGER_DURATION:
            is_doing_dagger = False
            dagger_counter = 0
            shift_x = 0
            shift_y = 0
            normal_shift = 1 if random.random()<.5 else -1
                
        ########################
        # Moving the car

        # speed limit and turn agg
        if frame_ix % DRIVE_STYLE_CHANGE_IX: reset_drive_style()

        fps = random.gauss(20, 2)

        dist_car_travelled = current_speed_mps * (1 / fps)

        # always using close wp for ap, to keep right on traj
        current_tire_angle = angle_to_target_wp_ap * lateral_kP
        _wheelbase = 1.5 #CRV_WHEELBASE # NOTE this isn't correct, but i want ego to be turning faster, otherwise taking turns too wide
        _vehicle_turn_rate = current_tire_angle * (current_speed_mps/_wheelbase) # rad/sec # we're using target wp angle as the tire_angle
        vehicle_heading_delta = _vehicle_turn_rate * (1/fps) # radians

        # we get slightly out of sync bc our wps are going along based on curve, but we're controlling cube like a vehicle. 
        # Keeping totally synced so wp angles remain perfect. This isn't what moves the car, that's below
        dist_car_travelled_corrected = dist_car_travelled + (MIN_WP_M - wp_dists_actual[0])*.25
        distance_along_loop += dist_car_travelled_corrected

        get_node("pitch", make_vehicle_nodes).outputs["Value"].default_value = wp_rotations[current_wp_ix, 0]
        
        current_heading = current_heading - vehicle_heading_delta
        get_node("heading", make_vehicle_nodes).outputs["Value"].default_value = current_heading

        _current_heading = current_heading+(np.pi/2) # TODO can rationalize this away
        angle_to_future_vehicle_loc = _current_heading - (vehicle_heading_delta/2) # this isn't the exact calc. See rollout_utils for more accurate version
        delta_x = dist_car_travelled*np.cos(angle_to_future_vehicle_loc)
        delta_y = (dist_car_travelled*np.sin(angle_to_future_vehicle_loc))

        current_x = current_pos[0] + delta_x
        current_y = current_pos[1] + delta_y
        current_z = waypoints[current_wp_ix][2] # Just setting this manually to the wp itself. Not "driving" it like XY

        get_node("pos_x", make_vehicle_nodes).outputs["Value"].default_value = current_x
        get_node("pos_y", make_vehicle_nodes).outputs["Value"].default_value = current_y 
        get_node("pos_z", make_vehicle_nodes).outputs["Value"].default_value = current_z
        
        current_pos = np.array([current_x, current_y, current_z])
        ###

        curvature_constrained_speed = get_curve_constrained_speed_from_wp_angles(angles_to_wps, wp_dists_actual, current_speed_mps, max_accel=max_accel)
        curvature_constrained_speed *= curve_speed_mult

        target_speed = min(curvature_constrained_speed, speed_limit)

        current_speed_mps += (target_speed - current_speed_mps)*long_kP

        get_node("roll", make_vehicle_nodes).outputs["Value"].default_value = roll_noise[overall_frame_counter]


    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(frame_change_post)


