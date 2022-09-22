import numpy as np
import os, random, sys, bpy, time, glob

sys.path.append("/home/beans/bespoke")
from constants import *
from traj_utils import *
from map_utils import *


def linear_to_cos(p):
    # p is linear from 0 to 1. Outputs smooth values from 0 to 1 to back to zero
    return (np.cos(p*np.pi*2)*-1 + 1) / 2

# def linear_to_sin_decay(p):
#     # p is linear from 0 to 1. Outputs smooth values from 1 to 0
#     return np.sin(p*np.pi+np.pi/2)
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

    lateral_kP = random.uniform(.75, .95)
    long_kP = random.uniform(.02, .05)
    curve_speed_mult = random.uniform(.7, 1.2)
    turn_slowdown_sec_before = random.uniform(.25, .75)
    max_accel = random.uniform(1.0, 2.0)

import time

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

    get_node("pos_override", make_vehicle_nodes).outputs["Value"].default_value = 0 # We start off placing the car on the curve and aligning it to curve tangent
    
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
    DAGGER_FREQ = random.randint(400, 1000) #random.randint(200, 400)

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
    global lats, lons, way_ids
    dg = bpy.context.evaluated_depsgraph_get()
    cone = bpy.data.objects["Cone"].evaluated_get(dg)
    curve_pos = [(d.vector[0], d.vector[1]) for d in cone.data.attributes["curve_position"].data]
    way_ids = [d.value for d in cone.data.attributes["way_id"].data]
    lats, lons = [d[0] for d in curve_pos], [d[1] for d in curve_pos]
    assert len(lats) == len(way_ids)
    lats, lons, way_ids = add_noise_rds_to_map(lats, lons, way_ids)

    lats = np.array(lats, dtype='float64')
    lons = np.array(lons, dtype='float64')
    way_ids = np.array(way_ids, dtype='int')

    refresh_nav_map_freq = random.choice([3,4,5]) # alternatively, can use vehicle speed and heading to interpolate ala kalman TODO

    # global gps_tracker
    # gps_tracker = GPSTracker()

    print(f"Preparing the nav map took {round(time.time()-t0, 3)} seconds. Map has {len(curve_pos)} nodes")

    def frame_change_post(scene, dg):
        global current_speed_mps, counter, targets_container, overall_frame_counter
        global shift_x, shift_y
        global wp_m_offset, speed_limit, lateral_kP, long_kP, curve_speed_mult, turn_slowdown_sec_before, max_accel
        global current_tire_angle, is_lined

        cube = bpy.data.objects["Cube"].evaluated_get(dg)

        frame_ix = scene.frame_current

        # TODO these names are misleading. This is not the cam, per se, as we're now perturbing cam calib, this is taken from the loop itself
        # TODO URGENT this is probably not accurate, as we're taking the tangent from the curve itself, not the cube, so that will
        # be wrong when doing dagger. Should prob get this directly from the cube itself inside geonodes
        cam_loc = cube.data.attributes["cam_loc"].data[0].vector
        cam_rotation = cube.data.attributes["cam_heading"].data[0].vector #pitch, roll, yaw(heading). Roll always flat. in flat town pitch always flat. yaw==heading.
        cam_normal = cube.data.attributes["cam_normal"].data[0].vector
        sec_to_undagger = 2 #3 # TODO shift should prob be variable, in which case this should also be variable as a fn of shift
        meters_to_undagger = current_speed_mps * sec_to_undagger

        cam_heading = cam_rotation[2]+(np.pi/2)
        traj, wp_dists = [], []
        for i, wp_dist in enumerate(traj_wp_dists):
            wp = cube.data.attributes[f"wp{i}"].data[0].vector 

            if i==0: dist_to_closest_wp = dist(cam_loc, wp)

            if abs(shift_x)>0 or abs(shift_y)>0:
                perc_into_undaggering = wp_dist / meters_to_undagger # the corrective traj
                p = np.clip(linear_to_sin_decay(perc_into_undaggering), 0, 1)
                wp = [wp[0] + shift_x*p, wp[1] + shift_y*p, wp[2]]
            
            #TODO can move all this out of loop for perf sake
            angle_to_wp = get_angle_to(cam_loc[:2], 
                                        cam_heading, 
                                        wp[:2])

            wp_dist_actual = dist(wp, cam_loc)
            targets_container[counter, i] = angle_to_wp
            targets_container[counter, i+N_WPS] = wp_dist_actual
            targets_container[counter, i+2*N_WPS] = wp[2]-cam_loc[2]

            traj.append(angle_to_wp) 
            wp_dists.append(wp_dist_actual)

        traj = np.array(traj) # This is the possibly dagger-shifted traj. Targets are already stored in container, but we don't use them anymore here
        wp_dists = np.array(wp_dists)

        # target wp close to vehicle, used for steering AP to keep it tight on traj TODO consolidate w above apparatus
        wp = cube.data.attributes[f"wpcloseposition"].data[0].vector 
        if abs(shift_x)>0 or abs(shift_y)>0: #TODO is this correct?
            CLOSE_WP_DIST = 3
            perc_into_undaggering = CLOSE_WP_DIST / meters_to_undagger
            p = np.clip(linear_to_sin_decay(perc_into_undaggering), 0, 1)
            wp = [wp[0] + shift_x*p, wp[1] + shift_y*p, wp[2]]
        angle_to_target_wp_ap = get_angle_to(cam_loc[:2], 
                                            cam_heading, 
                                            wp[:2])

        global lats, lons, way_ids, small_map
        # gps doesn't refresh as fast as do frames
        if overall_frame_counter % refresh_nav_map_freq == 0: # This always has to be called on first frame otherwise small_map is none
            close_buffer = CLOSE_RADIUS # TODO is this correct?
            current_lat, current_lon = cam_loc[0], cam_loc[1]
            #heading = gps_tracker.step(current_lat, current_lon, current_speed_mps)

            small_map = get_map(lats, lons, way_ids, 
                                current_lat + maps_noise_position[overall_frame_counter], 
                                current_lon + maps_noise_position[overall_frame_counter], 
                                cam_heading+(np.pi/2)+maps_noise[overall_frame_counter], 
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


        ########################
        # DAGGER
        global shift_x_max, shift_y_max, DAGGER_FREQ, is_doing_dagger, dagger_counter, normal_shift

        DAGGER_DURATION = sec_to_undagger*2*20 # frames TODO don't hardcode frames here

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
        target_wp_angle = angle_to_target_wp_ap #, _, _ = get_target_wp(traj, current_speed_mps, wp_m_offset=wp_m_offset) # comes out as float, frac is the amount btwn the two wps
        
        current_tire_angle = target_wp_angle * lateral_kP
        vehicle_turn_rate = current_tire_angle * (current_speed_mps/CRV_WHEELBASE) # rad/sec # we're using target wp angle as the tire_angle
        vehicle_heading_delta = vehicle_turn_rate * (1/fps) # radians

        # we get slightly out of sync bc our wps are going along based on curve, but we're controlling cube like a vehicle. 
        # Keeping totally synced so wp angles remain perfect. This isn't what moves the car, that's below
        dist_car_travelled_corrected = dist_car_travelled + (MIN_WP_M - dist_to_closest_wp)*.25
        get_node("distance_along_loop", make_vehicle_nodes).outputs["Value"].default_value += dist_car_travelled_corrected

        get_node("heading", make_vehicle_nodes).outputs["Value"].default_value = cam_rotation[2] - vehicle_heading_delta
        get_node("pos_override", make_vehicle_nodes).outputs["Value"].default_value = 0 if frame_ix <=1 else 1 # first frames, set to orientation along curve. After that set manually.

        angle_to_future_vehicle_loc = cam_heading - (vehicle_heading_delta/2) # this isn't the exact calc. See rollout_utils for more accurate version
        delta_x = dist_car_travelled*np.cos(angle_to_future_vehicle_loc)
        delta_y = (dist_car_travelled*np.sin(angle_to_future_vehicle_loc))

        get_node("pos_x", make_vehicle_nodes).outputs["Value"].default_value = cam_loc[0] + delta_x
        get_node("pos_y", make_vehicle_nodes).outputs["Value"].default_value = cam_loc[1] + delta_y 

        # # TODO use the same apparatus we using in rw rollouts. As rw rollouts get better, can improve this to better match
        # # This isn't exact, bc we're taking the angle slightly ahead of actual steer angle, and our lookup is for actual angles
        # # right way to do this would be to estimate, in one second what turn angle will i have to implement
        # angle_for_curve_limit, _, _ = get_target_wp(traj, current_speed_mps, wp_m_offset=current_speed_mps*turn_slowdown_sec_before) 
        # # half second ahead of target wp. So our kP long will have to be fast enough to always get us down low enough

        # curvature_constrained_speed = np.interp(abs(angle_for_curve_limit), max_speed_bps, max_speed_vals)
        # curvature_constrained_speed *= curve_speed_mult

        curvature_constrained_speed = get_curve_constrained_speed_from_wp_angles(traj, wp_dists, current_speed_mps, max_accel=max_accel)
        curvature_constrained_speed *= curve_speed_mult

        target_speed = min(curvature_constrained_speed, speed_limit)

        current_speed_mps += (target_speed - current_speed_mps)*long_kP

        get_node("cam_roll_add", make_vehicle_nodes).outputs["Value"].default_value = roll_noise[overall_frame_counter]


    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(frame_change_post)


