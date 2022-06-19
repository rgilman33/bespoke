import numpy as np
import os, random, sys, bpy, time, glob

sys.path.append("/media/beans/ssd/bespoke")
from constants import *
from traj_utils import *

def reset_dagger_params():
    global DAGGER_MAX_X, DAGGER_MAX_Y, DAGGER_DURATION, DAGGER_FREQ

    DAGGER_LIMIT = .3
    DAGGER_FREQ = random.randint(80, 120) #random.randint(60, 120)
    DAGGER_DURATION = random.randint(24, 64)
    DAGGER_MAX_X = random.uniform(-DAGGER_LIMIT, DAGGER_LIMIT)
    DAGGER_MAX_Y = random.uniform(-DAGGER_LIMIT, DAGGER_LIMIT)

def reset_drive_style():
    global wp_m_offset, speed_limit, lateral_kP, long_kP, curve_speed_mult, turn_slowdown_sec_before
    global is_highway

    wp_m_offset = -8 # telling to always be right on top of traj
    if is_highway:
        speed_limit = random.uniform(15, 18) if random.random()<.05 else random.uniform(25, 29) if random.random()<.05 else random.uniform(18, 25)
    else:
        speed_limit = random.uniform(8, 22) # mps

    lateral_kP = random.uniform(.7, .9)
    long_kP = random.uniform(.02, .05)
    curve_speed_mult = random.uniform(.85, 1.1)
    turn_slowdown_sec_before = random.uniform(.25, .75)

is_highway = False
def set_frame_change_post_handler(bpy, save_data=False, run_root=None, _is_highway=False):
    global current_speed_mps, counter, targets_container, overall_frame_counter
    global wp_m_offset, speed_limit, lateral_kP, long_kP, curve_speed_mult 
    global current_tire_angle
    global is_highway
    is_highway = _is_highway

    current_tire_angle = 0

    DRIVE_STYLE_CHANGE_IX = random.randint(200, 600)
    reset_drive_style()

    make_vehicle_nodes = bpy.data.node_groups['MakeVehicle'].nodes 

    get_node("pos_override", make_vehicle_nodes).outputs["Value"].default_value = 0 # We start off placing the car on the curve and aligning it to curve tangent

    targets_container = np.zeros((SEQ_LEN, N_WPS_TO_USE), dtype=np.float32)
    aux_container = np.zeros((SEQ_LEN, N_AUX), dtype=np.float32)
    counter = 0
    overall_frame_counter = 0

    current_speed_mps = speed_limit / 2 # starting a bit slower in case in curve

    # dagger
    global DAGGER_MAX_X, DAGGER_MAX_Y, DAGGER_DURATION, DAGGER_FREQ
    global is_doing_dagger, dagger_counter, shift_x, shift_y

    is_doing_dagger = False
    dagger_counter = 0
    reset_dagger_params()
    shift_x, shift_y = 0, 0

    def frame_change_post(scene, dg): #TODO move the vehicle movement things BEFORE the data saving so we don't have to do the staggering in the dataloader. Actually i dunno...
        global current_speed_mps, counter, targets_container, overall_frame_counter
        global shift_x, shift_y
        global wp_m_offset, speed_limit, lateral_kP, long_kP, curve_speed_mult, turn_slowdown_sec_before
        global current_tire_angle

        cube = bpy.data.objects["Cube"].evaluated_get(dg) #; print("TEST 2", [a for a in cube.data.attributes])
        frame_ix = scene.frame_current
        cam_loc = cube.data.attributes["cam_loc"].data[0].vector 
        cam_heading = cube.data.attributes["cam_heading"].data[0].vector #pitch, roll, yaw(heading). Roll always flat. in flat town pitch always flat. yaw==heading.
        cam_normal = cube.data.attributes["cam_normal"].data[0].vector

        traj = []
        for i in range(N_WPS_TO_USE):
            wp = cube.data.attributes[f"wp{i}"].data[0].vector 
            # radians. Left is negative.
            angle_to_wp = get_angle_to(cam_loc[:2], 
                                        cam_heading[2]+(np.pi/2), 
                                        wp[:2])

            targets_container[counter, i] = angle_to_wp

            if i==0:
                dist_to_closest_wp = dist(cam_loc, wp)

            # # shifted traj for dagger
            # if abs(shift_x)>0 or abs(shift_y)>0:
            #     shifted_wp = [wp[0] + shift_x, wp[1] + shift_y]
            #     angle_to_wp_shifted = get_angle_to(cam_loc[:2], 
            #                     cam_heading[2]+(np.pi/2), 
            #                     shifted_wp)
            #     traj.append(angle_to_wp_shifted)
            # else:
            #     traj.append(angle_to_wp)
            traj.append(angle_to_wp)

        traj = np.array(traj) # This is the possible dagger-shifted traj. Targets are already stored in container, but we don't use them anymore here

        ############
        # save data
        aux_container[counter, 2] = mps_to_kph(current_speed_mps)
        aux_container[counter, 4] = current_tire_angle

        if (counter+1) == SEQ_LEN:
            if save_data:
                np.save(f"{run_root}/aux_{overall_frame_counter}.npy", aux_container)  
                np.save(f"{run_root}/targets_{overall_frame_counter}.npy", targets_container)  
                next_seq_path = f"{run_root}/targets_{overall_frame_counter+SEQ_LEN}.npy"
                if os.path.exists(next_seq_path):
                    os.remove(next_seq_path)

            targets_container[:,:] = 0.0
            aux_container[:,:] = 0.0
            counter = 0
        else:
            counter += 1
        
        overall_frame_counter += 1

        ###################
        # updating vehicle location

        # # speed limit and turn agg
        # if frame_ix % DRIVE_STYLE_CHANGE_IX:
        #     reset_drive_style()

        # # DAGGER
        # global DAGGER_MAX_X, DAGGER_MAX_Y, DAGGER_DURATION, DAGGER_FREQ
        # global is_doing_dagger, dagger_counter

        # if frame_ix % DAGGER_FREQ == 0:
        #     is_doing_dagger = True
        # if is_doing_dagger:
        #     dagger_counter += 1
        #     h = DAGGER_DURATION//2
    
        #     if dagger_counter <= h:
        #         r = dagger_counter/h
        #     else:
        #         r = 1 - (dagger_counter/h - 1)

        #     shift_x = r * DAGGER_MAX_X
        #     shift_y = r * DAGGER_MAX_Y
            
        # if dagger_counter == DAGGER_DURATION:
        #     is_doing_dagger = False
        #     dagger_counter = 0
        #     reset_dagger_params()
        #     shift_x = 0
        #     shift_y = 0
                
        ########################
        fps = random.gauss(20, 2)

        dist_car_travelled = current_speed_mps * (1 / fps)
        target_wp_angle, _, _ = get_target_wp(traj, current_speed_mps, wp_m_offset=wp_m_offset) # comes out as float, frac is the amount btwn the two wps
        current_tire_angle = target_wp_angle * lateral_kP
        vehicle_turn_rate = current_tire_angle * (current_speed_mps/CRV_WHEELBASE) # rad/sec # we're using target wp angle as the tire_angle
        vehicle_heading_delta = vehicle_turn_rate * (1/fps) # radians

        # we get slightly out of sync bc our wps are going along based on curve, but we're controlling cube like a vehicle. 
        # Keeping totally synced to wp angles remain perfect. This isn't what moves the car, that's below
        dist_car_travelled_corrected = dist_car_travelled + (MIN_WP_M - dist_to_closest_wp)*.25
        get_node("distance_along_loop", make_vehicle_nodes).outputs["Value"].default_value += dist_car_travelled_corrected

        get_node("heading", make_vehicle_nodes).outputs["Value"].default_value = cam_heading[2] - vehicle_heading_delta
        get_node("pos_override", make_vehicle_nodes).outputs["Value"].default_value = 0 if frame_ix <=1 else 1 # first frames, set to orientation along curve. After that set manually.

        angle_to_future_vehicle_loc = cam_heading[2]+(np.pi/2) - (vehicle_heading_delta/2) # this isn't the exact calc. See rollout_utils for more accurate version
        delta_x = dist_car_travelled*np.cos(angle_to_future_vehicle_loc)
        delta_y = (dist_car_travelled*np.sin(angle_to_future_vehicle_loc))

        # this way is the technically correct one, but keep in mind it's nearly identical to above bc things are nearly linear in these
        # short time periods TODO doesn't work
        # _vehicle_heading = cam_heading[2] - vehicle_heading_delta # don't know why sometimes subtract pi/2 or whatever. all janky.

        # if _vehicle_heading==0:
        #     delta_x = 0
        #     delta_y = dist_car_travelled
        # else:
        #     r = dist_car_travelled / _vehicle_heading
        #     delta_y = np.sin(_vehicle_heading)*r
        #     delta_x = r - (np.cos(_vehicle_heading)*r)

        dagger_offset = 0 #2 if frame_ix%200==0 else 0
        get_node("pos_x", make_vehicle_nodes).outputs["Value"].default_value = cam_loc[0] + delta_x + cam_normal[0]*dagger_offset
        get_node("pos_y", make_vehicle_nodes).outputs["Value"].default_value = cam_loc[1] + delta_y + cam_normal[1]*dagger_offset

        # get_node("normal_shift", make_vehicle_nodes).outputs["Value"].default_value = normal_shift

        # This isn't exact, bc we're taking the angle slightly ahead of actual steer angle, and our lookup is for actual angles
        # right way to do this would be to estimate, in one second what turn angle will i have to implement
        angle_for_curve_limit, _, _ = get_target_wp(traj, current_speed_mps, wp_m_offset=current_speed_mps*turn_slowdown_sec_before) 
        # half second ahead of target wp. So our kP long will have to be fast enough to always get us down low enough

        curvature_constrained_speed = np.interp(abs(angle_for_curve_limit), max_speed_bps, max_speed_vals)
        curvature_constrained_speed *= curve_speed_mult

        target_speed = min(curvature_constrained_speed, speed_limit)
        current_speed_mps += (target_speed - current_speed_mps)*long_kP

    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(frame_change_post)


