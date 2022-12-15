import numpy as np
import os, random, sys, bpy, time, glob
import pandas as pd

sys.path.append("/home/beans/bespoke")
from constants import *
from traj_utils import *
from map_utils import *



class Autopilot():
    def __init__(self, save_data=True, is_highway=False, pitch_perturbation=0, yaw_perturbation=0, \
                    run_root=None, ap_id=None, is_ego=False, just_go_straight=False):
        self.is_highway, self.save_data, self.run_root, self.ap_id, self.is_ego = is_highway, save_data, run_root, ap_id, is_ego
        self.reset_drive_style()
        self.counter, self.overall_frame_counter = 0, 0
        self.current_speed_mps = self.speed_limit / 2
        self.current_tire_angle = 0
        self.DRIVE_STYLE_CHANGE_IX = random.randint(200, 600)
        self.DAGGER_FREQ = random.randint(400, 1000)
        self.reset_dagger()

        self.targets_container = np.zeros((SEQ_LEN, N_WPS*4), dtype=np.float32) # angles, dists, rolls, z deltas
        self.aux_container = np.zeros((SEQ_LEN, N_AUX_TO_SAVE), dtype=np.float32)
        self.maps_container = np.zeros((SEQ_LEN, MAP_HEIGHT, MAP_WIDTH, 3), dtype='uint8')

        self.episode_info = np.zeros(N_EPISODE_INFO, dtype=np.float32)

        self.pitch_perturbation, self.yaw_perturbation = pitch_perturbation, yaw_perturbation
        self.small_map = None
        self.should_yield = False
        self.route_is_done = False

        self.N_NEGOTIATION_WPS = 100
        NEGOTIATION_TRAJ_LEN_SEC = random.uniform(2.0, 3.5) # follow dist. TODO uncouple follow dist and negotiation traj
        self.S_PER_WP = NEGOTIATION_TRAJ_LEN_SEC / self.N_NEGOTIATION_WPS

        NO_ROUTE_PROB = .4 # Only applied when just_go_straight TODO update this when doing turns again
        self.draw_route = False if (random.random() < NO_ROUTE_PROB and just_go_straight) else True

        ######
        # Noise for the map
        num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity. can go even lower here, eg to mimic bumpy gravel rd
        ROLL_MAX_DEG = 1.2
        do_roll = random.random() < .2
        roll_noise_mult = random.uniform(.001, np.radians(ROLL_MAX_DEG)) if do_roll else 0
        self.roll_noise = get_random_roll_noise(num_passes=num_passes) * roll_noise_mult

        r = random.random()
        gps_state = "BAD" if (r < .05 and just_go_straight) else "MED" if r <.1 else "NORMAL" #TODO update these probs when doing turns again
        # noise for the map, heading. This is in addition to what we'll get from the noisy pos
        num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity
        maps_noise_mult = np.radians(10) if gps_state=="BAD" else np.radians(5) if gps_state=="MED" else random.uniform(.001, np.radians(3)) # radians
        self.maps_noise_heading = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult

        # noise for the map, position
        num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity
        maps_noise_mult = 60 if gps_state=="BAD" else 30 if gps_state=="MED" else random.uniform(.001, 10) # meters NOTE this is for each of x and y, so actual will be higher
        self.maps_noise_x = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult
        self.maps_noise_y = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult

        self.negotiation_traj = np.empty((self.N_NEGOTIATION_WPS, 3), dtype=np.float32)
        self.negotiation_traj_ixs = np.empty((self.N_NEGOTIATION_WPS), dtype=int)
        self.m_per_wp = 1

        # Stopsigns
        self.stopsign_state = "NONE"
        self.stopped_counter = 0
        self.stopped_at_ix = 0
        self.stop_dist = DIST_NA_PLACEHOLDER


        # Lead car. Set in TM
        self.has_lead = False
        self.lead_dist = DIST_NA_PLACEHOLDER
        self.lead_relative_speed = 100

        ######
        # save episode info
        self.episode_info[0] = just_go_straight
        # add more when needed

        if self.save_data:
            np.save(f"{self.run_root}/episode_info.npy", self.episode_info)


    def set_route(self, route):
                                        
        self.waypoints = np.empty((len(route), 3), dtype="float64")
        self.waypoints[:,0] = moving_average(route.pos_x.to_numpy(), 30)
        self.waypoints[:,1] = moving_average(route.pos_y.to_numpy(), 30)
        self.waypoints[:,2] = moving_average(route.pos_z.to_numpy(), 20) 
        # smoothing wp z and pitch bc we don't drive them, we just use them directly. 
        # Smoothing XY of pos and rots was I believe causing a bit of spazz, don't know for certain why but removal did the trick
        # no didn't remove it, but perhaps lessened freq

        self.wp_is_stop = route.wp_is_stop.to_numpy()
        
        self.wp_normals = np.empty((len(route), 3), dtype="float64")
        self.wp_normals[:, 0] = route.normal_x.to_numpy()
        self.wp_normals[:, 1] = route.normal_y.to_numpy()
        self.wp_normals[:, 2] = route.normal_z.to_numpy()

        self.wp_rotations = np.empty((len(route), 3), dtype="float64") # pitch, roll, yaw
        self.wp_rotations[:, 0] = moving_average(route.curve_pitch.to_numpy(), 40) ## pitch
        self.wp_rotations[:, 1] = moving_average(route.rd_roll.to_numpy(), 20) #route.curve_roll.to_numpy() ## roll
        self.wp_rotations[:, 2] = route.curve_heading.to_numpy() # yaw


        START_IX = 30
        self.current_pos = self.waypoints[START_IX]
        self.current_rotation = self.wp_rotations[START_IX]
        self.distance_along_loop = START_IX * WP_SPACING

        # self.wp_uids = route.wp_uid.to_numpy()
        # self.traj_wp_uids =  self.wp_uids[START_IX:START_IX+1]
        # self.current_wp_uid =  self.wp_uids[START_IX]
        self.traj_granular = self.waypoints[START_IX:START_IX+1]

        self.start_section_id = route.wps_section_id.iloc[0]
        self.visited_intersections = list(route.intersection_id.unique())

        # for drawing route wps on map
        ROUTE_WP_SPACING = 6 #20 # meters
        wps_per_route_wp = int(ROUTE_WP_SPACING/WP_SPACING)
        is_route_wps = np.empty(len(route), dtype=bool)
        is_route_wps[:] = False
        is_route_wps[::wps_per_route_wp] = True
        self.is_route_wps = is_route_wps
        self.route_x_offset, self.route_y_offset = random.uniform(-4,4), random.uniform(-4,4)

    def set_nav_map(self, coarse_map_df):
        self.way_ids = coarse_map_df.way_id.to_numpy()
        self.map_xs, self.map_ys = coarse_map_df.pos_x.to_numpy(), coarse_map_df.pos_y.to_numpy()
        n_noise_rds = 0 if random.random()<.1 else random.randint(5, 25)
        self.map_ys, self.map_xs, self.way_ids = add_noise_rds_to_map(self.map_ys, self.map_xs, self.way_ids, n_noise_rds=n_noise_rds)

        self.refresh_nav_map_freq = random.choice([3,4,5]) # alternatively, can use vehicle speed and heading to interpolate ala kalman TODO
        self.small_map = None # doesn't update every frame

        self.heading_tracker = HeadingTracker()

    def set_should_yield(self, should_yield):
        self.should_yield = should_yield
        
    def step(self):

        sec_to_undagger = 2 # TODO shift should prob be variable, in which case this should also be variable as a fn of shift
        meters_to_undagger = self.current_speed_mps * sec_to_undagger
        meters_to_undagger = max(meters_to_undagger, 5)

        current_wp_ix = int(round(self.distance_along_loop / WP_SPACING, 0))
        cam_normal = self.wp_normals[current_wp_ix]


        # getting targets
        wp_ixs = current_wp_ix + TRAJ_WP_IXS

        if wp_ixs[-1] >= len(self.waypoints):
            self.route_is_done = True
            print("ROUTE IS DONE", self.ap_id)
            return

        # # storing info for traffic manager to use in determining lead car, other
        # self.current_wp_uid = self.wp_uids[current_wp_ix]
        # if self.is_ego: # doing this for perf. NPCs don't need this.
        #     self.traj_wp_uids = self.wp_uids[current_wp_ix:current_wp_ix+int(LEAD_LOOKAHEAD_DIST/WP_SPACING)] # filled-out version of what we're using for targets. First wp is car loc.
        if self.is_ego: # doing this for perf. NPCs don't need this.
            self.traj_granular = self.waypoints[current_wp_ix:current_wp_ix+int(LEAD_LOOKAHEAD_DIST/WP_SPACING)] # filled-out version of what we're using for targets. First wp is car loc.

        traj_wps = self.waypoints[wp_ixs]
        deltas = traj_wps - self.current_pos
        xs, ys = deltas[:, 0], deltas[:, 1]

        if abs(self.shift_x) > 0 or abs(self.shift_y) > 0:
            perc_into_undaggering = TRAJ_WP_DISTS_NP / meters_to_undagger 
            p = np.clip(linear_to_sin_decay(perc_into_undaggering), 0, 1)
            xs += self.shift_x*p
            ys += self.shift_y*p

        angles_to_wps = get_angles_to(xs, ys, -self.current_rotation[2])
        wp_dists_actual = np.sqrt(xs**2 + ys**2)

        if self.save_data:
            # Navmap
            # gps doesn't refresh as fast as do frames
            if self.overall_frame_counter % self.refresh_nav_map_freq == 0: # This always has to be called on first frame otherwise small_map is none
                close_buffer = CLOSE_RADIUS
                current_x = self.current_pos[0] + self.maps_noise_x[self.overall_frame_counter]
                current_y = self.current_pos[1] + self.maps_noise_y[self.overall_frame_counter]
                heading_for_map = self.heading_tracker.step(x=current_x, 
                                                        y=current_y, 
                                                        current_speed_mps=self.current_speed_mps) + self.maps_noise_heading[self.overall_frame_counter]                 

                BEHIND_BUFFER_M, FORWARD_BUFFER_M = 30, 400
                _min = max(0, current_wp_ix-int(BEHIND_BUFFER_M/WP_SPACING))
                _max = current_wp_ix+int(FORWARD_BUFFER_M/WP_SPACING)
                route_wps = self.waypoints[_min:_max]
                is_route_wps_ixs = self.is_route_wps[_min:_max]
                route_wps = route_wps[is_route_wps_ixs]
                route_xs, route_ys = route_wps[:, 0]+self.route_x_offset, route_wps[:, 1]+self.route_y_offset

                self.small_map = get_map(self.map_xs, 
                                        self.map_ys, 
                                        self.way_ids, 
                                        route_xs,
                                        route_ys,
                                        current_x, 
                                        current_y,
                                        heading_for_map,
                                        close_buffer,
                                        self.draw_route)

            wp_rolls = self.wp_rotations[wp_ixs, 1]

            # saving
            c = self.counter
            self.maps_container[c,:,:,:] = self.small_map

            self.targets_container[c, :N_WPS] = angles_to_wps
            self.targets_container[c, N_WPS:N_WPS*2] = wp_dists_actual
            self.targets_container[c, N_WPS*2:N_WPS*3] = wp_rolls
            self.targets_container[c, N_WPS*3:] = deltas[:,2] # z delta w respect to ego

            # aux
            self.aux_container[c, AUX_PITCH_IX] = self.pitch_perturbation
            self.aux_container[c, AUX_YAW_IX] = self.yaw_perturbation
            self.aux_container[c, AUX_SPEED_IX] = mps_to_kph(self.current_speed_mps)
            self.aux_container[c, AUX_TIRE_ANGLE_IX] = self.current_tire_angle

            self.aux_container[c, AUX_APPROACHING_STOP_IX] = self.stopsign_state=="APPROACHING_STOP" 
            self.aux_container[c, AUX_STOPPED_IX] = self.stopsign_state=="STOPPED"
            self.aux_container[c, AUX_STOP_DIST_IX] = self.stop_dist

            self.aux_container[c, AUX_HAS_LEAD_IX] = self.has_lead
            self.aux_container[c, AUX_LEAD_DIST_IX] = self.lead_dist
            self.aux_container[c, AUX_LEAD_SPEED_IX] = self.lead_relative_speed

            self.aux_container[c, AUX_SHOULD_YIELD_IX] = self.should_yield

            if (c+1) == SEQ_LEN:
                np.save(f"{self.run_root}/aux_{self.overall_frame_counter}.npy", self.aux_container)  
                np.save(f"{self.run_root}/targets_{self.overall_frame_counter}.npy", self.targets_container)
                np.save(f"{self.run_root}/maps_{self.overall_frame_counter}.npy", self.maps_container)  

                next_seq_path = f"{self.run_root}/targets_{self.overall_frame_counter+SEQ_LEN}.npy"
                if os.path.exists(next_seq_path):
                    os.remove(next_seq_path)

                self.targets_container[:,:] = 0.0 # TODO shouldn't need this, but it's for safety?
                self.aux_container[:,:] = 0.0
                self.counter = 0
            else:
                self.counter += 1
        
        self.overall_frame_counter += 1

        ############
        # target wp close to vehicle, used for steering AP to keep it tight on traj
        # This just follows the cos dagger traj directly. This is the wp we aim towards, not the wp we use for targets
        CLOSE_WP_DIST = np.interp(self.current_speed_mps, [9, 30], [3.0, 4.5])
        target_wp_ix = current_wp_ix + int(round(CLOSE_WP_DIST/WP_SPACING))
        wp = self.waypoints[target_wp_ix]
        if abs(self.shift_x)>0 or abs(self.shift_y)>0:
            wp = [wp[0] + self.shift_x, wp[1] + self.shift_y, wp[2]]

        deltas = wp - self.current_pos
        xs, ys = deltas[0:1], deltas[1:2]
        angle_to_target_wp_ap = get_angles_to(xs, ys, -self.current_rotation[2])[0] # blender counterclockwise is positive

        self.target_wp_pos = wp
        self.target_wp_rot = self.wp_rotations[target_wp_ix]

        #############
        # DAGGER setting

        DAGGER_DURATION = sec_to_undagger*2*FPS # frames. TODO this doesn't need to be tied w sec_to_undagger

        shift_x_max = cam_normal[0]*self.normal_shift
        shift_y_max = cam_normal[1]*self.normal_shift

        DAGGER_MIN_SPEED = mph_to_mps(12)
        if self.is_ego and (self.overall_frame_counter % self.DAGGER_FREQ == 0) and (self.current_speed_mps > DAGGER_MIN_SPEED):
            self.is_doing_dagger = True
        if self.is_doing_dagger:
            self.dagger_counter += 1
            r = linear_to_cos(self.dagger_counter/DAGGER_DURATION)
            
            self.shift_x = r * shift_x_max
            self.shift_y = r * shift_y_max
            
        if self.dagger_counter == DAGGER_DURATION: self.reset_dagger()
                
        ###############
        # Moving the car

        # speed limit and turn agg
        if self.overall_frame_counter % self.DRIVE_STYLE_CHANGE_IX: self.reset_drive_style()

        fps = np.clip(random.gauss(20, 2), 17, 23)

        dist_car_travelled = self.current_speed_mps * (1 / fps) if self.current_speed_mps>0 else 0

        # always using close wp for ap, to keep right on traj
        self.current_tire_angle = angle_to_target_wp_ap * self.lateral_kP # TODO should update this incrementally by kP?
        _wheelbase = 1.5 #CRV_WHEELBASE # NOTE this isn't correct, but i want ego to be turning faster, otherwise taking turns too wide
        _vehicle_turn_rate = self.current_tire_angle * (self.current_speed_mps/_wheelbase) # rad/sec # we're using target wp angle as the tire_angle
        vehicle_heading_delta = _vehicle_turn_rate * (1/fps) # radians


        # we get slightly out of sync bc our wps are going along based on curve, but we're controlling cube like a vehicle. 
        # Keeping totally synced so wp angles remain perfect.
        correction = (MIN_WP_M - wp_dists_actual[0])*.25 # TODO don't like this
        dist_car_travelled_corrected = dist_car_travelled + correction
        self.distance_along_loop += dist_car_travelled_corrected

        _current_heading = self.current_rotation[2]+(np.pi/2) # TODO can rationalize this away
        angle_to_future_vehicle_loc = _current_heading - (vehicle_heading_delta/2) # this isn't the exact calc. See rollout_utils for more accurate version
        delta_x = dist_car_travelled*np.cos(angle_to_future_vehicle_loc)
        delta_y = (dist_car_travelled*np.sin(angle_to_future_vehicle_loc))

        self.current_rotation[0] = self.wp_rotations[current_wp_ix][0] # Setting pitch manually based on wps, ie not 'driving' it
        self.current_rotation[1] = self.wp_rotations[current_wp_ix][1] + self.roll_noise[current_wp_ix]
        self.current_rotation[2] -= vehicle_heading_delta

        self.current_pos[0] += delta_x
        self.current_pos[1] += delta_y
        self.current_pos[2] = self.waypoints[current_wp_ix][2] # Just setting this manually to the wp itself. Not "driving" it like XY

        ###############
        # Update speed

        # ccs
        curvature_constrained_speed = get_curve_constrained_speed_from_wp_angles(angles_to_wps, wp_dists_actual, self.current_speed_mps, max_accel=self.max_accel)
        curvature_constrained_speed *= self.curve_speed_mult
        

        # stopsign constrained speed
        STOPSIGN_DECEL = 2.0
        EXTRA_STOP_BUFFER_M = 1.5
        STOPPED_SPEED = mph_to_mps(1.6)
        # n_advance_wps_for_stop = int((STOP_LOOKAHEAD_SEC * self.current_speed_mps) / WP_SPACING)
        # n_advance_wps_for_stop = max(n_advance_wps_for_stop, int(MIN_STOP_LOOKAHEAD_M/WP_SPACING))
        n_advance_wps_for_stop = int(STOP_LOOKAHEAD_DIST / WP_SPACING) # always looking 100 m ahead
        upcoming_wps_is_stop = self.wp_is_stop[current_wp_ix:current_wp_ix+n_advance_wps_for_stop]

        # Normal stopsign logic
        # NONE -> "APPROACHING_STOP" or "STOPPED"
        if True in upcoming_wps_is_stop and not self.stopsign_state=="LEAVING_STOP":
            stop_ix = np.where(upcoming_wps_is_stop==True)[0][0]
            stop_dist = abs(stop_ix * WP_SPACING - EXTRA_STOP_BUFFER_M)
            stop_dist = stop_dist if stop_dist > .1 else 0
            stop_sign_constrained_speed = np.sqrt(STOPSIGN_DECEL * stop_dist) 
            # the fastest we can be going now in order to hit zero m/s in the given dist at given decel
            self.stopsign_state = "APPROACHING_STOP" if self.current_speed_mps > STOPPED_SPEED else "STOPPED"
            self.stop_dist = stop_dist
        else:
            stop_sign_constrained_speed = 1e6
            stop_dist = DIST_NA_PLACEHOLDER
        
        # if stopped, increment counter
        # STOPPED -> LEAVING_STOP
        if self.stopsign_state=="STOPPED": 
            # Once stopped at stopsign long enough, proceed
            if self.stopped_counter >= (self.pause_at_stopsign_s * FPS):
                self.stopsign_state = "LEAVING_STOP"
                self.stopped_at_ix = current_wp_ix
                self.stopped_counter = 0
            else:
                self.stopped_counter += 1

        # if recently stopped, but now leaving stop-zone, reset stopsign apparatus
        # LEAVING_STOP -> NONE
        RESET_STOPSIGN_M = 5
        if self.stopsign_state == "LEAVING_STOP" and (current_wp_ix-self.stopped_at_ix) > RESET_STOPSIGN_M/WP_SPACING:
            self.stopsign_state = "NONE"

        if self.stopsign_state in ["APPROACHING_STOP", "STOPPED"]: 
            self.is_in_yield_zone = True

        # if ego runs the stopsign, above apparatus doesn't work, so just reset to NONE
        if True not in upcoming_wps_is_stop:
            self.stopsign_state = "NONE"
            stop_sign_constrained_speed = 1e6
            self.stop_dist = DIST_NA_PLACEHOLDER
            self.stopped_counter = 0
            self.stopped_at_ix = None

        # if self.is_ego:
        #     if self.has_lead:
        #         print(f"Lead dist {round(self.lead_dist, 2)} speed {round(self.lead_relative_speed, 2)}")
        #     if self.stopsign_state != "NONE":
        #         print(self.stopsign_state, round(stop_dist, 2))


        if self.obeys_stops:
            target_speed = min([curvature_constrained_speed, self.speed_limit, stop_sign_constrained_speed]) 
        else:
            target_speed = min(curvature_constrained_speed, self.speed_limit)

        target_speed = target_speed if not self.should_yield else 0

        max_accel_frame = self.max_accel / FPS # 
        delta = target_speed - self.current_speed_mps
        delta = np.clip(delta, -max_accel_frame, max_accel_frame)
        self.current_speed_mps += delta #(delta)*self.long_kP

        # negotiation traj
        m_per_wp = self.current_speed_mps * self.S_PER_WP
        frame_steps_per_wp = max(int(round(m_per_wp / WP_SPACING)), 1)
        self.negotiation_traj_ixs[:] = list(range(0, frame_steps_per_wp*self.N_NEGOTIATION_WPS, frame_steps_per_wp))
        negotiation_wps_ixs = current_wp_ix + self.negotiation_traj_ixs
        self.negotiation_traj[:,:] = self.waypoints[negotiation_wps_ixs, :]
        self.m_per_wp = m_per_wp


    def reset_dagger(self):
        self.is_doing_dagger = False
        self.dagger_counter = 0
        self.shift_x, self.shift_y = 0, 0
        self.normal_shift = NORMAL_SHIFT_MAX if random.random()<.5 else -NORMAL_SHIFT_MAX

    def reset_drive_style(self):
        rr = random.random() 
        if self.is_highway:
            self.speed_limit = random.uniform(16, 19) if rr<.1 else random.uniform(19, 25) if rr < .7 else random.uniform(25, 27)
        else:
            self.speed_limit = random.uniform(8, 12) if rr < .1 else random.uniform(12, 18) if rr < .2 else random.uniform(18, 26) # mps

        if not self.is_ego:
            self.speed_limit *= .8 # hack to make npcs go a bit slower than ego, to get more time w npcs as lead car

        self.lateral_kP = .9 #.85 #random.uniform(.75, .95)
        # self.long_kP = .5 #random.uniform(.02, .05)
        self.curve_speed_mult = random.uniform(.7, 1.25)
        self.turn_slowdown_sec_before = random.uniform(.25, .75)
        self.max_accel = random.uniform(1.5, 4.5)
        self.pause_at_stopsign_s = 0 if random.random()<.5 else random.randint(0,4)

        OBEYS_STOPS_PROB = .1
        self.obeys_stops = random.random()<OBEYS_STOPS_PROB