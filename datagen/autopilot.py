import numpy as np
import os, random, sys, bpy, time, glob

sys.path.append("/home/beans/bespoke")
from constants import *
from traj_utils import *
from map_utils import *

class Autopilot():
    def __init__(self, episode_info, run_root=None, ap_id=None, is_ego=False):
        self.episode_info, self.run_root, self.ap_id, self.is_ego = episode_info, run_root, ap_id, is_ego

        self.reset_drive_style()
        self.overall_frame_counter = 0 # bc incrementing before save. 1 here corresponds to frame 1, the first frame

        self.DRIVE_STYLE_CHANGE_IX = random.randint(300, 600)
        self.DAGGER_FREQ = random.randint(500, 800) # NOTE min can't exceed dagger duration
        self.dagger_freq_offset = random.randint(0, 10_000)

        # doing these here so they don't change mid stopsign
        self.pause_at_stopsign_s = 0 if random.random()<.5 else random.randint(0,4)
        OBEYS_STOPS_PROB = .3
        self.obeys_stops = random.random()<OBEYS_STOPS_PROB

        self.reset_dagger()

        if self.is_ego:
            self.targets_container = get_targets_container(1,1)[0] # angles, dists, rolls, z deltas
            self.aux = get_aux_container(1,1)[0] # returns w batch, but just seq here
            self.maps_container = get_maps_container(1, 1)[0]

            ######
            # Noise for the map
            num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity. can go even lower here, eg to mimic bumpy gravel rd
            ROLL_MAX_DEG = 1.2
            do_roll = random.random() < .4
            roll_noise_mult = random.uniform(.001, np.radians(ROLL_MAX_DEG)) if do_roll else 0
            self.roll_noise = get_random_roll_noise(num_passes=num_passes) * roll_noise_mult
            episode_info.roll_noise_mult = roll_noise_mult

            # noise for the map, position 
            BAD_GPS_PROB = .1
            if random.random()<BAD_GPS_PROB:
                maps_noise_mult = 60 # NOTE for each x and y, so actual will be higher
                min_num_passes_exp = np.interp(maps_noise_mult, [0, 10, 60], [1, 2, 3])
                num_passes = int(3 * 10**random.uniform(min_num_passes_exp, 3)) # more passes makes for longer periodocity
                self.maps_noise_x = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult # actually a bit time-consuming at higher num_passes
                self.maps_noise_y = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult
            else:
                self.maps_noise_x = np.zeros_like(self.roll_noise)
                self.maps_noise_y = np.zeros_like(self.roll_noise)
                maps_noise_mult = 0
            
            episode_info.maps_noise_mult = maps_noise_mult
            
            # #
            # tire_angle_noise_mult = random.uniform(.001, .015) # radians
            # self.tire_angle_noise = get_random_roll_noise(num_passes=random.randint(30,300)) * tire_angle_noise_mult
            # self.tire_angles_hist = [0 for _ in range(6)]
            # self.tire_angles_lag = random.randint(2,6) # two is most recent obs

            #
            self.EMPTY_MAP = np.zeros_like(self.maps_container[0])
            
        self.small_map = None
        self.should_yield = False
        self.route_is_done = False

        self.is_rando_yielding = False
        self.rando_yield_counter = 0
        self.RANDO_YIELD_FREQ = random.randint(600, 800) if not (episode_info.is_highway or random.random()<.4) else 100_000
        self.RANDO_YIELD_DURATION = random.randint(30, 120)
        self.rando_yield_offset = random.randint(0, 10_000)

        self.N_NEGOTIATION_WPS = 100
        NEGOTIATION_TRAJ_LEN_SEC = random.uniform(2.5, 3.5) # follow dist. TODO uncouple follow dist and negotiation traj
        self.S_PER_WP = NEGOTIATION_TRAJ_LEN_SEC / self.N_NEGOTIATION_WPS

        # 
        self.negotiation_traj = np.empty((self.N_NEGOTIATION_WPS, 3), dtype=np.float32)
        self.negotiation_traj_ixs = np.empty((self.N_NEGOTIATION_WPS), dtype=int)
        self.m_per_wp = 1

        # Stopsigns
        self.stopsign_state = "NONE"
        self.stopped_counter = 0
        self.stopped_at_ix = 0
        self.stop_dist = DIST_NA_PLACEHOLDER

        # Lead car. Set in TM
        self.lead_dist = DIST_NA_PLACEHOLDER
        self.lead_relative_speed = 100
        self.is_in_yield_zone = False

        self.should_yield = False

        self.wp_keep_up_correction = 0



    def set_route(self, route):

        self.waypoints = np.empty((len(route), 3), dtype="float64")
        XY_SMOOTH = 160 #120 #60 # 180 visibly cuts corners a bit, but still not as much as humans. less than 160 is not sharp enough w some settings
        # NOTE will have to be wary of this smoothing when we're doing intersections, as this may be too much. Also our close wp dist.
        self.waypoints[:,0] = moving_average(route.pos_x.to_numpy(), XY_SMOOTH)
        self.waypoints[:,1] = moving_average(route.pos_y.to_numpy(), XY_SMOOTH)
        self.waypoints[:,2] = moving_average(route.pos_z.to_numpy(), 20) 
        # smoothing wp z and pitch bc we don't drive them, we just use them directly. 
        # Smoothing XY of pos and rots was I believe causing a bit of spazz, don't know for certain why but removal did the trick
        # no didn't remove it, but perhaps lessened freq

        self.wp_is_stop = route.wp_is_stop.to_numpy()
        STOP_MOVE_BACK_M = 5.
        STOP_MOVE_BACK = int(STOP_MOVE_BACK_M / WP_SPACING)
        self.wp_is_stop[:-STOP_MOVE_BACK] = self.wp_is_stop[STOP_MOVE_BACK:]
        
        self.wp_normals = np.empty((len(route), 3), dtype="float64")
        self.wp_normals[:, 0] = route.normal_x.to_numpy()
        self.wp_normals[:, 1] = route.normal_y.to_numpy()
        self.wp_normals[:, 2] = route.normal_z.to_numpy()

        self.wp_rotations = np.empty((len(route), 3), dtype="float64") # pitch, roll, yaw
        self.wp_rotations[:, 0] = moving_average(route.curve_pitch.to_numpy(), 40) ## pitch
        self.wp_rotations[:, 1] = moving_average(route.rd_roll.to_numpy(), 20) #route.curve_roll.to_numpy() ## roll
        self.wp_rotations[:, 2] = route.curve_heading.to_numpy() # yaw

        self.start_section_id = route.wps_section_id.iloc[0]
        self.visited_intersections = list(route.intersection_id.unique())

        # for drawing route wps on map
        ROUTE_WP_SPACING = 6 #20 # meters
        wps_per_route_wp = int(ROUTE_WP_SPACING/WP_SPACING)
        is_route_wps = np.empty(len(route), dtype=bool)
        is_route_wps[:] = False
        is_route_wps[::wps_per_route_wp] = True
        self.is_route_wps = is_route_wps

        self.is_left_turn = route.left_turn.to_numpy()
        self.is_right_turn = route.right_turn.to_numpy()
        if self.is_ego: print("left, right turn wps", self.is_left_turn.sum(), self.is_right_turn.sum())

        self.route_curvature = route.route_curvature.to_numpy()

        self._route_way_ids = list(route.way_id.unique())

        self.route_len = len(self.waypoints)
        self.reset_position()

    def reset_position(self):
        START_IX = 120
        self.current_pos = self.waypoints[START_IX]
        self.current_rotation = self.wp_rotations[START_IX]
        self.distance_along_loop = START_IX * WP_SPACING
        self.current_wp_ix = int(round(self.distance_along_loop / WP_SPACING, 0))
        self.traj_granular = self.waypoints[START_IX:START_IX+1]

        self.current_speed_mps = self.speed_limit / 2
        self.current_tire_angle = 0
        
    def set_nav_map(self, coarse_map_df):
        self.way_ids = coarse_map_df.way_id.to_numpy()
        self.map_xs, self.map_ys = coarse_map_df.pos_x.to_numpy(), coarse_map_df.pos_y.to_numpy()
        n_noise_rds = 0 if random.random()<.1 else random.randint(5, 25)
        self.map_ys, self.map_xs, self.way_ids = add_noise_rds_to_map(self.map_ys, self.map_xs, self.way_ids, n_noise_rds=n_noise_rds)

        self.refresh_nav_map_freq = random.choice([3,4,5]) # alternatively, can use vehicle speed and heading to interpolate ala kalman TODO
        self.small_map = self.EMPTY_MAP # doesn't update every frame

        route_df = coarse_map_df[coarse_map_df.way_id.isin(self._route_way_ids)]
        # route_df['way_id'] = pd.Categorical(route_df['way_id'], categories=self._route_way_ids, ordered=True)
        # route_df = route_df.sort_values('way_id')
        # route_df.reset_index(drop=True, inplace=True)
        self.route_xs, self.route_ys = route_df.pos_x.to_numpy(), route_df.pos_y.to_numpy()
        self.route_way_ids = route_df.way_id.to_numpy()

        self.heading_tracker = HeadingTracker()

    def set_should_yield(self, should_yield):
        self.should_yield = should_yield
        
    def _sec_to_undagger(self, dagger_shift):
        # fixed, for targets. Constant time return to proper traj based on dagger shift
        return np.interp(abs(dagger_shift), [NORMAL_SHIFT_MIN, NORMAL_SHIFT_MAX], [1, 4]) #2 NOTE these need to remain scaled

    def step(self):
        
        ##############################
        # Move car
        ##############################

        # speed limit and turn agg
        if (self.overall_frame_counter+1) % self.DRIVE_STYLE_CHANGE_IX: self.reset_drive_style()

        # Check for route done
        max_ix_will_travel_in_step = int((30/FPS) / WP_SPACING)
        is_done_lookahead_m = TRAJ_WP_DISTS[-1]+max_ix_will_travel_in_step # if self.is_ego else 20 TODO NPC shouldn't need all this traj
        if self.is_ego and self.current_wp_ix + int(is_done_lookahead_m/WP_SPACING) >= self.route_len:
            # Ego gets reset back to beginning
            # print("Ego ran out of route.")
            # self.reset_position()
            self.route_is_done = True
        elif not self.is_ego and self.current_wp_ix + int(is_done_lookahead_m/WP_SPACING) >= self.route_len: # 
            # NPCs get done
            self.route_is_done = True
            print("NPC is done", self.ap_id)
            return
        
        # if self.is_ego:
        #     print("current wp ix", self.current_wp_ix, "route len", self.route_len, "is done", self.route_is_done)

        #############
        # DAGGER setting. This is what the ap car will take. 
        # wp ix is from prev step. Car is not yet moved

        cam_normal = self.wp_normals[self.current_wp_ix]
        shift_x_max = cam_normal[0]*self.normal_shift
        shift_y_max = cam_normal[1]*self.normal_shift

        if ((self.overall_frame_counter+self.dagger_freq_offset) % self.DAGGER_FREQ == 0) and not self.is_doing_dagger: # don't want overlap
            self.is_doing_dagger = True
        if self.is_doing_dagger and self.current_speed_mps>mph_to_mps(6): # only increment dagger progress when above certain speed, otherwise holds dagger shift. Doesn't affect targets of course
            self.dagger_counter += 1
        if self.is_doing_dagger:
            r = linear_to_cos(self.dagger_counter/self.dagger_duration)
            self.shift_x = r * shift_x_max
            self.shift_y = r * shift_y_max
            self.dagger_shift = self.normal_shift * r # for logging 
        if self.dagger_counter == self.dagger_duration: self.reset_dagger()

        ############
        # Get AP wp

        CLOSE_WP_DIST = np.interp(self.current_speed_mps, [9, 30], [3., 8.]) #

        target_wp_ix = self.current_wp_ix + int(round(CLOSE_WP_DIST/WP_SPACING))
        wp = self.waypoints[target_wp_ix]
        if abs(self.shift_x)>0 or abs(self.shift_y)>0:
            wp = [wp[0] + self.shift_x, wp[1] + self.shift_y, wp[2]]

        deltas = wp - self.current_pos
        xs, ys = deltas[0:1], deltas[1:2]
        angle_to_target_wp_ap = get_angles_to(xs, ys, -self.current_rotation[2])[0] # blender counterclockwise is positive

        self.target_wp_pos = wp # Used for debugging, it's the red object leading ego
        self.target_wp_rot = self.wp_rotations[target_wp_ix]

        fps = np.clip(random.gauss(20, 2), 17, 23)

        ############
        # Move the car
        _vehicle_turn_rate = self.current_tire_angle * (self.current_speed_mps/self.wheelbase) # rad/sec # Tire angle from prev step
        vehicle_heading_delta = _vehicle_turn_rate * (1/fps) # radians
        
        _current_heading = self.current_rotation[2]+(np.pi/2)
        angle_to_future_vehicle_loc = _current_heading - (vehicle_heading_delta/2) # this isn't the exact calc. See rollout_utils for more accurate version. Approximating it w a triangle.
        dist_car_travelled = self.current_speed_mps * (1 / fps) if self.current_speed_mps>0 else 0
        delta_x = dist_car_travelled*np.cos(angle_to_future_vehicle_loc)
        delta_y = (dist_car_travelled*np.sin(angle_to_future_vehicle_loc))

        self.distance_along_loop += (dist_car_travelled+self.wp_keep_up_correction) 
        # keep-up correction bc we're moving vehicle manually, need it to remain aligned as we increment along the route
        # correction is the value from last step

        # Updating current wp ix
        self.current_wp_ix = int(round(self.distance_along_loop / WP_SPACING, 0))

        self.current_rotation[0] = self.wp_rotations[self.current_wp_ix][0] # Setting pitch manually based on wps, ie not 'driving' it
        self.current_rotation[1] = self.wp_rotations[self.current_wp_ix][1] 
        self.current_rotation[2] -= vehicle_heading_delta

        if self.is_ego:
            self.current_rotation[1] += self.roll_noise[self.current_wp_ix]

        self.current_pos[0] += delta_x
        self.current_pos[1] += delta_y
        self.current_pos[2] = self.waypoints[self.current_wp_ix][2] # Just setting this manually to the wp itself. Not "driving" it like XY

        # updating tire angle for next round. The tire angle used for driving ap, not the one recorded in aux, which is further out
        self.current_tire_angle += (angle_to_target_wp_ap - self.current_tire_angle) * self.lateral_kP # update tire angle for next round
        
        ############
        # get target WPs
        self.wp_ixs = self.current_wp_ix + TRAJ_WP_IXS

        if self.is_ego: # doing this for perf. NPCs don't need this.
            LEAD_OUTER_DIST = 100
            self.traj_granular = self.waypoints[self.current_wp_ix:self.current_wp_ix+int(LEAD_OUTER_DIST/WP_SPACING)] # filled-out version of what we're using for targets. First wp is car loc.

        traj_wps = self.waypoints[self.wp_ixs]
        self.deltas = traj_wps - self.current_pos
        xs, ys = self.deltas[:, 0], self.deltas[:, 1]
        xs_d, ys_d = xs.copy(), ys.copy()

        # dagger targets take ego smoothly back to proper traj
        if abs(self.dagger_shift):
            sec_to_undagger = self._sec_to_undagger(self.dagger_shift)
            meters_to_undagger = self.current_speed_mps * sec_to_undagger
            meters_to_undagger = max(meters_to_undagger, 12)
            
            perc_into_undaggering = np.clip(TRAJ_WP_DISTS_NP/meters_to_undagger, 0, 1)
            p = linear_to_sin_decay(perc_into_undaggering)
            xs += self.shift_x*p
            ys += self.shift_y*p

            xs_d += self.shift_x # used to get a rough idea of where the actual tire is angled. Not very good.
            ys_d += self.shift_y

        self.angles_to_wps = get_angles_to(xs, ys, -self.current_rotation[2]) # dagger corrected, used for targets
        self.angles_to_wps_d = get_angles_to(xs_d, ys_d, -self.current_rotation[2]) # not corrected, used to get tire angle

        self.wp_dists_actual = np.sqrt(xs**2 + ys**2)
        
        self.wp_keep_up_correction = (MIN_WP_M - self.wp_dists_actual[0])*.05 

        ###############
        # Update speed

        # ccs
        curvature_constrained_speed = get_curve_constrained_speed_from_wp_angles(self.angles_to_wps, self.wp_dists_actual, self.current_speed_mps, max_accel=self.max_accel)
        curvature_constrained_speed *= self.curve_speed_mult
        
        # stopsign constrained speed
        STOPPED_SPEED = mph_to_mps(1.6)
        STOP_OUTER_DIST = 80
        n_advance_wps_for_stop = int(STOP_OUTER_DIST / WP_SPACING) # always looking 100 m ahead
        upcoming_wps_is_stop = self.wp_is_stop[self.current_wp_ix:self.current_wp_ix+n_advance_wps_for_stop]

        # Normal stopsign logic
        # NONE -> "APPROACHING_STOP" or "STOPPED"
        if True in upcoming_wps_is_stop and not self.stopsign_state=="LEAVING_STOP":
            stop_ix = np.where(upcoming_wps_is_stop==True)[0][0]
            stop_dist = abs(stop_ix * WP_SPACING)
            stop_dist = stop_dist if stop_dist > .1 else 0
            stop_sign_constrained_speed = np.sqrt(self.max_accel * stop_dist) 
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
                self.stopped_at_ix = self.current_wp_ix
                self.stopped_counter = 0
            else:
                self.stopped_counter += 1

        # if recently stopped, but now leaving stop-zone, reset stopsign apparatus
        # LEAVING_STOP -> NONE
        RESET_STOPSIGN_M = 10
        if self.stopsign_state == "LEAVING_STOP" and (self.current_wp_ix-self.stopped_at_ix) > RESET_STOPSIGN_M/WP_SPACING:
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
        #     if self.lead_dist < DIST_NA_PLACEHOLDER:
        #         print(f"Lead dist {round(self.lead_dist, 2)} speed {round(self.lead_relative_speed, 2)}")
        #     if self.stopsign_state != "NONE":
        #         print(self.stopsign_state, round(stop_dist, 2), round(stop_sign_constrained_speed, 2))
        #     if self.is_rando_yielding:
        #         print("Rando yielding", self.rando_yield_counter)
        #     if abs(self.dagger_shift) > .0:
        #         print("Dagger shift", self.dagger_shift, self.dagger_counter)

        if self.obeys_stops:
            target_speed = min([curvature_constrained_speed, self.speed_limit, stop_sign_constrained_speed]) 
        else:
            target_speed = min(curvature_constrained_speed, self.speed_limit)

        # rando yielding to break speed / stopsign corr
        if ((self.overall_frame_counter+self.rando_yield_offset) % self.RANDO_YIELD_FREQ == 0):
            self.is_rando_yielding = True
        if self.is_rando_yielding:
            self.rando_yield_counter += 1
        if self.rando_yield_counter > self.RANDO_YIELD_DURATION:
            self.is_rando_yielding = False
            self.rando_yield_counter = 0

        target_speed = target_speed if not (self.should_yield or self.is_rando_yielding) else 0
        max_accel_frame = self.max_accel / FPS # 
        delta = target_speed - self.current_speed_mps
        delta = np.clip(delta, -max_accel_frame, max_accel_frame)
        self.current_speed_mps += delta #(delta)*self.long_kP
        self.current_speed_mps = np.clip(self.current_speed_mps, 0, np.inf)

        if self.route_is_done:
            self.current_speed_mps = 0 # Just emergency stop at end of route

        #############
        # negotiation traj
        m_per_wp = self.current_speed_mps * self.S_PER_WP
        min_traj_m = 12
        m_per_wp = max(m_per_wp, min_traj_m/self.N_NEGOTIATION_WPS)
        frame_steps_per_wp = max(int(round(m_per_wp / WP_SPACING)), 1)
        self.negotiation_traj_ixs[:] = list(range(0, frame_steps_per_wp*self.N_NEGOTIATION_WPS, frame_steps_per_wp))
        negotiation_wps_ixs = self.current_wp_ix + self.negotiation_traj_ixs
        self.negotiation_traj[:,:] = self.waypoints[negotiation_wps_ixs, :]
        self.m_per_wp = m_per_wp

        # Navmap
        # gps doesn't refresh as fast as do frames. 
        if self.is_ego and self.overall_frame_counter%self.refresh_nav_map_freq==0:

            self.current_x = self.current_pos[0] + self.maps_noise_x[self.overall_frame_counter]
            self.current_y = self.current_pos[1] + self.maps_noise_y[self.overall_frame_counter] 
            self.heading_for_map = self.heading_tracker.step(x=self.current_x, 
                                                    y=self.current_y, 
                                                    current_speed_mps=self.current_speed_mps) #+ self.maps_noise_heading[self.overall_frame_counter]                 


        self.overall_frame_counter += 1



    #if self.save_data and self.overall_frame_counter%FRAME_CAPTURE_N==0:
    def save_data(self):
        # print("Saving data", self.overall_frame_counter)
        
        # rd maneuvers
        RD_MANEUVER_LOOKAHEAD = 60 # meters
        left_turn = any(self.is_left_turn[self.current_wp_ix:self.current_wp_ix+int(RD_MANEUVER_LOOKAHEAD/WP_SPACING)])
        right_turn = any(self.is_right_turn[self.current_wp_ix:self.current_wp_ix+int(RD_MANEUVER_LOOKAHEAD/WP_SPACING)])

        ego_in_intx = any(self.is_left_turn[self.current_wp_ix:self.current_wp_ix+int(8/WP_SPACING)]) or \
                        any(self.is_right_turn[self.current_wp_ix:self.current_wp_ix+int(8/WP_SPACING)])
            
        # Maps
        HAS_MAP_PROB = .5 if self.episode_info.just_go_straight else 1
        HAS_ROUTE_PROB = .5 if self.episode_info.just_go_straight else 1
        if self.overall_frame_counter < 10: HAS_MAP_PROB, HAS_ROUTE_PROB = 0, 0 # First few obs, no maps bc heading tracker janky and rds cutoff
        
        self.has_map = random.random() < HAS_MAP_PROB
        self.has_route = self.has_map and random.random()<HAS_ROUTE_PROB # keeping inside here to keep always synced w aux, bc gps hertz slower
    
        self.small_map = get_map(self.map_xs, 
                                self.map_ys, 
                                self.way_ids, 
                                self.route_xs,
                                self.route_ys,
                                self.route_way_ids,
                                self.current_x, 
                                self.current_y,
                                self.heading_for_map,
                                CLOSE_RADIUS,
                                draw_route=self.has_route) if self.has_map else self.EMPTY_MAP

        # will get repeated ~4 times in a row bc gps less hz
        c = 0
        self.maps_container[c,:,:,:] = self.small_map 
        self.aux[c, "has_map"] = int(self.has_map)
        self.aux[c, "has_route"] = int(self.has_route)
        self.aux[c, "heading"] = self.heading_tracker.heading
        # self.aux[c, "pos_x"] = self.current_pos[0] # Doesn't have noise, but should. Refine this if important. Shouldn't update every step.
        # self.aux[c, "pos_y"] = self.current_pos[1]

        # Wps
        self.targets_container[c, :N_WPS] = self.angles_to_wps
        self.targets_container[c, N_WPS:N_WPS*2] = self.wp_dists_actual
        self.targets_container[c, N_WPS*2:N_WPS*3] = self.wp_rotations[self.wp_ixs, 1] # wp rolls
        self.targets_container[c, N_WPS*3:] = self.deltas[:,2] # z delta w respect to ego

        # aux
        # angle_to_wp = get_target_wp_angle(self.angles_to_wps, self.current_speed_mps) # gathering this here for convenience, just for logging. The target wp for rw driver, not the close up one ap is using.
        angle_to_wp_d = get_target_wp_angle(self.angles_to_wps_d, self.current_speed_mps) # gathering this here for convenience, just for logging. The target wp for rw driver, not the close up one ap is using.
        # self.tire_angles_hist.append(angle_to_wp_d)

        # get_clf_range = lambda s : np.clip(s*5.0, 40., 100) # looking five seconds out, but clamped. TODO should prob be more sec ahead
        # r = get_clf_range(self.current_speed_mps)

        stop_angle = angle_to_wp_from_dist_along_traj(self.angles_to_wps, self.stop_dist)
        lead_angle = angle_to_wp_from_dist_along_traj(self.angles_to_wps, self.lead_dist)
        
        self.aux[c, "speed"] = self.current_speed_mps
        self.aux[c, "tire_angle"] = angle_to_wp_d #self.current_tire_angle
        self.aux[c, "tire_angle_ap"] = self.current_tire_angle # actual tire angle used for ap steering
        self.aux[c, "has_stop"] = self.stop_dist < STOP_DIST_MAX and (abs(stop_angle) < .6) #smooth_dist_clf(self.stop_dist, 60, 80) #self.stopsign_state=="APPROACHING_STOP"
        self.aux[c, "stop_dist"] = self.stop_dist
        self.aux[c, "has_lead"] = (self.lead_dist < LEAD_DIST_MAX) and (abs(lead_angle) < .6) # .6 is directly out of frame #smooth_dist_clf(self.lead_dist, 80, 100)
        self.aux[c, "lead_dist"] = self.lead_dist
        self.aux[c, "lead_speed"] = self.lead_relative_speed
        self.aux[c, "should_yield"] = self.should_yield
        self.aux[c, "dagger_shift"] = self.dagger_shift
        self.aux[c, "left_turn"] = left_turn
        self.aux[c, "right_turn"] = right_turn
        self.aux[c, "curvature_at_ego"] = self.route_curvature[self.current_wp_ix]
        self.aux[c, "ego_in_intx"] = ego_in_intx

        # Passing episode info forward. These will be repeated. Not efficient but more robust -- now everything consolidated in aux.
        episode_info_dict = self.episode_info.__dict__
        for p in EPISODE_PROPS: 
            self.aux[c, p] = episode_info_dict[p]
        
        _i = self.overall_frame_counter // FRAME_CAPTURE_N
        np.save(f"{self.run_root}/aux/{_i}.npy", self.aux)
        np.save(f"{self.run_root}/targets/{_i}.npy", self.targets_container)
        np.save(f"{self.run_root}/maps/{_i}.npy", self.maps_container)  
        
       

    def reset_dagger(self):
        # Get max dagger shift

        normal_shift = random.uniform(NORMAL_SHIFT_MIN, NORMAL_SHIFT_MAX)
        self.normal_shift = normal_shift if random.random()<.5 else -normal_shift

        # reset state
        self.is_doing_dagger = False
        self.dagger_counter = 0
        self.shift_x, self.shift_y, self.dagger_shift = 0, 0, 0

        # How long dagger will take, ie this is what ap will follow. Based on normal shift, but with some randomness bc
        # doesn't need to be tied, and want agent to be able to not be too distracted when off policy human driving.
        self.dagger_duration = int(self._sec_to_undagger(normal_shift)*2*FPS*random.uniform(1., 2.)) # frames NOTE don't exceed frequency


    def reset_drive_style(self):
        rr = random.random() 
        if self.episode_info.is_highway:
            self.speed_limit = random.uniform(16, 19) if rr<.1 else random.uniform(19, 27) if rr < .7 else random.uniform(27, 30)
        else:
            self.speed_limit = random.uniform(8, 12) if rr < .1 else random.uniform(12, 18) if rr < .2 else random.uniform(18, 28) # mps

        if not self.is_ego:
            self.speed_limit *= .8 # hack to make npcs go a bit slower than ego, to get more time w npcs as lead car

        self.lateral_kP = .95 #.85 #random.uniform(.75, .95)
        # self.long_kP = .5 #random.uniform(.02, .05)
        self.curve_speed_mult = random.uniform(.7, 1.25)
        self.turn_slowdown_sec_before = random.uniform(.25, .75)
        self.max_accel = random.uniform(2.6, 3.5) #

        self.wheelbase = random.uniform(2., 2.66) # crv is 2.66, but that begins to be too slow on the turns, like a yacht



NORMAL_SHIFT_MIN = .25
NORMAL_SHIFT_MAX = 1.0