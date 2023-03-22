import numpy as np
import os, random, sys, bpy, time, glob

sys.path.append("/home/beans/bespoke")
from constants import *
from traj_utils import *
from map_utils import *

class Autopilot():
    def __init__(self, episode_info, save_data=True, run_root=None, ap_id=None, is_ego=False):
        self.episode_info, self.save_data, self.run_root, self.ap_id, self.is_ego = episode_info, save_data, run_root, ap_id, is_ego

        self.reset_drive_style()
        self.overall_frame_counter = 0
        self.current_speed_mps = self.speed_limit / 2
        self.current_tire_angle = 0
        self.DRIVE_STYLE_CHANGE_IX = random.randint(300, 600)
        self.DAGGER_FREQ = random.randint(400, 800)

        # doing these here so they don't change mid stopsign
        self.pause_at_stopsign_s = 0 if random.random()<.5 else random.randint(0,4)
        OBEYS_STOPS_PROB = .5
        self.obeys_stops = random.random()<OBEYS_STOPS_PROB

        self.reset_dagger()

        self.targets_container = get_targets_container(1,1)[0] # angles, dists, rolls, z deltas
        self.aux = get_aux_container(1,1)[0] # returns w batch, but just seq here
        self.maps_container = get_maps_container(1, 1)[0]

        self.small_map = None
        self.should_yield = False
        self.route_is_done = False

        self.is_rando_yielding = False
        self.rando_yield_counter = 0
        self.RANDO_YIELD_FREQ = random.randint(400, 800) if not (episode_info.is_highway or random.random()<.5) else 10_000
        self.RANDO_YIELD_DURATION = random.randint(30, 120)

        self.N_NEGOTIATION_WPS = 100
        NEGOTIATION_TRAJ_LEN_SEC = random.uniform(2.5, 3.5) # follow dist. TODO uncouple follow dist and negotiation traj
        self.S_PER_WP = NEGOTIATION_TRAJ_LEN_SEC / self.N_NEGOTIATION_WPS

        ######
        # Noise for the map
        num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity. can go even lower here, eg to mimic bumpy gravel rd
        ROLL_MAX_DEG = 1.2
        do_roll = random.random() < .2
        roll_noise_mult = random.uniform(.001, np.radians(ROLL_MAX_DEG)) if do_roll else 0
        self.roll_noise = get_random_roll_noise(num_passes=num_passes) * roll_noise_mult
        episode_info.roll_noise_mult = roll_noise_mult

        r = random.random()
        gps_state = "BAD" if (r < .05 and self.episode_info.just_go_straight) else "MED" if r <.1 else "NORMAL"
        # # noise for the map, heading. This is in addition to what we'll get from the noisy pos
        # num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity
        # maps_noise_mult = np.radians(10) if gps_state=="BAD" else np.radians(5) if gps_state=="MED" else random.uniform(.001, np.radians(3)) # radians
        # self.maps_noise_heading = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult

        # noise for the map, position
        maps_noise_mult = 60 if gps_state=="BAD" else 30 if gps_state=="MED" else random.uniform(.001, 10) # meters NOTE this is for each of x and y, so actual will be higher
        min_num_passes_exp = np.interp(maps_noise_mult, [0, 10, 60], [1, 2, 3])
        num_passes = int(3 * 10**random.uniform(min_num_passes_exp, 3)) # more passes makes for longer periodocity
        self.maps_noise_x = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult # actually a bit time-consuming at higher num_passes
        self.maps_noise_y = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult
        episode_info.maps_noise_mult = maps_noise_mult

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

        # 
        tire_angle_noise_mult = random.uniform(.001, .015) # radians
        self.tire_angle_noise = get_random_roll_noise(num_passes=random.randint(30,300)) * tire_angle_noise_mult
        self.tire_angles_hist = [0 for _ in range(6)]
        self.tire_angles_lag = random.randint(2,6) # two is most recent obs

        # # Final stop for episode info datagen NOTE don't use this anymore afterwards. It gets consolidated in aux
        # if self.save_data: save_object(self.episode_info, f"{self.run_root}/episode_info.pkl")
        

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

        START_IX = 120
        self.current_pos = self.waypoints[START_IX]
        self.current_rotation = self.wp_rotations[START_IX]
        self.distance_along_loop = START_IX * WP_SPACING

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

        self.route_normal_shift = random.uniform(0,3)

        self.is_left_turn = route.left_turn.to_numpy()
        self.is_right_turn = route.right_turn.to_numpy()
        if self.is_ego: print("left, right turn wps", self.is_left_turn.sum(), self.is_right_turn.sum())

        self.route_curvature = route.route_curvature.to_numpy()

        self._route_way_ids = list(route.way_id.unique())

    def set_nav_map(self, coarse_map_df):
        self.way_ids = coarse_map_df.way_id.to_numpy()
        self.map_xs, self.map_ys = coarse_map_df.pos_x.to_numpy(), coarse_map_df.pos_y.to_numpy()
        n_noise_rds = 0 if random.random()<.1 else random.randint(5, 25)
        self.map_ys, self.map_xs, self.way_ids = add_noise_rds_to_map(self.map_ys, self.map_xs, self.way_ids, n_noise_rds=n_noise_rds)

        self.refresh_nav_map_freq = random.choice([3,4,5]) # alternatively, can use vehicle speed and heading to interpolate ala kalman TODO
        self.small_map = None # doesn't update every frame

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
        return np.interp(abs(dagger_shift), [NORMAL_SHIFT_MIN, NORMAL_SHIFT_MAX], [2, 4]) #2

    def step(self):

        current_wp_ix = int(round(self.distance_along_loop / WP_SPACING, 0))
        cam_normal = self.wp_normals[current_wp_ix]

        # getting targets
        wp_ixs = current_wp_ix + TRAJ_WP_IXS

        if wp_ixs[-1] >= len(self.waypoints):
            self.route_is_done = True
            print("ROUTE IS DONE", self.ap_id)
            return

        if self.is_ego: # doing this for perf. NPCs don't need this.
            self.traj_granular = self.waypoints[current_wp_ix:current_wp_ix+int(LEAD_OUTER_DIST/WP_SPACING)] # filled-out version of what we're using for targets. First wp is car loc.

        traj_wps = self.waypoints[wp_ixs]
        deltas = traj_wps - self.current_pos
        xs, ys = deltas[:, 0], deltas[:, 1]
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

        angles_to_wps = get_angles_to(xs, ys, -self.current_rotation[2]) # dagger corrected, used for targets
        angles_to_wps_d = get_angles_to(xs_d, ys_d, -self.current_rotation[2]) # not corrected, used to get tire angle

        wp_dists_actual = np.sqrt(xs**2 + ys**2)

        # rd maneuvers
        RD_MANEUVER_LOOKAHEAD = 60 # meters
        left_turn = any(self.is_left_turn[current_wp_ix:current_wp_ix+int(RD_MANEUVER_LOOKAHEAD/WP_SPACING)])
        right_turn = any(self.is_right_turn[current_wp_ix:current_wp_ix+int(RD_MANEUVER_LOOKAHEAD/WP_SPACING)])

        if self.save_data:
            HAS_MAP_PROB = .5 if self.episode_info.just_go_straight else 1
            HAS_ROUTE_PROB = .5 if self.episode_info.just_go_straight else 1
            if self.overall_frame_counter < 10: HAS_MAP_PROB, HAS_ROUTE_PROB = 0, 0 # First few obs, no maps bc heading tracker janky and rds cutoff
            # Navmap
            # gps doesn't refresh as fast as do frames. 
            # This has to be called on first frame otherwise small_map and has_map and route indicators is None
            if self.overall_frame_counter % self.refresh_nav_map_freq == 0:
                self.has_route = random.random() < HAS_ROUTE_PROB # keeping inside here to keep always synced w aux, bc gps hertz slower
                self.has_map = random.random() < HAS_MAP_PROB
                current_x = self.current_pos[0] + self.maps_noise_x[self.overall_frame_counter]
                current_y = self.current_pos[1] + self.maps_noise_y[self.overall_frame_counter] 
                close_buffer = CLOSE_RADIUS
                heading_for_map = self.heading_tracker.step(x=current_x, 
                                                        y=current_y, 
                                                        current_speed_mps=self.current_speed_mps) #+ self.maps_noise_heading[self.overall_frame_counter]                 

                # BEHIND_BUFFER_M, FORWARD_BUFFER_M = 80, 400
                # _min = max(1, current_wp_ix-int(BEHIND_BUFFER_M/WP_SPACING)) # this has to be one. Cuts off the first route wp. Don't fully understand, it's buggy. With zero, get janky route line connecting to origin.
                # _max = current_wp_ix+int(FORWARD_BUFFER_M/WP_SPACING)
                # route_wps = self.waypoints[_min:_max]
                # route_wps_normals = self.wp_normals[_min:_max] # shift on normals, otherwise route is on right side of rd
                # is_route_wps_ixs = self.is_route_wps[_min:_max]
                # route_wps = route_wps[is_route_wps_ixs]
                # route_wps_normals = route_wps_normals[is_route_wps_ixs]
                # route_wps -= route_wps_normals*self.route_normal_shift
                # route_xs, route_ys = route_wps[:, 0], route_wps[:, 1]

                self.small_map = get_map(self.map_xs, 
                                        self.map_ys, 
                                        self.way_ids, 
                                        self.route_xs,
                                        self.route_ys,
                                        self.route_way_ids,
                                        current_x, 
                                        current_y,
                                        heading_for_map,
                                        close_buffer,
                                        draw_route=self.has_route) if self.has_map else np.zeros_like(self.maps_container[0])
            
            # Maps
            # will get repeated ~4 times in a row bc gps less hz
            c = 0
            self.maps_container[c,:,:,:] = self.small_map 
            self.aux[c, "has_map"] = int(self.has_map)
            self.aux[c, "has_route"] = int(self.has_route and self.has_map)
            self.aux[c, "heading"] = self.heading_tracker.heading
            # self.aux[c, "pos_x"] = self.current_pos[0] # Doesn't have noise, but should. Refine this if important. Shouldn't update every step.
            # self.aux[c, "pos_y"] = self.current_pos[1]

            # Wps
            self.targets_container[c, :N_WPS] = angles_to_wps
            self.targets_container[c, N_WPS:N_WPS*2] = wp_dists_actual
            self.targets_container[c, N_WPS*2:N_WPS*3] = self.wp_rotations[wp_ixs, 1] # wp rolls
            self.targets_container[c, N_WPS*3:] = deltas[:,2] # z delta w respect to ego

            # aux
            angle_to_wp = get_target_wp_angle(angles_to_wps, self.current_speed_mps) # gathering this here for convenience, just for logging. The target wp for rw driver, not the close up one ap is using.
            angle_to_wp_d = get_target_wp_angle(angles_to_wps_d, self.current_speed_mps) # gathering this here for convenience, just for logging. The target wp for rw driver, not the close up one ap is using.
            self.tire_angles_hist.append(angle_to_wp_d)

            # get_clf_range = lambda s : np.clip(s*5.0, 40., 100) # looking five seconds out, but clamped. TODO should prob be more sec ahead
            # r = get_clf_range(self.current_speed_mps)

            stop_angle = angle_to_wp_from_dist_along_traj(angles_to_wps, self.stop_dist)
            lead_angle = angle_to_wp_from_dist_along_traj(angles_to_wps, self.lead_dist)
            
            HAS_TIRE_ANGLE_PROB = 0 #.6
            self.aux[c, "speed"] = self.current_speed_mps
            self.aux[c, "tire_angle"] = angle_to_wp_d #self.current_tire_angle
            self.aux[c, "tire_angle_ap"] = self.current_tire_angle # actual tire angle used for ap steering
            self.aux[c, "tire_angle_dagger_corrected"] = angle_to_wp
            # self.aux[c, "tire_angle_lagged"] = self.tire_angles_hist[-self.tire_angles_lag] + self.tire_angle_noise[self.overall_frame_counter]
            self.aux[c, "has_tire_angle"] = int(random.random() < HAS_TIRE_ANGLE_PROB)
            self.aux[c, "has_stop"] = self.stop_dist < 60 and (abs(stop_angle) < .65) #smooth_dist_clf(self.stop_dist, 60, 80) #self.stopsign_state=="APPROACHING_STOP"
            self.aux[c, "stop_dist"] = self.stop_dist
            self.aux[c, "has_lead"] = (self.lead_dist < 80) and (abs(lead_angle) < .65) # .65 is directly out of frame #smooth_dist_clf(self.lead_dist, 80, 100)
            self.aux[c, "lead_dist"] = self.lead_dist
            self.aux[c, "lead_speed"] = self.lead_relative_speed
            self.aux[c, "should_yield"] = self.should_yield
            self.aux[c, "dagger_shift"] = self.dagger_shift
            self.aux[c, "left_turn"] = left_turn
            self.aux[c, "right_turn"] = right_turn
            self.aux[c, "curvature_at_ego"] = self.route_curvature[current_wp_ix]



            # Passing episode info forward. These will be repeated. Not efficient but more robust -- now everything consolidated in aux.
            episode_info_dict = self.episode_info.__dict__
            for p in EPISODE_PROPS: 
                self.aux[c, p] = episode_info_dict[p]
            
            np.save(f"{self.run_root}/aux/{self.overall_frame_counter}.npy", self.aux)
            np.save(f"{self.run_root}/targets/{self.overall_frame_counter}.npy", self.targets_container)
            np.save(f"{self.run_root}/maps/{self.overall_frame_counter}.npy", self.maps_container)  
        
        self.overall_frame_counter += 1

        #############
        # DAGGER setting. This is what the ap car will take. 

        shift_x_max = cam_normal[0]*self.normal_shift
        shift_y_max = cam_normal[1]*self.normal_shift

        DAGGER_MIN_SPEED = mph_to_mps(12)
        # if self.is_ego and (self.overall_frame_counter % self.DAGGER_FREQ == 0) and (self.current_speed_mps > DAGGER_MIN_SPEED) and not self.is_rando_yielding:
        if (self.overall_frame_counter % self.DAGGER_FREQ == 0) and (self.current_speed_mps > DAGGER_MIN_SPEED) and not self.is_rando_yielding:
            self.is_doing_dagger = True
        if self.is_doing_dagger:
            self.dagger_counter += 1
            r = linear_to_cos(self.dagger_counter/self.dagger_duration)
            
            self.shift_x = r * shift_x_max
            self.shift_y = r * shift_y_max
            self.dagger_shift = self.normal_shift * r # for logging 

            # print(f"Doing dagger. Counter: {self.dagger_counter}, duration: {self.dagger_duration}, dagger shift: {round(self.dagger_shift, 3)}, normal shift: {round(self.normal_shift, 3)}")
            
        if self.dagger_counter == self.dagger_duration: self.reset_dagger()

        ############
        # target wp close to vehicle, used for steering AP to keep it tight on traj
        # This just follows the cos dagger traj directly. This is the wp AP aims towards, not the wp we use for targets
        # CLOSE_WP_DIST = np.interp(self.current_speed_mps, [9, 30], [5., 11.]) #
        CLOSE_WP_DIST = np.interp(self.current_speed_mps, [9, 30], [3., 8.]) #
        # The above is an important number. If cutting turns too close in AP w traj still in middle, model thinks traj is usually on the right
        # during turns, ie it only sees turns from that perspective. Want to keep agent on traj as much as possible, while also balancing smoothness,
        # bc if too on top of target wp, jittery
        target_wp_ix = current_wp_ix + int(round(CLOSE_WP_DIST/WP_SPACING))
        wp = self.waypoints[target_wp_ix]
        if abs(self.shift_x)>0 or abs(self.shift_y)>0:
            wp = [wp[0] + self.shift_x, wp[1] + self.shift_y, wp[2]]

        deltas = wp - self.current_pos
        xs, ys = deltas[0:1], deltas[1:2]
        angle_to_target_wp_ap = get_angles_to(xs, ys, -self.current_rotation[2])[0] # blender counterclockwise is positive

        self.target_wp_pos = wp # who uses these?
        self.target_wp_rot = self.wp_rotations[target_wp_ix]

        ###############
        # Moving the car TODO this whole section can use a deep look

        # speed limit and turn agg
        if self.overall_frame_counter % self.DRIVE_STYLE_CHANGE_IX: self.reset_drive_style()

        fps = np.clip(random.gauss(20, 2), 17, 23)
        dist_car_travelled = self.current_speed_mps * (1 / fps) if self.current_speed_mps>0 else 0

        # always using close wp for ap, to keep right on traj
        _wheelbase = CRV_WHEELBASE #1.6 #CRV_WHEELBASE # NOTE CRV wheelbase may make it turn too wide. That's why was using smaller wheelbase. This interacts w kP and with target wp ix interp
        _vehicle_turn_rate = self.current_tire_angle * (self.current_speed_mps/_wheelbase) # rad/sec # Tire angle from prev step
        vehicle_heading_delta = _vehicle_turn_rate * (1/fps) # radians
        
        self.current_tire_angle += (angle_to_target_wp_ap - self.current_tire_angle) * self.lateral_kP # update tire angle for next round
        # This is not the same as tire_angle in our aux, bc there we're aiming further ahead, in same way as drive rw.

        # we get slightly out of sync bc our wps are going along based on curve, but we're controlling cube like a vehicle. 
        # Keeping totally synced so wp angles remain perfect.
        correction = (MIN_WP_M - wp_dists_actual[0])*.05 
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
        STOPPED_SPEED = mph_to_mps(1.6)
        n_advance_wps_for_stop = int(STOP_OUTER_DIST / WP_SPACING) # always looking 100 m ahead
        upcoming_wps_is_stop = self.wp_is_stop[current_wp_ix:current_wp_ix+n_advance_wps_for_stop]

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
        #     if self.lead_dist < DIST_NA_PLACEHOLDER:
        #         print(f"Lead dist {round(self.lead_dist, 2)} speed {round(self.lead_relative_speed, 2)}")
        #     if self.stopsign_state != "NONE":
        #         print(self.stopsign_state, round(stop_dist, 2), round(stop_sign_constrained_speed, 2))
        #     if self.is_rando_yielding:
        #         print("Rando yielding", self.rando_yield_counter)

        if self.obeys_stops:
            target_speed = min([curvature_constrained_speed, self.speed_limit, stop_sign_constrained_speed]) 
        else:
            target_speed = min(curvature_constrained_speed, self.speed_limit)

        # rando yielding to break speed / stopsign corr
        if (self.overall_frame_counter % self.RANDO_YIELD_FREQ == 0):
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

        # negotiation traj
        m_per_wp = self.current_speed_mps * self.S_PER_WP
        min_traj_m = 12
        m_per_wp = max(m_per_wp, min_traj_m/self.N_NEGOTIATION_WPS)
        frame_steps_per_wp = max(int(round(m_per_wp / WP_SPACING)), 1)
        self.negotiation_traj_ixs[:] = list(range(0, frame_steps_per_wp*self.N_NEGOTIATION_WPS, frame_steps_per_wp))
        negotiation_wps_ixs = current_wp_ix + self.negotiation_traj_ixs
        self.negotiation_traj[:,:] = self.waypoints[negotiation_wps_ixs, :]
        self.m_per_wp = m_per_wp


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
        self.dagger_duration = int(self._sec_to_undagger(normal_shift)*2*FPS*random.uniform(.6, 1.1)) # frames


    def reset_drive_style(self):
        rr = random.random() 
        if self.episode_info.is_highway:
            self.speed_limit = random.uniform(16, 19) if rr<.1 else random.uniform(19, 25) if rr < .7 else random.uniform(25, 27)
        else:
            self.speed_limit = random.uniform(8, 12) if rr < .1 else random.uniform(12, 18) if rr < .2 else random.uniform(18, 26) # mps

        if not self.is_ego:
            self.speed_limit *= .8 # hack to make npcs go a bit slower than ego, to get more time w npcs as lead car

        self.lateral_kP = .95 #.85 #random.uniform(.75, .95)
        # self.long_kP = .5 #random.uniform(.02, .05)
        self.curve_speed_mult = random.uniform(.7, 1.25)
        self.turn_slowdown_sec_before = random.uniform(.25, .75)
        self.max_accel = random.uniform(2.6, 3.5) #



NORMAL_SHIFT_MIN = .4
NORMAL_SHIFT_MAX = .8 #1.0