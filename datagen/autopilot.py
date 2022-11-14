import numpy as np
import os, random, sys, bpy, time, glob
import pandas as pd

sys.path.append("/home/beans/bespoke")
from constants import *
from traj_utils import *
from map_utils import *

N_NEGOTIATION_WPS = 100
NEGOTIATION_TRAJ_LEN_SEC = 3.
S_PER_WP = NEGOTIATION_TRAJ_LEN_SEC / N_NEGOTIATION_WPS

class Autopilot():
    def __init__(self, save_data=True, is_highway=False, pitch_perturbation=0, yaw_perturbation=0, run_root=None, ap_id=None, is_ego=False):
        self.is_highway, self.save_data, self.run_root, self.ap_id, self.is_ego = is_highway, save_data, run_root, ap_id, is_ego
        self.reset_drive_style()
        self.counter, self.overall_frame_counter = 0, 0
        self.current_speed_mps = self.speed_limit / 2
        self.current_tire_angle = 0
        self.DRIVE_STYLE_CHANGE_IX = random.randint(200, 600)
        self.DAGGER_FREQ = random.randint(400, 1000)
        self.reset_dagger()

        self.targets_container = np.zeros((SEQ_LEN, N_WPS*3), dtype=np.float32)
        self.aux_container = np.zeros((SEQ_LEN, N_AUX), dtype=np.float32)
        self.maps_container = np.zeros((SEQ_LEN, MAP_HEIGHT, MAP_WIDTH, 3), dtype='uint8')

        self.pitch_perturbation, self.yaw_perturbation = pitch_perturbation, yaw_perturbation
        self.small_map = None
        self.should_yield = False
        self.route_is_done = False

        ######
        # Noise for the map
        num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity. can go even lower here, eg to mimic bumpy gravel rd
        ROLL_MAX_DEG = 5
        do_roll = random.random() < .2
        roll_noise_mult = random.uniform(.001, np.radians(ROLL_MAX_DEG)) if do_roll else 0
        self.roll_noise = get_random_roll_noise(num_passes=num_passes) * roll_noise_mult

        gps_bad = random.random() < .05
        # noise for the map, heading
        num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity
        maps_noise_mult = np.radians(30) if gps_bad else random.uniform(.001, np.radians(5)) # radians
        self.maps_noise = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult

        # noise for the map, position
        num_passes = int(3 * 10**random.uniform(1, 2)) # more passes makes for longer periodocity
        maps_noise_mult = 50 if gps_bad else random.uniform(.001, 10) # meters
        self.maps_noise_position = get_random_roll_noise(num_passes=num_passes) * maps_noise_mult

        self.negotiation_traj = np.empty((N_NEGOTIATION_WPS, 3), dtype=np.float32)
        self.negotiation_traj_ixs = np.empty((N_NEGOTIATION_WPS), dtype=int)


    def set_route(self, route):
                                        
        self.waypoints = np.empty((len(route), 3), dtype="float64")
        self.waypoints[:,0] = moving_average(route.pos_x.to_numpy(), 30)
        self.waypoints[:,1] = moving_average(route.pos_y.to_numpy(), 30)
        self.waypoints[:,2] = moving_average(route.pos_z.to_numpy(), 20) 
        # smoothing wp z and pitch bc we don't drive them, we just use them directly. 
        # Smoothing XY of pos and rots was I believe causing a bit of spazz, don't know for certain why but removal did the trick
        # no didn't remove it, but perhaps lessened freq
        
        self.wp_normals = np.empty((len(route), 3), dtype="float64")
        self.wp_normals[:, 0] = route.normal_x.to_numpy()
        self.wp_normals[:, 1] = route.normal_y.to_numpy()
        self.wp_normals[:, 2] = route.normal_z.to_numpy()

        self.wp_rotations = np.empty((len(route), 3), dtype="float64") # pitch, roll, yaw
        self.wp_rotations[:, 0] = moving_average(route.curve_pitch.to_numpy(), 40) ## pitch
        self.wp_rotations[:, 1] = route.curve_roll.to_numpy() ## roll
        self.wp_rotations[:, 2] = route.curve_heading.to_numpy() # yaw

        START_IX = 30
        self.current_pos = self.waypoints[START_IX]
        self.current_rotation = self.wp_rotations[START_IX]
        self.distance_along_loop = START_IX * WP_SPACING

        self.start_intersection_section_id = route.intersection_section_id.iloc[0]
        self.visited_intersections = list(route.intersection_id.unique())

        ROUTE_WP_SPACING = 20 # meters
        wps_per_route_wp = int(ROUTE_WP_SPACING/WP_SPACING)
        is_route_wps = np.empty(len(route), dtype=bool)
        is_route_wps[:] = False
        is_route_wps[::wps_per_route_wp] = True
        self.is_route_wps = is_route_wps

    def set_nav_map(self, coarse_map_df):
        self.way_ids = coarse_map_df.way_id.to_numpy()
        self.lats, self.lons = coarse_map_df.pos_x.to_numpy(), coarse_map_df.pos_y.to_numpy()
        # lats, lons, way_ids = add_noise_rds_to_map(lats, lons, way_ids)
        self.refresh_nav_map_freq = random.choice([3,4,5]) # alternatively, can use vehicle speed and heading to interpolate ala kalman TODO
        self.small_map = None # doesn't update every frame

        # global gps_tracker
        # gps_tracker = GPSTracker()
    def set_should_yield(self, should_yield):
        self.should_yield = should_yield
        
    def step(self):

        sec_to_undagger = 2 # TODO shift should prob be variable, in which case this should also be variable as a fn of shift
        meters_to_undagger = self.current_speed_mps * sec_to_undagger

        current_wp_ix = int(round(self.distance_along_loop / WP_SPACING, 0))
        cam_normal = self.wp_normals[current_wp_ix]

        # getting targets
        wp_ixs = current_wp_ix + TRAJ_WP_IXS

        if wp_ixs[-1] >= len(self.waypoints):
            self.route_is_done = True
            print("ROUTE IS DONE", self.ap_id)
            return

        current_wps = self.waypoints[wp_ixs]
        deltas = current_wps - self.current_pos
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
                current_lat, current_lon = self.current_pos[0], self.current_pos[1]
                #heading = gps_tracker.step(current_lat, current_lon, current_speed_mps)

                BEHIND_BUFFER_M = 30
                FORWARD_BUFFER_M = 400
                _min = max(0, current_wp_ix-int(BEHIND_BUFFER_M/WP_SPACING))
                _max = current_wp_ix+int(FORWARD_BUFFER_M/WP_SPACING)
                route_wps = self.waypoints[_min:_max]
                is_route_wps_ixs = self.is_route_wps[_min:_max]
                route_wps = route_wps[is_route_wps_ixs]
                self.small_map = get_map(self.lats, self.lons, self.way_ids, route_wps,
                                    current_lat + self.maps_noise_position[self.overall_frame_counter], 
                                    current_lon + self.maps_noise_position[self.overall_frame_counter], #TODO refine this noising. Don't do same on x and y.
                                    self.current_rotation[2]+(np.pi)+self.maps_noise[self.overall_frame_counter], 
                                    close_buffer)

            # saving
            c = self.counter
            self.maps_container[c,:,:,:] = self.small_map

            self.targets_container[c, :N_WPS] = angles_to_wps
            self.targets_container[c, N_WPS:N_WPS*2] = wp_dists_actual
            self.targets_container[c, N_WPS*2:] = deltas[:,2]

            self.aux_container[c, 2] = mps_to_kph(self.current_speed_mps)
            self.aux_container[c, 4] = self.current_tire_angle
            self.aux_container[c, 0] = self.pitch_perturbation
            self.aux_container[c, 1] = self.yaw_perturbation

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

        if self.is_ego and (self.overall_frame_counter % self.DAGGER_FREQ == 0):
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

        fps = random.gauss(20, 2)

        dist_car_travelled = self.current_speed_mps * (1 / fps)

        # always using close wp for ap, to keep right on traj
        self.current_tire_angle = angle_to_target_wp_ap * self.lateral_kP # TODO should update this incrementally by kP?
        _wheelbase = 1.5 #CRV_WHEELBASE # NOTE this isn't correct, but i want ego to be turning faster, otherwise taking turns too wide
        _vehicle_turn_rate = self.current_tire_angle * (self.current_speed_mps/_wheelbase) # rad/sec # we're using target wp angle as the tire_angle
        vehicle_heading_delta = _vehicle_turn_rate * (1/fps) # radians

        # we get slightly out of sync bc our wps are going along based on curve, but we're controlling cube like a vehicle. 
        # Keeping totally synced so wp angles remain perfect.
        dist_car_travelled_corrected = dist_car_travelled + (MIN_WP_M - wp_dists_actual[0])*.25 # TODO don't like this
        self.distance_along_loop += dist_car_travelled_corrected

        _current_heading = self.current_rotation[2]+(np.pi/2) # TODO can rationalize this away
        angle_to_future_vehicle_loc = _current_heading - (vehicle_heading_delta/2) # this isn't the exact calc. See rollout_utils for more accurate version
        delta_x = dist_car_travelled*np.cos(angle_to_future_vehicle_loc)
        delta_y = (dist_car_travelled*np.sin(angle_to_future_vehicle_loc))

        self.current_rotation[0] = self.wp_rotations[current_wp_ix][0] # Setting pitch manually based on wps, ie not 'driving' it
        self.current_rotation[1] = self.roll_noise[current_wp_ix] #self.wp_rotations[current_wp_ix][1] # roll of rd is zero, so just adding our noise
        self.current_rotation[2] -= vehicle_heading_delta

        self.current_pos[0] += delta_x
        self.current_pos[1] += delta_y
        self.current_pos[2] = self.waypoints[current_wp_ix][2] # Just setting this manually to the wp itself. Not "driving" it like XY

        # Update speed
        curvature_constrained_speed = get_curve_constrained_speed_from_wp_angles(angles_to_wps, wp_dists_actual, self.current_speed_mps, max_accel=self.max_accel)
        curvature_constrained_speed *= self.curve_speed_mult
        target_speed = min(curvature_constrained_speed, self.speed_limit)
        target_speed = target_speed if not self.should_yield else 0
        self.current_speed_mps += (target_speed - self.current_speed_mps)*self.long_kP

        # negotiation traj
        m_per_wp = self.current_speed_mps * S_PER_WP
        frame_steps_per_wp = max(int(round(m_per_wp / WP_SPACING)), 1)
        self.negotiation_traj_ixs[:] = list(range(0, frame_steps_per_wp*N_NEGOTIATION_WPS, frame_steps_per_wp))
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
            self.speed_limit = random.uniform(16, 19) if rr<.05 else random.uniform(19, 25) if rr < .5 else random.uniform(25, 27)
        else:
            self.speed_limit = random.uniform(8, 12) if rr < .05 else random.uniform(12, 18) if rr < .1 else random.uniform(18, 26) # mps

        self.lateral_kP = .95 #.85 #random.uniform(.75, .95)
        self.long_kP = random.uniform(.02, .05)
        self.curve_speed_mult = random.uniform(.7, 1.25)
        self.turn_slowdown_sec_before = random.uniform(.25, .75)
        self.max_accel = random.uniform(1.0, 4.0)