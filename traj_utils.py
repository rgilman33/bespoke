from constants import *
import numpy as np
import math


def max_speed_from_steer(steer_angle_perc_max):
    steer_angle_perc_max = abs(steer_angle_perc_max)
    max_speed = int(round(np.interp(steer_angle_perc_max, 
                            max_speed_bps, 
                            max_speed_vals)))
    
    return max_speed


def angle_to_wp_from_dist_along_traj(traj, m):
    wp_ix = m - MIN_WP_M

    # float, where frac is the perc in between the two target wps
    wp_ix = min(max(0, wp_ix), len(traj_wp_dists)-1) #+ 2 # magic number

    wp_ix_0, wp_ix_1 = math.floor(wp_ix), math.ceil(wp_ix)
    perc_along_inbetween = wp_ix - wp_ix_0 # btwn zero and one, wps are one meter apart

    # weighted avg for wp_dist
    wp_dist = traj_wp_dists[wp_ix_1]*perc_along_inbetween + traj_wp_dists[wp_ix_0]*(1-perc_along_inbetween)

    # weighted avg for angle to wp
    target_wp_angle = traj[wp_ix_1]*perc_along_inbetween + traj[wp_ix_0]*(1-perc_along_inbetween)

    return target_wp_angle, wp_dist, wp_ix


def get_target_wp(traj, speed_mps, wp_m_offset=None):
    # negative value brings target wp closer, making turns more centered. Pos makes further, making turns tighter
    if wp_m_offset:
        wp_m = np.interp(mps_to_kph(speed_mps), 
                                min_dist_bps, 
                                [v+wp_m_offset for v in min_dist_vals]) 
    else:
        wp_m = np.interp(mps_to_kph(speed_mps), 
                                min_dist_bps, 
                                min_dist_vals) 
    

    target_wp_angle, wp_dist, wp_ix = angle_to_wp_from_dist_along_traj(traj, wp_m)

    return target_wp_angle, wp_dist, wp_ix



def gather_preds(preds_all, speeds):
    """ takes in preds for entire traj (seqlen, n_preds), 
    gathers them based on the given speed kph (seqlen, 1). 
    Adjusts for kP, returns the actual angles commanded """ 
    # used only in rw offline eval
    
    wp_angles = []
    for i in range(len(preds_all)):
        traj = preds_all[i]
        s = speeds[i]
        angle_to_wp, _, _ = get_target_wp(traj, kph_to_mps(s))
        wp_angles.append(angle_to_wp)
    return np.array(wp_angles) 



# used for blender autopilot

def get_vehicle_updated_position(current_speed, traj):
    # current_speed in mps, traj as 1d np array of angles to wps in rad
    dist_car_travelled_during_lag = current_speed * (1 / FPS)

    # local space, used for directing the car
    # everything in the frame of reference of the car at t0, the origin
    target_wp_angle, _, _ = get_target_wp(traj, current_speed, wp_m_offset=0) # comes out as float, frac is the amount btwn the two wps

    # assuming the tire angle will be the angle to the target wp
    vehicle_turn_rate = target_wp_angle * (current_speed/CRV_WHEELBASE) # rad/sec
    vehicle_heading_delta = vehicle_turn_rate * (1/FPS) # radians
    # the vehicle won't be as turned as the tires, proportional to wheelbase

    if vehicle_heading_delta==0:
        vehicle_x_delta = 0
        vehicle_y_delta = dist_car_travelled_during_lag
    else:
        r = dist_car_travelled_during_lag / vehicle_heading_delta
        vehicle_y_delta = np.sin(vehicle_heading_delta)*r
        vehicle_x_delta = r - (np.cos(vehicle_heading_delta)*r)
    
    return vehicle_x_delta, vehicle_y_delta, vehicle_heading_delta


def get_wp_xy_from_ix(traj, wp_ix):
    # wp_ix as int TODO this actually isn't correct, will get worse further along traj you go
    angle_to_wp = traj[wp_ix]
    dist = wp_ix + MIN_WP_M
    wp_x = np.sin(angle_to_wp) * dist
    wp_y = np.cos(angle_to_wp) * dist
    return wp_x, wp_y


def get_headings_from_traj(traj):
    headings = [0]
    for i in range(len(traj)-1):
        this_wp_x, this_wp_y = get_wp_xy_from_ix(traj, i)
        next_wp_x, next_wp_y = get_wp_xy_from_ix(traj, i+1)
        heading = np.arcsin((next_wp_x - this_wp_x)/(next_wp_y - this_wp_y))
        headings.append(heading)
    return headings


HEADING_BPS = [0] + [wp_dist+.5 for wp_dist in traj_wp_dists][:-1]

def get_heading_at_dist_along_traj(traj, dist):
    heading_values = get_headings_from_traj(traj)
    heading = np.interp(dist, HEADING_BPS, heading_values)
    
    return heading





# Used for autolong in OP
# no reason this lookup shouldn't be the same as the other. No need conceptually for separate lookups. TODO delete this one?
max_speed_lookup_rollout = [ # estimated from run260, nabq. 
    (.005, 100), # 62 mph
    (.01, 80), # 50 mph         don't know about this one, research more, this could be dangerous
    (.0175, 60), # 37 mph
    (.035, 50), # 31 mph
    (.065, 40), # 25 mph
    (.12, 30), # 18 mph
    (.23, 20), # 12 mph
    (.3, 15),
    (.42, 10), # 6 mph
]

max_speed_bps_rollout = [x[0] for x in max_speed_lookup_rollout]
max_speed_vals_rollout = [kph_to_mps(x[1]) for x in max_speed_lookup_rollout]

CURVE_PREP_SLOWDOWN_TIME = 2.0 # seconds

#TODO this still isn't principled. Should be getting the angle to target wp from the future location
def get_curve_constrained_speed(traj, speed_mps):
    # N seconds from now, what is the speed we'll want to be going based on the turning we'll be doing?
    
    car_future_heading = get_heading_at_dist_along_traj(traj, speed_mps*CURVE_PREP_SLOWDOWN_TIME)

    # magic number to convert the heading at that place on the traj into the approximate steer that will be commanded at that point
    # goal is simply to shift the steer curve to the left, ie 'what steer are we going to command in n seconds'?
    # this magic number was eyeballed using the trn data, trying to shift the actual steer forward by n sec
    future_steer = car_future_heading / 5.5 # 1.5 slowdown time good w 4; 2.0 good w 5.5

    curve_max_speed_kph = np.interp(abs(future_steer), 
                        max_speed_bps_rollout, 
                        max_speed_vals_rollout)

    return curve_max_speed_kph





def get_angle_to(pos, theta, target):
    pos = np.array(pos); target = np.array(target)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    aim = R.T.dot(target - pos)
    angle = np.arctan2(-aim[1], aim[0])
    angle = 0.0 if np.isnan(angle) else angle 

    return angle

def dist(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return (x**2 + y**2)**(1/2)