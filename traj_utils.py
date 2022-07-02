from constants import *
import numpy as np
import math



def wp_ix_from_dist_along_traj(dist_m):
    # Returns a float where the decimal is the fraction the pt is btwn the two wps
    if dist_m <= 25:
        wp_ix = dist_m - MIN_WP_M # at dist of 25.0, will return wp ix 19, which is our last one-meter spaced wp
    else:
        n_wps_into_five_m_wps = (dist_m - 25 ) / 5 # Now the decimal is perc out of 5
        wp_ix = 19 + n_wps_into_five_m_wps

    wp_ix = min(max(0, wp_ix), len(traj_wp_dists)-1) # ix can't be less than our closest wp or more than our farthest wp

    return wp_ix
    
#TODO make sure we're also always capping min max on distance as well, in all parts of codebase


def angle_to_wp_from_dist_along_traj(traj, dist_m):

    wp_ix = wp_ix_from_dist_along_traj(dist_m)

    wp_ix_0, wp_ix_1 = math.floor(wp_ix), math.ceil(wp_ix)
    perc_along_inbetween = wp_ix - wp_ix_0 # btwn zero and one, closer wps this is perc out of one, further it's perc out of five

    # weighted avg for wp_dist
    wp_dist = traj_wp_dists[wp_ix_1]*perc_along_inbetween + traj_wp_dists[wp_ix_0]*(1-perc_along_inbetween)

    # weighted avg for angle to wp
    target_wp_angle = traj[wp_ix_1]*perc_along_inbetween + traj[wp_ix_0]*(1-perc_along_inbetween)

    return target_wp_angle, wp_dist, wp_ix


def get_target_wp(traj, speed_mps, wp_m_offset=None):
    # negative value brings target wp closer along the traj, making turns more centered. Pos makes further, making turns tighter
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
    # wp_ix as int 
    angle_to_wp = traj[wp_ix]
    dist = traj_wp_dists[wp_ix] # TODO this actually isn't correct, will get worse the more the traj bends. 
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


# These are the dists at all the midway pts btwn our wps, plus a zero in the beginning
# they're staggered by .5 m for closer wps, then 2.5m for farther ones
HEADING_BPS = [0] + [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 27.5] 
HEADING_BPS += [32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5] # last heading is the one btwn our second to last and our last wp

# TODO we should just use np.interp like below for our manual interps above
def get_heading_at_dist_along_traj(traj, dist):
    heading_values = get_headings_from_traj(traj)
    heading = np.interp(dist, HEADING_BPS, heading_values)
    
    return heading


mph_to_mps = lambda x : x*.44704
# estimated from mostly-agent run on vv, sabq, pow, 214. This one appears more permissive than the one below? NOTE these are in mph whereas below and elsewhere they're kph
max_speed_lookup_rollout = [
    (.157, 16.6),
    (.11, 20),
    (.071, 25.5),
    (.052, 28.7),
    (.037, 34.1),
    (.027, 39.3),
    (.021, 44.7),
    (.018, 50), # this is a guess, TODO update this
    (.01, 60), #
]
CURVE_SPEED_MULT = .8
max_speed_bps_rollout = [x[0] for x in max_speed_lookup_rollout][::-1]
max_speed_vals_rollout = [mph_to_mps(x[1])*CURVE_SPEED_MULT for x in max_speed_lookup_rollout][::-1]


def get_curve_constrained_speed(traj, speed_mps, curve_prep_slowdown_time_sec=3):
    # N seconds from now, what is the speed we'll want to be going based on the turning we'll be doing?
    # more accurate the smaller the slowdown time, eg projecting four sec into future is harder than one sec
    
    car_future_heading = get_heading_at_dist_along_traj(traj, speed_mps*curve_prep_slowdown_time_sec)

    # magic number to convert the heading at that place on the traj into the approximate steer that will be commanded at that point
    # goal is simply to shift the steer curve to the left, ie 'what steer are we going to command in n seconds'? Eyeballed from trn data
    magic_smallerizer = np.interp(curve_prep_slowdown_time_sec, [1,0, 2.0, 3.0, 4.0], [2.6, 5.5, 8.0, 11])
    future_steer = car_future_heading / magic_smallerizer

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