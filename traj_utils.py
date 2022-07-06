from constants import *
import numpy as np
import math

SPACING_BTWN_FAR_WPS = 10
LAST_NEAR_WP_DIST_M = 25
LAST_NEAR_WP_IX = 19

def wp_ix_from_dist_along_traj(dist_m):
    # Returns a float where the decimal is the fraction the pt is btwn the two wps
    if dist_m <= LAST_NEAR_WP_DIST_M:
        wp_ix = dist_m - MIN_WP_M # at dist of 25.0, will return wp ix 19, which is our last one-meter spaced wp
    else:
        n_wps_into_five_m_wps = (dist_m - LAST_NEAR_WP_DIST_M ) / SPACING_BTWN_FAR_WPS # Now the decimal is perc out of 10
        wp_ix = LAST_NEAR_WP_IX + n_wps_into_five_m_wps

    wp_ix = min(max(0, wp_ix), len(traj_wp_dists)-1) # ix can't be less than our closest wp or more than our farthest wp

    return wp_ix
    
#TODO make sure we're also always capping min max on distance as well, in all parts of codebase


def angle_to_wp_from_dist_along_traj(traj, dist_m):

    wp_ix = wp_ix_from_dist_along_traj(dist_m)

    wp_ix_0, wp_ix_1 = math.floor(wp_ix), math.ceil(wp_ix)
    perc_along_inbetween = wp_ix - wp_ix_0 # btwn zero and one, closer wps this is perc out of one, further it's perc out of five

    # weighted avg for wp_dist #TODO why don't we just use dist_m...
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
    # the vehicle won't be as turned as the tires, proportional to wheelbase DUMB intuition check out updated in model_wrapper

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
# they're staggered by .5 m for closer wps, then 5m for farther ones
WP_HALFWAYS = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 30.0] 
WP_HALFWAYS += [40, 50, 60, 70, 80, 90, 100, 110, 120] # last heading is the one btwn our second to last and our last wp
# halfways are the halfway pt on each segment, there is one less than the number of wps

HEADING_BPS = [0] + WP_HALFWAYS

SEGMENT_DISTS = np.array(TRAJ_WP_DISTS[1:]) - np.array(TRAJ_WP_DISTS[:-1]) # 19 ones then 10 tens. There is one fewer segment than there are wps

# TODO we should just use np.interp like below for our manual interps above
def get_heading_at_dist_along_traj(traj, dist):
    heading_values = get_headings_from_traj(traj)
    heading = np.interp(dist, HEADING_BPS, heading_values)
    
    return heading


mph_to_mps = lambda x : x*.44704


# max_speed_lookup_rollout = [
#     (.157, 11),
#     (.11, 16),
#     (.071, 20.5),
#     (.052, 26.7),
#     (.037, 34.1),
#     (.027, 39.3),
#     (.021, 44.7),
#     (.018, 50), # this is a guess, TODO update this
#     (.01, 66), #
# ]
# CURVE_SPEED_MULT = 1.
# max_speed_bps_rollout = [x[0] for x in max_speed_lookup_rollout][::-1]
# max_speed_vals_rollout = [mph_to_mps(x[1])*CURVE_SPEED_MULT for x in max_speed_lookup_rollout][::-1]

# # estimates future steer using angle to future wp TODO this is going away in favor of more principled version
# def get_curve_constrained_speed(traj, speed_mps, curve_prep_slowdown_time_sec=4):
#     # N seconds from now, what is the speed we'll want to be going based on the turning we'll be doing?
#     # more accurate the smaller the slowdown time, eg projecting four sec into future is harder than one sec
    
#     angle_to_far_wp, _, _ = angle_to_wp_from_dist_along_traj(traj, speed_mps*curve_prep_slowdown_time_sec)

#     # magic number to convert the heading at that place on the traj into the approximate steer that will be commanded at that point
#     # goal is simply to shift the steer curve to the left, ie 'what steer are we going to command in n seconds'? Eyeballed from trn data
#     magic_smallerizer = np.interp(curve_prep_slowdown_time_sec, [3.0, 4.0, 5.0], [3.7, 4.5, 5.3])
#     future_steer = angle_to_far_wp / magic_smallerizer

#     curve_max_speed_kph = np.interp(abs(future_steer), 
#                         max_speed_bps_rollout, 
#                         max_speed_vals_rollout)

#     return curve_max_speed_kph




def get_headings_from_traj(traj):
    wp_xs = np.sin(traj) * TRAJ_WP_DISTS #TODO this is wrong, especially now that our traj is so long!
    wp_ys = np.cos(traj) * TRAJ_WP_DISTS

    # these headings are at half-wp marks
    xd_over_yd = (wp_xs[1:] - wp_xs[:-1])/(wp_ys[1:] - wp_ys[:-1])
    headings = np.arcsin(np.clip(xd_over_yd, -1.0, 1.0)) # can't have arcsine outside range -1 to 1. This was only happening at end of traj, sometimes. So prob we're not even looking at this part of traj, but be aware of this
    # cat a zero on the front bc we know the heading is zero at vehicle (by definition)
    headings = np.concatenate([np.array([0], dtype=np.float16), headings])
    
    # interp back to our original wp dists so they're nice and aligned w our other data
    headings = np.interp(TRAJ_WP_DISTS, HEADING_BPS, headings)
    
    return headings

def get_curvatures_from_headings(headings):
    
    # these will be at the half-wp marks
    curvatures = abs(headings[1:] - headings[:-1]) / SEGMENT_DISTS
    
    # we're extrapolating for both the first and last wp, that's fine
    curvatures = np.interp(TRAJ_WP_DISTS, WP_HALFWAYS, curvatures)
    
    return curvatures


def tire_angles_to_max_speeds(tire_angles):
    magic = 5.0 # even the official formula has a magic number. We're just taking this from our own runs. 
    max_speeds = np.sqrt(1/(tire_angles+.0001)) * magic # units is mph bc that's how we eyeballed it
    max_speeds = mph_to_mps(max_speeds)
    return max_speeds


ACCEL = 2.0 #m/s/s 3 to 5 is considered avg for an avg driver in terms of stopping, the latter as a sort of max decel
def get_curve_constrained_speed(traj, current_speed_mps):
    # given a traj, what is the max speed we can be going right now to ensure we're able to hit all the upcoming wps at a speed appropriate for each one?
    # and given a maximum allowed deceleration. This speed may be higher than the speed limit, it is simply the speed based on upcoming curvature, so
    # in theory a perfectly straight rd could have limit of infinity
    # current_speed_mps is only used to truncate the results bc we don't need to support beyond 5s

    headings = get_headings_from_traj(traj)
    curvatures = get_curvatures_from_headings(headings)
    tire_angles = curvatures * CRV_WHEELBASE
    max_speeds_at_each_wp = tire_angles_to_max_speeds(tire_angles)
    # For each wp, there is a max speed we can be going now to make sure we're going the proper speed when we hit that wp
    current_max_speed_w_respect_to_each_wp = np.sqrt(np.array(TRAJ_WP_DISTS) * ACCEL) + max_speeds_at_each_wp # adds elementwise

    long_consideration_max_m = CURVE_PREP_SLOWDOWN_S_MAX * current_speed_mps # we currently support up to 5s out
    long_consideration_max_ix = math.ceil(wp_ix_from_dist_along_traj(long_consideration_max_m))
    curve_constrained_speed = current_max_speed_w_respect_to_each_wp[:long_consideration_max_ix].min()

    return curve_constrained_speed
    



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