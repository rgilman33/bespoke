from constants import *
import numpy as np
import math


def wp_ix_from_dist_along_traj(dist_m):
    # Returns a float where the decimal is the fraction the pt is btwn the two wps
    if dist_m <= LAST_NEAR_WP_DIST_M:
        wp_ix = dist_m - MIN_WP_M # at dist of 25.0, will return wp ix 19, which is our last one-meter spaced wp
    else:
        n_wps_into_five_m_wps = (dist_m - LAST_NEAR_WP_DIST_M ) / SPACING_BTWN_FAR_WPS # Now the decimal is perc out of 10
        wp_ix = LAST_NEAR_WP_IX + n_wps_into_five_m_wps

    wp_ix = min(max(0, wp_ix), len(TRAJ_WP_DISTS)-1) # ix can't be less than our closest wp or more than our farthest wp

    return wp_ix

#TODO make sure we're also always capping min max on distance as well, in all parts of codebase


def angle_to_wp_from_dist_along_traj(traj, dist_m):

    wp_ix = wp_ix_from_dist_along_traj(dist_m)

    wp_ix_0, wp_ix_1 = math.floor(wp_ix), math.ceil(wp_ix)
    perc_along_inbetween = wp_ix - wp_ix_0 # btwn zero and one, closer wps this is perc out of one, further it's perc out of five

    # weighted avg for wp_dist #TODO why don't we just use dist_m...
    wp_dist = TRAJ_WP_DISTS[wp_ix_1]*perc_along_inbetween + TRAJ_WP_DISTS[wp_ix_0]*(1-perc_along_inbetween)

    # weighted avg for angle to wp
    target_wp_angle = traj[wp_ix_1]*perc_along_inbetween + traj[wp_ix_0]*(1-perc_along_inbetween)

    return target_wp_angle, wp_dist, wp_ix

def get_target_wp_dist(speed_mps, wp_m_offset=0):
    # takes in speed, returns dist along traj to target wp
    wp_m = np.interp(speed_mps,
                    min_dist_bps,
                    [v+wp_m_offset for v in min_dist_vals])
    return wp_m

def get_target_wp(traj, speed_mps, wp_m_offset=0):
    # negative values of wp_m_offset brings target wp closer along the traj, making turns more centered. Pos makes further, making turns tighter
    wp_m = get_target_wp_dist(speed_mps, wp_m_offset=wp_m_offset)

    target_wp_angle, wp_dist, wp_ix = angle_to_wp_from_dist_along_traj(traj, wp_m)

    return target_wp_angle, wp_dist, wp_ix



def gather_preds(preds_all, speeds):
    """ takes in preds for entire traj (seqlen, n_preds),
    gathers them based on the given speed kph (seqlen, 1). """

    wp_angles = []
    for i in range(len(preds_all)):
        traj = preds_all[i]
        s = speeds[i]
        angle_to_wp, _, _ = get_target_wp(traj, s)
        wp_angles.append(angle_to_wp)
    return np.array(wp_angles)


# Dumb, but have to do this bc can't get the jitter out of blender curve
# i don't like this fn
def smooth_near_wps(traj):
    # Only use for headings and for wp_angles, as those are both zero at ego origin
    traj_orig = traj.copy()

    # Value is zero at origin. Including beginning and end tips to aid the smoothing
    meterly_values = np.interp(list(range(0,40)), [0]+TRAJ_WP_DISTS, np.insert(traj, 0, 0)) # getting a value for every meter from 0 to 40

    # tt returns same size as the original, so we only want first part
    traj_orig[0:N_DENSE_WPS] = moving_average(meterly_values, 5)[MIN_WP_M:MIN_WP_M+N_DENSE_WPS]

    return traj_orig


def get_heading_and_pos_delta(tire_angle, speed, wheelbase=CRV_WHEELBASE, n_sec=1/FPS):
    # TODO use this same fn in model wrapper

    curvature = tire_angle/wheelbase # rad/m
    vehicle_turn_rate_sec = curvature * speed # rad/sec
    future_vehicle_heading = vehicle_turn_rate_sec * n_sec # radians, w respect to original ego heading

    dist_travelled = speed * n_sec # meters
    r = dist_travelled / (future_vehicle_heading)
    future_vehicle_y = np.sin(future_vehicle_heading)*r
    future_vehicle_x = r - (np.cos(future_vehicle_heading)*r)

    return future_vehicle_heading, future_vehicle_x, future_vehicle_y

# def get_te_loss(wp_angles, wp_dists, aux):
#     # batch, seq, n
#     xs, ys = xy_from_angle_dist(wp_angles, wp_dists)
#     t0_xs, t0_ys = xs[:,:-1], ys[:,:-1]
#     t1_xs, t1_ys = xs[:,1:], ys[:,1:]

#     # shift and rotate t1 according to vehicle motion. Using aux values (tire angle, speed) from t0
#     # this puts the t1 values in the coord space of t0
#     t1h, t1x, t1y = get_heading_and_pos_delta(aux[:,:-1,'tire_angle'], aux[:,:-1,'speed']) #TODO change to tire_angle_ap
#     t1_xs += t1x
#     t1_ys += t1y
#     t1_xs, t1_ys = rotate_around_origin(t1_xs, t1_ys, t1h)

#     # Now all in absolute coords relative to t0

#     #TODO instead, rotate t0 points to get traj_consistency targets for t1. Can get those fully in np, detached,
#     # and then use them as targets

#     # This is where t1 points should be, if they remain on t0 traj
#     dist_travelled = aux[:,:-1,'speed'] * (1/FPS) # m
#     t1_xs_targets = np.interp(TRAJ_WP_DISTS+dist_travelled, TRAJ_WP_DISTS, t0_xs)
#     t1_ys_targets = np.interp(TRAJ_WP_DISTS+dist_travelled, TRAJ_WP_DISTS, t0_ys)

#     loss = np.sqrt((t1_xs_targets - t1_xs)**2 + (t1_ys_targets - t1_ys)**2) # loss for each wp
#     loss = loss[:,:,:20] # just for the first 20, can refine this later
#     loss = loss.mean()

#     return loss

def get_consistency_targets(wp_angles, wp_dists, aux):
    """
    Given traj at t0, and vehicle movement, what will targets be at t1 to remain completely on the traj?
    """
    # batch, seq, n
    xs, ys = xy_from_angle_dist(wp_angles, wp_dists)

    # puts the t0 xy values in the coord space of t1
    hd, xd, yd = get_heading_and_pos_delta(aux[:,:,'tire_angle_ap'], aux[:,:,'speed'])
    xs -= xd; ys -= yd
    xs, ys = rotate_around_origin(xs, ys, hd)

    # This is where t1 points should be, if they remain on t0 traj
    dist_travelled = aux[:,:,'speed'] * (1/FPS) # m
    xs_targets = np.interp(TRAJ_WP_DISTS-dist_travelled, TRAJ_WP_DISTS, xs)
    ys_targets = np.interp(TRAJ_WP_DISTS-dist_travelled, TRAJ_WP_DISTS, ys)

    wp_angles_t = np.arctan2(xs_targets, ys_targets)
    wp_angles_t = wp_angles_t[:,:-1,:] # not taking last one as there's no interp target

    return wp_angles_t


def rotate_around_origin(x, y, angle):
    xx = x * np.cos(angle) + y * np.sin(angle)
    yy = -x * np.sin(angle) + y * np.cos(angle)

    return xx, yy

def xy_from_angle_dist(wp_angles, wp_dists):
    #NOTE this should always be done in full float precision
    wp_xs = np.sin(wp_angles) * wp_dists
    wp_ys = np.cos(wp_angles) * wp_dists

    return wp_xs, wp_ys

def get_headings_from_traj(wp_angles, wp_dists):
    #NOTE this should always be done in full float precision
    wp_xs, wp_ys = xy_from_angle_dist(wp_angles, wp_dists)

    # these headings are at half-wp marks
    headings = np.arctan2(wp_xs[1:] - wp_xs[:-1], wp_ys[1:] - wp_ys[:-1])
    # cat a zero on the front bc we know the heading is zero at vehicle, by definition
    headings = np.concatenate([np.array([0], dtype=np.float32), headings])

    # interp back to our original wp dists so they're aligned w our other data
    headings = np.interp(TRAJ_WP_DISTS, HEADING_BPS, headings)

    return headings


# used for ccs apparatus
MAX_ACCEL = .6 #1.0 #2.0 #m/s/s 3 to 5 is considered avg for an avg driver in terms of stopping, the latter as a sort of max decel

# used to generate blender data. Should match the batched apparatus used in blender dataloader for making trn data curvatures
# would be nice to actually use the exact same fns, but would need to update the fns below to accept both batched and non-batched data
# the only reason not using same fn is bc calling the batched versions in the dataloader, calling the single versions here
def get_curve_constrained_speed_from_wp_angles(wp_angles, wp_dists, current_speed_mps, max_accel=MAX_ACCEL):
    wp_angles = smooth_near_wps(wp_angles)
    headings = get_headings_from_traj(wp_angles, wp_dists)
    headings = smooth_near_wps(headings)
    curvatures = get_curvatures_from_headings(headings)
    curve_constrained_speed = get_curve_constrained_speed(curvatures, current_speed_mps, max_accel=max_accel)

    return curve_constrained_speed


def smooth_near_wps_batch(batch):
    bs, seq_len, _ = batch.shape
    smoothed = np.empty_like(batch)
    for b in range(bs):
        for s in range(seq_len):
            smoothed[b,s,:] = smooth_near_wps(batch[b,s,:])
    return smoothed

def get_headings_from_traj_batch(wp_angles, wp_dists):
    bs, seq_len, _ = wp_angles.shape
    # TODO this should be vectorized. Just using this for convenience right now
    wp_headings = np.empty_like(wp_angles)
    for b in range(bs):
        for s in range(seq_len):
            wp_headings[b,s,:] = get_headings_from_traj(wp_angles[b,s,:], wp_dists[b,s,:])
    return wp_headings


def get_curvatures_from_headings(headings):

    # these will be at the half-wp marks
    curvatures = (headings[1:] - headings[:-1]) / SEGMENT_DISTS # rad / m

    # we're extrapolating for both the first and last wp, that's fine
    curvatures = np.interp(TRAJ_WP_DISTS, WP_HALFWAYS, curvatures)

    return curvatures

def get_curvatures_from_headings_batch(headings):
    bs, seq_len, _ = headings.shape
    curvatures = np.empty_like(headings)
    for b in range(bs):
        for s in range(seq_len):
            curvatures[b,s,:] = get_curvatures_from_headings(headings[b,s,:])
    return curvatures


def tire_angles_to_max_speeds(tire_angles):
    magic = 4.5 #4.1 #3.8 #5.0 # even the official formula has a magic number. We're just taking this from our own runs. 5.0 was estimated from human run as "correct" value.
    max_speeds = np.sqrt(1/(abs(tire_angles)+.0001)) * magic # units is mph bc that's how we eyeballed it
    max_speeds = mph_to_mps(max_speeds)
    return max_speeds

def max_ix_from_speed(speed_mps):
    long_consideration_max_m = max_pred_m_from_speeds(speed_mps)
    long_consideration_max_ix = math.ceil(wp_ix_from_dist_along_traj(long_consideration_max_m))
    return long_consideration_max_ix

MAX_SPEED_CCS = 30.0

def _get_curve_constrained_speed(curvatures, current_speed_mps, max_accel=MAX_ACCEL, preempt_sec=1.0):
    # given a traj, what is the max speed we can be going right now to ensure we're able to hit all the upcoming wps at a speed appropriate for each one?
    # and given a maximum allowed deceleration. This speed may be higher than the speed limit, it is simply the speed based on upcoming curvature, so
    # in theory a perfectly straight rd could have limit of infinity
    # current_speed_mps is only used to truncate the results bc we don't need to support beyond 5s
    # returns mps
    if current_speed_mps < 5: return 30

    #curvatures = get_curvatures_from_headings(headings)
    tire_angles = curvatures * CRV_WHEELBASE
    max_speeds_at_each_wp = tire_angles_to_max_speeds(tire_angles)

    curve_preempt_m = preempt_sec * current_speed_mps # we want to be going the desired speed BEFORE we hit the wp, not AT the wp, is that true?
    max_speeds_at_each_wp_preempted = np.interp(np.array(TRAJ_WP_DISTS) + curve_preempt_m, TRAJ_WP_DISTS, max_speeds_at_each_wp)
    # For each wp, there is a max speed we can be going now to make sure we're going the proper speed when we hit that wp
    current_max_speed_w_respect_to_each_wp = np.sqrt(np.array(TRAJ_WP_DISTS) * max_accel) + max_speeds_at_each_wp_preempted #max_speeds_at_each_wp # adds elementwise

    long_consideration_max_ix = max_ix_from_speed(current_speed_mps)
    curve_constrained_speed = current_max_speed_w_respect_to_each_wp[:long_consideration_max_ix].min()

    curve_constrained_speed = min(curve_constrained_speed, MAX_SPEED_CCS) # just for visual cleanliness

    return curve_constrained_speed

def get_curve_constrained_speed(curvatures, current_speed_mps, max_accel=MAX_ACCEL):
    # returns mps
    ccs = min([_get_curve_constrained_speed(curvatures, current_speed_mps, preempt_sec=s, max_accel=MAX_ACCEL) for s in [0.1, .5, 1.0, 1.5, 2.0]])

    # this magic is in addition to the magic number in curve-to-speeds formula above
    ccs *= np.interp(mps_to_mph(current_speed_mps), [20, 40], [1.0, 1.12]) # rw driving we seem to accept more torque at higher speeds? maybe not, maybe rds are generally banked at higher speeds?

    return ccs


def get_angles_to(xs, ys, heading):
    # wps w respect to current pos, ie centered at zero, ie pos already subtracted out. Radians in and out.
    # current heading can be range 0 to 2*pi or -pi to pi
    # replacing below function "get_angle_to" so 1) i can understand it and 2) is vectorized

    angles = np.arctan2(xs, ys) # from -pi to pi, zero is up

    angles[angles<0] += 2*np.pi # rotate to range 0 to 2*pi

    angles -= heading # subtract out current heading

    # rotate into range -pi to pi
    angles[angles<-np.pi] += 2*np.pi
    angles[angles>np.pi] -= 2*np.pi

    return angles


MAX_CCS_UNROLL_ACCEL = 1.0
class CurveConstrainedSpeedCalculator():
    def __init__(self):
        self.reset()

    def step(self, wp_curvatures, current_speed):
        # Curve constrained speed
        curve_constrained_speed_mps = get_curve_constrained_speed(wp_curvatures, current_speed)
        self.curve_speeds_history.append(curve_constrained_speed_mps)

        CURVE_SPEED_UNROLL_DELAY_S = 5.0
        FPS = 20
        CURVE_SPEED_UNROLL_DELAY_IX = int(CURVE_SPEED_UNROLL_DELAY_S*FPS)
        curve_constrained_speed_mps = min(self.curve_speeds_history[-CURVE_SPEED_UNROLL_DELAY_IX:])

        # don't accel after a turn too quickly
        max_speed_increase_per_step = MAX_CCS_UNROLL_ACCEL / FPS
        curve_constrained_speed_mps = min(curve_constrained_speed_mps, self.prev_commanded_ccs+max_speed_increase_per_step)
        self.prev_commanded_ccs = curve_constrained_speed_mps

        return curve_constrained_speed_mps

    def reset(self):
        self.curve_speeds_history = [MAX_SPEED_CCS for _ in range(20)]
        self.prev_commanded_ccs = MAX_SPEED_CCS


# "A Policy on Geometric Design of Highways and Streets recommends 3.4 m/s2
# a comfortable deceleration rate for most drivers, as the deceleration threshold for determining adequate stopping sight distance"
SPEED_MASK_DECEL = 2.0 #2.5 # m/s/s # NOTE pay attn to this, are we still training far enough on the traj? Tradeoff here in terms of apportioning trn load
MAX_PRED_S = 8.0 #6.0
MIN_M_TRAJ_PRED = 40.

def max_pred_m_from_speeds(speeds_mps):
    # can be single obs, seq, or batch,seq. Returns same dims, just swaps speeds for meters.

    # Pred out as far as it will take us to comfortably to get to full stop
    max_pred_s = speeds_mps / SPEED_MASK_DECEL

    # But don't pred any further than MAX_PRED_S, TODO increase this to 10s. Now that we're clipping based on speed above, we won't hit this as often
    max_pred_s = np.clip(max_pred_s, 0., MAX_PRED_S)

    max_pred_dists_m = speeds_mps * max_pred_s

    max_pred_dists_m = np.clip(max_pred_dists_m, MIN_M_TRAJ_PRED, np.inf) # always pred out to at least n meters, even when stopped

    return max_pred_dists_m

def get_speed_mask(speed):
    # can be single obs, seq, or seq,batch. Returns same but w extra dim of sz 30 on the end.

    if type(speed)!=np.ndarray: speed = np.array([speed])

    # same shape as speeds, just swapped for distance
    max_pred_dists_m = max_pred_m_from_speeds(speed)

    # Pad to broadcast. Result is eg (batch,seq,30) if input have batch seq, or ()
    speed_mask = (np.array(TRAJ_WP_DISTS) <= max_pred_dists_m[...,None]).astype(np.float32)

    return speed_mask[0] # otherwise padded first dim


#TODO change to mps. Use same fns as use for speed limiting, but inverse for tire angles
TORQUE_ABS_MAX = 10_000 #TODO move all these constants into single place
TORQUE_DELTA_MAX = 800

class TorqueLimiter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_published_tire_angle_deg = 0


    def step(self, desired_tire_angle_deg, current_speed_kph):
        # Torque limits
        BASELINE = self.prev_published_tire_angle_deg # current_tire_angle_deg

        max_angle_bc_abs_torque = TORQUE_ABS_MAX / (current_speed_kph**2 + 1)
        max_angle_delta_bc_td = TORQUE_DELTA_MAX / (current_speed_kph**2 + 1)

        commanded_td = (desired_tire_angle_deg - BASELINE) * current_speed_kph**2
        commanded_torque = desired_tire_angle_deg * current_speed_kph**2

        torque_limited_angle_deg = np.clip(desired_tire_angle_deg, -max_angle_bc_abs_torque, max_angle_bc_abs_torque)
        is_abs_torque_limited = torque_limited_angle_deg != desired_tire_angle_deg

        td_limited_angle_deg = np.clip(desired_tire_angle_deg,
                                        BASELINE - max_angle_delta_bc_td,
                                        BASELINE + max_angle_delta_bc_td)
        is_td_limited = td_limited_angle_deg != desired_tire_angle_deg

        if (is_td_limited or is_abs_torque_limited):
            tire_angle_deg = td_limited_angle_deg if abs(td_limited_angle_deg)<abs(torque_limited_angle_deg) else torque_limited_angle_deg
        else:
            tire_angle_deg = desired_tire_angle_deg

        self.prev_published_tire_angle_deg = tire_angle_deg

        return tire_angle_deg, is_abs_torque_limited, is_td_limited, commanded_torque, commanded_td
