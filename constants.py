import numpy as np

BESPOKE_ROOT = '/home/beans/bespoke'

# webcam_img_height = 480
# webcam_img_width = 640
# IMG_WIDTH = webcam_img_width

# BOTTOM_CHOP = 150
# TOP_CHOP = 150 + 60

# IMG_HEIGHT = 120 #
# assert IMG_HEIGHT == (webcam_img_height - TOP_CHOP - BOTTOM_CHOP)

# OP_UI_BACKGROUND_WIDTH = 1164
# OP_UI_BACKGROUND_HEIGHT = 874
# OP_UI_MARGIN = 300

webcam_img_height = 1080
webcam_img_width = 1920

SIDE_CHOP = 240 # cropping out an eigth on each side which we eyeballed to equal our prev frame, and gets rid of distortion and lens at edges
IMG_WIDTH = webcam_img_width - SIDE_CHOP - SIDE_CHOP # 1440
assert IMG_WIDTH == 1440

BOTTOM_CHOP = 310 #330 #360
TOP_CHOP = 410 #390 #360

IMG_HEIGHT = 360
assert IMG_HEIGHT == (webcam_img_height - TOP_CHOP - BOTTOM_CHOP)

OP_UI_BACKGROUND_WIDTH = 1164
OP_UI_BACKGROUND_HEIGHT = 874
OP_UI_MARGIN = 300




N_CHANNELS = 6 #3
BPTT = 1 #4 #8 #9

DATA_CONSUMPTION_RATIO_LIMIT = 3 #1.

MIN_WP_M = 6 #8
TRAJ_WP_DISTS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] + [35, 45, 55, 65, 75, 85, 95, 105, 115, 125]
assert MIN_WP_M==TRAJ_WP_DISTS[0]
N_WPS = len(TRAJ_WP_DISTS)

N_TARGETS = N_WPS*5 # currently is wp_angle, curvature, heading, roll, z-delta
N_WPS_TO_USE = N_WPS
traj_wp_dists = TRAJ_WP_DISTS

AUX_PITCH_IX = 0
AUX_YAW_IX = 1
AUX_SPEED_IX = 2
AUX_TIRE_ANGLE_IX = 4
AUX_APPROACHING_STOP_IX = 5
AUX_STOPPED_IX = 6
AUX_STOP_DIST_IX = 7
AUX_HAS_LEAD_IX = 8
AUX_LEAD_DIST_IX = 9
AUX_LEAD_SPEED_IX = 10
AUX_SHOULD_YIELD_IX = 11

N_AUX_TO_SAVE = 20
N_EPISODE_INFO = 10 

N_AUX_MODEL_IN = 5
N_AUX_CALIB_IN = 4

N_AUX_TARGETS = 12

STEER_RATIO = 16. # taken from OP, specific for crv-5g. Don't change this willy nilly

mps_to_kph = lambda x: (x/1000)*60*60
kph_to_mps = lambda x: (x*1000)/(60*60)
mps_to_mph = lambda x: x*2.23694
mph_to_mps = lambda x : x*.44704
kph_to_mph = lambda x : x*0.621371

# 6.3, still using 6.7
min_dist_lookup = [ # TODO change this name to be more accurate
    (20,6), #12 mph 
    (30,6.), #18 mph
    (40,7.5), # 24 mph
    (50,9.5), # 30 mph
    (60,12), # 36 mph
    (70,16.5), # 43 mph
    (80,22), # 50 mph
]
min_dist_bps = [x[0] for x in min_dist_lookup]
min_dist_vals = [x[1] for x in min_dist_lookup]


# NOTE tuned recently. This is the one to use TODO maybe use the other one instead
max_speed_lookup = [ # estimated from run260, abq. 
    (.005, 100),
    (.01, 80),
    (.0175, 60),
    (.035, 50),
    (.065, 40),
    (.12, 30),
    (.23, 20),
    (.3, 15),
    (.42, 10),
]
max_speed_bps = [x[0] for x in max_speed_lookup]
max_speed_vals = [kph_to_mps(x[1]) for x in max_speed_lookup]

CRV_WHEELBASE = 2.66 # both OP and internet agree, rw measurement confirms

device = 'cuda'

FPS = 20 # WARNING this is hardcoded throughout codebase. Don't rely on this. TODO consolidate all the places we've hardcoded this

BLENDER_MEMBANK_ROOT = "/home/beans/blender_membank"
#BLENDER_MEMBANK_ROOT = "/dev/shm/blender_membank"

SEQ_LEN = 116 * 1
EPISODE_LEN = SEQ_LEN * 10
RUNS_TO_STORE_PER_PROCESS = 30
N_RUNNERS = 12

get_node = lambda label, nodes : [n for n in nodes if n.label==label][0]

import glob, os

def get_should_stop():
    return np.load(f"{BLENDER_MEMBANK_ROOT}/should_stop.npy")[0] == 1

def set_should_stop(should_stop):
    np.save(f"{BLENDER_MEMBANK_ROOT}/should_stop.npy", np.array([1 if should_stop else 0], dtype='uint8'))

def report_obs_per_sec(dataloader_root, obs_per_sec):
    np.save(f"{dataloader_root}/obs_per_sec.npy", np.array([obs_per_sec], dtype=np.float32))

def get_obs_per_sec():
    try:
        paths = glob.glob(f"{BLENDER_MEMBANK_ROOT}/**/obs_per_sec.npy", recursive=True)
        obs_per_sec_arr = np.array([np.load(p)[0] for p in paths])
        return obs_per_sec_arr.sum(), obs_per_sec_arr.min()
    except:
        return 0, 0
        
def clear_obs_per_sec():
    paths = glob.glob(f"{BLENDER_MEMBANK_ROOT}/**/obs_per_sec.npy", recursive=True)
    for p in paths:
        os.remove(p)


def set_lr(lr):
    np.save(f"{BLENDER_MEMBANK_ROOT}/lr.npy", np.array([lr], dtype='float'))

def get_lr():
    return np.load(f"{BLENDER_MEMBANK_ROOT}/lr.npy")[0]


# These are used in rw rollout to get avg, to smooth out long
CURVE_PREP_SLOWDOWN_S_MIN, CURVE_PREP_SLOWDOWN_S_MAX = 6., 6. #2.5, 3.5 TODO this will go away. Seconds will depend on velocity delta to be covered

MAX_ACCEL = .6 #1.0 #2.0 #m/s/s 3 to 5 is considered avg for an avg driver in terms of stopping, the latter as a sort of max decel

MAP_WIDTH = 120 #80
assert MAP_WIDTH%2==0
MAP_HEIGHT = 180 #120 #IMG_HEIGHT

GPS_HZ = 5

CRV_WIDTH = 1.85 


NORMAL_SHIFT_MAX = 1.0
ROUTE_LEN_M = 1700
WP_SPACING = .1
TRAJ_WP_IXS = np.round(np.array(TRAJ_WP_DISTS) / WP_SPACING).astype('int')

TRAJ_WP_DISTS_NP = np.array(TRAJ_WP_DISTS, dtype='float32')

def linear_to_cos(p):
    # p is linear from 0 to 1. Outputs smooth values from 0 to 1 to back to zero
    return (np.cos(p*np.pi*2)*-1 + 1) / 2

def linear_to_sin_decay(p):
    # p is linear from 0 to 1. Outputs smooth values from 1 to 0
    return np.sin(p*np.pi+np.pi/2) / 2 + .5


def linear_to_sin(p):
    # p is linear from 0 to 1. Outputs smooth values from 0 to 1
    return (np.sin(p*np.pi+np.pi/2) / 2 + .5)*-1 + 1


MAX_N_NPCS = 10

GPS_HZ = 5

# ROUTES_DIR = "/media/beans/ssd/routes"
ROUTES_DIR = "/home/beans/openpilot/routes"

def get_auxs(aux):
    bs, bptt, _ = aux.shape
    # model in
    aux_model = np.zeros((bs, bptt, N_AUX_MODEL_IN), dtype=np.float16)
    aux_model[:,:,2] = aux[:,:,AUX_SPEED_IX]

    # calib in
    aux_calib = np.zeros((bs, bptt, N_AUX_CALIB_IN), dtype=np.float16)
    aux_calib[:,:,0] = aux[:,:,AUX_PITCH_IX]
    aux_calib[:,:,1] = aux[:,:,AUX_YAW_IX]

    # aux targets
    aux_targets = np.zeros((bs, bptt, N_AUX_TARGETS), dtype=np.float16)
    aux_targets[:,:,0] = aux[:,:, AUX_APPROACHING_STOP_IX]
    aux_targets[:,:,1] = aux[:,:, AUX_STOP_DIST_IX]
    aux_targets[:,:,2] = aux[:,:, AUX_STOPPED_IX]

    aux_targets[:,:,3] = aux[:,:, AUX_HAS_LEAD_IX]
    aux_targets[:,:,4] = aux[:,:, AUX_LEAD_DIST_IX]
    aux_targets[:,:,5] = aux[:,:, AUX_LEAD_SPEED_IX]

    return aux_model, aux_calib, aux_targets

DIST_NA_PLACEHOLDER = 150
STOP_LOOKAHEAD_DIST = 60 #100 low res, just very hard to even see stopsigns. 
LEAD_LOOKAHEAD_DIST = 100 # can be bigger than stop lookahead bc cars are bigger objects

import time
class Logger():
    def __init__(self):
        self.tracker = {}
        
    def log(self, to_log):
        for k,v in to_log.items():
            if k in self.tracker:
                self.tracker[k].append(v)
            else:
                self.tracker[k] = [v]
    
    def finish(self):
        r = self.tracker
        for k in r: r[k] = np.round(np.nanmean(np.array(r[k])), 8)
        self.tracker = {}
        return r

class Timer():
    def __init__(self, timer_name):
        self.init_time = time.time()
        self.results = {}
        self.last_milestone_time = self.init_time
        self.timer_name = timer_name

    def log(self, milestone):
        t = time.time() 
        self.results[f"timing/{milestone}"] = t - self.last_milestone_time
        self.last_milestone_time = t
    
    def finish(self):
        self.results[f"timing/{self.timer_name}"] = time.time() - self.init_time
        return self.results

# Img cat
LOOKBEHIND = 1
N_LOOKBEHINDS = 3
SEQ_START_IX = LOOKBEHIND * N_LOOKBEHINDS


def bwify_seq(_img):
    return _img[:, :,:,:1]//3 + _img[:, :,:,1:2]//3 + _img[:, :,:,2:3]//3

def bwify_img(_img):
    return bwify_seq(_img[None, ...])[0]

def cat_imgs(imgs, imgs_bw):
    # bs, seqlen, h, w, c
    # seq ends w current img, and goes back SEQ_START_IX into the past
    img = imgs[:, LOOKBEHIND*3:, :,:,:]
    img_1 = imgs_bw[:, LOOKBEHIND*2:-LOOKBEHIND, :,:,:]
    img_2 = imgs_bw[:, LOOKBEHIND:-LOOKBEHIND*2, :,:,:]
    img_3 = imgs_bw[:, :-LOOKBEHIND*3, :,:,:]
    
    img = np.concatenate([img, img_1, img_2, img_3], axis=-1)
    return img

# def cat_imgs(imgs, imgs_bw):
#     # bs, seqlen, h, w, c
#     img = imgs[:, LOOKBEHIND*3:, :,:,:] # same ixs, these two
#     img_bw = imgs_bw[:, LOOKBEHIND*3:, :,:,:]
#     DIFF_1_N, DIFF_2_N, DIFF_3_N = 1, 2, 3
#     img_1_bw = imgs_bw[:, LOOKBEHIND*3-DIFF_1_N:-DIFF_1_N, :,:,:]
#     img_2_bw = imgs_bw[:, LOOKBEHIND*3-DIFF_2_N:-DIFF_2_N, :,:,:]
#     img_3_bw = imgs_bw[:, LOOKBEHIND*3-DIFF_3_N:-DIFF_3_N, :,:,:]
#     diff = img_bw - img_1_bw

#     # img_1 = imgs_bw[:, LOOKBEHIND*2:-LOOKBEHIND, :,:,:]
#     img_2 = imgs_bw[:, LOOKBEHIND:-LOOKBEHIND*2, :,:,:]
#     img_3 = imgs_bw[:, :-LOOKBEHIND*3, :,:,:]
    
#     # img = np.concatenate([img, img_1, img_2, img_3], axis=-1)
#     # img = np.concatenate([img, diff, img_2, img_3], axis=-1)
#     img = np.concatenate([img, img_1_bw, img_2_bw, img_3_bw], axis=-1)

#     return img

TRT_MODEL_PATH = f"{BESPOKE_ROOT}/trt_models/backbone_trt.jit.pt"