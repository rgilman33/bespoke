import numpy as np

BESPOKE_ROOT = '/media/beans/ssd/bespoke'

webcam_img_height = 480
webcam_img_width = 640
IMG_WIDTH = webcam_img_width

BOTTOM_CHOP = 150
TOP_CHOP = 150 + 80

IMG_HEIGHT = 100 #
assert IMG_HEIGHT == (webcam_img_height - TOP_CHOP - BOTTOM_CHOP)

OP_UI_BACKGROUND_WIDTH = 1164
OP_UI_BACKGROUND_HEIGHT = 874
OP_UI_MARGIN = 300

SEQ_LEN = 116 #232 #24
N_CHANNELS = 3
BPTT = 9 #8 #4 #8

DATA_CONSUMPTION_RATIO_LIMIT = 3 #1.

MIN_WP_M = 6 #8
TRAJ_WP_DISTS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] + [35, 45, 55, 65, 75, 85, 95, 105, 115, 125]

N_PRED = len(TRAJ_WP_DISTS)
N_WPS_TO_USE = 30 #22 #15
traj_wp_dists = TRAJ_WP_DISTS[:N_WPS_TO_USE] # Now using all of them
assert traj_wp_dists == TRAJ_WP_DISTS
assert MIN_WP_M==TRAJ_WP_DISTS[0]

aux_properties = [
    'left_blinker',
    'right_blinker',
    'current_speed',
    'speed_as_percent_of_limit',
    'current_tire_angle_rad'
]

aux_norm_constants = np.array([1., 1., 20., 1., 1.], dtype=np.float16)

STEER_RATIO = 16. # taken from OP, specific for crv-5g. Don't change this willy nilly

mps_to_kph = lambda x: (x/1000)*60*60
kph_to_mps = lambda x: (x*1000)/(60*60)


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



BLENDER_MEMBANK_ROOT = "/media/beans/ssd/blender_membank"
EPISODE_LEN = SEQ_LEN * 10
RUNS_TO_STORE_PER_PROCESS = 30
N_RUNNERS = 12


N_TARGETS = N_WPS_TO_USE
N_AUX = len(aux_properties)

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
        obs_per_sec = np.array([np.load(p)[0] for p in paths]).sum()
        return obs_per_sec
    except:
        return 0
        
def clear_obs_per_sec():
    paths = glob.glob(f"{BLENDER_MEMBANK_ROOT}/**/obs_per_sec.npy", recursive=True)
    for p in paths:
        os.remove(p)




# These are used in rw rollout to get avg, to smooth out long
CURVE_PREP_SLOWDOWN_S_MIN, CURVE_PREP_SLOWDOWN_S_MAX = 5., 5. #2.5, 3.5 TODO this will go away. Seconds will depend on velocity delta to be covered