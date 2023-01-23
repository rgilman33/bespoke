import numpy as np


BESPOKE_ROOT = '/home/beans/bespoke'


###########################
# img, map dimensions
###########################

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


MAP_WIDTH = 120 #80
assert MAP_WIDTH%2==0
MAP_HEIGHT = 180 #120 #IMG_HEIGHT


###########################
# WPs
###########################

MIN_WP_M = 6 #8
TRAJ_WP_DISTS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] + [35, 45, 55, 65, 75, 85, 95, 105, 115, 125]
assert MIN_WP_M==TRAJ_WP_DISTS[0]
N_WPS = len(TRAJ_WP_DISTS)

N_WPS_TARGETS = N_WPS*5 # currently is wp_angle, curvature, heading, roll, z-delta

SPACING_BTWN_FAR_WPS = 10
LAST_NEAR_WP_DIST_M = 25
LAST_NEAR_WP_IX = 19
# These are the dists at all the midway pts btwn our wps, plus a zero in the beginning
# they're staggered by .5m for closer wps, then 5m for farther ones
WP_HALFWAYS = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 30.0] 
WP_HALFWAYS += [40, 50, 60, 70, 80, 90, 100, 110, 120] # last heading is the one btwn our second to last and our last wp
# halfways are the halfway pt on each segment, there is one less than the number of wps

HEADING_BPS = [0] + WP_HALFWAYS

SEGMENT_DISTS = np.array(TRAJ_WP_DISTS[1:]) - np.array(TRAJ_WP_DISTS[:-1]) # 19 ones then 10 tens. There is one fewer segment than there are wps


###########################
# Conversions
###########################
mps_to_mph = lambda x: x*2.23694 # just used for human readability
mph_to_mps = lambda x : x*.44704

###########################
# rw
###########################

# 6.3, still using 6.7
min_dist_lookup = [ # TODO change this name to be more accurate. NOTE are we looking up constant time here? ie is this a constant mult of eg .8s ahead?
    (8.33, 6.), #18 mph (speed mps, wp dist m)
    (11.11, 7.5), # 24 mph
    (13.89, 9.5), # 30 mph
    (16.67, 12), # 36 mph
    (19.44, 16.5), # 43 mph
    (22.22, 22), # 50 mph
]
min_dist_bps = [x[0] for x in min_dist_lookup]
min_dist_vals = [x[1] for x in min_dist_lookup]


CRV_WHEELBASE = 2.66 # both OP and internet agree, rw measurement confirms
STEER_RATIO = 16. # taken from OP, specific for crv-5g. Don't change this willy nilly
CRV_WIDTH = 1.85 

GPS_HZ = 5

ROUTES_DIR = "/media/beans/ssd/routes"


###########################
# Trn logistics
###########################

import glob, os

device = 'cuda'

FPS = 20 # WARNING this is hardcoded throughout codebase. Don't rely on this. TODO consolidate all the places we've hardcoded this

BLENDER_MEMBANK_ROOT = "/home/beans/blender_membank"
#BLENDER_MEMBANK_ROOT = "/dev/shm/blender_membank"


SEQ_LEN = 16 #64 #116 * 1
EPISODE_LEN = SEQ_LEN * (1280//SEQ_LEN)
RUNS_TO_STORE_PER_PROCESS = 128 #30
N_RUNNERS = 12

BPTT = 1 #4 #8 #9

DATA_CONSUMPTION_RATIO_LIMIT = 3 #1.

# trn loader
def get_loader_should_stop():
    return np.load(f"{BESPOKE_ROOT}/tmp/trnloader_should_stop.npy")[0]

def set_loader_should_stop(should_stop):
    np.save((f"{BESPOKE_ROOT}/tmp/trnloader_should_stop.npy"), np.array([should_stop], dtype='bool'))

# trainer
def get_trainer_should_stop():
    return np.load(f"{BESPOKE_ROOT}/tmp/trainer_should_stop.npy")[0]

def set_trainer_should_stop(should_stop):
    np.save((f"{BESPOKE_ROOT}/tmp/trainer_should_stop.npy"), np.array([should_stop], dtype='bool'))


# datagen
def get_should_stop():
    return np.load(f"{BLENDER_MEMBANK_ROOT}/should_stop.npy")[0] == 1

def set_should_stop(should_stop):
    np.save(f"{BLENDER_MEMBANK_ROOT}/should_stop.npy", np.array([1 if should_stop else 0], dtype='uint8'))

def report_obs_per_sec(dataloader_root, obs_per_sec):
    np.save(f"{dataloader_root}/{OBS_PER_SEC_F}", np.array([obs_per_sec], dtype=np.float32))

def get_obs_per_sec():
    runners_roots = glob.glob(f"{BLENDER_MEMBANK_ROOT}/*/{OBS_PER_SEC_F}")
    obs_per_sec_arr = np.empty(len(runners_roots))
    for i,f in enumerate(runners_roots):
        try: # every blue moon getting error on this, maybe bc it's currently being written to
            runner_obs_per_sec = np.load(f) if os.path.exists(f) else 0
        except:
            runner_obs_per_sec = 0
            print("couldn't get obs per sec")
        obs_per_sec_arr[i] = runner_obs_per_sec
    s = round(obs_per_sec_arr.sum(),2)
    return (0,0) if s==0 else (s, round(obs_per_sec_arr.min(),2))

        
def clear_obs_per_sec():
    paths = glob.glob(f"{BLENDER_MEMBANK_ROOT}/**/{OBS_PER_SEC_F}", recursive=True)
    for p in paths:
        os.remove(p)

def set_lr(lr):
    np.save(f"{BLENDER_MEMBANK_ROOT}/lr.npy", np.array([lr], dtype='float'))

def get_lr():
    return np.load(f"{BLENDER_MEMBANK_ROOT}/lr.npy")[0]

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

OBS_PER_SEC_F = "obs_per_sec.npy"

def get_mins_since_slowest_runner_reported():
    # each runner updates its obs_per_sec file when it completes a run. We're checking the timestamps of those files directly.
    dl_paths = glob.glob(f"{BLENDER_MEMBANK_ROOT}/*/{OBS_PER_SEC_F}")
    mins_since_last_update = []
    for dl in dl_paths:
        mins_since_last_update.append((time.time() - os.path.getmtime(dl)) / 60)
    return max(mins_since_last_update) if len(mins_since_last_update)>0 else 0


###########################
# Datagen sim
###########################

NORMAL_SHIFT_MAX = 1.0
ROUTE_LEN_M = 1700
WP_SPACING = .1
TRAJ_WP_IXS = np.round(np.array(TRAJ_WP_DISTS) / WP_SPACING).astype('int')
TRAJ_WP_DISTS_NP = np.array(TRAJ_WP_DISTS, dtype='float32')

MAX_N_NPCS = 12

DIST_NA_PLACEHOLDER = 150

LEAD_OUTER_DIST, LEAD_INNER_DIST = 100, 80
STOP_OUTER_DIST, STOP_INNER_DIST = 80, 60

class EpisodeInfo():
    def __init__(self):
        pass

###########################
# Util fns
###########################

get_node = lambda label, nodes : [n for n in nodes if n.label==label][0]

def linear_to_cos(p):
    # p is linear from 0 to 1. Outputs smooth values from 0 to 1 to back to zero
    return (np.cos(p*np.pi*2)*-1 + 1) / 2

def linear_to_sin_decay(p):
    # p is linear from 0 to 1. Outputs smooth values from 1 to 0
    return np.sin(p*np.pi+np.pi/2) / 2 + .5


def linear_to_sin(p):
    # p is linear from 0 to 1. Outputs smooth values from 0 to 1
    return (np.sin(p*np.pi+np.pi/2) / 2 + .5)*-1 + 1

def sigmoid_python(x):
    return 1.0 / (1.0 + np.exp(-x))

def smooth_dist_clf(arr, _min, _max):
    # takes in dists eg stop and lead, outputs smooth interp from min to max 0 to one instead of stepwise binary clf
    arr -= _min
    arr /= (_max-_min)
    arr = np.clip(arr, 0,1)
    arr = linear_to_sin_decay(arr)
    return arr

import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
        return obj

def moving_average(arr, w):
    return np.convolve(arr, np.ones(w), 'same') / w

def dist(a, b):
    x = float(a[0]) - float(b[0])
    y = float(a[1]) - float(b[1])
    return (x**2 + y**2)**(1/2)


def get_random_roll_noise(window_size=20, num_passes=2): #TODO rename this, it's a general fn
    # returns smoothed noise btwn -1 and 1
    roll_noise = np.random.random(EPISODE_LEN*FPS) - .5
    for i in range(num_passes):
        roll_noise = moving_average(roll_noise, 20)
    roll_noise = roll_noise / abs(roll_noise).max()
    return roll_noise

def side_crop(img, crop=20):
    # Side crop. The model will lean towards the side you crop off of. ie the model steers away from the way the cam is yaw tilted
    # eg if cam tilted to the right car will try and correct to the left
    _,_,_, H, W = img.shape
    img = img[0,:, :, :, :-crop]
    #img = img[0,:, :, :, crop:]
    #img = torchvision.transforms.Resize((H, W))(img) # commented out torchvision bc of cuda version incompatability. Can bring back in if reinstall
    img = img.unsqueeze(0)
    return img


def gamma_correct_auto(img):
    # expects imgs in range 0 - 255
    img = (img/255.)
    mean = img.mean()
    target_mean = .5
    gamma = np.log(mean) / np.log(target_mean)
    
    img = img**(1./gamma) # this is the part that takes the most time, 3 out of the 4 ms
    img = (img*255.).astype('uint8')
    return img

# https://numpy.org/doc/stable/user/basics.subclassing.html
# https://stackoverflow.com/questions/54295616/changing-behavior-of-getitem-and-setitem-in-numpy-array-subclass
class NamedArray(np.ndarray):
    """
    Wrap a np array so that can use str ix on last dim. Takes in normal array, and list of names. Is otherwise completely same as normal arr.
    """
    def __new__(cls, input_array, names):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.names = list(names) # in case coming in as np array
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj): # do we actually need this?
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.names = getattr(obj, 'names', None)
        
    def _swap_name_for_ix_if_necessary(self, item):
        if type(item) == tuple and type(item[-1]) == str: 
            item = list(item)
            item[-1] = self.names.index(item[-1])
            item = tuple(item)
        elif type(item) == str:
            item = self.names.index(item)
        return item
    
    def __getitem__(self, item):
        item = self._swap_name_for_ix_if_necessary(item)
        return super().__getitem__(item)
    
    def __setitem__(self, item, value):
        item = self._swap_name_for_ix_if_necessary(item)
        return super().__setitem__(item, value)

    # https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
    # otherwise lose names attr on pickling
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(NamedArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.names,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.names = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(NamedArray, self).__setstate__(state[0:-1])

def na(arr, names): return NamedArray(arr, names)

###########################
# Assembling img input
###########################

# Img cat
FAST_LB_IX = 0 #2
SLOW_1_LB_IX = 0 #20
SLOW_2_LB_IX = 0 #40

assert IMG_WIDTH%2==0 and IMG_HEIGHT%2==0
# SMALL_IMG_HEIGHT = IMG_HEIGHT//2
# SMALL_IMG_WIDTH = IMG_WIDTH//2
IMG_HEIGHT_MODEL = IMG_HEIGHT #+SMALL_IMG_HEIGHT
IMG_WIDTH_MODEL = IMG_WIDTH
N_CHANNELS_MODEL = 3 #4

def bwify_seq(_img):
    #return _img[:, :,:,:1]//3 + _img[:, :,:,1:2]//3 + _img[:, :,:,2:3]//3
    return _img[:, :,:,:1] # just returning r channel, for perf

def bwify_img(_img):
    return bwify_seq(_img[None, ...])[0]

def cat_imgs(imgs, imgs_bw, maps, aux, seq_len=1):
    # single seqs, no batch. All ixs calculated from the end, so seqs can have extra in beginning, but all must end on current obs
    aux = aux[-seq_len:, :]
    img_final = np.zeros((seq_len, IMG_HEIGHT_MODEL, IMG_WIDTH_MODEL, N_CHANNELS_MODEL), dtype='uint8')

    # Current img w current map
    # setting them all in place on final img to avoid making copies
    current_img = imgs[-seq_len:, :,:,:]
    img_final[:, :IMG_HEIGHT,:,:3] = current_img
    maps = maps[-seq_len:, :,:,:] # maps can come in same len as imgs (rw) or shorter (trn). Doesn't matter bc we're right ixing
    # img_final[:, -MAP_HEIGHT-SMALL_IMG_HEIGHT:-SMALL_IMG_HEIGHT,-MAP_WIDTH:,:3] = maps # current img gets current map in bottom right corner, no others get map
    img_final[:, -MAP_HEIGHT:,-MAP_WIDTH:,:3] = maps # current img gets current map in bottom right corner, no others get map

    # # Fast lb on current img channelwise
    # fast_lookbehind_img = imgs_bw[-seq_len-FAST_LB_IX:-FAST_LB_IX, :,:,:]
    # img_final[:, :IMG_HEIGHT,:,3:] = fast_lookbehind_img

    # HUD
    hud = get_hud(aux)
    _, h,w,c = hud.shape
    img_final[:, -h:,-w:,:c] = hud

    return img_final

def get_hud(aux):
    HUD_SQUARE_SZ = 10
    get_hud_square = lambda : np.zeros((len(aux), HUD_SQUARE_SZ, HUD_SQUARE_SZ, 3), dtype='uint8')

    # Speed
    speed = get_hud_square()
    speed[:, :,:,0] = np.interp(aux[:, "speed"], [0, 30], [0, 255])[:, None,None]
    speed[:, :,:,1] = np.interp(aux[:, "speed"], [30, 0], [0, 255])[:, None,None]
    speed[:, :,:,2] = np.interp(aux[:, "speed"], [10, 20], [0, 255])[:, None,None]

    # Map and route presence
    has_map_route = get_hud_square()
    has_map_route[:, :,:,0] = (aux[:, "has_map"]*255)[:, None,None]
    has_map_route[:, :,:,1] = (aux[:, "has_route"]*255)[:, None,None]
    has_map_route[:, :,:,2] = ((aux[:, "has_route"]*-1+1)*255)[:, None,None]

    hud = np.concatenate([has_map_route, speed], axis=-3) # stack on height dim
    return hud
    
###########################
# Misc
###########################

TRT_MODEL_PATH = f"{BESPOKE_ROOT}/trt_models/backbone_trt.jit.pt"

SIM_RUN_ID = "sim"

###########################
# Property reference
###########################
import pandas as pd
propref = pd.read_csv(f"{BESPOKE_ROOT}/propref.csv").fillna(0)
propref['ix'] = list(range(len(propref)))
AUX_PROPS = list(propref.prop.values)

propref_aux_target = propref[propref.aux_target==1]
AUX_TARGET_PROPS = list(propref_aux_target.prop.values)
AUX_TARGET_IXS = list(propref_aux_target.ix.values)
propref_aux_target['ixx'] = list(range(len(propref_aux_target))) # dumb self ix so can grab sigmoid from within this smaller df
AUX_TARGET_SIGMOID_IXS = list(propref_aux_target[propref_aux_target.sigmoid==1].ixx)

propref_obsnet = propref[propref.obsnet==1]
OBSNET_PROPS = list(propref_obsnet.prop.values)
OBSNET_IXS = propref_obsnet.ix.values

propref_model = propref[propref.model==1]
AUX_MODEL_PROPS = list(propref_model.prop.values)
AUX_MODEL_IXS = propref_model.ix.values

propref_episode = propref[propref.episode_info==1]
EPISODE_PROPS = list(propref_episode.prop.values)

propref_rollout = propref[propref.rollout==1]
ROLLOUT_PROPS = list(propref_rollout.prop.values)

get_img_container = lambda bs, seqlen : np.empty((bs, seqlen, IMG_HEIGHT_MODEL, IMG_WIDTH_MODEL, N_CHANNELS_MODEL), dtype='uint8')
get_aux_container = lambda bs, seqlen : na(np.empty((bs, seqlen, len(AUX_PROPS)), dtype=np.float32), AUX_PROPS)
get_targets_container = lambda bs, seqlen : np.empty((bs, seqlen, N_WPS*4), dtype=np.float32)
get_maps_container = lambda bs, seqlen : np.zeros((bs, seqlen, MAP_HEIGHT, MAP_WIDTH, 3), dtype='uint8')


AUX_NORM_SHIFTS = propref.norm_shift.values
AUX_NORM_SCALES = propref.norm_scale.values

SHOW_WORST_PROPS = list(propref[propref.show_worst==1].prop.values)
VIEW_INFO_PROPS = list(propref[propref.view_info==1].prop.values)
