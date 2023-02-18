from constants import *
from imports import *

# All norming accepts bs, seqlen, n, or single seq, or single obs

###########################
# img
###########################

IMG_NORM_MEAN = .5
IMG_NORM_STD = .25

# TODO try this, might be faster
# def norm_img_uint8(img):
#     m = 128
#     std = 64
#     return ((img-m) / std)

def norm_img(img):
    return (img-IMG_NORM_MEAN) / IMG_NORM_STD

def denorm_img(img):
    return (img*IMG_NORM_STD) + IMG_NORM_MEAN

def prep_img(img, is_for_trt=False):
    # bs, seqlen, h, w, c in np or single obs
    n = len(img.shape)
    new_shape = (0,1, 4,2,3) if n==5 else (2,0,1)
    img = np.transpose(img, new_shape) # channels first for pytorch
    img = torch.from_numpy(img) #img should still be uint8 at this point. <1ms
    img = img.to('cuda') # ~40ms as uint8, more as half #TODO try the pinned memory thing here. Faster?
    img = img / 255. # <1ms # implicitly converts to float
    img = norm_img(img) # <1msnorming should return float32
    if not is_for_trt: img = img.half() # <1ms
    return img

def unprep_img(img):
    img = (denorm_img(img.detach()) * 255).cpu().numpy().astype('uint8')
    n = len(img.shape)
    new_shape = (0,1, 3,4,2) if n==5 else (1,2,0)
    img = np.transpose(img, new_shape) # channels last for np
    return img



###########################
# WPS
###########################
# these should roughly match the stds of the properties, using the masked values
# Masking only affects angles and heading and zs, as these are relative to ego and get more extreme at distance
WP_NORM_ANGLE = .02 * np.linspace(1, 10, num=N_WPS).astype('float32') 
WP_NORM_HEADING = .04 * np.linspace(1, 10, num=N_WPS).astype('float32')
WP_NORM_CURVATURE = .006 * np.ones(N_WPS, dtype='float32')
WP_NORM_ROLL = .02 * np.ones(N_WPS, dtype='float32')
WP_NORM_Z = 3. * np.ones(N_WPS, dtype='float32')

WPS_NORM = np.concatenate([WP_NORM_ANGLE, WP_NORM_HEADING, WP_NORM_CURVATURE, WP_NORM_ROLL, WP_NORM_Z]) #NOTE this will rely on broadcasting. Don't actually like it. Would rather be explicit, but i also like being agnostic to if have batch, seq dims or not

def prep_wps(wps):    
    wps /= WPS_NORM
    wps = torch.from_numpy(wps).to('cuda') #.pin_memory().to("cuda", non_blocking=True) # "When non_blocking, tries to convert asynchronously with respect to the host if possible, e.g., converting a CPU Tensor with pinned memory to a CUDA Tensor."
    wps = wps.half()
    return wps

def unprep_wps(wps):
    wps = wps.detach().cpu().numpy().astype(np.float32)
    wps *= WPS_NORM
    return wps

###########################
# Aux
###########################

def norm_aux(aux):
    return (aux - AUX_NORM_SHIFTS) / AUX_NORM_SCALES

def denorm_aux(aux):
    return (aux * AUX_NORM_SCALES) + AUX_NORM_SHIFTS

def prep_aux(aux, is_for_trt=False):
    aux = norm_aux(aux)
    aux = torch.from_numpy(aux).to(device)
    if not is_for_trt:
        # trt keeps as full float inputs, but will change them to halves itself
        aux = aux.half()
    return aux

def unprep_aux(aux):
    aux = aux.detach().cpu().numpy().astype('float32')
    aux = denorm_aux(aux)
    aux = na(aux, AUX_PROPS) # we lose the names when go through pt
    return aux


###########################
# Misc
###########################

def unprep_obsnet(obsnet):
    obsnet = obsnet.detach().cpu().numpy().astype(np.float32)
    obsnet = (obsnet * AUX_NORM_SCALES[OBSNET_IXS]) + AUX_NORM_SHIFTS[OBSNET_IXS]
    obsnet = na(obsnet, OBSNET_PROPS)
    return obsnet

def unprep_aux_targets(aux_targets):
    aux_targets = aux_targets.detach().cpu().numpy()
    aux_targets = (aux_targets * AUX_NORM_SCALES[AUX_TARGET_IXS]) + AUX_NORM_SHIFTS[AUX_TARGET_IXS]
    aux_targets = na(aux_targets, AUX_TARGET_PROPS)
    return aux_targets
