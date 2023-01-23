from constants import *
from imports import *
from train_utils import *
from traj_utils import *
from norm import *


# Camera matrix got from rw calib using the webcam. Notebook on old laptop?
dist = np.float32([[0,0,0,0,0]])
rvec = np.float32([[0],[0],[0]])
tvec = np.float32([[0],[0],[0]])
# mtx = np.float32([[398, 0, 320], # This was the nexigo webcam
#                  [0, 389, 240],
#                  [0, 0, 1.]])

# # the new arducam. 75 deg horizontal, 640 x 480
# mtx = np.float32([[420, 0, 320], #
#                  [0, 420, 240],
#                  [0, 0, 1.]])

# the new arducam. 75 deg horizontal, 1440 x 360
mtx = np.float32([[945, 0, 1440//2], 
                 [0, 945, 1080//2],
                 [0, 0, 1.]])

# [focal_len_px, 0, x_mid]
# [0, focal_len_px, y_mid]
# [0, 0, 0]
# We're using non-distorted ie square px, so fx and fy are the same. I believe doubling res should double focal len. 
# Just using img center as cam center

def xz_from_angles_and_distances(angles, wp_dists):
    wp_xs = np.sin(angles) * wp_dists
    wp_zs = np.cos(angles) * wp_dists
    return wp_xs, wp_zs

def draw_wps(image, wp_angles, wp_dists=np.array(TRAJ_WP_DISTS), color=(255,0,0), thickness=1, speed_mps=None):
    image = copy.deepcopy(image)

    if speed_mps is not None:
        max_ix = max_ix_from_speed(speed_mps)
        wp_angles = wp_angles[:max_ix]
        wp_dists = wp_dists[:max_ix]

    wp_xs, wp_zs = xz_from_angles_and_distances(wp_angles, wp_dists)
    wp_ys = np.ones_like(wp_xs) * -1.5
    wps_3d = np.array([[x,y,z] for x, y, z in zip(wp_xs, wp_ys, wp_zs)])
    wps_2d, _ = cv2.projectPoints(wps_3d, rvec, tvec, mtx, dist) # these are projected onto the full img, ie before crop
    wps_2d[:,:,1] = IMG_HEIGHT - (wps_2d[:,:,1]-BOTTOM_CHOP)

    for i in range(len(wps_2d)):
        wp = tuple(wps_2d[i][0].astype(int))
        image = cv2.circle(image, wp, radius=4, color=color, thickness=thickness)
    return image


def get_text_box(d):
    n_entries = len(d)
    f = 12 # font mult
    text_box = np.zeros((n_entries*f, 140, 3))
    for i, text in enumerate(d):
        text_box = cv2.putText(text_box, text, (0, f*i+f//2), cv2.FONT_HERSHEY_SIMPLEX, .33, white, 1)
    return text_box


def enrich_img(img=None, wps=None, wps_p=None, aux=None, aux_targets_p=None, obsnet_out=None):
    # takes in img np (h, w, c), denormed. Adds trajs and info to it.

    # Trajs, targets if have them
    angles_p, curvatures_p, headings_p, rolls_p, zs_p = np.split(wps_p, 5, -1)
    speed = aux["speed"]
    img = draw_wps(img, angles_p, speed_mps=speed)
    if wps is not None:
        angles, curvatures, headings, rolls, zs = np.split(wps, 5, -1)
        img = draw_wps(img, angles, color=(100, 200, 200), speed_mps=speed)

    # Target wp, pred
    target_wp_angle, wp_dist, _ = get_target_wp(angles_p, speed)
    img = draw_wps(img, np.array([target_wp_angle]), wp_dists=np.array([wp_dist]), color=(200, 50, 100), thickness=-1)

    # Target wp, actual. Using common wp dist, which is fine bc just based on speed. The only actual wp rw has
    img = draw_wps(img, np.array([aux["tire_angle"]]), wp_dists=np.array([wp_dist]), color=(50, 150, 150), thickness=-1)

    # Longitudinal wp, ie end of traj, pred
    far_long_wp_dist = max_pred_m_from_speeds(speed)
    far_long_wp_angle, _, _ = angle_to_wp_from_dist_along_traj(angles_p, far_long_wp_dist)
    img = draw_wps(img, np.array([far_long_wp_angle]), wp_dists=np.array([far_long_wp_dist]), color=(50, 50, 255), thickness=-1)

    # Additional props. Put in aux so gets caught up below
    aux["ccs_p"] = get_curve_constrained_speed(curvatures_p, speed)

    # text box
    a = [f"{p}: {round(aux_targets_p[p],2)}, {round(aux[p], 2)}" for p in AUX_TARGET_PROPS]
    v = [vv for vv in VIEW_INFO_PROPS if vv not in AUX_TARGET_PROPS] # clean up the rest
    a += [f"{p}: {round(aux[p],2)}" for p in v]

    a.append(f"unc_p: {round(obsnet_out['unc_p'], 2)}")

    # Roll
    _, _, target_wp_ix = get_target_wp(rolls_p, speed) # doesn't matter if rolls or whatever, we're just getting ix
    target_wp_ix = int(target_wp_ix)
    if wps is not None:
        a.append(f"roll: {round(float(rolls_p[target_wp_ix]), 3)}, {round(float(rolls[target_wp_ix]), 3)}")
    else:
        a.append(f"roll: {round(float(rolls_p[target_wp_ix]), 3)}")

    text_box = get_text_box(a)
    h,w,_ = text_box.shape
    img[:h, IMG_WIDTH-w:,:3] = text_box

    #img = draw_guidelines(img)
    return img


def draw_guidelines(img):
    height, width, channels = img.shape
    w2, h2 = width//2, height//2
    img[:,w2-1:w2+1,:] = 20
    img[h2-1:h2+1:,:,:] = 20
    return img

red =  (255, 100, 100)
white = (255, 255, 255)



q_lookup = {
    'traj': 6e-4,
    'stop': 4e-5,
    'lead': 6e-5,
}
def _get_actgrad(m, img, aux, backwards_fn, clip_neg=False, do_abs=False, viz_loc=5, backwards_target='traj'):
    grads = []
    m.zero_grad()
    with torch.cuda.amp.autocast(): model_out = m(prep_img(img[None,None, :,:,:]), prep_aux(aux[None,None,:])) 
        
    to_backwards = backwards_fn(model_out, m, aux)
    to_backwards.backward()
    acts = m.acts[viz_loc].mean(1, keepdims=True) # (bs, 1, height, width), mean of channels
    grads = m.grads[viz_loc].mean(1, keepdims=True) #NOTE explore these channel by channel rather than flattened
    actgrad = (acts * grads)[0][0].astype(np.float32)
    actgrad = cv2.resize(actgrad, (IMG_WIDTH,IMG_HEIGHT))

    actgrad /= q_lookup[backwards_target] # rescale to clip dist for viewing
    actgrad = np.clip(actgrad, -1, 1) # clip to -1, 1 for viewing

    # need one or the other. Actgrad in range zero to one.
    if do_abs: actgrad = abs(actgrad)
    elif clip_neg: grads = np.clip(grads, 0, 1)
    else: print("need to get actgrads in range zero to one")
    
    actgrad = np.expand_dims(actgrad, -1) # from zero to one in shape (IMG_HEIGHT, IMG_WIDTH, 1)

    return actgrad
    

def _backwards_lead(model_out, m, aux):
    wp_preds, pred_aux, obsnet_out = model_out
    return (pred_aux[:,: AUX_TARGET_PROPS.index("has_lead")]).mean() # has lead

def _backwards_stop(model_out, m, aux):
    wp_preds, pred_aux, obsnet_out = model_out
    return (pred_aux[:,:,AUX_TARGET_PROPS.index("has_stop")]).mean() # has stop

def _backwards_traj(model_out, m, aux):
    to_pred_mask = get_speed_mask(aux["speed"])
    to_pred_mask = torch.from_numpy(to_pred_mask).to(device)
    wp_preds, pred_aux, obsnet_out = model_out
    wp_angles_p, _, _, _, _ = torch.chunk(wp_preds, 5, -1)
    return ((wp_angles_p * to_pred_mask)*400).mean() #NOTE 400 is a magic to get the right scale

def get_actgrad(m, img, aux, actgrad_target='traj', viz_loc=5):
    m.viz_loc = viz_loc
    if actgrad_target=='traj': return _get_actgrad(m, img, aux, _backwards_traj, do_abs=True, backwards_target='traj', viz_loc=viz_loc)
    elif actgrad_target=='stop': return _get_actgrad(m, img, aux, _backwards_stop, clip_neg=True, backwards_target='stop', viz_loc=viz_loc)
    elif actgrad_target=='lead': return _get_actgrad(m, img, aux, _backwards_lead, clip_neg=True, backwards_target='lead', viz_loc=viz_loc)
    else: print("actgrad target not valid")

def combine_img_actgrad(img, actgrad, color=(8,255,8)):
    """ actgrad in range zero to one, img in range 0 to 255 uint8 """
    actgrad_mask = (actgrad > .01).astype(int)
    actgrad *= actgrad_mask # do we want to do this?
    actgrad_fillin = (actgrad * np.array(color)).astype(np.uint8)
    img_actgrad = (actgrad_fillin + img*(1-actgrad)).astype('uint8')
    return img_actgrad


# q_lookup = {
#     'traj': 6e-4,
#     'stop': 4e-5,
#     'lead': 6e-5,
# }
# def _get_actgrad(m, img, aux_model, aux_calib, backwards_fn, clip_neg=False, do_abs=False, q=.005, backwards_target='traj'):
#     grads = []
#     img.requires_grad_(True)
#     img.register_hook(lambda grad : grads.append(grad.detach().cpu()))
#     with torch.cuda.amp.autocast():
#         m.zero_grad()
#         model_out = m(img, aux_model, aux_calib) 
        
#         to_backwards = backwards_fn(model_out, m, aux_model.detach().cpu().numpy())
#         to_backwards.backward()
    
#     actgrad_source = "level5"
#     if actgrad_source=='img':
#         # The img itself, in normalized state
#         acts = img[0][0].permute(1,2,0).detach().cpu().numpy()
#         grads = grads[0][0][0].permute(1,2,0).numpy()
#         actgrad = grads * acts # the actual actgrads, nothing rescaled yet. This is the actual grad*act
#         actgrad = (actgrad[:,:,:3].mean(-1)) # not taking the movement channel for now
#         actgrad /= q

#     elif actgrad_source=='level5':
#         acts = m.activations.mean(1, keepdim=True).numpy() # (bs, 1, 13, 80), mean of channels
#         grads = m.gradients.mean(1, keepdim=True).numpy()
#         actgrad = (acts * grads)[0][0].astype(np.float32)
#         actgrad = cv2.resize(actgrad, (IMG_WIDTH,IMG_HEIGHT))
#         actgrad /= q_lookup[backwards_target] # rescale to clip dist for viewing

#     actgrad = np.clip(actgrad, -1, 1) # this works in conjunction w the quantile-based rescaling, eliminating outliers
#     # and scaling to -1, 1

#     # need one or the other. Actgrad in range zero to one.
#     if do_abs: actgrad = abs(actgrad)
#     elif clip_neg: grads = np.clip(grads, 0, 1)
#     else: print("need to get actgrads in range zero to one")
    
#     actgrad = np.expand_dims(actgrad, -1) # from zero to one in shape (IMG_HEIGHT, IMG_WIDTH, 1)

#     return actgrad
    
# def _backwards_lead(model_out, m, aux_model):
#     wp_preds, pred_aux, obsnet_out = model_out
#     return (pred_aux[:,:,3]).mean() # has lead

# def _backwards_stop(model_out, m, aux_model):
#     wp_preds, pred_aux, obsnet_out = model_out
#     return (pred_aux[:,:,0]).mean() # has stop

# def _backwards_traj(model_out, m, aux_model):
#     aux_model *= model_in_aux_norm_constants
#     to_pred_mask = get_speed_mask(aux_model)
#     to_pred_mask = torch.from_numpy(to_pred_mask).to(device)
#     wp_preds, pred_aux, obsnet_out = model_out
#     wp_angles_p, _, _, _, _ = torch.chunk(wp_preds, 5, -1)
#     return ((wp_angles_p * to_pred_mask)*400).mean() #TODO need this provided

# def get_actgrad(m, img, aux_model, aux_calib, actgrad_target='traj'):
#     if actgrad_target=='traj': return _get_actgrad(m, img, aux_model, aux_calib, _backwards_traj, do_abs=True, q=.13, backwards_target='traj')
#     elif actgrad_target=='stop': return _get_actgrad(m, img, aux_model, aux_calib, _backwards_stop, clip_neg=True, q=.006, backwards_target='stop')
#     elif actgrad_target=='lead': return _get_actgrad(m, img, aux_model, aux_calib, _backwards_lead, clip_neg=True, q=.058, backwards_target='lead')
#     else: print("actgrad target not valid")

# def combine_img_actgrad(img, actgrad, color=(8,255,8)):
#     """ actgrad in range zero to one, img in range 0 to 255 uint8 """
#     actgrad_mask = (actgrad > .01).astype(int)
#     actgrad *= actgrad_mask # do we want to do this?
#     actgrad_fillin = (actgrad * np.array(color)).astype(np.uint8)
#     img_actgrad = (actgrad_fillin + img*(1-actgrad)).astype('uint8')
#     return img_actgrad




def write_vid(img, filename):
    # just write the frames to a vid, no actgrad or anything
    
    height, width, channels = img[0].shape
    fps = 20
    video = cv2.VideoWriter(f'/home/beans/bespoke_vids/{filename}.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, (width,height))

    for i in range(len(img)-1):
        video.write(img[i][:,:,::-1])

    video.release()
    print(f"{filename} done!")


def plot_aux(aux, props):
    n_cols = 3
    n_rows = math.ceil(len(props)/n_cols)
    fig, subplots = plt.subplots(n_rows,n_cols, figsize=(n_cols*5,n_rows*3)) # figsize is w,h
    subplots = subplots.flatten()

    for i,p in enumerate(props):
        ax = subplots[i]
        data = aux[:,:,p].flatten() 
        if "_dist" in p:
            data = data[data!=DIST_NA_PLACEHOLDER]
        elif p=="lead_speed":
            data = data[data<100]
        ax.hist(data, bins=40)
        ax.set_title(p, fontdict={"fontsize":12})

def plot_wps(ws, speed_mask):
    titles = ["angles", "headings", "curvatures", "rolls", "zs"]
    n_cols = 2
    n_rows = math.ceil(len(ws)/n_cols)
    fig, subplots = plt.subplots(n_rows,n_cols, figsize=(n_cols*8,n_rows*6)) # figsize is w,h
    subplots = subplots.flatten()

    for i in range(len(ws)):
        ax = subplots[i]
        ax.hist(ws[i].flatten(), bins=80)
        w_masked = ws[i]*speed_mask
        ax.hist(w_masked.flatten(), bins=80, alpha=.6)
        ax.set_title(f"{titles[i]}, std {round(float(ws[i].std()), 3)}, {round(float(w_masked.std()),3)}", fontdict={"fontsize":12})