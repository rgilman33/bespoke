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
        # clip y so never go out of bounds
        wp = (wp[0], np.clip(wp[1], 0, IMG_HEIGHT-1))
        image = cv2.circle(image, wp, radius=3, color=color, thickness=thickness)
    return image


def get_text_box(d):
    n_entries = len(d)
    f = 12 # font mult
    text_box = np.zeros((n_entries*f, 160, 3))
    for i, text in enumerate(d):
        text_box = cv2.putText(text_box, text, (0, f*i+f//2), cv2.FONT_HERSHEY_SIMPLEX, .33, white, 1)
    return text_box

black = (0,0,0)

def add_traj_preds(img, angles_p, speed, color=(255, 10, 10)):
    # Traj pred
    img = draw_wps(img, angles_p, speed_mps=speed, color=color, thickness=-1)

    # Target wp, pred
    target_wp_angle = get_target_wp_angle(angles_p, speed)
    wp_dist = get_target_wp_dist(speed)

    img = draw_wps(img, np.array([target_wp_angle]), wp_dists=np.array([wp_dist]), color=color, thickness=-1)
    img = draw_wps(img, np.array([target_wp_angle]), wp_dists=np.array([wp_dist]), color=black, thickness=1)

    # Longitudinal wp, ie end of traj, pred
    far_long_wp_dist = max_pred_m_from_speeds(speed)
    far_long_wp_angle = angle_to_wp_from_dist_along_traj(angles_p, far_long_wp_dist)
    img = draw_wps(img, np.array([far_long_wp_angle]), wp_dists=np.array([far_long_wp_dist]), color=color, thickness=-1)
    img = draw_wps(img, np.array([far_long_wp_angle]), wp_dists=np.array([far_long_wp_dist]), color=black, thickness=1)
    return img

def enrich_img(img=None, wps=None, wps_p=None, wps_p2=None, aux=None, aux_targets_p=None, aux_targets_p2=None, obsnet_out=None):
    # takes in img np (h, w, c), denormed. Adds trajs and info to it.

    img = img.copy() # have to do this bc otherwise was strange error w our new lstm loader. 
    # Trajs, targets if have them
    speed = aux["speed"]

    ##########
    # Pred wps
    ##########
    if wps_p2 is not None:
        angles_p2, curvatures_p2, headings_p2, rolls_p2, zs_p2 = np.split(wps_p2, 5, -1)
        img = add_traj_preds(img, angles_p2, speed, color=(125, 45, 45)) # brown

    angles_p, curvatures_p, headings_p, rolls_p, zs_p = np.split(wps_p, 5, -1)
    img = add_traj_preds(img, angles_p, speed, color=(255, 40, 40))


    ##########
    # Target wps
    ##########

    # if have target traj, draw it
    if wps is not None:
        angles, curvatures, headings, rolls, zs = np.split(wps, 5, -1)
        img = draw_wps(img, angles, color=(100, 200, 200), speed_mps=speed)

    # Target wp, actual. The only actual wp rw has
    wp_dist = get_target_wp_dist(speed)
    img = draw_wps(img, np.array([aux["tire_angle"]]), wp_dists=np.array([wp_dist]), color=(100, 200, 200), thickness=-1)


    ##########
    # Aux info
    ##########

    # text box
    if aux_targets_p is not None:
        a = [f"{p}: {round(aux_targets_p[p],2)}, {round(aux_targets_p2[p],2) if aux_targets_p2 is not None else ''}, {round(aux[p], 2)}" for p in AUX_TARGET_PROPS]
        ccs = f"{round(get_curve_constrained_speed(curvatures_p, speed), 0)}"
        ccs2 = f"{round(get_curve_constrained_speed(curvatures_p2, speed),0)}" if wps_p2 is not None else ""
        a += [f"ccs: {ccs}, {ccs2}"]
    else:
        a = []

    if obsnet_out is not None:
        a.append(f"unc_p: {round(obsnet_out['unc_p'], 2)}")

    # a.append(f"maps, route: {aux['has_map']}, {aux['has_route']}")

    # Roll, using value closest to ego
    if wps is not None:
        a.append(f"roll: {round(float(rolls_p[0]), 2)}, {round(float(rolls[0]), 2)}")
    else:
        a.append(f"roll: {round(float(rolls_p[0]), 2)}, {round(float(rolls_p2[0]), 2) if wps_p2 is not None else ''}")

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

def make_enriched_vid_trn(rollout):
    # quick hack to get it to work w sim, mostly just the fn below
    height, width, channels = IMG_HEIGHT, IMG_WIDTH, 3
    fps = 20 // FRAME_CAPTURE_N
    filename = f"sim_{rollout.model_stem}"
    video = cv2.VideoWriter(f'/home/beans/bespoke_vids/{filename}.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, (width,height))

    for i in range(len(rollout.img)):
        if i%1000==0: print(i)
        wps_p2, aux_targets_p2 = None, None
        img = enrich_img(img=rollout.img[i][:,:,:3], 
                         wps=rollout.wps[i],
                         wps_p=rollout.wps_p[i], wps_p2=wps_p2, 
                         aux_targets_p=rollout.aux_targets_p[i], aux_targets_p2=aux_targets_p2,
                        aux=rollout.aux[i], 
                        obsnet_out=rollout.obsnet_outs[i])
        video.write(img[:,:,::-1])

    video.release()
    print(f"{filename} done!")

def make_enriched_vid(run_id, model_stem, model_stem_b=None): # rw
    rollout = load_object(f"{BESPOKE_ROOT}/tmp/{run_id}_{model_stem}_rollout.pkl")
    if model_stem_b is not None: rollout_b = load_object(f"{BESPOKE_ROOT}/tmp/{run_id}_{model_stem_b}_rollout.pkl")
    run = load_object(f"{BESPOKE_ROOT}/tmp/runs/{run_id}.pkl")

    height, width, channels = IMG_HEIGHT, IMG_WIDTH, 3
    fps = 20
    filename = f"{rollout.run_id}_{rollout.model_stem}" if model_stem_b is None else f"{rollout.run_id}_{rollout.model_stem}_VS_{rollout_b.model_stem}"
    video = cv2.VideoWriter(f'/home/beans/bespoke_vids/{filename}.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, (width,height))

    for i in range(len(run.img_chunk[0])):
        if i%1000==0: print(i)
        wps_p2, aux_targets_p2 = None, None
        if model_stem_b is not None:
            wps_p2 = rollout_b.wps_p[i]
            aux_targets_p2 = rollout_b.aux_targets_p[i]
        img = enrich_img(img=run.img_chunk[0][i][:,:,:3], 
                         wps_p=rollout.wps_p[i], wps_p2=wps_p2, 
                         aux_targets_p=rollout.aux_targets_p[i], aux_targets_p2=aux_targets_p2,
                        aux=rollout.aux[i], 
                        obsnet_out=rollout.obsnet_outs[i])
        video.write(img[:,:,::-1])

    video.release()
    print(f"{filename} done!")

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


