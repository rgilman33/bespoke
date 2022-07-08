from constants import *
from imports import *
from train_utils import *
from traj_utils import *

# Camera matrix got from rw calib using the webcam. Notebook on old laptop?
dist = np.float32([[0,0,0,0,0]])
rvec = np.float32([[0],[0],[0]])
tvec = np.float32([[0],[0],[0]])
# mtx = np.float32([[398, 0, 320], # This was the nexigo webcam
#                  [0, 389, 240],
#                  [0, 0, 1.]])

# the new arducam. 75 deg horizontal
mtx = np.float32([[420, 0, 320], #
                 [0, 420, 240],
                 [0, 0, 1.]])

def xz_from_angles_and_distances(angles, wp_dists):
    wp_xs = np.sin(angles) * wp_dists
    wp_zs = np.cos(angles) * wp_dists
    return wp_xs, wp_zs

def draw_wps(image, wp_angles, wp_dists=np.array(TRAJ_WP_DISTS), color=(255,0,0), thickness=1, speed_mps=None):
    image = copy.deepcopy(image)

    if speed_mps is not None:
        max_ix = max_ix_from_speed(speed_mps) + 1 # TODO want this +1?
        wp_angles = wp_angles[:max_ix]
        wp_dists = wp_dists[:max_ix]

    wp_xs, wp_zs = xz_from_angles_and_distances(wp_angles, wp_dists)
    wp_ys = np.ones_like(wp_xs) * -1.5
    wps_3d = np.array([[x,y,z] for x, y, z in zip(wp_xs, wp_ys, wp_zs)])
    wps_2d, _ = cv2.projectPoints(wps_3d, rvec, tvec, mtx, dist) # these are projected onto the full img of size 480 x 640
    wps_2d[:,:,1] = IMG_HEIGHT - (wps_2d[:,:,1]-BOTTOM_CHOP)
    #wps_2d[:,:,1] = wps_2d[:,:,1]-BOTTOM_CHOP

    for i in range(len(wps_2d)):
        wp = tuple(wps_2d[i][0].astype(int))
        image = cv2.circle(image, wp, radius=3, color=color, thickness=thickness) #-1 fills the circle
    return image



def combine_img_cam(act_grad, img, cutoff):
    """ act_grad np float, just the salmap*gradients. img np out of 255"""
    
    # resize to same as img
    act_grad = cv2.resize(act_grad, (img.shape[1],img.shape[0]))
    
    # Get mask for zeroing out
    mask = np.where(act_grad, (abs(act_grad)>(cutoff)), 0)
    mask = np.expand_dims(mask, -1)

    # -1 to 1, centered at zero 
    # TODO this is wrong, shouldn't be normalizing each img separately
    m = max(act_grad.max(), abs(act_grad.min()))
    act_grad = act_grad/m

    # 0 to 255, centered at 127.5. COLORMAP_JET. Positive is red, negative gradients blue, green in middle
    act_grad = ((act_grad*127.5)+127.5).astype(np.uint8)
    
    # Get heatmap and zero out the boring middle ground
    heatmap = cv2.applyColorMap(act_grad, cv2.COLORMAP_JET)
    heatmap = (heatmap*mask)
     
    cam = heatmap/2 + img*np.clip(mask*-1 + 1.5, 0, 1)
    cam = np.clip(cam, 0, 255).astype('uint8')
    
    return cam

def get_rnn_gradcam(act, grad, img, cutoff=.2):
    
    ag = (act * grad).astype('float32')
    ag = cv2.resize(ag, (int(IMG_HEIGHT/2), IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    # at this point they're about -.5 to .5
    
    ag = ag*2 # TODO kindof a trap, just hardcoding for now to get more in the range of -1 to 1
    
    # Get mask for zeroing out
    mask = np.where(ag, (abs(ag)>(cutoff)), 0)
    mask = np.expand_dims(mask, -1)
    
    # 0 to 255, centered at 127.5. COLORMAP_JET. Positive is red, negative gradients blue, green in middle
    ag = ((ag*127.5)+127.5).astype(np.uint8)
    
    # Get heatmap 
    heatmap = cv2.applyColorMap(ag, cv2.COLORMAP_JET)
    heatmap = heatmap * mask
    
    r = np.concatenate([img, heatmap], axis=1).astype('uint8')
    return r


from models import EffNet
from input_prep import *

def get_viz_rollout(model_stem, img, aux, do_gradcam=True, GRADCAM_WP_IX=10):
    cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled=False # otherwise can't do backward through RNN w cudnn

    m = EffNet(model_arch="efficientnet_b3", is_for_viz=True).to(device)
    m.load_state_dict(torch.load(f"/media/beans/ssd/bespoke/models/m_{model_stem}.torch"))
    m.eval()
    bs = 1
    m.reset_hidden(bs)
    m.convert_backbone_to_sequential() # required for the viz version of the model

    rnn_activations, rnn_grads, cnn_activations, cnn_grads, wp_angles_all, wp_headings_all, obsnet_outs = [], [], [], [], [], [], []
    chunk_len, ix = 24, 0

    while ix < len(img):
        aux_ = pad(aux[ix:ix+chunk_len]) # current speed is important!!!
        img_ = pad(img[ix:ix+chunk_len])
        ix += chunk_len
        img_, aux_ = prep_inputs(img_, aux_)

        with torch.cuda.amp.autocast():
            m.zero_grad()
            pred, obsnet_out = m(img_, aux_) 
            wp_angles, wp_headings, _ = torch.chunk(pred, 3, -1)
            wp_angles_all.append(wp_angles[0,:,:].detach().cpu().numpy())
            wp_headings_all.append(wp_headings[0,:,:].detach().cpu().numpy())
            obsnet_outs.append(obsnet_out[0,:,:].detach().cpu().numpy())
            
            if do_gradcam:
                cnn_activations.append(m.activations.mean(1, keepdim=True).numpy()) # (24, 1, 13, 80), mean of channels
                rnn_activations.append(m.rnn_activations[0].numpy()) #torch.Size([24, 512]), 
                
                wp_angles[:,:,GRADCAM_WP_IX].sum().backward() # gradients w respect to lateral preds
                #obsnet_out[:,:,0].sum().backward() # gradients w respect to undertainty preds
                
                cnn_grads.append(m.gradients.mean(1, keepdim=True).numpy())
                rnn_grads.append(m.rnn_gradients[0].numpy()) #torch.Size([24, 512])
                
            del wp_angles
            
    wp_angles_all = np.concatenate(wp_angles_all)
    wp_headings_all = np.concatenate(wp_headings_all)
    obsnet_outs = np.concatenate(obsnet_outs)

    if do_gradcam:
        cnn_activations, cnn_grads = np.concatenate(cnn_activations, axis=0), np.concatenate(cnn_grads, axis=0)
        rnn_activations, rnn_grads = np.concatenate(rnn_activations, axis=0), np.concatenate(rnn_grads, axis=0)
        
    seqlen, n_acts = rnn_activations.shape
    rnn_activations = rnn_activations.reshape(seqlen, 32, 16)
    rnn_grads = rnn_grads.reshape(seqlen, 32, 16)

    # a bit dumb, have to pad before denorming
    wp_angles_all = np.expand_dims(wp_angles_all,0) * TARGET_NORM.cpu().numpy()
    wp_angles_all = wp_angles_all[0] 

    wp_headings_all = np.expand_dims(wp_headings_all,0) * TARGET_NORM_HEADINGS.cpu().numpy()
    wp_headings_all = wp_headings_all[0] 

    torch.backends.cudnn.enabled = cudnn_was_enabled

    return wp_angles_all, wp_headings_all, obsnet_outs, cnn_activations, cnn_grads, rnn_activations, rnn_grads



def _make_vid(model_stem, run_id, wp_angles_pred, wp_headings_pred, img, aux, wp_angles_targets,
             cnn_grads, cnn_activations, rnn_grads, rnn_activations,
            temporal_error):
    
    height, width, channels = img[0].shape
    w2, h2 = width//2, height//2
    height*=2 # stacking two
    width += 50 # for rnn gradcam
    fps = 20
    cutoff = 2.4e-6 # adjust this manually when necessary
    MAX_CLIP_TE_VIZ = .2 # viz will be white at this point
    video = cv2.VideoWriter(f'/home/beans/bespoke_vids/{run_id}_m_{model_stem}_gradcam.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, (width,height))

    for i in range(len(img)-1):

        # Gradcam
        g = cnn_grads[i][0].astype(np.float32)
        s = cnn_activations[i][0].astype(np.float32)
        r = combine_img_cam(s*g, img[i], cutoff=cutoff) # this could also be on the processed img

        speed_mps = max(0, kph_to_mps(aux[i, 2])) # TODO why is this max() necessary? why would speed be coming in negative here, even if only slightly? neg values here were breaking draw_wp apparatus

        if wp_angles_targets is not None:
            r = draw_wps(r, wp_angles_targets[i], color=(100, 200, 200), thickness=-1, speed_mps=speed_mps)

        # wps
        traj = wp_angles_pred[i]
        r = draw_wps(r, traj, speed_mps=speed_mps)

        target_wp_angle, wp_dist, _ = get_target_wp(traj, speed_mps)
        r = draw_wps(r, np.array([target_wp_angle]), wp_dists=np.array([wp_dist]), color=(50, 255, 50), thickness=-1)

        close_long_wp_dist, far_long_wp_dist = CURVE_PREP_SLOWDOWN_S_MIN*speed_mps, CURVE_PREP_SLOWDOWN_S_MAX*speed_mps
        close_long_wp_angle, _, _ = angle_to_wp_from_dist_along_traj(traj, close_long_wp_dist)
        far_long_wp_angle, _, _ = angle_to_wp_from_dist_along_traj(traj, far_long_wp_dist)
        r = draw_wps(r, np.array([close_long_wp_angle, far_long_wp_angle]), wp_dists=np.array([close_long_wp_dist, far_long_wp_dist]), color=(50, 50, 255), thickness=-1)
        
        # info
        r[:50, :120, :] = 0
        headings = wp_headings_pred[i]
        curve_limited_speed = get_curve_constrained_speed(headings, speed_mps)
        r = cv2.putText(r, f"cls (mph): {round(mps_to_mph(curve_limited_speed))}", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        r = cv2.putText(r, f"s (mph): {round(mps_to_mph(speed_mps))}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        # Guidelines
        r[:,w2-1:w2+1,:] -= 20 # darker line vertical center
        r[h2-1:h2+1:,:,:] -= 20 # darker line horizontal center

        # RNN actgrad
        r = get_rnn_gradcam(rnn_activations[i], rnn_grads[i], r)

        # te
        if temporal_error is not None:
            te = np.clip(temporal_error[i] / MAX_CLIP_TE_VIZ, 0, 1.0)
            r[:25, :130, :] = 0
            r[:20, :100, :] = 255*te

        r = np.clip(np.flip(r,-1), 0, 255)

        img_ = np.flip(get_rnn_gradcam(rnn_activations[i], rnn_grads[i], img[i]), -1)

        r = np.concatenate([r, img_], axis=0)

        video.write(r)

    video.release()


def make_vid(run_id, model_stem, img, aux, targets=None, tire_angle_rad=None):
    wp_angles_all, wp_headings_all, obsnet_outs, cnn_activations, cnn_grads, rnn_activations, rnn_grads = get_viz_rollout(model_stem, img, aux)
    print(wp_angles_all.shape, cnn_activations.shape, cnn_grads.shape)

    temporal_error = None
    if tire_angle_rad is not None:
        speeds_mps = kph_to_mps(aux[:,2])
        trajs = torch.FloatTensor(wp_angles_all[:,:N_WPS_TO_USE]).to('cuda')
        traj_xs, traj_ys = get_trajs_world_space(trajs, speeds_mps, tire_angle_rad, CRV_WHEELBASE)
        temporal_error = np.sqrt(get_temporal_error(traj_xs.cpu(), traj_ys.cpu(), speeds_mps))

    _make_vid(model_stem, run_id, wp_angles_all, wp_headings_all, img, aux, targets, cnn_grads, cnn_activations, rnn_grads, rnn_activations, temporal_error)
    print("Made vid!")
