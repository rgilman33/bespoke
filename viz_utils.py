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

def get_rnn_gradcam(act, grad, img, cutoff=.1):
    
    ag = (act * grad).astype('float32')
    ag = cv2.resize(ag, (50, IMG_HEIGHT), interpolation=cv2.INTER_AREA) # width chosen manually
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
    if do_gradcam:
        cudnn_was_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled=False # otherwise can't do backward through RNN w cudnn

    m = EffNet(model_arch="efficientnet_b3", is_for_viz=do_gradcam).to(device)
    m.load_state_dict(torch.load(f"{BESPOKE_ROOT}/models/m_{model_stem}.torch"), strict=False)
    m.eval()
    bs = 1
    m.reset_hidden(bs)

    if do_gradcam:
        m.convert_backbone_to_sequential() # required for the viz version of the model

    rnn_activations, rnn_grads, cnn_activations, cnn_grads, wp_angles_all, wp_headings_all, wp_curvatures_all, obsnet_outs = [], [], [], [], [], [], [], []
    chunk_len, ix = 48, 0 # 24, 0

    while ix < len(img):
        aux_ = pad(aux[ix:ix+chunk_len]) # current speed is important!!!
        img_ = pad(img[ix:ix+chunk_len])
        aux_model, aux_calib, aux_targets = get_auxs(aux_)

        speed_mask = get_speed_mask(aux_model)

        img_, aux_model, aux_calib = prep_inputs(img_, aux_model, aux_calib)

        with torch.cuda.amp.autocast():
            m.zero_grad()
            pred, pred_aux, obsnet_out = m(img_, aux_model, aux_calib) 
            wp_angles, wp_headings, wp_curvatures = torch.chunk(pred, 3, -1)
            wp_angles_all.append(wp_angles[0,:,:].detach().cpu().numpy())
            wp_headings_all.append(wp_headings[0,:,:].detach().cpu().numpy())
            wp_curvatures_all.append(wp_curvatures[0,:,:].detach().cpu().numpy())
            obsnet_outs.append(obsnet_out[0,:,:].detach().cpu().numpy())
            
            if do_gradcam:
                cnn_activations.append(m.activations.mean(1, keepdim=True).numpy()) # (24, 1, 13, 80), mean of channels
                rnn_activations.append(m.rnn_activations[0].numpy()) #torch.Size([24, 512]), 

                (wp_angles * torch.from_numpy(speed_mask).to(device)).mean().backward()
                
                cnn_grads.append(m.gradients.mean(1, keepdim=True).numpy())
                rnn_grads.append(m.rnn_gradients[0].numpy()) #torch.Size([24, 512])
                
            del wp_angles

        ix += chunk_len
        
    wp_angles_all = np.concatenate(wp_angles_all)
    wp_headings_all = np.concatenate(wp_headings_all)
    wp_curvatures_all = np.concatenate(wp_curvatures_all)
    obsnet_outs = np.concatenate(obsnet_outs)

    if do_gradcam:
        cnn_activations, cnn_grads = np.concatenate(cnn_activations, axis=0), np.concatenate(cnn_grads, axis=0)
        rnn_activations, rnn_grads = np.concatenate(rnn_activations, axis=0), np.concatenate(rnn_grads, axis=0)
        
        seqlen, n_acts = rnn_activations.shape
        rnn_activations = rnn_activations.reshape(seqlen, 32, 16) # splitting rnn hidden vector into a rectangle for viewing
        rnn_grads = rnn_grads.reshape(seqlen, 32, 16)

    # a bit dumb, have to pad before denorming
    wp_angles_all = np.expand_dims(wp_angles_all,0) * TARGET_NORM.cpu().numpy()
    wp_angles_all = wp_angles_all[0] 
    wp_headings_all = np.expand_dims(wp_headings_all,0) * TARGET_NORM_HEADINGS.cpu().numpy()
    wp_headings_all = wp_headings_all[0] 
    wp_curvatures_all = np.expand_dims(wp_curvatures_all,0) * TARGET_NORM_CURVATURES
    wp_curvatures_all = wp_curvatures_all[0] 

    if do_gradcam:
        torch.backends.cudnn.enabled = cudnn_was_enabled

    return wp_angles_all, wp_headings_all, wp_curvatures_all, obsnet_outs, cnn_activations, cnn_grads, rnn_activations, rnn_grads



def _make_vid(model_stem, run_id, wp_angles_pred, wp_headings_pred, wp_curvatures_pred, img, aux, wp_angles_targets,
             cnn_grads, cnn_activations, rnn_grads, rnn_activations, add_charts):
    
    height, width, channels = img[0].shape
    w2, h2 = width//2, height//2
    height*= 3 if add_charts else 2 # stacking two, and charts
    width += 50 # for rnn gradcam
    fps = 20
    cutoff = 2.4e-8 #2.4e-6 # adjust this manually when necessary
    MAX_CLIP_TE_VIZ = .2 # viz will be white at this point
    print(height, width, channels)
    video = cv2.VideoWriter(f'/home/beans/bespoke_vids/{run_id}_m_{model_stem}_gradcam.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, (width,height))
    curve_constrained_speed_calculator = CurveConstrainedSpeedCalculator()
    torque_limiter = TorqueLimiter()
    torques_hist, tds_hist, steers_hist = [], [], []

    for i in range(len(img)-1):

        # Gradcam
        g = cnn_grads[i][0].astype(np.float32)
        s = cnn_activations[i][0].astype(np.float32)
        r = combine_img_cam(s*g, img[i], cutoff=cutoff) # this could also be on the processed img

        speed_mps = max(0, kph_to_mps(aux[i, 2])) # TODO why is this max() necessary? why would speed be coming in negative here, even if only slightly? neg values here were breaking draw_wp apparatus

        # traj targets
        if wp_angles_targets is not None:
            r = draw_wps(r, wp_angles_targets[i], color=(100, 200, 200), thickness=-1, speed_mps=speed_mps)

        # traj preds
        traj = wp_angles_pred[i]
        r = draw_wps(r, traj, speed_mps=speed_mps)

        target_wp_angle, wp_dist, _ = get_target_wp(traj, speed_mps)
        r = draw_wps(r, np.array([target_wp_angle]), wp_dists=np.array([wp_dist]), color=(50, 255, 50), thickness=-1)

        close_long_wp_dist, far_long_wp_dist = CURVE_PREP_SLOWDOWN_S_MIN*speed_mps, CURVE_PREP_SLOWDOWN_S_MAX*speed_mps
        close_long_wp_angle, _, _ = angle_to_wp_from_dist_along_traj(traj, close_long_wp_dist)
        far_long_wp_angle, _, _ = angle_to_wp_from_dist_along_traj(traj, far_long_wp_dist)
        r = draw_wps(r, np.array([close_long_wp_angle, far_long_wp_angle]), wp_dists=np.array([close_long_wp_dist, far_long_wp_dist]), color=(50, 50, 255), thickness=-1)

        ##########
        ### info

        r[:70, :120, :] = 0
        headings = wp_headings_pred[i]

        # ccs
        curvatures = wp_curvatures_pred[i]
        curve_constrained_speed = curve_constrained_speed_calculator.step(curvatures, speed_mps)
        r = cv2.putText(r, f"ccs (mph): {round(mps_to_mph(curve_constrained_speed))}", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        r = cv2.putText(r, f"s (mph): {round(mps_to_mph(speed_mps))}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        
        # Torque
        target_wp_angle_deg = np.degrees(target_wp_angle)
        tire_angle_deg, is_abs_torque_limited, is_td_limited, commanded_torque, commanded_td = torque_limiter.step(target_wp_angle_deg, mps_to_kph(speed_mps))
        red =  (255, 100, 100)
        white = (255, 255, 255)
        torques_hist.append(round(abs(commanded_torque)))
        tds_hist.append(round(abs(commanded_td)))
        M = 7
        display_torque = max(torques_hist[-M:])
        display_td = max(tds_hist[-M:])
        r = cv2.putText(r, f"t: {display_torque}", (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, red if is_abs_torque_limited else white, 1)
        r = cv2.putText(r, f"td: {display_td}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, red if is_td_limited else white, 1)

        # steer angle
        steers_hist.append(target_wp_angle_deg)
        display_steer = sum(steers_hist[-M:])/M
        r = cv2.putText(r, f"angle: {round(display_steer, 2)}", (0, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, white, 1)
        
        ####

        # Guidelines
        r[:,w2-1:w2+1,:] -= 20 # darker line vertical center
        r[h2-1:h2+1:,:,:] -= 20 # darker line horizontal center
        
        # RNN actgrad
        r = get_rnn_gradcam(rnn_activations[i], rnn_grads[i], r)

        # Aux charts
        if add_charts:
            charts_fig = get_pts_and_headings_fig(traj, TRAJ_WP_DISTS, headings, curvatures)
            charts = fig_to_img(charts_fig, (IMG_HEIGHT, width, 3))

        r = np.clip(np.flip(r,-1), 0, 255)

        img_ = np.flip(get_rnn_gradcam(rnn_activations[i], rnn_grads[i], img[i]), -1) #DUMB catting on rnn grad just to match sz

        r = np.concatenate([r, img_, charts], axis=0) if add_charts else np.concatenate([r, img_], axis=0)

        video.write(r)

    video.release()


def make_vid(run_id, model_stem, img, aux, targets=None, add_charts=False):
    rollout_data = get_viz_rollout(model_stem, img, aux)
    wp_angles_all, wp_headings_all, wp_curvatures_all, obsnet_outs, cnn_activations, cnn_grads, rnn_activations, rnn_grads = rollout_data
    print(wp_angles_all.shape, cnn_activations.shape, cnn_grads.shape)

    _make_vid(model_stem, run_id, wp_angles_all, wp_headings_all, wp_curvatures_all, img, aux, targets, cnn_grads, cnn_activations, rnn_grads, rnn_activations, add_charts)
    print("Made vid!")
    return rollout_data


def combine_vids(m_path_1, m_path_2, run_id):
    fps = 20
    v1 = cv2.VideoCapture(f'/home/beans/bespoke_vids/{run_id}_m_{m_path_1}_gradcam.avi')
    v2 = cv2.VideoCapture(f'/home/beans/bespoke_vids/{run_id}_m_{m_path_2}_gradcam.avi')
    ret, frame_1 = v1.read()
    ret, frame_2 = v2.read()
    height, width, channels = frame_1.shape

    video = cv2.VideoWriter(f'/home/beans/bespoke_vids/{run_id}_m_{m_path_1}_{m_path_2}.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, (width,height))
    
    while True:
        ret_1, frame_1 = v1.read()
        ret_2, frame_2 = v2.read()
        if not (ret_1 and ret_2): break
        f = np.concatenate([frame_1[:IMG_HEIGHT,:,:], frame_2[:IMG_HEIGHT,:,:]], axis=0)
        video.write(f)

    video.release()
    print("combined!")


        
def get_pts_and_headings_fig(wp_angles, wp_dists, wp_headings, wp_curvatures):

    #plt.close('all') # dunno about this. HACK
    
    xs = np.sin(wp_angles) * wp_dists
    ys = np.cos(wp_angles) * wp_dists
    
    # Near traj
    fig, (ax, ax2, ax3) = plt.subplots(1,3, figsize=(12,3), gridspec_kw={'width_ratios': [1, 1, 2]})
    
    ax.scatter(xs[:20], ys[:20])

    for i in range(0,20-1):
        h = wp_headings[i]
        xd = np.sin(h)*SEGMENT_DISTS[i]*.6
        yd = np.cos(h)*SEGMENT_DISTS[i]*.6
        ax.plot([xs[i], xs[i]+xd], [ys[i], ys[i]+yd], 'Blue')

    ax.set_title("Traj (near)", fontdict={"fontsize":16})
    ax.set_yticks([6, 12, 24])
    
    xmax = max(abs(xs[:19]))
    x_axis_max = .1 if xmax <= .1 else .5 if xmax <= .5 else 1 if xmax <= 1 else 2 if xmax <=2 else 3 if xmax <=3 else 5
    ax.set_xticks([-x_axis_max, 0, x_axis_max])
    
    # far traj
    ax2.scatter(xs[20:], ys[20:])

    for i in range(20, 30-1):
        h = wp_headings[i]
        xd = np.sin(h)*SEGMENT_DISTS[i]*.6
        yd = np.cos(h)*SEGMENT_DISTS[i]*.6
        ax2.plot([xs[i], xs[i]+xd], [ys[i], ys[i]+yd], 'Blue')

    ax2.set_title("Traj (far)", fontdict={"fontsize":16})
    #ax2.set_yticks([35, 125])
    
    xmax = max(abs(xs[20:]))
    x_axis_max = 1 if xmax <= 1 else 5 if xmax <= 5 else 10 if xmax <=10 else 20
    ax2.set_xticks([-x_axis_max, 0, x_axis_max])
    
    # curvatures
    maxc = max(abs(wp_curvatures))
    y_axis_max = .001 if maxc<=.001 else .005 if maxc <= .005 else .01 if maxc <= .01 else .02 if maxc<=.02 else .03
    ax3.plot(wp_dists, wp_curvatures)
    ax3.set_title("Curvatures", fontdict={"fontsize":16})
    ax3.set_yticks([-y_axis_max, 0, y_axis_max])
    ax3.set_xticks([6, 125])
    
    return fig


from skimage.transform import resize
import cv2

def fig_to_img(fig, size):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    chart = (resize(data, size) * 255).astype('uint8')
    return chart