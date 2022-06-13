from constants import *
from imports import *

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

def draw_wps(image, wp_angles, color=(255,0,0)):
    image = copy.deepcopy(image)
    wp_xs, wp_zs = xz_from_angles_and_distances(wp_angles, np.array(TRAJ_WP_DISTS)[:N_WPS_TO_USE])
    wp_ys = np.ones_like(wp_xs) * -1.5
    wps_3d = np.array([[x,y,z] for x, y, z in zip(wp_xs, wp_ys, wp_zs)])
    wps_2d, _ = cv2.projectPoints(wps_3d, rvec, tvec, mtx, dist) # these are projected onto the full img of size 480 x 640
    wps_2d[:,:,1] = IMG_HEIGHT - (wps_2d[:,:,1]-BOTTOM_CHOP)
    #wps_2d[:,:,1] = wps_2d[:,:,1]-BOTTOM_CHOP

    for i in range(len(wps_2d)):
        wp = tuple(wps_2d[i][0].astype(int))
        image = cv2.circle(image, wp, radius=3, color=color, thickness=1)
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

    rnn_activations, rnn_grads, cnn_activations, cnn_grads, preds, obsnet_outs = [], [], [], [], [], []
    chunk_len, ix = 24, 0

    while ix < len(img):
        aux_ = pad(aux[ix:ix+chunk_len]) # current speed is important!!!
        img_ = pad(img[ix:ix+chunk_len])
        ix += chunk_len
        img_, aux_ = prep_inputs(img_, aux_)
        #img_ = side_crop(img_, crop=8)

        with torch.cuda.amp.autocast():
            m.zero_grad()
            pred, obsnet_out = m(img_, aux_) 
            preds.append(pred[0,:,:].detach().cpu().numpy())
            obsnet_outs.append(obsnet_out[0,:,:].detach().cpu().numpy())
            
            if do_gradcam:
                cnn_activations.append(m.activations.mean(1, keepdim=True).numpy()) # (24, 1, 13, 80), mean of channels
                rnn_activations.append(m.rnn_activations[0].numpy()) #torch.Size([24, 512]), 
                
                pred[:,:,GRADCAM_WP_IX].sum().backward() # gradients w respect to lateral preds
                #obsnet_out[:,:,0].sum().backward() # gradients w respect to undertainty preds
                
                cnn_grads.append(m.gradients.mean(1, keepdim=True).numpy())
                rnn_grads.append(m.rnn_gradients[0].numpy()) #torch.Size([24, 512])
                
            del pred
            
    preds = np.concatenate(preds)
    obsnet_outs = np.concatenate(obsnet_outs)

    if do_gradcam:
        cnn_activations, cnn_grads = np.concatenate(cnn_activations, axis=0), np.concatenate(cnn_grads, axis=0)
        rnn_activations, rnn_grads = np.concatenate(rnn_activations, axis=0), np.concatenate(rnn_grads, axis=0)
        
    seqlen, n_acts = rnn_activations.shape
    rnn_activations = rnn_activations.reshape(seqlen, 32, 16)
    rnn_grads = rnn_grads.reshape(seqlen, 32, 16)

    torch.backends.cudnn.enabled = cudnn_was_enabled

    return preds, obsnet_outs, cnn_activations, cnn_grads, rnn_activations, rnn_grads



def make_vid(model_stem, run_id, preds_all, img, 
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

        # wps
        r = draw_wps(r, preds_all[i])

        # Guidelines
        r[:,w2-1:w2+1,:] -= 20 # darker line vertical center
        r[h2-1:h2+1:,:,:] -= 20 # darker line horizontal center

        # RNN actgrad
        r = get_rnn_gradcam(rnn_activations[i], rnn_grads[i], r)

        # agent driving (enabled) or human
        te = np.clip(temporal_error[i] / MAX_CLIP_TE_VIZ, 0, 1.0)
        r[:25, :130, :] = 0
        r[:20, :100, :] = 255*te

        r = np.clip(np.flip(r,-1), 0, 255)

        img_ = np.flip(get_rnn_gradcam(rnn_activations[i], rnn_grads[i], img[i]), -1)

        r = np.concatenate([r, img_], axis=0)

        video.write(r)

    video.release()