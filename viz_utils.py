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


"""
### 161d
nov30. back on RR mostly agent. couple veers left bc agent. 30 mph mostly

### 161b
nov30 back on PP all human. Gathered specifically for training. 40 - 50 mph

### 161e
nov30. human back on RR gravel.

### 172
back from PO all the way. Mostly garbage agent

### 176
An on and off cloudy ish day. Halfway through put on polarizing filter. All human
176b cam a santa ana
176c mineral de luz
176e and 176f out and back towards la concepcion
176g paved lined GTO to DH out a bit and back

### 178a, 178b, 179a
agent run w 2_23 model. Same course as 176. 
"""