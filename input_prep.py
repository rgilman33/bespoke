from constants import *
from imports import *

def norm_aux(aux):
    aux = aux / torch.from_numpy(aux_norm_constants).to('cuda')
    aux = aux.clip(-8,8) # TODO pay attn to this, we clipping anything important? 
    return aux

pad = lambda x: np.expand_dims(x,0)

IMG_NORM_MEAN = .5
IMG_NORM_STD = .25

#wp_ix_norm = torch.from_numpy(np.linspace(1, 4.27, num=N_WPS_TO_USE).astype('float16')) # this was when pred 22 out
wp_ix_norm = torch.from_numpy(np.linspace(1, 5.5, num=N_WPS).astype('float32')) 
wp_ix_norm = wp_ix_norm.unsqueeze(0).unsqueeze(0)#.to('cuda')
# TARGET_NORM = .035 * wp_ix_norm 
TARGET_NORM = .015 * wp_ix_norm 

wp_ix_norm_headings = torch.from_numpy(np.linspace(1, 5.5, num=N_WPS).astype('float32')) 
wp_ix_norm_headings = wp_ix_norm_headings.unsqueeze(0).unsqueeze(0)#.to('cuda')
# TARGET_NORM = .035 * wp_ix_norm 
TARGET_NORM_HEADINGS = .025 * wp_ix_norm_headings

TARGET_NORM_CURVATURES = .015 


def norm_img(img):
    return (img-IMG_NORM_MEAN) / IMG_NORM_STD

def denorm_img(img):
    return (img*IMG_NORM_STD) + IMG_NORM_MEAN

def prep_inputs(image, aux, targets=None, is_single_obs=False):
    """ np to pytorch totally prepped for model in """

    image = image.astype(np.float16)
    aux = aux.astype(np.float16)

    if is_single_obs:
        image = pad(pad(image))
        aux = pad(pad(aux))
    
    aux = torch.from_numpy(aux).to('cuda') # from_numpy matches dtype, uses same mem
    aux = norm_aux(aux) 

    image = torch.from_numpy(image).to('cuda')
    image = image.permute(0,1,4,2,3)
    image /= 255.
    
    # img is from zero to one, now let's standardize. Could have just done it above
    image = norm_img(image)

    if targets is not None:
        wp_angles, wp_headings, wp_curvatures = targets #TODO norm and return headings

        wp_angles = torch.from_numpy(wp_angles).to('cuda')
        wp_angles = wp_angles / TARGET_NORM.to('cuda')
        wp_angles = wp_angles.half()

        wp_headings = torch.from_numpy(wp_headings).to('cuda')
        wp_headings = wp_headings / TARGET_NORM_HEADINGS.to('cuda')
        wp_headings = wp_headings.half()

        wp_curvatures = torch.from_numpy(wp_curvatures).to('cuda')
        wp_curvatures = wp_curvatures / TARGET_NORM_CURVATURES
        wp_curvatures = wp_curvatures.half()
        
        return (image, aux, wp_angles, wp_headings, wp_curvatures)

    return (image, aux)


from skimage.draw import line_aa

def draw_compass(img, gps_nav_traj):
    compass_line_len = 16
    margin = 0
    compass_base_x = IMG_WIDTH - compass_line_len - margin
    compass_base_y = compass_line_len + margin
    img[0:compass_line_len+4, IMG_WIDTH-compass_line_len*2:IMG_WIDTH, :] = 0

    wp_angle = gps_nav_traj[0] - math.pi/2 # bc 0 is pointing straight to the right

    rr, cc, val = line_aa(compass_base_y, 
                            compass_base_x, 
                            compass_base_y + int(compass_line_len * np.sin(wp_angle)), 
                            compass_base_x + int(compass_line_len * np.cos(wp_angle)))
    val = np.expand_dims(val,-1)
    img[rr, cc] = val * 255

    return img

def draw_turn_indicator(img, turn_command, copy=False):
    if copy:
        img = np.copy(img)
    compass_line_len = 16
    img[0:compass_line_len+4, IMG_WIDTH-compass_line_len*2:IMG_WIDTH, :] = 0
    
    if turn_command=='LEFT':
        #img[0:compass_line_len+4, IMG_WIDTH-compass_line_len*2:IMG_WIDTH-compass_line_len, :] = 255
        img[0:compass_line_len//4, IMG_WIDTH-compass_line_len*2:IMG_WIDTH-compass_line_len, :] = 255
    elif turn_command=='RIGHT':
        #img[0:compass_line_len+4, IMG_WIDTH-compass_line_len:IMG_WIDTH, :] = 255
        img[0:compass_line_len//4, IMG_WIDTH-compass_line_len:IMG_WIDTH, :] = 255
    return img


def side_crop(img, crop=20):
    # Side crop. The model will lean towards the side you crop off of. ie the model steers away from the way the cam is yaw tilted
    # eg if cam tilted to the right car will try and correct to the left
    _,_,_, H, W = img.shape
    img = img[0,:, :, :, :-crop]
    #img = img[0,:, :, :, crop:]
    img = torchvision.transforms.Resize((H, W))(img)
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

