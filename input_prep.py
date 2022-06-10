from constants import *
from imports import *

def norm_aux(aux):
    aux = aux / torch.from_numpy(aux_norm_constants).to('cuda')
    aux = aux.clip(-8,8)
    return aux

pad = lambda x: np.expand_dims(x,0)

IMG_NORM_MEAN = .5
IMG_NORM_STD = .25

# wps at ix 15 are about three times as big as ix 0. Actually measured the value off a dataloader, it was 2.56
# verified this for blender dataloader also. It was similar.
#wp_ix_norm = torch.from_numpy(np.linspace(1, 2.56, num=N_WPS_TO_USE).astype('float16')) # this was when pred 15 wps out
wp_ix_norm = torch.from_numpy(np.linspace(1, 4.27, num=N_WPS_TO_USE).astype('float16'))
wp_ix_norm = wp_ix_norm.unsqueeze(0).unsqueeze(0)#.to('cuda')
TARGET_NORM = .035 * wp_ix_norm #.02


def norm_img(img):
    return (img-IMG_NORM_MEAN) / IMG_NORM_STD

def denorm_img(img):
    return (img*IMG_NORM_STD) + IMG_NORM_MEAN

def prep_inputs(image, aux, targets=None, is_single_obs=False):
    """ np to pytorch totally prepped for model in """

    image = image.astype(np.float16)
    aux = aux.astype(np.float16)

    #assert image.max() > 1 # make sure these haven't already been normalized

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
        targets = torch.from_numpy(targets.astype(np.float16)).to('cuda')
        targets = targets[:,:,:N_WPS_TO_USE]
        targets = targets / TARGET_NORM.to('cuda') # RR maxes out at around 5%, so we'll make that one
        return (image, aux, targets)

    return (image, aux)

# ###############################################
# from torchvision import transforms

# # all take in torch.float16 already totally prepped for pytorch, returns the same

# def bwify(imgs):
#     return imgs.mean(2, keepdim=True)

# def norm_seqly(front_container):
#     """ Normalize by sequence within the batch so all seqs have approximate same brightness. 
#     Takes in b&w image from 0 to 1. Outputs the same, just scaled and shifted by mean and std.
#     Want to minimize any differences btwn test and train """
#     f = front_container.permute(1,0,2,3,4)
#     means = torch.flatten(f.float(),start_dim=1,end_dim=-1).mean(1)
#     stds = torch.flatten(f.float(),start_dim=1,end_dim=-1).std(1)
    
#     means = means.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#     stds = stds.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
#     front_container = front_container - means
#     front_container = front_container / stds
#     front_container = (front_container + 3.) / 6.
#     front_container = torch.clamp(front_container, 0., 1.)
    
#     return front_container

# def posterize(imgs):
#     imgs = (imgs*255).type(torch.uint8)
#     imgs = transforms.functional.posterize(imgs, 5)
#     imgs = imgs.type(torch.float16)/255.
#     return imgs

# def simplify_imgs(imgs):
#     imgs = bwify(imgs)
#     imgs = norm_seqly(imgs)
#     imgs = posterize(imgs)
    
#     return imgs
# #####################################################

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

