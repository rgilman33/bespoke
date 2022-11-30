from constants import *
from imports import *
from traj_utils import *
from input_prep import *
from viz_utils import *
import wandb

##########################################
### temporal consistency helpers

TIME_DELTA_PER_STEP = .05 # 1 / hz #TODO this should probably be in constants
 
def rotate_around_origin(x, y, angle):
    xx = x * np.cos(angle) + y * np.sin(angle)
    yy = -x * np.sin(angle) + y * np.cos(angle)

    return xx, yy

def temporal_consistency_loss(model_out, speeds_mps_all, tire_angles_rad_all, wheelbase=CRV_WHEELBASE):
    temporal_error = torch.FloatTensor([0]).to('cuda')
    bs = len(model_out)
    model_out_rad = model_out * TARGET_NORM.to('cuda') # now they're angles in rad
    for b in range(bs):
        trajs, speeds_mps, tire_angles_rad = model_out_rad[b], speeds_mps_all[b], tire_angles_rad_all[b]
        traj_xs, traj_ys = get_trajs_world_space(trajs, speeds_mps, tire_angles_rad, wheelbase)
        te = get_temporal_error(traj_xs, traj_ys, speeds_mps) 
        temporal_error += te.mean()
    temporal_error /= bs
    return temporal_error


def get_trajs_world_space(trajs, speeds_mps, tire_angles_rad, wheelbase):
    """ 
    Converts a trajectory of relative angles into trajectory of xy locations in absolute world space 
    """
    vehicle_heading, vehicle_location_x, vehicle_location_y = 0, 0, 0
    device = 'cuda'
    
    traj_wp_dists_torch = torch.HalfTensor(traj_wp_dists).to(device)
    
    traj_xs = torch.FloatTensor(len(trajs), len(traj_wp_dists)).to(device)
    traj_ys = torch.FloatTensor(len(trajs), len(traj_wp_dists)).to(device)
    
    for i in range(len(trajs)):

        current_speed_mps = speeds_mps[i]
        traj = trajs[i]

        ################
        # also at this instant, converting those local points to absolute space
        # this calculation is not technically correct. TODO
        # these are the payload of this fn
        xs = torch.sin(traj+vehicle_heading) * traj_wp_dists_torch + vehicle_location_x # abs world space
        ys = torch.cos(traj+vehicle_heading) * traj_wp_dists_torch + vehicle_location_y
        traj_xs[i] = xs
        traj_ys[i] = ys
        #################

        # this is an estimate of our current tire angle
        # Makes a big diff where we get this value. Using the real values, it matches better
        tire_angle = tire_angles_rad[i]
        vehicle_turn_rate = tire_angle * (current_speed_mps/wheelbase) # rad/sec
        vehicle_heading_delta = vehicle_turn_rate * TIME_DELTA_PER_STEP # radians
        # by the end of this step, our vehicle heading will have changed this much

        dist_car_will_travel_over_step = TIME_DELTA_PER_STEP * current_speed_mps # 20hz
        # by the end of this step, our vehicle will have travelled this far

        # simple linear way, not technically correct
        # /=2 bc that will be the avg angle during the turn
        #vehicle_delta_x = np.sin(vehicle_heading + (vehicle_heading_delta/2)) * dist_car_will_travel_over_step
        #vehicle_delta_y = np.cos(vehicle_heading + (vehicle_heading_delta/2)) * dist_car_will_travel_over_step

        # the technically correct way, though makes little difference
        # https://math.dartmouth.edu/~m8f19/lectures/m8f19curvature.pdf
        # TODO do these need to be in torch also, ie do we need to diff batck through these?
        if vehicle_heading_delta==0:
            vehicle_delta_x = 0
            vehicle_delta_y = dist_car_will_travel_over_step
        else:
            r = dist_car_will_travel_over_step / vehicle_heading_delta
            vehicle_delta_y = np.sin(vehicle_heading_delta)*r
            vehicle_delta_x = r - (np.cos(vehicle_heading_delta)*r)
        vehicle_delta_x, vehicle_delta_y = rotate_around_origin(vehicle_delta_x, vehicle_delta_y, vehicle_heading)   

        vehicle_heading += vehicle_heading_delta
        vehicle_location_x += vehicle_delta_x
        vehicle_location_y += vehicle_delta_y
    
    return traj_xs, traj_ys


def get_temporal_error(traj_xs, traj_ys, speeds_mps):
    """
    Takes in traj of xy points and calculates temporal inconsistency btwn them
    """
    
    # The initial positions
    t0x = traj_xs[:-1]
    t0y = traj_ys[:-1]
    
    speeds_mps = np.expand_dims(speeds_mps[:-1], -1)
    
    # the ending positions
    t1x = traj_xs[1:, :-1]
    t1y = traj_ys[1:, :-1]
    
    # the x and y distances btwn each wp and the following wp
    xd = t0x[:, 1:] - t0x[:, :-1]
    yd = t0y[:, 1:] - t0y[:, :-1]

    dist_travelled = torch.FloatTensor(speeds_mps * TIME_DELTA_PER_STEP).to('cuda')
    
    # the estimates at t1 should be dist_travelled along the traj estimated at t0
    # these are the 'targets', they are all locations along the original traj
    tx = t0x[:, :-1] + xd*dist_travelled # TODO this only works bc wps are one meter apart. Won't work above speeds of about 40mph bc then will travel more than 1m during a timestep
    ty = t0y[:, :-1] + yd*dist_travelled

    y_error = (t1y - ty)**2
    x_error = (t1x - tx)**2
    
    return (x_error + y_error).mean(axis=-1)


mse_loss = torch.nn.MSELoss().cuda()
_mse_loss_no_reduce = torch.nn.MSELoss(reduction='none').cuda()

def mse_loss_no_reduce(targets, preds, mask=None, weights=None):
    loss_no_reduce = _mse_loss_no_reduce(targets, preds)
    if mask is not None:
        loss_no_reduce *= mask # only asking to pred angles below certain threshold
    if weights is not None:
        loss_no_reduce *= weights # we care less about further wps, they're mostly used only for speed est at this pt

    return loss_no_reduce, loss_no_reduce.mean()

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


def torch_to_np_img(img):
    # takes in img as it was given to model. Denorms it, permutes back into np dims, puts on cpu and returns as np 0 to 255
    img = img.permute(1,2,0).cpu().numpy()
    img = denorm_img(img)
    img = (img*255).astype(np.uint8)
    return img

def add_trajs_to_img(img, pred, targets, speed_mps=None):
    """ Takes img in as it was directly as fed into model. Preds and targets same, right into or out of the model """
    img = torch_to_np_img(img)

    target_norm = TARGET_NORM[0,0,:].to('cuda') # squeezing what was unsqueezed
    trj = (pred * target_norm).detach().cpu().numpy()
    trj_targets = (targets * target_norm).detach().cpu().numpy()
    img = draw_wps(img, trj, speed_mps=speed_mps)
    img = draw_wps(img, trj_targets, color=(100, 200, 200), speed_mps=speed_mps)
    return img


LOSS_EPS = 1 - 1/200 

avg_control_loss = .01
avg_td_loss = 15
avg_torque_loss = 2000
avg_te_loss = .02

avg_approaching_stop_loss = .04
avg_stop_dist_loss = .005
avg_stopped_loss = .0002
avg_has_lead_loss = .04
avg_lead_dist_loss = .04

# we're saying we don't care as much about the further wps, regardless of angle
loss_weights = torch.from_numpy(pad(pad(np.concatenate([np.ones(20), np.linspace(1., .5, 10)]))).astype(np.float16)).to(device)

sigmoid = torch.nn.Sigmoid()

def get_worst(losses, has_three_dims=False):
    if has_three_dims: # control loss has wp dim to also collapse ie (bs, bptt, n_wps). Others are just (bs, bptt)
        losses,_ = losses.max(-1) # collapse the wp traj
    a, bptt_worst_ixs = losses.max(-1) # collapse the bptt traj
    worst_loss, batch_worst_ix = a.max(0)
    bptt_worst_ix = bptt_worst_ixs[batch_worst_ix]
    return worst_loss, batch_worst_ix, bptt_worst_ix
    
def run_epoch(dataloader, #TODO prob put this in own file, it's a big one
              model, 
              opt=None, 
              scaler=None, 
              train=True, 
              backwards=True, 
              log_wandb=True,
              log_cadence=128, 
              updates_per_epoch=5120,
              wandb=None):
    
    global avg_control_loss, avg_td_loss, avg_torque_loss, avg_te_loss, avg_approaching_stop_loss, avg_stop_dist_loss, avg_stopped_loss, avg_has_lead_loss, avg_lead_dist_loss
    #TODO awkward

    model.train(train)
    logger = Logger()
    train_pause = 0 #seconds
    model.reset_hidden(dataloader.bs)

    worst_control_loss_all, worst_approaching_stop_loss_all, worst_has_lead_loss_all = -np.inf, -np.inf, -np.inf
    worst_control_loss_all_img, worst_approaching_stop_img, worst_has_lead_img = None, None, None

    for update_counter in range(updates_per_epoch):
        t1 = time.time()
        batch = dataloader.get_batch()
        if not batch: break
        (img, aux_model, aux_calib, wp_angles, wp_headings, wp_curvatures, aux_targets, to_pred_mask), is_first_in_seq = batch

        if is_first_in_seq: 
            model.reset_hidden(dataloader.bs)

        with torch.cuda.amp.autocast(): 
            wps_preds, aux_preds, obs_net_out  = model(img, aux_model, aux_calib)

        wp_angles_pred, wp_headings_pred, wp_curvatures_pred = torch.chunk(wps_preds, 3, -1)
        
        control_loss_no_reduce, control_loss = mse_loss_no_reduce(wp_angles, wp_angles_pred, mask=to_pred_mask, weights=loss_weights)
        _, headings_loss = mse_loss_no_reduce(wp_headings, wp_headings_pred, mask=to_pred_mask, weights=loss_weights)
        _, curvatures_loss = mse_loss_no_reduce(wp_curvatures, wp_curvatures_pred, mask=to_pred_mask, weights=loss_weights)

        # aux targets
        aux_targets = torch.clip(aux_targets, -150, 150) #TODO just make this lower in datagen. Was getting overflow in halfs

        approaching_stop_loss_no_reduce, approaching_stop_loss = mse_loss_no_reduce(aux_targets[:,:,0], sigmoid(aux_preds[:,:,0]))
        stop_dist_loss_no_reduce, stop_dist_loss = mse_loss_no_reduce(aux_targets[:,:,1], aux_preds[:,:,1], mask=aux_targets[:,:,0]) # dist only care about when approaching stop
        
        stopped_loss = mse_loss(aux_targets[:,:,2], sigmoid(aux_preds[:,:,2]))

        has_lead_loss_no_reduce, has_lead_loss = mse_loss_no_reduce(aux_targets[:,:,3], sigmoid(aux_preds[:,:,3]))
        _, lead_dist_loss = mse_loss_no_reduce(aux_targets[:,:,4], aux_preds[:,:,4], mask=aux_targets[:,:,3])

        # calib loss
        pitch_loss = mse_loss(aux_calib[:,:,0], obs_net_out[:,:,0])
        yaw_loss = mse_loss(aux_calib[:,:,1], obs_net_out[:,:,1])

        # Vanilla steer cost
        steer_cost = mse_loss(wp_angles_pred[:,1:,:], wp_angles_pred[:,:-1,:])

        # Torque losses
        torque = get_torque(wp_angles_pred, aux_model[:,:,2:2+1]*20) #TODO this hardcoded denorm is dangerous)
        torque_loss = get_torque_loss(torque)
        torque_delta_loss = get_torque_delta_loss(torque)
        has_torque_loss = torque_loss.item() > 0
        has_torque_delta_loss = torque_delta_loss.item() > 0

        # keep our running avgs up to date
        avg_control_loss = avg_control_loss*LOSS_EPS + control_loss.item()*(1-LOSS_EPS)
        if has_torque_delta_loss:
            avg_td_loss = avg_td_loss*LOSS_EPS + torque_delta_loss.item()*(1-LOSS_EPS)
        if has_torque_loss:
            avg_torque_loss = avg_torque_loss*LOSS_EPS + torque_loss.item()*(1-LOSS_EPS)
        avg_approaching_stop_loss = avg_approaching_stop_loss*LOSS_EPS + approaching_stop_loss.item()*(1-LOSS_EPS)
        avg_stop_dist_loss = avg_stop_dist_loss*LOSS_EPS + stop_dist_loss.item()*(1-LOSS_EPS)
        avg_stopped_loss = avg_stopped_loss*LOSS_EPS + stopped_loss.item()*(1-LOSS_EPS)
        avg_has_lead_loss = avg_has_lead_loss*LOSS_EPS + has_lead_loss.item()*(1-LOSS_EPS)
        avg_lead_dist_loss = avg_lead_dist_loss*LOSS_EPS + lead_dist_loss.item()*(1-LOSS_EPS)

        # loss weights
        headings_loss /= 40
        curvatures_loss /= 2
        loss = control_loss + headings_loss + curvatures_loss + torque_delta_loss*(.03*avg_control_loss/avg_td_loss)

        loss += approaching_stop_loss*(.1*avg_control_loss/avg_approaching_stop_loss)
        loss += stop_dist_loss*(.1*avg_control_loss/avg_stop_dist_loss)
        loss += stopped_loss*(.03*avg_control_loss/avg_stopped_loss)
        loss += has_lead_loss*(.1*avg_control_loss/avg_has_lead_loss)
        loss += lead_dist_loss*(.1*avg_control_loss/avg_lead_dist_loss)

        loss += ((pitch_loss + yaw_loss)/300)

        if backwards:
            scaler.scale(loss).backward() 
            scaler.unscale_(opt)

            CLIP_GRAD_NORM = 10
            torch.nn.utils.clip_grad_norm_(model.fcs_1.parameters(), CLIP_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(model._rnn.parameters(), CLIP_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(model.fcs_2.parameters(), CLIP_GRAD_NORM)
            
            scaler.step(opt)
            scaler.update() 
            opt.zero_grad()
            time.sleep(train_pause)

        # Logging
        logger.log({f"lat_losses/{dataloader.path_stem}_control_loss": control_loss.item(),
                    f"lat_losses/{dataloader.path_stem}_headings_loss": headings_loss.item(),   
                    f"lat_losses/{dataloader.path_stem}_curvatures_loss": curvatures_loss.item(),   
                    f"lon_losses/{dataloader.path_stem}_approaching_stop_loss": approaching_stop_loss.item(),
                    f"lon_losses/{dataloader.path_stem}_stop_dist_loss": stop_dist_loss.item(),   
                    f"lon_losses/{dataloader.path_stem}_stopped_loss": stopped_loss.item(),  
                    f"lon_losses/{dataloader.path_stem}_has_lead_loss": has_lead_loss.item(),
                    f"lon_losses/{dataloader.path_stem}_lead_dist_loss": lead_dist_loss.item(),   
                    f"consistency losses/{dataloader.path_stem}_steer_cost":steer_cost.item(),
                    # # f"aux losses/{dataloader.path_stem}_uncertainty_loss":uncertainty_loss.item(),
                    f"aux losses/{dataloader.path_stem}_pitch_loss":pitch_loss.item(),
                    f"aux losses/{dataloader.path_stem}_yaw_loss":yaw_loss.item(),
                    # f"consistency losses/{dataloader.path_stem}_%_updates_w_torque_loss":1 if has_torque_loss else 0,
                    f"consistency losses/{dataloader.path_stem}_%_updates_w_torque_delta_loss":1 if has_torque_delta_loss else 0,
                   })
        if has_torque_loss:
            logger.log({f"consistency losses/{dataloader.path_stem}_torque_loss":torque_loss.item()})
        if has_torque_delta_loss: 
            logger.log({f"consistency losses/{dataloader.path_stem}_torque_delta_loss":torque_delta_loss.item()})

        # Obs w worst loss
        with torch.no_grad():
            # wp angles loss
            worst_control_loss, batch_worst_ix, bptt_worst_ix = get_worst(control_loss_no_reduce, has_three_dims=True) # has wp dim to also collapse
            if worst_control_loss > worst_control_loss_all:
                speed_mps = kph_to_mps(aux_model[batch_worst_ix, bptt_worst_ix, 2]*20) #TODO sloppy, this hardcoded denorm, and hardcoded speed ix
                worst_control_loss_all_img = add_trajs_to_img(img[batch_worst_ix, bptt_worst_ix, :, :, :], 
                                                                wp_angles_pred[batch_worst_ix, bptt_worst_ix, :], 
                                                                wp_angles[batch_worst_ix, bptt_worst_ix, :],
                                                                speed_mps=speed_mps)
            # stopsigns
            worst_approaching_stop_loss, batch_worst_ix, bptt_worst_ix = get_worst(approaching_stop_loss_no_reduce)
            if worst_approaching_stop_loss > worst_approaching_stop_loss_all:
                worst_approaching_stop_img = torch_to_np_img(img[batch_worst_ix, bptt_worst_ix, :,:,:])

            # Lead car
            worst_has_lead_loss, batch_worst_ix, bptt_worst_ix = get_worst(has_lead_loss_no_reduce)
            if worst_has_lead_loss > worst_has_lead_loss_all:
                worst_has_lead_img = torch_to_np_img(img[batch_worst_ix, bptt_worst_ix, :,:,:])

        # Periodic save and report
        if (update_counter+1)%log_cadence==0: 
            torch.save(model.state_dict(), f"{BESPOKE_ROOT}/models/m.torch")
            max_param = max([p.max() for p in model.parameters()]).item()
            logger.log({"logistical/max_param": max_param,
                        "logistical/lr":opt.param_groups[0]['lr']})
            stats = logger.finish(); print(stats)
            worst_control_loss_all, worst_approaching_stop_loss_all, worst_has_lead_loss_all = -np.inf, -np.inf, -np.inf
            if log_wandb: 
                # random imgs to log
                random_img_0 = add_trajs_to_img(img[0, 3, :,:,:], wp_angles_pred[0, 3, :], wp_angles[0, 3, :], speed_mps=kph_to_mps(aux_model[0,3,2]*20)) #TODO sloppy hardcoded denorm
                random_img_1 = add_trajs_to_img(img[1, -1, :,:,:], wp_angles_pred[1, -1, :], wp_angles[1, -1, :], speed_mps=kph_to_mps(aux_model[1,-1,2]*20))
                random_img_2 = add_trajs_to_img(img[-1, 0, :,:,:], wp_angles_pred[-1, 0, :], wp_angles[-1, 0, :], speed_mps=kph_to_mps(aux_model[-1,0,2]*20))

                three_randos = wandb.Image(np.concatenate([random_img_0, random_img_1, random_img_2], axis=0))
                worst_imgs = np.concatenate([worst_control_loss_all_img, worst_approaching_stop_img, worst_has_lead_img], axis=0)
                wandb.log(stats)
                wandb.log({"imgs/worst_losses": wandb.Image(worst_imgs),
                            "imgs/random":three_randos,
                        })

        # Timing
        t2 = time.time()
        n_obs_consumed = dataloader.bs * BPTT
        obs_consumed_per_second = n_obs_consumed / (t2 - t1)
        logger.log({"logistical/obs_consumed_per_second":round(obs_consumed_per_second)})
        t1 = t2
        
        # Manually keeping data consumption ratio down if training speed exceeds data gen speed
        obs_generated_per_second, slowest_runner_obs_per_sec = dataloader.get_obs_per_second()
        if obs_generated_per_second > 0:
            data_consumption_ratio = obs_consumed_per_second / obs_generated_per_second
            if data_consumption_ratio > DATA_CONSUMPTION_RATIO_LIMIT: 
                train_pause += .01
            elif data_consumption_ratio < DATA_CONSUMPTION_RATIO_LIMIT:
                train_pause -= .01
            train_pause = max(train_pause, 0)
            logger.log({"logistical/obs_generated_per_second":round(obs_generated_per_second),
                        "logistical/slowest_runner_obs_per_sec":round(slowest_runner_obs_per_sec),
                        "logistical/data_consumption_ratio":data_consumption_ratio,
                        "logistical/manual_train_pause":train_pause,
                       })
    
    # Final save and reporting
    stats = logger.finish(); print(stats)
    if log_wandb: wandb.log(stats)



##################
# Torque losses

def gather_ixs(preds, speeds_kph):
    bs, seqlen, _ = preds.shape
    ixs = torch.LongTensor(bs, seqlen, 1)
    for b in range(bs):
        for s in range(seqlen):
            traj = preds[b,s,:]
            speed = speeds_kph[b,s,0]
            _, _, wp_ix = get_target_wp(traj, kph_to_mps(speed)) # TODO to get the ix, shouldn't need traj at all
            ixs[b,s,0] = int(round(wp_ix))
    return ixs

MAX_ACCEPTABLE_TORQUE = 6000
MAX_ACCEPTABLE_TORQUE_DELTA = 1_000 #700
rad_to_deg = lambda x: x*57.2958

def get_torque(pred, speeds):
    speed_kph = speeds
    angles_deg = rad_to_deg(pred*TARGET_NORM.to('cuda'))
    ixs = gather_ixs(angles_deg, speed_kph.cpu().numpy()).to('cuda') # not backwards through angles_deg here, just getting ix based on speed
    applied_angles = torch.gather(angles_deg, -1, ixs)
    torque = applied_angles * (speed_kph**2)
    torque = torch.nan_to_num(torque, nan=0, posinf=MAX_ACCEPTABLE_TORQUE*1.1, neginf=-MAX_ACCEPTABLE_TORQUE*1.1) #TODO don't actually like this, as will result in wrong torque delta calcs. Fix it for real, torque should always be finite
    return torque

def get_torque_loss(torque):
    torque = abs(torque)
    unacceptable_mask = (torque > MAX_ACCEPTABLE_TORQUE)
    torque_loss = (torque * unacceptable_mask).mean()
    return torque_loss

def get_torque_delta_loss(torque):
    torque_delta = abs(torque[:,1:,:] - torque[:,:-1,:])
    unacceptable_mask = (torque_delta > MAX_ACCEPTABLE_TORQUE_DELTA)
    torque_delta_loss = (torque_delta * unacceptable_mask).mean()
    return torque_delta_loss



def _eval_rw(dataloader, model):
    
    model.eval()
    bptt = 128
    
    preds_all = []
    targets_all = []
    
    with torch.no_grad():
        while True:
            chunk = dataloader.get_chunk()
            if not chunk: break
            img_chunk, aux_chunk, targets_chunk = chunk
            bs, seq_len, _, _, _ = img_chunk.shape
            ix = 0
            model.reset_hidden(bs)
            
            preds_seq = []
            targets_seq = []

            while ix < (seq_len-bptt):
                img = img_chunk[:, ix:ix+bptt, :, :, :]
                aux = aux_chunk[:, ix:ix+bptt, :]
                targets = targets_chunk[:, ix:ix+bptt, :]
                img, aux, targets = prep_inputs(img, aux, targets=targets) #TODO we need to denorm these before reporting
                #img = side_crop(img, crop=8)
                ix += bptt

                with torch.cuda.amp.autocast(): 
                    pred, obsnet_out = model(img, aux)
                
                preds_at_wp_ix = gather_preds(pred[0].cpu().numpy(), aux[0,:,2].cpu().numpy())
                targets = targets[0,:,0].cpu().numpy()
                
                preds_seq.append(preds_at_wp_ix)
                targets_seq.append(targets)
            
            preds_seq = np.concatenate(preds_seq)
            targets_seq = np.concatenate(targets_seq)
            
            preds_all.append(preds_seq)
            targets_all.append(targets_seq)

    # preds_all *= TARGET_NORM # put into meaningful units, ie radians TODO
    # targets_all *= TARGET_NORM

    return preds_all, targets_all

import matplotlib.pyplot as plt

def eval_rw(rw_dataloader, m, wandb):
    ps, ts = _eval_rw(rw_dataloader, m)
    for i in range(len(ps)):
        p, t = ps[i], ts[i]
        run_id = rw_dataloader.runs[i]

        plt.figure(figsize=(20,5))
        plt.plot(t)
        plt.plot(p)
        plt.title(run_id)

        wandb.log({f"{run_id}":plt,
                    f"rw eval/mse {run_id}": mse(p, t),
                    f"rw eval/steer cost {run_id}": steer_cost_as_percent_of_target(p, t),
                   f"rw eval/bias {run_id}": absolute_avg_loss(p,t),
                   f"rw eval/pearsonr {run_id}": pearsonr(p, t)[0],
                  })
        plt.close()



def mse(p,t):
    return np.mean((t-p)**2)

def steer_cost_as_percent_of_target(p,t):
    pc = np.mean((p[1:] - p[:-1])**2)
    tc = np.mean((t[1:] - t[:-1])**2)
    return pc / tc

def absolute_avg_loss(p,t):
    return np.mean(t-p)


###########################################
# lives here so don't break OP. OP env doesn't have alb, bc it breaks the cv2 version or something? could debug easily, but no need

import albumentations as A

BRIGHTNESS_LIMIT = .2
BLUR_LIMIT = (1, 3) #(3,5)
ISO_NOISE_MAX = .4 #.8
GAUSS_NOISE_MAX = 200 # 400
COMPRESSION_QUALITY_MIN = 35 # 15
GAMMA_MAX = 130

transform = A.Compose([
    A.Blur(blur_limit=BLUR_LIMIT, p=.2),
    A.HueSaturationValue(hue_shift_limit=10,sat_shift_limit=70,val_shift_limit=(-20, 30)),
    A.OneOf([
        A.GaussNoise(var_limit=GAUSS_NOISE_MAX, p=.2),
        A.ISONoise(intensity=(.2, ISO_NOISE_MAX), p=.2)
    ]),
    A.Cutout(max_h_size=90, max_w_size=60, num_holes=2, p=.1),
    A.OneOf([
        A.ImageCompression(quality_lower=COMPRESSION_QUALITY_MIN, quality_upper=80, compression_type=0, p=.2),
        A.ImageCompression(quality_lower=COMPRESSION_QUALITY_MIN, quality_upper=80, compression_type=1, p=.2),
        A.JpegCompression(quality_lower=COMPRESSION_QUALITY_MIN, quality_upper=80, p=.2)
    ]),
    A.RandomBrightnessContrast(brightness_limit=BRIGHTNESS_LIMIT, contrast_limit=.4),
    A.RandomGamma(gamma_limit=(45,GAMMA_MAX), p=.25),
], additional_targets={f"image{i}":"image" for i in range(1,BPTT)})

def aug_imgs(img):
    bs, seqlen, _,_,_= img.shape # seqlen may be shorter than BPTT at the end of each seq
    AUG_SEQ_PROB = .8
    AUTO_GAMMA_PROB = 0 #.1
    SEQ_CONSTANT_AUG_PROB = .7 #.8
    for b in range(bs):
        # Sometimes don't aug at all
        if random.random() > AUG_SEQ_PROB: continue

        # Sometimes do the same auto gamma correct we do in rw
        auto_gamma = random.random() < AUTO_GAMMA_PROB
        if auto_gamma: 
            for s in range(seqlen): img[b][s] = gamma_correct_auto(img[b][s])
    
        # Aug
        if random.random() < SEQ_CONSTANT_AUG_PROB:
            # All frames in seq get same aug
            aug_input = {"image":img[b][0]}
            aug_input.update({f"image{i}":img[b][i] for i in range(1,seqlen)})
            augged_imgs = transform(**aug_input)
            img[b][0] = augged_imgs['image']
            for i in range(1,seqlen):
                img[b][i] = augged_imgs[f'image{i}']
        else:
            # different aug for each img in seq
            for s in range(seqlen):
                img[b][s] = transform(image=img[b][s])['image']

    return img
#################################################### 


def train_only_parts_of_model(m, part_to_train=""):
    for n, p in m.named_parameters():
        if part_to_train in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
def unfreeze_model(m):
    for p in m.parameters():
        p.requires_grad = True