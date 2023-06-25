from constants import *
from norm import *
from imports import *
from traj_utils import *
from viz_utils import *
import wandb




mse_loss = torch.nn.MSELoss().cuda()
_mse_loss_no_reduce = torch.nn.MSELoss(reduction='none').cuda()

def mse_loss_no_reduce(targets, preds, weights=None, clip=None):
    loss_no_reduce = _mse_loss_no_reduce(targets, preds)
    if weights is not None:
        loss_no_reduce *= weights # we care less about further wps, they're mostly used only for speed est at this pt
    if clip is not None:
        loss_no_reduce = torch.clip(loss_no_reduce, min=clip[0], max=clip[1])
    return loss_no_reduce
    

class LossManager():
    """
    Decouples the loss magnitude and the amount of data in the dataset from the stimulus provided to the model.
    A loss weight of e.g. .2 means roughly that 20% of the gradient stim should come from this item
    """
    def __init__(self):
        self.baseline_loss_name = "wp_angles"
        self.loss_weights = { 
            "has_stop":.1, 
            "stop_dist":.1, 
            "has_lead":.1, 
            "lead_dist":.1,
            "lead_speed":.02, 
            "lane_width":.02,
            "dagger_shift":.05,
            "rd_is_lined":.2,
            "left_turn":.05,
            "right_turn":.05,
            "td":.03,
            "pitch":.1,
            "yaw":.1,
            "unc":.1,
            "te":.07,
        }
        self.loss_weights.update({   
                    "wp_angles": 1,
                    "wp_curvatures":.2,
                    "wp_headings":.02,
                    "wp_rolls":.2,
                    "wp_zs":.02,
                    })
        # for k,v in wps_losses.items():
        #     self.loss_weights[k] = v
        #     self.loss_weights[f"{k}_i"] = v * .5 #.1 # manual downweighting. Remember this is the proportion w respect to normal control loss, not the weight itself

        self.loss_emas = {k:10. for k in self.loss_weights.keys()} # init to be large so baseline loss will dominate
        self.loss_emas[self.baseline_loss_name] = 1. # init to be small to dominate first updates until ema stabilizes
        self.LOSS_EPS = 1. - 1./80.
        self.counter = 0
        self.update_emas = True

    def _step_loss(self, loss_name, loss):
        if not np.isfinite(loss.item()):
            print(f"loss is not finite ({loss}) for {loss_name}. Skipping")
            # return torch.HalfTensor([0]).to(device)
            return torch.FloatTensor([0]).to(device)

        # update avg
        if self.update_emas:
            self.loss_emas[loss_name] = self.loss_emas[loss_name]*self.LOSS_EPS + float(loss.item())*(1-self.LOSS_EPS) # full prec
        weight = self.loss_weights[loss_name] * (self.loss_emas[self.baseline_loss_name] / (self.loss_emas[loss_name]+1e-8))
        weighted_loss = weight*loss
        # print(loss_name, loss.item(), weighted_loss.item())
        return weighted_loss

    def step(self, losses_dict, logger=None):
        # total_loss = torch.HalfTensor([0]).to(device)
        total_loss = torch.FloatTensor([0]).to(device)
        for loss_name, loss_value in losses_dict.items():
            # if self.counter%200==0: print(loss_name, loss_value.item(), loss_value.dtype)
            weighted_loss = self._step_loss(loss_name, loss_value)
            total_loss += weighted_loss
            if logger: logger.log({loss_name:loss_value.item()/(LOSS_SCALER if "wp_" in loss_name else 1)})
        self.counter+=1
        return total_loss


def get_worst(losses):
    a, bptt_worst_ixs = losses.max(-1) # collapse the bptt traj
    worst_loss, batch_worst_ix = a.max(0)
    bptt_worst_ix = bptt_worst_ixs[batch_worst_ix]
    return worst_loss, batch_worst_ix, bptt_worst_ix
    

def get_and_enrich_img(b, b_ix, seq_ix):
    """
    Convenience wrapper for `enrich_img`. Takes in batch in pytorch, normed. Plucks at ixs, returns enriched, np single obs
    """ 
    img, wps, wps_p, aux, aux_targets_p, obsnet_out = b
    
    return enrich_img(img=unprep_img(img[b_ix, seq_ix, :3,:,:]), 
                        wps=unprep_wps(wps[b_ix, seq_ix,:]), 
                        wps_p=unprep_wps(wps_p[b_ix, seq_ix,:]), 
                        aux=unprep_aux(aux[b_ix, seq_ix,:]),
                        aux_targets_p=unprep_aux_targets(aux_targets_p[b_ix, seq_ix,:]),
                        obsnet_out=unprep_obsnet(obsnet_out[b_ix, seq_ix,:]))

LOSS_SCALER = 100

class Trainer():
    def __init__(self,
                dataloader,
                model, 
                model_stem=None,
                opt=None, 
                backwards=True, 
                log_wandb=True,
                log_cadence=512, updates_per_epoch=5120, total_epochs=3000,
                wandb=None,
                rw_evaluator=None,
                use_rnn=False,
                ):

        self.dataloader = dataloader; self.model = model; self.opt = opt; self.model_stem = model_stem; self.use_rnn = use_rnn
        self.backwards, self.log_wandb = backwards, log_wandb
        self.log_cadence, self.updates_per_epoch, self.wandb = log_cadence, updates_per_epoch, wandb
        self.wandb = wandb
        if self.backwards:
            self.scaler = torch.cuda.amp.GradScaler() 
        self.loss_manager = LossManager()
        set_trainer_should_stop(False)
        self.should_stop = False
        self.current_epoch = 0
        self.total_epochs = total_epochs
        self.rw_evaluator = rw_evaluator

        # we're saying we don't care as much about the further wps, regardless of angle
        # self.LOSS_WEIGHTS = torch.from_numpy(np.concatenate([np.ones(20), np.linspace(1., .5, 10)]).astype(np.float16)).to(device)[None,None,:]  
        self.LOSS_WEIGHTS = torch.from_numpy(np.concatenate([np.ones(20), np.linspace(1., .2, 10)])).to(device)[None,None,:]  

    def train(self):
        while self.current_epoch < self.total_epochs:
            self._run_epoch()
            if self.should_stop: 
                print("ending training early")
                break
            # if self.rw_evaluator is not None and (self.current_epoch+1) % 2 == 0: # every 1.5 hours currently TODO UNDO let back in
            #     self.model.model_stem = f"{self.model_stem}_e{self.current_epoch}"
            #     self.rw_evaluator.evaluate()
            self.current_epoch += 1
        
    def reset_worsts(self):
        self.worsts = {p:[-np.inf, None] for p in ['angles', "angles_i"]+SHOW_WORST_PROPS}

    def update_worst(self, loss_name, loss_values_no_reduce, batch):
        worst_loss, batch_worst_ix, seq_worst_ix = get_worst(loss_values_no_reduce)
        current_worst_loss = self.worsts[loss_name][0]
        if worst_loss > current_worst_loss:
            self.worsts[loss_name][1] = get_and_enrich_img(batch, batch_worst_ix, seq_worst_ix)
            self.worsts[loss_name][0] = worst_loss 

    def _run_epoch(self):
        model, dataloader = self.model, self.dataloader
        logger = Logger()
        train_pause = 0 #seconds
        model.reset_hidden(dataloader.bs)
        self.reset_worsts()

        f = lambda t : (~torch.isfinite(t)).any() # True if has infs or nans
        for update_counter in range(self.updates_per_epoch):
            t1 = time.time()
            timer = Timer("trn update")

            # get and unpack batch
            batch = dataloader.get_batch()
            if not batch: break
            timer.log("get batch from dataloader")

            if self.use_rnn:
                # zs, img, aux, wps, (to_pred_mask, is_first_in_seq) = batch # for Zs loader
                img, aux, wps, (to_pred_mask, is_first_in_seq) = batch
                if is_first_in_seq: model.reset_hidden(dataloader.bs)

                # with torch.cuda.amp.autocast(): model_out = model.rnn_head(zs)
                # zs = zs.float(); 
                #assert not f(zs), f"zs has nonfinite: {zs}"
                # if f(zs):
                #     print("zs has nonfinite") # this happens sometimes, was causing m outputs to have nonfinite. 
                #     continue

                with torch.cuda.amp.autocast(): z = model.cnn_features(img, aux) 
                # aux=aux.float(); wps=wps.float(); to_pred_mask=to_pred_mask.float() # full precision
                model_out = model.rnn_head(z.float()) # keeping lstm in full precision
            else:
                img, aux, wps, (to_pred_mask, is_first_in_seq) = batch  
                with torch.cuda.amp.autocast(): model_out= model.forward_cnn(img, aux)

            self.img_for_viewing = img # for inspection
            wps_p, aux_targets_p, obsnet_out = model_out
            #assert torch.isnan(aux).sum() == 0, f"aux has nans: {aux}"
            timer.log("model forward")

            # making full precision explicit from here on out
            # All losses calculated in full precision
            wps_p, aux_targets_p, obsnet_out = wps_p.float(), aux_targets_p.float(), obsnet_out.float() # full precision

            aux_targets = aux[:,:,AUX_TARGET_IXS] 

            #############
            # wp losses
            #############
            loss_weights = self.LOSS_WEIGHTS #.float() if self.use_rnn else self.LOSS_WEIGHTS
            weights = (to_pred_mask*loss_weights).repeat((1,1,5)).float() # why was this float64
            assert torch.isnan(wps).sum() == 0, f"wps has nans: {wps}"

            if f(wps_p):
                print("wps_p has nonfinite", wps_p)
                # self.should_stop = True
                # break
                continue
            if f(aux_targets_p):
                print("aux_targets_p has nans")
                self.should_stop = True
                break
            if f(obsnet_out).sum():
                print("obsnet_out has nans")
                self.should_stop = True
                break
            # if state_dict_has_nans(model.state_dict()): #NOTE this takes 50ms, so shouldn't do so often
            #     print("model has nans")
            #     self.img_for_viewing = img # for inspection
            #     self.nans_counter += 1
            #     self.should_stop = True
            #     break
            ego_in_intx = aux[:,:,AUX_PROPS.index("ego_in_intx")].bool()

            _wps_loss = mse_loss_no_reduce(wps, wps_p, weights=weights) * LOSS_SCALER #TODO can prob get rid of scaling now that all is full float
            angles_loss, headings_loss, curvatures_loss, rolls_loss, zs_loss = torch.chunk(_wps_loss, 5, -1)
            _mm = 400
            cm = lambda t : torch.clamp(t, -_mm, _mm).mean()
            losses = {
                "wp_angles": cm(angles_loss),
                "wp_headings": cm(headings_loss),
                "wp_curvatures": cm(curvatures_loss),
                "wp_rolls":cm(rolls_loss),
                "wp_zs":cm(zs_loss), 
            }
            intx_angles_loss_perc_total = angles_loss[ego_in_intx.unsqueeze(-1).expand(-1,-1,N_WPS)].sum() / angles_loss.sum()
            logger.log({"logistical/intx_angles_loss_perc_total":intx_angles_loss_perc_total.item()})
            logger.log({"logistical/intx_perc_total":ego_in_intx.float().mean().item()})

            # # te. When in doubt, stay close to prev preds
            # _, bptt, _ = wps_p.shape
            # if bptt > 1:
            #     te_loss = mse_loss_no_reduce(wps_p[:, :-1, :].detach(), wps_p[:, 1:, :], weights=weights[:, 1:, :]) 
            #     te_loss *= 100
            #     losses.update({"te": te_loss.mean()})

            #############
            # aux losses
            #############
                        
            aux_targets_p[:,:,AUX_TARGET_SIGMOID_IXS] = aux_targets_p[:,:,AUX_TARGET_SIGMOID_IXS].sigmoid()
            aux_targets_losses = mse_loss_no_reduce(aux_targets, aux_targets_p) 
            aux_targets_losses *= 100 #TODO can prob get rid of scaling now that all is full float
            #######################
            # Zero out a buffer right at end of horizon, no need to be strict about exact horizon end
            aux_np = unprep_aux(aux)
            has_stop, has_lead = aux_targets[:,:,AUX_TARGET_PROPS.index("has_stop")], aux_targets[:,:,AUX_TARGET_PROPS.index("has_lead")]

            # Leads
            lead_dist = aux_np[:,:,"lead_dist"]
            leads_always_enforce = torch.from_numpy((lead_dist<LEAD_DIST_MIN) | (lead_dist>LEAD_DIST_MAX)).to(device)
            m_sees_lead = (aux_targets_p[:,:,AUX_TARGET_PROPS.index("has_lead")].detach() > .3)
            m_sees_true_lead = has_lead.bool() & m_sees_lead # already sigmoided
            leads_enforce = leads_always_enforce | m_sees_true_lead
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("has_lead")] *= leads_enforce
            
            # Stops
            stop_dist = aux_np[:,:,"stop_dist"]
            stops_min_buffer = -100 if self.use_rnn else 8
            stops_always_enforce = torch.from_numpy(((stops_min_buffer<stop_dist) & (stop_dist<STOP_DIST_MIN)) | (stop_dist>STOP_DIST_MAX)).to(device)
            m_sees_stop = (aux_targets_p[:,:,AUX_TARGET_PROPS.index("has_stop")].detach() > .3)
            m_sees_true_stop = has_stop.bool() & m_sees_stop # already sigmoided
            stops_enforce = stops_always_enforce | m_sees_true_stop
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("has_stop")] *= stops_enforce

            # Rd is lined NOTE can't do this when just starting trn TODO put back in UNDO
            lenient_is_lined = False
            if lenient_is_lined:
                m_sees_rd_is_lined = (aux_targets_p[:,:,AUX_TARGET_PROPS.index("rd_is_lined")].detach() > .3)
                rd_is_lined = aux_targets[:,:,AUX_TARGET_PROPS.index("rd_is_lined")]
                rd_is_lined_enforce = (rd_is_lined.bool() & m_sees_rd_is_lined) | ~rd_is_lined.bool()
                aux_targets_losses[:,:,AUX_TARGET_PROPS.index("rd_is_lined")] *= rd_is_lined_enforce

            # Only enforce stops and lead metrics when we have stops and leads
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("stop_dist")] *= m_sees_true_stop #has_stop
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("lead_dist")] *= m_sees_true_lead
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("lead_speed")] *= m_sees_true_lead

            # when cnn only, don't enforce stops when up close, don't enforce some losses in intx
            if not self.use_rnn: 
                aux_targets_losses[:,:,AUX_TARGET_PROPS.index("rd_is_lined")] *= ~ego_in_intx
                aux_targets_losses[:,:,AUX_TARGET_PROPS.index("lane_width")] *= ~ego_in_intx
            #######################

            losses_to_include = AUX_TARGET_PROPS if self.use_rnn else [p for p in AUX_TARGET_PROPS if p not in ["lead_speed"]]
            losses.update({p:aux_targets_losses[:,:,AUX_TARGET_PROPS.index(p)].mean() for p in losses_to_include})

            #############
            # Obsnet losses
            #############
            # pitch_loss = mse_loss(aux[:,:,AUX_PROPS.index('pitch')], obsnet_out[:,:,OBSNET_PROPS.index('pitch')])
            # yaw_loss = mse_loss(aux[:,:,AUX_PROPS.index('yaw')], obsnet_out[:,:,OBSNET_PROPS.index('yaw')])
            unc_p = obsnet_out[:,:,OBSNET_PROPS.index('unc_p')]
            unc_loss = mse_loss(torch.log(angles_loss.mean(dim=-1).detach()), unc_p)
            losses.update({
                #"pitch":pitch_loss,
                #"yaw":yaw_loss,
                "unc":unc_loss,
            })
            logger.log({f"avg_unc": unc_p.mean().item()})

            loss = self.loss_manager.step(losses, logger=logger)
            timer.log("calc losses")
            # if is_first_in_seq:
            #     print("\n\n is first in seq")
            # print("Total loss:", loss.item(), loss.dtype)

            opt = self.opt
            if self.backwards:
                # if self.use_rnn: # rnn, full precision no amp or scaler
                # loss.backward() 
                # torch.nn.utils.clip_grad_value_(model.parameters(), 2.0) 
                # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 60.0) 
                # opt.step()
                # opt.zero_grad()

                # else: # cnn
                scaler = self.scaler
                scaler.scale(loss).backward() 
                nonfinite_sum, none_sum, finite_sum = gradients_with_nonfinite(model)
                if nonfinite_sum > 0: print("got nonfinite grad")
                logger.log({"logistical/nonfinite grads": nonfinite_sum})
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_value_(model.parameters(), 2.0) 
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 60.0)
                scaler.step(opt)
                scaler.update() 
                opt.zero_grad()

                logger.log({"logistical/grad_norm": total_norm.item()})
                logger.log({"logistical/amp scale": scaler.get_scale()})
                time.sleep(train_pause)
            timer.log("backwards")
        
            # Obs w worst loss
            CHECK_FOR_WORST_FREQ = 10 # takes a long time, so not each update
            if update_counter%CHECK_FOR_WORST_FREQ==0 and self.log_wandb and img is not None:
                with torch.no_grad():
                    # collapse the wp traj for any traj related losses
                    b = (img, wps, wps_p, aux, aux_targets_p, obsnet_out)

                    angles_loss,_ = angles_loss.max(-1)
                    angles_loss_no_intx = angles_loss * ~ego_in_intx
                    self.update_worst("angles", angles_loss_no_intx, b)

                    angles_loss_intx = angles_loss * ego_in_intx
                    self.update_worst("angles_i", angles_loss_intx, b)

                    # Report worsts for our aux targets
                    for p in SHOW_WORST_PROPS:
                        loss_no_reduce = aux_targets_losses[:,:,AUX_TARGET_PROPS.index(p)]
                        self.update_worst(p, loss_no_reduce, b)
            timer.log("get worst")
            # Periodic save and report
            if (update_counter+1)%self.log_cadence==0: 
                self.save_state()                
                max_param = max([p.max() for p in model.parameters()]).item()
                
                snrs = get_snr(opt) if opt is not None else np.zeros(3)
                logger.log({"logistical/max_param": max_param,
                            "logistical/lr":opt.param_groups[0]['lr'] if opt is not None else 0,
                            "logistical/mins_since_slowest_runner_reported":get_mins_since_slowest_runner_reported(), # this is the true number, should be using this for gen obs_per_sec
                            "logistical/snr":snrs.mean(),
                            })
                stats = logger.finish()
                dataloader_stats = dataloader.report_logs()
                print("\n\n",stats); print("\n",dataloader_stats)
                if self.log_wandb: 
                    # random imgs to log
                    batch = (img, wps, wps_p, aux, aux_targets_p, obsnet_out)
                    if img is not None: 
                        for k,v in self.worsts.items(): self.wandb.log({f"imgs/worst_{k}": wandb.Image(v[1])})
                    if img is not None: self.wandb.log({"imgs/random":wandb.Image(get_and_enrich_img(batch, b_ix=0, seq_ix=0))})
                    self.wandb.log(stats)
                    self.wandb.log(dataloader_stats)

                    plt.scatter(range(len(snrs)), snrs)
                    plt.title("snrs by layer")
                    self.wandb.log({"logistical_charts/snr_by_layer":plt})
                self.reset_worsts()
                
            timer.log("logging")

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

            timer.log("calc timing")
            logger.log(timer.finish())

            if get_trainer_should_stop():
                self.should_stop =  True
                break

        # During normal training, save each epoch. Models will be used for ensembling
        if self.backwards: 
            self.save_state()
            torch.save(self.model.state_dict(), f"{BESPOKE_ROOT}/models/m{self.model_stem}_e{self.current_epoch}.torch")

        # Final save and reporting
        stats = logger.finish(); print(stats)
        if self.log_wandb: self.wandb.log(stats)

    def save_state(self):
        torch.save(self.model.state_dict(), f"{BESPOKE_ROOT}/models/m.torch")
        if self.opt is not None: torch.save(self.opt.state_dict(), f"{BESPOKE_ROOT}/models/opt.torch")
        trainer_state = {
            "current_epoch":self.current_epoch,
            "loss_manager_emas":self.loss_manager.loss_emas,
        }
        save_object(trainer_state, f"{BESPOKE_ROOT}/tmp/trainer_state.pkl")
    
    def reload_state(self, reload_model=True):
        self.should_stop = False
        set_trainer_should_stop(False)
        if reload_model:
            m_path = f"{BESPOKE_ROOT}/models/m.torch"
            self.model.load_state_dict(torch.load(m_path))
            print(f"Loading model from {m_path}. \nLast modified {round((time.time() - os.path.getmtime(m_path)) / 60)} min ago.")
            if self.opt is not None: 
                opt_path = f"{BESPOKE_ROOT}/models/opt.torch"
                print(f"Loading opt from {opt_path}. \nLast modified {round((time.time() - os.path.getmtime(opt_path)) / 60)} min ago.")
                self.opt.load_state_dict(torch.load(opt_path))
            
        # reload trainer state. Manually keep these updated.
        trainer_state = load_object(f"{BESPOKE_ROOT}/tmp/trainer_state.pkl")
        # for k,v in trainer_state.items():setattr(self, k, v)
        self.current_epoch = trainer_state["current_epoch"]
        self.loss_manager.loss_emas = trainer_state["loss_manager_emas"]

        print(f"Currently at epoch {self.current_epoch}.")


def get_snr(opt):
    sd = opt.state_dict()
    mms, vms, snrs = [], [], []
    for k in sd['state'].keys():
        m = sd['state'][k]["exp_avg"]    
        v = sd['state'][k]["exp_avg_sq"]
        m = m.abs()
        v = v.sqrt()+1e-8
        snr = (m/v)
        snrs.append(snr.mean().item())
        # mms.append(m.mean().item())
        # vms.append(v.mean().item())
    snrs = np.array(snrs)
    # mms = np.array(mms)
    # vms = np.array(vms)
    return snrs #, mms.mean(), vms.mean()

def gradients_with_nonfinite(model):
    nonfinite_sum, none_sum, finite_sum = 0, 0, 0
    for p in model.parameters():
        if p.grad is None:
            none_sum += 1
        elif not torch.isfinite(p.grad).all():
            nonfinite_sum += 1
        else:
            finite_sum += 1
    return nonfinite_sum, none_sum, finite_sum

def state_dict_has_nans(state_dict):
    for k,v in state_dict.items():
        if torch.isnan(v).sum()>0:
            print(k, "has nans")
            return True
    return False

class ClipActivationHook(): # Doesn't do anything, just prints out big activations. Modified from gpt.
    def __init__(self, min_val=-2**15, max_val=2**15-1):
        self.min_val = min_val
        self.max_val = max_val
        
    def __call__(self, module, _input):
        #if isinstance(module, torch.nn.modules.activation.Activation):
        #_input[0].clamp_(self.min_val, self.max_val)
        _max = _input[0].flatten().max().item()
        if _max > 2**15: print("\n BIG ACTIVATION", module.__class__.__name__, _max)

def clip_activations(model, min_val=-2**15, max_val=2**15-1):
    clip_hook = ClipActivationHook(min_val, max_val)
    for module in model.modules():
        handle = module.register_forward_pre_hook(clip_hook)
    return model

###########################################
# lives here so don't break OP. OP env doesn't have alb, bc it breaks the cv2 version or something? could debug easily, but no need

import albumentations as A


random_color = lambda : (random.randint(0,255), random.randint(0,255), random.randint(0,255))
def get_transform(seqlen, diff="easy"):
    # spatter is strongest when cutout is at or below mean. Best is within .1 of mean.
    spatter_mean = random.uniform(.1, .4)
    spatter_cutout = spatter_mean + random.uniform(.06, .12)
    d = .3
    COMPRESSION_QUALITY_MIN = 30
    cutout_prob = .01 if diff=="easy" else .1 # right now just used for cnn
    cutout_max_n_holes = 10 if diff=="easy" else 40

    transforms = [
        A.AdvancedBlur(p=d, blur_limit=(3, 3), sigmaX_limit=(0.1, 1.55), sigmaY_limit=(0.1, 1.51), rotate_limit=(-81, 81), beta_limit=(0.5, 8.0), noise_limit=(0.01, 22.05)),
        A.HueSaturationValue(hue_shift_limit=10,sat_shift_limit=70,val_shift_limit=0, p=d), # not doing brightness, doing it below
        A.OneOf([ # Noise
            A.GaussNoise(var_limit=250, p=d),
            A.ISONoise(intensity=(.2, .5), p=d),
        ]),
        A.OneOf([ # distractors
            A.CoarseDropout(p=cutout_prob, max_holes=cutout_max_n_holes, max_height=120, max_width=120, min_holes=3, min_height=10, min_width=10, fill_value=random_color(), mask_fill_value=None),
            A.Spatter(p=.1, mean=spatter_mean, std=(0.25, 0.35), gauss_sigma=(.8, 1.6), intensity=(-.3, 0.3), cutout_threshold=spatter_cutout, mode=['rain', 'mud']),
            A.PixelDropout(p=.1, dropout_prob=random.uniform(.01, .05), per_channel=random.choice([0,1]), drop_value=random_color(), mask_drop_value=None),
        ]),
        A.OneOf([ # compression artefacting
            A.ImageCompression(quality_lower=COMPRESSION_QUALITY_MIN, quality_upper=80, compression_type=0, p=d),
            A.ImageCompression(quality_lower=COMPRESSION_QUALITY_MIN, quality_upper=80, compression_type=1, p=d),
            A.JpegCompression(quality_lower=COMPRESSION_QUALITY_MIN, quality_upper=80, p=d)
        ]),
        A.OneOf([ # brightness
            A.RandomGamma(gamma_limit=(50,150), p=d), # higher is darker
            A.RandomBrightnessContrast(p=d, brightness_limit=(-0.2, 0.22), contrast_limit=(-.4, .4)),
        ]),
        A.OneOf([  # other 
            # A.Sharpen(p=.1, alpha=(0.2, 0.5), lightness=(0.5, 1.0)), # rw imgs are blurrier, never sharper
            A.CLAHE(p=.05, clip_limit=(1, 4), tile_grid_size=(8, 8)),
            A.Emboss(p=.05, alpha=(0.2, 0.5), strength=(0.2, 0.7)),
        ])
    ]
    random.shuffle(transforms)
    transform = A.Compose(transforms, additional_targets={f"image{i}":"image" for i in range(seqlen)})
    return transform

def aug_imgs(img, constant_seq_aug, diff="easy"):
    seqlen, _,_,_= img.shape # seqlen may be shorter than BPTT at the end of each seq
    transform = get_transform(seqlen, diff=diff)

    if random.random() < constant_seq_aug: #TODO only works w len 2, fix it. How to pass in param names programatically? wtf shitty api
        transformed = transform(image=img[1], image0=img[0])
        img[1, :,:,:] = transformed['image']
        img[0, :,:,:] = transformed['image0']
    else:
        # different aug for each img in seq
        for s in range(seqlen):
            AUG_PROB = .95 #1.0 # to speed up a bit.
            if random.random() < AUG_PROB:
                img[s] = transform(image=img[s])['image']

    return img
#################################################### 

def unfreeze_part_of_model(m, part_to_unfreeze):
    for n, p in m.named_parameters():
        if part_to_unfreeze in n:
            p.requires_grad = True
            
def freeze_model(m, should_freeze):
    for p in m.parameters():
        p.requires_grad = not should_freeze
