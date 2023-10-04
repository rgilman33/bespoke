from constants import *
from norm import *
from imports import *
from traj_utils import *
from viz_utils import *
from utils import *

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
    
_mae_loss_no_reduce = nn.L1Loss(reduction='none').cuda()
def mae_loss_no_reduce(targets, preds, weights=None):
    loss_no_reduce = _mae_loss_no_reduce(targets, preds)
    if weights is not None:
        loss_no_reduce *= weights
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
            "semseg":10., #.1,
            "depth":10., #.1,
            "bev":10.,
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
    img, wps, wps_p, bev, bev_p, perspective, perspective_p, aux, aux_targets_p, obsnet_out = b
    
    return enrich_img(img=unprep_img(img[b_ix, seq_ix, :3,:,:]), 
                        wps=unprep_wps(wps[b_ix, seq_ix,:]), 
                        wps_p=unprep_wps(wps_p[b_ix, seq_ix,:]), 
                        aux=unprep_aux(aux[b_ix, seq_ix,:]),
                        aux_targets_p=unprep_aux_targets(aux_targets_p[b_ix, seq_ix,:]),
                        obsnet_out=unprep_obsnet(obsnet_out[b_ix, seq_ix,:]),
                        bev=unprep_bev(bev[b_ix, seq_ix,:,:,:]),
                        bev_p=unprep_bev(bev_p[b_ix, seq_ix,:,:,:]),
                        perspective=unprep_perspective(perspective[b_ix, seq_ix,:,:,:]),
                        perspective_p=unprep_perspective(perspective_p[b_ix, seq_ix,:,:,:]),
                        )

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
        median_tracker = MedianTracker()
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
            img, aux, wps, bev, perspective, (to_pred_mask, is_first_in_seq) = batch

            if self.use_rnn:
                if is_first_in_seq: model.reset_hidden(dataloader.bs)
                with torch.cuda.amp.autocast(): 
                    z = model.cnn_features(img, aux) 
                model_out = model.rnn_head(z.float()) # keeping lstm in full precision
            else:
                with torch.cuda.amp.autocast(): 
                    z, bev_p, perspective_p = model.cnn_features(img, aux, return_blocks_out=True)
                    model_out= model.cnn_head(z)

            self.img_for_viewing = img # for inspection
            wps_p, aux_targets_p, obsnet_out = model_out
            #assert torch.isnan(aux).sum() == 0, f"aux has nans: {aux}"
            timer.log("model forward")

            # making full precision explicit from here on out
            # All losses calculated in full precision
            wps_p, aux_targets_p, obsnet_out = wps_p.float(), aux_targets_p.float(), obsnet_out.float() # full precision

            aux_targets_p[:,:,AUX_TARGET_SIGMOID_IXS] = aux_targets_p[:,:,AUX_TARGET_SIGMOID_IXS].sigmoid()

            bev_p = bev_p.float(); bev = bev.float()
            bev_p = bev_p.sigmoid()
            perspective_p = perspective_p.float(); perspective = perspective.float() # all losses in full precision
            perspective_p[:,:, :3,:,:] = perspective_p[:,:, :3,:,:].sigmoid() # sigmoid in place

            aux_targets = aux[:,:,AUX_TARGET_IXS] 

            # helper fn to save bad imgs
            def save_if_necessary(obswise_criterion, tag):
                # bool of shape bs
                if obswise_criterion.sum().item()>0:
                    batch = (img[obswise_criterion], wps[obswise_criterion], wps_p[obswise_criterion], 
                            bev[obswise_criterion], bev_p[obswise_criterion],
                            perspective[obswise_criterion], perspective_p[obswise_criterion],
                            aux[obswise_criterion], aux_targets_p[obswise_criterion], obsnet_out[obswise_criterion])
                    _img = get_and_enrich_img(batch, b_ix=0, seq_ix=0) # just taking first, if more than one
                    plt.imsave(f"{BESPOKE_ROOT}/tmp/bad_imgs/{tag}_e{self.current_epoch}u{update_counter}.png", _img)

            #############
            # Nan checks
            #############
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

            losses = {}

            #############
            # aux losses 
            #############      
            ego_in_intx = aux[:,:,AUX_PROPS.index("ego_in_intx")].bool()

            aux_targets_losses = mse_loss_no_reduce(aux_targets, aux_targets_p) 
            aux_targets_losses *= 100 #TODO can prob get rid of scaling now that all is full float            
            
            # is-lined
            m_sees_rd_is_lined = (aux_targets_p[:,:,AUX_TARGET_PROPS.index("rd_is_lined")].detach() > .3) # threshold chosen manually
            rd_is_lined = aux_targets[:,:,AUX_TARGET_PROPS.index("rd_is_lined")]
            # rd_is_lined_enforce = (rd_is_lined.bool() & m_sees_rd_is_lined) | ~rd_is_lined.bool()

            m_doesnt_see_lines = (rd_is_lined.bool() & ~m_sees_rd_is_lined & ~ego_in_intx)[:,0] # Not removing when intx, bc not enforcing anyways. bs, just taking first of bptt NOTE bptt? 
            save_if_necessary(m_doesnt_see_lines, f"lines_")
            logger.log({"logistical/maskout_bc_lines_perc":m_doesnt_see_lines.float().mean().item()})

            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("rd_is_lined")][~rd_is_lined.bool()] *= 4 # upweighting the loss when we're not lined, make is-lined detector less senstive

            ##################
            # wp losses
            ##################
            to_pred_mask = (to_pred_mask * self.LOSS_WEIGHTS).float() # why was this float64?
            weights = (to_pred_mask).repeat((1,1,5)).float() # why was this float64?
            _wps_loss = mae_loss_no_reduce(wps, wps_p, weights=weights) * LOSS_SCALER #TODO can prob get rid of scaling now that all is full float
            angles_loss, headings_loss, curvatures_loss, rolls_loss, zs_loss = torch.chunk(_wps_loss, 5, -1)
            
            # collapse wps dim and bptt dim. Now have single loss for each batch item
            # Our unit of analysis here is the batch item. Collapsing bptt is a nontrivial thing to do, but i like the simplicity of 
            # dealing w batch items only when it comes to weighting and eliminating observations

            # weighted-mean followed by mean over bptt dim
            sm = lambda l : l.mean(-1) #l.sum(-1) / to_pred_mask.sum(-1) # smart mean, not using zeroed out values at end of traj

            # using for unc_p. Should use for everything.
            angles_loss_sm = angles_loss.sum(-1) / to_pred_mask.sum(-1) # bs, bptt

            angles_loss = sm(angles_loss).mean(-1)
            headings_loss = sm(headings_loss).mean(-1); curvatures_loss = sm(curvatures_loss).mean(-1)
            rolls_loss = sm(rolls_loss).mean(-1); zs_loss = sm(zs_loss).mean(-1)

            median_tracker.step({"wp_angles":angles_loss,
                                "wp_headings":headings_loss,
                                "wp_curvatures":curvatures_loss,
                                "wp_rolls":rolls_loss,
                                "wp_zs":zs_loss,
                                })
            median_emas = median_tracker.median_emas

            CLAMP_MEDIAN_MULTIPLE, MASKOUT_MEDIAN_MULTIPLE = 2, 30
            angles_loss_exceeds_thresh = (angles_loss > median_emas["wp_angles"]*MASKOUT_MEDIAN_MULTIPLE)
            if angles_loss_exceeds_thresh.sum() > 0:
                bad_img_angles_loss_med = round(angles_loss[angles_loss_exceeds_thresh][0].item() / median_emas["wp_angles"]) # store for use below
                save_if_necessary(angles_loss_exceeds_thresh, f"angles_m{bad_img_angles_loss_med}_")

            # Maskout the worst, not using their loss at all, discarding the data. Using wp_angles to maskout all wp losses
            # The assumption is this is useless, harmful data, bc our procgen isn't perfect. 

            # log maskout percentage, ensure we're not discarding too much
            logger.log({"logistical/maskout_perc":angles_loss_exceeds_thresh.float().mean().item()})

            # also masking out if there are lines but m doesn't see them. Lines affects traj loc, and our data is often unfair on this point.
            maskout = angles_loss_exceeds_thresh | m_doesnt_see_lines
            maskout = maskout.detach() # for safety

            if maskout.sum() > 0:
                # print("masking out obs")
                angles_loss[maskout] = 0; headings_loss[maskout] = 0; curvatures_loss[maskout] = 0; rolls_loss[maskout] = 0; zs_loss[maskout] = 0

            # Clamping all losses to multiple of their respective medians 
            # Each item in the batch will never account for more than clamped proportion of loss. 
            # Focusing attn on fine-grained details rather than coarse outliers.
            c = lambda l, ln : torch.clamp(l, max=median_emas[ln]*CLAMP_MEDIAN_MULTIPLE)

            # Log the amount loss is reduced by the clamping, for wp_angles only. Diagnostic
            angles_loss_clamped = c(angles_loss, "wp_angles")
            _a = angles_loss.mean().item()
            logger.log({"logistical/angles_loss_clamp_reduction":(_a - angles_loss_clamped.mean().item())/_a})
            logger.log({"logistical/angles_loss_clamp_perc_obs":(angles_loss > median_emas["wp_angles"]*CLAMP_MEDIAN_MULTIPLE).float().mean().item()})

            losses.update({
                "wp_angles": angles_loss_clamped.mean(),
                "wp_headings": c(headings_loss, "wp_headings").mean(),
                "wp_curvatures": c(curvatures_loss, "wp_curvatures").mean(),
                # "wp_rolls":c(rolls_loss, "wp_rolls").mean(), # roll usually zero, so median is often zero, so don't median-clamp
                "wp_rolls":rolls_loss.mean(),
                "wp_zs":c(zs_loss, "wp_zs").mean(), 
            })

            # Logging the proportion of loss that is coming from intx, rd_is_lined, as proportion of total loss
            intx_angles_loss_perc = angles_loss_clamped[ego_in_intx[:,0]].sum().item() / angles_loss_clamped.sum().item() # taking first bptt item of ego_in_intx
            logger.log({"logistical/intx_angles_loss_perc":intx_angles_loss_perc})
            logger.log({"logistical/intx_perc":ego_in_intx[:,0].float().mean().item()})

            rd_is_lined = aux[:,:,AUX_PROPS.index("rd_is_lined")].bool()
            rd_is_lined_angles_loss_perc = angles_loss_clamped[rd_is_lined[:,0]].sum().item() / angles_loss_clamped.sum().item() # taking first bptt item 
            logger.log({"logistical/rd_is_lined_loss_perc":rd_is_lined_angles_loss_perc})
            logger.log({"logistical/rd_is_lined_perc":rd_is_lined[:,0].float().mean().item()})


            # # te. When in doubt, stay close to prev preds
            # _, bptt, _ = wps_p.shape
            # if bptt > 1:
            #     te_loss = mse_loss_no_reduce(wps_p[:, :-1, :].detach(), wps_p[:, 1:, :], weights=weights[:, 1:, :]) 
            #     te_loss *= 100
            #     losses.update({"te": te_loss.mean()})
            
            # #############
            # # bev
            # #############
            # bev_loss = mae_loss_no_reduce(bev_p, bev)

            # npc_mask = is_npc(bev).expand(-1,-1, 3,-1,-1)
            # npc_mask_p = is_npc(bev_p).expand(-1,-1, 3,-1,-1).detach()

            # rd_markings_mask = is_rd_markings(bev).expand(-1,-1, 3,-1,-1)
            # rd_markings_mask_p = is_rd_markings(bev_p).expand(-1,-1, 3,-1,-1).detach()

            # edge_mask = is_edge(bev).expand(-1,-1, 3,-1,-1)

            # bev_loss[maskout] = 0
            # q = torch.quantile(bev_loss, .85).item()

            # rd_markings_false_pos_loss = bev_loss[rd_markings_mask_p & ~rd_markings_mask].mean()
            # npc_false_pos_loss = bev_loss[npc_mask_p & ~npc_mask].mean()

            # bev_loss_final = bev_loss[bev_loss>q].mean() + bev_loss[npc_mask].mean()*10 + bev_loss[rd_markings_mask].mean()*10 + bev_loss[edge_mask].mean()
            # fp_loss = rd_markings_false_pos_loss + npc_false_pos_loss
            # if fp_loss.item()>0: bev_loss_final += fp_loss
            # losses.update({"bev":bev_loss_final})

            #############
            # Perspective semseg and depth loss
            #############

            seg = perspective[:,:, :3,:,:]
            depth = perspective[:,:, 3:,:,:]
            seg_p = perspective_p[:,:, :3,:,:]
            depth_p = perspective_p[:,:, 3:,:,:]

            faraway_maskout = (depth[:,:, 0:1,:,:]<2).expand(-1,-1, 3,-1,-1).detach() #TODO FIX this hardcoded nonsense 

            semseg_loss = mae_loss_no_reduce(seg_p, seg) 

            # median, for watching only right now
            semseg_loss_by_obs = semseg_loss.mean((-1,-2,-3,-4)) # collapse all except batch
            median_tracker.step({"semseg":semseg_loss_by_obs}) # mean for each obs
            semseg_loss_exceeds_thresh = semseg_loss_by_obs > median_tracker.median_emas["semseg"]*6
            if semseg_loss_exceeds_thresh.sum().item()>0:
                bad_semseg_loss_med = round(semseg_loss_by_obs[semseg_loss_exceeds_thresh][0].item()/median_tracker.median_emas["semseg"])
                save_if_necessary(semseg_loss_exceeds_thresh, f"semseg_m{bad_semseg_loss_med}_")

            npc_mask = is_npc(seg).expand(-1,-1, 3,-1,-1)
            npc_mask_p = is_npc(seg_p).expand(-1,-1, 3,-1,-1).detach()

            rd_markings_mask = is_rd_markings(seg).expand(-1,-1, 3,-1,-1)
            rd_markings_mask_p = is_rd_markings(seg_p).expand(-1,-1, 3,-1,-1).detach()

            stops_mask = is_stopsign(seg).expand(-1,-1, 3,-1,-1)
            stops_mask_p = is_stopsign(seg_p).expand(-1,-1, 3,-1,-1).detach()

            semseg_loss[maskout] = 0 # not fair to do these when masked out
            semseg_loss[faraway_maskout] = 0
            
            q = torch.quantile(semseg_loss[:,:, :,::2,::2], .85).item() # downsampling bc quantile can't take so big
            semseg_loss_normal = semseg_loss[semseg_loss>q].mean()
            loss_npc = semseg_loss[npc_mask].mean()
            npc_false_pos = semseg_loss[npc_mask_p & ~npc_mask].mean()
            loss_rd_markings = semseg_loss[rd_markings_mask].mean()
            rd_markings_false_pos_loss = semseg_loss[rd_markings_mask_p & ~rd_markings_mask].mean()
            loss_stops = semseg_loss[stops_mask].mean()
            loss_stops_fp = semseg_loss[stops_mask_p & ~stops_mask].mean()

            semseg_loss = semseg_loss_normal + loss_npc*3  + loss_rd_markings*3 + loss_stops*3
            fp_loss = npc_false_pos + rd_markings_false_pos_loss + loss_stops_fp
            if fp_loss.item()>0: semseg_loss += fp_loss
            losses.update({"semseg":semseg_loss})

            depth_loss = mae_loss_no_reduce(depth_p, depth) 
            _depth_loss_mask = depth_loss_mask(depth).expand(-1,-1, 3,-1,-1) 
            depth_loss[maskout] = 0
            depth_loss[faraway_maskout] = 0
            depth_loss[~_depth_loss_mask] = 0
            qd = torch.quantile(depth_loss[:,:, :,::2,::2], .75).item()
            depth_loss_base = depth_loss[depth_loss>qd].mean()
            depth_loss = depth_loss_base
            losses.update({"depth":depth_loss})

            if update_counter %10==0: ##obs_has_npcs.sum() > 0:
                # print(f"semseg loss for {obs_has_npcs.sum()} obs", semseg_loss.item())

                batch = (img[:], wps[:], wps_p[:], 
                        bev[:], bev_p[:], perspective[:], perspective_p[:],
                        aux[:], aux_targets_p[:], obsnet_out[:])
                _img = get_and_enrich_img(batch, b_ix=0, seq_ix=0) # just taking first, if more than one
                plt.imsave(f"{BESPOKE_ROOT}/tmp/npcs_imgs/e{self.current_epoch}u{update_counter}.png", _img)


            #######################
            # Aux losses: Stops and leads
            aux_np = unprep_aux(aux)
            has_stop, has_lead = aux_targets[:,:,AUX_TARGET_PROPS.index("has_stop")], aux_targets[:,:,AUX_TARGET_PROPS.index("has_lead")]

            # Leads
            lead_dist = aux_np[:,:,"lead_dist"]
            leads_always_enforce = torch.from_numpy((lead_dist<LEAD_DIST_MIN) | (lead_dist>LEAD_DIST_MAX)).to(device)
            m_sees_lead = (aux_targets_p[:,:,AUX_TARGET_PROPS.index("has_lead")].detach() > .3)
            m_sees_true_lead = has_lead.bool() & m_sees_lead # already sigmoided
            leads_enforce = leads_always_enforce | m_sees_true_lead
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("has_lead")] *= leads_enforce
            leads_buffer = torch.from_numpy(((LEAD_DIST_MAX-20)<lead_dist) & (lead_dist<LEAD_DIST_MAX)).to(device)
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("has_lead")][leads_buffer] = 0 # deadzone, don't trn either direction
            
            ######################
            # Stops
            ######################
            stop_dist = aux_np[:,:,"stop_dist"]
            stops_min_buffer = -100 if self.use_rnn else 15
            stops_always_enforce_pos = torch.from_numpy((stops_min_buffer<stop_dist) & (stop_dist<STOP_DIST_MIN)) & has_stop.bool().cpu()
            stops_always_enforce_neg = torch.from_numpy(stop_dist>STOP_DIST_MAX)
            stops_always_enforce = (stops_always_enforce_pos | stops_always_enforce_neg).to(device)
            m_sees_stop = (aux_targets_p[:,:,AUX_TARGET_PROPS.index("has_stop")].detach() > .3)
            m_sees_true_stop = has_stop.bool() & m_sees_stop # already sigmoided
            stops_enforce = stops_always_enforce | m_sees_true_stop
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("has_stop")] *= stops_enforce
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("has_stop")][m_sees_true_stop] *= 2 # upping has_stop weight to make detector more sensitive
            
            # buffer zone, don't enforce up or down
            stops_buffer_end = torch.from_numpy(((STOP_DIST_MAX-20)<stop_dist) & (stop_dist<STOP_DIST_MAX))
            stops_buffer_beginning = torch.from_numpy(stop_dist<2) 
            stops_buffer = (stops_buffer_beginning | stops_buffer_end).to(device)
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("has_stop")][stops_buffer] = 0 # deadzone, don't trn either direction
            
            # logging the stops that we're forcing model to learn even when m doesn't see them
            m_doesnt_see_stop = (stops_always_enforce_pos.to(device) & ~m_sees_stop)[:,0]
            save_if_necessary(m_doesnt_see_stop, f"stop_forced_")

            # logging the stops that we're being lenient on but m maybe should actually see
            stops_in_lenient_zone = torch.from_numpy((STOP_DIST_MIN<stop_dist) & (stop_dist<40)).to(device)
            m_doesnt_see_stop_lenient = (stops_in_lenient_zone & ~m_sees_stop)[:,0]
            save_if_necessary(m_doesnt_see_stop_lenient, f"stop_lenient_")

            # Only enforce stops and lead metrics when we have stops and leads
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("stop_dist")] *= m_sees_true_stop #has_stop
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("lead_dist")] *= m_sees_true_lead
            aux_targets_losses[:,:,AUX_TARGET_PROPS.index("lead_speed")] *= m_sees_true_lead

            # when cnn only, don't enforce stops when up close, don't enforce some losses in intx
            if not self.use_rnn: 
                aux_targets_losses[:,:,AUX_TARGET_PROPS.index("rd_is_lined")] *= ~ego_in_intx
                aux_targets_losses[:,:,AUX_TARGET_PROPS.index("lane_width")] *= ~ego_in_intx
                aux_targets_losses[:,:,AUX_TARGET_PROPS.index("dagger_shift")] *= ((~ego_in_intx).float()*.95 + .05)
            #######################

            # masking out all losses for the obs we're discarding above based on angle. Across entire bptt.
            aux_targets_losses[maskout] = 0

            losses_to_include = AUX_TARGET_PROPS if self.use_rnn else [p for p in AUX_TARGET_PROPS if p not in ["lead_speed"]]
            losses.update({p:aux_targets_losses[:,:,AUX_TARGET_PROPS.index(p)].mean() for p in losses_to_include})

            #############
            # Obsnet losses
            #############
            # pitch_loss = mse_loss(aux[:,:,AUX_PROPS.index('pitch')], obsnet_out[:,:,OBSNET_PROPS.index('pitch')])
            # yaw_loss = mse_loss(aux[:,:,AUX_PROPS.index('yaw')], obsnet_out[:,:,OBSNET_PROPS.index('yaw')])
            unc_p = obsnet_out[:,:,OBSNET_PROPS.index('unc_p')]
            angles_loss_sm = torch.clamp(angles_loss_sm, max=median_emas["wp_angles"]*10) # giving a bit more room for pred unc. Also these units aren't the same bc median taken w dumb median
            unc_loss = mse_loss(angles_loss_sm.detach(), unc_p) 
            losses.update({
                #"pitch":pitch_loss,
                #"yaw":yaw_loss,
                "unc":unc_loss,
            })
            logger.log({f"avg_unc": unc_p.mean().item()})

            loss = self.loss_manager.step(losses, logger=logger)

            #loss = semseg_loss 
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

                    b = (img, wps, wps_p, bev, bev_p, perspective, perspective_p, aux, aux_targets_p, obsnet_out)

                    # using the masked-out but not clamped losses for viewing wps losses. We've collapsed along bptt for wps losses above
                    angles_loss = angles_loss[:,None] # padding the bptt dim bc that's what fn expects. Collapsed bptt above w new apparatus. Can update this fn later
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
                # if nans, stop. Not doing this every update bc takes 50ms. This should allow to resume from prev save
                # wout nans in state dict
                if state_dict_has_nans(model.state_dict()):
                    print("Nans in model state dict. Ending training")
                    self.should_stop = True
                    break
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
                    if img is not None: 
                        for k,v in self.worsts.items(): self.wandb.log({f"imgs/worst_{k}": wandb.Image(v[1])})
                    batch = (img, wps, wps_p, bev, bev_p, perspective, perspective_p, aux, aux_targets_p, obsnet_out)
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
        # don't save anything if should_stop flag has been thrown, might be bc nans or something
        if self.backwards and not self.should_stop: 
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
    cutout_prob = .005 if diff=="easy" else .1 # right now just used for cnn
    spatter_prob = .03 if diff=="easy" else .1
    pixdrop_prob = .03 if diff=="easy" else .1
    cutout_max_n_holes = 5 if diff=="easy" else 40

    cutout_color = (0,0,0) if random.random()<.8 else random_color() # this is mostly to mimic maskout in viz suite at this point
    transforms = [
        A.AdvancedBlur(p=d, blur_limit=(3, 3), sigmaX_limit=(0.1, 1.55), sigmaY_limit=(0.1, 1.51), rotate_limit=(-81, 81), beta_limit=(0.5, 8.0), noise_limit=(0.01, 22.05)),
        A.HueSaturationValue(hue_shift_limit=10,sat_shift_limit=70,val_shift_limit=0, p=d), # not doing brightness, doing it below
        A.OneOf([ # Noise
            A.GaussNoise(var_limit=250, p=d),
            A.ISONoise(intensity=(.2, .5), p=d),
        ]),
        A.OneOf([ # distractors
            A.CoarseDropout(p=cutout_prob, max_holes=cutout_max_n_holes, max_height=120, max_width=120, min_holes=3, min_height=10, min_width=10, fill_value=cutout_color, mask_fill_value=None),
            A.Spatter(p=spatter_prob, mean=spatter_mean, std=(0.25, 0.35), gauss_sigma=(.8, 1.6), intensity=(-.3, 0.3), cutout_threshold=spatter_cutout, mode=['rain', 'mud']),
            A.PixelDropout(p=pixdrop_prob, dropout_prob=random.uniform(.01, .05), per_channel=random.choice([0,1]), drop_value=random_color(), mask_drop_value=None),
        ]),
        A.OneOf([ # compression artefacting
            A.ImageCompression(quality_lower=COMPRESSION_QUALITY_MIN, quality_upper=80, compression_type=0, p=d),
            A.ImageCompression(quality_lower=COMPRESSION_QUALITY_MIN, quality_upper=80, compression_type=1, p=d),
            A.JpegCompression(quality_lower=COMPRESSION_QUALITY_MIN, quality_upper=80, p=d)
        ]),
        A.OneOf([ # brightness
            A.RandomGamma(gamma_limit=(60,135), p=d), # higher is darker
            A.RandomBrightnessContrast(p=d, brightness_limit=(-0.18, 0.2), contrast_limit=(-.3, .3)),
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
