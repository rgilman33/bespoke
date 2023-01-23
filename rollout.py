from constants import *
from imports import *
from train_utils import *
from traj_utils import *
from models import *


class Rollout():
    def __init__(self, run, model_stem=None, m=None):

        self.model_stem = model_stem
        self.run_id = run.run_id
        self.is_rw = run.is_rw
        self.m = m
        self._get_rollout(run)

    def _get_rollout(self, run):
        
        # If we passed a model stem, load new model. Else use the model we passed in
        if self.m is None: 
            m = EffNet().to(device)
            m.load_state_dict(torch.load(f"{BESPOKE_ROOT}/models/m{self.model_stem}.torch"))
            # if self.do_actgrad: m.set_for_viz() # Has to be AFTER load state_dict
        else:
            m = self.m

        m.eval()
        m.save_backbone_out = True

        img, aux, wps = [], [], [] 
        final_acts, wps_p, obsnet_outs, aux_targets_p = [], [], [], []

        with torch.no_grad():
            while True:
                # get and unpack batch
                batch = run.get_batch()
                if not batch: break
                _img, _aux, _wps, extras = batch
                m.zero_grad()
                with torch.cuda.amp.autocast(): _wps_p, _aux_targets_p, _obsnet_out = m(_img, _aux)
                img.append(unprep_img(_img))
                aux.append(unprep_aux(_aux))
                wps.append(unprep_wps(_wps))
                wps_p.append(unprep_wps(_wps_p))
                obsnet_outs.append(unprep_obsnet(_obsnet_out))
                aux_targets_p.append(unprep_aux_targets(_aux_targets_p))
                final_acts.append(m.backbone_out_acts.detach().cpu().numpy())

        # Aux, inputs, targets
        self.img = np.concatenate(img, axis=1) #NOTE can't use the imgs from the run bc sometimes use TrnLoader, which will have updated chunks
        self.aux = na(np.concatenate(aux, axis=1), AUX_PROPS)
        self.wps = np.concatenate(wps, axis=1)

        # Model out
        self.wps_p = np.concatenate(wps_p, axis=1)
        self.final_acts = np.concatenate(final_acts, axis=1) # NOTE these will be a problem bc of threading
        self.obsnet_outs = na(np.concatenate(obsnet_outs, axis=1), OBSNET_PROPS)
        self.aux_targets_p = na(np.concatenate(aux_targets_p, axis=1), AUX_TARGET_PROPS)

        self.bs, self.seq_len, _ = self.aux.shape

        calc_rollout_results(self)
        flatten_rollout(self)

        print("Rollout complete!")

def skipify(rollout, skip=20):
    for a in ["img", "aux", "wps", "wps_p", "final_acts", "obsnet_outs", "aux_targets_p", "additional_results"]:
        setattr(rollout, a, getattr(rollout, a)[::skip])

def calc_rollout_results(rollout):
    # still in shape (bs, seqlen, n)
    # Do all calculations on rollout results here

    wp_angles_p, wp_headings_p, wp_curvatures_p, wp_rolls_p, wp_zs_p = np.split(rollout.wps_p, 5, -1)

    additional_results = na(np.zeros((rollout.bs, rollout.seq_len, len(ROLLOUT_PROPS))), ROLLOUT_PROPS)
    # Calculate some derivative properties #TODO not inside the rollout
    speeds = rollout.aux[:, :, 'speed']
    for b in range(rollout.bs):
        additional_results[b, :, "tire_angle_p"] = gather_preds(wp_angles_p[b], speeds[b])

    additional_results[:,:,'te'] = get_te(additional_results[:,:,'tire_angle_p'])
    speed_mask = get_speed_mask(speeds)
    additional_results[:,:,'traj_max_angle_p'] = np.abs((wp_angles_p * speed_mask)).max(axis=-1)

    # ccs 
    for b in range(rollout.bs):
        curve_constrained_speed_calculator = CurveConstrainedSpeedCalculator()
        for i in range(rollout.seq_len):
            ccs = curve_constrained_speed_calculator.step(wp_curvatures_p[b, i], speeds[b, i])
            additional_results[b,i,'ccs_p'] = ccs

    additional_results[:,:,'tire_angle_loss'] = abs(rollout.aux[:,:,"tire_angle"] - additional_results[:,:,"tire_angle_p"])

    # df['tire_angle_abs'] = df.tire_angle.abs().clip(0, .08)
    # df['unc_p_normed'] = (df.unc_p / (df.traj_max_angle.abs()+.001)).clip(-1000, 0)
    # df['te_log'] = np.clip(np.log(df.te), -25, 0)
    rollout.additional_results = additional_results


def flatten_batch_seq(a):
    bs, seqlen, o = a.shape[0], a.shape[1], a.shape[2:]
    new_shape = tuple(np.insert(o, 0, bs*seqlen))
    return np.reshape(a, new_shape)

def flatten_rollout(rollout):
    # flattens batch dimension, stacking batch items seqwise
    # All rollout results must be accounted for. That's one reason for keeping things in big containers until here

    # Inputs, targets, aux
    rollout.img = flatten_batch_seq(rollout.img)
    rollout.aux = flatten_batch_seq(rollout.aux)
    rollout.wps = flatten_batch_seq(rollout.wps)

    # Model out
    rollout.wps_p = flatten_batch_seq(rollout.wps_p); rollout.final_acts = flatten_batch_seq(rollout.final_acts)
    rollout.obsnet_outs = flatten_batch_seq(rollout.obsnet_outs); rollout.aux_targets_p = flatten_batch_seq(rollout.aux_targets_p)

    # Additional props
    rollout.additional_results = flatten_batch_seq(rollout.additional_results)

def df_from_rollout(rollout):
    # put into df to facilitate stacking of multiple rollouts, everything seqwise

    df = pd.DataFrame()
    # Ground truth aux
    for p in AUX_PROPS: df[p] = rollout.aux[:, p]
    for p in AUX_TARGET_PROPS: df[f"{p}_p"] = rollout.aux_targets_p[:, p]
    for p in ROLLOUT_PROPS: df[p] = rollout.additional_results[:, p]
    df["unc_p"] = rollout.obsnet_outs[:, "unc_p"]

    # rollout metadata
    df["run_id"] = rollout.run_id
    df["model_stem"] = rollout.model_stem
    df['is_rw'] = 1 if rollout.is_rw else 0

    return df

def get_te(tire_angle):
    # bs, seqlen
    te = np.zeros_like(tire_angle)
    te[:,1:] = abs(tire_angle[:, 1:] - tire_angle[:, :-1]) # first entry is a zero
    return te


import threading
def evaluate_run(run, m, save_rollouts, a):
    # run is a Run object, m is a model
    rollout = Rollout(run, model_stem=m.model_stem, m=m)
    if save_rollouts:
        save_object(rollout, f"{BESPOKE_ROOT}/tmp/{rollout.run_id}_{m.model_stem}_rollout.pkl")
    a.append(rollout)

import multiprocessing
class RwEvaluator():
    def __init__(self, run_ids, m, wandb=None, save_rollouts=False):
        run_paths = [f"{BESPOKE_ROOT}/tmp/runs/{run_id}.pkl" for run_id in run_ids] # loading from pickled Run object rather than from scratch
        self.runs = [load_object(p) for p in run_paths]
        self.wandb = wandb
        self.m = m
        self.save_rollouts = save_rollouts

    def evaluate(self):
        threads = [] # not using multiprocessing Process bc of complications with gpu
        a = []
        for i in range(len(self.runs)):
            run = self.runs[i]
            t = threading.Thread(target=evaluate_run, args=(run, self.m, self.save_rollouts, a)) # this takes 3.5 minutes for four runs, opposed to 6.5 w no threads
            t.start()
            threads.append(t)
        for t in threads:t.join()
        for r in self.runs:r.reset()
        self.m.save_backbone_out = False # return model to trn state
        self.m.train()
        print("down w rollouts, reporting")
        for rollout in a:
            print(rollout.run_id)
            te = rollout.additional_results[:,"te"].mean()
            tire_angle_loss = rollout.additional_results[:,"tire_angle_loss"].mean()
            unc_p = rollout.obsnet_outs[:,"unc_p"].mean()
            plt.figure(figsize=(20,2))
            plt.plot(rollout.aux[:,"tire_angle"])
            plt.plot(rollout.additional_results[:,"tire_angle_p"])
            self.wandb.log({
                f"rw/te_{rollout.run_id}": te,
                f"rw/tire_angle_loss_{rollout.run_id}": tire_angle_loss,
                f"rw/avg_unc_{rollout.run_id}": unc_p,
                f"rw_plots/tire_angle_{rollout.run_id}": plt,
            })
