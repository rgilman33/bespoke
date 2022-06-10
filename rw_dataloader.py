
from constants import *
from imports import *
from input_prep import prep_inputs, gamma_correct_auto, pad
from train_utils import aug_imgs

import threading


class RealWorldDataloader():
    runs = [
                "run_286a", # new cam, abq, oscillaty same below
                "run_286b" # new cam, sabq
            ]

    path_stem = "real_world"

    def __init__(self):
        paths = [f"/media/beans/ssd/bespoke_logging/{run_id}" for run_id in self.runs]

        # memory here is simply a list of seqs, none of which have a batch dim. Not placing in normal symmetrical np array bc all seqs diff len
        # this is conceptually the same, just variable len seqs. Can still think of batch as first dim
        self.imgs_all_seqs, self.aux_all_seqs, self.targets_all_seqs = [], [], []

        for run_path in paths:
            MAX_LEN = 4000 if '176' in run_path else 7000

            car_state = np.stack([np.load(f) for f in sorted(glob.glob(f"{run_path}/car_state/*"))[:MAX_LEN]]) # (6469, 4)
            aux = np.stack([np.load(f) for f in sorted(glob.glob(f"{run_path}/aux/*"))[:MAX_LEN]])  # (6469, 4)
            imgs = np.stack([np.load(f) for f in sorted(glob.glob(f"{run_path}/img/*"))[:MAX_LEN]]) #(6469, 3, 80, 256)

            # Ensure all same length
            sl = min([len(car_state), len(aux), len(imgs)])
            car_state = car_state[:sl]; aux = aux[:sl]; imgs = imgs[:sl]

            #targets = (car_state[:,:1] / (STEER_RATIO * MAX_TIRE_ANGLE_DEG) * -1)
            targets = np.radians(car_state[:,:1] / (STEER_RATIO) * -1) # the tire angle in radians directly

            print(imgs.shape, aux.shape, targets.shape)

            self.imgs_all_seqs.append(imgs); self.aux_all_seqs.append(aux); self.targets_all_seqs.append(targets)

        self.i = 0

    def get_chunk(self):
        # returns an entire run at a time, batch padded in front
        i = self.i
        if i==len(self.imgs_all_seqs):
            self.i = 0
            return None
        img = pad(self.imgs_all_seqs[i]) # padding batch dim
        aux = pad(self.aux_all_seqs[i])
        targets = pad(self.targets_all_seqs[i])

        self.i += 1

        return img, aux, targets, '', #targets_traj, ''
