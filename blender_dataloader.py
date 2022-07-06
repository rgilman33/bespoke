
from input_prep import prep_inputs, pad
from train_utils import aug_imgs
import threading
from constants import *
from imports import *

get_img_container = lambda bs : np.empty((bs, SEQ_LEN, IMG_HEIGHT, IMG_WIDTH, 3), dtype='uint8')
get_aux_container = lambda bs : np.empty((bs, SEQ_LEN, N_AUX), dtype=np.float32)
get_targets_container = lambda bs : np.empty((bs, SEQ_LEN, N_TARGETS), dtype=np.float32)

class BlenderDataloader():

    def __init__(self, path_stem, bs):
        
        self.img_chunk = get_img_container(bs)
        self.aux_chunk = get_aux_container(bs)
        self.targets_chunk = get_targets_container(bs)

        self.img_chunk_backup = get_img_container(bs)
        self.aux_chunk_backup = get_aux_container(bs)
        self.targets_chunk_backup = get_targets_container(bs)

        self.path_stem = path_stem # path stems: trn, val, real_world
        self.bs = bs
        self.seq_ix = 0

        self._refresh_chunk(self.img_chunk, self.aux_chunk, self.targets_chunk)
        self.refresh_backup_chunk()

        self.queued_batches = []
        self.retry_counter = 0
        self.queue_up_batch()

    def refresh_backup_chunk(self):
        self._refresh_chunk(self.img_chunk_backup, self.aux_chunk_backup, self.targets_chunk_backup)

    def _refresh_chunk(self, img_chunk, aux_chunk, targets_chunk):
        # sets them in place. Takes about 2 sec. Is this still true?

        _offset = random.randint(0, 10_000) # so we sample runners evenly
        for b in range(self.bs):

            #TODO keep an eye here. Won't likely run into targets file not exist, but may get the occasion where imgs begin to overwrite.
            # ie we grab a targets right before it's deleted, and imgs begin to be overwritten before we can grab them all. Also note this modulo
            # move is to keep sampling more even
            datagen_id = ("00"+str((b+_offset) % N_RUNNERS))[-2:] 
            ts = glob.glob(f"{BLENDER_MEMBANK_ROOT}/dataloader_{datagen_id}/run_{random.randint(0, RUNS_TO_STORE_PER_PROCESS-1)}/targets_*.npy")
            t = random.choice(ts)

            t_split = t.split('/')
            p = t_split[-1]
            t_stem = '/'.join(t_split[:-1])

            end_ix = int(p.split('.npy')[0].split('_')[1])
            
            targets = np.load(t)
            targets_chunk[b, :, :] = targets

            aux = np.load(f"{t_stem}/aux_{end_ix}.npy")
            aux_chunk[b, :, :] = aux

            img_paths = sorted(glob.glob(f"{t_stem}/imgs/*"))[end_ix-SEQ_LEN+1:end_ix+1] # imgs 1 indexed, targets zero indexed

            for i, p in enumerate(img_paths):
                img_chunk[b, i, :,:,:] = cv2.imread(p)[:,:,::-1] #bgr to rgb
        
        targets_chunk[:,:-1,:] = targets_chunk[:,1:,:] # moving targets forward by one bc of the hypths that we're misaligned
        aux_chunk[:,:-1,:] = aux_chunk[:,1:,:] #TODO should maybe do this further upstream actually

    def queue_up_batch(self):
        bptt = BPTT
        img_chunk, aux_chunk, targets_chunk = self.img_chunk, self.aux_chunk, self.targets_chunk
        _, seq_len, _, _, _ = img_chunk.shape
        ix = self.seq_ix
        is_first_in_seq = (ix==0)

        img = img_chunk[:, ix:ix+bptt, :, :, :].copy()
        
        aux = aux_chunk[:, ix:ix+bptt, :].copy()
        current_tire_angles_rad = aux[:,:,4].copy()
        current_speeds_mps = kph_to_mps(aux[:,:,2])
        pitch = aux[:,:,0].copy()
        yaw = aux[:,:,1].copy()
        aux[:,:,4] = 0.0
        aux[:,:,0] = 0.0
        aux[:,:,1] = 0.0
        aux[:,:,3] = 0.0 # temporary HACK
        
        pitch = torch.FloatTensor(pitch).to('cuda')
        current_tire_angles_rad = torch.FloatTensor(current_tire_angles_rad).to('cuda')
        current_speeds_mps = torch.FloatTensor(current_speeds_mps).to('cuda')
        yaw = torch.FloatTensor(yaw).to('cuda')

        targets = targets_chunk[:, ix:ix+bptt, :].copy()


        # mask out wps more than n seconds ahead
        MAX_PRED_S = 6.0
        avg_speeds_mps = kph_to_mps(aux[:,:,2:3].mean(axis=1, keepdims=True))
        max_pred_dists_m = avg_speeds_mps * MAX_PRED_S
        speed_mask = pad(pad(np.array(TRAJ_WP_DISTS, dtype=np.float16)))
        speed_mask = (speed_mask <= max_pred_dists_m).astype(np.float16) 
        # this will give us a shape of (bs, 1, 30), where 30 is the number of wps in our traj. The broadcasting above 'just works' bc thank you np

        MAX_ANGLE_TO_PRED = .18 #.16
        to_pred_mask = torch.from_numpy((np.abs(targets) < MAX_ANGLE_TO_PRED).astype(np.float16)).to(device)
        to_pred_mask = (to_pred_mask*.9) + .1 # 1.0 for all normal angles, .1 for all big angles

        ZERO_THRESH = .36 #.24
        zero_mask = torch.from_numpy((np.abs(targets) < ZERO_THRESH).astype(np.float16)).to(device)
        to_pred_mask = to_pred_mask*zero_mask # totally zero out above this threshold

        to_pred_mask = to_pred_mask * torch.from_numpy(speed_mask).to(device)

        img = aug_imgs(img) # aug when still as uint8. Ton of time spent here, way inefficient. Is that still true now?
        img, aux, targets = prep_inputs(img, aux, targets=targets) # we're actually spending substantial time here.

        self.seq_ix += bptt

        if self.seq_ix > seq_len-bptt: 
            self.seq_ix = 0

            self.img_chunk[:,:,:,:,:] = self.img_chunk_backup[:,:,:,:,:] 
            self.aux_chunk[:,:,:] = self.aux_chunk_backup[:,:,:]
            self.targets_chunk[:,:,:] = self.targets_chunk_backup[:,:,:]

            threading.Thread(target=self.refresh_backup_chunk).start() 


        self.queued_batches.append(((img, 
                                    aux, 
                                    targets,
                                    to_pred_mask,
                                    current_tire_angles_rad, # Extras
                                    current_speeds_mps, 
                                    pitch, 
                                    yaw), 
                                        is_first_in_seq))

    def get_batch(self):

        while len(self.queued_batches)==0 and self.retry_counter < 200:
            #print("Waiting for next chunk...")
            self.retry_counter += 1
            time.sleep(.1)

        thread = threading.Thread(target=self.queue_up_batch)
        thread.start()
        
        if len(self.queued_batches)==0:
            print("Finished dataloader!")
            return None
        else:
            self.retry_counter = 0
            return self.queued_batches.pop(0)

    def get_obs_per_second(self):
        return get_obs_per_sec()

