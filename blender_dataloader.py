
from input_prep import prep_inputs, pad, prep_targets
from train_utils import aug_imgs
import threading
from constants import *
from imports import *
from traj_utils import *

get_img_container = lambda bs : np.empty((bs, SEQ_LEN, IMG_HEIGHT, IMG_WIDTH, 3), dtype='uint8')
get_img_bw_container = lambda bs : np.empty((bs, SEQ_LEN, IMG_HEIGHT, IMG_WIDTH, 1), dtype='uint8')
get_maps_container = lambda bs : np.empty((bs, SEQ_LEN, MAP_HEIGHT, MAP_WIDTH, 3), dtype='uint8')
get_aux_container = lambda bs : np.empty((bs, SEQ_LEN, N_AUX_TO_SAVE), dtype=np.float32)
get_targets_container = lambda bs : np.empty((bs, SEQ_LEN, N_WPS*4), dtype=np.float32)
get_episode_info_container = lambda bs : np.empty((bs, N_EPISODE_INFO), dtype=np.float32)


class BlenderDataloader():

    def __init__(self, path_stem, bs):
        
        self.img_chunk = get_img_container(bs)
        self.img_bw_chunk = get_img_bw_container(bs)
        self.maps_chunk = get_maps_container(bs)
        self.aux_chunk = get_aux_container(bs)
        self.targets_chunk = get_targets_container(bs)
        self.episode_info = get_episode_info_container(bs)

        self.img_chunk_backup = get_img_container(bs)
        self.img_bw_chunk_backup = get_img_bw_container(bs)
        self.maps_chunk_backup = get_maps_container(bs)
        self.aux_chunk_backup = get_aux_container(bs)
        self.targets_chunk_backup = get_targets_container(bs)
        self.episode_info_backup = get_episode_info_container(bs)

        self.logger = Logger()
        self.path_stem = path_stem # path stems: trn, val, real_world
        self.bs = bs
        self.seq_ix = SEQ_START_IX

        self.backup_chunk_is_ready = False
        threading.Thread(target=self.refresh_backup_chunk).start() 
        self._refresh_chunk(self.img_chunk, self.img_bw_chunk, self.maps_chunk, self.aux_chunk, self.targets_chunk, self.episode_info)
        #self.refresh_backup_chunk()

        self.queued_batches = []
        self.retry_counter = 0
        self.queue_up_batch()

    def refresh_backup_chunk(self):
        t1 = time.time()
        self.backup_chunk_is_ready = False
        self._refresh_chunk(self.img_chunk_backup, self.img_bw_chunk_backup, self.maps_chunk_backup, \
                                self.aux_chunk_backup, self.targets_chunk_backup, self.episode_info_backup)
        self.backup_chunk_is_ready = True
        self.logger.log({"timing/refresh_chunk":time.time()-t1})

    def _refresh_chunk(self, img_chunk, img_bw_chunk, maps_chunk, aux_chunk, targets_chunk, episode_info):
        # sets them in place. Takes about 2 sec per item
        _offset = random.randint(0, 10_000) # so we sample runners evenly
        for b in range(self.bs):
            timer = Timer("get_batch_item")
            #TODO keep an eye here. Won't likely run into targets file not exist, but may get the occasion where imgs begin to overwrite.
            # ie we grab a targets right before it's deleted, and imgs begin to be overwritten before we can grab them all. Also note this modulo
            # move is to keep sampling more even
            datagen_id = ("00"+str((b+_offset) % N_RUNNERS))[-2:] 
            ts = []
            while len(ts)==0: # So we can start sooner than when all runs fully populated
                ts = glob.glob(f"{BLENDER_MEMBANK_ROOT}/dataloader_{datagen_id}/run_{random.randint(0, RUNS_TO_STORE_PER_PROCESS-1)}/targets_*.npy")
            t = random.choice(ts)

            t_split = t.split('/')
            p = t_split[-1]
            t_stem = '/'.join(t_split[:-1])

            end_ix = int(p.split('.npy')[0].split('_')[1])
            timer.log("get paths")

            targets = np.load(t)
            targets_chunk[b, :, :] = targets

            aux = np.load(f"{t_stem}/aux_{end_ix}.npy")
            aux_chunk[b, :, :] = aux

            maps = np.load(f"{t_stem}/maps_{end_ix}.npy")
            maps_chunk[b, :, :,:,:] = maps

            timer.log("load targets, aux, maps")

            img_paths = sorted(glob.glob(f"{t_stem}/imgs/*"))[end_ix-SEQ_LEN+1:end_ix+1] # imgs 1 indexed, targets zero indexed

            for i, p in enumerate(img_paths):
                img_chunk[b, i, :,:,:] = cv2.imread(p)[:,:,::-1] #bgr to rgb

            timer.log("load imgs")

            # # Moving this here to move time from batch load to chunk load
            # moving it doubled the chunk time from 30s to 60s. Note that doing two chunks simultaneously doesn't slow them down
            _img = aug_imgs(img_chunk[b:b+1,:, :,:,:])[0] # padding the batch dimension 
            timer.log("aug imgs")
            img_chunk[b,:, :,:,:] = _img
            img_bw_chunk[b,:, :,:,:] = bwify_seq(_img) # faster than mean? appears so, 60-70s vs 90s. Didn't test a lot.
            timer.log("bw imgs")

            _episode_info = np.load(f"{t_stem}/episode_info.npy")

            just_go_straight = bool(_episode_info[0])
            HAS_MAPS_PROB = .8 if just_go_straight else 1.0
            img_chunk[b,:, :,-MAP_WIDTH:,:] = maps if random.random()<HAS_MAPS_PROB else 0

            episode_info[b,:] = _episode_info
            timer.log("add maps")
            self.logger.log(timer.finish())

        targets_chunk[:,:-1,:] = targets_chunk[:,1:,:] # moving targets forward by one bc of they're off by one
        aux_chunk[:,:-1,:] = aux_chunk[:,1:,:] #TODO should maybe do this further upstream actually
        # maps_chunk[:,:-1,:,:,:] = maps_chunk[:, 1:,:,:,:] This lines them up, without it we're actually off by one lagged

    def queue_up_batch(self):
        timer = Timer("queue_batch")

        bptt = BPTT
        img_chunk, maps_chunk, aux_chunk, targets_chunk, episode_info = self.img_chunk, self.maps_chunk, self.aux_chunk, self.targets_chunk, self.episode_info
        img_bw_chunk = self.img_bw_chunk

        _, seq_len, _, _, _ = img_chunk.shape
        ix = self.seq_ix
        is_first_in_seq = (ix==SEQ_START_IX)

        ##############
        img_subchunk = img_chunk[:, ix-LOOKBEHIND*3:ix+bptt, :,:,:] # same ixs, these two
        img_subchunk_bw = img_bw_chunk[:, ix-LOOKBEHIND*3:ix+bptt, :,:,:]
        img = cat_imgs(img_subchunk, img_subchunk_bw)

        timer.log("cat hist")

        ###################
        
        ######### TODO this can be moved to get_chunk apparatus for extra performance in get_batch
        # Aux
        aux_model, aux_calib, aux_targets = get_auxs(aux_chunk[:, ix:ix+bptt, :])

        targets = targets_chunk[:, ix:ix+bptt, :].copy()
        wp_angles, wp_dists, wp_rolls, wp_zs = np.split(targets, 4, axis=2)
        
        wp_angles_smoothed = smooth_near_wps_batch(wp_angles)

        wp_headings = get_headings_from_traj_batch(wp_angles_smoothed, wp_dists)
        wp_headings = smooth_near_wps_batch(wp_headings)
        wp_headings = smooth_near_wps_batch(wp_headings) # el doble

        wp_curvatures = get_curvatures_from_headings_batch(wp_headings)
        ######
        # mask out loss for wps more than n seconds ahead

        speed_mask = get_speed_mask(aux_model)

        MAX_ANGLE_TO_PRED = .48 #.36 #.18 #.16
        to_pred_mask = torch.from_numpy((np.abs(wp_angles) < MAX_ANGLE_TO_PRED).astype(np.float16)).to(device)
        to_pred_mask = (to_pred_mask*.9) + .1 # 1.0 for all normal angles, .1 for all big angles

        ZERO_THRESH = 1.0
        zero_mask = torch.from_numpy((np.abs(wp_angles) < ZERO_THRESH).astype(np.float16)).to(device)
        to_pred_mask = to_pred_mask*zero_mask # totally zero out above this threshold

        to_pred_mask = to_pred_mask * torch.from_numpy(speed_mask).to(device)
        #to_pred_mask = torch.from_numpy(speed_mask).to(device)
        timer.log("wps prep and mask")

        ########
        # Prep for delivery

        img, aux_model, aux_calib  = prep_inputs(img, aux_model, aux_calib)
        wp_angles, wp_headings, wp_curvatures, wp_rolls, wp_zs, aux_targets = prep_targets(wp_angles, wp_headings, wp_curvatures, wp_rolls, wp_zs, aux_targets)

        self.seq_ix += bptt
        timer.log("prep for delivery")

        # If at the end of seq, move the backup into active position and grab a new backup
        # This will pause until backup is ready
        if self.seq_ix > seq_len-bptt: 
            self.promote_backup_chunk()
        timer.log("promote backup chunk")

        self.logger.log(timer.finish())

        self.queued_batches.append(((img, 
                                    aux_model,
                                    aux_calib, 
                                    wp_angles,
                                    wp_headings,
                                    wp_curvatures,
                                    wp_rolls,
                                    wp_zs,
                                    aux_targets,
                                    to_pred_mask), 
                                        is_first_in_seq))


    def promote_backup_chunk(self):
        while not self.backup_chunk_is_ready:
            print("backup chunk not yet ready")
            time.sleep(1)

        self.seq_ix = SEQ_START_IX

        self.img_chunk[:,:,:,:,:] = self.img_chunk_backup[:,:,:,:,:] 
        self.img_bw_chunk[:,:,:,:,:] = self.img_bw_chunk_backup[:,:,:,:,:] 
        self.maps_chunk[:,:,:,:,:] = self.maps_chunk_backup[:,:,:,:,:] 
        self.aux_chunk[:,:,:] = self.aux_chunk_backup[:,:,:]
        self.targets_chunk[:,:,:] = self.targets_chunk_backup[:,:,:]
        self.episode_info[:,:] = self.episode_info_backup[:,:]

        threading.Thread(target=self.refresh_backup_chunk).start() 

    def get_batch(self):

        while len(self.queued_batches)==0 and self.retry_counter < 1200:
            if (self.retry_counter+1)%100==0:
                print("Waiting for next chunk...")
            self.retry_counter += 1
            time.sleep(.1)

        thread = threading.Thread(target=self.queue_up_batch)
        thread.start()

        self.logger.log({"timing/wait bc batch not ready":.1*self.retry_counter})
        
        if len(self.queued_batches)==0:
            print("Finished dataloader!")
            return None
        else:
            self.retry_counter = 0
            return self.queued_batches.pop(0)

    def get_obs_per_second(self):
        return get_obs_per_sec()
    
    def report_logs(self):
        stats = self.logger.finish()
        return stats
