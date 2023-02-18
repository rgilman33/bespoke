
from train_utils import aug_imgs
import threading
from constants import *
from imports import *
from traj_utils import *

format_ix = lambda ix: ("0000"+str(ix))[-4:]

def get_current_run(dataloader_root):
    # Rare error here, prob writing to it at same time
    try:
        current_run = np.load(f"{dataloader_root}/run_counter.npy")[0]
        return current_run
    except:
        print("Failed to load run_counter.npy. Retrying...")
        time.sleep(1)
        return get_current_run(dataloader_root)
    
def update_seq_inplace(*inputs):
    try:
        _update_seq_inplace(*inputs)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Ending thread.")
        return
    except Exception as e:
        print("Error in thread, trying again\n\n", e)
        time.sleep(1)
        update_seq_inplace(*inputs)

def _update_seq_inplace(img_chunk, aux_chunk, targets_chunk, b, is_done, offset, constant_seq_aug, timings_container):

    timer = Timer("update_seq_inplace")

    datagen_id = ("00"+str((b+offset) % N_RUNNERS))[-2:] 
    dataloader_root = f"{BLENDER_MEMBANK_ROOT}/dataloader_{datagen_id}"

    current_run = get_current_run(dataloader_root)

    available_runs = glob.glob(f"{dataloader_root}/*/")
    available_runs = [r for r in available_runs if f"run_{current_run}" not in r]

    run_path = random.choice(available_runs)

    timer.log("get run_path")

    # Targets and aux start at 0, imgs start at one. This actually lines up correctly if we match ixs. ie we throw away the first 
    # target and aux. img1 corresponds to targets1, aux1. One is our first obs.

    bptt = img_chunk.shape[1] #TODO this should't be called bptt, it's seqlen
    ix = random.randint(bptt+FS_LOOKBACK, EPISODE_LEN-1) # ix will be the current obs, ie the latest obs
    
    # Load targets, aux, maps
    maps = []
    ixs = list(range(ix+1-bptt, ix+1)) # ends in ix
    for i, _ix in enumerate(ixs):
        targets = np.load(f"{run_path}/targets/{_ix}.npy")
        targets_chunk[b, i, :] = targets# rare error here, why? this should always exist
        aux_chunk[b, i, :] = na(np.load(f"{run_path}/aux/{_ix}.npy"), AUX_PROPS)
        maps.append(np.load(f"{run_path}/maps/{_ix}.npy"))
    maps = np.concatenate(maps, axis=0)
    timer.log("load targets, aux, maps")

    # smoothing targets along sequence dimension. Later on we smooth along the traj itself.
    if bptt > 1: #NOTE this is a sensitive move. Pay attn to this.
        w = 5
        targets_chunk[b, :, :] = moving_average_n(targets_chunk[b, :,:], w)
    timer.log("smooth targets")

    # imgs
    img_paths = [f"{run_path}/imgs/{format_ix(i)}.jpg" for i in range(ix+1-bptt-FS_LOOKBACK, ix+1)]
    imgs = np.stack([cv2.imread(img_path) for img_path in img_paths])[:, :,:,::-1] # bgr to rgb
    timer.log("load imgs")

    constant_seq_aug = 0 #TODO remove this
    imgs = aug_imgs(imgs, constant_seq_aug)
    imgs_bw = bwify_seq(imgs)
    timer.log("aug imgs")

    # Assemble img for model
    aux_chunk[b,:,"has_tire_angle"] = 0 # temporary
    img_chunk[b,:, :,:,:] = cat_imgs(imgs[FS_LOOKBACK:], imgs_bw[:-FS_LOOKBACK], maps, aux_chunk[b,:,:]) # each has seq dim, no batch
    timer.log("assemble img")

    timings_container.append(timer.finish())
    is_done[b] = 1 # why does this work, but didn't in my eval rw fn?

from multiprocessing import Queue, Process

class TrnLoader():
    def __init__(self, bs, n_batches=np.inf, bptt=BPTT, seqlen=1, constant_seq_aug=.5):
        self.bs = bs
        self.bptt = bptt
        self.logger = Logger()
        self.path_stem = "trn"
        self.run_id = "trn" # for use w Rollout apparatus
        self.is_rw = False
        self.n_batches = n_batches
        self.batches_delivered = 0
        self.seqlen = seqlen
        self.constant_seq_aug = constant_seq_aug

        self.seq_ix = 0
        self.queued_batches = []
        self.retry_counter = 0
        set_loader_should_stop(False)

        self.img_chunk = get_img_container(bs, seqlen)
        self.aux_chunk = get_aux_container(bs, seqlen)
        self.targets_chunk = get_targets_container(bs, seqlen)

        self.img_chunk_b = get_img_container(bs, seqlen)
        self.aux_chunk_b = get_aux_container(bs, seqlen)
        self.targets_chunk_b = get_targets_container(bs,seqlen)

        self.should_stop = False
        N_WORKERS = 3
        self.chunks_queue = Queue(maxsize=N_WORKERS)
        print(f"Launching {N_WORKERS} loader workers")
        for i in range(N_WORKERS):
            p = Process(target=self.make_chunks)
            setattr(self, f"chunks_worker{i}", p)
            p.start()

        self._refresh_backup_chunk()
        self.promote_backup_chunk()

        print("Got first chunk")
        # self.queue_up_batch()
        

    def make_chunks(self): # done in separate process to remove from main process. Still have substantial time spent in handoff though, consider shm
        while True:
            if get_loader_should_stop(): 
                print("worker process received order to stop")
                break
            if not self.chunks_queue.full():
                timer = Timer("make chunks")
                img_chunk = get_img_container(self.bs, self.seqlen)
                aux_chunk = get_aux_container(self.bs, self.seqlen)
                targets_chunk = get_targets_container(self.bs, self.seqlen)
                timer.log("get containers")
                timings = self._fill_chunk_inplace(img_chunk, aux_chunk, targets_chunk)
                timer.log("fill chunk inplace")
                timings.update(timer.finish())
                self.chunks_queue.put(((img_chunk, aux_chunk, targets_chunk), timings))
            else:
                time.sleep(.01)
        print("done making chunks!")

    def _fill_chunk_inplace(self, img_chunk, aux_chunk, targets_chunk):
        # sets them in place. Takes about 2 sec per seq of len 116. Threading important for perf, as most of the time is io reading in imgs
        is_done = np.zeros(self.bs)
        threads = []
        timings_container = []
        for b in range(self.bs):
            t = threading.Thread(target=update_seq_inplace, args=(img_chunk, aux_chunk, targets_chunk, b, is_done, random.randint(0,10_000), self.constant_seq_aug, timings_container))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        logger = Logger() # using logger just for averaging
        for timings in timings_container: logger.log(timings)
        timings = logger.finish()

        assert is_done.sum() == self.bs
        return timings
        
    def _refresh_backup_chunk(self):
        timer = Timer("get chunk from queue")
        self.backup_chunk_is_ready = False
        self.seq_ix = 0
        if self.chunks_queue.qsize() == 0:
            print("queue empty!")
        (self.img_chunk_b, self.aux_chunk_b, self.targets_chunk_b), timings = self.chunks_queue.get()
        self.aux_chunk_b = na(self.aux_chunk_b, AUX_PROPS) #Why necessary?
        # get the number of things in the queue
        self.logger.log({"timing/queue size": self.chunks_queue.qsize()})
        self.logger.log(timer.finish())
        self.logger.log(timings)
        self.backup_chunk_is_ready = True

    def promote_backup_chunk(self):
        i = 0
        while not self.backup_chunk_is_ready:
            if (i+1)%300==0: print("backup chunk not yet ready")
            time.sleep(.01)
            i+=1
        self.seq_ix = 0
        self.img_chunk[:,:, :,:,:] = self.img_chunk_b[:,:, :,:,:] 
        self.aux_chunk[:,:,:] = self.aux_chunk_b[:,:,:]
        self.targets_chunk[:,:,:] = self.targets_chunk_b[:,:,:]

        threading.Thread(target=self._refresh_backup_chunk).start() # the theory being this is mostly unpickling time, which would benefit from threading

    def queue_up_batch(self):
        timer = Timer("queue_batch")
        bptt = self.bptt
        _, seq_len, _, _, _ = self.img_chunk.shape
        batch = get_batch_at_ix(self.img_chunk, self.aux_chunk, self.targets_chunk, self.seq_ix, bptt, timer=timer)
        self.seq_ix += bptt
        timer.log("get_batch_at_ix")

        # If at the end of seq, move the backup into active position and grab a new backup
        # This will pause until backup is ready
        if (self.seq_ix > seq_len-bptt): # Trn, not taking any w len shorter than bptt
            self.promote_backup_chunk()
        timer.log("promote backup chunk")
        self.logger.log(timer.finish())

        self.queued_batches.append(batch)

    def get_batch(self):
        return _iter_batch(self)

    def get_obs_per_second(self):
        return get_obs_per_sec()
    
    def report_logs(self):
        stats = self.logger.finish()
        return stats

class ZLoader():
    def __init__(self, m, bs, bptt, seqlen, constant_seq_aug=.0):
        self.m = m 
        self.bs = bs
        self.bptt = bptt
        self.seqlen = seqlen
        self.constant_seq_aug = constant_seq_aug
        self.normal_loader = TrnLoader(bs=bs, bptt=7, seqlen=seqlen, constant_seq_aug=constant_seq_aug)
        self.queued_batches = []
        self.batches_delivered = 0
        self.retry_counter = 0
        self.n_batches = np.inf
        self.logger = self.normal_loader.logger
    

    def queue_up_batch(self):
        n_img_batches_per_z_batch = self.bptt // self.normal_loader.bptt
        n_batches = 0
        zs, imgs, auxs, wpss, to_pred_masks, is_first_in_seqs = [], [], [], [], [], []

        while n_batches < n_img_batches_per_z_batch:
            batch = self.normal_loader.get_batch()
            img, aux, wps, (to_pred_mask, is_first_in_seq) = batch
            with torch.no_grad():
                with torch.cuda.amp.autocast(): z = self.m.cnn_features(img, aux)
            zs.append(z); auxs.append(aux); wpss.append(wps); to_pred_masks.append(to_pred_mask)
            #imgs.append(img.cpu());
            is_first_in_seqs.append(is_first_in_seq)
            n_batches += 1
        
        zs = torch.cat(zs, dim=1); auxs = torch.cat(auxs, dim=1); wpss = torch.cat(wpss, dim=1); to_pred_masks = torch.cat(to_pred_masks, dim=1)
        imgs = None #torch.cat(imgs, dim=1)
        is_first_in_seq = is_first_in_seqs[0]

        self.queued_batches.append((zs, imgs, auxs, wpss, (to_pred_masks, is_first_in_seq)))
    
    def get_batch(self):
        return _iter_batch(self)

    def get_obs_per_second(self):
        return self.normal_loader.get_obs_per_second()

    def report_logs(self):
        stats = self.normal_loader.logger.finish()
        return stats


    
    


from map_utils import *

class RunLoader():
    def __init__(self, run_path, is_rw=True, redraw_maps=False, _gamma_correct_auto=False):
        self.logger = Logger()
        self.retry_counter = 0
        self.bptt = 32 #12 # didn't seem to affect throughput speed, tested from 12 to 64 all similar.
        self.bs = 1
        self.is_rw = is_rw
        self.n_batches = np.inf
        self.gamma_correct_auto = _gamma_correct_auto

        if is_rw: 
            self.run_id = run_path.split("/")[-1]
            self._load_rw(run_path, redraw_maps)
        else:
            self.run_id = SIM_RUN_ID
            self._load_sim(run_path)

        self._finalize_loader_init()

        self.reset()

    def reset(self):
        self.seq_ix = 0
        self.batches_delivered = 0
        self.queued_batches = []
        # self.queue_up_batch()

    def _load_rw(self, run_path, redraw_maps):

        # TODO do a check here to ensure all obs align, using ix of file names, in case OP exit mid save
        s, e = 0, -100
        if self.gamma_correct_auto:
            self.img_chunk = np.stack([gamma_correct_auto(np.load(f)) for f in sorted(glob.glob(f"{run_path}/img/*"))[s:e]]) # caching this only about 20% faster
        else:
            self.img_chunk = np.stack([np.load(f) for f in sorted(glob.glob(f"{run_path}/img/*"))[s:e]]) # caching this only about 20% faster

        self.aux_chunk = na(np.stack([np.load(f) for f in sorted(glob.glob(f"{run_path}/aux/*"))[s:e]]), AUX_PROPS)
        print("loaded imgs and aux")
        if redraw_maps:
            self.maps = get_maps(self.aux_chunk)
        else:
            self.maps = np.stack([np.load(f) for f in sorted(glob.glob(f"{run_path}/navmap/*"))[s:e]])

        self.targets_chunk = np.zeros((self.img_chunk.shape[0], N_WPS*4)) # For convenience. at some point we could actually fill these in using kalman filter

    def _load_sim(self, run_path):

        self.aux_chunk = na(np.concatenate([np.load(f'{run_path}/aux/{i}.npy') for i in range(1,EPISODE_LEN+1)], axis=0), AUX_PROPS)
        self.targets_chunk = np.concatenate([np.load(f'{run_path}/targets/{i}.npy') for i in range(1,EPISODE_LEN+1)], axis=0)
        self.maps = np.concatenate([np.load(f'{run_path}/maps/{i}.npy') for i in range(1,EPISODE_LEN+1)], axis=0)
        self.img_chunk = np.stack([cv2.imread(i)[:,:,::-1].astype(np.uint8) for i in sorted(glob.glob(f"{run_path}/imgs/*"))]) 
        print(self.img_chunk.shape, self.maps.shape, self.aux_chunk.shape, self.targets_chunk.shape)

    def _finalize_loader_init(self):
        # Common to both sim and rw paths
        L = FS_LOOKBACK 
        imgs_bw = bwify_seq(self.img_chunk).copy() # need copy otherwise they also have maps
        self.img_chunk = cat_imgs(self.img_chunk[L:], imgs_bw[:-L], self.maps[L:], self.aux_chunk[L:])
        self.seq_len = len(self.img_chunk)

        # Pad batch dimension to keep common dims of (batch,seq,n) across all loaders
        self.img_chunk = self.img_chunk[None,:, :,:,:]
        self.aux_chunk = self.aux_chunk[None,L:,:] # shorten to match img_chunk
        self.targets_chunk = self.targets_chunk[None,L:,:]

        # Chunks are now loaded in common format: np, real units

    def queue_up_batch(self):
        timer = Timer("queue_batch") # not used, but simpler to just pass in
        ix = self.seq_ix
        if ix%(self.bptt*100)==0:print(ix)
        if (ix >= self.seq_len): 
            self.is_done = True
            return
        batch = get_batch_at_ix(self.img_chunk, self.aux_chunk, self.targets_chunk, ix, self.bptt, timer=timer)
        self.seq_ix += self.bptt
        self.queued_batches.append(batch)

    def get_batch(self):
        return _iter_batch(self)
    
    def report_logs(self):
        stats = self.logger.finish()
        return stats


def _iter_batch(loader):
    # On first call, queue up first batch
    if loader.batches_delivered == 0:
        loader.queue_up_batch()

    # On last call, indicate finished. Only trn hits this.
    if loader.batches_delivered == loader.n_batches:
        return None

    while len(loader.queued_batches)==0 and loader.retry_counter < 12_000:
        if hasattr(loader, "is_done") and loader.is_done: 
            print("loader is done")
            return None

        if (loader.retry_counter+1)%1000==0:
            print("Waiting for batch...")
        loader.retry_counter += 1
        time.sleep(.01)

    thread = threading.Thread(target=loader.queue_up_batch)
    thread.start()
    loader.logger.log({"timing/wait bc batch not ready":.01*loader.retry_counter})
    
    if len(loader.queued_batches)==0:
        print("Finished dataloader!")
        return None
    else:
        loader.batches_delivered+=1
        loader.retry_counter = 0
        return loader.queued_batches.pop(0)

from norm import *

def get_batch_at_ix(img_chunk, aux_chunk, targets_chunk, ix, bptt, timer=None):

    is_first_in_seq = (ix==0)

    # Inputs
    img = img_chunk[:, ix:ix+bptt, :,:,:]
    aux = aux_chunk[:, ix:ix+bptt, :]
    speed = aux[:,:,"speed"] # grab this before putting on gpu. Need it for speed mask below

    img = prep_img(img)
    timer.log("prep image")

    aux = prep_aux(aux)
    timer.log("prep aux")

    # Wps targets TODO time for these to go into prep chunk
    wps, to_pred_mask = None, None
    if targets_chunk is not None:
        targets = targets_chunk[:, ix:ix+bptt, :].copy()
        wp_angles, wp_dists, wp_rolls, wp_zs = np.split(targets, 4, axis=2)
        
        wp_angles_smoothed = smooth_near_wps_batch(wp_angles)
        wp_headings = get_headings_from_traj_batch(wp_angles_smoothed, wp_dists)
        wp_headings = smooth_near_wps_batch(wp_headings)
        wp_headings = smooth_near_wps_batch(wp_headings) # el doble
        wp_curvatures = get_curvatures_from_headings_batch(wp_headings)
        timer.log("calc wp targets")

        # Mask
        MAX_ANGLE_TO_PRED = .48 #.36 #.18 #.16 #TODO get rid of this, along w loss weights
        to_pred_mask = (np.abs(wp_angles) < MAX_ANGLE_TO_PRED).astype(np.float16)
        to_pred_mask = (to_pred_mask*.9) + .1 # 1.0 for all normal angles, .1 for all big angles

        ZERO_THRESH = 1.0
        zero_mask = (np.abs(wp_angles) < ZERO_THRESH).astype(np.float16)
        to_pred_mask = to_pred_mask*zero_mask # totally zero out above this threshold

        speed_mask = get_speed_mask(speed) # mask out loss for wps more than n seconds ahead
        to_pred_mask *= speed_mask

        to_pred_mask = torch.from_numpy(speed_mask).to('cuda')
        timer.log("assemble mask")

        wps = np.concatenate([wp_angles, wp_headings, wp_curvatures, wp_rolls, wp_zs], axis=-1)
        wps = prep_wps(wps)
        timer.log("prep wps")

    return img, aux, wps, (to_pred_mask, is_first_in_seq)



