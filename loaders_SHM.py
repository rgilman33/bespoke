
from train_utils import aug_imgs
import threading
from constants import *
from imports import *
from traj_utils import *


def update_seq_inplace(img_chunk, aux_chunk, targets_chunk, b, is_done, offset):
    #TODO keep an eye here. Won't likely run into targets file not exist, but may get the occasion where imgs begin to overwrite.
    # ie we grab a targets right before it's deleted, and imgs begin to be overwritten before we can grab them all. Also note this modulo
    # move is to keep sampling more even. Offset is so first runners don't get preference.
    datagen_id = ("00"+str((b+offset) % N_RUNNERS))[-2:] 

    # Grab a target file. That's how we choose a seq.
    ts = []
    first_ix = SEQ_LEN - 1
    while len(ts)==0: # So we can start sooner than when all runs fully populated
        ts = glob.glob(f"{BLENDER_MEMBANK_ROOT}/dataloader_{datagen_id}/run_{random.randint(0, RUNS_TO_STORE_PER_PROCESS-1)}/targets_*.npy")
        ts = [f for f in ts if f"targets_{first_ix}.npy" not in f] # don't load first seq, not enough history, it's a buffer for second seq
    t = random.choice(ts)

    # Parse filename for end_ix identifier
    t_split = t.split('/')
    p = t_split[-1]
    t_stem = '/'.join(t_split[:-1])
    end_ix = int(p.split('.npy')[0].split('_')[1])

    # Load targets
    targets = np.load(t)
    targets_chunk[b, :, :] = targets

    # Load aux. Will use and update below before caching to chunk
    aux = na(np.load(f"{t_stem}/aux_{end_ix}.npy"), AUX_PROPS)
    aux_chunk[b, :, :] = aux

    # Maps
    maps = np.load(f"{t_stem}/maps_{end_ix}.npy") 

    # moving targets forward by one bc of they're off by one. Without this targets are lagged. This is critical.
    targets_chunk[b,:-1,:] = targets_chunk[b,1:,:] 
    aux_chunk[b,:-1,:] = aux_chunk[b,1:,:] 
    maps[:-1, :,:,:] = maps[1:, :,:,:] # maps have to remain synced w aux, though it's less important that they're lagged
    
    # imgs. These are the vanilla imgs, there are more of them than will be in our img_chunk, which is assembled img for model
    # load extra imgs on the back. Won't fail bc we never load first seq
    L = SLOW_2_LB_IX+FAST_LB_IX
    # img_paths = sorted(glob.glob(f"{t_stem}/imgs/*"))[end_ix-SEQ_LEN+1:end_ix+1] 
    img_paths = sorted(glob.glob(f"{t_stem}/imgs/*"))[end_ix-SEQ_LEN+1-L:end_ix+1] # imgs 1 indexed, targets zero indexed
    imgs = np.empty((len(img_paths), IMG_HEIGHT, IMG_WIDTH, 3), dtype='uint8')
    for i, p in enumerate(img_paths): imgs[i, :,:,:] = cv2.imread(p) # NOTE the most time consuming part of this fn
    imgs = imgs[:, :,:,::-1] #bgr to rgb

    # Aug images. NOTE time consuming TODO MAYBE potentially capture augs here in aux
    imgs = aug_imgs(imgs[None,:, :,:,:])[0] # fn requires batch dimension 

    # Bwify imgs. NOTE also time consuming
    img_bw = bwify_seq(imgs)

    # Assemble img for model
    img_chunk[b,:, :,:,:] = cat_imgs(imgs, img_bw, maps, aux, seq_len=SEQ_LEN)

    is_done[b] = 1

from multiprocessing import Queue, Process
from multiprocessing import shared_memory


def _fill_chunk_inplace(img_chunk, aux_chunk, targets_chunk):
    # sets them in place. Takes about 2 sec per seq of len 116. Threading important for perf, as most of the time is io reading in imgs
    bs, _,_ = aux_chunk.shape
    is_done = np.zeros(bs)
    for b in range(bs):
        threading.Thread(target=update_seq_inplace, args=(img_chunk, aux_chunk, targets_chunk, b, is_done, random.randint(0,10_000))).start() 
        #update_seq_inplace(img_chunk, aux_chunk, targets_chunk, b, is_done, random.randint(0,10_000))
    while is_done.sum() < bs:
        time.sleep(.1)

class TrnLoader():
    def __init__(self, bs, n_batches=np.inf, bptt=BPTT):
        self.bs = bs
        self.bptt = bptt
        self.logger = Logger()
        self.path_stem = "trn"
        self.run_id = "trn" # for use w Rollout apparatus
        self.is_rw = False
        self.n_batches = n_batches
        self.batches_delivered = 0

        self.seq_ix = 0
        self.queued_batches = []
        self.retry_counter = 0
        
        self.img_chunk = get_img_container(bs, SEQ_LEN)
        self.aux_chunk = get_aux_container(bs, SEQ_LEN)
        self.targets_chunk = get_targets_container(bs, SEQ_LEN)

        img_shm = shared_memory.SharedMemory(create=True, name="img_shm", size=self.img_chunk.nbytes)
        aux_shm = shared_memory.SharedMemory(create=True, name="aux_shm", size=self.aux_chunk.nbytes)
        targets_shm = shared_memory.SharedMemory(create=True, name="targets_shm", size=self.targets_chunk.nbytes)

        self.img_chunk_b = np.ndarray(self.img_chunk.shape, dtype=self.img_chunk.dtype, buffer=img_shm.buf)
        self.aux_chunk_b = np.ndarray(self.aux_chunk.shape, dtype=self.aux_chunk.dtype, buffer=aux_shm.buf)
        self.targets_chunk_b = np.ndarray(self.targets_chunk.shape, dtype=self.targets_chunk.dtype, buffer=targets_shm.buf)

        controller_props = ["should_stop", "backup_chunk_ready"]
        controller_shm = shared_memory.SharedMemory(create=True, name="controller_shm", size=len(controller_props))
        self.controller = np.ndarray((len(controller_props),), dtype='uint8', buffer=controller_shm.buf)

        self.chunks_worker = Process(target=self.keep_backup_chunks_fresh).start()
        
        self.promote_backup_chunk()

        print("Got first chunk")
        self.queue_up_batch()

    def keep_backup_chunks_fresh(self): # done in separate process to remove from main process. Still have substantial time spent in handoff though, consider shm
        img_shm = shared_memory.SharedMemory(name="img_shm")
        aux_shm = shared_memory.SharedMemory(name="aux_shm")
        targets_shm = shared_memory.SharedMemory(name="targets_shm")
        controller_props = ["should_stop", "backup_chunk_ready"]
        controller_shm = shared_memory.SharedMemory(name="controller_shm")

        controller = np.ndarray((len(controller_props),), dtype='uint8', buffer=controller_shm.buf)
        img_chunk_b = np.ndarray(self.img_chunk.shape, dtype=self.img_chunk.dtype, buffer=img_shm.buf)
        aux_chunk_b = np.ndarray(self.aux_chunk.shape, dtype=self.aux_chunk.dtype, buffer=aux_shm.buf)
        targets_chunk_b = np.ndarray(self.targets_chunk.shape, dtype=self.targets_chunk.dtype, buffer=targets_shm.buf)

        while True:
            print("dd", controller)
            if controller[0]: break

            if not controller[1]:
                _fill_chunk_inplace(img_chunk_b, aux_chunk_b, targets_chunk_b)
                controller[1] = 1
            else:
                time.sleep(.2)
        print("done making chunks!")

    def promote_backup_chunk(self):
        print("promoting backup chunk")
        while not self.controller[1]:
            print("backup chunk not yet ready")
            time.sleep(.2)

        print(self.controller)
        self.seq_ix = 0
        self.img_chunk[:,:, :,:,:] = self.img_chunk_b[:,:, :,:,:] 
        self.aux_chunk[:,:,:] = self.aux_chunk_b[:,:,:]
        self.targets_chunk[:,:,:] = self.targets_chunk_b[:,:,:]

        self.controller[1] = 0

    def queue_up_batch(self):
        print("queueing batch")
        timer = Timer("queue_batch")
        bptt = self.bptt
        _, seq_len, _, _, _ = self.img_chunk.shape
        batch = get_batch_at_ix(self.img_chunk, self.aux_chunk, self.targets_chunk, self.seq_ix, bptt, timer=timer)
        self.seq_ix += bptt
        timer.log("get_batch_at_ix")

        # If at the end of seq, move the backup into active position and grab a new backup
        # This will pause until backup is ready
        if (self.seq_ix > seq_len-bptt): # Trn, not taking any w len shorter than bptt
            print("going to promote backup chunk")
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


from map_utils import *

class RunLoader():
    def __init__(self, run_path, is_rw=True, redraw_maps=False, _gamma_correct_auto=False):
        self.logger = Logger()
        self.queued_batches = []
        self.seq_ix = 0
        self.retry_counter = 0
        self.bptt = 12
        self.is_rw = is_rw
        self.batches_delivered = 0
        self.n_batches = np.inf
        self.gamma_correct_auto = _gamma_correct_auto

        if is_rw: 
            self.run_id = run_path.split("/")[-1]
            self._load_rw(run_path, redraw_maps)
        else:
            self.run_id = SIM_RUN_ID
            self._load_sim(run_path)

        self._finalize_loader_init()

        self.queue_up_batch()

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
            self.maps = get_maps(self.aux)
        else:
            self.maps = np.stack([np.load(f) for f in sorted(glob.glob(f"{run_path}/navmap/*"))[s:e]])

        self.targets_chunk = None

    def _load_sim(self, run_path):
        chunks_range = range(SEQ_LEN-1, EPISODE_LEN, SEQ_LEN)
        self.aux_chunk = na(np.concatenate([np.load(f'{run_path}/aux_{i}.npy') for i in chunks_range], axis=0), AUX_PROPS)
        self.targets_chunk = np.concatenate([np.load(f'{run_path}/targets_{i}.npy') for i in chunks_range], axis=0)
        self.maps = np.concatenate([np.load(f'{run_path}/maps_{i}.npy') for i in chunks_range], axis=0)
        self.img_chunk = np.stack([cv2.imread(i)[:,:,::-1].astype(np.uint8) for i in sorted(glob.glob(f"{run_path}/imgs/*"))]) 
            
        # staggering by one, otherwise targets and aux are lagged. This is important. Current blender setup is off by one.
        self.targets_chunk[:-1,:] = self.targets_chunk[1:,:]
        self.aux_chunk[:-1,:] = self.aux_chunk[1:,:] 
        self.maps[:-1, :,:,:] = self.maps[1:, :,:,:]

    def _finalize_loader_init(self):
        # Common to both sim and rw paths
        
        img_bw = bwify_seq(self.img_chunk)
        self.seq_len = len(self.img_chunk) - SLOW_2_LB_IX - FAST_LB_IX # cat hist shortens seqs
        self.img_chunk = cat_imgs(self.img_chunk, img_bw, self.maps, self.aux_chunk, seq_len=self.seq_len)

        # Assembling imgs shortens the seq. Ensure everything else is shortened equally.
        assert len(self.img_chunk)==self.seq_len
        self.aux_chunk = self.aux_chunk[-self.seq_len:]
        self.targets_chunk = self.targets_chunk[-self.seq_len:]

        # Pad batch dimension to keep common dims of (batch,seq,n) across all loaders
        self.img_chunk = self.img_chunk[None,:, :,:,:]
        self.aux_chunk = self.aux_chunk[None,:,:]

        if self.targets_chunk is not None: self.targets_chunk = self.targets_chunk[None,:,:]

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
    if loader.batches_delivered == loader.n_batches:
        return None

    while len(loader.queued_batches)==0 and loader.retry_counter < 1200:
        if hasattr(loader, "is_done") and loader.is_done: 
            print("loader is done")
            return None

        if (loader.retry_counter+1)%100==0:
            print("Waiting for batch...")
        loader.retry_counter += 1
        time.sleep(.1)

    thread = threading.Thread(target=loader.queue_up_batch)
    thread.start()
    loader.logger.log({"timing/wait bc batch not ready":.1*loader.retry_counter})
    
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

    # Wps targets
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



