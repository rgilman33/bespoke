def get_pts_and_headings_fig(wp_angles, wp_dists, wp_headings, wp_curvatures):

    #plt.close('all') # dunno about this. HACK
    
    xs = np.sin(wp_angles) * wp_dists
    ys = np.cos(wp_angles) * wp_dists
    
    # Near traj
    fig, (ax, ax2, ax3) = plt.subplots(1,3, figsize=(12,3), gridspec_kw={'width_ratios': [1, 1, 2]})
    
    ax.scatter(xs[:20], ys[:20])

    for i in range(0,20-1):
        h = wp_headings[i]
        xd = np.sin(h)*SEGMENT_DISTS[i]*.6
        yd = np.cos(h)*SEGMENT_DISTS[i]*.6
        ax.plot([xs[i], xs[i]+xd], [ys[i], ys[i]+yd], 'Blue')

    ax.set_title("Traj (near)", fontdict={"fontsize":16})
    ax.set_yticks([6, 12, 24])
    
    xmax = max(abs(xs[:19]))
    x_axis_max = .1 if xmax <= .1 else .5 if xmax <= .5 else 1 if xmax <= 1 else 2 if xmax <=2 else 3 if xmax <=3 else 5
    ax.set_xticks([-x_axis_max, 0, x_axis_max])
    
    # far traj
    ax2.scatter(xs[20:], ys[20:])

    for i in range(20, 30-1):
        h = wp_headings[i]
        xd = np.sin(h)*SEGMENT_DISTS[i]*.6
        yd = np.cos(h)*SEGMENT_DISTS[i]*.6
        ax2.plot([xs[i], xs[i]+xd], [ys[i], ys[i]+yd], 'Blue')

    ax2.set_title("Traj (far)", fontdict={"fontsize":16})
    #ax2.set_yticks([35, 125])
    
    xmax = max(abs(xs[20:]))
    x_axis_max = 1 if xmax <= 1 else 5 if xmax <= 5 else 10 if xmax <=10 else 20
    ax2.set_xticks([-x_axis_max, 0, x_axis_max])
    
    # curvatures
    maxc = max(abs(wp_curvatures))
    y_axis_max = .001 if maxc<=.001 else .005 if maxc <= .005 else .01 if maxc <= .01 else .02 if maxc<=.02 else .03
    ax3.plot(wp_dists, wp_curvatures)
    ax3.set_title("Curvatures", fontdict={"fontsize":16})
    ax3.set_yticks([-y_axis_max, 0, y_axis_max])
    ax3.set_xticks([6, 125])
    
    return fig


from skimage.transform import resize
import cv2

def fig_to_img(fig, size):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    chart = (resize(data, size) * 255).astype('uint8')
    return chart









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
    angles_deg = rad_to_deg(pred*torch.from_numpy(TARGET_NORM).to('cuda'))
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



def combine_img_cam(act_grad, img, cutoff, color=[255, 0, 0]): # TODO delete this. Deprecated in favor of below
    """ act_grad np float, just the salmap*gradients. img np out of 255"""
    
    # resize to same as img
    act_grad = cv2.resize(act_grad, (img.shape[1],img.shape[0]))
        
    act_grad = (abs(act_grad)>(cutoff)).astype(int)
    act_grad_mask = np.expand_dims(act_grad, -1)

    act_grad = (act_grad_mask * np.array(color)).astype(np.uint8)

    cam = act_grad*.5 + img*((act_grad_mask*-1+1)*.5+.5)
    cam = np.clip(cam, 0, 255).astype('uint8')
    
    return cam


def make_simple_vid(run):
    t0 = time.time()
    imgs = run.imgs
    maps = run.maps
    w2, h2 = IMG_WIDTH//2, IMG_HEIGHT//2
    _imgs = np.empty_like(run.imgs)

    for i in range(len(imgs)-1):

        img = imgs[i].copy()
        img[-MAP_HEIGHT:, -MAP_WIDTH:, :] = maps[i]
        
        # wps
        if run.wp_angles is not None:
            img = draw_wps(img, run.wp_angles[i])
        
        # Guidelines
        img[:,w2-1:w2+1,:] -= 20 # darker line vertical center
        img[h2-1:h2+1:,:,:] -= 20 # darker line horizontal center
        
        _imgs[i, :,:,:] = img

    write_vid(_imgs, f"{run.run_id}_simple")
    print(f"Simple vid created in {round((time.time()-t0)/60, 2)} minutes.")

def combine_vids(m_path_1, m_path_2, run_id):
    fps = 20
    f1, f2 = f"{run_id}_{m_path_1}", f"{run_id}_{m_path_2}"
    print(f"Combining {f1} and {f2}")
    v1 = cv2.VideoCapture(f'/home/beans/bespoke_vids/{f1}.avi')
    v2 = cv2.VideoCapture(f'/home/beans/bespoke_vids/{f2}.avi')
    ret, frame_1 = v1.read()
    ret, frame_2 = v2.read()
    height1, width1, channels = frame_1.shape
    height2, width2, channels = frame_2.shape
    print(f"{f1} shape {frame_1.shape}, {f2} shape {frame_2.shape}")
    #assert width1==width2 # currently supporting older vids TODO UNDO
    height = height1 *2 #+ height2
    width = width1

    video = cv2.VideoWriter(f'/home/beans/bespoke_vids/{run_id}_{m_path_1}vs{m_path_2}.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, (width,height))
    
    while True:
        ret_1, frame_1 = v1.read()
        ret_2, frame_2 = v2.read()
        if not (ret_1 and ret_2): break
        f = np.concatenate([frame_1, frame_2[:IMG_HEIGHT, :IMG_WIDTH, :]], axis=0) 
        video.write(f)

    video.release()
    print("combined!")



def add_noise(activations, std=.1):
    noise = torch.randn(activations.shape, dtype=activations.dtype).to(device) * std
    noise += 1 # centered at one w std of std
    return activations * noise

def dropout_no_rescale(activations, p=.2): 
    mask = (torch.rand_like(activations) > p).half()
    return activations * mask





# NOTE tuned recently. This is the one to use TODO maybe use the other one instead
max_speed_lookup = [ # estimated from run260, abq. 
    (.005, 100),
    (.01, 80),
    (.0175, 60),
    (.035, 50),
    (.065, 40),
    (.12, 30),
    (.23, 20),
    (.3, 15),
    (.42, 10),
]
max_speed_bps = [x[0] for x in max_speed_lookup]
max_speed_vals = [kph_to_mps(x[1]) for x in max_speed_lookup]