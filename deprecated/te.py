#########################################
## temporal consistency helpers

TIME_DELTA_PER_STEP = .05 # 1 / hz #TODO this should probably be in constants
 

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