from imports import *
from constants import *
from models import EffNet
from input_prep import TARGET_NORM
from traj_utils import *

class ModelWrapper():
    def __init__(self, wheelbase=None, model_dict=None):
        self.model = EffNet(model_arch=model_dict["arch"]).to(device)
        self.model.load_state_dict(torch.load(f"{BESPOKE_ROOT}/models/{model_dict['stem']}.torch"))
        self.model.eval()
        self.model.reset_hidden(1)

        self.llc = LaggedLateralCalculator(wheelbase=wheelbase)
        print(f"Initializing model w {model_dict['stem']}")


    def step(self, image_pytorch, aux_pytorch, current_speed, current_tire_angle):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                model_out, obsnet_out = self.model(image_pytorch, aux_pytorch)

        model_out = model_out * TARGET_NORM.to(device) # back into radians
        model_out = model_out[0][0].cpu().detach().numpy()

        return self.llc.step(model_out, current_speed, current_tire_angle) #TODO refactor this to just be outside of wrapper?


class LaggedLateralCalculator():
    def __init__(self, wheelbase=None):

        self.wheelbase = wheelbase

        # abs world space w initial location at origin facing up
        self.traj_xs, self.traj_ys = [], []
        self.vehicle_heading = 0 # rad
        self.vehicle_location_x = 0
        self.vehicle_location_y = 0


    def step(self, model_out, current_speed, current_tire_angle):
        # speed mps, tire_angle rad

        LAG_S = .3 #.15 # should this be .15? or even .2? was .1
        WHEELBASE = self.wheelbase 

        dist_car_travelled_during_lag = current_speed * LAG_S

        # local space, used for directing the car
        # everything in the frame of reference of the car at t0, the origin
        WP_ADJUSTMENT = 0 #-2 works well for the 3.21avg model. All carla trained of that gen, prob
        # changing adjustment to 0 bc we updated the lookup table itself
        target_wp_angle, wp_dist, _ = get_target_wp(model_out, current_speed, wp_m_offset=dist_car_travelled_during_lag+WP_ADJUSTMENT) # comes out as float, frac is the amount btwn the two wps
        wp_x = np.sin(target_wp_angle) * wp_dist #TODO is this correct? why is x w sin? ok checked, this looks correct
        wp_y = np.cos(target_wp_angle) * wp_dist


        vehicle_turn_rate = current_tire_angle * (current_speed/WHEELBASE) # rad/sec
        future_vehicle_heading = vehicle_turn_rate * LAG_S # radians
        # the vehicle won't be as turned as the tires, proportional to wheelbase
        # future_vehicle_heading = get_heading_at_dist_along_traj(model_out, dist_car_travelled_during_lag)

        # # Most of the following is approximate. Wrongness gets wronger the steeper the angle.
        # avg_vehicle_heading_during_turn = future_vehicle_heading / 2.
        # future_vehicle_x_simple = np.sin(avg_vehicle_heading_during_turn) * dist_car_travelled_during_lag # TODO this prob isn't right. Need to account for wheelbase. It will be less than this, at least on x value. Will x value be exactly half? i think it will
        # future_vehicle_y_simple = np.cos(avg_vehicle_heading_during_turn) * dist_car_travelled_during_lag # actually i think if we just /=2 tire_angle before csin cos that will be it, be the avg angle will be halfway btwn start and end pt
        # # the proper way to do this would be trace the circle (obtained from curvature) and measure how far we've travelled along it to get future x and y. But i think it will be close to the same bc these angles are small and we're not going that fast.

        # this way is the technically correct one, but keep in mind it's nearly identical to above bc things are nearly linear in these
        # short time periods
        if future_vehicle_heading==0:
            future_vehicle_x = 0
            future_vehicle_y = dist_car_travelled_during_lag
        else:
            r = dist_car_travelled_during_lag / future_vehicle_heading
            future_vehicle_y = np.sin(future_vehicle_heading)*r
            future_vehicle_x = r - (np.cos(future_vehicle_heading)*r) #TODO check this. Ok, looks fine from quick spot check

        # recenter at future vehicle
        wp_x -= future_vehicle_x
        wp_y -= future_vehicle_y

        # as if vehicle hadn't turned at all, ie heading hadn't changed
        target_wp_angle_future = np.arctan(wp_x/wp_y) 
        # subtracting out what vehicle turned
        target_wp_angle_future -= future_vehicle_heading

        return target_wp_angle_future
