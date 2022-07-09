from imports import *
from constants import *
from models import EffNet
from input_prep import TARGET_NORM, TARGET_NORM_HEADINGS
from traj_utils import *

class ModelWrapper():
    def __init__(self, wheelbase=None, model_dict=None):
        self.model = EffNet(model_arch=model_dict["arch"]).to(device)
        self.model.load_state_dict(torch.load(f"{BESPOKE_ROOT}/models/{model_dict['stem']}.torch"))
        self.model.eval()
        self.model.reset_hidden(1)

        self.llc = LaggedLateralCalculator(wheelbase=wheelbase)

        self.target_norm = TARGET_NORM.to(device).to(torch.float32)
        self.target_norm_headings = TARGET_NORM_HEADINGS.to(device).to(torch.float32)

        print(f"Initializing model w {model_dict['stem']}")


    def step(self, image_pytorch, aux_pytorch, current_speed, current_tire_angle):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                model_out, obsnet_out = self.model(image_pytorch, aux_pytorch)

        wp_angles, wp_headings, _ = torch.chunk(model_out.to(torch.float32), 3, -1)
        wp_angles *= self.target_norm # back into radians, full precision
        wp_headings *= self.target_norm_headings
        wp_angles = wp_angles[0][0].cpu().detach().numpy()
        wp_headings = wp_headings[0][0].cpu().detach().numpy()

        return self.llc.step(wp_angles, wp_headings, current_speed, current_tire_angle)


class LaggedLateralCalculator():
    def __init__(self, wheelbase=None):

        self.wheelbase = wheelbase

        # abs world space w initial location at origin facing up
        self.traj_xs, self.traj_ys = [], []
        self.vehicle_heading = 0 # rad
        self.vehicle_location_x = 0
        self.vehicle_location_y = 0
        self.curve_speeds_hist = [] # TODO prob make this a queue or whatever


    def step(self, wp_angles, wp_headings, current_speed, current_tire_angle):
        # speed mps, tire_angle rad

        LAG_S = .3 
        WHEELBASE = self.wheelbase 

        dist_car_travelled_during_lag = current_speed * LAG_S

        # local space, used for directing the car
        # everything in the frame of reference of the car at t0, the origin
        WP_ADJUSTMENT = 0
        target_wp_angle, wp_dist, _ = get_target_wp(wp_angles, current_speed, wp_m_offset=dist_car_travelled_during_lag+WP_ADJUSTMENT) # comes out as float, frac is the amount btwn the two wps
        wp_x = np.sin(target_wp_angle) * wp_dist
        wp_y = np.cos(target_wp_angle) * wp_dist

        curvature = current_tire_angle/WHEELBASE # rad/m #TODO WARNING made this change and haven't tested yet, just eyeball checked
        vehicle_turn_rate_sec = curvature * current_speed # rad/sec
        future_vehicle_heading = vehicle_turn_rate_sec * LAG_S # radians, w respect to original ego heading
        # future_vehicle_heading = get_heading_at_dist_along_traj(model_out, dist_car_travelled_during_lag) #TODO test this again, might be better

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
            future_vehicle_x = r - (np.cos(future_vehicle_heading)*r)

        # recenter at future vehicle
        wp_x -= future_vehicle_x
        wp_y -= future_vehicle_y

        # as if vehicle hadn't turned at all, ie heading hadn't changed
        target_wp_angle_future = np.arctan(wp_x/wp_y) 
        # subtracting out what vehicle turned
        target_wp_angle_future -= future_vehicle_heading

        curve_constrained_speed_mps = get_curve_constrained_speed(wp_headings, current_speed)
        self.curve_speeds_hist.append(curve_constrained_speed_mps)
        CURVE_SPEEDS_N_AVG = 4
        curve_constrained_speed_mps = sum(self.curve_speeds_hist[-CURVE_SPEEDS_N_AVG:])/CURVE_SPEEDS_N_AVG
        
        return target_wp_angle_future, curve_constrained_speed_mps
