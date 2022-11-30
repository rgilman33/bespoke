import numpy as np
import cv2, random
from constants import *

EARTH_RADIUS = 6373000.0  # approximate radius of earth in meters
CLOSE_RADIUS = 400 #200 # meters
CLOSE_BUFFER = np.degrees(CLOSE_RADIUS / EARTH_RADIUS)
MAP_SZ_PX = 400 # we'll crop out of this
assert MAP_SZ_PX%2==0

# filtering down the FAR_RADIUS chunk of map data into the CLOSE_RADIUS. We'll draw this entire close area 
# then chop it down to the actual size we need, which will be about half
# this takes .1 - .3 ms w small chunk, 1 ms w silverton area chunk.

def filter_pts(xs, ys, way_ids, current_x, current_y, close_buffer):
    close_filter_xs = (xs > (current_x-close_buffer)) & (xs < (current_x+close_buffer))
    close_filter_ys = (ys > (current_y-close_buffer)) & (ys < (current_y+close_buffer))
    close_filter = close_filter_xs & close_filter_ys

    xs = xs[close_filter]
    ys = ys[close_filter]
    way_ids = way_ids[close_filter]
    
    return xs, ys, way_ids

def prepare_small_map_nodes(xs, ys, way_ids, current_x, current_y, vehicle_heading, close_buffer):
    """
    Takes in cached big map nodes. Filters down to small map nodes, rotates, and scales points to pixels
    in preparation for drawing.
    heading in radians, clockwise positive, -pi to pi, zero is up
    conceptually, origin is bottom left. Lons are xs, lats are ys. Real-world data, lons should have been scaled already.
    Mapping conceptually all in x-y space.
    """
    xs, ys, way_ids = filter_pts(xs, ys, way_ids, current_x, current_y, close_buffer)
    
    # center and rotate, still in latlon units but not actually latlon bc of the rotation
    x, y = xs-current_x, ys-current_y
    c, s = np.cos(-vehicle_heading), np.sin(-vehicle_heading) 
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])
    m = np.asarray(m)
    xs, ys = m[0], m[1]
    
    # scale for drawing
    # -1 to 1
    xs /= close_buffer
    ys /= close_buffer

    # 0 to 1
    xs = (xs + 1)/2
    ys = (ys + 1)/2

    # scale to pixels
    xs *= MAP_SZ_PX
    ys *= MAP_SZ_PX

    # this makes lines less smooth, but we need ints bc this is scaling directly to pixels
    xs, ys = xs.round().astype(np.int32), ys.round().astype(np.int32)
    
    return xs, ys, way_ids

def draw_small_map(xs, ys, way_ids, route_xs=None, route_ys=None):
    
    # We'll draw each line using open-cv. We want to draw each separately so as to not connect ends of some rds w starts of others.
    is_last_node_on_way_filter = (way_ids[:-1] != way_ids[1:])
    ixs_of_last_nodes_on_ways = list(is_last_node_on_way_filter.nonzero()[0]+1) # this is actually first node of next way?
    ixs_of_last_nodes_on_ways = [0] + ixs_of_last_nodes_on_ways + [len(xs)]
    
    small_map = np.zeros((MAP_SZ_PX, MAP_SZ_PX, 3), dtype='uint8')    
    pts = np.stack([xs, ys], axis=-1)
    pts = pts.reshape((-1, 1, 2))  # quirk of cv2
    
    # Drawing
    isClosed = False
    color = (255, 0, 0)
    thickness = 2

    for i in range(len(ixs_of_last_nodes_on_ways)-1):
        start_ix, end_ix = ixs_of_last_nodes_on_ways[i], ixs_of_last_nodes_on_ways[i+1]
        ptsSegment = pts[start_ix:end_ix,:,:]
        small_map = cv2.polylines(small_map, [ptsSegment], isClosed, color, thickness)
    
    # route
    if route_xs is not None:
        route_pts = np.stack([route_xs, route_ys], axis=-1)
        route_pts = route_pts.reshape((-1, 1, 2))  # just copying above convention, don't even know what shape this is now
        small_map = cv2.polylines(small_map, [route_pts], isClosed, (150,150,255), 2)

    # ego vehicle, for human viewing. Center of current square.
    h = MAP_SZ_PX//2
    small_map = cv2.circle(small_map, (h, h), radius=2, color=(0,255,255), thickness=-1)

    # # now we crop the square down to the rect we want. 
    small_map = np.flipud(small_map) # bc cv2 origin is top left rather than bottom left
    w2 = MAP_WIDTH // 2
    top_start = 90 # Want ego close to bottom but not fully at bottom
    small_map = small_map[top_start:top_start+MAP_HEIGHT, h-w2:h+w2, :]
    
    return small_map


def get_map(map_xs, map_ys, way_ids, route_xs, route_ys, current_x, current_y, vehicle_heading, close_buffer, draw_route=False):
    # helper, wrap the two big fns above
    # background map, filtered down and rotated
    xs, ys, way_ids = prepare_small_map_nodes(map_xs, map_ys, way_ids, current_x, current_y, vehicle_heading, close_buffer)

    # route
    if draw_route:
        route_xs, route_ys, _ = prepare_small_map_nodes(route_xs, route_ys, np.ones(len(route_xs)), current_x, current_y, vehicle_heading, close_buffer)
        small_map = draw_small_map(xs, ys, way_ids, route_xs=route_xs, route_ys=route_ys)
    else:
        small_map = draw_small_map(xs, ys, way_ids)

    return small_map



# only used by blender
def add_noise_rds_to_map(lats, lons, way_ids, n_noise_rds=15):
    for i in range(n_noise_rds):
        ix = random.randint(0, len(lats)-1)
        start_lat, start_lon = lats[ix], lons[ix]

        min_dist, max_dist = 2, 100 # meters
        end_lat_add, end_lon_add = random.randint(min_dist, max_dist), random.randint(min_dist, max_dist)
        if random.random()>.5: end_lat_add*=-1
        if random.random()>.5: end_lon_add*=-1

        r = random.random()
        if r < .3: # just altering if make lat lon or none curvy. A hack to get curvature in noise rds
            rd_lats = [start_lat+(end_lat_add*linear_to_sin(p)) for p in [0, .2, .4, .6, .8, 1.0]]
            rd_lons = [start_lon+(end_lon_add*p) for p in [0, .2, .4, .6, .8, 1.0]]
        elif r < .6:
            rd_lats = [start_lat+(end_lat_add*p) for p in [0, .2, .4, .6, .8, 1.0]]
            rd_lons = [start_lon+(end_lon_add*linear_to_sin(p)) for p in [0, .2, .4, .6, .8, 1.0]]
        else:
            rd_lats = [start_lat+(end_lat_add*p) for p in [0, .2, .4, .6, .8, 1.0]]
            rd_lons = [start_lon+(end_lon_add*p) for p in [0, .2, .4, .6, .8, 1.0]]
        lats = np.concatenate([lats, rd_lats])
        lons = np.concatenate([lons, rd_lons])
        way_ids = np.concatenate([way_ids, ([i+300]* len(rd_lats))]) # just making sure isn't same id as real rds HACK this whole fn is hacky. beware.
    return lats, lons, way_ids

# Distance btwn lon lines is a fn of lat. Mult rw lon values by this mult to make it approximately an xy grid eg as we get from blender
# no need for haversine formula, just make it all an xy grid, easier to reason about This is totally fine for how we're using maps, and 
# our big map distances are on the order of tens of km. 
def get_lon_mult(lat_deg):
    return np.cos(np.radians(lat_deg))


HEADING_ESTIMATION_M = 10.

# import pyproj
import math
class HeadingTracker():
    def __init__(self):
        self.xs_hist = [0]
        self.ys_hist = [0]
        # self.geodesic = pyproj.Geod(ellps='WGS84')
        self.heading = 0 # zero is up, -pi to pi

    def step(self, x=None, y=None, current_speed_mps=None):

        if current_speed_mps > 2.0: # only update heading if moving faster than threshold

            headings_estimation_seconds_lookback = HEADING_ESTIMATION_M / current_speed_mps
            headings_estimation_ix_lookback = int(math.ceil(GPS_HZ * headings_estimation_seconds_lookback))
            headings_estimation_ix_lookback = min(len(self.xs_hist), headings_estimation_ix_lookback)
            prev_x, prev_y = self.xs_hist[-headings_estimation_ix_lookback], self.ys_hist[-headings_estimation_ix_lookback]

            # fwd_azimuth, back_azimuth, distance = self.geodesic.inv(prev_lon, prev_lat, current_lon, current_lat) # using real latlon to compute heading, though could just use our fixed up xy pseudo latlons
            # self.heading = np.radians(fwd_azimuth)
            self.heading = np.arctan2(x-prev_x, y-prev_y) # -pi to pi

        self.xs_hist.append(x)
        self.ys_hist.append(y)

        return self.heading


