import numpy as np
import cv2, random
from constants import *

EARTH_RADIUS = 6373000.0  # approximate radius of earth in meters
CLOSE_RADIUS = 600 #400 #200 # meters
CLOSE_BUFFER = np.degrees(CLOSE_RADIUS / EARTH_RADIUS)
MAP_SZ_PX = 600 #400 # we'll crop out of this
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
    Mapping conceptually all in euclidean x-y space no geodesics.
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


def _draw_map(small_map, xs, ys, way_ids, color=(255, 0, 0), thickness=3):    
    # We'll draw each line using open-cv. We want to draw each separately so as to not connect ends of some rds w starts of others.
    is_last_node_on_way_filter = (way_ids[:-1] != way_ids[1:])
    ixs_of_last_nodes_on_ways = list(is_last_node_on_way_filter.nonzero()[0]+1) # this is actually first node of next way?
    ixs_of_last_nodes_on_ways = [0] + ixs_of_last_nodes_on_ways + [len(xs)]
    
    pts = np.stack([xs, ys], axis=-1)
    pts = pts.reshape((-1, 1, 2))  # quirk of cv2
    
    # Drawing
    isClosed = False
    for i in range(len(ixs_of_last_nodes_on_ways)-1):
        start_ix, end_ix = ixs_of_last_nodes_on_ways[i], ixs_of_last_nodes_on_ways[i+1]
        ptsSegment = pts[start_ix:end_ix,:,:]
        small_map = cv2.polylines(small_map, [ptsSegment], isClosed, color, thickness)
    
    return small_map


def draw_small_map(xs, ys, way_ids, route_xs=None, route_ys=None, route_way_ids=None):
    
    small_map = np.zeros((MAP_SZ_PX, MAP_SZ_PX, 3), dtype='uint8')   
    small_map = _draw_map(small_map, xs, ys, way_ids, color=(255, 0, 0), thickness=3)
    if route_xs is not None: small_map = _draw_map(small_map, route_xs, route_ys, route_way_ids, color=(150,150,255), thickness=2)

    # ego vehicle, for human viewing. Center of current square.
    h = MAP_SZ_PX//2
    small_map = cv2.circle(small_map, (h, h), radius=2, color=(0,255,255), thickness=-1)

    # # now we crop the square down to the rect we want. 
    small_map = np.flipud(small_map) # bc cv2 origin is top left rather than bottom left
    w2 = MAP_WIDTH // 2
    top_start = 150 #90 # Want ego close to bottom but not fully at bottom. Eyeballed
    small_map = small_map[top_start:top_start+MAP_HEIGHT, h-w2:h+w2, :]
    
    return small_map


def get_map(map_xs, map_ys, way_ids, route_xs, route_ys, route_way_ids, current_x, current_y, vehicle_heading, close_buffer, draw_route=False):
    # helper, wrap the two big fns above

    # background map, filtered down and rotated
    xs, ys, way_ids = prepare_small_map_nodes(map_xs, map_ys, way_ids, current_x, current_y, vehicle_heading, close_buffer)

    # route
    if draw_route:
        # route_way_ids = np.ones(len(route_xs))
        route_xs, route_ys, route_way_ids = prepare_small_map_nodes(route_xs, route_ys, route_way_ids, current_x, current_y, vehicle_heading, close_buffer)
        small_map = draw_small_map(xs, ys, way_ids, route_xs=route_xs, route_ys=route_ys, route_way_ids=route_way_ids)
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


HEADING_ESTIMATION_M = 20.

import math
class HeadingTracker():
    def __init__(self):
        self.xs_hist = []
        self.ys_hist = []
        self.heading = 0 # zero is up, -pi to pi

    def step(self, x=None, y=None, current_speed_mps=None):

        if current_speed_mps > 2.0 and len(self.xs_hist)>0: # only update heading if moving faster than threshold. Don't update on first step.

            headings_estimation_seconds_lookback = HEADING_ESTIMATION_M / current_speed_mps
            headings_estimation_ix_lookback = int(math.ceil(GPS_HZ * headings_estimation_seconds_lookback))
            headings_estimation_ix_lookback = min(len(self.xs_hist), headings_estimation_ix_lookback)
            prev_x, prev_y = self.xs_hist[-headings_estimation_ix_lookback], self.ys_hist[-headings_estimation_ix_lookback]

            self.heading = np.arctan2(x-prev_x, y-prev_y) # -pi to pi

        self.xs_hist.append(x)
        self.ys_hist.append(y)

        return self.heading



def get_maps(aux):
    # redraws in accordance w latest apparatus, which may have changed since rw gather. Used in rollouts to update maps from rw runs
    # so don't have to regather rw each time update maps
    print("Redrawing maps using latest apparatus")

    big_map_lats, big_map_lons, way_ids = get_big_map()
    LOCAL_LON_MULT = get_lon_mult(big_map_lats.mean())
    big_map_xs, big_map_ys = big_map_lons*LOCAL_LON_MULT, big_map_lats
    heading_tracker = HeadingTracker()

    # route_name = "home_to_end_rr" # this actually goes to cold springs?
    route_name = "home_to_po"
    route = np.load(f"{ROUTES_DIR}/{route_name}.npy")
    draw_route = True
    route_xs = route[:,0]*LOCAL_LON_MULT
    route_ys = route[:,1]
    lons, lats = aux[:,'pos_x'], aux[:,'pos_y']
    maps = np.empty((len(lats), MAP_HEIGHT, MAP_WIDTH, 3), dtype='uint8')

    for i in range(len(lats)):
        if i % 1000 == 0: print(i)
        current_x, current_y = lons[i]*LOCAL_LON_MULT, lats[i]
        current_speed_mps = aux[i,'speed']
        if i%4==0: vehicle_heading = heading_tracker.step(current_x, current_y, current_speed_mps) # map updates at 5hz, frames are at 20hz

        small_map = get_map(big_map_xs, 
                            big_map_ys, 
                            way_ids, 
                            route_xs,
                            route_ys,
                            current_x, 
                            current_y, 
                            vehicle_heading, 
                            CLOSE_BUFFER,
                            draw_route)
        maps[i] = small_map

    return maps



################################
# Big maps
################################


import subprocess

OSM_DB_PATH = "/media/beans/ssd/osm/db"
silverton_area_bbox_str = f'{str(44.85017681566702)},{-122.83272781942897},{45.07848012852752},{-122.49831789424171}'
boulder_mtns_bbox_str = f'{str(39.9619)},{-105.5429},{40.0930},{-105.2961}'
boulder_bbox_str = f'{str(39.9619)},{-105.5429},{40.0930},{-105.2143}'


def get_big_map():
    import xmltodict # using this instead of overpy to parse OSM data manually ourselves

    # We'll call this once in the beginning. Just keep loaded in memory. When venturing further out, 
    # we'll need to refresh this every so often.
    # total process of fetching 10 - 20 km radius in each direction will be a couple hundred milliseconds
    # in production version, this will probably be running every minute or so. We have time to keep a healthy buffer

    """far_radius = 40_000 #10_000
    lat, lon = 45.023867, -122.741862
    bbox_angle = np.degrees(far_radius / EARTH_RADIUS)
    bbox_str = f'{str(lat - bbox_angle)},{str(lon - bbox_angle)},{str(lat + bbox_angle)},{str(lon + bbox_angle)}'"""

    #bbox_str = silverton_area_bbox_str
    # bbox_str = boulder_mtns_bbox_str
    bbox_str = boulder_bbox_str
    q = """
        way(""" + bbox_str + """)
          [highway]
          [highway!~"^(footway|path|corridor|bridleway|steps|cycleway|construction|bus_guideway|escape|service|track)$"];
        (._;>;);
        out;
        """

    # grabbing osm data from our local db. Could also grab this from online, if wanted eg big dump of certain area, could then
    # store that offline and not need to deal w local db at all
    completion = subprocess.run(["/home/beans/osm-3s_v0.7.56.9/bin/osm3s_query", f"--db-dir={OSM_DB_PATH}", f'--request={q}'], 
                                                                    check=True, capture_output=True)
    
    # Manual parsing of osm data, which is just xml after all
    osm_db_out = xmltodict.parse(completion.stdout)

    # Rearranging our data in a flat way so we can filter quickly based on node distance from ego, THEN group by way
    # this cell and the next outputs flat tabular style data w three properties: way_id, lat, lon
    # prep nodes_dict for faster lookup below
    ways = osm_db_out['osm']['way']
    nodes = osm_db_out['osm']['node']
    nodes_dict = {}
    for n in nodes:
        nodes_dict[n['@id']] = (float(n["@lat"]), float(n["@lon"]))
    # create the flattened data, nodes are in order bc we're arranging using the 'nd' attribute
    way_ids, lats, lons = [], [], []
    for way in (ways if type(ways)==list else [ways]):
        way_id = int(way["@id"])
        for nd in way['nd']:
            nd_id = nd['@ref']
            node = nodes_dict[nd_id]
            way_ids.append(way_id)
            lats.append(node[0])
            lons.append(node[1])
    lats, lons, way_ids = np.array(lats), np.array(lons), np.array(way_ids)
    
    return lats, lons, way_ids