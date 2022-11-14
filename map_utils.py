import numpy as np
import cv2, random
from constants import *

EARTH_RADIUS = 6373000.0  # approximate radius of earth in meters
CLOSE_RADIUS = 400 #200 # meters
CLOSE_BUFFER = np.degrees(CLOSE_RADIUS / EARTH_RADIUS)
LAT_SZ_PX, LON_SZ_PX = 400, 400 # we'll crop out of this
assert LAT_SZ_PX%2==0

# filtering down the FAR_RADIUS chunk of map data into the CLOSE_RADIUS. We'll draw this entire close area 
# then chop it down to the actual size we need, which will be about half
# this takes .1 - .3 ms w small chunk, 1 ms w silverton area chunk.

def filter_pts(lats, lons, way_ids, current_lat, current_lon, close_buffer):
    close_filter_lats = (lats > (current_lat-close_buffer)) & (lats < (current_lat+close_buffer))
    close_filter_lons = (lons > (current_lon-close_buffer)) & (lons < (current_lon+close_buffer))
    close_filter = close_filter_lons & close_filter_lats

    lats = lats[close_filter]
    lons = lons[close_filter]
    way_ids = way_ids[close_filter]
    
    return lats, lons, way_ids

def prepare_small_map_nodes(lats, lons, way_ids, current_lat, current_lon, vehicle_heading, close_buffer):
    """
    Takes in cached big map nodes. Filters down to small map nodes, rotates, and scales points to pixels
    in preparation for drawing.
    """
    lats, lons, way_ids = filter_pts(lats, lons, way_ids, current_lat, current_lon, close_buffer)
    
    # center and rotate, still in latlon units but not actually latlon bc of the rotation
    x, y = lats-current_lat, lons-current_lon
    c, s = np.cos(vehicle_heading), np.sin(vehicle_heading)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])
    m = np.asarray(m)
    lats, lons = m[0], m[1]
    
    # scale for drawing
    # -1 to 1
    lats /= close_buffer
    lons /= close_buffer

    # 0 to 1
    lats = (lats + 1)/2
    lons = (lons + 1)/2

    # scale to pixels
    lats *= LAT_SZ_PX
    lons *= LON_SZ_PX

    # this makes lines less smooth, but we need ints bc this is scaling directly to pixels
    lats, lons = lats.round().astype(np.int32), lons.round().astype(np.int32)
    
    return lats, lons, way_ids

def draw_small_map(lats, lons, way_ids, route_lats=None, route_lons=None):
    
    # We'll draw each line using open-cv. We want to draw each separately so as to not connect ends of some rds w starts of others.
    is_last_node_on_way_filter = (way_ids[:-1] != way_ids[1:])
    ixs_of_last_nodes_on_ways = list(is_last_node_on_way_filter.nonzero()[0]+1) # this is actually first node of next way?
    ixs_of_last_nodes_on_ways = [0] + ixs_of_last_nodes_on_ways + [len(lats)]
    #print([(way_ids[i], lats[i]) for i in range(len(lats))])
    
    small_map = np.zeros((LAT_SZ_PX, LON_SZ_PX, 3), dtype='uint8')    
    pts = np.stack([lats, lons], axis=-1)
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
    if route_lats is not None:
        for i in range(len(route_lats)):
            small_map = cv2.circle(small_map, (route_lats[i], route_lons[i]), radius=2, color=(150,150,255), thickness=-1)
            
    # ego vehicle, for human viewing
    h = LAT_SZ_PX//2
    small_map = cv2.circle(small_map, (h, h), radius=2, color=(0,255,255), thickness=-1)

    # now we crop to where we want. 
    w2 = MAP_WIDTH // 2
    top_start = 90 # Want ego close to bottom but not fully at bottom
    small_map = small_map[top_start:top_start+MAP_HEIGHT, h-w2:h+w2, :]
    
    return small_map


def get_map(lats, lons, way_ids, route, current_lat, current_lon, vehicle_heading, close_buffer):
    # helper, wrap the two big fns above
    # map
    lats, lons, way_ids = prepare_small_map_nodes(lats, lons, way_ids, current_lat, current_lon, vehicle_heading, close_buffer)

    # route
    route_lats, route_lons, _ = prepare_small_map_nodes(route[:,0], route[:,1], np.ones(len(route)), current_lat, current_lon, vehicle_heading, close_buffer)

    small_map = draw_small_map(lats, lons, way_ids, route_lats=route_lats, route_lons=route_lons)


    return small_map



# only used by blender
def add_noise_rds_to_map(lats, lons, way_ids, n_noise_rds=10):
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
        lats += rd_lats
        lons += rd_lons
        way_ids += ([i+300]* len(rd_lats)) # just making sure isn't same id as real rds HACK this whole fn is hacky. beware.

    return lats, lons, way_ids

# Distance btwn lon lines is a fn of lat. Mult rw lon values by this mult to make it approximately an xy grid eg as we get from blender
# no need for haversine formula, just make it all an xy grid, easier to reason about This is totally fine for how we're using maps, and 
# our big map distances are on the order of tens of km. 
def get_lon_mult(lat_deg):
    return np.cos(np.radians(lat_deg))
