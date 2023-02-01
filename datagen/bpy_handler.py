import numpy as np
import os, random, sys, bpy, time, glob
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
sys.path.append("/home/beans/bespoke")
from constants import *
from traj_utils import *
from map_utils import *

sys.path.append("/home/beans/bespoke/datagen")
from autopilot import Autopilot


def get_next_possible_wps(prev_wp, all_start_wps):

    threshold = WP_SPACING*2 # meters. should work w slightly above WP_SPACING

    all_start_wps = all_start_wps[(all_start_wps.way_id != prev_wp.way_id)] # no u-turns

    dist_to_prev_wp = ((all_start_wps.pos_x - prev_wp.pos_x)**2 + (all_start_wps.pos_y - prev_wp.pos_y)**2)**.5
    close_start_wps = all_start_wps[dist_to_prev_wp < threshold]
    return close_start_wps

def _get_route(df, start_locs_df, route_len, just_go_straight, is_ego):
    # route_len is in meters
    segments = []
    start_wp = start_locs_df.sample(1).iloc[0] if start_locs_df is not None else df.sample(1).iloc[0]
    wps_segment_uid = start_wp.wps_segment_uid

    all_start_wps = df[(df.ix_in_curve==0) & (df.wp_no_go==False)]
    if just_go_straight:
        all_start_wps = all_start_wps[all_start_wps.wp_curve_id==2]

    RANDOM_START_OFFSET_M_MAX = 50 if is_ego else 200
    front_chop = random.randint(0, int(RANDOM_START_OFFSET_M_MAX/WP_SPACING))
    n_wps = 0
    visited_sections = []
    while n_wps < ((route_len/WP_SPACING)+front_chop):
        
        segment_df = df[df.wps_segment_uid == wps_segment_uid]
        segment_df = segment_df.sort_values("ix_in_curve")
        segments.append(segment_df.iloc[:-1]) # Not taking last wp of each segment bc it's the same loc as first wp of next segment?
        n_wps += len(segment_df)

        last_wp = segment_df.iloc[-1]

        # only visit each segment once, but don't care about connecting slips. 
        # This could be rationalized a bit. Include to- from- info and it's good w no if statement.
        if not last_wp.wps_connect:
            visited_sections.append(last_wp.intersection_section_id) 

        close_start_wps = get_next_possible_wps(last_wp, all_start_wps)

        close_start_wps = close_start_wps[close_start_wps.intersection_section_id.isin(visited_sections)==False]

        if len(close_start_wps)==0: break
        next_wp = close_start_wps.sample(1).iloc[0]
        wps_segment_uid = next_wp.wps_segment_uid
        
    route_wps = pd.concat(segments)
    route_wps = route_wps.iloc[front_chop:-1]

    # calculate the curvatures and headings of the chosen route
    # this is the curvature of the underlying rd network, not the curvature of the vehicle's path bc the vehicle does dagger
    # using this to eliminate routes w too high curvature
    if is_ego and len(route_wps) > 100: # safeguard, if this short we're not using it anyways
        M_TO_CALC_CURVATURE = 3
        c_skip = int(M_TO_CALC_CURVATURE/WP_SPACING)
        route_wps_skip = route_wps.iloc[::c_skip]

        headings = np.arctan2(route_wps_skip.pos_x.values[1:] - route_wps_skip.pos_x.values[:-1], route_wps_skip.pos_y.values[1:] - route_wps_skip.pos_y.values[:-1])
        headings_d = (headings[1:] - headings[:-1])
        headings_d[headings_d>np.pi] -= 2*np.pi
        headings_d[headings_d<-np.pi] += 2*np.pi
        curvatures = headings_d / M_TO_CALC_CURVATURE # rad / m
        curvatures = np.insert(curvatures, 0, 0) # lose two above
        curvatures = np.insert(curvatures, 0, 0)

        #headings, curvatures = calc_headings_curvatures(route_wps_skip.pos_x.values, route_wps_skip.pos_y.values)
        # print("SHAPES", curvatures.shape, route_wps_skip.shape)
        route_wps_skip["route_curvature"] = curvatures
        
        route_wps['route_curvature'] = route_wps_skip.route_curvature
        route_wps["route_curvature"] = route_wps["route_curvature"].fillna(method='ffill').fillna(method='bfill')
        # print("max curvature in the route", route_wps.route_curvature.max())
    else:
        route_wps["route_curvature"] = 0
    return route_wps

# # from chatgpt
# def calc_headings_curvatures(x, y):
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     headings = np.arctan2(dy, dx)
#     d_headings = np.gradient(headings)

#     d_headings[d_headings>np.pi] -= 2*np.pi
#     d_headings[d_headings<-np.pi] += 2*np.pi

#     curvatures = d_headings / np.hypot(dx, dy)
#     return headings, curvatures

# repeat each item in a list n times
def repeat_list(l, n):
    return [item for item in l for i in range(n)]

def get_route(wp_df, start_locs_df=None, route_len=2000, just_go_straight=False, is_ego=False):
    # route len in meters
    _route_len = 0
    counter = 0
    max_curve = np.inf
    while _route_len < route_len or max_curve > .25: # .25 chosen manually, cuts out about 1 in ten routes, is still pretty sharp. Can maybe go up to .3
        route = _get_route(wp_df, start_locs_df, route_len, just_go_straight, is_ego)
        max_curve = route.route_curvature.abs().max()
        _route_len = len(route)*WP_SPACING
        counter += 1
        if counter>100: 
            print("wtf can't seem to get route of sufficient length?")
            return None
    return route

def get_wp_df(wps_holder_object):
    t0 = time.time()

    df = pd.DataFrame()
    df['wps'] = [d.value for d in wps_holder_object.data.attributes["wps"].data]

    # When all turnoffs connect then all wps are go and the attribute doesn't show up
    if "wp_no_go" in wps_holder_object.data.attributes:
        df['wp_no_go'] = [d.value for d in wps_holder_object.data.attributes["wp_no_go"].data]
    else:
        df['wp_no_go'] = False

    t0 = time.time()
    pos = [d.vector for d in wps_holder_object.data.attributes["curve_position"].data] # .2 sec
    print("getting pos", time.time() - t0)
    t0 = time.time()
    df["pos_x"] = [d[0] for d in pos] # adding to pandas .3 s
    df["pos_y"] = [d[1] for d in pos]
    df["pos_z"] = [d[2] for d in pos]
    print("setting pos in df", time.time() - t0)

    normals = [d.vector for d in wps_holder_object.data.attributes["curve_normal"].data]
    df["normal_x"] = [d[0] for d in normals]
    df["normal_y"] = [d[1] for d in normals]
    df["normal_z"] = [d[2] for d in normals]

    rotations = [d.vector for d in wps_holder_object.data.attributes["curve_rotation"].data]
    df["curve_pitch"] = [d[0] for d in rotations]
    df["curve_roll"] = [d[1] for d in rotations]
    df["curve_heading"] = [d[2] for d in rotations]

    df["rd_roll"] = [d.value for d in wps_holder_object.data.attributes["rd_roll"].data]

    df["intersection_id"] = [d.value for d in wps_holder_object.data.attributes["intersection_id"].data] # .06 sec
    df["section_id"] = [d.value for d in wps_holder_object.data.attributes["section_id"].data]
    df["intersection_section_id"] = (df.intersection_id.astype(str) + df.section_id.astype(str)).astype(str)
    df["to_section"] = [d.value for d in wps_holder_object.data.attributes["to_section"].data]
    df["from_section"] = [d.value for d in wps_holder_object.data.attributes["from_section"].data]

    df["left_turn"] = ((df.from_section==1) & (df.to_section==5)) | \
                        ((df.from_section==10) & (df.to_section==2) | \
                        ((df.from_section==7) & (df.to_section==11)) | \
                        ((df.from_section==4) & (df.to_section==8))
                        ).astype('int')

    df["right_turn"] = ((df.from_section==1) & (df.to_section==11)) | \
                        ((df.from_section==10) & (df.to_section==8) | \
                        ((df.from_section==7) & (df.to_section==5)) | \
                        ((df.from_section==4) & (df.to_section==2))
                        ).astype('int')
    print("left turn wps", df.left_turn.sum(), "right turn wps", df.right_turn.sum())

    # dumb packing up of one-hot into single categorical
    wp_left = [d.value for d in wps_holder_object.data.attributes["wp_left"].data]
    wp_l1 = [d.value for d in wps_holder_object.data.attributes["wp_l1"].data]
    wp_l2 = [d.value for d in wps_holder_object.data.attributes["wp_l2"].data]
    wp_l3 = [d.value for d in wps_holder_object.data.attributes["wp_l3"].data]
    wp_right = [d.value for d in wps_holder_object.data.attributes["wp_right"].data]
    wp_curve_id = []
    for i in range(len(wp_left)):
        wp_curve_id.append(1 if wp_left[i] else 2 if wp_l1[i] else 3 if wp_l2[i] else 4 if wp_l3[i] else 5 if wp_right[i] else 0)
    df["wp_curve_id"] = wp_curve_id

    df["wps_segment_uid"] = (df.intersection_id.astype(str) + df.section_id.astype(str) + df.wp_curve_id.astype(str) + \
                                df.from_section.astype(str) + df.to_section.astype(str)).astype(int)

    df["wps_section_id"] = (df.intersection_id.astype(str) + df.section_id.astype(str)).astype(int)
    # these last two properties necessary to differentiate slips within an intersection

    df["ix_in_curve"] = [d.value for d in wps_holder_object.data.attributes["ix_in_curve"].data]

    # Using for overlap detection when map is especially noised
    df["is_curve_tip"] = [d.value for d in wps_holder_object.data.attributes["is_curve_tip"].data] # currently only on core curves. Dist is 10m.
    df["core_h"] = [d.value for d in wps_holder_object.data.attributes["core_h"].data]
    df["core_v"] = [d.value for d in wps_holder_object.data.attributes["core_v"].data]

    # unique for each wp
    df["wp_uid"] = (df.wps_segment_uid.astype(str) + df.ix_in_curve.astype(str)).astype("int64")
    _df_wps = df[df.wps]
    print("UNIQUE wps, len df", len(_df_wps.wp_uid.unique()), len(_df_wps))

    df["wps_connect"] = [d.value for d in wps_holder_object.data.attributes["wps_connect"].data]

    # used for coarse wp navmap
    df["arm_id"] = [d.value for d in wps_holder_object.data.attributes["arm_id"].data]
    df["way_id"] = (df.intersection_id.astype(str) + df.arm_id.astype(str)).astype(int)

    # longitudinal
    df["wp_is_stop"] = [d.value for d in wps_holder_object.data.attributes["wp_is_stop"].data]

    return df


def update_ap_object(nodes, ap):
    get_node("pitch", nodes).outputs["Value"].default_value = ap.current_rotation[0]
    get_node("roll", nodes).outputs["Value"].default_value = ap.current_rotation[1]
    get_node("heading", nodes).outputs["Value"].default_value = ap.current_rotation[2]
    get_node("pos_x", nodes).outputs["Value"].default_value = ap.current_pos[0]
    get_node("pos_y", nodes).outputs["Value"].default_value = ap.current_pos[1]
    get_node("pos_z", nodes).outputs["Value"].default_value = ap.current_pos[2]


def reset_npc_objects(bpy):
    """
    In blender, clears the existing copies of the master npc object. Makes MAX_N_NPCS copies of the master npc. 
    To be run once at beginning of datagen to make sure npcs are up to date.
    """
    # remove old npcs
    for o in bpy.data.objects:
        if "npc." in o.name:
            bpy.data.objects.remove(o, do_unlink=True)

    # make n copies of master npc. Make the max num, we'll mute the others
    for i in range(MAX_N_NPCS):
        c = bpy.data.objects["npc"].copy()
        c.data = c.data.copy()
        c.modifiers["GeometryNodes"].node_group = c.modifiers["GeometryNodes"].node_group.copy()
        bpy.context.collection.objects.link(c)
        # we now have a copy of the form "npc.001" in the blender scene, which we'll grab and use later

class NPC():
    def __init__(self, ap, nodes, blender_object, dist_from_ego_start_go):
        self.ap, self.nodes, self.blender_object = ap, nodes, blender_object
        self.dist_from_ego_start_go = dist_from_ego_start_go
        self.is_done = False

class TrafficManager():
    def __init__(self, ego_ap, wp_df, episode_info):
        self.episode_info = episode_info
        self.ego = ego_ap # this is ego.ap actually
        # npcs only start along the route, not randomly scattered on map
        start_locs_df = wp_df[wp_df.intersection_id.isin(ego_ap.visited_intersections)] 
        # no NPCs right next to ego at start
        start_locs_df = start_locs_df[(start_locs_df.wps_section_id != ego_ap.start_section_id)]
        if ego_ap.start_section_id==41:
            oncoming_sections = [62, 57,52, 47,42]
        elif ego_ap.start_section_id==67:  
            oncoming_sections = [48, 51, 58, 61, 68]
        else:
            # this happens when on side street
            oncoming_sections = [48, 51, 58, 61, 68] + [62, 57,52, 47,42] # For now not caring about oncoming distinction when on side street

        start_locs_df_oncoming = start_locs_df[start_locs_df.wps_section_id.isin(oncoming_sections)]

        self.wp_df = wp_df
        ONLY_ONCOMING_NPCS_PROB = .5 
        self.npcs_only_oncoming = random.random() < ONLY_ONCOMING_NPCS_PROB
        self.start_locs_df = start_locs_df_oncoming if self.npcs_only_oncoming else start_locs_df

        # existing blender_objects for npcs
        self.existing_npc_objects = [o for o in bpy.data.objects if "npc." in o.name]
        self.npcs = []


    def add_npcs(self, n_npcs):
        # use existing objects rather than create from scratch. For perf bc don't want to be deleting and remaking all the time. And
        # there was fear of buildup of things in blender behind the scenes (eg names were incrementing each time). It's simpler to clear and remake
        # but we were also getting a substantial linear slowdown in datagen which is consistent with possible buildup. Reusing now to avoid that, despite
        # the slightly more amount of complexity
        for i,npc_object in enumerate(self.existing_npc_objects):
            if i < n_npcs:
                npc_object.hide_render = False
                nodes = npc_object.modifiers["GeometryNodes"].node_group.nodes

                npc_ap = Autopilot(self.episode_info, ap_id=i, save_data=False, is_ego=False)
                route = get_route(self.wp_df, start_locs_df=self.start_locs_df, route_len=200 if self.npcs_only_oncoming else 700)
                npc_ap.set_route(route)

                update_ap_object(nodes, npc_ap)
                dist_from_ego_start_go = 200 if self.npcs_only_oncoming else random.uniform(110, 160)
                npc = NPC(npc_ap, nodes, npc_object, dist_from_ego_start_go)
                self.npcs.append(npc)
            else:
                npc_object.hide_render = True

    def step(self):

        self.ego.set_should_yield(False)
        self.ego.lead_dist = DIST_NA_PLACEHOLDER

        for npc in self.npcs:
            if npc.ap.route_is_done:
                npc.blender_object.hide_render = True
                npc.is_done = True
                continue

            npc.ap.set_should_yield(False)

            if dist(npc.ap.current_pos[:2], self.ego.current_pos[:2]) < TRAJ_WP_DISTS[-1]: # Only check detailed trajs if within distance TODO this dist should be dynamic based on the speeds of both parties
                # Checking for collisions, setting yield states
                ego_traj = self.ego.negotiation_traj[:, :2]
                npc_traj = npc.ap.negotiation_traj[:, :2]
                # Result is traj_len x traj_len matrix w dist btwn every point in A to every pt in B
                result = (ego_traj[:, None, :] - npc_traj[None, :, :])**2
                result = np.sqrt(result[:,:,0] + result[:,:,1])
                if result.min() < CRV_WIDTH * 1.2: 
                    argmin_ix_a, argmin_ix_b = np.unravel_index(result.argmin(), result.shape) # np magic to find argmin in 2d array. Don't fully understand.
                    ego_dist_to_collision = argmin_ix_a * self.ego.m_per_wp
                    npc_dist_to_collision = argmin_ix_b * npc.ap.m_per_wp
                    if (ego_dist_to_collision > npc_dist_to_collision) or self.ego.is_in_yield_zone:
                        self.ego.set_should_yield(True)                        
                    else:
                        npc.ap.set_should_yield(True)

                # lead car status
                # overlap_ixs = np.where(self.ego.traj_wp_uids==npc.ap.current_wp_uid)[0] # Doesn't work bc wps overlaps, left right straight. 
                dists = (self.ego.traj_granular - npc.ap.current_pos)**2
                dists = np.sqrt(dists[:,0] + dists[:,1])
                IS_LEAD_TRAJ_DIST_THRESH = 1.0
                overlap_ixs = np.where(dists<IS_LEAD_TRAJ_DIST_THRESH)[0] # Not perfect, would fail if vehicle was directly oncoming. Should also taking into account heading
                npc_is_ego_lead = len(overlap_ixs)>0
                if npc_is_ego_lead:
                    lead_dist = overlap_ixs[0] * WP_SPACING
                    if lead_dist < self.ego.lead_dist:
                        self.ego.lead_dist = lead_dist
                        self.ego.lead_relative_speed = npc.ap.current_speed_mps - self.ego.current_speed_mps

            # move if visible to ego
            if dist(npc.ap.current_pos[:2], self.ego.current_pos[:2]) < npc.dist_from_ego_start_go:
                npc.ap.step()
                update_ap_object(npc.nodes, npc.ap)

        self.npcs = [npc for npc in self.npcs if not npc.is_done] 

def check_map_has_overlapping_rds(df):
    t0 = time.time()
    df = df[df.is_curve_tip==False]
    h = df[df.core_h][["pos_x", "pos_y"]].values
    v = df[df.core_v][["pos_x", "pos_y"]].values
    print(h.shape)

    # distance matrix
    result = (v[:, None, :] - h[None, :, :])**2 # padding so that broadcasting gives us matrix
    result = np.sqrt(result[:,:,0] + result[:,:,1])
    m = result.min()
    print("Min distance btwn v and h rds", m)
    #print("Checking distance took", time.time()-t0)
    return m < 3.5 # chosen manually. Tips are 10m. This should be greater than dists btwn those pts themselves, which is currently 2m. Also using to keep intersections not too sharp.


def set_frame_change_post_handler(bpy, episode_info, save_data=False, run_root=None):
    global ap, tm
    bpy.app.handlers.frame_change_post.clear()

    make_vehicle_nodes = bpy.data.node_groups['MakeVehicle'].nodes 

    t0 = time.time()
    dg = bpy.context.evaluated_depsgraph_get()
    wps_holder_object = bpy.data.objects["wps_holder"].evaluated_get(dg)
    wp_df = get_wp_df(wps_holder_object)
    coarse_map_df = wp_df[wp_df.wps==False]
    map_has_overlapping_rds = check_map_has_overlapping_rds(coarse_map_df)
    if map_has_overlapping_rds:
        print("Map has overlapping rds. Trying again.")
        return False

    wp_df = wp_df[wp_df.wps]
    print("get wp df", time.time() - t0)

    # single-rd. Maybe just straight maybe turns ok.
    JUST_GO_STRAIGHT_PROB = .5
    just_go_straight = random.random()<JUST_GO_STRAIGHT_PROB
    episode_info.just_go_straight = just_go_straight
    if episode_info.is_single_rd:
        s = wp_df[(wp_df.ix_in_curve==0)] # start wps
        if just_go_straight:
            s = s[(s.wp_curve_id==2)] # only straights
            s = s[((s.intersection_id==4) & (s.section_id.isin([1]))) | ((s.intersection_id==6) & (s.section_id.isin([7])))]
        else:
            # s = s[((s.intersection_id==4) & (s.section_id.isin([1, 10, 4]))) | ((s.intersection_id==6) & (s.section_id.isin([7, 4, 10])))] 
            s = s[((s.intersection_id==4) & (s.section_id.isin([10, 4]))) | ((s.intersection_id==6) & (s.section_id.isin([4, 10])))] 
        route = get_route(wp_df, start_locs_df=s, route_len=ROUTE_LEN_M, just_go_straight=just_go_straight, is_ego=True)
    else: # start anywhere, turning allowed
        route = get_route(wp_df, route_len=ROUTE_LEN_M, is_ego=True)

    if route is None: # when can't get route of sufficient len. Don't know if this safeguard is ever tripped
        return False

    print("get route", time.time() - t0)
    ap = Autopilot(episode_info, run_root=run_root, save_data=save_data, ap_id=-1, is_ego=True)

    ap.set_route(route)
    ap.set_nav_map(coarse_map_df)
    update_ap_object(make_vehicle_nodes, ap)

    tm = TrafficManager(wp_df=wp_df, ego_ap=ap, episode_info=episode_info)
    tm.add_npcs(random.randint(0, MAX_N_NPCS) if episode_info.has_npcs else 0)

    print("total setup time", time.time() - t0)
        
    wp_sphere_nodes = bpy.data.node_groups['wp_sphere_nodes'].nodes 
    get_node("pitch", wp_sphere_nodes).outputs["Value"].default_value = ap.current_rotation[0]
    get_node("roll", wp_sphere_nodes).outputs["Value"].default_value = ap.current_rotation[1]
    get_node("heading", wp_sphere_nodes).outputs["Value"].default_value = ap.current_rotation[2]
    get_node("pos_x", wp_sphere_nodes).outputs["Value"].default_value = ap.current_pos[0]
    get_node("pos_y", wp_sphere_nodes).outputs["Value"].default_value = ap.current_pos[1]
    get_node("pos_z", wp_sphere_nodes).outputs["Value"].default_value = ap.current_pos[2]

    def frame_change_post(scene, dg):
        global ap, tm
        ap.step()
        update_ap_object(make_vehicle_nodes, ap)

        # debugging
        get_node("pitch", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_rot[0]
        get_node("roll", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_rot[1]
        get_node("heading", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_rot[2]
        get_node("pos_x", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_pos[0]
        get_node("pos_y", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_pos[1]
        get_node("pos_z", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_pos[2] + 1

        tm.step()

    bpy.app.handlers.frame_change_post.append(frame_change_post)

    return True


