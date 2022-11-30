import numpy as np
import os, random, sys, bpy, time, glob
import pandas as pd

sys.path.append("/home/beans/bespoke")
from constants import *
from traj_utils import *
from map_utils import *

sys.path.append("/home/beans/bespoke/datagen")
from autopilot import Autopilot

#########################################
"""
We're currently getting some of our wps by shifting the white lines inwards, getting others by creating a new sigmoid
curve. The point of the latter is to be able to smooth the curve, don't want to always just follow the lines, humans smooth them
out. 
"""

# illegal_connections = {
#     1:[2],
#     2: [1],
#     3: [12],
# }

def get_next_possible_wps(prev_wp, all_start_wps):

    threshold = WP_SPACING*2 # meters. should work w slightly above WP_SPACING

    # seg_id = prev_wp.segment_id 

    # # illegal internal moves
    # if seg_id in illegal_connections.keys():
    #     all_start_wps = all_start_wps[all_start_wps.segment_id.isin(illegal_connections[seg_id]) == False]

    all_start_wps = all_start_wps[(all_start_wps.way_id != prev_wp.way_id)] # no u-turns

    dist_to_prev_wp = ((all_start_wps.pos_x - prev_wp.pos_x)**2 + (all_start_wps.pos_y - prev_wp.pos_y)**2)**.5
    close_start_wps = all_start_wps[dist_to_prev_wp < threshold]
    return close_start_wps

def _get_route(df, start_locs_df, route_len, just_go_straight):
    # route_len is in meters
    segments = []
    start_wp = start_locs_df.sample(1).iloc[0] if start_locs_df is not None else df.sample(1).iloc[0]
    wps_segment_uid = start_wp.wps_segment_uid

    all_start_wps = df[(df.ix_in_curve==0) & (df.wp_no_go==False)]
    if just_go_straight:
        all_start_wps = all_start_wps[all_start_wps.wp_curve_id==2]

    n_wps = 0
    visited_sections = []
    while n_wps < (route_len/WP_SPACING):
        
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

        # TODO need to refine that a bit, was taking longer. Maybe limit start wps to ends. And mark loose edges as no-go.
        # # go straight more often than turn
        # straight_wp_segments = [2,3,4] # not left or right TODO is this time consuming?
        # go_straight_options = close_start_wps[close_start_wps.wp_curve_id.isin(straight_wp_segments)]
        # if random.random() < .6 and len(go_straight_options) > 0:
        #     close_start_wps = go_straight_options

        if len(close_start_wps)==0: break
        next_wp = close_start_wps.sample(1).iloc[0]
        wps_segment_uid = next_wp.wps_segment_uid
        
    route_wps = pd.concat(segments)
    RANDOM_START_OFFSET_M_MAX = 30
    route_wps = route_wps.iloc[random.randint(0, int(RANDOM_START_OFFSET_M_MAX/WP_SPACING)):-1]
    return route_wps

def get_route(wp_df, start_locs_df=None, route_len=2000, just_go_straight=False):
    # route len in meters
    _route_len = 0
    while _route_len < route_len:
        route = _get_route(wp_df, start_locs_df, route_len, just_go_straight)
        _route_len = len(route)*WP_SPACING
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

    df["intersection_id"] = [d.value for d in wps_holder_object.data.attributes["intersection_id"].data] # .06 sec
    df["section_id"] = [d.value for d in wps_holder_object.data.attributes["section_id"].data]
    df["intersection_section_id"] = (df.intersection_id.astype(str) + df.section_id.astype(str)).astype(str)
    df["to_section"] = [d.value for d in wps_holder_object.data.attributes["to_section"].data]
    df["from_section"] = [d.value for d in wps_holder_object.data.attributes["from_section"].data]

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
    # these last two properties necessary to differentiate slips within an intersection


    df["ix_in_curve"] = [d.value for d in wps_holder_object.data.attributes["ix_in_curve"].data]

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
####################################


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
    To be run once at beginning to datagen to make sure npcs are up to date.
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
    def __init__(self, ap, nodes, blender_object):
        self.ap, self.nodes, self.blender_object = ap, nodes, blender_object
        self.dist_from_ego_start_go = random.uniform(100, 200)
        self.is_done = False

class TrafficManager():
    def __init__(self, ego_ap, wp_df, is_highway=False):
        # self.npc_nodes, self.npc_aps, self.npc_objects = [], [], []
        self.is_highway = is_highway
        self.ego = ego_ap # this is ego.ap actually
        # npcs only start along the route, not randomly scattered on map
        start_locs_df = wp_df[wp_df.intersection_id.isin(ego_ap.visited_intersections)] 
        # no NPCs right next to ego at start
        start_locs_df = start_locs_df[(start_locs_df.intersection_section_id != ego_ap.start_intersection_section_id)]
        self.wp_df = wp_df
        self.start_locs_df = start_locs_df

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

                npc_ap = Autopilot(is_highway=self.is_highway, ap_id=i, save_data=False, is_ego=False)
                route = get_route(self.wp_df, start_locs_df=self.start_locs_df, route_len=800)
                npc_ap.set_route(route)

                update_ap_object(nodes, npc_ap)

                npc = NPC(npc_ap, nodes, npc_object)
                self.npcs.append(npc)
            else:
                npc_object.hide_render = True

    def step(self):

        self.ego.set_should_yield(False)
        self.ego.has_lead = False
        self.ego.lead_dist = 1e6

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
                result = (ego_traj[:, None, :] - npc_traj[None, :, :])**2 # padding + broadcasting is forcing np to give us the matrix we want
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
                IS_LEAD_TRAJ_DIST_THRESH = .5
                overlap_ixs = np.where(dists<IS_LEAD_TRAJ_DIST_THRESH)[0] # Not perfect, would fail if vehicle was directly oncoming. Should also taking into account heading
                npc_is_ego_lead = len(overlap_ixs)>0
                if npc_is_ego_lead:
                    self.ego.has_lead = True
                    lead_dist = overlap_ixs[0] * WP_SPACING
                    self.ego.lead_dist = min(lead_dist, self.ego.lead_dist)

            # move if visible to ego
            if dist(npc.ap.current_pos[:2], self.ego.current_pos[:2]) < npc.dist_from_ego_start_go:
                npc.ap.step()
                update_ap_object(npc.nodes, npc.ap)

        self.npcs = [npc for npc in self.npcs if not npc.is_done] 


def set_frame_change_post_handler(bpy, save_data=False, run_root=None,
                                _is_highway=False, _is_lined=False, _pitch_perturbation=0, _yaw_perturbation=0,
                                has_npcs=True, _is_single_rd=True):
    global ap, tm

    make_vehicle_nodes = bpy.data.node_groups['MakeVehicle'].nodes 

    t0 = time.time()
    dg = bpy.context.evaluated_depsgraph_get()
    wps_holder_object = bpy.data.objects["wps_holder"].evaluated_get(dg)
    wp_df = get_wp_df(wps_holder_object)
    coarse_map_df = wp_df[wp_df.wps==False]
    wp_df = wp_df[wp_df.wps]
    print("get wp df", time.time() - t0)

    # single-rd. Maybe just straight maybe turns ok.
    JUST_GO_STRAIGHT_PROB = .2
    just_go_straight = random.random()<JUST_GO_STRAIGHT_PROB
    if _is_single_rd:
        s = wp_df[(wp_df.ix_in_curve==0) & (wp_df.wp_curve_id==2)] # start wps, first lane (not turning)
        if just_go_straight:
            s = s[((s.intersection_id==4) & (s.section_id.isin([1]))) | ((s.intersection_id==6) & (s.section_id.isin([7])))]
            route = get_route(wp_df, start_locs_df=s, route_len=ROUTE_LEN_M, just_go_straight=True)
        else:
            s = s[((s.intersection_id==4) & (s.section_id.isin([1, 10, 4]))) | ((s.intersection_id==6) & (s.section_id.isin([7, 4, 10])))]
            route = get_route(wp_df, start_locs_df=s, route_len=ROUTE_LEN_M, just_go_straight=False)
    else: # start anywhere, turning allowed
        route = get_route(wp_df, route_len=ROUTE_LEN_M)

    print("get route", time.time() - t0)
    ap = Autopilot(is_highway=_is_highway, pitch_perturbation=_pitch_perturbation, yaw_perturbation=_yaw_perturbation, 
                        run_root=run_root, save_data=save_data, ap_id=-1, is_ego=True, just_go_straight=just_go_straight)

    ap.set_route(route)
    ap.set_nav_map(coarse_map_df)
    update_ap_object(make_vehicle_nodes, ap)

    tm = TrafficManager(wp_df=wp_df, ego_ap=ap, is_highway=_is_highway)

    tm.add_npcs(random.randint(6, MAX_N_NPCS) if has_npcs else 0)

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




    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(frame_change_post)


