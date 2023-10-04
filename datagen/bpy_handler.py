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

# repeat each item in a list n times
def repeat_list(l, n):
    return [item for item in l for i in range(n)]

def get_route(wp_df, start_locs_df=None, route_len=2000, just_go_straight=False, is_ego=False):
    # route len in meters
    _route_len = 0
    counter = 0
    max_curve = np.inf
    while _route_len < route_len or max_curve > .25: # chosen manually, cuts out about 1 in ten routes, is still pretty sharp
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

    Also makes N_NPC_ARCHETYPES copies of the master npc body archetype and material
    NOTE also resetting rdsigns, prob others later. Change name to more general
    """

    # remove old npcs and bodies
    for o in bpy.data.objects:
        if len([n for n in ["npc.", "_npc_body.", "rd_signs."] if n in o.name])>0: # NPCs
            bpy.data.objects.remove(o, do_unlink=True)

    # Remove old npc materials
    for m in bpy.data.materials:
        if len([n for n in ["npc.", "rdsigns."] if n in m.name])>0:
            bpy.data.materials.remove(m, do_unlink=True)

    ###################
    # NPCs
    available_bodies = []
    # make n copies of master archetype body and material
    for i in range(N_NPC_ARCHETYPES):
        # material
        material = bpy.data.materials["npc"].copy()

        # Body
        c = bpy.data.objects["_npc_body"].copy() # copy master
        c.data = c.data.copy()
        c.modifiers["GeometryNodes"].node_group = c.modifiers["GeometryNodes"].node_group.copy()
        bpy.data.collections['npc_bodies'].objects.link(c) # link directly into collection
        c.data.materials.append(material) # doesn't actually have to be on that specific object i don't think
        get_node("npc_material_assigner", c.modifiers["GeometryNodes"].node_group.nodes).inputs[2].default_value = material # this is what actually assigns material in the geonodes
        available_bodies.append(c)

    # make n copies of master npc. Make the max num, we'll mute the others
    for i in range(MAX_N_NPCS):
        c = bpy.data.objects["npc"].copy()
        c.data = c.data.copy()
        c.modifiers["GeometryNodes"].node_group = c.modifiers["GeometryNodes"].node_group.copy()
        bpy.data.collections['npcs'].objects.link(c) # link directly into collection
        # we now have a copy of the form "npc.001" in the blender scene, which we'll grab and use later

        get_node("npc_body_assigner", c.modifiers["GeometryNodes"].node_group.nodes).inputs[0].default_value = available_bodies[i%N_NPC_ARCHETYPES]

    ###################
    # Rdsigns
    # make n copies of master archetype body and material
    for i in range(N_RDSIGN_ARCHETYPES):
        # material
        material = bpy.data.materials["rdsigns"].copy()

        # Body
        c = bpy.data.objects["rd_signs"].copy() # copy master
        c.data = c.data.copy()
        c.modifiers["GeometryNodes"].node_group = c.modifiers["GeometryNodes"].node_group.copy()
        bpy.data.collections['rdsign_bodies'].objects.link(c) # link directly into collection
        c.data.materials.append(material) # doesn't actually have to be on that specific object i don't think
        get_node("rdsign_material_assigner", c.modifiers["GeometryNodes"].node_group.nodes).inputs[2].default_value = material # this is what actually assigns material in the geonodes


class NPC():
    def __init__(self, ap, nodes, blender_object, dist_from_ego_start_go):
        self.ap, self.nodes, self.blender_object = ap, nodes, blender_object
        self.dist_from_ego_start_go = dist_from_ego_start_go
        self.is_done = False

class TrafficManager():
    def __init__(self, ego_ap, wp_df, episode_info, n_npcs=6):
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
        ONLY_ONCOMING_NPCS_PROB = .4
        self.npcs_only_oncoming = random.random() < ONLY_ONCOMING_NPCS_PROB
        self.start_locs_df = start_locs_df_oncoming if self.npcs_only_oncoming else start_locs_df

        # existing blender_objects for npcs
        self.existing_npc_objects = [o for o in bpy.data.objects if "npc." in o.name]

        self.active_npcs = []

        # Add npcs
        self.n_npcs = n_npcs
        self._add_npcs()


    def _add_npcs(self):
        # use existing objects rather than create from scratch. For perf bc don't want to be deleting and remaking all the time. And
        # there was fear of buildup of things in blender behind the scenes (eg names were incrementing each time). It's simpler to clear and remake
        # but we were also getting a substantial linear slowdown in datagen which is consistent with possible buildup. Reusing now to avoid that, despite
        # the slightly more amount of complexity
        for i,npc_object in enumerate(self.existing_npc_objects):
            if i < self.n_npcs:
                npc_object.hide_render = False
                nodes = npc_object.modifiers["GeometryNodes"].node_group.nodes

                npc_ap = Autopilot(self.episode_info, ap_id=i, is_ego=False)
                route = get_route(self.wp_df, start_locs_df=self.start_locs_df, route_len=200 if self.npcs_only_oncoming else 700)
                npc_ap.set_route(route)

                update_ap_object(nodes, npc_ap)
                # dist_from_ego_start_go = 200 if self.npcs_only_oncoming else random.uniform(110, 160)
                dist_from_ego_start_go = 130 if self.npcs_only_oncoming else random.uniform(50, 90) #TODO UNDO back to original above
                npc = NPC(npc_ap, nodes, npc_object, dist_from_ego_start_go)
                self.active_npcs.append(npc)
            else:
                npc_object.hide_render = True

    def reset_active_npcs(self):
        # has to be called after _add_npcs, only works on active_npcs
        for npc in self.active_npcs:
            npc.blender_object.hide_render = False
            npc.is_done = False
            npc.ap.reset()
            update_ap_object(npc.nodes, npc.ap)

    def step(self):

        self.ego.set_should_yield(False)
        self.ego.lead_dist = DIST_NA_PLACEHOLDER

        active_npcs = [npc for npc in self.active_npcs if not npc.is_done] 
        for npc in active_npcs:
            npc_dist_to_ego = dist(npc.ap.current_pos[:2], self.ego.current_pos[:2])

            if npc.ap.route_is_done:
                npc.blender_object.hide_render = True
                npc.is_done = True
                continue

            npc.ap.set_should_yield(False)

            if npc_dist_to_ego < TRAJ_WP_DISTS[-1]: # Only check detailed trajs if within distance TODO this dist should be dynamic based on the speeds of both parties
                # # Checking for collisions, setting yield states
                # ego_traj = self.ego.negotiation_traj[:, :2]
                # npc_traj = npc.ap.negotiation_traj[:, :2]
                # # Result is traj_len x traj_len matrix w dist btwn every point in A to every pt in B
                # result = (ego_traj[:, None, :] - npc_traj[None, :, :])**2
                # result = np.sqrt(result[:,:,0] + result[:,:,1])
                # if result.min() < CRV_WIDTH * 1.2: 
                #     argmin_ix_a, argmin_ix_b = np.unravel_index(result.argmin(), result.shape) # np magic to find argmin in 2d array. Don't fully understand.
                #     ego_dist_to_collision = argmin_ix_a * self.ego.m_per_wp
                #     npc_dist_to_collision = argmin_ix_b * npc.ap.m_per_wp
                #     if (ego_dist_to_collision > npc_dist_to_collision) or self.ego.is_in_yield_zone:
                #         self.ego.set_should_yield(True)                        
                #     else:
                #         npc.ap.set_should_yield(True) TODO UNDO candidate for semseg repro bug. Yes i believe this was it
                # when let back in, perhaps round. Also find out why was overflow. These need to be deterministic. 

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
                        
                        # hack for now, tm needs some redoing anyways. Disappearing lead if too close
                        npc_too_close = npc_dist_to_ego < 8 
                        if npc_too_close: 
                            npc.ap.route_is_done = True
                            npc.blender_object.hide_render = True # hiding immediately

            # move if visible to ego
            if npc_dist_to_ego < npc.dist_from_ego_start_go:
                npc.ap.step()
                update_ap_object(npc.nodes, npc.ap)


def check_map_has_overlapping_rds(df, episode_info):
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
    t = 3.5 if (episode_info.rd_is_lined or episode_info.has_stops) else 1.5 # only narrow gravel can be this close
    return m < t # chosen manually. Tips are 10m. This should be greater than dists btwn those pts themselves, which is currently 2m. Also using to keep intersections not too sharp.

def get_map_data(bpy, episode_info, timer):

    dg = bpy.context.evaluated_depsgraph_get()
    timer.log("get dg")

    wps_holder_object = bpy.data.objects["wps_holder"].evaluated_get(dg)
    timer.log("get wps_holder_object")

    wp_df = get_wp_df(wps_holder_object)
    timer.log("get wp df")

    coarse_map_df = wp_df[wp_df.wps==False]
    map_has_overlapping_rds = check_map_has_overlapping_rds(coarse_map_df, episode_info)
    if map_has_overlapping_rds:
        print("Map has overlapping rds. Trying again.")
        return None, None, False

    wp_df = wp_df[wp_df.wps]
    timer.log("check wp df")

    return wp_df, coarse_map_df, True

def get_ego_route(wp_df, episode_info, start_left):

    JUST_GO_STRAIGHT_PROB = .2
    just_go_straight = random.random()<JUST_GO_STRAIGHT_PROB
    episode_info.just_go_straight = just_go_straight

    s = wp_df[(wp_df.ix_in_curve==0)] # start wps

    if episode_info.just_go_straight:
        s = s[(s.wp_curve_id==2)] # only straights
        s = s[((s.intersection_id==4) & (s.section_id.isin([1]))) if start_left else ((s.intersection_id==6) & (s.section_id.isin([7])))]
    else:
        s = s[((s.intersection_id==4) & (s.section_id.isin([10, 4]))) if start_left else ((s.intersection_id==6) & (s.section_id.isin([4, 10])))] 

    route = get_route(wp_df, start_locs_df=s, route_len=ROUTE_LEN_M, just_go_straight=episode_info.just_go_straight, is_ego=True)

    return route

def _update_wp_sphere(wp_sphere_nodes, ap):
    get_node("pitch", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_rot[0]
    get_node("roll", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_rot[1]
    get_node("heading", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_rot[2]
    get_node("pos_x", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_pos[0]
    get_node("pos_y", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_pos[1]
    get_node("pos_z", wp_sphere_nodes).outputs["Value"].default_value = ap.target_wp_pos[2] + 1

def create_ap_tm(bpy, wp_df, coarse_map_df, ego_route, episode_info, timer, run_root=None, dataloader_id=None, run_id=None):
    
    ###################
    # Create AP and TM

    make_vehicle_nodes = bpy.data.node_groups['MakeVehicle'].nodes 

    ap = Autopilot(episode_info, run_root=run_root, ap_id=-1, is_ego=True, dataloader_id=dataloader_id, run_id=run_id)
    timer.log("create ap")
    ap.set_route(ego_route) # this also calls ap.reset()
    timer.log("set route")
    ap.set_nav_map(coarse_map_df)
    timer.log("set nav map")
    update_ap_object(make_vehicle_nodes, ap)
    timer.log("update ap object")

    n_npcs = random.randint(MAX_N_NPCS//4, MAX_N_NPCS) if episode_info.has_npcs else 0
    tm = TrafficManager(wp_df=wp_df, ego_ap=ap, episode_info=episode_info, n_npcs=n_npcs)
    timer.log("prep tm")

    return ap, tm

def reset_ap_tm(bpy, ap, tm):
    make_vehicle_nodes = bpy.data.node_groups['MakeVehicle'].nodes 

    ap.reset()
    update_ap_object(make_vehicle_nodes, ap)

    tm.reset_active_npcs()

def toggle_semseg(bpy, is_semseg):
    dirt_gravel_nodes = bpy.data.materials["Dirt Gravel"].node_tree.nodes
    main_map_nodes = bpy.data.node_groups['main_map'].nodes 
    npc_materials = [m.node_tree.nodes for m in bpy.data.materials if "npc." in m.name]
    # place_obstacles_nodes = bpy.data.node_groups['place_obstacles'].nodes
    background_hdri_nodes = bpy.data.scenes["Scene"].world.node_tree.nodes
    road_marking_material_nodes = bpy.data.materials["Road Markings"].node_tree.nodes

    rd_base_nodes = bpy.data.node_groups['get_rd_base'].nodes

    npc_archetypes_nodes = [o.modifiers["GeometryNodes"].node_group.nodes for o in bpy.data.objects if "_npc_body." in o.name]
    stopsign_nodes = bpy.data.node_groups['stopsign_nodes'].nodes
    stopsign_material = bpy.data.materials["stopsign"].node_tree.nodes      

    if is_semseg:
        get_node("semseg_gate", road_marking_material_nodes).mute = False
        get_node("semseg_gate", rd_base_nodes).mute = False
        get_node("semseg_gate", stopsign_material).mute = False


        get_node("voronoi_noise_gate", dirt_gravel_nodes).mute = True
        get_node("shadows_gate", dirt_gravel_nodes).mute = True

        get_node("set_distractors_semseg_mat", main_map_nodes).mute = False
        get_node("semseg_hide_billboards", main_map_nodes).outputs["Value"].default_value = 1

        # get_node("constant_rd_edge_switch", dirt_gravel_nodes).outputs["Value"].default_value = 0 #1 no longer constraining rd, bc letting lines in
        # get_node("no_rdside_npcs_switch", place_obstacles_nodes).outputs["Value"].default_value = 0 #1 now allowing rdside npcs
        get_node("hdri_strength", background_hdri_nodes).outputs["Value"].default_value = 0

        # no rdside npcs for now when bev, will need diff color
        get_node("semseg_gate", road_marking_material_nodes).mute = False

        for n in npc_archetypes_nodes: get_node("assign_semseg_mat", n).mute = False
        get_node("assign_semseg_npc_mat", main_map_nodes).mute = False

    else: # not semseg
        get_node("semseg_gate", road_marking_material_nodes).mute = True
        get_node("semseg_gate", rd_base_nodes).mute = True
        get_node("semseg_gate", stopsign_material).mute = True

        get_node("voronoi_noise_gate", dirt_gravel_nodes).mute = False
        get_node("shadows_gate", dirt_gravel_nodes).mute = False

        get_node("set_distractors_semseg_mat", main_map_nodes).mute = True
        get_node("semseg_hide_billboards", main_map_nodes).outputs["Value"].default_value = 0

        # get_node("constant_rd_edge_switch", dirt_gravel_nodes).outputs["Value"].default_value = 0
        # get_node("no_rdside_npcs_switch", place_obstacles_nodes).outputs["Value"].default_value = 0
        get_node("hdri_strength", background_hdri_nodes).outputs["Value"].default_value = 1

        for n in npc_archetypes_nodes: get_node("assign_semseg_mat", n).mute = True
        get_node("assign_semseg_npc_mat", main_map_nodes).mute = True


def save_depth(bpy, should_save):
    get_node("depth_out", bpy.data.scenes["Scene"].node_tree.nodes).mute = not should_save
    # get_node("normals_out", bpy.data.scenes["Scene"].node_tree.nodes).mute = not should_save


def toggle_bev(bpy, is_bev, pitch_perturbation=0):
    z_adj_nodes = bpy.data.node_groups['apply_z_adjustment'].nodes 
    meshify_lines_nodes = bpy.data.node_groups['meshify_lines'].nodes
    # npc_body_nodes = bpy.data.node_groups['npc_body_nodes'].nodes
    npc_archetypes_nodes = [o.modifiers["GeometryNodes"].node_group.nodes for o in bpy.data.objects if "_npc_body." in o.name]
    npc_materials = [m for m in bpy.data.materials if "npc." in m.name]

    if is_bev:
        # Orthographic bev

        # get_node("semseg_out", bpy.data.scenes["Scene"].node_tree.nodes).mute = True

        bpy.data.scenes["Scene"].render.resolution_x = 256 #BEV_WIDTH #180 
        bpy.data.scenes["Scene"].render.resolution_y = 256 #BEV_HEIGHT #120

        # bpy.data.objects["Camera"].location[0] = .27
        bpy.data.objects["Camera"].location[1] = -24 # neg to move ego more outside frame to the right
        bpy.data.objects["Camera"].location[2] = -50 # move in the air

        bpy.data.objects["Camera"].rotation_euler[0] = np.radians(0) # facing straight down
        bpy.data.objects["Camera"].rotation_euler[1] = np.radians(180)
        bpy.data.objects["Camera"].rotation_euler[2] = np.radians(90) # rotating so ego faces horizontal to the right

        bpy.data.objects["Camera"].data.type = 'ORTHO'
        bpy.data.objects["Camera"].data.ortho_scale = 32 # this is the amount in meters that is captured. 

        get_node("markings_z_adj", z_adj_nodes).outputs["Value"].default_value = .5 
        # otherwise z fighting. This is high up, but bc ortho doesn't matter.
        get_node("constant_line_hwidth_switch", meshify_lines_nodes).outputs["Value"].default_value = 1 # constant extra-wide for better bev
        # NOTE the above doesn't matter bc not using lines when low res

        for n in npc_archetypes_nodes: get_node("npcs_to_bb", n).outputs["Value"].default_value = 1
        for n in npc_materials: n.blend_method = "BLEND"
    else:
        # Perspective egocentric

        # get_node("semseg_out", bpy.data.scenes["Scene"].node_tree.nodes).mute = False

        bpy.data.scenes["Scene"].render.resolution_x = 1440 
        bpy.data.scenes["Scene"].render.resolution_y = 360
        # bpy.data.objects["Camera"].data.clip_end = 1000 

        bpy.data.objects["Camera"].location[0] = .27
        bpy.data.objects["Camera"].location[1] = 0
        bpy.data.objects["Camera"].location[2] = -.97
        
        # Camera calib
        # NOTE a bit of dr here, would rather it consolidated in episode, but this allows for all camera control in one place 
        # so can toggle bev / perspective
        bpy.data.objects["Camera"].rotation_euler[0] = np.radians(BASE_PITCH + pitch_perturbation)
        bpy.data.objects["Camera"].rotation_euler[1] = np.radians(180)
        bpy.data.objects["Camera"].rotation_euler[2] = np.radians(BASE_YAW)

        bpy.data.objects["Camera"].data.type = 'PERSP'
        bpy.data.objects["Camera"].data.angle = np.radians(71)

        get_node("markings_z_adj", z_adj_nodes).outputs["Value"].default_value = .005
        get_node("constant_line_hwidth_switch", meshify_lines_nodes).outputs["Value"].default_value = 0

        for n in npc_archetypes_nodes: get_node("npcs_to_bb", n).outputs["Value"].default_value = 0
        for n in npc_materials: n.blend_method = "OPAQUE"



def reset_scene(bpy, _ap, _tm, timer=None, save_data=False, render_filepath=None):
    
    ###################
    # Reset scene 
    bpy.app.handlers.frame_change_post.clear() # has to be before we manually set the frame number. Not true i don't think w our new method.
    if timer is not None: timer.log("clear handler")

    bpy.data.scenes["Scene"].render.image_settings.file_format = 'JPEG' #"AVI_JPEG"
    bpy.data.scenes["Scene"].render.image_settings.quality = 100 #random.randint(50, 100) # zero to 100. Default 50. Going to 30 didn't speed up anything, but we're prob io bound now so test again later when using ramdisk

    # bpy.data.scenes["Scene"].render.image_settings.file_format = 'PNG'
    # bpy.data.scenes["Scene"].render.image_settings.color_mode = 'RGBA'
    # bpy.data.scenes["Scene"].render.image_settings.compression = 0

    # set_bev_cam(bpy)

    # Render samples slows datagen down linearly.
    # Too low and get aliasing around edges, harsh looking. More is softer. We're keeping low samples, trying to make up for it 
    # in data aug w blur and other distractors
    bpy.data.scenes["Scene"].eevee.taa_render_samples = random.randint(2, 5) 

    if render_filepath is not None: bpy.data.scenes["Scene"].render.filepath = render_filepath
    frame_start = 1
    bpy.data.scenes["Scene"].frame_start = frame_start
    bpy.data.scenes["Scene"].frame_end = EPISODE_LEN
    bpy.data.scenes["Scene"].render.fps = 20 // FRAME_CAPTURE_N
    if timer is not None: timer.log("set scene params")

    # bpy.data.scenes["Scene"].frame_set(frame_start) # Handler has to be cleared before this, otherwise triggers frame_change_post
    bpy.data.scenes["Scene"].frame_current = frame_start 
    # i think this one doesn't call the handlers, do other stuff. Much faster but i believe is just deferring the time cost
    # until later, ie first call on the new map
    if timer is not None: timer.log("set frame")

    ###################
    # Attach handler
    global ap, tm
    ap, tm = _ap, _tm
    # debugging
    wp_sphere_nodes = bpy.data.node_groups['wp_sphere_nodes'].nodes 
    # _update_wp_sphere(wp_sphere_nodes, ap)

    make_vehicle_nodes = bpy.data.node_groups['MakeVehicle'].nodes 
    if timer is not None: timer.log("get wp and vehicle nodes")

    def frame_change_post(scene, dg):
        # Update ego location and traj targets -> update TM (lead targets) -> save targets/aux/info data -> the frame is rendered
        global ap, tm

        # Update ego location. All traj targets will be current now.
        for i in range(FRAME_CAPTURE_N):
            # print("frame_counter", ap.overall_frame_counter, "scene.frame_current", scene.frame_current)
            ap.step()
            update_ap_object(make_vehicle_nodes, ap)

        # debugging
        _update_wp_sphere(wp_sphere_nodes, ap)

        # Move NPCs and update lead status
        for i in range(FRAME_CAPTURE_N):
            tm.step()

        # Save data
        if save_data: ap.save_data() # save data after tm steps to get correct lead status

        if scene.frame_current == scene.frame_end:
            distance_travelled = ap.current_wp_ix * WP_SPACING
            print(f"AP travelled {round(distance_travelled)} out of {round(len(ap.waypoints)*WP_SPACING)} meters in the route")
            print(f"AP route had finished: {ap.route_is_done}")

    assert len(bpy.app.handlers.frame_change_post)==0, "frame change post handler needs to be cleared"
    bpy.app.handlers.frame_change_post.append(frame_change_post) 
    if timer is not None: timer.log("attach handler")

    # Imgs is the expected amount, eg goes from 1 - 170 whereas others go to 171. We're just throwing the last one out.
    

