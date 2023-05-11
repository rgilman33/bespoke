import bpy, random, glob, time, sys
import numpy as np

sys.path.append("/home/beans/bespoke")
from constants import *

STATIC_ROOT = "/home/beans/static"
TEXTURES_ROOT = f"{STATIC_ROOT}/textures"
HDRIS_ROOT = f"{STATIC_ROOT}/hdris" 
MEGASCANS_DOWNLOADED_ROOT = f"{STATIC_ROOT}/Megascans Library/Downloaded" #TODO must update this in bridge app
OPEN_IMGS_ROOT = f"{STATIC_ROOT}/open_imgs/test"

# Getting nodes and filepaths
rd_noise_nodes = bpy.data.node_groups['getRdNoise'].nodes 
npc_nodes = bpy.data.node_groups['npc'].nodes
main_map_nodes = bpy.data.node_groups['main_map'].nodes # 
z_adjustment_nodes = bpy.data.node_groups['apply_z_adjustment'].nodes
random_value_nodes = bpy.data.node_groups['getRandomValue'].nodes
meshify_lines_nodes = bpy.data.node_groups['meshify_lines'].nodes
make_vehicle_nodes = bpy.data.node_groups['MakeVehicle'].nodes
get_map_nodes = bpy.data.node_groups['get_map'].nodes
get_rds_nodes = bpy.data.node_groups['get_rds'].nodes
get_section_nodes = bpy.data.node_groups['get_section'].nodes
get_stop_line_meshes_nodes = bpy.data.node_groups['get_stop_line_meshes'].nodes


get_variables_nodes = bpy.data.node_groups['getVariables'].nodes

dirt_gravel_nodes = bpy.data.materials["Dirt Gravel"].node_tree.nodes
background_hdri_nodes = bpy.context.scene.world.node_tree.nodes
building_material = bpy.data.materials["buildings"].node_tree.nodes
npc_material = bpy.data.materials["npc"].node_tree.nodes
rdside_thing_material = bpy.data.materials["rdside_thing"].node_tree.nodes
grass_trees_material = bpy.data.materials["grass_trees"].node_tree.nodes


stopsign_material = bpy.data.materials["stopsign"].node_tree.nodes
stopsign_nodes = bpy.data.node_groups['stopsign_nodes'].nodes

rdsigns_material = bpy.data.materials["rdsigns"].node_tree.nodes
rdsigns_nodes = bpy.data.node_groups['rdsigns_nodes'].nodes

grass_material = bpy.data.materials["rdside_grass_master"].node_tree.nodes

buildings_nodes = bpy.data.node_groups['buildings_nodes'].nodes
npc_body_nodes = bpy.data.node_groups['npc_body_nodes'].nodes

all_background_hdris = glob.glob(f"{HDRIS_ROOT}/*")
all_background_hdris = [x for x in all_background_hdris if "night" not in x]

all_albedos = glob.glob(f"{MEGASCANS_DOWNLOADED_ROOT}/surface/*/*_2K_Albedo.jpg")#; print([x.split('/')[-2] for x in all_albedos])
all_normals = glob.glob(f"{MEGASCANS_DOWNLOADED_ROOT}/surface/*/*_2K_Normal.jpg")

literally_all_albedos = glob.glob(f"{MEGASCANS_DOWNLOADED_ROOT}/**/*_2K_Albedo.jpg", recursive=True)
literally_all_normals = glob.glob(f"{MEGASCANS_DOWNLOADED_ROOT}/**/*_2K_Normal.jpg", recursive=True)

img_textures = glob.glob(f"{TEXTURES_ROOT}/*.jpg") 
literally_all_albedos += img_textures

open_imgs = glob.glob(f"{OPEN_IMGS_ROOT}/*.jpg")

rd_surface_keywords = ["gravel", "rock", "concrete", "soil", "mud", "asphalt", "sand", "road"] #TODO put in snow sometimes

rd_surfaces = [s for s in all_albedos if len([k for k in rd_surface_keywords if k in s])>0]
snow_surfaces = [s for s in all_albedos if "snow" in s]
#rd_surfaces += snow_surfaces # These generally look good, no need to deal w separately. There were 14 of them vs 390 of others last time counted

print(f"{len(all_albedos)} total surfaces. {len(rd_surfaces)} appropriate for rd surface. {len(snow_surfaces)} snow surfaces.")

slow_grasses = ["houseplant_flowerless_ulrtcjsia", "plants_3d_slfpffjr", "houseplant_flowering_vgztealha", "plant_shrub_wdvhberja", "ground cover_plant_vdfnehiia"]
# /home/beans/static/Megascans Library/Downloaded/3dplant/plants_3d_slfpffjr/Var4/Var4_LOD3.fbx
# /home/beans/static/Megascans Library/Downloaded/3dplant/ground cover_plant_vdfnehiia/Var7/Var7_LOD3.fbx

plants_folders = glob.glob(f"{MEGASCANS_DOWNLOADED_ROOT}/3dplant/*")
plants_folders = [f for f in plants_folders if len([s for s in slow_grasses if s in f])==0]

def make_map(timer):

    episode_info = EpisodeInfo() 

    is_single_rd = True #random.random() < 1 
    get_node("is_single_rd", get_variables_nodes).outputs["Value"].default_value = 1 if is_single_rd else 0

    HAS_LANELINES_PROB = .7 # .8
    rd_is_lined = random.random() < HAS_LANELINES_PROB

    is_highway = rd_is_lined and random.random() < .3 # highways are faster, wider laned, always lined, no bumps, more often banked, no smaller-scale XY noise
    wide_shoulder_add = random.uniform(.2, 6) if (rd_is_lined and random.random() < .2) else 0 # no wide shoulder add when dirtgravel

    is_just_straight = random.random()<.01

    left_shift = 0
    if rd_is_lined:
        lane_width = random.uniform(3.1, 4.1) if is_highway else random.uniform(2.8, 3.9)
    else: # dirt gravel
        if random.random()<.85:
            lane_width = 4.
            if random.random() < .05: # manually forcing a bit more wide dirtgravel, as it's especially challenging
                left_shift = -.4
            else:
                left_shift = -random.uniform(.4, lane_width/2)
        else: # narrow dirt gravel
            lane_width = 2.2 #random.uniform(2.4, 4)
            left_shift = -lane_width/2

    lane_width_actual = lane_width + left_shift

    has_stops = 1 if rd_is_lined or lane_width_actual>3. else 0 # wide gravel also can have stops
    get_node("has_stops", get_variables_nodes).outputs["Value"].default_value = has_stops
    
    is_wide_laned = lane_width>3.4 or wide_shoulder_add>0

    get_node("lane_width", get_variables_nodes).outputs["Value"].default_value = lane_width
    get_node("lane_width_actual", get_variables_nodes).outputs["Value"].default_value = lane_width_actual
    print(f"lane_width_actual: {lane_width_actual}")

    get_node("has_lanelines", get_variables_nodes).outputs["Value"].default_value = 1 if rd_is_lined else 0

    ONLY_YELLOW_LINES_PROB = .2 # applied after lanelines prob
    is_only_yellow_lined = random.random() < ONLY_YELLOW_LINES_PROB and rd_is_lined and not wide_shoulder_add>0 and lane_width<3.3
    get_node("is_only_yellow_lined", meshify_lines_nodes).outputs["Value"].default_value = 1 if is_only_yellow_lined else 0

    IS_COUNTRY_MTN_PROB = .2 # interspersed very curvy with totally straight. Train slowdowns and sharp curves.
    is_country_mtn = random.random() < IS_COUNTRY_MTN_PROB and lane_width < 3.6 and not (is_highway or is_just_straight)
    just_mtn = random.random() < .1 # subset of country_mtn, curviness without the straights
    get_node("country_mtn_border_noise", rd_noise_nodes).outputs["Value"].default_value = 0 if just_mtn else 16

    # CRV is 1.85m wide. 
    # dist from back wheel to cam loc is 1.78m
    # i believe cam was ~1.45m from the ground, should measure again. These values are hardcoded in the blendfile. Cam is also hardcoded to
    # the cube in a janky way, but i believe is fine.

    # Rd roll / supereleveation
    RD_IS_BANKED_PROB = 0 if (not rd_is_lined or is_just_straight) else .7 if is_highway else .4
    rd_is_banked = random.random()<RD_IS_BANKED_PROB
    get_node("max_rd_roll", z_adjustment_nodes).outputs["Value"].default_value = random.uniform(.02, .05) if is_country_mtn else random.uniform(.1, .14) # max roll is 8 deg (.14 rad) on rural rds in WY
    get_node("max_roll_at_this_curvature", z_adjustment_nodes).outputs["Value"].default_value = random.uniform(.06, .16)
    get_node("rd_is_banked", z_adjustment_nodes).outputs["Value"].default_value = 1 if rd_is_banked else 0


    get_node("loop_random_seed_x", rd_noise_nodes).outputs["Value"].default_value = random.randint(-1e6, 1e6) # TODO different noises should get their own seeds, to increase combinations
    get_node("loop_random_seed_y", rd_noise_nodes).outputs["Value"].default_value = random.randint(-1e6, 1e6)
    get_node("loop_random_seed_z", rd_noise_nodes).outputs["Value"].default_value = random.randint(-1e6, 1e6)

    # XY noise scale one
    noise_scale_1 = random.uniform(.15, .45) # /= 100 in blender
    noise_mult_1_max = np.interp(noise_scale_1, [.15, .45], [1000, 500])
    get_node("loop_noise_scale_1", rd_noise_nodes).outputs["Value"].default_value = noise_scale_1
    nm1 = 0 if (is_just_straight or is_country_mtn) else noise_mult_1_max*random.uniform((.7 if rd_is_banked else .3), 1)
    get_node("loop_noise_mult_1", rd_noise_nodes).outputs["Value"].default_value = nm1

    # XY noise scale two
    noise_scale_2 = random.uniform(.6, 1.1) #  /= 100 in blender
    get_node("loop_noise_scale_2", rd_noise_nodes).outputs["Value"].default_value = noise_scale_2
    nm2_r = random.random()
    no_noise_2 = (nm2_r<.7 or is_highway or is_wide_laned or is_just_straight)
    nm2 = random.uniform(150,300) if is_country_mtn else 0 if no_noise_2 else random.uniform(20, 60) if nm2_r<.9 else random.uniform(60, 100)
    get_node("loop_noise_mult_2", rd_noise_nodes).outputs["Value"].default_value = nm2

    # Z noise

    # large hills
    r = random.random()
    z0_mult = 0 if r < .2 else random.uniform(10, 100) if r < .95 else random.uniform(100,200)
    get_node("loop_noise_mult_z_0", rd_noise_nodes).outputs["Value"].default_value = z0_mult

    # small rises
    z1_scale = .5 * 10**random.uniform(0, 1) # /= 100 in blender
    z1_mult_max_mult = 1 if is_country_mtn else .8
    z1_mult_max = np.interp(z1_scale, [.5, 1, 3, 5], [25, 18, 6, 3]) * z1_mult_max_mult # too much large-scale hills makes intx strange
    z1_mult_max = z1_mult_max*.3 if rd_is_banked else z1_mult_max # small scale hills w banking isn't realistic, i don't think. Pay attn. 
    get_node("loop_noise_scale_z_1", rd_noise_nodes).outputs["Value"].default_value = z1_scale
    z1_mult = 0 if random.random()<.05 else random.uniform(0, z1_mult_max/2) if random.random()<.8 else random.uniform(z1_mult_max/2, z1_mult_max)
    get_node("loop_noise_mult_z_1", rd_noise_nodes).outputs["Value"].default_value = z1_mult

    # rd bumpiness
    z2_mult = 0 if (random.random() < .5 or is_highway or rd_is_banked) else random.uniform(.01, .05) if rd_is_lined else random.uniform(.02, .1) # rd bumpiness
    get_node("loop_noise_mult_z_2", rd_noise_nodes).outputs["Value"].default_value = z2_mult

    # Rd hump
    rd_hump = random.uniform(.2, .8) 
    get_node("rd_hump", z_adjustment_nodes).outputs["Value"].default_value = rd_hump
    get_node("rd_hump_rampup", z_adjustment_nodes).outputs["Value"].default_value = 16 #random.uniform(4, 10) 

    # Camera calib
    BASE_PITCH = 89
    BASE_YAW = 180
    pitch_perturbation = random.uniform(-2, 2)
    yaw_perturbation = 0 #random.uniform(-2, 2)
    bpy.data.objects["Camera"].rotation_euler[0] = np.radians(BASE_PITCH + pitch_perturbation)
    bpy.data.objects["Camera"].rotation_euler[2] = np.radians(BASE_YAW + yaw_perturbation)

    # lanelines
    r = random.random()
    yellow_type = 'single' if r < .2 or not rd_is_lined else 'half_dashed' if r < .4 else 'double' # gravel setting want 0 add in the middle of rd
    yellow_is_double = yellow_type=="half_dashed" or yellow_type=="double"
    yellow_is_dashed = yellow_type=="half_dashed" or yellow_type=="single"
    yellow_spacing_hwidth = random.uniform(.08, .2) if yellow_type=="half_dashed" else 0 if yellow_type=="single" else random.uniform(.08, .25)

    # Setup map
    intersection_1 = get_node("intersection_1", get_map_nodes)
    intersection_2 = get_node("intersection_2", get_map_nodes)
    intersection_3 = get_node("intersection_3", get_map_nodes)

    single_rd = [intersection_1, intersection_2, intersection_3]
    for intersection in single_rd:
        set_intersection_property(intersection, "has_top_r", True if random.random()<.2 else False)

        for n in ["top_right", "top_left", "bottom_right", "bottom_left"]:
            set_intersection_property(intersection, f"{n}_has_outer_2", True if (random.random()<.02 and rd_is_lined) else False)
        
    # town
    intersection_a1 = get_node("intersection_a1", get_map_nodes)
    intersection_a2 = get_node("intersection_a2", get_map_nodes)
    intersection_a3 = get_node("intersection_a3", get_map_nodes)
    intersection_b1 = get_node("intersection_b1", get_map_nodes)
    intersection_b2 = get_node("intersection_b2", get_map_nodes)
    intersection_b3 = get_node("intersection_b3", get_map_nodes)
    intersection_c1 = get_node("intersection_c1", get_map_nodes)
    intersection_c2 = get_node("intersection_c2", get_map_nodes)
    intersection_c3 = get_node("intersection_c3", get_map_nodes)

    road_a = [intersection_a1, intersection_a2, intersection_a3]
    road_b = [intersection_b1, intersection_b2, intersection_b3]
    road_c = [intersection_c1, intersection_c2, intersection_c3]

    road_1 = [intersection_a1, intersection_b1, intersection_c1]
    road_2 = [intersection_a2, intersection_b2, intersection_c2]
    road_3 = [intersection_a3, intersection_b3, intersection_c3]

    vertical_rds = [road_a, road_b, road_c] + [single_rd]
    horizontal_rds = [road_1, road_2, road_3] + [single_rd] #TODO this is kindof confusing
    
    for rd in vertical_rds:
        set_intersection_property_all(rd, "v_has_left_turn_lane", False)
        set_intersection_property_all(rd, "v_left_has_l2", False)
        set_intersection_property_all(rd, "v_left_has_l3", False)
        set_intersection_property_all(rd, "v_right_has_l2", False)
        set_intersection_property_all(rd, "v_right_has_l3", False)
        set_intersection_property_all(rd, "v_lane_width", lane_width)
        set_intersection_property_all(rd, "v_yellow_spacing", yellow_spacing_hwidth*2) # this refers to the full width
        set_intersection_property_all(rd, "v_yellow_is_double", yellow_is_double)
        set_intersection_property_all(rd, "v_yellow_top_is_dashed", yellow_is_dashed)
        set_intersection_property_all(rd, "left_shift", left_shift) # Used for dirtgravel. Shifts lane markings and wps to the left, creates overlap in the middle.

    for rd in horizontal_rds:
        set_intersection_property_all(rd, "h_has_left_turn_lane", False)
        set_intersection_property_all(rd, "h_top_has_l2", False)
        set_intersection_property_all(rd, "h_top_has_l3", False)
        set_intersection_property_all(rd, "h_bottom_has_l2", False)
        set_intersection_property_all(rd, "h_bottom_has_l3", False)
        set_intersection_property_all(rd, "h_lane_width", lane_width)
        set_intersection_property_all(rd, "h_yellow_spacing", yellow_spacing_hwidth*2) #NOTE must be zero when dirtgravel
        set_intersection_property_all(rd, "h_yellow_is_double", yellow_is_double) # can only be single when yellow_spacing is zero. Must be single when dirtgravel
        set_intersection_property_all(rd, "h_yellow_top_is_dashed", yellow_is_dashed)
        set_intersection_property_all(rd, "left_shift", left_shift)

    timer.log("setup map")

    # end randomize appearance

    get_node("episode_seed", get_variables_nodes).outputs["Value"].default_value = random.randint(-1e6, 1e6)
    get_node("wp_spacing", get_variables_nodes).outputs["Value"].default_value = WP_SPACING

    # variable len stencils to give angles to rds. .33 and .67 would give perfect grid. .1 and .15 would give heavily skewed.
    # s1 = .33 #random.uniform(.1, .4)
    # s2 = .67 #s1 + random.uniform(.1, .4)
    # # NOTE hardcoding these to even bc was having to filter out lots of inits, and that's a bottleneck right now. 
    # # Can allow back in later if eg have sidecams, or otherwise want more sharp turns
    s1 = random.uniform(.1, .4)
    s2 = s1 + random.uniform(.1, .4)
    get_node("trisplit_loc_1", get_rds_nodes).outputs["Value"].default_value = s1
    get_node("trisplit_loc_2", get_rds_nodes).outputs["Value"].default_value = s2

    # No NPCs when too narrow gravel rds
    if not rd_is_lined and lane_width_actual < 3.0:
        bpy.data.objects["npc"].hide_render = True
        has_npcs = False
    else:
        bpy.data.objects["npc"].hide_render = False
        has_npcs = True


    episode_info.is_highway = is_highway
    episode_info.rd_is_lined = rd_is_lined
    episode_info.pitch = pitch_perturbation
    episode_info.yaw = yaw_perturbation
    episode_info.has_npcs = has_npcs
    episode_info.is_single_rd = is_single_rd
    episode_info.lane_width = lane_width_actual # left shift is zero for lined, for dirtgravel this gives effective lane width. left shift is neg
    episode_info.has_stops = has_stops
    episode_info.is_only_yellow_lined = is_only_yellow_lined
    episode_info.wide_shoulder_add = wide_shoulder_add

    return episode_info    



def randomize_appearance(timer, episode_info, run_counter):
    ############################################################
    # Randomize appearance -- nothing below changes any targets
    ############################################################

    # rdside hills
    get_node("rdside_hills_falloff_add", main_map_nodes).outputs["Value"].default_value = random.uniform(.8, 2) + episode_info.wide_shoulder_add*1.2
    get_node("rdside_hills_falloff_range", main_map_nodes).outputs["Value"].default_value = random.uniform(1.5, 10)
    get_node("rdside_hills_noise_scale", main_map_nodes).outputs["Value"].default_value = .03 * 10**random.uniform(0, 1)
    get_node("rdside_hills_noise_mult", main_map_nodes).outputs["Value"].default_value = 3 * 10**random.uniform(0, 1)

    # # gutter
    # # if wide shoulder add, no gutter
    # get_node("gutter_noise_in", main_map_nodes).outputs["Value"].default_value = 1 if rd_is_lined else 3. # needs to travel w shift to remain aligned
    # get_node("gutter_noise_mult", main_map_nodes).outputs["Value"].default_value = 0 if wide_shoulder_add>0 else random.uniform(.4, 1.6)
    # get_node("gutter_shift", main_map_nodes).outputs["Value"].default_value = random.uniform(.8, 1.5) if rd_is_lined else random.uniform(0, 1.5)
    # get_node("gutter_hwidth", main_map_nodes).outputs["Value"].default_value = random.uniform(.2, 1.2)
    # has_gutter = random.random() < .6 and not wide_shoulder_add>0 
    # get_node("gutter_depth_mult", main_map_nodes).outputs["Value"].default_value = random.uniform(.5, .8) if has_gutter else 0

    get_node("mtns_mult", main_map_nodes).outputs["Value"].default_value = random.uniform(20, 70)

    buildings_group_center_modulo = random.uniform(50, 300)
    get_node("buildings_group_center_modulo", main_map_nodes).outputs["Value"].default_value = buildings_group_center_modulo
    get_node("buildings_group_size", main_map_nodes).outputs["Value"].default_value = buildings_group_center_modulo * random.uniform(.1, .9)
    get_node("buildings_density", main_map_nodes).outputs["Value"].default_value = random.uniform(.003, .03)

    grass_group_center_modulo = random.uniform(50, 300)
    get_node("grass_group_center_modulo", main_map_nodes).outputs["Value"].default_value = grass_group_center_modulo
    get_node("grass_group_size", main_map_nodes).outputs["Value"].default_value = grass_group_center_modulo * random.uniform(.1, .9)
    get_node("grass_density", main_map_nodes).outputs["Value"].default_value = random.uniform(.04, .4)

    timer.log("randomize -- buildings, grass modulos")

    ###########################
    # Lanelines

    get_node("yellow_hwidth", meshify_lines_nodes).outputs["Value"].default_value = random.uniform(.04, .12)
    get_node("white_hwidth", meshify_lines_nodes).outputs["Value"].default_value = random.uniform(.04, .12)

    # lanelines mod 
    get_node("y_mod_period", get_section_nodes).outputs["Value"].default_value = random.uniform(8, 24) 
    get_node("y_mod_space", get_section_nodes).outputs["Value"].default_value = random.uniform(.5, .8)

    get_node("w_mod_period", get_section_nodes).outputs["Value"].default_value = random.uniform(8, 18)
    get_node("w_mod_space", get_section_nodes).outputs["Value"].default_value = random.uniform(.5, .8)


    ######################
    # Background
    ######################

    background_hdri_nodes["Environment Texture"].image.filepath = random.choice(all_background_hdris) if random.random()<.8 else random.choice(open_imgs)
    #get_node("hdri_rotation_x", background_hdri_nodes).outputs["Value"].default_value = random.uniform(-.3, .3)
    #get_node("hdri_rotation_y", background_hdri_nodes).outputs["Value"].default_value = random.uniform(-.3, .3)
    get_node("hdri_rotation_z", background_hdri_nodes).outputs["Value"].default_value = random.uniform(0, 6.28)

    get_node("hdri_hue", background_hdri_nodes).outputs["Value"].default_value = random.uniform(.45, .55)
    get_node("hdri_sat", background_hdri_nodes).outputs["Value"].default_value = random.uniform(.5, 1.5)
    get_node("hdri_brightness", background_hdri_nodes).outputs["Value"].default_value = random.uniform(.5, 1.5) #random.uniform(.3, 3.5)

    ######################
    # Lanelines, material
    ######################
    # whitelines also control stoplines

    white_lines_opacity = get_node("white_line_opacity", dirt_gravel_nodes)
    yellow_lines_opacity = get_node("yellow_line_opacity", dirt_gravel_nodes)

    if episode_info.rd_is_lined: # can consider going back down to min .2, just that sometimes too hard to see when combined w noise
        white_lines_opacity.outputs["Value"].default_value = .25 * 10**random.uniform(0, .6) # Deleting the mesh itself now when only yellow
        yellow_lines_opacity.outputs["Value"].default_value = random.uniform(.5, 1.) if episode_info.is_only_yellow_lined else .25*10**random.uniform(0, .6) # max is 1.0. less than min, sometimes just not visible, especially w aug
        # get_node("white_line_gate", dirt_gravel_nodes).mute = False
        # get_node("yellow_line_gate", dirt_gravel_nodes).mute = False
    else:
        white_lines_opacity.outputs["Value"].default_value = 0
        yellow_lines_opacity.outputs["Value"].default_value = 0
        # get_node("white_line_gate", dirt_gravel_nodes).mute = True # NOTE i'd like to do this for speed, but don't know the effects. Can test later
        # get_node("yellow_line_gate", dirt_gravel_nodes).mute = True 

    get_node("yellow_line_hue", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.45, .60)
    get_node("yellow_line_sat", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.9, 2)
    get_node("yellow_line_brightness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.5, 3.0)
    
    yellow_line_noise_mult_large_max = 4 if episode_info.is_only_yellow_lined else 20
    get_node("yellow_line_noise_mult_small", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(2, 40)
    get_node("yellow_line_noise_mult_large", dirt_gravel_nodes).outputs["Value"].default_value = 0 if random.random() < .2 else random.uniform(1, yellow_line_noise_mult_large_max)
    #TODO why don't have same mult setup for white lines?

    get_node("yellow_line_specular", dirt_gravel_nodes).outputs["Value"].default_value = 0 if random.random() < .8 else random.uniform(.2, .5)
    get_node("yellow_line_roughness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .7)
    get_node("white_line_specular", dirt_gravel_nodes).outputs["Value"].default_value = 0 if random.random() < .8 else random.uniform(.2, .5)
    get_node("white_line_roughness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .7)

    mm = random.uniform(.4, .8)
    get_node("white_line_mid", dirt_gravel_nodes).outputs["Value"].default_value = mm
    get_node("white_line_outer", dirt_gravel_nodes).outputs["Value"].default_value = mm * random.uniform(.0, .8)


    timer.log("randomize -- lanelines, backgrounds")

    ######################
    # Road / shoulder
    ######################

    rd_img_albedo = random.choice(rd_surfaces)
    shoulder_img_albedo = rd_img_albedo if random.random()<.2 else random.choice(rd_surfaces+snow_surfaces)

    get_node("rd_img_albedo", dirt_gravel_nodes).image.filepath = rd_img_albedo
    get_node("shoulder_img_albedo", dirt_gravel_nodes).image.filepath = shoulder_img_albedo #rd_img_albedo if random.random() < SHOULDER_SAME_ALBEDO_AS_RD_PROB else random.choice(all_albedos)
    print("RD IMG ALBEDO", rd_img_albedo)

    get_node("shoulder_img_normal", dirt_gravel_nodes).image.filepath = shoulder_img_albedo.replace("Albedo", "Normal")
    get_node("rd_img_normal", dirt_gravel_nodes).image.filepath = rd_img_albedo.replace("Albedo", "Normal") #random.choice(all_normals)


    ############### # all is relative to edge of rd, ie outer white line
    # inner_shoulder is the same surface as rd

    inner_shoulder_width = .25 if episode_info.is_only_yellow_lined else .8 if not episode_info.rd_is_lined else (episode_info.wide_shoulder_add*.5 + random.uniform(.2, 1.5))

    # outer shoulder is different surface from rd
    outer_shoulder_width = random.uniform(0.01, .3)

    # constant inner-shoulder width when only-yellow or dirtgravel, and narrower edge fade (.3 -> 1.5) rather than (.3, 3.0)
    rd_start_fade = inner_shoulder_width 
    rd_fade_width = .3*(10**random.uniform(0, .7)) if (episode_info.is_only_yellow_lined or not episode_info.rd_is_lined) else .3*(10**random.uniform(0, 1.))
    outer_shoulder_start_fade = rd_start_fade + outer_shoulder_width # inner + outer shoulder widths
    outer_shoulder_fade_width = random.uniform(0.1, 1.)

    get_node("rd_start_fade", dirt_gravel_nodes).outputs["Value"].default_value = rd_start_fade
    get_node("rd_fade_width", dirt_gravel_nodes).outputs["Value"].default_value = rd_fade_width
    get_node("shoulder_start_fade", dirt_gravel_nodes).outputs["Value"].default_value = outer_shoulder_start_fade
    get_node("shoulder_fade_width", dirt_gravel_nodes).outputs["Value"].default_value = outer_shoulder_fade_width 
    #################

    # small scale z noise and shift that begins where outer shoulder ends. Smaller scale and more constant than rdside hills
    get_node("terrain_top_noise_scale", main_map_nodes).outputs["Value"].default_value  = .2 * 10**random.uniform(0, 1)
    get_node("terrain_top_noise_mult", main_map_nodes).outputs["Value"].default_value  = .3 * 10**random.uniform(.4, 1)
    get_node("terrain_up_shift", main_map_nodes).outputs["Value"].default_value  = random.uniform(-.5, .8)
    get_node("terrain_up_falloff_add", main_map_nodes).outputs["Value"].default_value  = outer_shoulder_start_fade + .5 # terrain begins up at end of outer shoulder
    get_node("terrain_up_falloff_range", main_map_nodes).outputs["Value"].default_value  = random.uniform(1.0, 3.0)


    get_node("rd_normal_strength", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(0, 2.)
    get_node("shoulder_normal_strength", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(0, 5)

    get_node("rd_roughness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.3, .9)
    get_node("shoulder_roughness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.6, 1.0)

    get_node("rd_specular", dirt_gravel_nodes).outputs["Value"].default_value = 0 if random.random() < .8 else random.uniform(.0, .5)
    get_node("shoulder_specular", dirt_gravel_nodes).outputs["Value"].default_value = 0 if random.random() < .8 else random.uniform(.0, .5)

    get_node("rd_overlay_mixer", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .8)

    ##
    get_node("rd_edge_noise_scale", dirt_gravel_nodes).outputs["Value"].default_value = 2*10**random.uniform(1.0, 2.5)
    # get_node("rd_edge_noise_mult", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, 1.0) if is_only_yellow_lined else random.uniform(.3, 2.0)
    get_node("rd_edge_noise_mult", dirt_gravel_nodes).outputs["Value"].default_value = .1*(10**random.uniform(1, 2.0))

    get_node("shoulder_edge_noise_scale", dirt_gravel_nodes).outputs["Value"].default_value = 10**random.uniform(1.0, 4.0)
    get_node("shoulder_edge_noise_mult", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, 6.0)
    ##

    get_node("rd_hue", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.48, .52)
    get_node("rd_sat", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, 1.2)
    get_node("rd_brightness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.7, 1.2)

    get_node("shoulder_hue", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.48, .52)
    get_node("shoulder_sat", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, 1.4)
    get_node("shoulder_brightness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.5, 2)


    # rd overlay
    get_node("rd_overlay_s", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .5)
    if random.random() < .8:
        get_node("rd_overlay_h", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.0, 1) # can be any hue when value is low
        get_node("rd_overlay_v", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.03, .1)
    else:
        get_node("rd_overlay_h", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.03, .07) # hue must be constrained when value is high
        get_node("rd_overlay_v", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .5) 

    timer.log("randomize -- rd, terrain")

    # shadows, dapples
    is_puddles = random.random() < .2
    get_node("shadow_roughness", dirt_gravel_nodes).outputs["Value"].default_value = 0 if is_puddles else random.uniform(.1, .9)
    get_node("shadow_specular", dirt_gravel_nodes).outputs["Value"].default_value = 1 if is_puddles else 0

    is_colored = random.random() < .0 #.2 # actually thinking i don't like the model thinking bright colors in rd are ok. If bright colors, that's prob not rd...
    get_node("shadow_hue", dirt_gravel_nodes).outputs["Value"].default_value = 0 if not is_colored else random.uniform(0, 1)
    get_node("shadow_sat", dirt_gravel_nodes).outputs["Value"].default_value = 0 if not is_colored else random.uniform(1, 2)
    shadow_v = 0.0 if random.random() < .6 else 1.0 # value 0 blacks out colors anyways
    get_node("shadow_v", dirt_gravel_nodes).outputs["Value"].default_value = shadow_v

    NO_SHADOWS_PROB = .6 #.3 if rd_is_lined else .7
    has_shadows = random.random()<NO_SHADOWS_PROB
    if has_shadows:
        #get_node("shadows_gate", dirt_gravel_nodes).mute = False #NOTE for perf. Can test effects later

        shadow_strength = random.uniform(.1, .7) if shadow_v==1.0 else random.uniform(.2, 1.05)
        get_node("shadow_strength", dirt_gravel_nodes).outputs["Value"].default_value = shadow_strength
        get_node("shadow_shape_subtracter", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(-.04, .1) if shadow_v==0.0 else random.uniform(-.01, .1)
        # shadow_noise_scale = ((10**random.uniform(0, 1.0)) / 10) - .05 # .05 to .95
        shadow_noise_scale = random.uniform(.02, .85)
        get_node("shadow_noise_scale_small", dirt_gravel_nodes).outputs["Value"].default_value = shadow_noise_scale
        get_node("shadow_uv_rotate", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(0, 6.28)
        get_node("shadow_uv_stretch", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(1, np.interp(shadow_noise_scale, [.02, .85], [15, 3]))

        get_node("shadow_noise_scale_large", dirt_gravel_nodes).outputs["Value"].default_value = .01 * 10**random.uniform(0,1)
    else:
        # get_node("shadows_gate", dirt_gravel_nodes).mute = True
        get_node("shadow_strength", dirt_gravel_nodes).outputs["Value"].default_value = 0
        

    # Directionality, uv-stretching
    HAS_DIRECTIONALITY_PROB = .4 if episode_info.rd_is_lined else .8
    has_directionality = random.random() < HAS_DIRECTIONALITY_PROB
    get_node("directionality_w", dirt_gravel_nodes).outputs["Value"].default_value = episode_info.lane_width + inner_shoulder_width
    get_node("d_gate_normal", dirt_gravel_nodes).mute = not has_directionality # Directly muting these, otherwise still get slowdown. Mute node is only way to avoid slowdown, setting to zero still get slowdown.
    get_node("d_gate_albedo", dirt_gravel_nodes).mute = not has_directionality
    get_node("directionality_override", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.3, .5)
    # one overrides to just use underlying, non-directioned material. .3 is pretty directioned
    

    # Directionality, old version
    HAS_DIRECTIONALITY_PROB = .4 #.3 if rd_is_lined else .6
    has_directionality = random.random() < HAS_DIRECTIONALITY_PROB
    directionality_mult = random.uniform(.3, .8) if has_directionality else 0
    get_node("directionality_mixer", dirt_gravel_nodes).mute = not has_directionality
    get_node("directionality_mult", dirt_gravel_nodes).outputs["Value"].default_value = directionality_mult
    # get_node("directionality_maskout_noise_add", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .6)
    get_node("d1_width", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.05, .2)
    get_node("d2_width", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.05, .2)
    get_node("d3_width", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.05, .2)
    get_node("directionality_noise_mult", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .5)
    get_node("directionality_noise_scale", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .8)

    timer.log("randomize -- shadows, directionality")

    # vertex spacing.
    # Startup time perf very sensitive to this. Directionality lines only show up where are edges, so this strongly affects directionality appearance.
    #TODO should depend on if rd is lined. Why? Bc startup time? changed it to depend on directionlity mult
    get_node("rd_base_vertex_spacing", get_variables_nodes).outputs["Value"].default_value = random.uniform(.4 if directionality_mult>0 else .5, .7)

    timer.log("randomize -- vertex spacing")

    # # tiremarks
    # HAS_TIREMARKS_PROB = 0 if directionality_mult>0 else .1 if rd_is_lined else .4 # one or the other
    # get_node("tiremarks_noise_scale", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .6)
    # get_node("tiremarks_noise_mult", dirt_gravel_nodes).outputs["Value"].default_value = 0 if random.random()<.2 else random.uniform(.2, 1.)
    # get_node("tiremarks_maskout_noise_add", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(0, .4)
    # get_node("wheel_width", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.4, .9) if rd_is_lined else random.uniform(.3, .5) # using dirtgravel as proxy for narrower, bc we want narrower tiremarks when narrow rd otherwise whole rd is worn out
    # tiremarks_mult = random.uniform(.1 if rd_is_lined else .3, .8) if random.random() < HAS_TIREMARKS_PROB else 0
    # get_node("tiremarks_mult", dirt_gravel_nodes).outputs["Value"].default_value = tiremarks_mult



    ######################
    # Terrain
    ######################

    # meta_img = None
    # dataloader_root = f"{BLENDER_MEMBANK_ROOT}/dataloader_0{random.randint(0,8)}" # random dataloader
    # current_run_f = f"{dataloader_root}/run_counter.npy"
    # if os.path.exists(current_run_f):
    #     run_counter = np.load(current_run_f)[0]
    #     meta_img_paths = glob.glob(f"{dataloader_root}/run_{run_counter}/imgs/*")
    #     if len(meta_img_paths)>0:
    #         meta_img = random.choice(meta_img_paths)
    # print("meta_img path", meta_img)

    terrain_albedo = rd_img_albedo if random.random() < .6 else random.choice(all_albedos)
    get_node("terrain_img_albedo", dirt_gravel_nodes).image.filepath = random.choice(open_imgs) if random.random() < .5 else terrain_albedo
    get_node("terrain_img_normal", dirt_gravel_nodes).image.filepath = terrain_albedo.replace("Albedo", "Normal")

    get_node("terrain_hue", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(.46, .54)
    get_node("terrain_saturation", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(.4, 1.2)
    get_node("terrain_brightness", dirt_gravel_nodes).outputs["Value"].default_value  = .3 * 10**random.uniform(0, 1.)

    get_node("terrain_roughness", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(.4, 1.0)

    # turning this off for now. Don't like the unnatural colors.
    get_node("terrain_overlay_fac", dirt_gravel_nodes).outputs["Value"].default_value  = 0 # if random.random() < .6 else random.uniform(0, 1.0)
    get_node("terrain_overlay_hue", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(0, 1.0)
    get_node("terrain_overlay_sat", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(.4, 2.0)
    get_node("terrain_overlay_value", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(.1, 3.0)


    # Rotate, scale UVs
    get_node("terrain_uv_rotate", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(0, 6.28)
    get_node("shoulder_uv_rotate", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(0, 6.28)
    get_node("rd_uv_rotate", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(0, 6.28)
    get_node("rd_uv_rotate_directioned", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(0, 6.28)

    get_node("rd_uv_scale", dirt_gravel_nodes).outputs["Value"].default_value = 1 / 10**random.uniform(1,3)
    get_node("rd_uv_scale_directioned", dirt_gravel_nodes).outputs["Value"].default_value  = .05 * 10**random.uniform(0,1) 
    get_node("shoulder_uv_scale", dirt_gravel_nodes).outputs["Value"].default_value  = 1 / 10**random.uniform(1,3)
    get_node("terrain_uv_scale", dirt_gravel_nodes).outputs["Value"].default_value  = 1 / 10**random.uniform(2,3)

    timer.log("randomize -- more terrain")

    ######################
    # Stopsigns
    ######################
    get_node("red_hue", stopsign_material).outputs["Value"].default_value  = random.uniform(-.1, .07)
    get_node("red_sat", stopsign_material).outputs["Value"].default_value  = random.uniform(.7, 1.0)
    get_node("red_val", stopsign_material).outputs["Value"].default_value  = random.uniform(.3, 1.0)
    get_node("metallic", stopsign_material).outputs["Value"].default_value = 0 if random.random()<.5 else random.uniform(.0, 1.0)
    get_node("roughness", stopsign_material).outputs["Value"].default_value = .5 if random.random()<.5 else random.uniform(.0, 1.0)
    get_node("base_sat", stopsign_material).outputs["Value"].default_value = 0 if random.random()<.5 else random.uniform(.3, .8)
    get_node("base_val", stopsign_material).outputs["Value"].default_value = random.uniform(.0, .2)

    stopsign_radius = random.uniform(.4, .6) # 30 - 36 inches is rw, we're going a bit bigger here
    text_max = np.interp(stopsign_radius, [.25, .5], [.19, .37])
    get_node("stopsign_radius", stopsign_nodes).outputs["Value"].default_value = stopsign_radius
    get_node("text_size", stopsign_nodes).outputs["Value"].default_value = random.uniform(text_max*.75, text_max)
    get_node("stopsign_y_scale", stopsign_nodes).outputs["Value"].default_value = random.uniform(.8, 1.2)
    get_node("rotation_x", stopsign_nodes).outputs["Value"].default_value = random.uniform(-.05, .05)
    get_node("rotation_y", stopsign_nodes).outputs["Value"].default_value = random.uniform(-.05, .05)
    get_node("rotation_z", stopsign_nodes).outputs["Value"].default_value = random.uniform(-.9, .2)
    get_node("rotation_ccw", stopsign_nodes).outputs["Value"].default_value = random.uniform(-.2, .2)

    stop_shift_y = random.uniform(-2, 0) # neg moves it along rd closer up before stopline
    get_node("shift_y", stopsign_nodes).outputs["Value"].default_value = stop_shift_y
    get_node("shift_x", stopsign_nodes).outputs["Value"].default_value = random.uniform(-1, -stop_shift_y/2) # i like 2m out, but then more often blocked, not fair, if could make fair i'd like that
    # neg moves closer to rd. When further up, can also be further out. 

    # don't shift in stopsign object itself, do so aftewards, otherwise was keeping origin of obj as center, stopsign itself was in rd
    get_node("shift_z", stopsign_nodes).outputs["Value"].default_value = random.uniform(1.8, 3.)

    get_node("stop_line_hwidth", get_stop_line_meshes_nodes).outputs["Value"].default_value = random.uniform(.2, .5)

    ######################
    # Rdsigns
    ######################
    get_node("front_hue", rdsigns_material).outputs["Value"].default_value  = random.uniform(.1, .9)
    if random.random()<.2: # white sign black text
        sat = 0
        val = random.uniform(1,3)
        text_val = 0
    else:
        sat, val, text_val = random.uniform(.7, 1.0), random.uniform(.3, 1.0), 1.0
    get_node("front_sat", rdsigns_material).outputs["Value"].default_value  = sat
    get_node("front_val", rdsigns_material).outputs["Value"].default_value  = val
    get_node("text_val", rdsigns_material).outputs["Value"].default_value  = text_val

    get_node("metallic", rdsigns_material).outputs["Value"].default_value = 0 if random.random()<.5 else random.uniform(.0, 1.0)
    get_node("roughness", rdsigns_material).outputs["Value"].default_value = .5 if random.random()<.5 else random.uniform(.0, 1.0)
    get_node("base_sat", rdsigns_material).outputs["Value"].default_value = 0 if random.random()<.5 else random.uniform(.3, .8)
    get_node("base_val", rdsigns_material).outputs["Value"].default_value = random.uniform(.0, .2)

    radius = random.uniform(.2, .5) 
    get_node("radius", rdsigns_nodes).outputs["Value"].default_value = radius
    n_vertices = 30 if random.random()<.05 else random.randint(3,4)
    get_node("n_vertices", rdsigns_nodes).outputs["Value"].default_value = n_vertices
    if n_vertices==3:
        rotation = random.choice([np.pi/2, -np.pi/2])
    elif n_vertices==4:
        rotation = random.choice([0, np.pi/4])
    else:
        rotation = 0
    get_node("rotation", rdsigns_nodes).outputs["Value"].default_value = rotation

    x_scale = random.uniform(1, 2)
    get_node("x_scale", rdsigns_nodes).outputs["Value"].default_value = x_scale
    get_node("y_scale", rdsigns_nodes).outputs["Value"].default_value = random.uniform(1, 2)

    random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    sign_text = "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(random.randint(0, 4))])
    bpy.data.node_groups["rdsigns_nodes"].nodes["String"].string = sign_text
    text_max = np.interp(radius*x_scale, [.25, .5, 1], [.19, .37, .8])
    get_node("text_size", rdsigns_nodes).outputs["Value"].default_value = random.uniform(text_max*.7, text_max)

    timer.log("randomize -- signs")


    ######################
    # Buildings
    ######################

    get_node("building_roughness", building_material).outputs["Value"].default_value = random.uniform(.0, 1.0)
    get_node("building_specular", building_material).outputs["Value"].default_value = random.uniform(.0, 1.0)
    get_node("building_img_normal", building_material).image.filepath = random.choice(literally_all_normals)
    get_node("building_img_albedo", building_material).image.filepath = random.choice(open_imgs) #random.choice(img_textures) if random.random()<.6 else random.choice(literally_all_albedos)
    get_node("building_hue", building_material).outputs["Value"].default_value = random.uniform(.0, 1.0)
    get_node("building_sat", building_material).outputs["Value"].default_value = random.uniform(.5, 3.0)
    get_node("building_brightness", building_material).outputs["Value"].default_value = random.uniform(.5, 1.2)
    get_node("building_uv_scale", building_material).outputs["Value"].default_value = random.uniform(.5, 10)
    get_node("building_uv_rotation", building_material).outputs["Value"].default_value = random.uniform(0, 6)

    get_node("buildings_seed", buildings_nodes).outputs["Value"].default_value = random.uniform(-1e-6, 1e6)
    get_node("buildings_noise_mult", buildings_nodes).outputs["Value"].default_value = 0 if random.random() < .3 else random.uniform(.2, .6) #TODO make this so can go higher, need to prevent intersect w rd


    # roadside things

    get_node("rst_img_albedo", rdside_thing_material).image.filepath = random.choice(open_imgs)
    get_node("rst_size", main_map_nodes).outputs["Value"].default_value = random.uniform(3., 12.)
    rst_group_center_modulo = random.uniform(50, 300)
    get_node("rst_group_center_modulo", main_map_nodes).outputs["Value"].default_value = rst_group_center_modulo
    get_node("rst_group_size", main_map_nodes).outputs["Value"].default_value = rst_group_center_modulo * random.uniform(.1, .9)
    #get_node("rst_density", main_map_nodes).outputs["Value"].default_value = random.uniform(.003, .03)

    # grass trees
    
    get_node("gt_img_albedo", grass_trees_material).image.filepath = terrain_albedo

    ######################
    # NPCs
    ######################

    get_node("npc_img_normal", npc_material).image.filepath = random.choice(literally_all_normals)
    get_node("npc_img_albedo", npc_material).image.filepath = random.choice(img_textures)
    get_node("npc_hue", npc_material).outputs["Value"].default_value = random.uniform(.0, 1.0)
    get_node("npc_brightness", npc_material).outputs["Value"].default_value = random.uniform(.5, 2)
    get_node("npc_sat", npc_material).outputs["Value"].default_value = random.uniform(.1, 1)

    get_node("npc_overlay_h", npc_material).outputs["Value"].default_value = random.uniform(0, 1)
    get_node("npc_overlay_s", npc_material).outputs["Value"].default_value = random.uniform(0, 1)
    get_node("npc_overlay_v", npc_material).outputs["Value"].default_value = random.uniform(0, 1)
    get_node("npc_overlay_mixer", npc_material).outputs["Value"].default_value = 0 if random.random()<.3 else random.uniform(.0, 1)
    get_node("npc_roughness", npc_material).outputs["Value"].default_value = random.uniform(0, 1)
    get_node("npc_metallic", npc_material).outputs["Value"].default_value = random.uniform(.3, 1.)
    get_node("npc_img_scale", npc_material).outputs["Value"].default_value = .01 * 10**random.uniform(0,2)
    get_node("npc_img_shift_x", npc_material).outputs["Value"].default_value = random.uniform(-1000,1000)
    get_node("npc_img_shift_y", npc_material).outputs["Value"].default_value = random.uniform(-1000,1000)
    get_node("npc_img_rotation", npc_material).outputs["Value"].default_value = random.uniform(0,360)

    npc_body_width = random.uniform(.5, 1.1)
    get_node("body_width", npc_body_nodes).outputs["Value"].default_value = npc_body_width 
    get_node("cabin_width", npc_body_nodes).outputs["Value"].default_value = random.uniform(.5, .9)
    get_node("cabin_height", npc_body_nodes).outputs["Value"].default_value = random.uniform(.4, 1.4)
    get_node("body_height", npc_body_nodes).outputs["Value"].default_value = random.uniform(.4, 1.4)

    get_node("cabin_bottom_len", npc_body_nodes).outputs["Value"].default_value = random.uniform(1, 2)
    get_node("cabin_top_mult", npc_body_nodes).outputs["Value"].default_value = random.uniform(.3, 1.1)

    get_node("body_bottom_len", npc_body_nodes).outputs["Value"].default_value = random.uniform(3, 5)
    get_node("body_top_mult", npc_body_nodes).outputs["Value"].default_value = random.uniform(.3, 1.2)

    get_node("body_offset", npc_body_nodes).outputs["Value"].default_value = random.uniform(-.3, .3)
    get_node("cabin_offset", npc_body_nodes).outputs["Value"].default_value = random.uniform(0, .7)

    get_node("wheel_hspan", npc_body_nodes).outputs["Value"].default_value = npc_body_width * random.uniform(.7, 1.1)
    get_node("back_axle_loc", npc_body_nodes).outputs["Value"].default_value = random.uniform(1.1, 1.8)
    get_node("front_axle_shift", npc_body_nodes).outputs["Value"].default_value = random.uniform(2.4, 3)

    get_node("wheel_hwidth", npc_body_nodes).outputs["Value"].default_value = random.uniform(.1, .2)
    get_node("wheel_radius", npc_body_nodes).outputs["Value"].default_value = random.uniform(.35, .5)
    get_node("wheel_thickness", npc_body_nodes).outputs["Value"].default_value = random.uniform(-.3, -.1)
    get_node("hub_offset", npc_body_nodes).outputs["Value"].default_value = random.uniform(.5, 1.1)

    # taillights also used for headlights
    get_node("taillight_spacing", npc_body_nodes).outputs["Value"].default_value = random.uniform(.3, .9) 
    get_node("taillight_noise_w", npc_body_nodes).outputs["Value"].default_value = random.uniform(-1e6, 1e6)
    get_node("taillight_radius", npc_body_nodes).outputs["Value"].default_value = random.uniform(.1, .3)

    get_node("license_plate_width", npc_body_nodes).outputs["Value"].default_value = random.uniform(.25, .35)
    get_node("license_plate_height", npc_body_nodes).outputs["Value"].default_value = random.uniform(.12, .18)

    get_node("taillight_hue", npc_material).outputs["Value"].default_value = random.uniform(.96, 1.04) % 1.0
    get_node("taillight_sat", npc_material).outputs["Value"].default_value = random.uniform(.9, 1.1)
    get_node("taillight_value", npc_material).outputs["Value"].default_value = random.uniform(.3, 1.)
    get_node("taillight_emission_strength", npc_material).outputs["Value"].default_value = 3 if random.random()<.15 else 0

    get_node("license_plate_hue", npc_material).outputs["Value"].default_value = random.uniform(0, 1)
    get_node("license_plate_value", npc_material).outputs["Value"].default_value = random.uniform(.5, 1.)

    get_node("headlights_emission_strength", npc_material).outputs["Value"].default_value = random.uniform(2, 30) if random.random()<.15 else 0

    timer.log("randomize -- buildings, npcs")

    ######################
    # Rdside Grass
    ######################

    if run_counter%5==0: # This takes by far the longest, only doing occasionally
        for o in bpy.data.objects:
            if "grass_mesh" in o.name or "LOD" in o.name: # the latter is bc sometimes they weren't being deleted, unsure why. Should just need the first one.
                bpy.data.objects.remove(o, do_unlink=True)

        grass_root = random.choice(plants_folders)
        grass_opacity_img = glob.glob(f"{grass_root}/Textures/Atlas/*_2K_Opacity.jpg")[0]

        mesh_files = glob.glob(f"{grass_root}/**/*.fbx")
        mesh_files = [f for f in mesh_files if "LOD0" not in f]
        mesh_files = [f for f in mesh_files if "LOD1" not in f] #TODO use only higher lods. We really don't want all these
        mesh_files_lower_poly = [f for f in mesh_files if "LOD2" not in f]
        mesh_files = mesh_files_lower_poly if len(mesh_files_lower_poly) > 0 else mesh_files
        if len(mesh_files)<1: print("No meshes of high enough LOD", grass_root)
        
        grass_mesh_path = random.choice(mesh_files)
        print("Using grass mesh ", grass_mesh_path)
        timer.log("randomize -- choose grass mesh") 

        bpy.ops.import_scene.fbx(filepath=grass_mesh_path)
        bpy.context.selected_objects[0].name = "grass_mesh"
        bpy.context.selected_objects[0].hide_render = True

        get_node("grass_mesh", main_map_nodes).inputs[0].default_value = bpy.data.objects["grass_mesh"]

        bpy.data.images["GrassOpacity"].filepath = grass_opacity_img
        bpy.data.images["GrassAlbedo"].filepath = grass_opacity_img.replace("Opacity", "Albedo") if random.random() < .3 else random.choice(all_albedos)
        timer.log("randomize -- apply grass") 

    get_node("grass_hue", grass_material).outputs["Value"].default_value = random.uniform(.4, .6)
    get_node("grass_sat", grass_material).outputs["Value"].default_value = random.uniform(.5, 1.5)
    get_node("grass_brightness", grass_material).outputs["Value"].default_value = 10**random.uniform(0, .5) - .5

    get_node("base_grass_size", main_map_nodes).outputs["Value"].default_value = random.uniform(1.5, 3.3)

    timer.log("randomize -- grass") 




def set_intersection_property_all(intersections, property_name, value):
    for intersection in intersections:
        set_intersection_property(intersection, property_name, value)

def set_intersection_property(intersection, property_name, value):
    for i in range(len(intersection.inputs)):
        if intersection.inputs[i].name == property_name:
            intersection.inputs[i].default_value = value
            return
    assert False