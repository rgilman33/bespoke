import bpy, random, glob, time, sys
import numpy as np

sys.path.append("/home/beans/bespoke")
from constants import *

STATIC_ROOT = "/home/beans/static"
TEXTURES_ROOT = f"{STATIC_ROOT}/textures"
HDRIS_ROOT = f"{STATIC_ROOT}/hdris" 
MEGASCANS_DOWNLOADED_ROOT = f"{STATIC_ROOT}/Megascans Library/Downloaded" #TODO must update this in bridge app

# Getting nodes and filepaths

rd_noise_nodes = bpy.data.node_groups['getRdNoise'].nodes #bpy.data.node_groups['GenerateLoop'].nodes
npc_nodes = bpy.data.node_groups['MakeNPC'].nodes
main_map_nodes = bpy.data.node_groups['MainMap'].nodes # 
main_map_nodes_v2 = bpy.data.node_groups['main_map'].nodes # 
z_adjustment_nodes = bpy.data.node_groups['apply_z_adjustment'].nodes
random_value_nodes = bpy.data.node_groups['getRandomValue'].nodes
get_yellow_nodes = bpy.data.node_groups['get_yellow'].nodes
meshify_lines_nodes = bpy.data.node_groups['meshifyLines'].nodes
rd_curves_nodes = bpy.data.node_groups['makeRdCurves'].nodes
make_vehicle_nodes = bpy.data.node_groups['MakeVehicle'].nodes
get_postion_along_loop_nodes = bpy.data.node_groups['GetPositionAlongLoop'].nodes
make_turnoff_nodes = bpy.data.node_groups['makeTurnoff'].nodes

get_variables_nodes = bpy.data.node_groups['getVariables'].nodes

dirt_gravel_nodes = bpy.data.materials["Dirt Gravel"].node_tree.nodes
background_hdri_nodes = bpy.context.scene.world.node_tree.nodes
building_material = bpy.data.materials["buildings"].node_tree.nodes
npc_material = bpy.data.materials["npc"].node_tree.nodes

grass_material = bpy.data.materials["rdside_grass_master"].node_tree.nodes

cube_nodes = bpy.data.node_groups['CubeNodes'].nodes

all_background_hdris = glob.glob(f"{HDRIS_ROOT}/*")
all_background_hdris = [x for x in all_background_hdris if "night" not in x]

all_albedos = glob.glob(f"{MEGASCANS_DOWNLOADED_ROOT}/surface/*/*_2K_Albedo.jpg")#; print([x.split('/')[-2] for x in all_albedos])
all_normals = glob.glob(f"{MEGASCANS_DOWNLOADED_ROOT}/surface/*/*_2K_Normal.jpg")

literally_all_albedos = glob.glob(f"{MEGASCANS_DOWNLOADED_ROOT}/**/*_2K_Albedo.jpg", recursive=True)
literally_all_normals = glob.glob(f"{MEGASCANS_DOWNLOADED_ROOT}/**/*_2K_Normal.jpg", recursive=True)

img_textures = glob.glob(f"{TEXTURES_ROOT}/*.jpg") 
literally_all_albedos += img_textures

rd_surface_keywords = ["gravel", "rock", "concrete", "soil", "mud", "asphalt", "sand", "road"]

rd_surfaces = [s for s in all_albedos if len([k for k in rd_surface_keywords if k in s])>0]

print(f"{len(all_albedos)} total surfaces. {len(rd_surfaces)} appropriate for rd surface")

plants_folders = glob.glob(f"{MEGASCANS_DOWNLOADED_ROOT}/3dplant/*")

slow_grasses = ["houseplant_flowerless_ulrtcjsia"]
# /home/beans/static/Megascans Library/Downloaded/3dplant/plants_3d_slfpffjr/Var4/Var4_LOD3.fbx

def randomize_appearance(rd_is_lined=True, lane_width=None, wide_shoulder_add=None, is_only_yellow_lined=False):
    print("Randomizing appearance")
    """ 
    Doesn't change value of underlying targets. Superficial changing of materials, distractors, etc
    """

    ######################
    # Background
    ######################

    background_hdri_nodes["Environment Texture"].image.filepath = random.choice(all_background_hdris)
    #get_node("hdri_rotation_x", background_hdri_nodes).outputs["Value"].default_value = random.uniform(-.3, .3)
    #get_node("hdri_rotation_y", background_hdri_nodes).outputs["Value"].default_value = random.uniform(-.3, .3)
    get_node("hdri_rotation_z", background_hdri_nodes).outputs["Value"].default_value = random.uniform(0, 6.28)

    get_node("hdri_hue", background_hdri_nodes).outputs["Value"].default_value = random.uniform(.45, .55)
    get_node("hdri_sat", background_hdri_nodes).outputs["Value"].default_value = random.uniform(.5, 1.5)
    get_node("hdri_brightness", background_hdri_nodes).outputs["Value"].default_value = random.uniform(.3, 2)

    ######################
    # Lanelines
    ######################

    white_lines_opacity = get_node("white_line_opacity", dirt_gravel_nodes)
    yellow_lines_opacity = get_node("yellow_line_opacity", dirt_gravel_nodes)

    if rd_is_lined:
        white_lines_opacity.outputs["Value"].default_value = random.uniform(.6, 1.2) # Deleting the mesh itself now when only yellow
        yellow_lines_opacity.outputs["Value"].default_value = random.uniform(.6, 1.2) 
    else:
        white_lines_opacity.outputs["Value"].default_value = 0
        yellow_lines_opacity.outputs["Value"].default_value = 0

    get_node("yellow_line_hue", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.45, .60)
    get_node("yellow_line_sat", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.9, 2)
    get_node("yellow_line_brightness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.5, 3.0)
    
    yellow_line_noise_mult_large_max = 4 if is_only_yellow_lined else 20
    get_node("yellow_line_noise_mult_small", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(2, 40)
    get_node("yellow_line_noise_mult_large", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(1, yellow_line_noise_mult_large_max)

    get_node("yellow_line_specular", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.4, .9)
    get_node("yellow_line_roughness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .7)

    get_node("mod_space_hwidth_perc", get_yellow_nodes).outputs["Value"].default_value = random.uniform(.5, .8)
    get_node("mod_period", get_yellow_nodes).outputs["Value"].default_value = random.uniform(8, 30)

    r = random.random()
    yellow_type = 'single' if r < .3 or not rd_is_lined else 'half_dashed' if r < .5 else 'double' # gravel setting want 0 add in the middle of rd

    if yellow_type=="single":
        get_node("bottom_is_dashed", get_yellow_nodes).outputs["Value"].default_value = 1
    elif yellow_type=="half_dashed":
        if random.random() < .5:
            get_node("top_is_dashed", get_yellow_nodes).outputs["Value"].default_value = 1
            get_node("bottom_is_dashed", get_yellow_nodes).outputs["Value"].default_value = 0
        else:
            get_node("top_is_dashed", get_yellow_nodes).outputs["Value"].default_value = 0
            get_node("bottom_is_dashed", get_yellow_nodes).outputs["Value"].default_value = 1
    elif yellow_type=="double":
        get_node("top_is_dashed", get_yellow_nodes).outputs["Value"].default_value = 0
        get_node("bottom_is_dashed", get_yellow_nodes).outputs["Value"].default_value = 0

    yellow_spacing_hwidth = random.uniform(.08, .2) if yellow_type=="half_dashed" else 0 if yellow_type=="single" else random.uniform(.08, .25)
    get_node("yellow_spacing_hwidth", get_variables_nodes).outputs["Value"].default_value = yellow_spacing_hwidth

    get_node("is_double", get_yellow_nodes).outputs["Value"].default_value = 0 if yellow_type=="single" else 1

    get_node("yellow_hwidth", meshify_lines_nodes).outputs["Value"].default_value = random.uniform(.04, .12)
    get_node("white_hwidth", meshify_lines_nodes).outputs["Value"].default_value = random.uniform(.04, .12)



    ######################
    # Road / shoulder
    ######################

    rd_img_albedo = random.choice(rd_surfaces)
    shoulder_img_albedo = random.choice(rd_surfaces)

    get_node("rd_img_albedo", dirt_gravel_nodes).image.filepath = rd_img_albedo
    get_node("shoulder_img_albedo", dirt_gravel_nodes).image.filepath = shoulder_img_albedo #rd_img_albedo if random.random() < SHOULDER_SAME_ALBEDO_AS_RD_PROB else random.choice(all_albedos)
    print("RD IMG ALBEDO", rd_img_albedo)

    get_node("shoulder_img_normal", dirt_gravel_nodes).image.filepath = shoulder_img_albedo.replace("Albedo", "Normal")
    get_node("rd_img_normal", dirt_gravel_nodes).image.filepath = rd_img_albedo.replace("Albedo", "Normal") #random.choice(all_normals)


    ###############
    # inner_shoulder is the same surface as rd
    inner_shoulder_width = .25 if is_only_yellow_lined or not rd_is_lined else (wide_shoulder_add*.5 + random.uniform(.3, .9))

    # outer shoulder is different surface from rd
    outer_shoulder_width = random.uniform(0.01, .3)

    # constant inner-shoulder width when only-yellow or dirt-gravel. 
    rd_start_fade = inner_shoulder_width 
    # relatively crisp rd-shoulder edge when no white line support
    rd_fade_width = random.uniform(0.01, .15) if is_only_yellow_lined or not rd_is_lined else random.uniform(0.01, 1.0)
    outer_shoulder_start_fade = rd_start_fade + outer_shoulder_width
    outer_shoulder_fade_width = random.uniform(0.001, 1.)

    get_node("rd_start_fade", dirt_gravel_nodes).outputs["Value"].default_value = rd_start_fade
    get_node("rd_fade_width", dirt_gravel_nodes).outputs["Value"].default_value = rd_fade_width
    get_node("shoulder_start_fade", dirt_gravel_nodes).outputs["Value"].default_value = outer_shoulder_start_fade
    get_node("shoulder_fade_width", dirt_gravel_nodes).outputs["Value"].default_value = outer_shoulder_fade_width
    #################

    get_node("rd_normal_strength", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .8)
    get_node("shoulder_normal_strength", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(2, 5)

    get_node("rd_roughness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.3, .9)
    get_node("shoulder_roughness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.6, 1.0)

    get_node("rd_specular", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.0, .1)

    get_node("rd_overlay_mixer", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(0.0, .6)

    ##
    get_node("rd_edge_noise_scale", dirt_gravel_nodes).outputs["Value"].default_value = 10**random.uniform(1.0, 4.0)
    get_node("rd_edge_noise_mult", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, 1.0) if is_only_yellow_lined else random.uniform(.3, 2.0)

    get_node("shoulder_edge_noise_scale", dirt_gravel_nodes).outputs["Value"].default_value = 10**random.uniform(1.0, 4.0)
    get_node("shoulder_edge_noise_mult", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, 6.0)
    ##

    get_node("rd_hue", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.48, .52)
    get_node("rd_sat", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, 1.2)
    get_node("rd_brightness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, 1.2)

    get_node("shoulder_hue", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.48, .52)
    get_node("shoulder_sat", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, 1.4)
    get_node("shoulder_brightness", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.5, 2)


    # rd overlay
    get_node("rd_overlay_h", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.03, .07)
    get_node("rd_overlay_s", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, .8)
    get_node("rd_overlay_v", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.05, .9)

    # shadows, dapples
    shadow_v = 0.0 if random.random() < .6 else 1.0
    get_node("shadow_v", dirt_gravel_nodes).outputs["Value"].default_value = shadow_v
    shadow_strength = random.uniform(.05, .5) if shadow_v==1.0 else random.uniform(.2, 1.0)
    get_node("shadow_strength", dirt_gravel_nodes).outputs["Value"].default_value = shadow_strength
    get_node("shadow_shape_subtracter", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(-.04, .1) if shadow_v==0.0 else random.uniform(-.01, .1)
    get_node("shadow_noise_scale_small", dirt_gravel_nodes).outputs["Value"].default_value = ((10**random.uniform(0, 1.0)) / 10) - .05 # .05 to .95

    # don't have the uv dims to do this easily, and i don't believe it was giving us much
    # get_node("halfshadow_noise_scale", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1, 1.0)
    # get_node("halfshadow_modulo", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(100, 300)
    # get_node("halfshadow_width", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(2, 20)
    # get_node("halfshadow_add", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(5, 15) * random.choice([1,-1])
    # get_node("halfshadow_inner_noise", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.3, 1.0) 


    # tiremarks
    HAS_TIREMARKS_PROB = .4 if rd_is_lined else .95
    get_node("tiremarks_noise_mult", dirt_gravel_nodes).outputs["Value"].default_value = .2 * 10**random.uniform(0,1)
    get_node("tiremarks_noise_2_mult", dirt_gravel_nodes).outputs["Value"].default_value = .2 * 10**random.uniform(0,1)
    get_node("tiremarks_maskout_noise_add", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.2 if rd_is_lined else .4, 1)
    get_node("wheel_width_span", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.6, 1.2)
    get_node("avg_wheelspan", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.9, 1.1)
    get_node("tiremarks_mult", dirt_gravel_nodes).outputs["Value"].default_value = random.uniform(.1 if rd_is_lined else .3, .8) if random.random() < HAS_TIREMARKS_PROB else 0


    ######################
    # Terrain
    ######################

    terrain_albedo = random.choice(all_albedos)
    get_node("terrain_img_albedo", dirt_gravel_nodes).image.filepath = terrain_albedo
    get_node("terrain_img_normal", dirt_gravel_nodes).image.filepath = terrain_albedo.replace("Albedo", "Normal")

    get_node("terrain_hue", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(.4, .6)
    get_node("terrain_saturation", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(.4, 1.2)
    get_node("terrain_brightness", dirt_gravel_nodes).outputs["Value"].default_value  = 10**random.uniform(0, 1.1) - .5

    get_node("terrain_roughness", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(.4, 1.0)

    get_node("terrain_overlay_fac", dirt_gravel_nodes).outputs["Value"].default_value  = 0 if random.random() < .6 else random.uniform(0, 1.0)
    get_node("terrain_overlay_hue", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(0, 1.0)
    get_node("terrain_overlay_sat", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(.4, 2.0)
    get_node("terrain_overlay_value", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(.1, 3.0)


    # Rotate, scale UVs
    get_node("terrain_uv_rotate", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(0, 6.28)
    get_node("shoulder_uv_rotate", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(0, 6.28)
    get_node("rd_uv_rotate", dirt_gravel_nodes).outputs["Value"].default_value  = random.uniform(0, 6.28)

    get_node("rd_uv_scale", dirt_gravel_nodes).outputs["Value"].default_value  = 1 / 10**random.uniform(1,3)
    get_node("shoulder_uv_scale", dirt_gravel_nodes).outputs["Value"].default_value  = 1 / 10**random.uniform(1,3)
    get_node("terrain_uv_scale", dirt_gravel_nodes).outputs["Value"].default_value  = 1 / 10**random.uniform(1,3)

    
    ######################
    # Buildings
    ######################

    get_node("building_roughness", building_material).outputs["Value"].default_value = random.uniform(.0, 1.0)
    get_node("building_specular", building_material).outputs["Value"].default_value = random.uniform(.0, 1.0)
    get_node("building_img_normal", building_material).image.filepath = random.choice(literally_all_normals)
    get_node("building_img_albedo", building_material).image.filepath = random.choice(img_textures) if random.random()<.6 else random.choice(literally_all_albedos)
    get_node("building_hue", building_material).outputs["Value"].default_value = random.uniform(.0, 1.0)
    get_node("building_sat", building_material).outputs["Value"].default_value = random.uniform(.5, 3.0)
    get_node("building_brightness", building_material).outputs["Value"].default_value = random.uniform(.5, 1.2)
    get_node("building_uv_scale", building_material).outputs["Value"].default_value = random.uniform(.5, 10)
    get_node("building_uv_rotation", building_material).outputs["Value"].default_value = random.uniform(0, 6)

    ######################
    # NPCs
    ######################

    get_node("npc_img_normal", npc_material).image.filepath = random.choice(literally_all_normals)
    get_node("npc_img_albedo", npc_material).image.filepath = random.choice(img_textures)
    get_node("npc_hue", npc_material).outputs["Value"].default_value = random.uniform(.0, 1.0)
    get_node("npc_brightness", npc_material).outputs["Value"].default_value = random.uniform(.5, 2)

    # used for both npcs and buildings
    get_node("cube_seed", cube_nodes).outputs["Value"].default_value = random.uniform(-1e-6, 1e6)

    get_node("npc_scale_x", npc_nodes).outputs["Value"].default_value = random.uniform(.8, 1.2)
    get_node("npc_scale_y", npc_nodes).outputs["Value"].default_value = random.uniform(1, 5)
    get_node("npc_scale_z", npc_nodes).outputs["Value"].default_value = random.uniform(.5, 3)
    
    print("Done randomizing")

    ######################
    # Rdside Grass
    ######################
    for o in bpy.data.objects:
        if "grass_mesh" in o.name:
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
    bpy.ops.import_scene.fbx(filepath=grass_mesh_path)
    bpy.context.selected_objects[0].name = "grass_mesh"

    grass_mesh = "grass_mesh"
    get_node("grass_mesh", main_map_nodes_v2).inputs[0].default_value = bpy.data.objects[grass_mesh]

    bpy.data.images["GrassOpacity"].filepath = grass_opacity_img
    bpy.data.images["GrassAlbedo"].filepath = grass_opacity_img.replace("Opacity", "Albedo") if random.random() < .3 else random.choice(all_albedos)

    get_node("grass_hue", grass_material).outputs["Value"].default_value = random.uniform(.4, .6)
    get_node("grass_sat", grass_material).outputs["Value"].default_value = random.uniform(.5, 1.5)
    get_node("grass_brightness", grass_material).outputs["Value"].default_value = 10**random.uniform(0, .5) - .5

    get_node("base_grass_size", main_map_nodes_v2).outputs["Value"].default_value = random.uniform(1.5, 3.3)


    ######################
    # Turnoffs
    ######################
    get_node("junction_angle", make_turnoff_nodes).outputs["Value"].default_value = random.uniform(-.5, .5)

    
def setup_map():
    """
    Changes the rd network. Is a targets-changing thing.
    """
    HAS_LANELINES_PROB = .8
    rd_is_lined = random.random() < HAS_LANELINES_PROB

    is_highway = rd_is_lined and random.random() < .3 # highways are faster, wider laned, always lined, less curvy
    wide_shoulder_add = random.uniform(.2, 6) if (rd_is_lined and random.random() < .2) else 0

    is_just_straight = random.random()<.01

    lane_width = random.uniform(3.1, 4.1) if is_highway else random.uniform(2.8, 3.9) if rd_is_lined else random.uniform(1.6, 3.3)
    get_node("lane_width", get_variables_nodes).outputs["Value"].default_value = lane_width

    get_node("has_lanelines", get_variables_nodes).outputs["Value"].default_value = 1 if rd_is_lined else 0

    ONLY_YELLOW_LINES_PROB = .2 # applied after lanelines prob
    is_only_yellow_lined = random.random() < ONLY_YELLOW_LINES_PROB and rd_is_lined and not wide_shoulder_add>0 and lane_width<3.3
    get_node("is_only_yellow_lined", meshify_lines_nodes).outputs["Value"].default_value = 1 if is_only_yellow_lined else 0

    IS_COUNTRY_MTN_PROB = .15 # interspersed very curvy with totally straight. Train slowdowns and sharp curves.
    is_country_mtn = random.random() < IS_COUNTRY_MTN_PROB and lane_width < 3.6 and not (is_highway or is_just_straight)

    get_node("has_turnoffs", rd_curves_nodes).outputs["Value"].default_value = 1 #0 if is_country_mtn else 1 # when too curvy, turnoffs interfere w rd

    lane_centerish = lane_width * .53
    vehicle_perpendicular_shift = lane_centerish if rd_is_lined else (lane_width - np.interp(lane_width, [1.5, 4.0], [1.5, 2.4]))
    get_node("perpendicular_shift", get_variables_nodes).outputs["Value"].default_value = vehicle_perpendicular_shift

    # CRV is 1.85m wide. 
    # dist from back wheel to cam loc is 1.78m
    # i believe cam was ~1.45m from the ground, should measure again. These values are hardcoded in the blendfile. Cam is also hardcoded to
    # the cube in a janky way, but i believe is fine.

    is_wide_laned = lane_width>3.3 or wide_shoulder_add>0

    get_node("loop_random_seed_x", rd_noise_nodes).outputs["Value"].default_value = random.randint(-1e6, 1e6) # TODO different noises should get their own seeds, to increase combinations
    get_node("loop_random_seed_y", rd_noise_nodes).outputs["Value"].default_value = random.randint(-1e6, 1e6)
    get_node("loop_random_seed_z", rd_noise_nodes).outputs["Value"].default_value = random.randint(-1e6, 1e6)

    # XY noise scale one
    get_node("loop_noise_scale_1", rd_noise_nodes).outputs["Value"].default_value = random.uniform(.0045, .0055)
    nm1 = 0 if (random.random()<.05 or is_just_straight or is_country_mtn) else random.uniform(70, (150 if is_wide_laned else 250))
    get_node("loop_noise_mult_1", rd_noise_nodes).outputs["Value"].default_value = nm1

    # XY noise scale two
    get_node("loop_noise_scale_2", rd_noise_nodes).outputs["Value"].default_value = random.uniform(.006, .011) if is_country_mtn else random.uniform(.01, .015) 
    nm2_r = random.random()
    nm2 = random.uniform(100,200) if is_country_mtn else 0 if (nm2_r<.8 or is_highway or is_wide_laned or is_just_straight) else random.uniform(20, 60) if nm2_r<.97 else random.uniform(60, 100)
    get_node("loop_noise_mult_2", rd_noise_nodes).outputs["Value"].default_value = nm2

    # Z noise TODO add variability in scale
    get_node("loop_noise_mult_z_0", rd_noise_nodes).outputs["Value"].default_value = random.uniform(0, 200) if random.random() < .95 else random.uniform(200,300)
    get_node("loop_noise_mult_z_1", rd_noise_nodes).outputs["Value"].default_value = random.uniform(0, 10) if random.random() < .8 else random.uniform(10,20)
    get_node("loop_noise_mult_z_2", rd_noise_nodes).outputs["Value"].default_value = random.uniform(.0, .08) if rd_is_lined else random.uniform(.0, .2)

    get_node("rd_hump", z_adjustment_nodes).outputs["Value"].default_value = random.uniform(.1, .4) 

    get_node("rdside_hills_falloff_add", main_map_nodes_v2).outputs["Value"].default_value = random.uniform(1, 6) + wide_shoulder_add*1.2
    get_node("rdside_hills_falloff_range", main_map_nodes_v2).outputs["Value"].default_value = random.uniform(1.5, 10)
    get_node("rdside_hills_noise_scale", main_map_nodes_v2).outputs["Value"].default_value = .03 * 10**random.uniform(0, 1)
    get_node("rdside_hills_noise_mult", main_map_nodes_v2).outputs["Value"].default_value = 3 * 10**random.uniform(0, 1)

    get_node("mtns_mult", main_map_nodes_v2).outputs["Value"].default_value = random.uniform(20, 70)

    buildings_group_center_modulo = random.uniform(50, 300)
    get_node("buildings_group_center_modulo", main_map_nodes_v2).outputs["Value"].default_value = buildings_group_center_modulo
    get_node("buildings_group_size", main_map_nodes_v2).outputs["Value"].default_value = buildings_group_center_modulo * random.uniform(.1, .9)
    get_node("buildings_density", main_map_nodes_v2).outputs["Value"].default_value = random.uniform(.003, .03)


    grass_group_center_modulo = random.uniform(50, 300)
    get_node("grass_group_center_modulo", main_map_nodes_v2).outputs["Value"].default_value = grass_group_center_modulo
    get_node("grass_group_size", main_map_nodes_v2).outputs["Value"].default_value = grass_group_center_modulo * random.uniform(.1, .9)
    get_node("grass_density", main_map_nodes_v2).outputs["Value"].default_value = random.uniform(.04, .4)


    # #bpy.data.objects["Camera"].location[2] = random.uniform(1.45, 1.65)*-1
    # #bpy.data.objects["Camera"].location[1] = random.uniform(-.2, 0.)
    
    BASE_PITCH = 89
    BASE_YAW = 180
    pitch_perturbation = 0 #TODO UNDO random.uniform(-2, 2)
    yaw_perturbation = 0 # TODO UNDO random.uniform(-2, 2)
    bpy.data.objects["Camera"].rotation_euler[0] = np.radians(BASE_PITCH + pitch_perturbation)
    bpy.data.objects["Camera"].rotation_euler[2] = np.radians(BASE_YAW + yaw_perturbation)

    randomize_appearance(rd_is_lined=rd_is_lined, lane_width=lane_width, wide_shoulder_add=wide_shoulder_add, is_only_yellow_lined=is_only_yellow_lined)

    get_node("episode_seed", get_variables_nodes).outputs["Value"].default_value = random.randint(-1e6, 1e6)


    # No NPCs when too narrow gravel rds
    if not rd_is_lined and lane_width < 3.2:
        bpy.data.objects["Cylinder"].hide_render = True
    else:
        bpy.data.objects["Cylinder"].hide_render = False
    # bpy.data.objects["Cone"].hide_render

    return is_highway, rd_is_lined, pitch_perturbation, yaw_perturbation
        