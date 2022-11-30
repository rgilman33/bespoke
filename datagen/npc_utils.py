
import bpy
import glob, random

def find_car_paint_materials(objects):
    car_paint_materials = []
    car_paint_objects = []
    for obj in objects:
        instance = obj.instance_collection
        instance_objects = objects if instance is None else instance.all_objects
        for instance_object in instance_objects:
            material = instance_object.active_material
            if material is None:
                continue
            name = material.name.lower()
            if "car" in name and "paint" in name:
                car_paint_materials.append(material)
                car_paint_objects.append(instance_object)
    return car_paint_materials, car_paint_objects

car_blends = glob.glob("/home/beans/Downloads/Tranportation_Pro_v4/Tranportation/data/vehicles/*/*/*lowpoly_ADJUSTED.blend")
no_include = ["Bicycle", "Plane", "Boat", "Tractor", "Train", "Tramway"]
car_blends = [f for f in car_blends if len([n for n in no_include if n in f])==0]
# car_blend = random.choice([c for c in car_blends if "Kia Xceed" in c])
car_blend = random.choice(car_blends)

with bpy.data.libraries.load(car_blend, link=False) as (data_src, data_dst):
    data_dst.collections = data_src.collections

collection = data_dst.collections[0]
instance = bpy.data.objects.new(name=collection.name, object_data=None)
instance.instance_collection = collection
instance.instance_type = 'COLLECTION'
#instance.location = location
bpy.context.scene.collection.objects.link(instance)
instance.select_set(True)
bpy.context.view_layer.objects.active = instance
instance['TP_car_file_path'] = car_blend

material_blends = glob.glob("/home/beans/Downloads/Tranportation_Pro_v4/Tranportation/data/materials/*/*/*.blend")

with bpy.data.libraries.load(random.choice(material_blends), link=False) as (data_src_material, data_dst_material):
    data_dst_material.materials = data_src_material.materials

material = data_dst_material.materials[0]

_, car_paint_objects = find_car_paint_materials(bpy.context.selected_objects)
for obj in car_paint_objects:
    obj.active_material = material