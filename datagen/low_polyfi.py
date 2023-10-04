import bpy

# welding before decimating is important, especially in the tires, otherwise they
# look like hairball. 

total_original_vertices = 0
total_vertices_after_adjustment = 0
for obj in bpy.data.objects:

    if hasattr(obj.data, "vertices"):
        n_vertices = len(obj.data.vertices)
        total_original_vertices += n_vertices
        
        if "Weld" in [m.name for m in obj.modifiers]:
            obj.modifiers.remove(obj.modifiers.get("Weld"))
            
        if n_vertices > 200:
            obj.modifiers.new("Weld", "WELD")
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier="Weld")
                
        # Not decimating body. It's very visible and after weld seems to only have 5k vertices or so
        if n_vertices > 200: # and "body" not in obj.name:
            print(f"Decimating {obj.name}. Original has {n_vertices} vertices.")
        
            if "Decimate" in [m.name for m in obj.modifiers]:
                obj.modifiers.remove(obj.modifiers.get("Decimate"))
                
            obj.modifiers.new("Decimate", "DECIMATE")
            obj.modifiers["Decimate"].ratio = .3

            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier="Decimate")
            print(f"obj has {len(obj.data.vertices)} vertices after decimation")

        n_vertices_adj = len(obj.data.vertices)
        total_vertices_after_adjustment += n_vertices_adj
        
    
bpy.ops.wm.save_mainfile()
print(f"Vertices reduced from {total_original_vertices} to {total_vertices_after_adjustment}")


#https://www.google.com/maps/@45.0071184,-122.7831857,3a,75y,157.18h,89.81t/
# data=!3m7!1e1!3m5!1sRqtnc84S_KwRDjVR6qFEvA!2e0!6s
# https:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fpanoid%3DRqtnc84S_KwRDjVR6qFEvA%26cb_client%3Dmaps_sv.tactile.gps%26w%3D203%26h%3D100%26yaw%3D222.44214%26pitch%3D0%26thumbfov%3D100!7i16384!8i8192?entry=ttu


