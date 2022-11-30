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
        obj.modifiers.new("Weld", "WELD")
        
        # Not decimating body. It's very visible and after weld seems to only have 5k vertices or so
        if n_vertices > 200: # and "body" not in obj.name:
            print(f"Decimating {obj.name}. Original has {n_vertices} vertices.")
        
            if "Decimate" in [m.name for m in obj.modifiers]:
                obj.modifiers.remove(obj.modifiers.get("Decimate"))
                
            obj.modifiers.new("Decimate", "DECIMATE")
            obj.modifiers["Decimate"].ratio = .1

            # bpy.context.scene.objects.active = obj
            # bpy.ops.object.modifier_apply(modifier="Decimate")
        n_vertices_adj = len(obj.data.vertices)
        total_vertices_after_adjustment += n_vertices_adj
    
bpy.ops.wm.save_mainfile()
print(f"Vertices reduced from {total_original_vertices} to {total_vertices_after_adjustment}")



