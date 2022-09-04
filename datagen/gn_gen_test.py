import bpy
from mathutils import Vector

"""
bpy.ops.curve.primitive_bezier_curve_add()
bpy.ops.object.modifier_add(type='NODES')  

curve = bpy.context.active_object

def new_GeometryNodes_group():
    ''' Create a new empty node group that can be used
        in a GeometryNodes modifier.
    '''
    node_group = bpy.data.node_groups.new('GeometryNodes', 'GeometryNodeTree')
    inNode = node_group.nodes.new('NodeGroupInput')
    inNode.outputs.new('NodeSocketGeometry', 'Geometry')
    outNode = node_group.nodes.new('NodeGroupOutput')
    outNode.inputs.new('NodeSocketGeometry', 'Geometry')
    node_group.links.new(inNode.outputs['Geometry'], outNode.inputs['Geometry'])
    inNode.location = Vector((-1.5*inNode.width, 0))
    outNode.location = Vector((1.5*outNode.width, 0))
    return node_group

# In 3.2 Adding the modifier no longer automatically creates a node group.
# This test could be done with versioning, but this approach is more general
# in case a later version of Blender goes back to including a node group.
if curve.modifiers[-1].node_group:
    node_group = curve.modifiers[-1].node_group    
else:
    node_group = new_GeometryNodes_group()
    curve.modifiers[-1].node_group = node_group

nodes = node_group.nodes
"""

cube = bpy.data.objects["Cube"]
node_group = cube.modifiers[-1].node_group

# n = node_group.nodes.new('GeometryNodeMeshToPoints')
n = node_group.nodes.new('GeometryNodeTransform')

n.inputs[3].default_value[0] = 4

n2 = node_group.nodes.new('GeometryNodeSetPosition')
n2.inputs[3].default_value[2] = 1

inNode = node_group.nodes["Group Input"]
outNode = node_group.nodes["Group Output"]

node_group.links.new(inNode.outputs['Geometry'], n.inputs['Geometry'])
node_group.links.new(n.outputs['Geometry'], n2.inputs['Geometry'])
node_group.links.new(n2.outputs['Geometry'], outNode.inputs['Geometry'])