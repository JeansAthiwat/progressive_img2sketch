import bpy
import os
import math
import json
from math import atan2, sqrt
import sys



# Parameters
RESOLUTION_X = 2048  # Set to your desired resolution
RESOLUTION_Y = 2048  # Set to your desired resolution
azimuth_step = 10
elevations = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# Now we render both mesh and freestyle, so we expect 2x the original count
expected_image_count = (360 // azimuth_step) * len(elevations) * 2

def setup_render(output_path):
    bpy.context.scene.render.engine = render_engine
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.resolution_x = RESOLUTION_X 
    bpy.context.scene.render.resolution_y = RESOLUTION_Y
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.film_transparent = True  # Enable transparent background

    # Set world background to pure white
    if not bpy.data.worlds:
        bpy.data.worlds.new("World")
    bpy.context.scene.world = bpy.data.worlds["World"]
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes["Background"]
    bg_node.inputs[0].default_value = (1, 1, 1, 1)  # RGBA for pure white background


def import_model(obj_path, limited_dissolve=False):
    bpy.ops.import_scene.obj(filepath=obj_path)
    bpy.context.view_layer.update()

    # Move the model to the origin
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    imported_objects = bpy.context.selected_objects

    for obj in imported_objects:
        obj.location = (0, 0, 0)

    # Apply limited dissolve if requested
    if limited_dissolve and imported_objects:
        obj = imported_objects[0]  # Use the first imported object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.dissolve_limited()
        bpy.ops.object.mode_set(mode='OBJECT')


# Clear default objects and remove imported models
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


# Set up lighting in the scene
def setup_lighting():
    # Remove existing lights
    for light in [obj for obj in bpy.data.objects if obj.type == "LIGHT"]:
        bpy.data.objects.remove(light, do_unlink=True)

    # Add multiple light sources to ensure good illumination
    light_locations = [(10, -10, 10), (-10, 10, 10), (10, 10, 10), (-10, -10, 10)]
    for loc in light_locations:
        bpy.ops.object.light_add(type="POINT", location=loc)
        light = bpy.context.object
        light.data.energy = (
            3000.0  # Increase light energy to ensure the scene is well-lit
        )

    # Add a sun light to provide directional lighting
    bpy.ops.object.light_add(type="SUN", location=(15, 15, 30))
    sun_light = bpy.context.object
    sun_light.data.energy = 10.0


def render_combined_views(output_folder, model_name):
    scene = bpy.context.scene
    view_layer = scene.view_layers["ViewLayer"]
    
    # Ensure there is a camera in the scene, if not create one
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        scene.camera = camera
    else:
        camera = bpy.data.objects["Camera"]

    # Compute bounding‐box size to set camera distance
    bpy.context.view_layer.update()
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        return
    obj = objs[0]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    max_dim = max(obj.dimensions)
    distance_factor = 2.0
    distance = distance_factor * max_dim

    # Setup Freestyle for line rendering
    scene.render.use_freestyle = True
    scene.render.line_thickness_mode = 'ABSOLUTE'
    scene.render.line_thickness = 0.2
    
    view_layer.use_freestyle = True
    fs = view_layer.freestyle_settings
    fs.as_render_pass = True
    fs.use_view_map_cache = False
    fs.crease_angle = math.radians(179)
    
    # Clear existing LineSets and add one for silhouettes, borders & creases
    for ls in list(fs.linesets):
        fs.linesets.remove(ls)
    ls = fs.linesets.new("LineSet")
    ls.select_silhouette = True
    ls.select_border = True
    ls.select_crease = True
    ls.select_edge_mark = False

    # Set white background
    if not bpy.data.worlds:
        bpy.data.worlds.new("World")
    scene.world = bpy.data.worlds["World"]
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)

    # Orbit parameters
    azimuths = range(0, 360, azimuth_step)

    for az in azimuths:
        for el in elevations:
            # compute spherical→cartesian camera pos
            rad_az = math.radians(az)
            rad_el = math.radians(el)
            x = distance * math.cos(rad_az) * math.cos(rad_el)
            y = distance * math.sin(rad_az) * math.cos(rad_el)
            z = distance * math.sin(rad_el)
            camera.location = (x, y, z)

            # point the camera at the object center
            direction = obj.location - camera.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()

            bpy.context.view_layer.update()

            # First render: Mesh render using WORKBENCH engine
            scene.render.engine = render_engine
            scene.use_nodes = True
            tree = scene.node_tree
            # Clear nodes
            for n in list(tree.nodes):
                tree.nodes.remove(n)
            rl = tree.nodes.new(type="CompositorNodeRLayers")
            composite = tree.nodes.new(type="CompositorNodeComposite")
            composite.location = 200, 0
            # Connect Image output for mesh render
            tree.links.new(rl.outputs["Image"], composite.inputs["Image"])
            
            mesh_output_path = os.path.join(
                output_folder,
                f"{model_name}_mesh_az{az:03d}_el{el:02d}.png"
            )
            setup_render(mesh_output_path)
            bpy.ops.render.render(write_still=True)

            # Second render: Freestyle render using CYCLES engine
            scene.render.engine = 'CYCLES'
            # Clear nodes again
            for n in list(tree.nodes):
                tree.nodes.remove(n)
            rl = tree.nodes.new(type="CompositorNodeRLayers")
            composite = tree.nodes.new(type="CompositorNodeComposite")
            composite.location = 200, 0
            # Connect Freestyle output for line render
            tree.links.new(rl.outputs["Freestyle"], composite.inputs["Image"])
            
            freestyle_output_path = os.path.join(
                output_folder,
                f"{model_name}_freestyle_az{az:03d}_el{el:02d}.png"
            )
            setup_render(freestyle_output_path)
            bpy.ops.render.render(write_still=True)


# Parse CLI args
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get args after "--"
scene_id = int(argv[0])
lod_level = int(argv[1])

# Paths
render_engine = "BLENDER_WORKBENCH"
base_path = "/home/athiwat/progressive_img2sketch/resources/LOD_data_50"
processed_base = f"/home/athiwat/progressive_img2sketch/resources/LOD50_Finer_orbit"
obj_path = os.path.join(base_path, str(scene_id), f"lod{lod_level}.obj")
output_folder = os.path.join(processed_base, str(scene_id), f"lod{lod_level}")

# Skip if already rendered
if os.path.exists(output_folder):
    existing = [f for f in os.listdir(output_folder) if f.endswith(".png")]
    if len(existing) >= expected_image_count:
        print(f"✔ Scene {scene_id} LOD {lod_level} already rendered. Skipping.")
        sys.exit(0)

# Create output dir
os.makedirs(output_folder, exist_ok=True)

clear_scene()
import_model(obj_path, limited_dissolve=True)
setup_lighting()

render_combined_views(output_folder, f"lod{lod_level}")

print(f"✅ Rendered Scene {scene_id} LOD {lod_level}")
