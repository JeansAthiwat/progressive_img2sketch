import bpy
import os
import math
import json
from math import atan2, sqrt

# from tqdm import tqdm


# Define the base path
base_path = "C:/aaaJAIST/progressive_img2sketch/resources/LOD_data_50"
# processed_base_path = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_model_512_60_angle_00"
# processed_base_path = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_whiteblock_512_60_30_test"
# processed_base_path = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_whiteblock_512_60_00"
# processed_base_path = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_whiteblock_512_num1_angle90_depth_map"
processed_image_base_path = "C:/aaaJAIST/progressive_img2sketch/resources/LOD_orbit_images"
processed_freestyle_base_path = "C:/aaaJAIST/progressive_img2sketch/resources/LOD_orbit_freestyles"

azimuth_step = 30
elevations = [0, 15, 30]  # in degrees

def setup_render(output_path, bg_color=(1, 1, 1)):
    bpy.context.scene.render.engine =  "BLENDER_WORKBENCH" # "CYCLES" #
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.image_settings.file_format = "PNG"
    if bg_color is not None:
        bpy.context.scene.world.color = bg_color
    else:
        bpy.context.scene.render.film_transparent = True  # Enable transparent background

    # Set world background to pure white
    if not bpy.data.worlds:
        bpy.data.worlds.new("World")
    bpy.context.scene.world = bpy.data.worlds["World"]
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes["Background"]
    bg_node.inputs[0].default_value = (1, 1, 1, 1)  # RGBA for pure white background


# Import model and apply transformations
def import_model(obj_path):
    bpy.ops.import_scene.obj(filepath=obj_path)
    bpy.context.view_layer.update()
    # Move the model to the origin
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    for obj in bpy.context.selected_objects:
        obj.location = (0, 0, 0)


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


def render_image_views(output_folder, model_name):
    # Ensure there is a camera in the scene, if not create one
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        bpy.context.scene.camera = camera
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
    max_dim = max(obj.dimensions)
    distance_factor = 2.0
    distance = distance_factor * max_dim

    # Prepare compositor for RGB (you can keep your mist/depth nodes if needed)
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)
    rl        = tree.nodes.new(type="CompositorNodeRLayers")
    composite = tree.nodes.new(type="CompositorNodeComposite")
    composite.location = 200, 0
    links = tree.links

    # We'll be rendering pure Image outputs
    links.new(rl.outputs["Image"], composite.inputs["Image"])

    # Orbit parameters
    azimuths   = range(0, 360, azimuth_step)

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

            # setup render
            output_path = os.path.join(
                output_folder,
                f"{model_name}_az{az:03d}_el{el:02d}.png"
            )
            setup_render(output_path, bg_color=None)  

            # render
            bpy.ops.render.render(write_still=True)

def render_freestyle_views(output_folder, model_name):
    import os, math, bpy

    scene      = bpy.context.scene
    view_layer = scene.view_layers["ViewLayer"]

    # 1) Render & Freestyle settings
    scene.render.engine            = 'CYCLES'
    scene.render.use_freestyle     = True
    scene.render.line_thickness_mode = 'ABSOLUTE'
    scene.render.line_thickness      = 0.4

    view_layer.use_freestyle = True
    fs = view_layer.freestyle_settings
    fs.as_render_pass     = True              # output lines as their own pass
    fs.use_view_map_cache = True              # speed up repeated renders
    fs.crease_angle       = math.radians(179) # global crease cutoff

    # 2) Clear any existing LineSets, then add one that selects silhouettes, borders & creases
    for ls in list(fs.linesets):
        fs.linesets.remove(ls)
    ls = fs.linesets.new("LineSet")
    ls.select_silhouette = True
    ls.select_border     = True
    ls.select_crease     = True
    ls.select_edge_mark  = False

    # 3) White background
    if not bpy.data.worlds:
        bpy.data.worlds.new("World")
    scene.world = bpy.data.worlds["World"]
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)

    # 4) Ensure Camera exists
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
    camera = bpy.data.objects["Camera"]
    scene.camera = camera

    # 5) Build a compositor that outputs **only** the Freestyle pass
    scene.use_nodes = True
    tree = scene.node_tree
    # clear nodes
    for n in list(tree.nodes):
        tree.nodes.remove(n)
    rl        = tree.nodes.new(type="CompositorNodeRLayers")
    composite = tree.nodes.new(type="CompositorNodeComposite")
    composite.location = (200, 0)
    # connect Freestyle pass → composite
    tree.links.new(rl.outputs["Freestyle"], composite.inputs["Image"])

    # 6) Recenter mesh & compute orbit radius
    bpy.context.view_layer.update()
    meshes = [o for o in scene.objects if o.type == "MESH"]
    if not meshes:
        return
    obj = meshes[0]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0,0,0)
    max_dim = max(obj.dimensions)
    radius  = max_dim * 2.0

    # 7) Orbit loop (30° steps, elevations 0°,15°,30°)
    for az in range(0, 360, azimuth_step):
        for el in elevations:
            rad_az = math.radians(az)
            rad_el = math.radians(el)
            x = radius * math.cos(rad_az) * math.cos(rad_el)
            y = radius * math.sin(rad_az) * math.cos(rad_el)
            z = radius * math.sin(rad_el)
            camera.location = (x,y,z)
            # aim at origin
            vec = obj.location - camera.location
            camera.rotation_euler = vec.to_track_quat('-Z','Y').to_euler()
            bpy.context.view_layer.update()

            # 8) Render
            out_path = os.path.join(
                output_folder,
                f"{model_name}_az{az:03d}_el{el:02d}.png"
            )
            setup_render(out_path, bg_color=(1,1,1))
            bpy.ops.render.render(write_still=True)



    
# Iterate over all subdirectories and render models
for folder in os.listdir(base_path):
    if folder.isdigit() and 0 <= int(folder) <= 3:
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for lod in range(1, 4):
                obj_file_path = os.path.join(folder_path, f"lod{lod}.obj")
                if os.path.exists(obj_file_path):
                    clear_scene()
                    import_model(obj_file_path)
                    setup_lighting()
                    output_image_folder = os.path.join(
                        processed_image_base_path, folder, f"lod{lod}"
                    )
                    if not os.path.exists(output_image_folder):
                        os.makedirs(output_image_folder)
                    render_image_views(output_image_folder, f"lod{lod}")
                    
                    # Render freestyle images
                    output_freestyle_folder = os.path.join(
                        processed_freestyle_base_path, folder, f"lod{lod}"
                    )
                    if not os.path.exists(output_freestyle_folder):
                        os.makedirs(output_freestyle_folder)
                    render_freestyle_views(output_freestyle_folder, f"lod{lod}")
                    
                    

print("Rendering completed for all models.")
