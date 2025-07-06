import bpy
import os
import math
import json
from math import atan2, sqrt

# from tqdm import tqdm


# Define the base path
base_path = "/home/jeans/progressive_img2sketch/resources/LOD_data_50"
# processed_base_path = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_model_512_60_angle_00"
# processed_base_path = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_whiteblock_512_60_30_test"
# processed_base_path = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_whiteblock_512_60_00"
# processed_base_path = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_whiteblock_512_num1_angle90_depth_map"
processed_base_path = "/home/jeans/progressive_img2sketch/resources/LOD_topdown_imgs"


def setup_render(output_path):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.image_settings.file_format = "PNG"
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


# Set the camera rotation and render
def render_views(output_folder, model_name):
    # Ensure there is a camera in the scene, if not create one
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        bpy.context.scene.camera = camera
    else:
        camera = bpy.data.objects["Camera"]

    # Calculate the bounding box size of the object to determine camera distance
    bpy.context.view_layer.update()
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if objs:
        bpy.ops.object.select_all(action="DESELECT")
        objs[0].select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        dimensions = objs[0].dimensions
        max_dimension = max(dimensions)
        # Automatically calculate the camera distance to fit the entire object in view
        distance_factor = (
            2.0  # Adjust this factor to control how much space is around the object
        )
        distance = distance_factor * max_dimension

        # Set the camera to look at the object
        # camera.location = (distance, distance, distance) # Original camera position setting参数
        camera.location = (0, 0, distance)  # Camera placed directly above the object
        direction = objs[0].location - camera.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        camera.rotation_euler = rot_quat.to_euler()

    # Enable mist for depth map rendering
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    rl = tree.nodes.new(type="CompositorNodeRLayers")
    composite = tree.nodes.new(type="CompositorNodeComposite")
    composite.location = 200, 0

    # Set up a normalization node to enhance depth map contrast
    normalize = tree.nodes.new(type="CompositorNodeNormalize")
    normalize.location = 100, 0
    links.new(rl.outputs["Mist"], normalize.inputs[0])
    links.new(normalize.outputs[0], composite.inputs["Image"])

    # # Set up a brightness/contrast node to further enhance depth map contrast
    # bright_contrast = tree.nodes.new(type="CompositorNodeBrightContrast")
    # bright_contrast.inputs['Bright'].default_value = -5.0
    # bright_contrast.inputs['Contrast'].default_value = 5.0  # Increase contrast further
    # bright_contrast.location = 300, 0
    # links.new(normalize.outputs[0], bright_contrast.inputs[0])
    # links.new(bright_contrast.outputs[0], composite.inputs['Image'])

    # # Set up a color ramp node to manually adjust the depth gradient
    # color_ramp = tree.nodes.new(type="CompositorNodeValToRGB")
    # color_ramp.location = 500, 0
    # color_ramp.color_ramp.interpolation = 'CONSTANT'
    # color_ramp.color_ramp.elements[0].position = 0.2  # Adjust positions to control contrast
    # color_ramp.color_ramp.elements[1].position = 0.8
    # links.new(bright_contrast.outputs[0], color_ramp.inputs[0])
    # links.new(color_ramp.outputs[0], composite.inputs['Image'])

    # Define camera angles
    # num_views = 16
    # num_views = 60 # Capture 60 images from different angles
    num_views = 1  # Capture only one image from the top view
    # elevation_angle = math.radians(0)  # Set elevation angle, e.g., 30 degrees
    elevation_angle = math.radians(
        90
    )  # Set elevation angle to 90 degrees for top-down view
    for i in range(num_views):
        angle = 2 * math.pi * i / num_views
        # camera.location.x = distance * math.cos(angle)
        # camera.location.y = distance * math.sin(angle)
        # camera.location.z = distance * 0.8  # Adjusted to ensure the object is fully in view

        # Adjust the camera’s height and distance based on elevation angle
        camera.location.x = distance * math.cos(angle) * math.cos(elevation_angle)
        camera.location.y = distance * math.sin(angle) * math.cos(elevation_angle)
        camera.location.z = distance * math.sin(elevation_angle)  # 调整z轴的高度

        direction = objs[0].location - camera.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        camera.rotation_euler = rot_quat.to_euler()
        bpy.context.view_layer.update()

        # Calculate azimuth and elevation
        azimuth = angle
        horizontal_distance = sqrt(camera.location.x**2 + camera.location.y**2)
        elevation = atan2(camera.location.z, horizontal_distance)

        # Calculate Rotation (R), Translation (T), Focal length (f), and Principal point (c)
        R = [list(row) for row in camera.matrix_world.to_quaternion().to_matrix()]
        T = list(camera.location)
        f = camera.data.lens
        c = (camera.data.shift_x, camera.data.shift_y)

        # # Save the current camera angle and parameters to a JSON file
        # angle_info = {
        #     "azimuth": azimuth,
        #     "elevation": elevation,
        #     "R": R,
        #     "T": T,
        #     "f": f,
        #     "c": c
        # }
        # angles_path = os.path.join(output_folder, f"{model_name}_view_{i:02d}_camera_angle.json")
        # with open(angles_path, 'w') as f:
        #     json.dump(angle_info, f)

        # Render RGB image
        output_path = os.path.join(output_folder, f"{model_name}_view_{i:02d}.png")
        setup_render(output_path)
        links.new(rl.outputs["Image"], composite.inputs["Image"])
        bpy.ops.render.render(write_still=True)

        # # Render Depth Map
        # bpy.context.scene.view_layers[0].use_pass_mist = True
        # bpy.context.scene.world.mist_settings.falloff = 'LINEAR'
        # bpy.context.scene.world.mist_settings.start = 0.1 * distance
        # bpy.context.scene.world.mist_settings.depth = distance * 1.5
        # output_path_depth = os.path.join(output_folder, f"{model_name}_view_{i:02d}_depth.png")
        # setup_render(output_path_depth)
        # links.new(rl.outputs['Mist'], composite.inputs['Image'])
        # bpy.ops.render.render(write_still=True)


# Iterate over all subdirectories and render models
for folder in os.listdir(base_path):
    if folder.isdigit() and 0 <= int(folder) <= 29:
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for lod in range(1, 5):
                obj_file_path = os.path.join(folder_path, f"lod{lod}.obj")
                if os.path.exists(obj_file_path):
                    clear_scene()
                    import_model(obj_file_path)
                    setup_lighting()
                    output_folder = os.path.join(
                        processed_base_path, folder, f"lod{lod}"
                    )
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    render_views(output_folder, f"lod{lod}")

print("Rendering completed for all models.")
