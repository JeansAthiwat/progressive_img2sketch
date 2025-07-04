import os
import trimesh
import pyrender
import numpy as np
import imageio

# --- Configuration ---
dataset_root = r"/home/jeans/progressive_img2sketch/rtsc-1.6/LOD_data_50"  # Corrected path for Linux
output_root = r"/home/jeans/progressive_img2sketch/rtsc-1.6/model_images"  # Corrected path for Linux

# Create output root directory if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# Define camera parameters
camera_angles_azimuth = np.arange(0, 360, 20)  # Azimuth angles (around Y-axis)
camera_elevations = np.arange(0, 30, 20)  # Elevation angles (up/down from horizontal)


# --- Helper Function for Camera Setup ---
def setup_camera(mesh, angle_azimuth, angle_elevation):
    # Calculate bounding box of the mesh
    bounds = mesh.bounds  # Min and max corners of the bounding box
    center = mesh.centroid  # Center of the mesh

    # Determine a suitable radius for the camera to view the whole model
    max_extent = np.max(np.linalg.norm(bounds - center, axis=1))
    view_distance_multiplier = 2.0  # Adjust this to zoom in/out
    radius = max_extent * view_distance_multiplier

    # Convert angles to radians
    azimuth_rad = np.radians(angle_azimuth)
    elevation_rad = np.radians(angle_elevation)

    # Calculate camera position in spherical coordinates (relative to the center)
    x = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = radius * np.sin(elevation_rad)
    z = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)

    camera_position = center + np.array([x, y, z])

    # Construct the view matrix (world to camera) manually
    forward = center - camera_position
    forward = forward / np.linalg.norm(forward)  # Normalize

    up_vec = np.array([0.0, 1.0, 0.0])  # World up vector (assuming Y-up)

    right = np.cross(forward, up_vec)
    # Handle cases where forward and up_vec are collinear (e.g., looking straight up/down)
    if (
        np.linalg.norm(right) < 1e-6
    ):  # if right vector is near zero, meaning forward is along Y-axis
        if forward[1] > 0:  # Looking straight up
            right = np.array([1.0, 0.0, 0.0])  # Arbitrary right vector
        else:  # Looking straight down
            right = np.array([-1.0, 0.0, 0.0])  # Arbitrary right vector

    right = right / np.linalg.norm(right)  # Normalize

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)  # Re-orthogonalize and normalize

    # The view matrix (world_to_camera)
    view_matrix = np.array(
        [
            [right[0], right[1], right[2], -np.dot(right, camera_position)],
            [up[0], up[1], up[2], -np.dot(up, camera_position)],
            [
                -forward[0],
                -forward[1],
                -forward[2],
                np.dot(forward, camera_position),
            ],  # Camera looks down its -Z axis
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Pyrender expects a camera_to_world matrix (pose matrix), so we invert the view matrix.
    camera_pose = np.linalg.inv(view_matrix)

    # Create a pyrender camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    # --- CHANGE HERE: Set the pose directly on the Node during creation ---
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)

    return camera_node  # Now only return the camera_node, as it contains its pose


# --- Main Script Logic ---
for scene_folder_name in sorted(
    os.listdir(dataset_root)
):  # Sort to ensure consistent order
    scene_folder_path = os.path.join(dataset_root, scene_folder_name)

    if not os.path.isdir(scene_folder_path) or not scene_folder_name.isdigit():
        # Skip if it's not a directory or not a numbered scene folder
        continue

    print(f"Processing scene: {scene_folder_name}")

    # Create output directory for the current scene
    scene_output_path = os.path.join(output_root, scene_folder_name)
    os.makedirs(scene_output_path, exist_ok=True)

    lod_meshes = {
        "lod1": os.path.join(scene_folder_path, "lod1.obj"),
        "lod2": os.path.join(scene_folder_path, "lod2.obj"),
        "lod3": os.path.join(scene_folder_path, "lod3.obj"),
        "lod4": os.path.join(scene_folder_path, "lod4.obj"),
    }

    # Initialize renderer outside the inner loop for efficiency if viewports are constant
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

    for lod_name, lod_mesh_path in lod_meshes.items():
        if not os.path.exists(lod_mesh_path):
            print(f"  Skipping {lod_name}: {lod_mesh_path} not found.")
            continue

        print(f"  Loading {lod_name} from {lod_mesh_path}")
        try:
            mesh = trimesh.load(lod_mesh_path)
            if isinstance(mesh, trimesh.scene.Scene):
                combined_meshes = []
                for geo_name, geo_obj in mesh.geometry.items():
                    if isinstance(geo_obj, trimesh.Trimesh):
                        combined_meshes.append(geo_obj)
                    elif isinstance(
                        geo_obj, trimesh.scene.Scene
                    ):  # Handle nested scenes
                        for sub_geo_name, sub_geo_obj in geo_obj.geometry.items():
                            if isinstance(sub_geo_obj, trimesh.Trimesh):
                                combined_meshes.append(sub_geo_obj)

                if combined_meshes:
                    mesh = trimesh.util.concatenate(tuple(combined_meshes))
                else:
                    print(
                        f"  Warning: {lod_mesh_path} is a Scene but contains no renderable meshes. Skipping."
                    )
                    continue

            if (
                not mesh.is_empty
                and mesh.vertices.shape[0] > 0
                and mesh.faces.shape[0] > 0
            ):
                pass
            else:
                print(
                    f"  Warning: {lod_mesh_path} is an empty or invalid mesh. Skipping."
                )
                continue

        except Exception as e:
            print(f"  Error loading {lod_mesh_path}: {e}")
            continue

        # Create output directory for the current LOD
        lod_output_path = os.path.join(scene_output_path, lod_name)
        os.makedirs(lod_output_path, exist_ok=True)

        # ... (rest of the script)

        for angle_azimuth in camera_angles_azimuth:
            for angle_elevation in camera_elevations:
                print(
                    f"    Rendering {lod_name} at angle {angle_azimuth}, elevation {angle_elevation}"
                )

                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])

                # --- CHANGE HERE: Remove the explicit material creation and assignment ---
                # This will allow pyrender to use the textures loaded by trimesh if available
                mesh_node = pyrender.Mesh.from_trimesh(mesh)

                scene.add(mesh_node)

                # --- Lighting for Canny Edge Contrast --- (keep this as is for good contrast)
                light_color = np.array([1.0, 1.0, 1.0])

                key_light_intensity = 20.0
                key_light_pose = np.eye(4)
                key_light_pose[:3, 3] = np.array([15.0, 15.0, 10.0])
                scene.add(
                    pyrender.DirectionalLight(
                        color=light_color, intensity=key_light_intensity
                    ),
                    pose=key_light_pose,
                )

                fill_light_intensity = 8.0
                fill_light_pose = np.eye(4)
                fill_light_pose[:3, 3] = np.array([-15.0, 10.0, 5.0])
                scene.add(
                    pyrender.DirectionalLight(
                        color=light_color, intensity=fill_light_intensity
                    ),
                    pose=fill_light_pose,
                )

                rim_light_intensity = 12.0
                rim_light_pose = np.eye(4)
                rim_light_pose[:3, 3] = np.array([0.0, 10.0, -20.0])
                scene.add(
                    pyrender.DirectionalLight(
                        color=light_color, intensity=rim_light_intensity
                    ),
                    pose=rim_light_pose,
                )

                camera_node = setup_camera(mesh, angle_azimuth, angle_elevation)
                scene.add_node(camera_node)

                color, depth = r.render(scene)

                image_filename = f"{angle_azimuth}_{angle_elevation}.png"
                image_filepath = os.path.join(lod_output_path, image_filename)

                imageio.imwrite(image_filepath, color)

        # ... (rest of the script)

    r.delete()  # Clean up renderer resources after all renders for the entire script run

print("Dataset processing complete!")
