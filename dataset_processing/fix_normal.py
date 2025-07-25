import os
import numpy as np

TARGET_RADIUS = 100.0
def correct_texture_path(mtl_dir, texture_filename, obj_name):
    """
    Correct the texture file path to ensure the folder name matches the main file's folder name.
    """
    # Extract folder names from the original texture path
    parts = texture_filename.split("/")
    if len(parts) > 1:
        parts[-2] = (
            obj_name  # Replace the second-to-last part with the correct folder name
        )
    corrected_texture_filename = "/".join(parts)
    return os.path.join(mtl_dir, corrected_texture_filename)

def read_vertices(obj_path):
    """Return (N×3) array of all 'v' lines in the OBJ."""
    verts = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()[1:4]
                verts.append(list(map(float, parts)))
    return np.array(verts)

def compute_scene_scale_and_centroids(scene_dir):
    """
    For all lod*.obj in scene_dir, compute each centroid & radius,
    then return {path: centroid}, and a global scale factor.
    """
    obj_paths = sorted(
        os.path.join(scene_dir, f)
        for f in os.listdir(scene_dir)
        if f.lower().endswith('.obj')
    )
    centroids = {}
    radii = {}
    for path in obj_paths:
        verts = read_vertices(path)
        # per‐mesh centroid = midpoint of its bbox
        c = (verts.max(axis=0) + verts.min(axis=0)) / 2
        centroids[path] = c
        # per‐mesh max distance from centroid
        radii[path] = np.max(np.linalg.norm(verts - c, axis=1))
    global_max = max(radii.values()) if radii else 1.0
    scale = TARGET_RADIUS / global_max
    return centroids, scale

def normalize_obj(obj_path, centroid, scale):
    """
    Rewrites obj_path so that each v = (v - centroid) * scale.
    """
    new_lines = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vals = list(map(float, line.strip().split()[1:4]))
                v = np.array(vals)
                v2 = (v - centroid) * scale
                new_lines.append(f"v {v2[0]} {v2[1]} {v2[2]}\n")
            else:
                new_lines.append(line)
    with open(obj_path, 'w') as f:
        f.writelines(new_lines)

def fix_mtl_map_kd_path(mtl_file_path, obj_name):
    """
    Fix the texture file reference paths in the MTL file by replacing relative paths with absolute paths, and ensure the folder names match the main OBJ file.
    """
    mtl_dir = os.path.dirname(mtl_file_path)
    new_lines = []

    # Read the MTL file
    with open(mtl_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            # Find lines containing "map_Kd" and fix their paths
            if line.startswith("map_Kd"):
                texture_filename = line.split()[1]
                corrected_texture_path = correct_texture_path(
                    mtl_dir, texture_filename, obj_name
                )
                if os.path.exists(corrected_texture_path):
                    absolute_texture_path = os.path.abspath(corrected_texture_path)
                    new_lines.append(f"map_Kd {absolute_texture_path}\n")
                else:
                    new_lines.append(
                        line
                    )  # If the texture file is not found, keep the original line
            else:
                new_lines.append(line)

    # Rewrite the MTL file with updated content
    with open(mtl_file_path, "w") as file:
        file.writelines(new_lines)
        
def fix_paths_only(obj_file_path):
    """
    Your existing fix_obj_mtl_path but with normalize=False,
    so it only corrects mtllib & map_Kd paths.
    """
    obj_dir = os.path.dirname(obj_file_path)
    obj_basename = os.path.basename(obj_file_path)
    obj_name, _ = os.path.splitext(obj_basename)
    new_lines = []

    # Read the OBJ file
    with open(obj_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            # Find lines containing "mtllib" and correct their names
            if line.startswith("mtllib"):
                mtl_filename = f"{obj_name}.mtl"
                mtl_file_path = os.path.join(obj_dir, mtl_filename)
                if os.path.exists(mtl_file_path):
                    absolute_mtl_path = os.path.abspath(mtl_file_path)
                    new_lines.append(f"mtllib {absolute_mtl_path}\n")
                    fix_mtl_map_kd_path(
                        absolute_mtl_path, obj_name
                    )  # Fix texture paths in the MTL file
                else:
                    new_lines.append(
                        line
                    )  # If the MTL file is not found, keep the original line
            else:
                new_lines.append(line)

    # Rewrite the OBJ file with updated content
    with open(obj_file_path, "w") as file:
        file.writelines(new_lines)

def traverse_and_fix(root_dir):
    """
    Assumes scenes are immediate subfolders of root_dir.
    For each scene folder:
      1) fix all paths (mtl & textures)
      2) compute global centroids & scale
      3) normalize every LOD obj with same scale
    """
    for scene in os.listdir(root_dir):
        scene_dir = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_dir):
            continue

        # 1) fix .mtl references & texture paths
        for fname in os.listdir(scene_dir):
            if fname.lower().endswith('.obj'):
                fix_paths_only(os.path.join(scene_dir, fname))

        # 2) compute global scale + centroids
        centroids, scale = compute_scene_scale_and_centroids(scene_dir)

        # 3) apply uniform normalization
        for obj_path, cent in centroids.items():
            normalize_obj(obj_path, cent, scale)

        print(f"Scene '{scene}': applied uniform scale={scale:.4f}")

# Example:
traverse_and_fix("/home/athiwat/progressive_img2sketch/resources/LOD_for_icp")
