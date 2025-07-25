import os
import numpy as np


def fix_obj_mtl_path(obj_file_path, normalize=True):
    """
    Fix the reference path to the MTL file in the OBJ file, ensuring it uses the same name as the OBJ file and uses an absolute path.
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

    # Normalize the model data in the OBJ file
    if normalize:
        normalize_obj_model(obj_file_path)


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


def normalize_obj_model(obj_file_path):
    """
    Normalize the model in the OBJ file: move the model center to the origin and scale it to a target radius, for example, 100.
    """
    vertices = []

    # Read the OBJ file and extract vertex data
    with open(obj_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("v "):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)

    # Calculate the model centroid and maximum radius
    vertices = np.array(vertices)
    centroid = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
    max_radius = np.max(np.linalg.norm(vertices - centroid, axis=1))

    # Calculate the scaling factor to make the max radius 100
    scale_factor = 100.0 / max_radius if max_radius > 0 else 1.0

    # Update vertex data
    new_lines = []
    for line in lines:
        if line.startswith("v "):
            parts = line.strip().split()
            vertex = np.array(list(map(float, parts[1:4])))
            normalized_vertex = (vertex - centroid) * scale_factor
            new_lines.append(
                f"v {normalized_vertex[0]} {normalized_vertex[1]} {normalized_vertex[2]}\n"
            )
        else:
            new_lines.append(line)

    # Rewrite the OBJ file
    with open(obj_file_path, "w") as file:
        file.writelines(new_lines)


def traverse_and_fix(root_dir):
    """
    Traverse all subfolders under the root directory and fix each OBJ file and its referenced MTL file.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".obj"):
                obj_file_path = os.path.join(subdir, file)
                fix_obj_mtl_path(obj_file_path)


# Example usage
# root_directory = "/ssd/du_dataset/mvdfusion/my_dataset_original_test_normal/"
root_directory = "/home/athiwat/progressive_img2sketch/resources/LOD_for_icp"
traverse_and_fix(root_directory, normalize=False)
