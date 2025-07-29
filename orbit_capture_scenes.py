# %% [markdown]
# # This notebook does the whole pipeline starting from the raw dataset

# %% [markdown]
# ### Import Libraries

# %%
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # must be set before importing pyrender

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import trimesh
from trimesh.visual import ColorVisuals
from trimesh.scene.lighting import DirectionalLight, PointLight

import pyrender
from pyrender import Primitive, Mesh as PyMesh, PerspectiveCamera, SpotLight, OffscreenRenderer
from pyrender.constants import RenderFlags
from pyrender import MetallicRoughnessMaterial

from collections import defaultdict

# %% [markdown]
# ## Functions for orbit captures

# %%

def center_scene_by_bbox(scene: trimesh.Scene) -> trimesh.Scene:
    """
    Center the scene at the origin based on its bounding-box center.
    """
    min_corner, max_corner = scene.bounds
    center = (min_corner + max_corner) / 2.0
    scene.apply_translation(-center)
    return scene

def get_registration_matrix(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    samples: int = 3000,
    icp_first: int = 1,
    icp_final: int = 30
) -> np.ndarray:
    """
    Compute the ICP transformation matrix that aligns source_mesh to target_mesh.
    """
    matrix, _ = trimesh.registration.mesh_other(
        source_mesh,
        target_mesh,
        samples=samples,
        scale=False,
        icp_first=icp_first,
        icp_final=icp_final
    )
    return matrix

def align_lods(scenes: dict[int, trimesh.Scene], center_before: bool = False):
    # — step 1: (optional) rough centering to help ICP converge —
    if center_before:
        for lod in scenes:
            scenes[lod] = center_scene_by_bbox(scenes[lod])

    # — step 2: extract single meshes for ICP —
    
    meshes = {
        lod: trimesh.util.concatenate(list(scenes[lod].geometry.values()))
        for lod in scenes
    }

    #show original bbox centers
    for lod, mesh in meshes.items():
        min_corner, max_corner = mesh.bounds
        center = (min_corner + max_corner) / 2.0
        print(f"LOD {lod} original center: {center}")
        
    # ICP: 2→1 then 3→2
    t2_1 = get_registration_matrix(meshes[2], meshes[1])
    t3_2 = get_registration_matrix(meshes[3], meshes[2])

    # apply those transforms
    scenes[2].apply_transform(t2_1)
    scenes[3].apply_transform(t2_1 @ t3_2)

    # show aligned bbox centers
    for lod, scene in scenes.items():
        min_corner, max_corner = scene.bounds
        center = (min_corner + max_corner) / 2.0
        print(f"LOD {lod} aligned center: {center}")
    # — step 3: **final centering** based on aligned LOD1 bbox —
    min1, max1 = scenes[1].bounds
    center1 = (min1 + max1) * 0.5
    for lod in scenes:
        scenes[lod].apply_translation(-center1)

    return scenes

def align_lods_1_2_only(meshes: dict[int, trimesh.base.Trimesh], center_before: bool = False, samples: int = 3000):
    
    # Show original bbox centers
    for lod, mesh in [(1, meshes[1]), (2, meshes[2])]:
        min_corner, max_corner = mesh.bounds
        center = (min_corner + max_corner) / 2.0
        print(f"LOD {lod} original center: {center}")
        
    # — step 1: (optional) rough centering to help ICP converge —
    if center_before:
        for lod in [1, 2]:
            meshes[lod] = center_scene_by_bbox(meshes[lod])

    # — step 2: extract meshes —

    mesh1 = meshes[1]
    mesh2 = meshes[2]
    # ICP: 2 → 1
    t2_1 = get_registration_matrix(mesh2, mesh1, samples=samples)
    meshes[2].apply_transform(t2_1)

    # Show aligned bbox centers
    for lod in [1, 2]:
        min_corner, max_corner = meshes[lod].bounds
        center = (min_corner + max_corner) / 2.0
        print(f"LOD {lod} aligned center: {center}")

    # — step 3: center both based on aligned LOD1 —
    min1, max1 = meshes[1].bounds
    center1 = (min1 + max1) * 0.5
    for lod in [1, 2]:
        meshes[lod].apply_translation(-center1)

    return meshes

def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Create a camera-to-world pose matrix for pyrender given eye, target, up vectors.
    """
    f = (target - eye)
    f /= np.linalg.norm(f)
    # avoid parallel up/f
    if np.isclose(np.linalg.norm(np.cross(f, up)), 0):
        up = np.array([0, 0, 1]) if np.isclose(abs(f.dot([0, 1, 0])), 1) else np.array([0, 1, 0])
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f); u /= np.linalg.norm(u)

    # view matrix (world→camera)
    view = np.array([
        [ s[0],  s[1],  s[2], -s.dot(eye)],
        [ u[0],  u[1],  u[2], -u.dot(eye)],
        [-f[0], -f[1], -f[2],  f.dot(eye)],
        [    0,     0,     0,           1]
    ])
    # invert → camera pose (camera→world)
    return np.linalg.inv(view)


def render_orbit_with_creases(mesh, line_mesh, lod_meshes, scene_number, lod, output_root,
                               azimuths, elevations, width=2048, height=2048):
    """
    Orbit render for one mesh+line using pyrender, saving both color and depth images.
    """
    max_bbox = max([m.bounding_box.extents.max() for m in lod_meshes.values()])
    radius = max_bbox * 1.5
    target = np.array([0.0, 0.0, 0.0])

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    for az in azimuths:
        for el in elevations:
            rad_az = np.deg2rad(az)
            rad_el = np.deg2rad(el)
            x = radius * np.cos(rad_el) * np.sin(rad_az)
            y = radius * np.sin(rad_el)
            z = radius * np.cos(rad_el) * np.cos(rad_az)
            eye = np.array([x, y, z])

            scene = pyrender.Scene(bg_color=[255,255,255, 0], ambient_light=[0.9, 0.9, 0.9])
            intensity = 10.0

            key = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
            key_pose = np.array([[ 0,  0,  1,  2], [ 0,  1,  0,  2], [ 1,  0,  0,  2], [ 0,  0,  0,  1]])
            scene.add(key, pose=key_pose)

            fill = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity * 0.85)
            fill_pose = np.array([[ 0,  0, -1, -2], [ 0,  1,  0,  1], [-1,  0,  0, -2], [ 0,  0,  0,  1]])
            scene.add(fill, pose=fill_pose)

            back = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity * 0.70)
            back_pose = np.array([[ 1,  0,  0, -2], [ 0,  0,  1, -2], [ 0,  1,  0,  2], [ 0,  0,  0,  1]])
            scene.add(back, pose=back_pose)

            cam_pose = look_at_matrix(eye, target, up=np.array([0, 1, 0]))
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=width / height)
            scene.add(camera, pose=cam_pose)

            # 1. Build textured mesh from Trimesh
            mesh_tex = PyMesh.from_trimesh(mesh, smooth=False)  # preserves original textures

            # 2. Force opaque alpha mode on every primitive
            for prim in mesh_tex.primitives:
                prim.material.alphaMode = "OPAQUE"

            # 4. Add to scene
            scene.add(mesh_tex)


            if line_mesh is not None:
                scene.add(line_mesh)
            else:
                print("!!No line mesh provided, skipping crease rendering!!")

            render_flags = RenderFlags.ALL_SOLID | RenderFlags.RGBA | RenderFlags.SKIP_CULL_FACES
            color, depth = renderer.render(scene, flags=render_flags)

            filename = f"lod{lod}_az{az:03d}_el{el:02d}.png"

            # Save color image
            save_dir_img = os.path.join(output_root, "image", str(scene_number), f"lod{lod}")
            os.makedirs(save_dir_img, exist_ok=True)
            save_path_img = os.path.join(save_dir_img, filename)
            if color.shape[-1] == 4:
                Image.fromarray(color, mode="RGBA").save(save_path_img)
            elif color.shape[-1] == 3:
                Image.fromarray(color, mode="RGB").save(save_path_img)
            else:
                raise ValueError(f"Unexpected image shape: {color.shape}")

            # — NEW depth export to match DPT demo exactly —
            save_dir_depth = os.path.join(output_root, "depth", str(scene_number), f"lod{lod}")
            os.makedirs(save_dir_depth, exist_ok=True)
            save_path_depth = os.path.join(save_dir_depth, filename)
            # 1) fill zero‐depth (misses) with farthest valid Z
            valid_mask = depth > 0.0
            print(f"Valid depth pixels: {valid_mask.sum()} out of {depth.size}")
            max_z = float(depth[valid_mask].max())
            depth_f = depth.copy()
            depth_f[~valid_mask] = max_z
            # 2) compute disparity = 1 / depth
            eps = 1e-6
            disp = 1.0 / (depth_f + eps)
            # 3) normalize disparity to [0…1]
            dmin, dmax = float(disp.min()), float(disp.max())
            disp_n = (disp - dmin) / (dmax - dmin)
            # 4) to uint8 [0…255], so that near=255 (white), far=0 (black)
            u8 = (disp_n * 255.0).clip(0, 255).astype(np.uint8)
            # 5) stack to 3 channels (H, W, 3) and save as RGB
            depth_rgb = np.stack([u8] * 3, axis=-1)
            Image.fromarray(depth_rgb, mode="RGB").save(save_path_depth)

    renderer.delete()


def line_segments_to_cylinders(vertices, edges, radius=0.001, sections=6):
    cylinders = []
    for edge in edges:
        start = vertices[edge[0]]
        end   = vertices[edge[1]]
        direction = end - start
        height = np.linalg.norm(direction)
        if height < 1e-6:
            continue

        # Create a base cylinder aligned to z-axis
        cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        cyl.apply_translation([0, 0, height / 2.0])  # base at origin

        # Rotate to align with actual direction
        cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction))

        # Translate to start point
        cyl.apply_translation(start)

        cylinders.append(cyl)

    if not cylinders:
        return None

    # Combine all into one mesh
    combined = trimesh.util.concatenate(cylinders)

    black_color = np.tile([0, 0, 0, 255], (len(combined.faces), 1))  # uint8 by default
    combined.visual.face_colors = black_color

    return pyrender.Mesh.from_trimesh(combined, smooth=False)

# %% [markdown]
# ### 2. Load the scene and orbit capture

# %%
# ─── Config ───────────────────────────────────────────────────────
# scene_num = 46
threshold_degrees = 5.0
angle_thresh = np.deg2rad(threshold_degrees)
# RAW_LOD_DATASET_ROOT = "/home/athiwat/progressive_img2sketch/resources/LOD50_opaque_normalized_1radius_triangulated_fix_normals"
RAW_LOD_DATASET_ROOT = "/home/athiwat/progressive_img2sketch/resources/LOD50_opaque_1000radius_triangulated"


# Step 4: Render together with the mesh
AZIMUTH_STEP = 10 
ELEVATIONS = [0,10,20,30,40,50,60] # [30]
OUTPUT_ROOT = "/home/athiwat/progressive_img2sketch/resources/LOD50_opaque_normalized_1radius_triangulated_fix_normals_orbits_with_depth"  # customize this

SCENES = range(48, 51)  # Assuming scenes are numbered from 0 to 50 inclusive
LODS = [1, 2]

for scene_num in SCENES:
    print(f"Processing scene {scene_num}...")
    # ─── 1. Load LOD meshes into dict ─────────────────────────────────
    lod_meshes = {}
    for lod in LODS:
        path = os.path.join(RAW_LOD_DATASET_ROOT, str(scene_num), f"lod{lod}.obj")
        loaded = trimesh.load(path, process=False)
        lod_mesh = (
            trimesh.util.concatenate(loaded.geometry.values())
            if isinstance(loaded, trimesh.Scene)
            else loaded
        )
        lod_meshes[lod] = lod_mesh
        
    print(f"Loaded LOD meshes for scene {scene_num}: {list(lod_meshes.keys())}")

    # ─── 2. Align meshes ──────────────────────────────────────────────
    # aligned_meshes = align_lods_1_2_only(lod_meshes, center_before=True, samples=4000)

    
    # # ─── 3. Build scene dict with crease lines ────────────────────────
    for lod, mesh in lod_meshes.items(): # for lod, mesh in aligned_meshes.items():
        # if lod == 1:
        #     print(f"Skipping LOD{lod} 1 for testing purposes")
        #     continue
        print(f"    Processing LOD{lod}...")

        # Step 1: Weld mesh for edge adjacency analysis
        welded = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                                faces=mesh.faces.copy(),
                                process=True)
        
        # Step 2: Detect creases
        fa = welded.face_adjacency_angles
        edges = welded.face_adjacency_edges
        mask = fa > angle_thresh
        
        # Filter to manifold edges only
        edge_count = defaultdict(int)
        for face in welded.faces:
            for i in range(3):
                e = tuple(sorted((face[i], face[(i+1)%3])))
                edge_count[e] += 1
        filtered_edges = [e for e in edges[mask] if edge_count[tuple(sorted(e))] == 2]

        # Step 3: Create pyrender line mesh for rendering
        line_mesh = line_segments_to_cylinders(welded.vertices, filtered_edges)

        render_orbit_with_creases(
            mesh=mesh,
            line_mesh=line_mesh,
            lod_meshes=lod_meshes, # aligned_meshes,
            scene_number=scene_num,
            lod=lod,
            output_root=OUTPUT_ROOT,
            azimuths=range(0, 360, AZIMUTH_STEP),
            elevations=ELEVATIONS
        )


