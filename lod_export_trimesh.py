import os
import math
import trimesh
import numpy as np
from trimesh.visual import ColorVisuals

# Force software GL if needed
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

# ─── User Options ─────────────────────────────────────────────────────────────
lod_path = "/home/athiwat/progressive_img2sketch/resources/LOD_data_50/29/lod2.obj"
output_folder = "./orbit_renders"
show_mesh = True
threshold_degrees = 5.0
angle_thresh = np.deg2rad(threshold_degrees)

# Orbit parameters
az_step = 30  # degrees per frame around Z
elevations = [0, 15, 30]
distance_factor = 2.0  # radius = bounding_sphere.radius × factor
# ────────────────────────────────────────────────────────────────────────────────

# 1. Load your OBJ without processing (preserve UVs)
loaded = trimesh.load(lod_path, process=False)

# 2. Extract (textured or untextured) mesh parts
textured_meshes = (
    list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
)

# 3. Create a Trimesh Scene and add your mesh(es)
scene = trimesh.Scene()
for mesh in textured_meshes:
    if show_mesh:
        scene.add_geometry(mesh)
    else:
        wm = mesh.copy()
        fc = np.tile([255, 255, 255, 255], (len(wm.faces), 1))
        wm.visual = ColorVisuals(mesh=wm, face_colors=fc)
        scene.add_geometry(wm)

# 4. Build a welded mesh for crease detection
clean_parts = []
for mesh in textured_meshes:
    clean_parts.append(
        trimesh.Trimesh(
            vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=True
        )
    )
clean_full = trimesh.util.concatenate(clean_parts)

# 5. Sharp‑edge test (face‑normal angle > threshold)
fa = clean_full.face_adjacency_angles  # θ = arccos(n₁·n₂)
mask = fa > angle_thresh
edges = clean_full.face_adjacency_edges[mask]
segments = clean_full.vertices[edges]

# 6. Add crease lines
crease = trimesh.load_path(segments)
crease.colors = np.tile([0, 0, 0, 255], (len(crease.entities), 1))
scene.add_geometry(crease)

# 7. Configure camera
scene.camera.resolution = (800, 600)
scene.camera.fov = (60.0, 45.0)  # degrees (horizontal, vertical)
scene.show()


# helper: build camera→world transform
def camera_to_world(eye, target, up=[0, 0, 1]):
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up_v = np.array(up, dtype=float)

    # forward vector
    f = target - eye
    f /= np.linalg.norm(f)

    # right vector
    r = np.cross(f, up_v)
    r /= np.linalg.norm(r)

    # true up
    u = np.cross(r, f)

    # assemble camera→world
    mat = np.eye(4)
    mat[0:3, 0] = r
    mat[0:3, 1] = u
    mat[0:3, 2] = f
    mat[0:3, 3] = eye
    return mat


# 8. Compute orbit center & radius
bs = clean_full.bounding_sphere
center = bs.center
radius = bs.primitive.radius * distance_factor

# 9. Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# 10. Loop over azimuth & elevation, render offscreen
for az in range(0, 360, az_step):
    for el in elevations:
        a = math.radians(az)
        e = math.radians(el)
        x = center[0] + radius * math.cos(a) * math.cos(e)
        y = center[1] + radius * math.sin(a) * math.cos(e)
        z = center[2] + radius * math.sin(e)
        eye = [x, y, z]

        # set camera transform
        scene.camera_transform = camera_to_world(eye, center, up=[0, 0, 1])

        # offscreen render (honors depth + lighting)
        png = scene.save_image(resolution=scene.camera.resolution, visible=True)

        # write PNG
        fname = f"az{az:03d}_el{el:02d}.png"
        path = os.path.join(output_folder, fname)
        with open(path, "wb") as f:
            f.write(png)

        print("Wrote", path)
