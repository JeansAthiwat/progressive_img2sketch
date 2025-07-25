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

SHOW_Y_AXIS = True  # Set to True to visualize the Y-axis

def render_spin(scenes: dict[int, trimesh.Scene], scene_number: int):
    """
    Spin each LOD model around the origin (yaw then pitch) under
    a fixed camera, and save images—now with a grazing‐angle key light.
    """
    # — use LOD3 to size the view —
    bounds = scenes[3].extents
    radius = np.max(bounds) / 2.0
    target = np.array([0.0, 0.0, 0.0])

    # 1) fixed camera at +Z
    renderer = pyrender.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)
    eye = np.array([0.0, 0.0, radius * 2.0])
    cam_pose = look_at_matrix(eye, target, np.array([0.0, 1.0, 0.0]))
    camera = pyrender.PerspectiveCamera(
        yfov=np.pi/3.0,
        aspectRatio=IMAGE_WIDTH/IMAGE_HEIGHT
    )

    # grazing‐light parameters
    KEY_AZIMUTH   = 45   # degrees around Y
    KEY_ELEVATION = 10   # degrees down from horizontal

    # precompute the key‐light rotation about the origin
    R_yaw_key = trimesh.transformations.rotation_matrix(
        np.deg2rad(KEY_AZIMUTH),   [0, 1, 0], point=target
    )
    R_pitch_key = trimesh.transformations.rotation_matrix(
        np.deg2rad(-KEY_ELEVATION), [1, 0, 0], point=target
    )
    pose_key = cam_pose @ R_yaw_key @ R_pitch_key

    for lod, orig_scene in scenes.items():
        # build a base scene (camera + lights)
        base_scene = pyrender.Scene()
        base_scene.add(camera, pose=cam_pose)

        # key from grazing angle
        key  = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        base_scene.add(key, pose=pose_key)

        # gentle fill from camera to recover shadow detail
        fill = pyrender.DirectionalLight(color=np.ones(3), intensity=0.5)
        base_scene.add(fill, pose=cam_pose)

        # optional back/rim for a faint outline
        back = pyrender.DirectionalLight(color=np.ones(3), intensity=0.3)
        base_scene.add(back, pose=cam_pose)

        # 2) spin through angles
        for az in range(0, 360, AZIMUTH_STEP):
            for el in ELEVATIONS:
                # object‐space rotation
                R_yaw   = trimesh.transformations.rotation_matrix(
                    np.deg2rad( az), [0, 1, 0], point=target
                )
                R_pitch = trimesh.transformations.rotation_matrix(
                    np.deg2rad( el), [1, 0, 0], point=target
                )
                M = R_pitch @ R_yaw

                # copy + rotate the mesh
                spin_scene = orig_scene.copy()
                spin_scene.apply_transform(M)

                # merge with camera+lights
                pyr_scene = pyrender.Scene.from_trimesh_scene(spin_scene)
                for node in base_scene.get_nodes():
                    pyr_scene.add_node(node)

                # render & save
                color, _ = renderer.render(pyr_scene)
                save_dir = os.path.join(OUTPUT_ROOT, str(scene_number), str(lod))
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{scene_number}_{lod}_{az}_{el}.png"
                plt.imsave(os.path.join(save_dir, fname), color)

    renderer.delete()



def render_orbit(scenes: dict[int, trimesh.Scene], scene_number: int):
    """
    For each LOD scene, render images at multiple azimuths and elevations and save.
    """
    # Use LOD3 to determine orbit radius
    bounds = scenes[3].extents
    radius = np.max(bounds) / 2.0
    target = np.array([0.0, 0.0, 0.0])

    renderer = pyrender.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)
    for lod, scene in scenes.items():
        pyrender_scene = pyrender.Scene.from_trimesh_scene(scene)
        # add a uniform ambient term (rgb)
        # pyrender_scene.ambient_light = np.array([0.05, 0.05, 0.05])
            # ←── insert SHOW_Y_AXIS here ──→
        if SHOW_Y_AXIS:
            # a very tall, thin box along Y to visualize the Y-axis
            bbox = trimesh.primitives.Box(extents=[1, max(scenes[3].extents)*2, 1])
            mat  = pyrender.Material(wireframe=True)
            mesh = pyrender.Mesh.from_trimesh(bbox, material=mat)
            pyrender_scene.add(mesh)
        
        # ← insert the wireframe overlay for LOD1 here
        # if lod == 1:
        #     combined = trimesh.util.concatenate(list(scene.geometry.values()))
        #     wire_mat = pyrender.Material( wireframe=True)
        #     wireframe = pyrender.Mesh.from_trimesh(combined, material=wire_mat, smooth=False)
        #     pyrender_scene.add(wireframe)
        
        # —– add Raymond lighting —–
        intensity = 3.0

        key = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        key_pose = np.array([
            [ 0,  0,  1,  2],
            [ 0,  1,  0,  2],
            [ 1,  0,  0,  2],
            [ 0,  0,  0,  1],
        ])
        pyrender_scene.add(key, pose=key_pose)

        fill = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity * 0.5)
        fill_pose = np.array([
            [ 0,  0, -1, -2],
            [ 0,  1,  0,  1],
            [-1,  0,  0, -2],
            [ 0,  0,  0,  1],
        ])
        pyrender_scene.add(fill, pose=fill_pose)

        back = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity * 0.3)
        back_pose = np.array([
            [ 1,  0,  0, -2],
            [ 0,  0,  1, -2],
            [ 0,  1,  0,  2],
            [ 0,  0,  0,  1],
        ])
        pyrender_scene.add(back, pose=back_pose)
        # —– end lights —–


        for az in range(0, 360, AZIMUTH_STEP):
            for el in ELEVATIONS:
                # spherical → cartesian
                rad_az = np.deg2rad(az)
                rad_el = np.deg2rad(el)
                x = radius * 2 * np.cos(rad_el) * np.sin(rad_az)
                y = radius * 2 * np.sin(rad_el)
                z = radius * 2 * np.cos(rad_el) * np.cos(rad_az)
                eye = np.array([x, y, z])

                # setup camera
                cam_pose = look_at_matrix(eye, target, np.array([0, 1, 0]))
                camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=IMAGE_WIDTH/IMAGE_HEIGHT)
                cam_node = pyrender_scene.add(camera, pose=cam_pose)

                # render
                color, _ = renderer.render(pyrender_scene)
                pyrender_scene.remove_node(cam_node)

                # save
                save_dir = os.path.join(OUTPUT_ROOT, str(scene_number), str(lod), str(az), str(el))
                os.makedirs(save_dir, exist_ok=True)
                file_name = f"{scene_number}_{lod}_{az}_{el}.png"
                plt.imsave(os.path.join(save_dir, file_name), color)

    renderer.delete()


import os
import numpy as np
import trimesh
from trimesh.visual.texture import TextureVisuals

def process_dataset():
    """
    Iterate through all scenes and LODs, align them, patch missing UVs, then render spin.
    """
    for scene_num in SCENES:
        # load scenes
        lod_scenes = {
            lod: trimesh.load(os.path.join(INPUT_ROOT, str(scene_num), f"lod{lod}.obj"))
            for lod in LODS
        }

        # convert single-mesh cases into Scenes
        for lod in LODS:
            if isinstance(lod_scenes[lod], trimesh.Trimesh):
                lod_scenes[lod] = trimesh.Scene(lod_scenes[lod])

        # align LODs
        aligned_scenes = align_lods(lod_scenes)

        # --- DUMMY‐UV PATCH: for any geom missing UV, give it a 1×1 white map ---
        for lod, scene in aligned_scenes.items():
            for name, geom in scene.geometry.items():
                existing_uv = getattr(geom.visual, 'uv', None)
                if existing_uv is None or len(existing_uv) == 0:
                    # planar UV from X,Y
                    verts2 = geom.vertices[:, :2]
                    uv = (verts2 - verts2.min(axis=0)) / np.ptp(verts2, axis=0)
                    # 1×1 white placeholder
                    placeholder = np.ones((1, 1, 3), dtype=np.uint8) * 255
                    geom.visual = TextureVisuals(uv=uv, image=placeholder)

        # render spin
        render_spin(aligned_scenes, scene_num)

def align_lods_1_2_only(scenes: dict[int, trimesh.Scene], center_before: bool = False):
    # — step 1: (optional) rough centering to help ICP converge —
    if center_before:
        for lod in [1, 2]:
            scenes[lod] = center_scene_by_bbox(scenes[lod])

    # — step 2: extract meshes —
    mesh1 = trimesh.util.concatenate(list(scenes[1].geometry.values()))
    mesh2 = trimesh.util.concatenate(list(scenes[2].geometry.values()))

    # Show original bbox centers
    for lod, mesh in [(1, mesh1), (2, mesh2)]:
        min_corner, max_corner = mesh.bounds
        center = (min_corner + max_corner) / 2.0
        print(f"LOD {lod} original center: {center}")

    # ICP: 2 → 1
    t2_1 = get_registration_matrix(mesh2, mesh1)
    scenes[2].apply_transform(t2_1)

    # Show aligned bbox centers
    for lod in [1, 2]:
        min_corner, max_corner = scenes[lod].bounds
        center = (min_corner + max_corner) / 2.0
        print(f"LOD {lod} aligned center: {center}")

    # — step 3: center both based on aligned LOD1 —
    min1, max1 = scenes[1].bounds
    center1 = (min1 + max1) * 0.5
    for lod in [1, 2]:
        scenes[lod].apply_translation(-center1)

    return scenes
