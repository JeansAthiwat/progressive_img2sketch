# triangulate_and_normals.py

import bpy, sys, os

def triangulate_and_fix_normals(input_path: str, output_path: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.obj(filepath=input_path)

    # Get all imported mesh objects
    imported = list(bpy.context.selected_objects)

    # Ensure correct selection context
    for obj in imported:
        if obj.type == 'MESH':
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            # Enter edit mode to triangulate & recalc normals
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')

            # Perform triangulation
            bpy.ops.mesh.quads_convert_to_tris()
            # Recalculate normals to outside
            bpy.ops.mesh.normals_make_consistent(inside=False)

            # Return to object mode
            bpy.ops.object.mode_set(mode='OBJECT')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bpy.ops.export_scene.obj(
        filepath=output_path,
        use_selection=False,
        use_materials=True,
        use_uvs=True,
        use_normals=True,
        axis_forward='-Z',
        axis_up='Y',
        path_mode='AUTO'
    )

def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background --python triangulate_and_normals.py -- input.obj output.obj")
        return
    inp, outp = argv[argv.index("--") + 1 : argv.index("--") + 3]
    triangulate_and_fix_normals(inp, outp)

if __name__ == "__main__":
    main()
