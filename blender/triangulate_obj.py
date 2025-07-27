# triangulate_obj.py
import bpy, sys, os

def triangulate_obj(input_path: str, output_path: str):
    # start with an empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # import the OBJ
    bpy.ops.import_scene.obj(filepath=input_path)

    # for each mesh, go into edit mode, select all, triangulate, back to object
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.quads_convert_to_tris()
            bpy.ops.object.mode_set(mode='OBJECT')

    # ensure output folder exists
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)

    # export back to OBJ (will write .obj + .mtl, preserving UVs/normals)
    bpy.ops.export_scene.obj(
        filepath=output_path,
        use_selection=False,
        use_materials=True,
        use_uvs=True,
        use_normals=True,
        use_triangles=False,   # we've already converted
        axis_forward='-Z',
        axis_up='Y',
        path_mode='AUTO'
    )

def main():
    # args after “--” are our input/output
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background --python triangulate_obj.py -- input.obj output.obj")
        return
    inp, outp = argv[argv.index("--")+1 : argv.index("--")+3]
    triangulate_obj(inp, outp)

if __name__ == "__main__":
    main()
