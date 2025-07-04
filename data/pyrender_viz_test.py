import trimesh
import pyrender
import numpy as np

lod_3_mesh = trimesh.load("/home/jeans/win/aaaJAIST/resources/LOD_data_50/1/lod3.obj")
# lod_3_mesh.show()

scene = pyrender.Scene.from_trimesh_scene(lod_3_mesh)

pyrender.Viewer(scene, use_raymond_lighting=True)
