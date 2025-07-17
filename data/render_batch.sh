#!/bin/bash

BLENDER_PATH="/home/athiwat/progressive_img2sketch/blender-3-6-23/blender"  # <-- update this
SCRIPT_PATH="/home/athiwat/progressive_img2sketch/data/render_one_scene.py"  # <-- update this

for scene_id in {0..50}; do
  for lod_level in {1..2}; do
    echo ">>> Rendering Scene $scene_id LOD $lod_level"
    "$BLENDER_PATH" -b --python "$SCRIPT_PATH" -- "$scene_id" "$lod_level"
  done
done
