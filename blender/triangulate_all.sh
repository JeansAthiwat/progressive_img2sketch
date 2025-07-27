#!/usr/bin/env bash
set -e

# point to your Blender binary if needed
BLENDER_EXEC="blender"  
# or e.g. BLENDER_EXEC="/home/athiwat/progressive_img2sketch/blender-3-6-23/blender"

INPUT_DIR="/home/athiwat/progressive_img2sketch/resources/LOD50_opaque_normalized"
OUTPUT_DIR="/home/athiwat/progressive_img2sketch/resources/LOD50_opaque_normalized_triangulated"
BLENDER_SCRIPT="/home/athiwat/progressive_img2sketch/blender/triangulate_obj.py"

# only these LODs will be triangulated
ALLOWED_LODS=(lod1 lod2 lod3)

mkdir -p "$OUTPUT_DIR"

for scene in $(seq 0 50); do
  in_scene="${INPUT_DIR}/${scene}"
  [ -d "$in_scene" ] || { echo "Skipping scene ${scene} (not found)"; continue; }

  out_scene="${OUTPUT_DIR}/${scene}"
  mkdir -p "$out_scene"

  for in_obj in "$in_scene"/*.obj; do
    [ -f "$in_obj" ] || continue

    base="$(basename "$in_obj" .obj)"   # e.g. "lod2"
    # skip any lod not in ALLOWED_LODS
    if [[ ! " ${ALLOWED_LODS[*]} " =~ " ${base} " ]]; then
      echo "  → skipping ${base}.obj"
      continue
    fi

    out_obj="${out_scene}/${base}.obj"
    echo "Triangulating ${in_obj} → ${out_obj}"
    "$BLENDER_EXEC" --background --python "$BLENDER_SCRIPT" -- "$in_obj" "$out_obj"

    # copy the .mtl
    if [ -f "${in_scene}/${base}.mtl" ]; then
      cp "${in_scene}/${base}.mtl" "${out_scene}/"
    fi

    # copy any texture folder (e.g. "lod3/…jpg")
    if [ -d "${in_scene}/${base}" ]; then
      cp -r "${in_scene}/${base}" "${out_scene}/"
    fi
  done
done
