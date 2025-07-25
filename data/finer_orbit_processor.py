import os
import cv2
import numpy as np

# --- Configuration ---
INPUT_DIR = '/home/athiwat/progressive_img2sketch/resources/LOD50_Finer_orbit'
OUTPUT_DIR = '/home/athiwat/progressive_img2sketch/resources/LOD50_Finer_orbit_processed'
MERGED_DIR = os.path.join(OUTPUT_DIR, 'merged_images')
CANNY_DIR = os.path.join(OUTPUT_DIR, 'canny_images')

# Canny parameters
LOW_THRESHOLD = 5
HIGH_THRESHOLD = 10

# Create output dirs
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- Main Processing ---
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith('.png') and 'freestyle' in file:
            freestyle_path = os.path.join(root, file)
            mesh_path = freestyle_path.replace('freestyle', 'mesh')

            # Output relative path
            rel_path = os.path.relpath(freestyle_path, INPUT_DIR)
            merged_output_path = os.path.join(MERGED_DIR, rel_path)
            canny_output_path = os.path.join(CANNY_DIR, rel_path)

            # Ensure output folder exists
            ensure_dir(os.path.dirname(merged_output_path))
            ensure_dir(os.path.dirname(canny_output_path))

            # Read input images
            freestyle_image = cv2.imread(freestyle_path, cv2.IMREAD_UNCHANGED)
            mesh_image = cv2.imread(mesh_path, cv2.IMREAD_UNCHANGED)

            if freestyle_image is None or mesh_image is None:
                print(f"[!] Skipping {file}: unable to read image(s).")
                continue

            if freestyle_image.shape != mesh_image.shape:
                print(f"[!] Skipping {file}: shape mismatch.")
                continue

            # Split channels
            fs_rgb = freestyle_image[..., :3]
            fs_alpha = freestyle_image[..., 3:] / 255.0

            mesh_rgb = mesh_image[..., :3]
            mesh_alpha = mesh_image[..., 3:] / 255.0

            # Alpha blend
            out_rgb = fs_rgb * fs_alpha + mesh_rgb * (1 - fs_alpha)
            out_alpha = fs_alpha + mesh_alpha * (1 - fs_alpha)

            # Merge RGBA
            merged_rgba = np.dstack((out_rgb, out_alpha * 255)).astype(np.uint8)
            merged_rgb = cv2.cvtColor(merged_rgba, cv2.COLOR_RGBA2RGB)

            # Save merged image
            cv2.imwrite(merged_output_path, merged_rgb)

            # Apply Canny
            canny_gray = cv2.Canny(merged_rgb, LOW_THRESHOLD, HIGH_THRESHOLD)
            canny_rgb = np.stack([canny_gray] * 3, axis=-1)

            # Save Canny image
            cv2.imwrite(canny_output_path, canny_rgb)

            print(f"[✓] Processed: {rel_path}")
