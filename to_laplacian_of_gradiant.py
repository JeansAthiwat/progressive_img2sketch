# %%
import os
import cv2
import numpy as np

# Configuration
root = "/home/athiwat/progressive_img2sketch/resources/LOD50_opaque_normalized_1radius_triangulated_fix_normals_orbits_with_depth_cropped_512x512"
image_root = os.path.join(root, "image")
LOW_THRESHOLD = 100
HIGH_THRESHOLD = 150
EDGE_METHOD = "log"  # Options: "canny", "log", "xdog"

# Output base
out_sub = {"canny": "canny", "log": "log", "xdog": "xdog"}[EDGE_METHOD]

# Create output directories
for sub in [out_sub, "sketch"]:
    for scene in range(51):
        for lod in ["lod1", "lod2"]:
            out_dir = os.path.join(root, sub, str(scene), lod)
            os.makedirs(out_dir, exist_ok=True)
# Process each scene and view
for scene in range(49, 51):  # Change this to range(51) to do all scenes
    scene_str = str(scene)
    print("Processing scene:", scene_str)    
    img1_dir = os.path.join(image_root, scene_str, "lod1")
    img2_dir = os.path.join(image_root, scene_str, "lod2")

    if not (os.path.isdir(img1_dir) and os.path.isdir(img2_dir)):
        continue

    for fname1 in os.listdir(img1_dir):
        if not fname1.endswith(".png"):
            continue

        suffix = fname1[len("lod1_"):]
        fname2 = "lod2_" + suffix

        path1_img = os.path.join(img1_dir, fname1)
        path2_img = os.path.join(img2_dir, fname2)

        if not (os.path.isfile(path2_img) and os.path.isfile(path1_img)):
            continue

        img1 = cv2.imread(path1_img, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(path2_img, cv2.IMREAD_UNCHANGED)

        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if EDGE_METHOD == "canny":
            blur1 = cv2.GaussianBlur(gray_img1, (5, 5), 0)
            blur2 = cv2.GaussianBlur(gray_img2, (5, 5), 0)
            edge1 = cv2.Canny(blur1, LOW_THRESHOLD, HIGH_THRESHOLD)
            edge2 = cv2.Canny(blur2, LOW_THRESHOLD, HIGH_THRESHOLD)

        elif EDGE_METHOD == "log":
            blur1 = cv2.GaussianBlur(gray_img1, (3, 3), 0)
            blur2 = cv2.GaussianBlur(gray_img2, (3, 3), 0)
            lap1 = cv2.Laplacian(blur1, cv2.CV_8U)
            lap2 = cv2.Laplacian(blur2, cv2.CV_8U)

            # Bias amount (lower threshold → more black)
            bias = 4

            # Otsu threshold, then subtract bias (clip to ≥0)
            t1, _ = cv2.threshold(lap1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t2, _ = cv2.threshold(lap2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            biased_t1 = max(t1 - bias, 0)
            biased_t2 = max(t2 - bias, 0)

            # Apply inverted binary threshold with biased value
            _, edge1 = cv2.threshold(lap1, biased_t1, 255, cv2.THRESH_BINARY_INV)
            _, edge2 = cv2.threshold(lap2, biased_t2, 255, cv2.THRESH_BINARY_INV)

        else:
            raise ValueError(f"Unsupported EDGE_METHOD: {EDGE_METHOD}")

        # Invert for sketch effect
        sketch_img1 = cv2.bitwise_not(edge1)
        sketch_img2 = cv2.bitwise_not(edge2)

        # Save
        cv2.imwrite(os.path.join(root, out_sub, scene_str, "lod1", fname1), edge1)
        cv2.imwrite(os.path.join(root, out_sub, scene_str, "lod2", fname2), edge2)
        cv2.imwrite(os.path.join(root, "sketch", scene_str, "lod1", fname1), sketch_img1)
        cv2.imwrite(os.path.join(root, "sketch", scene_str, "lod2", fname2), sketch_img2)

print(f"Done. Extracted edges using method: {EDGE_METHOD}")