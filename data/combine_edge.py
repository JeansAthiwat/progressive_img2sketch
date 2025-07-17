# This python script combines freestyle strokes from blender and extract canny edges from the original image and combines them into a single binary mask.

import os
import cv2
import numpy as np
from PIL import Image
# import gayporn # Assuming gayporn is a placeholder for the actual module you want to use
# from ultrakill import war_without_reason # goated game tbh

def extract_canny_edges(image_path, hysteresis_threshold=(10, 50), kernel_size=3):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    image = cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=1.0)
    out = cv2.Canny(image, hysteresis_threshold[0], hysteresis_threshold[1], apertureSize=kernel_size, L2gradient=True)
    out = np.where(out > 0, 255, 0).astype(np.uint8)
    out = cv2.bitwise_not(out)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    
    return out

def images_to_canny_pipe(input_image_folder, output_canny_folder, scene_range=range(51), lod_range=range(1, 4)):
    """
    Processes all images in the input directory and saves the Canny edge masks to the output directory.
    
    Args:
        input_image_folder (str): Path to the folder containing input images.
        output_canny_folder (str): Path to the folder where Canny edge masks will be saved.
    """
    os.makedirs(output_canny_folder, exist_ok=True)
        
    for scene_number in scene_range:  # Assuming scene numbers are from 0 to 50
        
        scene_folder = os.path.join(input_image_folder, f"{scene_number}")
        if not os.path.exists(scene_folder):
            print(f"Scene folder {scene_folder} does not exist. Skipping.")
            continue
        
        for lod in lod_range:
            
            lod_folder = os.path.join(scene_folder, f"lod{lod}")
            if not os.path.exists(lod_folder):
                print(f"LOD folder {lod_folder} does not exist. Skipping.")
                continue
            
            # for each files in lod_folder
            for filename in os.listdir(lod_folder):
                
                if not filename.endswith(".png"):
                    print(f"Skipping {filename} as it is not a PNG file.")
                    continue
                
                input_image_path = os.path.join(lod_folder, filename)
                output_canny_path = os.path.join(output_canny_folder, f"{scene_number}", f"lod{lod}", filename)
                
                os.makedirs(os.path.dirname(output_canny_path), exist_ok=True)
                canny_edges_rgb = extract_canny_edges(input_image_path)
                cv2.imwrite(output_canny_path, canny_edges_rgb)
                print(f"Processed {input_image_path} -> {output_canny_path}")

def combine_freestyle_and_canny(freestyle_folder, canny_folder, output_folder):
    """
    Combines freestyle strokes and Canny edges into a single binary mask.
    
    Args:
        freestyle_folder (str): Path to the folder containing freestyle stroke images.
        canny_folder (str): Path to the folder containing Canny edge images.
        output_folder (str): Path to the folder where combined masks will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for scene_number in range(49,51):  # Assuming scene numbers are from 0 to 50
        scene_freestyle_folder = os.path.join(freestyle_folder, f"{scene_number}")
        scene_canny_folder = os.path.join(canny_folder, f"{scene_number}")
        
        if not os.path.exists(scene_freestyle_folder) or not os.path.exists(scene_canny_folder):
            print(f"Scene {scene_number} folders do not exist. Skipping.")
            continue
        
        for lod in range(1, 4):
            lod_freestyle_folder = os.path.join(scene_freestyle_folder, f"lod{lod}")
            lod_canny_folder = os.path.join(scene_canny_folder, f"lod{lod}")
            
            if not os.path.exists(lod_freestyle_folder) or not os.path.exists(lod_canny_folder):
                print(f"LOD {lod} folders do not exist for scene {scene_number}. Skipping.")
                continue
            
            for filename in os.listdir(lod_freestyle_folder):
                if filename.endswith(".png"):
                    freestyle_path = os.path.join(lod_freestyle_folder, filename)
                    canny_path = os.path.join(lod_canny_folder, filename)
                    
                    if not os.path.exists(canny_path):
                        print(f"Canny image {canny_path} does not exist. Skipping.")
                        continue
                    
                    # Save individual images for debugging
                    debug_folder = os.path.join(output_folder, "debug", f"{scene_number}", f"lod{lod}")
                    os.makedirs(debug_folder, exist_ok=True)            
                    
                    # Load Freestyle RGBA image
                    freestyle_image = cv2.imread(freestyle_path, cv2.IMREAD_UNCHANGED)
                    if freestyle_image is None or freestyle_image.shape[2] != 4:
                        print(f"Error loading RGBA freestyle image: {freestyle_path}")
                        continue

                    # Separate RGBA
                    rgb = freestyle_image[:, :, :3].astype(np.float32)
                    alpha = freestyle_image[:, :, 3].astype(np.float32) / 255.0  # Normalize to [0,1]

                    # Define background gray value (0 = black, 255 = white)
                    bg_gray_value = 255  # White background
                    bg_gray = np.ones_like(rgb) * bg_gray_value

                    # Alpha blending: result = rgb * alpha + gray * (1 - alpha)
                    blended_rgb = (rgb * alpha[..., None] + bg_gray * (1 - alpha[..., None])).astype(np.uint8)

                    # Convert to grayscale
                    freestyle_gray = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2GRAY)
                    
                    # Load canny image
                    canny_image = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Check if images loaded correctly
                    if freestyle_gray is None or canny_image is None:
                        print(f"Error loading images for {filename}. Skipping.")
                        continue
                    
                    # Normalize both images to 0-255 range
                    if canny_image.max() <= 1:
                        # Canny is in 0-1 range, scale to 0-255
                        canny_normalized = (canny_image * 255).astype(np.uint8)
                    else:
                        # Already in 0-255 range
                        canny_normalized = canny_image.astype(np.uint8)
                    
                    # Freestyle is already in 0-255 range
                    freestyle_normalized = freestyle_gray.astype(np.uint8)
                    
                    print(f"Freestyle range: {freestyle_normalized.min()}-{freestyle_normalized.max()}")
                    print(f"Canny range: {canny_normalized.min()}-{canny_normalized.max()}")
                    
                    # Save debug images BEFORE combining
                    freestyle_debug_path = os.path.join(debug_folder, f"freestyle_{filename}")
                    cv2.imwrite(freestyle_debug_path, freestyle_normalized)
                    
                    canny_debug_path = os.path.join(debug_folder, f"canny_{filename}")
                    cv2.imwrite(canny_debug_path, canny_normalized)
                    
                    # Combine using minimum to keep the darkest (black) strokes from both
                    combined_mask = np.minimum(freestyle_normalized, canny_normalized)
                    print(f"Combined mask range: {combined_mask.min()}-{combined_mask.max()}")
                    combined_mask = np.where(combined_mask < 255, 0, 255).astype(np.uint8)  # Ensure binary mask
                    # Save the combined mask
                    output_path = os.path.join(output_folder, f"{scene_number}", f"lod{lod}", filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, combined_mask)
                    
                    print(f"Combined {freestyle_path} and {canny_path} -> {output_path}")
                    print(f"Debug images saved: {freestyle_debug_path}, {canny_debug_path}")
                    print(f"Combined range: {combined_mask.min()}-{combined_mask.max()}")
                    # break  # Remove this break to process all files


    
input_image_folder = "/home/athiwat/progressive_img2sketch/resources/LOD_orbit_images_BLENDER_WORKBENCH"
output_canny_folder = "/home/athiwat/progressive_img2sketch/resources/LOD_canny_images"

freestyle_folder = "resources/LOD_orbit_freestyles_BLENDER_WORKBENCH"
output_folder = "resources/LOD_combined_sketches"

# Extract Canny edges from the input images
# images_to_canny_pipe(input_image_folder, output_canny_folder)

# Combine freestyle strokes and Canny edges into a single binary mask
combine_freestyle_and_canny(freestyle_folder, output_canny_folder, output_folder)