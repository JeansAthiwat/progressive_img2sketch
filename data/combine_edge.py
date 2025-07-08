# This python script combines freestyle strokes from blender and extract canny edges from the original image and combines them into a single binary mask.

import os
import cv2
import numpy as np
from PIL import Image
# import gayporn # Assuming gayporn is a placeholder for the actual module you want to use
# from ultrakill import war_without_reason # goated game tbh

def extract_canny_edges(image_path, hysteresis_threshold=(10, 50), kernel_size=3):
    """
    Extracts Canny edges from the given image.
    
    Args:
        image_path (str): Path to the input image. (RGBA format is expected)
        hysteresis_threshold (tuple): Thresholds for Canny edge detection. if None just stop at non-max suppression.
        kernel_size (int): Size of the Gaussian kernel used in Canny edge detection.
    Returns:
        np.ndarray: Binary mask of the edges.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Convert to grayscale if the image has an alpha channel (fill transparent channel with black)
    if image.shape[2] == 4:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        alpha_channel = image[:, :, 3]
        gray_image[alpha_channel == 0] = 0  # Mask out fully transparent regions
    else:
        raise ValueError("Damn something wong:", image.shape)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    # Apply Canny edge detection
    if hysteresis_threshold is not None:
        edges = cv2.Canny(blurred_image, hysteresis_threshold[0], hysteresis_threshold[1])
    else:
        edges = cv2.Canny(blurred_image, 100, 200)
    # Convert edges to binary mask
    binary_mask = np.where(edges > 0, 255, 0).astype(np.uint8)
    #  inverse the mask to get white edges on black background
    binary_mask = cv2.bitwise_not(binary_mask)
    
    return binary_mask

def images_to_canny_pipe(input_image_folder, output_canny_folder):
    """
    Processes all images in the input directory and saves the Canny edge masks to the output directory.
    
    Args:
        input_image_folder (str): Path to the folder containing input images.
        output_canny_folder (str): Path to the folder where Canny edge masks will be saved.
    """

    # input file names are like os.path.join(input_image_folder,{scene_number(0to50)}, lod{1,2,3}, f"lod{1,2,3}_az{az:03d}_el{el:02d}.png")
    # output structure should be the same as input structure with different root directory (output_canny_folder)/{scene_number(0to50)}/lod{1,2,3}/f"lod{1,2,3}_az{az:03d}_el{el:02d}.png")
    if not os.path.exists(output_canny_folder):
        os.makedirs(output_canny_folder)
    for scene_number in range(51):  # Assuming scene numbers are from 0 to 50
        scene_folder = os.path.join(input_image_folder, f"{scene_number}")
        if not os.path.exists(scene_folder):
            print(f"Scene folder {scene_folder} does not exist. Skipping.")
            continue
        for lod in range(1, 4):
            lod_folder = os.path.join(scene_folder, f"lod{lod}")
            if not os.path.exists(lod_folder):
                print(f"LOD folder {lod_folder} does not exist. Skipping.")
                continue
            # for each files in lod_folder
            for filename in os.listdir(lod_folder):
                if filename.endswith(".png"):
                    input_image_path = os.path.join(lod_folder, filename)
                    output_canny_path = os.path.join(output_canny_folder, f"{scene_number}", f"lod{lod}", filename)
                    
                    # Create output directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_canny_path), exist_ok=True)
                    
                    # Extract Canny edges
                    canny_edges = extract_canny_edges(input_image_path)
                    
                    # Save the binary mask as an image
                    cv2.imwrite(output_canny_path, canny_edges)
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
    
    for scene_number in range(4):  # Assuming scene numbers are from 0 to 50
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
                    
                            
                    # Load images
                    freestyle_image = cv2.imread(freestyle_path, cv2.IMREAD_UNCHANGED)
                    freestyle_image = freestyle_image[:, :, :3]*255  # Convert to RGB if it has an alpha channel
                    freestyle_image = cv2.cvtColor(freestyle_image, cv2.COLOR_RGB2GRAY)
                    
                    print("frerestyuel shaope " , freestyle_image.shape)
                    print(freestyle_image[200:220, 200:220])
                    # Save freestyle image
                    freestyle_debug_path = os.path.join(debug_folder, f"freestyle_{filename}")
                    cv2.imwrite(freestyle_debug_path, freestyle_image)
                    
                    canny_image = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE)
                    print("canny max value", canny_image.max())
                    
                    # freestyle_image is grayscale, canny_image is a binary mask
                    if freestyle_image is None or canny_image is None:
                        print(f"Error loading images for {filename}. Skipping.")
                        continue
                    
                    # Combine the two binary masks using bitwise OR (union) or AND (intersection)
                    # Use OR to combine edges from both sources
                    combined_mask = cv2.bitwise_or(freestyle_image, canny_image)
                    
                    # Alternatively, use AND for intersection:
                    # combined_mask = cv2.bitwise_and(freestyle_image, canny_image)
                    
                    # Save the combined mask
                    output_path = os.path.join(output_folder, f"{scene_number}", f"lod{lod}", filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, combined_mask)
                    

                    

                    
                    # Save canny image
                    canny_debug_path = os.path.join(debug_folder, f"canny_{filename}")
                    cv2.imwrite(canny_debug_path, canny_image)
                    
                    print(f"Combined {freestyle_path} and {canny_path} -> {output_path}")
                    print(f"Debug images saved: {freestyle_debug_path}, {canny_debug_path}")
                    # break  # Remove this break to process all files


    
input_image_folder = "/home/athiwat/progressive_img2sketch/resources/LOD_orbit_images_BLENDER_WORKBENCH"
output_canny_folder = "/home/athiwat/progressive_img2sketch/resources/LOD_canny_images"

freestyle_folder = "resources/LOD_orbit_freestyles_BLENDER_WORKBENCH"
output_folder = "resources/LOD_combined_sketches"

# Extract Canny edges from the input images
# images_to_canny_pipe(input_image_folder, output_canny_folder)

# Combine freestyle strokes and Canny edges into a single binary mask
combine_freestyle_and_canny(freestyle_folder, output_canny_folder, output_folder)