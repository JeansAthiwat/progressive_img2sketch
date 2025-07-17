import os
import json
import glob



#This files generates the {"source": "", "target": "", "prompt": ""} manifest file for training ControlNet.

LOD_FROM = 3
LOD_TO = 2

PROMPT_JSON_PATH = f"/home/athiwat/progressive_img2sketch/ControlNet/training/LOD50/prompt_from{LOD_FROM}_to{LOD_TO}.json"

PATH_TO_REPO_ROOT = "/home/athiwat/progressive_img2sketch/"
DATASET_DIR = "resources/LOD_combined_sketches/"

# Editable prompt for all entries
TRAINING_PROMPT = "Simplified architectural drawing"

# Azimuth and elevation values (based on the file pattern)
AZIMUTH_VALUES = [f"{i:03d}" for i in range(0, 360, 15)]  # 000, 015, 030, ..., 345
ELEVATION_VALUES = ["00", "15", "30"]

def generate_manifest():
    """Generate manifest file for ControlNet training"""
    manifest_entries = []
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(PROMPT_JSON_PATH), exist_ok=True)
    
    # Iterate through all folders (00-50)
    for folder_num in range(0, 46): # left some as a testset
        folder_name = f"{folder_num}"
        
        # Check if both source and target LOD folders exist
        source_lod_path = os.path.join(PATH_TO_REPO_ROOT, DATASET_DIR, folder_name, f"lod{LOD_FROM}")
        target_lod_path = os.path.join(PATH_TO_REPO_ROOT, DATASET_DIR, folder_name, f"lod{LOD_TO}")
        
        if not os.path.exists(source_lod_path) or not os.path.exists(target_lod_path):
            print(f"Skipping folder {folder_name} - missing LOD directories")
            continue
        
        # Generate entries for all azimuth and elevation combinations
        for azimuth in AZIMUTH_VALUES:
            for elevation in ELEVATION_VALUES:
                source_filename = f"lod{LOD_FROM}_az{azimuth}_el{elevation}.png"
                target_filename = f"lod{LOD_TO}_az{azimuth}_el{elevation}.png"
                
                source_path = os.path.join(source_lod_path, source_filename)
                target_path = os.path.join(target_lod_path, target_filename)
                
                # Check if both files exist
                if os.path.exists(source_path) and os.path.exists(target_path):
                    # Create absolute paths for the manifest
                    source_abs_path = source_path
                    target_abs_path = target_path
                    
                    manifest_entry = {
                        "source": source_abs_path,
                        "target": target_abs_path,
                        "prompt": TRAINING_PROMPT
                    }
                    manifest_entries.append(manifest_entry)
                else:
                    print(f"Missing files for {folder_name} - az{azimuth}_el{elevation}")
    
    # Write manifest to file
    with open(PROMPT_JSON_PATH, 'w') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Generated manifest with {len(manifest_entries)} entries")
    print(f"Saved to: {PROMPT_JSON_PATH}")
    print(f"LOD mapping: {LOD_FROM} -> {LOD_TO}")
    print(f"Prompt: {TRAINING_PROMPT}")

if __name__ == "__main__":
    generate_manifest()
