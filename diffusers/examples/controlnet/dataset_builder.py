from PIL import Image
from torch.utils.data import Dataset
import os
import random
import torchvision.transforms as T
from transformers import CLIPTokenizer
import numpy as np
import cv2
from typing import Tuple
from transformers import CLIPTextModel


class SchoolRasterSketchDataset(Dataset):
    def __init__(self, data_root="/home/athiwat/progressive_img2sketch/resources/LOD50_opaque_normalized_1radius_triangulated_fix_normals_orbits_with_depth_cropped_512x512/log", 
                 tokenizer=None,
                 prompt="simplified architectural scribble, crisp black strokes, pure white background, no colours, no shading, no gradients, no windows no doors, no details",
                 pair_from_to=(2,1), 
                 resolution=512, 
                 augment=True):
        self.image_pairs = []
        self.resolution = resolution
        self.augment = augment
        self.good_scenes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13 ,15, 16, 18, 19,20,21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 39, 40, 41, 42, 44, 45, 46, 47] # lets just leave them for testing# 48broken, 49, 50], 49, 50
        self.prompt = prompt
        
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        self.tokenizer = tokenizer

        self.inputs = self.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        self.source_lod = pair_from_to[0]
        self.target_lod = pair_from_to[1]
        
        # Azimuth and elevation values (based on the file pattern)
        self.azimuth_values = [f"{i:03d}" for i in range(0, 360, 10)]  # 000, 010, 020, ..., 350
        self.elevation_values = ["00", "10", "20", "30", "40", "50", "60"]

        self.image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        
        self.conditioning_image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        
        #print out params info
        print(f"BuildingSketchDataset initialized with source LOD: {self.source_lod}, target LOD: {self.target_lod}, resolution: {self.resolution}, augment: {self.augment}")
        print(f"Prompt: {self.prompt}")
        
        self.collect_image_pairs(data_root)

    def collect_image_pairs(self, data_root):
        # Iterate through all folders (00-50)
        for folder_num in self.good_scenes:
            folder_name = f"{folder_num}"
            
            # Check if both source and target LOD folders exist
            source_lod_path = os.path.join(data_root, folder_name, f"lod{self.source_lod}")
            target_lod_path = os.path.join(data_root, folder_name, f"lod{self.target_lod}")
            
            if not os.path.exists(source_lod_path) or not os.path.exists(target_lod_path):
                print(f"Skipping folder {folder_name} - missing LOD directories")
                continue
            
            # Generate entries for all azimuth and elevation combinations
            for azimuth in self.azimuth_values:
                for elevation in self.elevation_values:
                    source_filename = f"lod{self.source_lod}_az{azimuth}_el{elevation}.png"
                    target_filename = f"lod{self.target_lod}_az{azimuth}_el{elevation}.png"
                    
                    source_path = os.path.join(source_lod_path, source_filename)
                    target_path = os.path.join(target_lod_path, target_filename)
                    
                    # Check if both files exist
                    if os.path.exists(source_path) and os.path.exists(target_path):
                        # Create absolute paths for the manifest
                        source_abs_path = source_path
                        target_abs_path = target_path

                        self.image_pairs.append((source_abs_path, target_abs_path))
                    else:
                        print(f"Missing files for {folder_name} - az{azimuth}_el{elevation}")
                        
        print(f"Processed {len(self.good_scenes)} with {len(self.image_pairs)} pairs.")
    


    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        sketch_path, image_path = self.image_pairs[idx]
        sketch = Image.open(sketch_path).convert("RGB")
        image = Image.open(image_path).convert("RGB")

        # Resize if resolution is not 512
        if self.resolution != 512:
            sketch = sketch.resize((self.resolution, self.resolution), Image.BICUBIC)
            image = image.resize((self.resolution, self.resolution), Image.BICUBIC)
            print(f"Resized images to {self.resolution}x{self.resolution}")

        if self.augment:
            sketch, image = self.apply_paired_augmentations(sketch, image)

        # To tensor
        sketch = self.image_transforms(sketch)
        image = self.image_transforms(image)

        return {
            "pixel_values": image,
            "conditioning_pixel_values": sketch,
            "input_ids": self.inputs.input_ids.squeeze(0),
        }


            
    def apply_paired_augmentations(self, sketch: Image.Image, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            sketch = T.functional.hflip(sketch)
            image = T.functional.hflip(image)

        angle = random.uniform(-15, 15)
        sketch = T.functional.rotate(sketch, angle, fill=255, interpolation=T.InterpolationMode.NEAREST)
        image = T.functional.rotate(image, angle, fill=255, interpolation=T.InterpolationMode.NEAREST)

        return sketch, image