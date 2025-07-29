from PIL import Image
from torch.utils.data import Dataset
import os
import random
import torchvision.transforms as T
from transformers import CLIPTokenizer
import numpy as np
import cv2
from typing import Tuple


class BuildingSketchDataset(Dataset):
    def __init__(self, data_root="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches", pair_from_to=(2,1), resolution=512, augment=False):
        self.image_pairs = []
        self.resolution = resolution
        self.augment = augment
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.prompt = "minimalist architectural line‑art, svg‑style, isometric view, crisp black strokes, pure white background, no colours, no shading, no gradients"
        # self.prompt = "A simplified architectural sketch of a building, with only black boundary lines and rough outlines on a white background. No colors, no shading."
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
        self.azimuth_values = [f"{i:03d}" for i in range(0, 360, 15)]  # 000, 015, 030, ..., 345
        self.elevation_values = ["00", "15", "30"]

        self.transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.RandomHorizontalFlip() if augment else T.Lambda(lambda x: x),
            T.ToTensor(),
            # T.Normalize([0.5], [0.5])
        ])
        #print out params info
        print(f"BuildingSketchDataset initialized with source LOD: {self.source_lod}, target LOD: {self.target_lod}, resolution: {self.resolution}, augment: {self.augment}")
        print(f"Prompt: {self.prompt}")
        
        self.collect_image_pairs(data_root)

    def collect_image_pairs(self, data_root):
        # Iterate through all folders (00-50)
        for folder_num in range(0, 48): # left some as a testset
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
                        
        print(f"Processed {folder_num + 1} with {len(self.image_pairs)} pairs.")
    


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

        if self.augment:
            sketch, image = self.apply_paired_augmentations(sketch, image)

        # To tensor
        sketch = T.ToTensor()(sketch)
        image = T.ToTensor()(image)

        return {
            "pixel_values": image,
            "conditioning_pixel_values": sketch,
            "input_ids": self.inputs.input_ids.squeeze(0),
        }


            
    def apply_paired_augmentations(self, sketch: Image.Image, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # 1. Random Horizontal Flip
        if random.random() < 0.5:
            sketch = T.functional.hflip(sketch)
            image = T.functional.hflip(image)

        # 2. Random Translation (white bg)
        max_dx = 0.05 * self.resolution
        max_dy = 0.05 * self.resolution
        translate_x = int(random.uniform(-max_dx, max_dx))
        translate_y = int(random.uniform(-max_dy, max_dy))
        sketch = T.functional.affine(
            sketch, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0,
            fill=255, interpolation=T.InterpolationMode.NEAREST
        )
        image = T.functional.affine(
            image, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0,
            fill=255, interpolation=T.InterpolationMode.NEAREST
        )

        # 3. Random Central Zoom (crop+resize)
        zoom_factor = random.uniform(0.8, 1.0)
        crop_size = int(self.resolution * zoom_factor)
        i = (self.resolution - crop_size) // 2
        sketch = T.functional.resized_crop(
            sketch, i, i, crop_size, crop_size, (self.resolution, self.resolution),
            interpolation=T.InterpolationMode.NEAREST
        )
        image = T.functional.resized_crop(
            image, i, i, crop_size, crop_size, (self.resolution, self.resolution),
            interpolation=T.InterpolationMode.NEAREST
        )

        # def dilate(img_pil, iterations):
        #     img_gray = np.array(img_pil.convert("L"))  # convert to grayscale
        #     inverted = 255 - img_gray                  # strokes become white

        #     kernel = np.ones((3, 3), np.uint8)
        #     dilated = cv2.dilate(inverted, kernel, iterations=iterations)

        #     result = 255 - dilated  # revert back to black strokes
        #     return Image.fromarray(result).convert("RGB")

        # # 4. Stroke Thickening (dilate both sketch + image)
        # iterations = random.randint(0, 1)
        # sketch = dilate(sketch, iterations)
        # image = dilate(image, iterations)


        angle = random.uniform(-15, 15)
        sketch = T.functional.rotate(sketch, angle, fill=255, interpolation=T.InterpolationMode.NEAREST)
        image = T.functional.rotate(image, angle, fill=255, interpolation=T.InterpolationMode.NEAREST)

        return sketch, image



class SchoolScribbleDataset(Dataset):
    def __init__(self, data_root="/home/athiwat/progressive_img2sketch/resources/LOD50_opaque_normalized_1radius_triangulated_fix_normals_orbits_with_depth_cropped_512x512/sketch", pair_from_to=(2,1), resolution=512, augment=True):
        self.image_pairs = []
        self.resolution = resolution
        self.augment = augment
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.prompt = "minimalist architectural line‑art, svg‑style, isometric view, crisp black strokes, pure white background, no colours, no shading, no gradients"
        # self.prompt = "A simplified architectural sketch of a building, with only black boundary lines and rough outlines on a white background. No colors, no shading."
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
        self.azimuth_values = [f"{i:03d}" for i in range(0, 360, 15)]  # 000, 015, 030, ..., 345
        self.elevation_values = ["00", "15", "30"]

        self.transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.RandomHorizontalFlip() if augment else T.Lambda(lambda x: x),
            T.ToTensor(),
            # T.Normalize([0.5], [0.5])
        ])
        #print out params info
        print(f"BuildingSketchDataset initialized with source LOD: {self.source_lod}, target LOD: {self.target_lod}, resolution: {self.resolution}, augment: {self.augment}")
        print(f"Prompt: {self.prompt}")
        
        self.collect_image_pairs(data_root)

    def collect_image_pairs(self, data_root):
        # Iterate through all folders (00-50)
        for folder_num in range(0, 48): # left some as a testset
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
                        
        print(f"Processed {folder_num + 1} with {len(self.image_pairs)} pairs.")
    


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

        if self.augment:
            sketch, image = self.apply_paired_augmentations(sketch, image)

        # To tensor
        sketch = T.ToTensor()(sketch)
        image = T.ToTensor()(image)

        return {
            "pixel_values": image,
            "conditioning_pixel_values": sketch,
            "input_ids": self.inputs.input_ids.squeeze(0),
        }


            
    def apply_paired_augmentations(self, sketch: Image.Image, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # 1. Random Horizontal Flip
        if random.random() < 0.5:
            sketch = T.functional.hflip(sketch)
            image = T.functional.hflip(image)

        # 2. Random Translation (white bg)
        max_dx = 0.05 * self.resolution
        max_dy = 0.05 * self.resolution
        translate_x = int(random.uniform(-max_dx, max_dx))
        translate_y = int(random.uniform(-max_dy, max_dy))
        sketch = T.functional.affine(
            sketch, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0,
            fill=255, interpolation=T.InterpolationMode.NEAREST
        )
        image = T.functional.affine(
            image, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0,
            fill=255, interpolation=T.InterpolationMode.NEAREST
        )

        # 3. Random Central Zoom (crop+resize)
        zoom_factor = random.uniform(0.8, 1.0)
        crop_size = int(self.resolution * zoom_factor)
        i = (self.resolution - crop_size) // 2
        sketch = T.functional.resized_crop(
            sketch, i, i, crop_size, crop_size, (self.resolution, self.resolution),
            interpolation=T.InterpolationMode.NEAREST
        )
        image = T.functional.resized_crop(
            image, i, i, crop_size, crop_size, (self.resolution, self.resolution),
            interpolation=T.InterpolationMode.NEAREST
        )

        # def dilate(img_pil, iterations):
        #     img_gray = np.array(img_pil.convert("L"))  # convert to grayscale
        #     inverted = 255 - img_gray                  # strokes become white

        #     kernel = np.ones((3, 3), np.uint8)
        #     dilated = cv2.dilate(inverted, kernel, iterations=iterations)

        #     result = 255 - dilated  # revert back to black strokes
        #     return Image.fromarray(result).convert("RGB")

        # # 4. Stroke Thickening (dilate both sketch + image)
        # iterations = random.randint(0, 1)
        # sketch = dilate(sketch, iterations)
        # image = dilate(image, iterations)


        angle = random.uniform(-15, 15)
        sketch = T.functional.rotate(sketch, angle, fill=255, interpolation=T.InterpolationMode.NEAREST)
        image = T.functional.rotate(image, angle, fill=255, interpolation=T.InterpolationMode.NEAREST)

        return sketch, image