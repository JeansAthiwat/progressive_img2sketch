import json
import cv2
import numpy as np

from torch.utils.data import Dataset
        

from PIL import Image
from torch.utils.data import Dataset
import os
import random
import torchvision.transforms as T
import numpy as np
import cv2
from typing import Tuple


class ProgressiveSketchDataset(Dataset):
    def __init__(self, root="/home/athiwat/progressive_img2sketch/"  ,lod_from=3, lod_to=2, sanity_check=True):
        self.data = []
        
        if sanity_check:
            manifest_path = root + f'ControlNet/training/LOD50/prompt_sanity_check.json'
        else:
            manifest_path = root + f'ControlNet/training/LOD50_combined_sketch/prompt_from{lod_from}_to{lod_to}.json'
            
        with open(manifest_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_file = item['source']
        target_file = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_file)
        target = cv2.imread(target_file)

        # Do not forget that OpenCV read images in BGR order.
        # print(f"Loading source image: {source_file}")
        # print(f"Loading target image: {target_file}")
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)



class BuildingSketchDataset(Dataset):
    def __init__(self, data_root="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches", pair_from_to=(2, 1), resolution=512, augment=False):
        self.image_pairs = []
        self.resolution = resolution
        self.augment = augment
        # self.prompt = "minimalist architectural line‑art, svg‑style, isometric view, crisp black strokes, pure white background, no colours, no shading, no gradients"
        self.prompt = "A simplified architectural line drawing of a building, crisp black outlines on a pure white background, no colors, no shading, no gradients."
        self.source_lod = pair_from_to[0]
        self.target_lod = pair_from_to[1]
        self.azimuth_values = [f"{i:03d}" for i in range(0, 360, 15)]
        self.elevation_values = ["00", "15", "30"]

        self.collect_image_pairs(data_root)

    def collect_image_pairs(self, data_root):
        for folder_num in range(0, 48):
            folder_name = f"{folder_num}"
            source_lod_path = os.path.join(data_root, folder_name, f"lod{self.source_lod}")
            target_lod_path = os.path.join(data_root, folder_name, f"lod{self.target_lod}")
            if not os.path.exists(source_lod_path) or not os.path.exists(target_lod_path):
                continue
            for azimuth in self.azimuth_values:
                for elevation in self.elevation_values:
                    source_filename = f"lod{self.source_lod}_az{azimuth}_el{elevation}.png"
                    target_filename = f"lod{self.target_lod}_az{azimuth}_el{elevation}.png"
                    source_path = os.path.join(source_lod_path, source_filename)
                    target_path = os.path.join(target_lod_path, target_filename)
                    if os.path.exists(source_path) and os.path.exists(target_path):
                        self.image_pairs.append((source_path, target_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        sketch_path, image_path = self.image_pairs[idx]
        sketch = Image.open(sketch_path).convert("RGB")
        image = Image.open(image_path).convert("RGB")

        if self.resolution != 512:
            sketch = sketch.resize((self.resolution, self.resolution), Image.BICUBIC)
            image = image.resize((self.resolution, self.resolution), Image.BICUBIC)

        if self.augment:
            sketch, image = self.apply_paired_augmentations(sketch, image)

        sketch = np.array(sketch).astype(np.float32) / 255.0  # [0, 1]
        image = (np.array(image).astype(np.float32) / 127.5) - 1.0  # [-1, 1]

        return {
            "jpg": image,
            "txt": self.prompt,
            "hint": sketch
        }

    def apply_paired_augmentations(self, sketch: Image.Image, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            sketch = T.functional.hflip(sketch)
            image = T.functional.hflip(image)

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

        angle = random.uniform(-15, 15)
        sketch = T.functional.rotate(sketch, angle, fill=255, interpolation=T.InterpolationMode.NEAREST)
        image = T.functional.rotate(image, angle, fill=255, interpolation=T.InterpolationMode.NEAREST)

        return sketch, image
    

class BuildingColoredDataset(Dataset):
    def __init__(self, data_root="/home/athiwat/progressive_img2sketch/resources/LOD_orbit_images_BLENDER_WORKBENCH", pair_from_to=(3, 1), resolution=512, augment=True):
        self.image_pairs = []
        self.resolution = resolution
        self.augment = augment
        # self.prompt = "minimalist architectural line‑art, svg‑style, isometric view, crisp black strokes, pure white background, no colours, no shading, no gradients"
        self.prompt = "A simplified architectural building, white background, primitive shapes."
        self.source_lod = pair_from_to[0]
        self.target_lod = pair_from_to[1]
        self.azimuth_values = [f"{i:03d}" for i in range(0, 360, 15)]
        self.elevation_values = ["00", "15", "30"]

        self.collect_image_pairs(data_root)

    def collect_image_pairs(self, data_root):
        for folder_num in range(0, 51):
            folder_name = f"{folder_num}"
            source_lod_path = os.path.join(data_root, folder_name, f"lod{self.source_lod}")
            target_lod_path = os.path.join(data_root, folder_name, f"lod{self.target_lod}")
            if not os.path.exists(source_lod_path) or not os.path.exists(target_lod_path):
                continue
            for azimuth in self.azimuth_values:
                for elevation in self.elevation_values:
                    source_filename = f"lod{self.source_lod}_az{azimuth}_el{elevation}.png"
                    target_filename = f"lod{self.target_lod}_az{azimuth}_el{elevation}.png"
                    source_path = os.path.join(source_lod_path, source_filename)
                    target_path = os.path.join(target_lod_path, target_filename)
                    if os.path.exists(source_path) and os.path.exists(target_path):
                        self.image_pairs.append((source_path, target_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        sketch_path, image_path = self.image_pairs[idx]
        sketch = Image.open(sketch_path).convert("RGB")
        image = Image.open(image_path).convert("RGB")

        if self.resolution != 512:
            sketch = sketch.resize((self.resolution, self.resolution), Image.BICUBIC)
            image = image.resize((self.resolution, self.resolution), Image.BICUBIC)

        if self.augment:
            sketch, image = self.apply_paired_augmentations(sketch, image)

        sketch = np.array(sketch).astype(np.float32) / 255.0  # [0, 1]
        image = (np.array(image).astype(np.float32) / 127.5) - 1.0  # [-1, 1]

        return {
            "jpg": image,
            "txt": self.prompt,
            "hint": sketch
        }

    def apply_paired_augmentations(self, sketch: Image.Image, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            sketch = T.functional.hflip(sketch)
            image = T.functional.hflip(image)

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

        angle = random.uniform(-15, 15)
        sketch = T.functional.rotate(sketch, angle, fill=255, interpolation=T.InterpolationMode.NEAREST)
        image = T.functional.rotate(image, angle, fill=255, interpolation=T.InterpolationMode.NEAREST)

        return sketch, image
        
#test my dataset
if __name__ == "__main__":
    dataset = ProgressiveSketchDataset()
    dataset = BuildingSketchDataset(
        data_root="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think",
        pair_from_to=(2, 1),
        resolution=512,
        augment=True
    )
    print(f"Dataset length: {len(dataset)}")
    for i in range(5):
        item = dataset[i]
        print(f"Item {i}:")
        print(f"  Prompt: {item['txt']}")
        print(f"  Source shape: {item['hint'].shape}")
        print(f"  Target shape: {item['jpg'].shape}")
        print(f"  Source max value: {item['hint'].max()}")
        print(f"  Target max value: {item['jpg'].max()}")
        print(f"  Source min value: {item['hint'].min()}")
        print(f"  Target min value: {item['jpg'].min()}")
        print(f"  Source dtype: {item['hint'].dtype}")  
        
