from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ----- Configuration -----
base_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
controlnet_path = "/mnt/nas/athiwat/ControlNet_models/22-07-2025-lod2to1-canny_with_freestylegreyscale"
input_dir = "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think"
output_dir = f"/home/athiwat/progressive_img2sketch/resources/{os.path.basename(controlnet_path)}-inference-output"
prompt = "minimalist architectural line‑art, svg‑style, isometric view, crisp black strokes, pure white background, no colours, no shading, no gradients"
max_samples = 16
selected_lod = "lod2"  # 🔄 Change this to "lod2" if needed

# ----- Load Pipeline -----
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float32
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# ----- Utility: grid image maker -----
def save_grid(control_imgs, result_imgs, file_names, scene_id, save_path):
    rows = len(control_imgs)
    fig, axes = plt.subplots(rows, 2, figsize=(6, rows * 3))
    for i in range(rows):
        for j in range(2):
            ax = axes[i, j] if rows > 1 else axes[j]
            img = control_imgs[i] if j == 0 else result_imgs[i]
            label = "Control" if j == 0 else "Generated"
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_title(f"Scene {scene_id} - {file_names[i]}", fontsize=8, loc='left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def inference_with_conditioning_scales(
    pipe,
    control_image: Image.Image,
    prompt: str,
    scales: list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    num_inference_steps: int = 20,
    generator_seed: int = 0,
    output_base_path: str = "./",
    image_name: str = "test"
):
    """Run inference on the same control image using multiple conditioning scales."""
    control_image = control_image.resize((512, 512))  # Make sure it's the right size
    generator = torch.manual_seed(generator_seed)

    results = []
    for scale in scales:
        print(f"Running inference with conditioning_scale={scale}")
        result = pipe(
            prompt=prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
            conditioning_scale=scale,  # 👈 This is the key
        ).images[0]
        results.append(result.convert("RGB"))

    # Plot comparison
    fig, axes = plt.subplots(1, len(scales) + 1, figsize=(3 * (len(scales) + 1), 3))
    axes[0].imshow(control_image.convert("RGB"))
    axes[0].set_title("Control")
    axes[0].axis("off")

    for i, (img, scale) in enumerate(zip(results, scales), 1):
        axes[i].imshow(img)
        axes[i].set_title(f"Scale {scale}")
        axes[i].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_base_path, f"{image_name}_scale_grid.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved conditioning scale test grid to {save_path}")


# ----- Inference Loop -----
os.makedirs(output_dir, exist_ok=True)

# Just test on one control image
test_image_path = "/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think/50/lod2/lod2_az075_el15.png"
test_image = load_image(test_image_path)

inference_with_conditioning_scales(
    pipe=pipe,
    control_image=test_image,
    prompt=prompt,
    output_base_path=output_dir,
    image_name="scene50_example"
)


# for scene_num in range(50, -1, -1):
#     scene = str(scene_num)
#     scene_path = os.path.join(input_dir, scene)
#     lod_path = os.path.join(scene_path, selected_lod)
#     if not os.path.isdir(lod_path):
#         continue

#     control_images = []
#     result_images = []
#     file_names = []
#     count = 0

#     for image_name in sorted(os.listdir(lod_path)):
#         if not image_name.endswith(".png"):
#             continue

#         file_names.append(image_name)
#         image_path = os.path.join(lod_path, image_name)
#         control_image = load_image(image_path)

#         generator = torch.manual_seed(0)
#         result = pipe(
#             prompt,
#             num_inference_steps=20,
#             generator=generator,
#             image=control_image
#         ).images[0]

#         control_images.append(control_image.convert("RGB"))
#         result_images.append(result.convert("RGB"))

#         count += 1
#         if count >= max_samples:
#             break

#     if control_images and result_images:
#         save_path = os.path.join(output_dir, f"{scene}_grid_{prompt.replace(',','').replace(' ','_')}.png")
#         save_grid(control_images, result_images, file_names, scene, save_path)
#         print(f"Saved: {save_path}")
