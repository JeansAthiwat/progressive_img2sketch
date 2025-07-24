#!/usr/bin/env python3
"""
Batch inference script for processing multiple images with trained ControlNet
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import argparse
from pathlib import Path
from tqdm import tqdm

from cldm.model import create_model
from cldm.ddim_hacked import DDIMSampler


def load_model_from_checkpoint(checkpoint_path, config_path, device):
    """Load the trained ControlNet model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = create_model(config_path).cpu()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict (Lightning checkpoint format)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict into model
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def preprocess_image(image_path, resolution=512):
    """Load and preprocess an input sketch image"""
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((resolution, resolution), Image.BICUBIC)
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor [C, H, W]
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def inference_single_image(model, hint, prompt, ddim_steps=50, guidance_scale=9.0, eta=0.0):
    """Run inference on a single image"""
    device = next(model.parameters()).device
    
    # Add batch dimension if needed
    if len(hint.shape) == 3:
        hint = hint.unsqueeze(0)
    hint = hint.to(device)
    
    # Prepare conditioning
    batch_size = hint.shape[0]
    c_cat = hint  # Control image
    c_cross = model.get_learned_conditioning([prompt] * batch_size)  # Text conditioning
    
    # Unconditional conditioning for guidance
    uc_cross = model.get_unconditional_conditioning(batch_size)
    uc_cat = c_cat  # Keep the same control image
    
    # Prepare conditioning dict
    cond = {"c_concat": [c_cat], "c_crossattn": [c_cross]}
    uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
    
    # Setup DDIM sampler
    ddim_sampler = DDIMSampler(model)
    
    # Sample
    with torch.no_grad():
        shape = (model.channels, hint.shape[2] // 8, hint.shape[3] // 8)  # Latent space shape
        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc_full,
            eta=eta
        )
        
        # Decode to image space
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
    return x_samples


def main():
    parser = argparse.ArgumentParser(description="Batch ControlNet inference")
    parser.add_argument("--input_dir", "-i", required=True, help="Input directory containing sketch images")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory for generated images")
    parser.add_argument("--prompt", "-p", default="architectural building sketch", help="Text prompt")
    parser.add_argument("--checkpoint", "-c", 
                        default="/home/athiwat/progressive_img2sketch/ControlNet/models/22-07-2025_LOD2to1_unfroze_at_finals_epoch/checkpoints/phase2-epoch=03-step=863.ckpt",
                        help="Checkpoint path")
    parser.add_argument("--config", 
                        default="./models/cldm_v15.yaml",
                        help="Model config path")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--guidance", type=float, default=9.0, help="Guidance scale")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg"], 
                        help="Image file extensions to process")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.config, device)
    
    # Find all image files
    input_path = Path(args.input_dir)
    image_files = []
    for ext in args.extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    failed = 0
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load and preprocess image
            hint = preprocess_image(str(image_file), args.resolution)
            if hint is None:
                failed += 1
                continue
            
            # Run inference
            result = inference_single_image(
                model, hint, args.prompt, 
                ddim_steps=args.steps, 
                guidance_scale=args.guidance
            )
            
            # Save results
            output_name = image_file.stem
            output_path = os.path.join(args.output_dir, f"{output_name}_generated.png")
            comparison_path = os.path.join(args.output_dir, f"{output_name}_comparison.png")
            
            save_image(result, output_path)
            
            # Create comparison
            hint_display = hint.unsqueeze(0) if len(hint.shape) == 3 else hint
            comparison = torch.cat([hint_display, result], dim=3)  # Concatenate horizontally
            save_image(comparison, comparison_path)
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            failed += 1
            continue
    
    print(f"✅ Batch inference complete!")
    print(f"Successfully processed: {successful} images")
    print(f"Failed: {failed} images")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
