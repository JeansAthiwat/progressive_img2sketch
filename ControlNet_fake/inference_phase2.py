import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

from progressive_sketch_dataset import BuildingSketchDataset
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config


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


def inference_single_image(model, hint, prompt, ddim_steps=50, guidance_scale=9.0, eta=0.0):
    """Run inference on a single image"""
    device = next(model.parameters()).device
    
    # Prepare inputs
    if isinstance(hint, np.ndarray):
        hint = torch.from_numpy(hint)
    if len(hint.shape) == 3:
        hint = hint.unsqueeze(0)  # Add batch dimension
    hint = hint.to(device)
    
    # Ensure hint is in correct format [B, C, H, W] and range [0, 1]
    if hint.shape[1] != 3:  # If [B, H, W, C]
        hint = hint.permute(0, 3, 1, 2)
    
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
    # ─── Configuration ────────────────────────────────────────────────
    checkpoint_path = "/home/athiwat/progressive_img2sketch/ControlNet/models/22-07-2025_LOD2to1_unfroze_at_finals_epoch/checkpoints/phase2-epoch=07-step=1727.ckpt"
    config_path = "./models/cldm_v15.yaml"
    output_dir = "./inference_outputs/22-07-2025_LOD2to1_phase2"
    
    # Inference settings
    batch_size = 4
    ddim_steps = 50
    guidance_scale = 9.0
    eta = 0.0
    max_samples = 100  # Set to None to process all images
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ─── Load Model ───────────────────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path, config_path, device)
    
    # ─── Load Dataset ─────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = BuildingSketchDataset(
        data_root="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think",
        pair_from_to=(2, 1),  # LOD 2 to LOD 1
        resolution=512,
        augment=False  # No augmentation for inference
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # ─── Inference ────────────────────────────────────────────────────
    print("Starting inference...")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            if max_samples is not None and idx >= max_samples:
                break
                
            # Extract batch data
            hint = batch["hint"]  # Input sketch [B, H, W, C] in [0, 1]
            prompt = batch["txt"][0] if isinstance(batch["txt"], list) else batch["txt"]
            gt = batch["jpg"]     # Ground truth [B, H, W, C] in [-1, 1]
            
            # Convert ground truth to [0, 1] for saving
            gt_normalized = (gt + 1.0) / 2.0
            gt_normalized = gt_normalized.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # Convert hint for saving (already in [0, 1])
            hint_for_save = hint.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # Run inference
            # Run inference
            try:
                result = inference_single_image(
                    model, hint, prompt,
                    ddim_steps=ddim_steps,
                    guidance_scale=guidance_scale,
                    eta=eta
                )  # result is on CUDA

                # ←---- ADD THIS LINE to move it to CPU
                result = result.cpu()  # now [B, C, H, W] on CPU

                # build one grid per batch: each row = one sample
                comps = []
                for j in range(result.size(0)):
                    inp   = hint_for_save[j]     # CPU
                    gen   = result[j]            # CPU
                    truth = gt_normalized[j]     # CPU
                    comps.append(torch.cat([inp, gen, truth], dim=2))
                comps = torch.stack(comps, dim=0)

                out_path = os.path.join(output_dir, f"{idx:04d}_batch_comparison.png")
                save_image(comps, out_path, nrow=1)

            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue


    
    print(f"✅ Inference complete! Results saved to: {output_dir}")
    print(f"Generated {min(max_samples or len(dataset), len(dataset))} samples")


if __name__ == "__main__":
    main()
