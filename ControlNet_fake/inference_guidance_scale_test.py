import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from progressive_sketch_dataset import BuildingSketchDataset
from cldm.model import create_model
from cldm.ddim_hacked import DDIMSampler
import config


def load_model_from_checkpoint(checkpoint_path, config_path, device):
    print(f"Loading model from {checkpoint_path}")
    model = create_model(config_path).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("✅ Model loaded")
    return model


def inference_single_batch(model, hints_np, prompts, ddim_steps, guidance_scale, eta):
    device = next(model.parameters()).device
    # [B,H,W,C] → [B,3,H,W] on `device`
    hints = torch.from_numpy(hints_np).permute(0,3,1,2).to(device)  
    B = hints.shape[0]

    c_cat   = hints
    c_cross = model.get_learned_conditioning(prompts)
    uc_cross= model.get_unconditional_conditioning(B)
    uc_cat  = c_cat

    cond   = {"c_concat":[c_cat],   "c_crossattn":[c_cross]}
    uc_full= {"c_concat":[uc_cat],  "c_crossattn":[uc_cross]}

    sampler = DDIMSampler(model)
    with torch.no_grad():
        shape = (model.channels, hints.shape[2]//8, hints.shape[3]//8)
        samples, _ = sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=B,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc_full,
            eta=eta
        )
        x = model.decode_first_stage(samples)
        x = torch.clamp((x + 1.0)/2.0, 0.0, 1.0)

    return x.cpu()  # back to CPU for easy concatenation


def main():
    # ─── Config ────────────────────────────────────────────────
    checkpoint_path = "/home/athiwat/progressive_img2sketch/ControlNet/models/22-07-2025_LOD2to1_unfroze_at_finals_epoch/checkpoints/phase2-epoch=07-step=1727.ckpt"

    config_path = "./models/cldm_v15.yaml"
    output_dir = "./inference_outputs/guidance_scale_experiment"
    os.makedirs(output_dir, exist_ok=True)

    batch_size     = 4
    ddim_steps     = 50
    eta            = 0.0
    guidance_scales= [1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ─── Load model & data ──────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path, config_path, device)

    ds = BuildingSketchDataset(
        data_root="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think",
        pair_from_to=(2,1),
        resolution=512,
        augment=False
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Dataset size: {len(ds)} samples, {len(loader)} batches")

    # ─── Grab a single batch ─────────────────────────────────────
    batch = next(iter(loader))
    hints_np = batch["hint"].numpy()       # [B,H,W,C]
    txt_field= batch["txt"]
    # unify to list of strings
    prompts = txt_field if isinstance(txt_field, list) else [txt_field]*hints_np.shape[0]

    # prepare ground-truth & input for saving ([B,C,H,W] CPU)
    gt = batch["jpg"]                       # [-1,1]
    gt = ((gt+1.0)/2.0).permute(0,3,1,2)     # [B,C,H,W]
    hints_cpu = torch.from_numpy(hints_np).permute(0,3,1,2)

    # ─── Run experiment ──────────────────────────────────────────
    for gs in guidance_scales:
        print(f"\n▶ Running batch with guidance_scale = {gs}")
        results = inference_single_batch(
            model, hints_np, prompts,
            ddim_steps, guidance_scale=gs, eta=eta
        )  # [B, C, H, W] on CPU

        # build one row per sample: [C, H, 3W]
        comps = []
        for i in range(results.size(0)):
            inp   = hints_cpu[i]
            gen   = results[i]
            truth = gt[i]
            comps.append(torch.cat([inp, gen, truth], dim=2))

        # stack into [B, C, H, 3W]
        grid = torch.stack(comps, dim=0)
        out_path = os.path.join(output_dir, f"guidance_{gs:.1f}.png")
        # save: each row=a sample, columns=[input｜gen｜gt]
        save_image(grid, out_path, nrow=1)
        print("Saved", out_path)

    print("\n✅ Done! Check your grids in", output_dir)


if __name__ == "__main__":
    main()
