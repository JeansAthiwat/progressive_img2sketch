import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from progressive_sketch_dataset import BuildingSketchDataset
from cldm.model import create_model, load_state_dict
from share import *

# ---------- Config ----------
checkpoint_path = "/home/athiwat/progressive_img2sketch/ControlNet/models/22-07-2025_LOD2to1_unfroze_at_finals_epoch/checkpoints/phase2-epoch=03-step=863.ckpt"  # Update this
output_dir = "./inference_outputs/22-07-2025_LOD2to1"
os.makedirs(output_dir, exist_ok=True)
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load model ----------
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(checkpoint_path, location='cpu'))
model = model.to(device)
model.eval()

# ---------- Load dataset ----------
dataset = BuildingSketchDataset(
    data_root="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think",
    pair_from_to=(2, 1),
    resolution=512,
    augment=False
)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)

# ---------- Inference ----------
with torch.no_grad():
    for idx, batch in enumerate(tqdm(dataloader)):
        hint = batch["hint"].permute(0, 3, 1, 2).to(device)          # [B, H, W, C] -> [B, C, H, W]
        prompt = batch["txt"]                                         # str
        gt = batch["jpg"].permute(0, 3, 1, 2).to(device)              # [0,1] range
        
        # ---- forward() should accept hint and prompt ----
        cond = {"hint": hint, "txt": prompt}
        result = model.sample(cond)
                # or model.forward(hint, prompt)

        # ---- Save ----
        save_image(hint, os.path.join(output_dir, f"{idx:04d}_input_hint.png"))
        save_image(result, os.path.join(output_dir, f"{idx:04d}_pred.png"))
        save_image(gt, os.path.join(output_dir, f"{idx:04d}_gt.png"))

print("✅ Inference complete.")
