#!/usr/bin/env python3
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from progressive_sketch_dataset import BuildingSketchDataset
from cldm.logger import ImageLogger
from cldm.model import create_model

def main():
    # ─── Config ───────────────────────────────────────────────────────
    run_name       = "22-07-2025_LOD2to1_unfroze_at_finals_epoch"
    output_root    = os.path.join("./models", run_name)
    checkpoint_dir = os.path.join(output_root, "checkpoints")
    resume_ckpt    = (
        "/home/athiwat/progressive_img2sketch/ControlNet/models/"
        "22-07-2025_LOD2to1_unfroze_at_finals_epoch/"
        "checkpoints/phase2-epoch=03-step=863.ckpt"
    )

    batch_size              = 4
    accumulate_grad_batches = 4
    logger_freq             = 200  # log images every 200 steps

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ─── Dataset & Dataloader ─────────────────────────────────────────
    dataset = BuildingSketchDataset(
        data_root="/home/athiwat/progressive_img2sketch/resources/"
                  "LOD_combined_sketches_best_i_think",
        pair_from_to=(2, 1),
        resolution=512,
        augment=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # ─── Model Setup ──────────────────────────────────────────────────
    model = create_model("./models/cldm_v15.yaml").cpu()
    # override Phase 2 hyperparams
    model.learning_rate    = 2e-6
    model.sd_locked        = False
    model.only_mid_control = False

    # ─── Logging & Checkpointing ─────────────────────────────────────
    image_logger = ImageLogger(batch_frequency=logger_freq)
    wandb_logger = WandbLogger(
        name=run_name + "_phase2",
        project="ControlNet_LOD_Simplification",
        save_dir=output_root,
        log_model=False,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="phase2-{epoch:02d}-{step}",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    # ─── Trainer (resume from epoch 3, continue → epoch 8) ─────────────
    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        logger=wandb_logger,
        callbacks=[image_logger, checkpoint_callback],
        max_epochs=8,                   # stops after epoch 8 total
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(model, dataloader, ckpt_path=resume_ckpt)

if __name__ == "__main__":
    main()
