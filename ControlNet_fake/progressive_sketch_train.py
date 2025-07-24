from share import *

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from progressive_sketch_dataset import BuildingSketchDataset, BuildingColoredDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# ---------------- Config ----------------
run_name = "23-07-2025_LOD2to1_colored_unfroze_at_finals_epoch"
output_root = os.path.join('./models', run_name)
checkpoint_dir = os.path.join(output_root, 'checkpoints')
initial_ckpt_path = os.path.join(output_root, "control_sd15_ini_23-07-2025_LOD2to1_colored.ckpt")

# Shared config
batch_size = 8
accumulate_grad_batches = 4
logger_freq = 200
image_logger = ImageLogger(batch_frequency=logger_freq)


# ---------------- Dataset ----------------
dataset = BuildingSketchDataset(
    data_root="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches_best_i_think",
    pair_from_to=(2, 1),
    resolution=512,
    augment=True
)
# dataset = BuildingColoredDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

# # ---------------- Trainer 1 (Phase 1) ----------------
# # Train 12 epochs with sd_locked = True, lr = 1e-5
# model1 = create_model('./models/cldm_v15.yaml').cpu()
# model1.load_state_dict(load_state_dict(initial_ckpt_path, location='cpu'))
# model1.learning_rate = 1e-5
# model1.sd_locked = True
# model1.only_mid_control = False

wandb_logger1 = WandbLogger(
    name=run_name + "_phase1",
    project="ControlNet_LOD_Simplification",
    save_dir=output_root,
    log_model=False,
)

checkpoint_callback1 = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="phase1-{epoch:02d}-{step}",
    save_top_k=-1,
    every_n_epochs=1,
    save_on_train_epoch_end=True
)

trainer1 = pl.Trainer(
    gpus=1,
    precision=32,
    logger=wandb_logger1,
    callbacks=[image_logger, checkpoint_callback1],
    max_epochs=16,
    accumulate_grad_batches=accumulate_grad_batches
)

# # trainer1.fit(model1, dataloader, ckpt_path="/home/athiwat/progressive_img2sketch/ControlNet/models/22-07-2025_LOD2to1_unfroze_at_finals_epoch/checkpoints/phase1-epochepoch=00-stepstep=215.ckpt")
# trainer1.fit(model1, dataloader)

# # ---------------- Trainer 2 (Phase 2) ----------------
# # Train 4 more epochs with sd_locked = False, lr = 2e-6

# Load from latest checkpoint from phase 1
latest_ckpt = checkpoint_callback1.best_model_path or trainer1.checkpoint_callback.last_model_path


model2 = create_model('./models/cldm_v15.yaml').cpu()
model2.load_state_dict(load_state_dict(latest_ckpt, location='cpu'))
model2.learning_rate = 2e-6
model2.sd_locked = False
model2.only_mid_control = False

wandb_logger2 = WandbLogger(
    name=run_name + "_phase2",
    project="ControlNet_LOD_Simplification",
    save_dir=output_root,
    log_model=False,
)

checkpoint_callback2 = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="phase2-{epoch:02d}-{step}",
    save_top_k=-1,
    every_n_epochs=1,
    save_on_train_epoch_end=True
)

trainer2 = pl.Trainer(
    gpus=1,
    precision=32,
    logger=wandb_logger2,
    callbacks=[image_logger, checkpoint_callback2],
    max_epochs=8,
    accumulate_grad_batches=accumulate_grad_batches
)

trainer2.fit(model2, dataloader,ckpt_path="/home/athiwat/progressive_img2sketch/ControlNet/models/22-07-2025_LOD2to1_unfroze_at_finals_epoch/checkpoints/phase2-epoch=03-step=863.ckpt")



# from share import *

# import os
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger

# from progressive_sketch_dataset import BuildingSketchDataset
# from cldm.logger import ImageLogger
# from cldm.model import create_model, load_state_dict

# # ---------------- Config ----------------
# run_name = "22-07-2025_LOD2to1_unfroze_at_finals_epoch"
# output_root = os.path.join('./models', run_name)
# checkpoint_dir = os.path.join(output_root, 'checkpoints')

# resume_path = '/home/athiwat/progressive_img2sketch/ControlNet/models/22-07-2025_LOD2to1_unfroze_at_finals_epoch/control_sd15_ini_22-07-2025_LOD2to1.ckpt'
# batch_size = 8
# logger_freq = 300
# learning_rate = 1e-5
# sd_locked = True
# only_mid_control = False
# max_epochs = 8

# # ---------------- Model Setup ----------------
# model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'))
# model.learning_rate = learning_rate
# model.sd_locked = sd_locked
# model.only_mid_control = only_mid_control

# # ---------------- Dataset & Dataloader ----------------
# dataset = BuildingSketchDataset(
#     data_root="/home/athiwat/progressive_img2sketch/resources/LOD_combined_sketches",
#     pair_from_to=(2, 1),
#     resolution=512,
#     augment=True
# )
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

# # ---------------- Logging & Callbacks ----------------
# wandb_logger = WandbLogger(
#     name=run_name,
#     project="ControlNet_LOD_Simplification",
#     save_dir=output_root,  # saves wandb logs to the same root
#     log_model=False,
# )

# image_logger = ImageLogger(batch_frequency=logger_freq)

# checkpoint_callback = ModelCheckpoint(
#     dirpath=checkpoint_dir,
#     filename="epoch{epoch:02d}-step{step}",
#     save_top_k=-1,
#     every_n_epochs=1,
#     save_on_train_epoch_end=True
# )

# # ---------------- Trainer ----------------
# trainer = pl.Trainer(
#     gpus=1,
#     precision=32,
#     logger=wandb_logger,
#     callbacks=[image_logger, checkpoint_callback],
#     max_epochs=max_epochs
# )

# # ---------------- Train ----------------
# trainer.fit(model, dataloader)
