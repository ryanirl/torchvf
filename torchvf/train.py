# Copyright 2022 Ryan Peters
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim 
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
import shutil
import time
import os

from dataloaders import *
from transforms import *
from numerics import *
from models import *
from losses import *
from utils import *

# I am actively looking into a better way to do configs.
from ml_collections.config_flags.config_flags import _ConfigFileParser

parser = argparse.ArgumentParser()
parser.add_argument("--config_dir", default = "./configs/training/bpcis_bact_fluor.py")
args = parser.parse_args()

FileParser = _ConfigFileParser(name = "train")
cfg = FileParser.parse(args.config_dir)

# When using GPU, have this on.
torch.backends.cudnn.benchmark = True

config_dir = os.path.join("./configs", cfg.CONFIG_DIR)
save_to    = next_model(cfg.WEIGHT_DIR)

shutil.copyfile(config_dir, os.path.join(save_to, "config.py"))

######################################## 
############# DATA LOADER ############## 
######################################## 

train_dataset = BPCIS(
    cfg.DATA.DIR, 
    split = cfg.DATA.SPLIT,
    vf = cfg.DATA.VF, 
    vf_delimeter = cfg.DATA.VF_DELIM,
    transforms = transforms[cfg.DATA.TRANSFORMS], 
    remove = cfg.DATA.REMOVE, 
    copy = cfg.DATA.COPY
)

train_dataloader = MultiEpochsDataLoader(
    train_dataset, 
    batch_size = cfg.BATCH_SIZE,
    pin_memory = True,
    drop_last = True,
    shuffle = True,
    num_workers = 4
)

######################################## 
############ MODEL & OPTIM ############# 
######################################## 

model = get_model(
    cfg.MODEL_TYPE,
    in_channels = cfg.DATA.C,
    out_channels = [1, 2],
    device = cfg.DEVICE
)

optimizer = optim.Adam(model.parameters(), lr = cfg.LR)

if cfg.PRETRAINED:
    model, optimizer, epoch = load_checkpoint(cfg.PRETRAINED_DIR, model, optimizer)  

######################################## 
############ LOSS FUNCTIONS ############ 
######################################## 

vf_losses  = []
sem_losses = []

if cfg.LOSS.IVP.APPLY:
    vf_losses.append([
        "IVP Loss", 
        IVPLoss(
            dx = cfg.LOSS.IVP.DX,
            n_steps = cfg.LOSS.IVP.STEPS,
            solver = cfg.LOSS.IVP.SOLVER,
            device = cfg.DEVICE 
        ) 
    ])

if cfg.LOSS.MSE.APPLY:
    vf_losses.append([
        "MSE Loss", 
        nn.MSELoss()
    ])

if cfg.LOSS.TVERSKY.APPLY:
    sem_losses.append([
        "Tversky Loss",
        TverskyLoss(
            alpha = cfg.LOSS.TVERSKY.ALPHA, 
            beta = cfg.LOSS.TVERSKY.BETA, 
            from_logits = cfg.LOSS.TVERSKY.FROM_LOGITS 
        )
    ])

if cfg.LOSS.BCE.APPLY:
    sem_losses.append([
        "BCE Loss", 
        nn.BCELoss()
    ])

######################################## 
############### LOGGERS ################ 
######################################## 

train_logger = setup_logger("Training",   save_to, filename = "train_log.txt")
delimiter    = " | "

######################################## 
############ TRAINING LOOP ############# 
######################################## 

ITERS = len(train_dataloader)
DEVICE = cfg.DEVICE

model.train(True)
total_time = time.time()
train_logger.info("Starting training.")
for epoch in range(1, cfg.EPOCHS + 1):
    epoch_time = time.time()
    for step, (image, vf, instance_mask) in enumerate(train_dataloader, 1):
        step_time = time.time()

        image         =         image.to(DEVICE, non_blocking = True).float()
        vf            =            vf.to(DEVICE, non_blocking = True).float()
        instance_mask = instance_mask.to(DEVICE, non_blocking = True)

        semantic = torch.where(instance_mask > 0, 1.0, 0.0)

        # Reduces the number of memory operations. 
        for param in model.parameters():
            param.grad = None

        pred_semantic, pred_vf = model(image)
#        pred_semantic = torch.sigmoid(pred_semantic)

        loss = 0
        loss_values = []
        for name, loss_f in vf_losses:
            loss_value = loss_f(pred_vf, vf)
            loss_values.append(f"{name}: {loss_value.item():.4f}")
            loss += loss_value

        for name, loss_f in sem_losses:
            loss_value = loss_f(pred_semantic, semantic)
            loss_values.append(f"{name}: {loss_value.item():.4f}")
            loss += loss_value

        loss_values.append(
            f"Total Loss: {loss.item():.4f}"
        )

        loss.backward()
        optimizer.step()

        if step % cfg.LOG.EVERY == 0:
            curr_time = time.time()
            train_logger.info(delimiter.join([
                f"Epoch: [{epoch:>{len(str(cfg.EPOCHS))}}/{cfg.EPOCHS}]", 
                f"Step: [{step:>{len(str(ITERS))}}/{ITERS}]",
                f"Step Time: {curr_time - step_time:.4f}",
                f"Total Time: {curr_time - total_time:.4f}",
                *loss_values
            ]))

    # Save the model checkpoint.
    if (epoch == 1) or (epoch % cfg.SAVE_EVERY == 0):
        state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict": model.state_dict(),
            "epoch": epoch
        }

        save_checkpoint(os.path.join(save_to, f"model_{epoch}.pth"), state)

    # Make sure everything looks normal. 
    if epoch % cfg.IMAGE_EVERY == 0:
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
        ax0.imshow(pred_vf[0][0].detach().cpu())
        ax1.imshow(pred_vf[0][1].detach().cpu())
        ax2.imshow(image[0][0].detach().cpu())
        ax3.imshow(pred_semantic[0][0].detach().cpu())
        plt.savefig(os.path.join(save_to, f"image_{epoch}.png"), dpi = 400)
        plt.close()





