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

import numpy as np
import argparse
import shutil
import time
import os

from dataloaders import *
from transforms import *
from numerics import *
from metrics import *
from models import *
from losses import *
from utils import *

# I am actively looking into a better way to do configs.
from ml_collections.config_flags.config_flags import _ConfigFileParser

parser = argparse.ArgumentParser()
parser.add_argument("--config_dir", default = "./configs/eval/bpcis_bact_fluor.py")
args = parser.parse_args()

FileParser = _ConfigFileParser(name = "eval")
cfg = FileParser.parse(args.config_dir)

#shutil.copyfile(
#    cfg.CONFIG_DIR, 
#    os.path.join(os.path.dirname(args.config_dir), "eval_config_modified.py")
#)

######################################## 
############# DATA LOADER ############## 
######################################## 

test_dataset = BPCIS(
    cfg.DATA.DIR, 
    split = cfg.DATA.SPLIT,
    vf = cfg.DATA.VF, 
    transforms = transforms[cfg.DATA.TRANSFORMS], 
    remove = cfg.DATA.REMOVE, 
    copy = cfg.DATA.COPY
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size = cfg.BATCH_SIZE,
    pin_memory = True,
    drop_last = True,
    shuffle = False,
    num_workers = 1
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

model, _, _ = load_checkpoint(cfg.MODEL_DIR, model) 

######################################## 
################ TILER ################# 
######################################## 

Tiler = get_tiler(cfg.TILE.TYPE)

model = Tiler(
    model,
    tile_size = cfg.TILE.SIZE,
    overlap = cfg.TILE.OVERLAP,
    batch_size = cfg.TILE.BATCH_SIZE,
    device = cfg.DEVICE
)

######################################## 
################ LOGGER ################ 
######################################## 

eval_logger = setup_logger("Eval", cfg.SAVE_DIR, filename = "eval_log.txt")
delimiter   = " | "

######################################## 
############## EVAL LOOP ############### 
######################################## 

model.train(False)
with torch.inference_mode():

    f1s = []
    ious = []
    FPSs = []
    pred_counts = []
    true_counts = []
    pred_instances = []
    true_instances = []

    ITERS = len(test_dataloader)

    start_time = time.time()
    for step, (image, target) in enumerate(test_dataloader, 1):
        image  =  image.to(cfg.DEVICE, non_blocking = True).float()
        target = target.to(cfg.DEVICE, non_blocking = True).float()
        true_semantic = target > 0 

        step_time = time.time()

        pred_semantic, pred_vf = model(image)
        pred_semantic = torch.sigmoid(pred_semantic) > 0.5

        semantic_iou = iou(pred_semantic, true_semantic)
        semantic_f1  = f1(pred_semantic,  true_semantic)

        # 1. Define continuous vf through bilinear interpolation. 
        vf = interp_vf(pred_vf, mode = cfg.IVP.INTERP)

        # 2. Get the initial-values to integrate through `f`. 
        init_values = init_values_semantic(pred_semantic, device = cfg.DEVICE)

        # 3. Perform integration for `n_steps` at step size `dx` with solver
        #    `solver`.
        solutions = ivp_solver(
            vf, 
            init_values, 
            dx = cfg.IVP.DX, 
            n_steps = cfg.IVP.STEPS, 
            solver = cfg.IVP.SOLVER
        )[-1] # Get the final solution. 

        # Clustering can only be done on the CPU. 
        solutions = solutions.cpu()
        pred_semantic = pred_semantic.cpu()

        # 4. Cluster the integrated semantic points to obtain the instance
        #    segmentation. 
        instance_segmentation = cluster(
            solutions, 
            pred_semantic[0], 
            eps = cfg.CLUSTERING.EPS, 
            min_samples = cfg.CLUSTERING.MIN_SAMPLES,
            snap_noise = cfg.CLUSTERING.SNAP_NOISE
        )

        solution_time = time.time() - step_time
        FPS = 1 / solution_time

        true_count = len(torch.unique(target)) - 1
        pred_count = len(torch.unique(instance_segmentation)) - 1

        target = target.cpu()
        image = image.cpu()

        pred_instance = instance_segmentation[0].numpy().astype(np.uint16)
        true_instance = target[0][0].numpy().astype(np.uint16)

        map_iou_score = map_iou(
            [pred_instance],
            [true_instance]
        )

        f1s.append(semantic_f1)
        ious.append(semantic_iou)
        FPSs.append(FPS)
        pred_counts.append(pred_count)
        true_counts.append(true_count)
        pred_instances.append(pred_instance)
        true_instances.append(true_instance)

        eval_logger.info(delimiter.join([
            f"Step: [{step:>{len(str(ITERS))}}/{ITERS}]",
            f"Sol Time: {solution_time:.4f}",
            f"FPS: {FPS:.4f}",
            f"IoU: {semantic_iou.item():.4f}",
            f"F1: {semantic_f1.item():.4f}",
            f"IoU @ 0.50: {map_iou_score[0]:.4f}",
            f"IoU @ 0.75: {map_iou_score[1]:.4f}",
            f"IoU @ 0.90: {map_iou_score[2]:.4f}",
            f"mAP IoU: {np.mean(map_iou_score):.4f}",
            f"Pred Count: {pred_count:.4f}",
            f"True Count: {true_count:.4f}",
        ]))


avg_f1 = torch.mean(torch.Tensor(f1s))
avg_iou = torch.mean(torch.Tensor(ious))
avg_fps = torch.mean(torch.Tensor(FPSs))

pred_counts = torch.Tensor(pred_counts)
true_counts = torch.Tensor(true_counts)
count_mae = torch.mean(torch.abs(pred_counts - true_counts))

eval_logger.info("Starting final results.")
eval_logger.info(delimiter.join([
    f"Total Time: {time.time() - start_time:.4f}",
    f"Average FPS: {avg_fps.item():.4f}",
    f"Average IoU: {avg_iou.item():.4f}",
    f"Average F1: {avg_f1.item():.4f}",
    f"Cell Counting Average MAE: {count_mae.item():.4f}",
]))

map_iou_score = map_iou(
    pred_instances,
    true_instances,
    eval_logger
)





