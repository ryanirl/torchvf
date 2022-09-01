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

import torchvision
import torch

import numpy as np
import yaml
import math
import glob
import os


def mask_bounding_box(mask):
    """
    Returns bounding box given a binary mask. Bounding box given in the form:

        [x_0, y_0, x_1, y_1]

    Where (x_0, y_0) is the top left pixel of the box and (x_1, y_1) is the
    bottom right pixel of the box. To index a numpy array do:

        np.array[y_0 : y_1, x_0 : x_1]

    Args:
        mask (np.ndarray): Binary Mask.

    Returns:
        list: Bounding box of binary mask.

    """
    non_zero_idx = np.where(mask == 1)

    x = non_zero_idx[1]
    y = non_zero_idx[0]

    bounding_box = [
        np.min(x),
        np.min(y),
        np.max(x) + 1,
        np.max(y) + 1
    ]

    return bounding_box


def reshape_min(x, min_value = 256, interp = None):
    _, C, H, W = x.shape
    
    # Only rescale if needed
    if (H >= min_value) and (W >= min_value):
        return x
    
    if H < W: 
        ratio = min_value / H
        new_shape = (min_value, math.ceil(W * ratio))
    else: 
        ratio = min_value / W
        new_shape = (math.ceil(H * ratio), min_value)
        
    resized_image = torchvision.transforms.Resize(new_shape)(x)
    
    return resized_image


def batch_of_n(arr, n):
    """ Given an array, split into batches of size n. """
    for i in range(0, len(arr), n): 
        yield arr[i : i + n]


def next_model(weight_dir):
    files = sorted(os.listdir(weight_dir))

    subdirs = []
    for file in files:
        if (os.path.isdir(os.path.join(weight_dir, file)) and file.startswith("model_")):
            subdirs.append(file)
    
    if not subdirs:
        model_dir = os.path.join(weight_dir, "model_0000/")
        os.makedirs(model_dir)
        return model_dir

    model_n    = int(subdirs[-1].split("_")[-1]) + 1
    next_model = f"model_{model_n:04}/"
    model_dir  = os.path.join(weight_dir, next_model) 
    os.makedirs(model_dir)

    return model_dir


def save_model(filename, model):
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        print(f"Save model dir does not exist, making dir '{dir_name}'")
        os.makedirs(dir_name)

    print(f"Saving model to: {filename}")
    torch.save(model.state_dict(), filename)


def load_model(filename, model):
    assert os.path.exists(filename), "Model dir does not exist!"

    print(f"Loading model from: {filename}")
    model.load_state_dict(torch.load(filename))

    return model


def save_checkpoint(filename, checkpoint):
    """
    state = {
        "optimizer_state_dict" : optimizer.state_dict(),
        "model_state_dict" : model.state_dict(),
        "epoch" : epoch
    }

    """
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        print(f"Save checkpoint dir does not exist, making dir '{dir_name}'")
        os.makedirs(dir_name)

    print(f"Saving checkpoint to: {filename}")
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer = None):
    """
    state = {
        "optimizer_state_dict" : optimizer.state_dict(),
        "model_state_dict" : model.state_dict(),
        "epoch" : epoch
    }

    """
    assert os.path.exists(filename), "Checkpoint dir does not exist!"

    print(f"Loading checkpoint from: {filename}")
    checkpoint = torch.load(filename, map_location = torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]

    return model, optimizer, epoch





