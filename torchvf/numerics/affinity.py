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

import torch.nn.functional as F
import torch.nn as nn 
import torch


def affinity_vf(instance_mask, kernel_size, device = "cpu"):
    """
    Computes the ground truth affinity vector field given an instance
    mask.

    Kernel sizes from 11 - 19 seem to work well on the BPCIS dataset. 

    Args:
        instance_mask (torch.Tensor): Of shape (B, C, H, W).

    Returns:
        (torch.Tensor): The vector field that is to be predicted by your 
            model. Will be of shape (B, 2, H, W)

    """
    B, _, H, W = instance_mask.shape

    semantic = instance_mask.clone() > 0
    sem_idx  = semantic.repeat(1, 2, 1, 1)

    target_kernel = apply_ones_kernel(instance_mask, kernel_size, gradient = False)
    target_kernel = target_kernel.reshape(B, kernel_size, kernel_size, H, W)

    vector_field = affinity_to_vf(target_kernel, device = device)
    vector_field[~sem_idx] = 0.0

    return vector_field


def apply_ones_kernel(x, kernel_size, gradient = False):
    B, C, H, W = x.shape
    mid = kernel_size // 2
    
    out = neighbor_view(x, kernel_size, padding_int = -1)

    if not gradient:
        out = out.reshape(B, kernel_size, kernel_size, H, W)

        instance_values = out[:, mid, mid].reshape(B, 1, 1, H, W)

        out = torch.where(out != instance_values, 0, 1)

        out = out.reshape(B, kernel_size * kernel_size, H, W)

    return out


def neighbor_view(x, n, padding_int = -1):
    """
    Uses PyTorch's im2col to get a n^2 convolution style neighbor 
    view of a 2D array. 

    Args:
        x (torch.Tensor): 4D tensor input of shape (B, C, H, W). 
        n (int): The number of n^2 neighbors. Must be an odd 
            number.

    Returns:
        torch.Tensor: The n^2-neighbor view of the image of
            shape (1, n, n, H, W).

    """
    x_dtype = x.dtype
    mid     = n // 2

    x = F.pad(x, (mid, mid, mid, mid), "constant", padding_int)

    out = F.unfold(x.float(), kernel_size = (n, n)).to(x_dtype)

    return out


def affinity_to_vf(target, device = "cpu"):
    """
    Given an affinity prediction of shape (B, K_H, K_W, H, W). This 
    will return the vector field.

    Args:
        target (torch.Tensor): This is the affinity neighbor view of
            your ground truth instance segmentation. This tensor must
            be of shape (B, K_H, K_W, H, W).

    Returns:
        vf (torch.Tensor): Of shape (2, H, W).
    
    """
    B, K_H, K_W, H, W = target.shape
    
    x_target = target.sum(axis = 1).reshape(B, K_W, H, W).permute(0, 2, 3, 1) / K_H
    y_target = target.sum(axis = 2).reshape(B, K_H, H, W).permute(0, 2, 3, 1) / K_H

    vec_product_l =  torch.ones(K_H // 2)
    vec_product_r = -torch.ones(K_H // 2)
    vec_product = torch.cat([vec_product_l, torch.Tensor([0]), vec_product_r], axis = 0)
    vec_product = vec_product.reshape(K_H, 1).to(device)

    vf_x = -torch.matmul(x_target.float(), vec_product.float()).permute(0, 3, 1, 2) / 2 
    vf_y = -torch.matmul(y_target.float(), vec_product.float()).permute(0, 3, 1, 2) / 2

    vf = torch.cat([vf_x, vf_y], axis = 1)

    return vf.float()





