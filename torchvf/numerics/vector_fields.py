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

import numpy as np

import edt

from torchvf.numerics.differentiation import *
from torchvf.utils import *


def compute_sdf(semantic_mask):
    """
    Given a semantic mask of shape (H, W) this function will compute the signed
    distance of each piel to the border of the mask. Used for the computation
    of the vector fields. 

    Args:
        semantic_mask (np.ndarray): The boolean semantic mask.

    Returns:
        np.ndarray: The SDF of the semantic_mask. Of shape: (H, W).

    """
    sdf = edt.sdf(
        semantic_mask, 
        order = 'C',
        parallel = -1 
    )

    return sdf


def compute_vector_field(labels, kernel_size, alpha = 10, device = "cpu"):
    """
    Computes the vector field as proposed in the "On Vector Fields for Instance
    Segmentation" article. This is done by computing the signed distance field
    (SDF) for each mask in `labels`, then computing the gradient via a large
    gaussian-smoothed finite differences kernel.

    Unfortunately there isn't a PyTorch implementation of the SDF (that I know
    of) so we must convert tensors to NumPy arrays first, then run the SDF
    computation on the CPU.

    Args:
        labels (torch.Tensor): The instance segmentation labels in the form of
            0 (background), 1 (first instance mask), 2 (second instance mask), ...,
            n (n'th instance mask). Shape must be (1, H, W).

        kernel_size (int): The size of the finite difference gaussian kernel
            used to compute the numeric gradient of the SDF. 

        alpha (int): The truncation value used on the SDF. 

    Returns:
        torch.Tensor: The vector field of shape (2, H, W).

    """
    _, H, W = labels.shape

    unique = torch.unique(labels)[1:]

    np_labels = labels[0].detach().cpu().numpy().astype(np.uint32)

    pad = kernel_size // 2

    vector_field = torch.zeros((2, H, W), device = device)
    for i in unique:
        curr = np_labels == int(i)

        x_0, y_0, x_1, y_1 = mask_bounding_box(curr)

        trimmed_curr = curr[y_0 : y_1 + 1, x_0 : x_1 + 1]

        padded_curr = np.pad(trimmed_curr, ((pad, pad), (pad, pad)))

        sdf = compute_sdf(padded_curr)
        sdf[sdf > alpha] = alpha

        # Use this as an example of why smoothing is important. 
#        out = np.stack(np.gradient(sdf), axis = 0)[None]

        # Compute the gradient. 
        out = gaussian_smoothed_finite_differences(
            torch.Tensor(sdf[None][None]).to(device), 
            kernel_size = kernel_size, 
            device = device
        )[0]

        vector_field[:, curr] = out[:, padded_curr]

    return vector_field.to(device)





