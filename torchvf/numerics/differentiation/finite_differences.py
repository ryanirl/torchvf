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

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvf.numerics.differentiation.kernels import *


def gaussian_smoothed_finite_differences(x, kernel_size, device = "cpu"):
    gaussian_x, gaussian_y = finite_gaussian_kernel(kernel_size)

    gaussian = torch.stack([gaussian_x, gaussian_y], dim = 0).to(device)

    out = F.conv2d(x, gaussian[:, None], padding = kernel_size // 2)

    return out


def finite_differences(x, kernel_size, device = "cpu"):
    finite_x, finite_y = finite_diff_kernel(kernel_size)

    finite = torch.stack([finite_x, finite_y], dim = 0).to(device)

    out = F.conv2d(x, finite[:, None], padding = kernel_size // 2)

    return out





