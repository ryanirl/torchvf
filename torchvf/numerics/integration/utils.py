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


def init_values_mesh(H, W, device = "cpu"):
    y, x = torch.meshgrid(
        torch.arange(0, H, device = device), 
        torch.arange(0, W, device = device),
        indexing = "ij"
    )

    init_values = torch.stack([x, y], dim = 0)

    return init_values


def init_values_mesh_batched(B, H, W, device = "cpu"):
    init_values = init_values_mesh(H, W, device)

    init_values = init_values.repeat(B, 1, 1, 1).float()

    return init_values


def init_values_semantic(semantic, device):
    B, _, H, W = semantic.shape

    init_values = init_values_mesh(H, W, device)

    init_values = init_values[:, semantic[0][0]].float()

    return init_values





