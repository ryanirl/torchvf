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

from torchvf.numerics import interp_vf, ivp_solver


class IVPLoss(nn.Module):
    """
    The initial value problem loss function as described in the "On Vector
    Fields for Instance Segmentation" article. This loss function aims to
    minimize the distance between trajectories of points under integration
    through the ground truth and predicted vector field. 

    Known issues:
        - Under interpolation we clip values that go beyond our image domain,
        therefore backpropogated gradients at the image borders might not be as
        expected. This isn't necessarily a problem with this loss function
        iteself as it is with the interpolation.

    """
    def __init__(self, dx = 0.5, n_steps = 8, solver = "euler", device = "cpu"):
        """
        Args:
            dx (float): Numeric integration step size.
            n_steps (int): Number of numeric integration steps.
            solver (str): Numeric integration solver. One of:
                - "euler"
                - "midpoint"
                - "runge_kutta"

        """
        super(IVPLoss, self).__init__()

        self.dx = dx
        self.n_steps = n_steps
        self.solver = solver
        self.device = device

    def _compute_init_values(self, B, H, W):
        y, x = torch.meshgrid(
            torch.arange(0, H, device = self.device), 
            torch.arange(0, W, device = self.device),
            indexing = "ij"
        )

        init_values = torch.stack([x, y], dim = 0)
        init_values = init_values.repeat(B, 1, 1, 1)

        return init_values

    def _compute_batched_trajectories(self, vf):
        B, C, H, W = vf.shape

        vf = interp_vf(vf, mode = "bilinear_batched")

        init_values = self._compute_init_values(B, H, W)

        trajectories = ivp_solver(
            vf, 
            init_values, 
            dx = self.dx, 
            n_steps = self.n_steps, 
            solver = self.solver
        )
        
        return trajectories

    def forward(self, vf_pred, vf_true):
        """
        Args:
            vf_pred (torch.Tensor): Of shape (B, 2, H, W).
            vf_true (torch.Tensor): Of shape (B, 2, H, W).

        Returns:
            torch.Tensor: The loss.

        """
        true_trajectories = self._compute_batched_trajectories(vf_true)
        pred_trajectories = self._compute_batched_trajectories(vf_pred)

        loss_batch = F.mse_loss(true_trajectories, pred_trajectories)

        return loss_batch





