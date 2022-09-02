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

from torchvf.losses.dice import DiceLoss


class TverskyLoss(DiceLoss):
    """
    This code is reworked from this GitHub repo:
        - https://github.com/qubvel/segmentation_models.pytorch
        - https://github.com/BloodAxe/pytorch-toolbelt

    Tversky loss for semantic segmentation. Notice this class inherits
    `DiceLoss` and adds a weight to the value of each TP and FP given by
    constants alpha and beta. With alpha == beta == 0.5, this loss becomes
    equal to the Dice loss. `y_pred` and `y_true` must be torch tensors of
    shape (B, C, H, W).

    """
    def __init__(self, from_logits = True, log_loss = False, smooth = 0.0, eps = 1e-7, 
                 alpha = 0.5, beta = 0.5, gamma = 1.0):
        """
        Args:
            from_logits (bool): If True, assumes y_pred are raw logits.
            log_loss (bool): If True, the loss is computed as `-log(dice_coeff)`,
                otherwise `1 - dice_coeff`.

            smooth (float): Smoothness constant for dice coefficient.
            eps (float): For numerical stability to avoid zero division error.
            alpha (float): Weight constant that penalize model for FPs.
            beta (float): Weight constant that penalize model for FNs.
            gamma (float): Constant that squares the error function. Defaults to `1.0`.

        """
        super(TverskyLoss, self).__init__(from_logits, log_loss, smooth, eps)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reduction(self, loss):
        return loss.mean() ** self.gamma

    def _compute_score(self, y_pred, y_true, smooth = 0.0, eps = 1e-7, dims = ()):
        assert y_pred.size() == y_true.size()

        intersection = torch.sum(y_pred * y_true, dim = dims)  
        fp = torch.sum(y_pred * (1.0 - y_true), dim = dims)
        fn = torch.sum((1 - y_pred) * y_true, dim = dims)

        tversky_score = (intersection + smooth) / (intersection + self.alpha * fp + self.beta * fn + smooth).clamp_min(eps)

        return tversky_score





