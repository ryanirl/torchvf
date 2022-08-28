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


class DiceLoss(nn.Module):
    """
    Binary dice loss for semantic segmentation. This code is reworked from
    these GitHub repos:
        - https://github.com/qubvel/segmentation_models.pytorch
        - https://github.com/BloodAxe/pytorch-toolbelt

    """
    def __init__(self, from_logits = True, log_loss = False, smooth = 0.0, eps = 1e-7):
        """
        Args:
            log_loss (bool): If True, the loss is computed as `-log(dice_coeff)`,
                otherwise `1 - dice_coeff`.

            from_logits (bool): If True, assumes y_pred are raw logits.
            smooth (float): Smoothness constant for dice coefficient.
            eps (float): For numerical stability to avoid zero division error.

        """
        super(DiceLoss, self).__init__()

        self.from_logits = from_logits
        self.log_loss = log_loss
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Of shape (B, C, H, W).
            y_true (torch.Tensor): Of shape (B, C, H, W).

        Returns:
            torch.Tensor: The loss.

        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = F.logsigmoid(y_pred).exp()

        bs   = y_true.size(0)
        dims = (0, 2)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        scores = self._compute_score(
            y_pred, 
            y_true.type_as(y_pred), 
            smooth = self.smooth, 
            eps = self.eps, 
            dims = dims
        )

        if self.log_loss: loss = -torch.log(scores.clamp_min(self.eps))
        else:             loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss = loss * mask.to(loss.dtype)

        return self._reduction(loss)

    def _reduction(self, loss):
        return loss.mean()

    def _compute_score(self, y_pred, y_true, smooth = 0.0, eps = 1e-7, dims = ()):
        assert y_pred.size() == y_true.size()

        intersection = torch.sum(y_pred * y_true, dim = dims)
        cardinality = torch.sum(y_pred + y_true, dim = dims)

        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

        return dice_score





