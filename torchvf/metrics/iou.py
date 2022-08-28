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


def iou(output, target):
    """ IoU or Jaccard score. """
    B, C, H, W = target.shape

    output = output.view(B, C, -1)
    target = target.view(B, C, -1)

    tp = torch.sum(output * target, dim = 2)
    fp = torch.sum(output, dim = 2) - tp
    fn = torch.sum(target, dim = 2) - tp

    # IoU Score
    score = tp / (tp + fp + fn)

    return score 





