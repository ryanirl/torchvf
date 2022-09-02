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

from torchvf.utils.tiling.base import TilerBase


class PaddingTiler(TilerBase):
    def tile_transform(self, image, overflow):
        H_overflow, W_overflow = overflow

        H_pad, W_pad     = int(H_overflow // 2), int(W_overflow // 2)
        H_extra, W_extra = int(H_overflow  % 2), int(W_overflow  % 2)

        padding = (W_pad, W_pad + W_extra, H_pad, H_pad + H_extra)

#        padded_image = F.pad(image, padding, mode = "constant", value = 0)
        try:
            padded_image = F.pad(image, padding, mode = "reflect")
        except:
            padded_image = F.pad(image, padding, mode = "replicate")


        return padded_image 

    def merge_transform(self, merged_pred, new_shape, original_shape):
        _, _,  H,  W = original_shape
        _, _, NH, NW = new_shape

        H_total_pad, W_total_pad = NH - H, NW - W
        H_pad, W_pad             = H_total_pad // 2, W_total_pad // 2
        H_pad_extra, W_pad_extra = H_total_pad % 2, W_total_pad % 2

        H_last = NH - (H_pad + H_pad_extra)
        W_last = NW - (W_pad + W_pad_extra)
        
        pred = merged_pred[:, :, H_pad : H_last, W_pad : W_last]

        return pred





