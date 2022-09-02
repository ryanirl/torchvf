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

from math import ceil


class DynamicOverlapTiler(TilerBase):
    def tile_transform(self, image, overflow):
        H_overflow, W_overflow = overflow

        H_pad, W_pad     = int(H_overflow // 2), int(W_overflow // 2)
        H_extra, W_extra = int(H_overflow  % 2), int(W_overflow  % 2)

        padding = (W_pad, W_pad + W_extra, H_pad, H_pad + H_extra)
        padded_image = F.pad(image, padding, mode = "constant", value = 0)

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

    def tile_image(self, image):
        B, C, H, W = image.shape

        assert B == 1, f"Batch size of 1 required. Found batch size {B}."

        if H < self.TH:
            image = self.tile_transform(image, (self.TH - H, 0))

        if W < self.TW:
            image = self.tile_transform(image, (0, self.TW - W))

        B, C, H, W = image.shape

        # Now find the indexes, should be shape: (tiles, 4)
        # 1. Find the number of tiles in X and Y direction.
        H_tiles = int(1 + (H - self.TH) / (self.TH - self.overlap)) + 1
        W_tiles = int(1 + (W - self.TW) / (self.TW - self.overlap)) + 1 

        # 2. Find the actual indides of shape: (tiles, 4) 
        # This is done from left to right, top to bottom.
        tiles = []
        for H_n in range(H_tiles):
            for W_n in range(W_tiles):
                left  = int(H_n * (self.TH - self.overlap))
                right = int(W_n * (self.TW - self.overlap))

                if left + self.TH > H: 
#                    print(left)
                    left = H - self.TH 
#                    print(left)

                if right + self.TW > W:
                    right = W - self.TW 

                # Going to be in form (y_0, x_0, y_1, x_1)
                tiles.append([left, right, left + self.TH, right + self.TW])

        # 3. Create the tiled images from the tiles computed
        # in step 2.
        tiled_images = []
        for y_0, x_0, y_1, x_1 in tiles:
            tiled_images.append(
                image[0][:, y_0 : y_1, x_0 : x_1]
            )
        
        tiled_images = torch.stack(tiled_images)

        return tiled_images, tiles, image.shape





