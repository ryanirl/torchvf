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

from math import floor, ceil

from torchvf.utils.utils import batch_of_n


class TilerBase(nn.Module):
    def __init__(self, model, tile_size = (256, 256), overlap = 128, batch_size = 2, device = "cpu"):
        super(TilerBase, self).__init__()

        self.model = model
        self.TH, self.TW = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.device = device

        self.stride_H = self.TH - self.overlap
        self.stride_W = self.TW - self.overlap

    def tile_image(self, image):
        B, C, H, W = image.shape

        assert B == 1, f"Batch size of 1 required. Found batch size {B}."

        overflow     = self._tiling_overflow(image.shape)
        padded_image = self.tile_transform(image, overflow)

        B, C, NH, NW = padded_image.shape

        # Make sure everything works as should. There should be 0 overflow with new
        # reshape size. 
        H_check, W_check = self._tiling_overflow((B, C, NH, NW))
        assert not H_check, "Something went wrong with tiling."
        assert not W_check, "Something went wrong with tiling."

        # Now find the indexes, should be shape: (tiles, 4)
        # 1. Find the number of tiles in X and Y direction.
        H_tiles = int(1 + (NH - self.TH) / (self.TH - self.overlap))
        W_tiles = int(1 + (NW - self.TW) / (self.TW - self.overlap))

        # 2. Find the actual indides of shape: (tiles, 4) 
        # This is done from left to right, top to bottom.
        tiles = []
        for H_n in range(H_tiles):
            for W_n in range(W_tiles):
                left  = int(H_n * (self.TH - self.overlap))
                right = int(W_n * (self.TW - self.overlap))

                # Going to be in form (y_0, x_0, y_1, x_1)
                tiles.append([left, right, left + self.TH, right + self.TW])

        # 3. Create the tiled images from the tiles computed
        # in step 2.
        tiled_images = []
        for y_0, x_0, y_1, x_1 in tiles:
            tiled_images.append(
                padded_image[0][:, y_0 : y_1, x_0 : x_1]
            )
        
        tiled_images = torch.stack(tiled_images)

        return tiled_images, tiles, padded_image.shape

    def merge_tiles(self, tiled_images, tiles, new_shape, original_shape):
        _, _, H, W   = original_shape
        _, C, TH, TW = tiled_images.shape
        _, _, NH, NW = new_shape

        # Don't want to include borders when stitching back 
        # together as they often include artifacts. 
        extra = int(self.overlap // 4)

        average = torch.zeros((1, C, NH, NW), device = self.device)
        merged_pred = torch.zeros((1, C, NH, NW), device = self.device)
        for i, (y_0, x_0, y_1, x_1) in enumerate(tiles):
            tiled_image = tiled_images[i]

            ty_0, tx_0, ty_1, tx_1 = 0, 0, self.TH, self.TW

            if y_0:
                y_0 = y_0 + extra
                ty_0 = extra 

            if x_0: 
                x_0 = x_0 + extra
                tx_0 = extra 

            if y_1 != NH: 
                y_1 = y_1 - extra
                ty_1 = self.TH - extra 

            if x_1 != NW: 
                x_1 = x_1 - extra
                tx_1 = self.TW - extra 

            average[:, :, y_0 : y_1, x_0 : x_1] += 1
            merged_pred[:, :, y_0 : y_1, x_0 : x_1] += tiled_image[:, ty_0 : ty_1, tx_0 : tx_1]

        merged_pred = merged_pred / average

        # For padding, this will reverse the padding for final predictions.
        # For reshape, this will reverse the reshape for final predictions. 
        pred = self.merge_transform(merged_pred, new_shape, original_shape)

        return pred

    def _tiling_overflow(self, image_size):
        B, C, H, W = image_size

        H_overflow = self.TH + self.stride_H * ceil(max((H - self.TH), 0) / self.stride_H) - H
        W_overflow = self.TW + self.stride_W * ceil(max((W - self.TW), 0) / self.stride_W) - W

        return H_overflow, W_overflow

    def forward(self, image):
        tiled_images, tiles, new_shape = self.tile_image(image)
        batches = batch_of_n(tiled_images, self.batch_size)

        # Predict VF and semantic on each tiling batch.
        vf_batched = []
        semantic_batched = []
        for batch in batches:
            semantic, vf = self.model(batch)

            semantic_batched.append(semantic)
            vf_batched.append(vf)

        vf_batched       = torch.vstack(vf_batched)
        semantic_batched = torch.vstack(semantic_batched)

        # Stitch the predicted tiles together.
        vf       = self.merge_tiles(vf_batched,       tiles, new_shape, image.shape)
        semantic = self.merge_tiles(semantic_batched, tiles, new_shape, image.shape)

        return semantic, vf





