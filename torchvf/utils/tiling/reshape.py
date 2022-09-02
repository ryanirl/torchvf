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

import torchvision

from torchvf.utils.tiling.base import TilerBase


class ResizeTiler(TilerBase):
    def tile_transform(self, image, overflow):
        B, C, H, W = image.shape
        H_overflow, W_overflow = overflow

        H_resize = H + H_overflow
        W_resize = W + W_overflow

        resized_image = torchvision.transforms.Resize((H_resize, W_resize))(image)

        return resized_image

    def merge_transform(self, merged_pred, new_shape, original_shape):
        _, _,  H,  W = original_shape

        pred = torchvision.transforms.Resize((H, W))(merged_pred)

        return pred





