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

import numpy as np
import numpy.random as random


def v_flip(image):
    return np.ascontiguousarray(image[:, ::-1, ...])


def h_flip(image):
    return np.ascontiguousarray(image[:, :, ::-1, ...])


def rot90(image, factor):
    return np.ascontiguousarray(np.rot90(image, factor, axes = [1, 2]))


def random_crop_coords(height, width, crop_height, crop_width):
    h_start = random.random()
    w_start = random.random()

    if (height < crop_height) or (width < crop_width):
        raise ValueError(
            f"Crop size ({crop_height}, {crop_width}) larger than the "
            f"image size ({height}, {width})."
        )

    y1 = int((height - crop_height + 1) * h_start)
    y2 = y1 + crop_height

    x1 = int((width - crop_width + 1) * w_start)
    x2 = x1 + crop_width

    return y1, y2, x1, x2





