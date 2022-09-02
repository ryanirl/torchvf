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

from torchvf.transforms.transforms import (
    Compose, 
    RandomCrop, 
    ToFloat, 
    VerticalFlip, 
    HorizontalFlip
)

train_transform = Compose([
    RandomCrop(256, 256),
    VerticalFlip(p = 0.25),
    HorizontalFlip(p = 0.25),
    ToFloat(255)
])

eval_transform = Compose([
    ToFloat(255)
])

transforms = {
    "train": train_transform,
    "eval": eval_transform,
    None: None
}





