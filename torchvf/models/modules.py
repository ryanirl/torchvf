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

import torch.nn as nn
import torch


def ConvBlock(in_channels, out_channels, kernel_size = 3, stride = 1, padding = "same"):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )

def MiddleBlock(in_channels, out_channels):
    return nn.Sequential(
        ConvBlock(in_channels, out_channels),
        nn.Dropout(p = 0.2),
        ConvBlock(out_channels, out_channels)
    )

def FinalBlock(in_channels, out_channels):
    return nn.Sequential(
        ConvBlock(in_channels, in_channels,  kernel_size = 1),
        ConvBlock(in_channels, in_channels,  kernel_size = 1),
        nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = "same")
    )





