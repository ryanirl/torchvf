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

from torchvf.models.modules import MiddleBlock


class ResidualBlock(nn.Module):
    """ Residual encoder block. """
    def __init__(self, in_channels, feature_maps, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = (2, 2), stride = None)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, feature_maps, kernel_size = (1, 1), stride = stride, bias = False),
            nn.BatchNorm2d(feature_maps)
        )

        self.conv1 = nn.Conv2d(in_channels, feature_maps,  kernel_size = (3, 3), stride = stride, padding = 1, bias = False)
        self.bn1   = nn.BatchNorm2d(feature_maps)

        self.conv2 = nn.Conv2d(feature_maps, feature_maps, kernel_size = (3, 3), stride = 1,      padding = 1, bias = False)
        self.bn2   = nn.BatchNorm2d(feature_maps)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        skip_connection = self.relu(x)

        x = self.maxpool(skip_connection)

        return x, skip_connection


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.Encoder_0 = ResidualBlock(in_channels = in_channels, feature_maps = 32)
        self.Encoder_1 = ResidualBlock(in_channels = 32,          feature_maps = 64)
        self.Encoder_2 = ResidualBlock(in_channels = 64,          feature_maps = 128)

        self.Middle = MiddleBlock(in_channels = 128, out_channels = 256)

    def forward(self, x):
        x, x0 = self.Encoder_0(x)
        x, x1 = self.Encoder_1(x)
        x, x2 = self.Encoder_2(x)

        x3 = self.Middle(x)

        return [x0, x1, x2, x3]





