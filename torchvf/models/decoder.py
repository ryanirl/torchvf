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

from torchvf.models.modules import ConvBlock


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.up = nn.Upsample(scale_factor = 2)

        self.conv_block_0 = ConvBlock(in_channels + out_channels, out_channels)
        self.conv_block_1 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)

        x = torch.cat((x, skip_connection), 1)

        x = self.conv_block_0(x)
        x = self.conv_block_1(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder_0 = DecoderBlock(256, 128) 
        self.decoder_1 = DecoderBlock(128, 64)
        self.decoder_2 = DecoderBlock(64, 32)

    def forward(self, x0, x1, x2, x3):
        x = self.decoder_0(x3, x2)
        x = self.decoder_1(x,  x1)
        x = self.decoder_2(x,  x0)

        return x 





