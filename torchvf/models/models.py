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

import torch.nn.functional as F 
import torch.nn as nn
import torch

from torchvf.models.modules import FinalBlock
from torchvf.models.encoder import Encoder
from torchvf.models.decoder import Decoder


class H2Model(nn.Module):
    def __init__(self, in_channels, out_channels = [1, 2]):
        super(H2Model, self).__init__()

        self.backbone = Encoder(in_channels)

        self.head_0  = Decoder()
        self.head_1  = Decoder()

        self.final_0 = FinalBlock(32, out_channels[0])
        self.final_1 = FinalBlock(32, out_channels[1])
        
    def forward(self, x):
        x0, x1, x2, x3 = self.backbone(x)

        semantic = self.head_0(x0, x1, x2, x3)
        semantic = self.final_0(semantic)

        vf = self.head_1(x0, x1, x2, x3)
        vf = self.final_1(vf)

        return semantic, vf

class H1Model(nn.Module):
    def __init__(self, in_channels, out_channels = [1, 2]):
        super(H1Model, self).__init__()

        self.backbone = Encoder(in_channels)
        self.head_0   = Decoder()

        self.final_0 = FinalBlock(32, out_channels[0])
        self.final_1 = FinalBlock(32, out_channels[1])
        
    def forward(self, x):
        x0, x1, x2, x3 = self.backbone(x)

        x = self.head_0(x0, x1, x2, x3)

        semantic = self.final_0(x)
        vf       = self.final_1(x)

        return semantic, vf


def get_model(model_type = "h1", in_channels = 1, out_channels = [1, 2], device = "cpu"):
    assert model_type in ["h1", "h2"], "Model type must be 'h1' or 'h2'."

    if model_type == "h1":
        model = H1Model(in_channels, out_channels)
    else:
        model = H2Model(in_channels, out_channels)

    return model.to(device)



if __name__ == "__main__":
    a = torch.rand(1, 1, 256, 256)

    h1_model = H1Model(1)
    h2_model = H2Model(1)

    h1_param_count = sum(p.numel() for p in h1_model.parameters())
    h2_param_count = sum(p.numel() for p in h2_model.parameters())

    print("==========================================")
    print(f"H1 Model Parameter Count: {h1_param_count}")
    print(f"H2 Model Parameter Count: {h2_param_count}")





