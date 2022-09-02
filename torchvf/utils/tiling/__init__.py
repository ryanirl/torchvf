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

from torchvf.utils.tiling.dynamic_overlap import DynamicOverlapTiler
from torchvf.utils.tiling.padding import PaddingTiler
from torchvf.utils.tiling.reshape import ResizeTiler

tilers = {
    "dynamic_overlap": DynamicOverlapTiler,
    "padding": PaddingTiler,
    "resize": ResizeTiler
}

def get_tiler(tiler_type):
    assert tiler_type in tilers, "Tiler type must be one of \
            'dynamic_overlap', 'padding', or 'resize'."

    return tilers[tiler_type]





