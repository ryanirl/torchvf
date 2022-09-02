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

from torchvf.numerics.interpolation.functional import (
    bilinear_interpolation_batched,
    bilinear_interpolation,
    nearest_interpolation
)


def _vf_bilinear_batched(vector_field):
    def _vf(p):
        out = bilinear_interpolation_batched(vector_field, p)
        return out
    return _vf


def _vf_bilinear(vector_field):
    def _vf(p):
        out = bilinear_interpolation(vector_field[0], p)
        return out
    return _vf


def _vf_nearest(vector_field):
    def _vf(p):
        out = nearest_interpolation(vector_field[0], p)
        return out
    return _vf


vf_interpolators = {
    "bilinear_batched": _vf_bilinear_batched,
    "bilinear": _vf_bilinear,
    "nearest": _vf_nearest
}


def interp_vf(vector_field, mode = "bilinear"):
    return vf_interpolators[mode](vector_field)





