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


def nearest_interpolation(vf, points):
    """ 
    Computes nearest-neighbor interpolation on the vector field 
    `vf` given `points`. Assumes `points` index `vf`, and that 
    `vf` is defined on a regular rectangular grid of equidistant 
    points.

    Args:
        vf (torch.float32): Vector field of shape (D, H, W). In 
            my case D is almost always 2. Must be of type float.
        points (torch.float32): The points of dimension D to be 
            interpolated. Of shape (D, N). Must be of type float.

    Returns:
        torch.float32: The interpolated values of shape (D, N).

    """
    _, H, W = vf.shape

    clipped = torch.empty_like(points)
    clipped[0] = torch.clip(points[0], 0, W - 1)
    clipped[1] = torch.clip(points[1], 0, H - 1)

    x, y = torch.round(clipped).long()

    return vf[:, y, x]


def bilinear_interpolation(vf, points):
    """ 
    Computes 2D bilinear interpolation on the vector field `vf` 
    given `points`. Assumes `points` index `vf`, and that `vf` 
    is defined on a regular rectangular grid of equidistant 
    points.

    Args:
        vf (torch.float32): Vector field of shape (2, H, W). Must
            be of type float. 
        points (torch.float32): The points of dimension 2 to be 
            interpolated. Of shape (2, N). Must be of type float.

    Returns:
        torch.float32: The interpolated values of shape (2, N).

    """
    _, H, W = vf.shape

    clipped = torch.empty_like(points)
    clipped[0] = torch.clip(points[0], 0, W - 1)
    clipped[1] = torch.clip(points[1], 0, H - 1)

    x0, y0 = torch.floor(clipped).long()
    x1, y1 = torch.ceil(clipped).long()
    x,  y  = clipped

    delta_x = x - x0
    delta_y = y - y0

    lerp_x0 = torch.lerp(
        vf[:, y0, x0],
        vf[:, y0, x1],
        delta_x 
    )

    lerp_x1 = torch.lerp(
        vf[:, y1, x0],
        vf[:, y1, x1],
        delta_x 
    )

    lerp_y = torch.lerp(
        lerp_x0,
        lerp_x1,
        delta_y 
    )

    return lerp_y


def bilinear_interpolation_batched(vf, points):
    """ 
    Computes a batched 2D bilinear interpolation on the vector 
    field `vf` given `points`. Because it's batched, all points
    on `vf` need to be considered and therefore the `points` shape 
    will be (B, 2, H, W). Assumes `points` index `vf`, and that 
    `vf` is defined on a regular rectangular grid of equidistant 
    points.

    Args:
        vf (torch.float32): Vector field of shape (B, 2, H, W). Must
            be of type float. 
        points (torch.float32): The points of dimension 2 to be 
            interpolated. Of shape (B, 2, H, W). Must be of type float.

    Returns:
        torch.float32: The interpolated values of shape (B, 2, H, W).

    """
    B, _, H, W = vf.shape

    points = points.permute(1, 0, 2, 3).float()

    clipped = torch.empty_like(points)
    clipped[0] = torch.clip(points[0], 0, W - 1)
    clipped[1] = torch.clip(points[1], 0, H - 1)

    x0, y0 = torch.floor(clipped.float()).long()
    x1, y1 = torch.ceil(clipped.float()).long()
    x,  y  = clipped

    delta_x = (x - x0).unsqueeze(1)
    delta_y = (y - y0).unsqueeze(1)

    # OPTION 1: Faster though I'm still testing to see if gradients are the same.
    # https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
    x0 = x0.contiguous()
    x1 = x1.contiguous()
    y0 = y0.contiguous()
    y1 = y1.contiguous()
    vf = vf.contiguous()

    x0 = x0.view(B, -1)
    x1 = x1.view(B, -1)
    y0 = y0.view(B, -1)
    y1 = y1.view(B, -1)
    vf = vf.view(B, 2, -1)

    one_0 = y0 * H + x0 
    two_0 = y0 * H + x1 
    one_1 = y1 * H + x0 
    two_1 = y1 * H + x1 

    one_0 = one_0.unsqueeze(1).expand(B, 2, -1)
    two_0 = two_0.unsqueeze(1).expand(B, 2, -1)
    one_1 = one_1.unsqueeze(1).expand(B, 2, -1)
    two_1 = two_1.unsqueeze(1).expand(B, 2, -1)

    one_0 = vf.gather(-1, one_0).view(B, 2, H, W)
    two_0 = vf.gather(-1, two_0).view(B, 2, H, W)
    one_1 = vf.gather(-1, one_1).view(B, 2, H, W)
    two_1 = vf.gather(-1, two_1).view(B, 2, H, W)

#    # OPTION 2: 2x slower though more safe atm.
#    one_0 = torch.zeros((B, _, H, W), device = "cuda:0")
#    two_0 = torch.zeros((B, _, H, W), device = "cuda:0")
#    one_1 = torch.zeros((B, _, H, W), device = "cuda:0")
#    two_1 = torch.zeros((B, _, H, W), device = "cuda:0")
#    for i in range(B):
#        x0_b, x1_b = x0[i], x1[i]
#        y0_b, y1_b = y0[i], y1[i]
#
#        one_0[i] = vf[i, :, y0_b, x0_b]
#        two_0[i] = vf[i, :, y0_b, x1_b]
#        one_1[i] = vf[i, :, y1_b, x0_b]
#        two_1[i] = vf[i, :, y1_b, x1_b]

    lerp_x0 = torch.lerp(
        one_0,
        two_0,
        delta_x
    )

    lerp_x1 = torch.lerp(
        one_1,
        two_1,
        delta_x,
    )

    lerp_y = torch.lerp(
        lerp_x0,
        lerp_x1,
        delta_y
    )

    return lerp_y





