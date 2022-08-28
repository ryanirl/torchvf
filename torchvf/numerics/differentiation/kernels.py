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


def mean_kernel(kernel_size):
    assert kernel_size % 2, "`kernel_size` must be divisible by 2."

    ones_kernel = torch.ones((1, kernel_size, kernel_size))
    mean_kernel = ones_kernel / ones_kernel.numel()

    return mean_kernel  


def gaussian_kernel(kernel_size, sigma = 0.5):
    assert kernel_size % 2, "`kernel_size` must be divisible by 2."

    t    = torch.linspace(-1, 1, kernel_size)
    x, y = torch.meshgrid(t, t, indexing = "ij")
    dst  = torch.sqrt(x * x + y * y)

    gaussian_kernel = torch.exp(-(dst ** 2 / (2.0 * sigma ** 2)))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel


def finite_diff_kernel(kernel_size):
    assert kernel_size % 2, "`kernel_size` must be divisible by 2."

    a = torch.ones((kernel_size, kernel_size // 2))
    b = torch.zeros((kernel_size, 1))
    c = -a.clone()

    finite_x = torch.cat([c, b, a], axis = 1)
    finite_y = finite_x.T

    return finite_x, finite_y


def finite_gaussian_kernel(kernel_size, sigma = 0.5):
    assert kernel_size % 2, "`kernel_size` must be divisible by 2."

    finite_x, finite_y = finite_diff_kernel(kernel_size)
    g_kernel = gaussian_kernel(kernel_size, sigma)

    finite_x = finite_x * g_kernel
    finite_y = finite_y * g_kernel

    return finite_x, finite_y



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.set_cmap("crest")

#    print(mean_kernel(3))
#    print(gaussian_kernel(7))
#    print(finite_diff_kernel(5))

    x, y = finite_gaussian_kernel(7)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(x)
    ax1.imshow(y)
    ax2.imshow(torch.abs(x) + torch.abs(y))

    plt.show()





