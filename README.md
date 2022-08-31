# TorchVF

WORK IN PROGRESS.

TorchVF is a unifying Python library for using vector fields for lightweight 
proposal-free instance segmentation. The TorchVF library provides generalizable
functions to automate ground truth vector field computation, interpolation of
discrete vector fields, numeric integration solvers, clustering functions, and
various other utilities. 

This repository also provides all configs, code, and tools necessary to
reproduce the results in my
[article](https://github.com/ryanirl/torchvf/blob/main/article/first_draft.pdf)
on vector field based methods.

## Quick Start

For anyone interested in learning about vector field based methods, see my
[article](https://github.com/ryanirl/torchvf/blob/main/article/first_draft.pdf).
TorchVF can be used to compute the instance segmentation given the semantic
segmentation and vector field via the following code: 

```Python
# Consider we have a vector field `vf` and semantic segmentation `semantic`, 
# we can derive the instance segmentation via the following code: 

from torchvf.numerics import *
from torchvf.utils import *

# Step 1: Convert our discretely sampled vector field into continuous vector
# field through bilinear interpolation. 
vf = interp_vf(vf, mode = "bilinear")

# Step 2. Convert our semantic segmentation `semantic` into a set of
# initial-values to be integrated through our vector field `vf`.
init_values = init_values_semantic(semantic, device = "cuda:0")

# Step 3. Integrate our initial-values `init_values` through our vector field
# `vf` for 25 steps with a step size of 0.1 using Euler's method for numeric 
# integration. 
solutions = ivp_solver(
    vf, 
    init_values, 
    dx = 0.1,
    n_steps = 25,
    solver = "euler"
)[-1] # Get the final solution. 

# Clustering can only be done on the CPU. 
solutions = solutions.cpu()
semantic = semantic.cpu()

# Step 4. Cluster the integrated semantic points `solutions` to obtain the
# instance segmentation. 
instance_segmentation = cluster(
    solutions, 
    semantic[0], 
    eps = 2.25,
    min_samples = 15,
    snap_noise = False
)

```

## Installation

Work in progress.

## Usage

Work in progress.

## Citation

Work in progress.

TODO:
 - Scripts to reproduce all experiments.
 - Add dependencies and PyPi.
 - Usage.
 - Contributions (Not accepting pulls ATM, but feel free to post issues or send me emails.)
 - Add image/diagram for vector field based methods.


## License

Distributed under the MIT License. See `LICENSE` for more information.






