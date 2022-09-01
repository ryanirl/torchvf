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

## Reproducability

This installation guide is for people who want to reproduce the results in my
[article](https://github.com/ryanirl/torchvf/blob/main/article/first_draft.pdf)
on vector field based methods using TorchVF as-is. 

I have a PyPI library for TorchVF, though for now just clone the repository as 
the API is a work in progress. 

### Installing the Weights

Weights include H1 and H2 for the bacterial phase contrast, bacterial
fluorescence, and worm subsets of the BPCIS dataset. And can be found
[here](https://drive.google.com/drive/folders/14fvNNZkr4ewuy0-Q2mwjCX-fbMVS7X90?usp=sharing)
(157.5 MB zipped | 185.5 MB unzipped). 

Once you download the weights:
 - Unzip the file.
 - Replace the `torchvf/weights` file with the `torchvf_weights` file. 
 - Rename `torchvf/torchvf_weights` to `torchvf/weights`.

### Installing the BPCIS Dataset

First, download the BPCIS dataset [here](http://www.cellpose.org/dataset_omnipose).
Then setup the file system this way:

```bash
├── torchvf/
├── data/
│   └── bpcis/
│       ├── bact_fluor_train/
│       ├── bact_fluor_test/
│       ├── bact_phase_train/
│       ├── bact_phase_test/
│       ├── worm_train/
│       └── worm_test/
├── weights/
└── ***
```

If you have cloned the library, downloaded the weights, and downloaded the
BPCIS dataset you *should* be able to do `python3 eval.py --config_dir
../weights/bact_fluor/h1/eval_config.py`. This will run evaluation on the
bacterial fluorescence subset using the downloaded weights. 

## Usage

Work in progress.

## Citation

Work in progress.


## License

Distributed under the Apache-2.0 license. See `LICENSE` for more information.






