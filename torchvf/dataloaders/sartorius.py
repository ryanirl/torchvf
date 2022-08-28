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

##############################################################
# The Sartorius dataset from the Sartorius Cell Instance 
# Segmentation Kaggle competition. More information can
# be found here: 
# - https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/overview
# 
##############################################################

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import time
import glob
import sys
import cv2
import os


def rle_decode(rle_list, shape = (520, 704), dtype = np.uint32):
    """
    Given a list of RLE encoded masks, decode them and
    return the numpy mask.

    Args:
        rle_list (list): List of RLE encoded masks for
        a whole image.

        shape (tuple): Tuple shape of the final image.
        
    Returns:
        np.ndarray 

    """
    mask = np.zeros((shape[0] * shape[1], 1))

    for idx, rle in enumerate(rle_list):
        rle    = rle.split()
        np_rle = np.array(rle, dtype = np.uint64)

        first_indices = np_rle[0 : : 2] - 1 
        lengths       = np_rle[1 : : 2]
        last_indices  = first_indices + lengths 

        for i in range(len(first_indices)):
            mask[first_indices[i] : last_indices[i]] = 1 + idx

    return mask.reshape(shape).astype(dtype)


class Sartorius(Dataset):
    """
    Dataloader for the Sartorius dataset. The link to download this
    dataset can be found on the Kaggle homepage here: 
        - https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data

    Args:
        data_dir (str): The location of the Sartorius dataset relative 
        to your current directory.

        cell_type (str): Options are:
            - "astro"
            - "cort"
            - "shsy5y"

        transforms: Albumentation transforms. 

    Returns:
        __getitem__: Return the Image and Mask respectively of index i. 

    """
    def __init__(self, data_dir, cell_type = None, transforms = None):
        self.transforms = transforms 
        self.train_csv = pd.read_csv(os.path.join(data_dir, "train.csv"))

        if cell_type: self.train_csv = self.train_csv[self.train_csv["cell_type"] == cell_type]

        self.train_ids   = sorted(self.train_csv["id"].unique().tolist())
        self.train_cache = [os.path.join(data_dir, "train/", f"{ID}.png") for ID in self.train_ids]

    def __getitem__(self, idx):
        input_dir = self.train_cache[idx]
        curr_id   = self.train_ids[idx]

        df_id = self.train_csv[self.train_csv["id"] == curr_id]

        cell_type   = df_id["cell_type"].tolist()[0]
        annotations = df_id["annotation"].tolist()

        mask  = rle_decode(annotations, (520, 704)).astype(np.int64)
        image = cv2.imread(input_dir, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if self.transforms is not None:
            transformed = self.transforms(image = image, mask = mask)

            image = transformed["image"]
            mask  = transformed["mask"]

        return image[None], mask[None]

    def __len__(self):
        return len(self.train_cache)



if __name__ == "__main__":
    data_dir = "../../../../kaggle/sartorius_cell_instance_segmentation_2021/data"

    shsy5y_dataset = Sartorius(data_dir, cell_type = "shsy5y")
    astro_dataset  = Sartorius(data_dir, cell_type = "astro")
    cort_dataset   = Sartorius(data_dir, cell_type = "cort")

    print("# SHSY5Y:", len(shsy5y_dataset))
    print("# ASTRO:", len(astro_dataset))
    print("# CORT:", len(cort_dataset))





