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

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

#import hdbscan


def cluster(points, semantic, eps = 2.2, min_samples = 15, snap_noise = True):
    """
    Will cluster the semantic points given by the affinity derived 
    offset coordinate system using DBSCAN.

    Args:
        points (torch.Tensor): Must be of shape (D, N)
        eps: ...
        min_samples: ...

    Returns:
        torch.Tensor: ...

    """
    points = points.T

    if points.shape[0] == 0:
        return semantic

#    clusterer = hdbscan.HDBSCAN(min_cluster_size = 15)
#    clusters = torch.Tensor(clusterer.fit_predict(points))
    clustering = DBSCAN(eps = eps, min_samples = min_samples, n_jobs = -1).fit(points)
    clusters = torch.Tensor(clustering.labels_)

    if snap_noise:
        outliers_idx = clusters == -1
        if torch.any(outliers_idx):
            non_outliers_idx = ~outliers_idx

            # If there are outliers but no non-outliers, then 
            # nothing was clustered and return semantic. 
            if not torch.any(non_outliers_idx):
                return semantic

            outliers     = points[outliers_idx]
            non_outliers = points[non_outliers_idx]

            NN = NearestNeighbors(metric = "euclidean")#, radius = eps)
            NN.fit(non_outliers)

            _, nn_idx = NN.kneighbors(outliers)
            nn_idx = torch.Tensor(nn_idx).long()

            nearest_n = clusters[non_outliers_idx][nn_idx]
            values = torch.mode(nearest_n, dim = 1)[0]

            # Give the outliers the instance value of their nearest neighbor.
            clusters[outliers_idx] = values

    # It sets first cluster to 0, be careful as this would
    # blend it into the background.
    instance_segmentation = semantic.clone().float()
    instance_segmentation[semantic] = clusters + 1

    return instance_segmentation





