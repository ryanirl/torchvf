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

#############################################################
# This metric is from: 
#   - https://www.kaggle.com/theoviel/competition-metric-map-iou
# 
#############################################################

import numpy as np 


def compute_iou(y_pred, y_true):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        y_true (np.ndarray): Actual mask.
        y_pred (np.ndarray): Predicted mask.

    Returns:
        np.ndarray: IoU matrix, of size true_objects x pred_objects.

    """
    pred_objects = len(np.unique(y_pred))
    true_objects = len(np.unique(y_true))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        y_pred.flatten(), y_true.flatten(), 
        bins = (pred_objects, true_objects)

    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.histogram(y_true, bins = true_objects)[0]

    area_pred = np.expand_dims(area_pred, -1)
    area_true = np.expand_dims(area_true, 0)

    # Compute union
    union = area_pred + area_true - intersection

    # exclude background
    intersection = intersection[1:, 1:] 
    union        = union[1:, 1:]

    union[union == 0] = 1e-9

    iou = intersection / union
    
    return iou     


def precision_at(threshold, iou):
    matches = iou > threshold

    true_positives  = np.sum(matches, axis = 1) >= 1  # Correct
    false_positives = np.sum(matches, axis = 1) == 0  # Extra 
    false_negatives = np.sum(matches, axis = 0) == 0  # Missed

    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives)
    )

    return tp, fp, fn


def map_iou(y_preds, y_trues, logger = None):
    """
    Computes the metric for the competition.

    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        y_preds (list of masks): Predictions.
        y_trues (list of masks): Ground truths.
        logger (Logger, optional): Logger for eval.

    Returns:
        float: mAP IoU.

    """
    ious = [compute_iou(y_pred, y_true) for y_pred, y_true in zip(y_preds, y_trues)]
    
    if logger:
        logger.info("Thresh & TP & FP & FN & Prec.")

    prec = []
    
    for t in [0.50, 0.75, 0.90]:
#    for t in np.arange(0.50, 1.00, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

        if logger:
            logger.info("{:1.2f} & {} & {} & {} & {:1.3f}".format(t, tps, fps, fns, p))

    if logger:
        logger.info("AP & & & & {:1.3f}".format(np.mean(prec)))

    return prec





