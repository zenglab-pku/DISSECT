"""
Copyright Zexian Zeng's lab, AAIS, Peking Universit. All Rights Reserved

@author: Yufeng He
"""

import csv
import gc
import json
import math
import os
import random
import time
from typing import Tuple
import numpy as np
import pandas as pd
import psutil
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from joblib import Parallel, delayed
from numba import njit
from scipy.ndimage import maximum_filter
from skimage.filters import threshold_multiotsu
from tqdm import tqdm

from .config import add_DISSECT_config
from .util.model_ema import add_model_ema_configs

def set_seed(seed):
    """
    Set the random seed for numpy, python, and torch.

    Parameters:
    seed (int): The seed to set.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_to_json(boxes, scores, meanps, areas, filename="boxes.json"):
    data = {
        "boxes": boxes.cpu().numpy().tolist(),
        "scores": scores.cpu().numpy().tolist(),
        "meanps": meanps.cpu().numpy().tolist(),
        "areas": areas.cpu().numpy().tolist(),
    }

    with open(filename, "w") as f:
        json.dump(data, f)

def read_json(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data

def generate_cell_box(cp_seg_cell):
    cell_box = {
        "cell_id": [],
        "center_x": [],
        "center_y": [],
        "height": [],
        "width": [],
    }
    for cell_idx in np.unique(cp_seg_cell):
        cell_region = np.where(cp_seg_cell == cell_idx)

        # Calculate the bounding box coordinates
        min_y, max_y = np.min(cell_region[0]), np.max(cell_region[0])
        min_x, max_x = np.min(cell_region[1]), np.max(cell_region[1])
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        height = max_y - min_y + 1
        width = max_x - min_x + 1

        # Append the values to the cell_box dictionary
        cell_box["cell_id"].append(cell_idx)
        cell_box["center_x"].append(center_x)
        cell_box["center_y"].append(center_y)
        cell_box["height"].append(height)
        cell_box["width"].append(width)
    return cell_box

class ModelGenerator:
    """
    Class to handle model building, loading, and other related tasks.
    """

    def __init__(self, config_file, weights_file, num_proposals):
        """
        Initialize the ModelHandler with a configuration file.

        Args:
        - config_file (str): Path to the configuration file.
        """
        self.cfg = self.setup(config_file, num_proposals)
        self.weights = weights_file
        self.model = self._build_model()

    def setup(self, config_file, num_proposals):
        """
        Set up the configuration for the model.

        Args:
        - config_file (str): Path to the configuration file.

        Returns:
        - cfg: Configuration object.
        """
        cfg = get_cfg()
        add_DISSECT_config(cfg)
        add_model_ema_configs(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()
        return cfg

    def _build_model(self):
        """
        Build the model based on the configuration using detectron2's build_model.

        Returns:
        - model: Built model object.
        """
        model = build_model(self.cfg)
        DetectionCheckpointer(model).load(self.weights)
        return model

class RegionExtractor:
    """
    A class to extract regions from provided images and matrices using various methods.
    """

    def __init__(
        self,
        img: np.ndarray,
        gene_df: pd.DataFrame,
        matrix: np.ndarray = None
    ):
        self.img = img
        self.matrix = matrix
        self.gene_df = gene_df

    def find_max_region(
        self, region_size: int, sort_index: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int, int]:
        """
        Find the region with the maximum total counts in the matrix.

        Parameters
        ----------
        region_size : int
            The size of the region to find.
        matrix : ndarray
            The matrix in which to find the region.
        sort_index: int
            The sort_index th maximum region to get.

        Returns
        -------
        region_matrix : ndarray
            The region matrix with the maximum total counts.
        max_region[0] : int
            The row index of the top-left corner of the region.
        max_region[1] : int
            The column index of the top-left corner of the region.
        """
        # Perform maximum filter operation
        filtered_matrix = maximum_filter(self.matrix, size=region_size)

        # Find the region with the maximum total counts
        max_total_counts = np.sort(np.unique(filtered_matrix))[-sort_index]
        max_indices = np.argwhere(filtered_matrix == max_total_counts)[0]
        max_region = tuple(max_indices)

        # Extract the region matrix with the maximum total counts
        region_matrix = self.matrix[
            max_region[0] : max_region[0] + region_size,
            max_region[1] : max_region[1] + region_size,
        ]
        img_cell = self.img[
            max_region[0] : max_region[0] + region_size,
            max_region[1] : max_region[1] + region_size,
        ]
        cp_seg_cell = self.cp_seg[
            max_region[0] : max_region[0] + region_size,
            max_region[1] : max_region[1] + region_size,
        ]
        # Filter gene_df for genes within the defined region
        region_genes = self.gene_df[
            (self.gene_df["x"] >= max_region[1])
            & (self.gene_df["x"] < max_region[1] + region_size)
            & (self.gene_df["y"] >= max_region[0])
            & (self.gene_df["y"] < max_region[0] + region_size)
        ].copy()
        region_genes.loc[:, "x"] -= max_region[1]
        region_genes.loc[:, "y"] -= max_region[0]

        return (
            img_cell,
            cp_seg_cell,
            region_matrix,
            region_genes,
            max_region[0],
            max_region[1],
        )

    def slice_region(
        self, min_x: int, min_y: int, length: int, width: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int, int]:
        """
        Slice a region from the matrix and rank its mean value.

        Parameters
        ----------
        min_x : int
            The minimum x-coordinate of the region.
        min_y : int
            The minimum y-coordinate of the region.
        max_x : int
            The maximum x-coordinate of the region.
        max_y : int
            The maximum y-coordinate of the region.
        """
        # Ensure the region is within the matrix boundaries
        min_x = max(0, min_x)
        max_x = min(self.img.shape[1] - 1, min_x + length)
        min_y = max(0, min_y)
        max_y = min(self.img.shape[0] - 1, min_y + width)

        # Slice the region from the matrix
        img_cell = self.img[min_y:max_y, min_x:max_x]

        # Filter gene_df for genes within the defined region
        region_genes = self.gene_df[
            (self.gene_df["x"] >= min_x)
            & (self.gene_df["x"] < max_x)
            & (self.gene_df["y"] >= min_y)
            & (self.gene_df["y"] < max_y)
        ].copy()

        # Adjust gene coordinates to synchronize with the sliced region
        region_genes.loc[:, "x"] -= min_x
        region_genes.loc[:, "y"] -= min_y

        return img_cell, region_genes

    def extract_region(
        self, start_x: int, start_y: int, region_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Extract a region of interest from the given images.

        Parameters:
        img (np.ndarray): The original image.
        cp_seg (np.ndarray): The cell segmentation mask.
        mtx (np.ndarray): The gene expression matrix.
        start_x (int): The x-coordinate of the top-left corner of the region.
        start_y (int): The y-coordinate of the top-left corner of the region.
        region_size (int): The size of the region.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The extracted regions from the original image, cell segmentation mask, and gene expression matrix.
        """
        end_x = start_x + region_size
        end_y = start_y + region_size

        img_cell = self.img[
            start_x : start_x + region_size, start_y : start_y + region_size
        ]
        cp_seg_cell = self.cp_seg[
            start_x : start_x + region_size, start_y : start_y + region_size
        ]
        mtx_cell = self.matrix[
            start_x : start_x + region_size, start_y : start_y + region_size
        ]

        # Filter gene_df for genes within the defined region
        region_genes = self.gene_df[
            (self.gene_df["x"] >= start_y)
            & (self.gene_df["x"] < end_y)
            & (self.gene_df["y"] >= start_x)
            & (self.gene_df["y"] < end_x)
        ].copy()

        # Adjust gene coordinates to synchronize with the extracted region
        region_genes.loc[:, "x"] -= start_y
        region_genes.loc[:, "y"] -= start_x

        return img_cell, cp_seg_cell, mtx_cell, region_genes
    
@njit(parallel=True)
def bbox_iou_np(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    if not x1y1x2y2:
        # Convert from center-width-height (xywh) to top-left-bottom-right (xyxy)
        box1 = np.array([box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, 
                         box1[0] + box1[2] / 2, box1[1] + box1[3] / 2])
        box2 = np.array([box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2, 
                         box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2]).T

    # Intersection area
    inter = np.clip(np.minimum(box1[2], box2[:, 2]) - np.maximum(box1[0], box2[:, 0]), 0, None) * \
            np.clip(np.minimum(box1[3], box2[:, 3]) - np.maximum(box1[1], box2[:, 1]), 0, None)

    # Union Area
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1] + eps
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1] + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        # Convex width and height
        cw = np.maximum(box1[2], box2[:, 2]) - np.minimum(box1[0], box2[:, 0])
        ch = np.maximum(box1[3], box2[:, 3]) - np.minimum(box1[1], box2[:, 1])

        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((box2[:, 0] + box2[:, 2] - box1[0] - box1[2]) ** 2 +
                    (box2[:, 1] + box2[:, 3] - box1[1] - box1[3]) ** 2) / 4
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / np.pi ** 2) * np.power(np.arctan(w2 / h2) - np.arctan(w1 / h1), 2)
                alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)
        else:  # GIoU
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
    else:
        return iou

def nms_numpy(boxes, scores, iou_thresh, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    sorted_indices = np.argsort(scores)[::-1]
    keep = np.zeros(len(boxes), dtype=np.int32)
    num_kept = 0

    while sorted_indices.size > 0:
        current_idx = sorted_indices[0]
        current_box = boxes[current_idx]
        keep[num_kept] = current_idx
        num_kept += 1

        if sorted_indices.size == 1:
            break

        other_boxes = boxes[sorted_indices[1:]]
        
        box1 = current_box
        box2 = other_boxes

        if not x1y1x2y2:
            # Convert from center-width-height (xywh) to top-left-bottom-right (xyxy)
            box1 = np.array([box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, 
                             box1[0] + box1[2] / 2, box1[1] + box1[3] / 2])
            box2 = np.array([box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2, 
                             box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2]).T

        # Intersection area
        inter = np.clip(np.minimum(box1[2], box2[:, 2]) - np.maximum(box1[0], box2[:, 0]), 0, None) * \
                np.clip(np.minimum(box1[3], box2[:, 3]) - np.maximum(box1[1], box2[:, 1]), 0, None)

        # Union Area
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1] + eps
        w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1] + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        ious = inter / (union + eps)

        if GIoU or DIoU or CIoU:
            # Convex width and height
            cw = np.maximum(box1[2], box2[:, 2]) - np.minimum(box1[0], box2[:, 0])
            ch = np.maximum(box1[3], box2[:, 3]) - np.minimum(box1[1], box2[:, 1])

            if CIoU or DIoU:
                c2 = cw ** 2 + ch ** 2 + eps
                rho2 = ((box2[:, 0] + box2[:, 2] - box1[0] - box1[2]) ** 2 +
                        (box2[:, 1] + box2[:, 3] - box1[1] - box1[3]) ** 2) / 4
                if DIoU:
                    ious = ious - rho2 / c2
                elif CIoU:
                    v = (4 / np.pi ** 2) * np.power(np.arctan(w2 / h2) - np.arctan(w1 / h1), 2)
                    alpha = v / ((1 + eps) - ious + v)
                    ious = ious - (rho2 / c2 + v * alpha)
            else:  # GIoU
                c_area = cw * ch + eps
                ious = ious - (c_area - union) / c_area
  
        # ious = bbox_iou_np(current_box, other_boxes, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU)
        
        # ious = bbox_iou_np(current_box, other_boxes, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU)

        # Filter out boxes with IoU greater than the threshold
        remaining_indices = np.where(ious <= iou_thresh)[0]
        if remaining_indices.size > 0:
            sorted_indices = sorted_indices[np.add(remaining_indices, 1)]
        else:
            break

    return np.array(keep[:num_kept])

class AnchorGenerator:
    def __init__(
        self,
        model=None,
        img_cell=None,
        threshold=None,
        edge_margin=3,
        repetition=2,
        batch_size=8,
        cell_area=350,
        maxarea=1600,
        
    ):
        self.model = model
        self.img_cell = img_cell
        self.threshold = threshold
        self.edge_margin = edge_margin
        self.repetition = repetition
        self.batch_size = batch_size
        self.model.to("cpu")
        self.area = cell_area
        self.maxarea = maxarea

    def NMS(self, boxes, scores, iou_thres, is_cuda=False, GIoU=False, DIoU=False, CIoU=False):
        """
        :param boxes:  (Tensor[N, 4])): are expected to be in ``(x1, y1, x2, y2)
        :param scores: (Tensor[N]): scores for each one of the boxes
        :param iou_thres: discards all overlapping boxes with IoU > iou_threshold
        :return:keep (Tensor): int64 tensor with the indices
                of the elements that have been kept
                by NMS, sorted in decreasing order of scores
        """
        if is_cuda and torch.cuda.is_available():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            boxes = boxes.to(device)
            scores = scores.to(device)
            B = torch.argsort(scores, dim=-1, descending=True)
            keep = []
            while B.numel() > 0:
                index = B[0]
                keep.append(index)
                if B.numel() == 1:
                    break
                iou = self.bbox_iou(
                    boxes[index, :], boxes[B[1:], :], GIoU=GIoU, DIoU=DIoU, CIoU=CIoU
                )
                inds = torch.nonzero(iou <= iou_thres).reshape(-1)
                B = B[inds + 1]
            return torch.tensor(keep).to('cpu')
        else:
            boxes = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
            scores = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores

            keep_indices = torch.tensor(nms_numpy(boxes, scores, iou_thres, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU))
            return keep_indices

    def local_NMS(self, boxes, scores, iou_thres, is_cuda=False, GIoU=False, DIoU=False, CIoU=False):
        """
        :param boxes:  (Tensor[N, 4])): are expected to be in ``(x1, y1, x2, y2)
        :param scores: (Tensor[N]): scores for each one of the boxes
        :param iou_thres: discards all overlapping boxes with IoU > iou_threshold
        :return:keep (Tensor): int64 tensor with the indices
                of the elements that have been kept
                by NMS, sorted in decreasing order of scores
        """
        # Filter boxes with x1 < x2 and y1 < y2
        if len(boxes) == 0:
            return torch.tensor([])
        
        if is_cuda and torch.cuda.is_available():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            boxes = boxes.to(device)
            scores = scores.to(device)
            B = torch.argsort(scores, dim=-1, descending=True)
            keep = []
            idx = 0
            while B.numel() > 0:
                index = B[0]
                keep.append(index.item())
                if B.numel() == 1:
                    break

                # if idx % 100 == 0:
                #     print(idx)
                idx += 1
                ref_x1, ref_y1, ref_x2, ref_y2 = boxes[index]
                potential_overlap = (
                    (boxes[B[1:], 0] < ref_x2) | (boxes[B[1:], 2] > ref_x1)
                ) & ((boxes[B[1:], 1] < ref_y2) | (boxes[B[1:], 3] > ref_y1))
                if potential_overlap.sum() == 0:
                    B = B[1:]
                    continue
                iou = self.bbox_iou(
                    boxes[index],
                    boxes[B[1:][potential_overlap]],
                    GIoU=GIoU,
                    DIoU=DIoU,
                    CIoU=CIoU,
                )
                inds = torch.nonzero(iou > iou_thres).reshape(-1)
                kept_indices = torch.ones_like(potential_overlap, dtype=torch.bool)
                kept_indices[torch.nonzero(potential_overlap).reshape(-1)[inds]] = False
                adjusted_inds = torch.nonzero(kept_indices).reshape(-1)
                B = B[adjusted_inds + 1]
            return torch.tensor(keep).to('cpu')
        else:
            boxes = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
            scores = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
            self.boxes = boxes
            self.scores = scores
            keep_indices = nms_numpy(boxes, scores, iou_thres, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU)
            return torch.tensor(keep_indices)

    def bbox_iou(
        self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9
    ):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.T

        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
            torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
        ).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(
                b1_x1, b2_x1
            )  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if (
                CIoU or DIoU
            ):  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw**2 + ch**2 + eps  # convex diagonal squared
                rho2 = (
                    (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                    + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
                ) / 4  # center distance squared
                if DIoU:
                    return iou - rho2 / c2  # DIoU
                elif (
                    CIoU
                ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi**2) * torch.pow(
                        torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                    )
                    with torch.no_grad():
                        alpha = v / ((1 + eps) - iou + v)
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                return iou - (c_area - union) / c_area  # GIoU
        else:
            return iou  # IoU

    def exclude_edge_boxes(
        self, boxes, scores, meanps, areas, img_shape, edge_margin=3
    ):
        """
        Exclude boxes that are at or near the edges of the image.

        Args:
        - boxes (torch.Tensor): Bounding boxes.
        - img_shape (tuple): Shape of the image as (height, width).
        - edge_margin (int): Margin value to define what constitutes 'near the edge'. Default is 10 pixels.

        Returns:
        - torch.Tensor: Filtered boxes that are not near the edges.
        """
        h, w = img_shape
        valid_indices = (
            (boxes[:, 0] > edge_margin)
            & (boxes[:, 1] > edge_margin)
            & (boxes[:, 2] < w - edge_margin)
            & (boxes[:, 3] < h - edge_margin)
        )
        return (
            boxes[valid_indices],
            scores[valid_indices],
            meanps[valid_indices],
            areas[valid_indices],
        )

    def get_boxes_and_scores(self, model, imgs_test, threshold):
        """
        Get bounding boxes and scores after processing the image with the model.

        Args:
        - model: The trained model.
        - img_test: The input image.
        - threshold: Threshold for nms.
        - repetitions: Number of times to repeat the model predictions. Default set to 3.

        Returns:
        - boxes: Bounding boxes.
        - scores: Corresponding scores.
        """
        if len(imgs_test.shape) == 2:
            img_test = [
                {
                    "image": torch.as_tensor(
                        imgs_test[np.newaxis, :, :].astype(np.float16)
                    ).to("cpu")
                }
            ]
        elif len(imgs_test.shape) == 3:
            img_test = [
                {
                    "image": torch.as_tensor(
                        imgs_test[i][np.newaxis, :, :].astype(np.float16)
                    ).to("cpu")
                }
                for i in range(len(imgs_test))
            ]
        
        self.model.eval()
        all_boxes, all_scores, all_mean = (
            [[] for _ in range(len(img_test))],
            [[] for _ in range(len(img_test))],
            [[] for _ in range(len(img_test))],
        )
        
        for _ in range(self.repetition):
            output = model(img_test)
            for batch_idx in range(len(img_test)):
                scores = output[batch_idx]["instances"].scores.cpu().numpy()
                boxes = output[batch_idx]["instances"].pred_boxes.tensor.cpu()
                extraction = [imgs_test[batch_idx][box[1]:box[3], box[0]:box[2]] for box in boxes.numpy().astype(int)]
                mean_pixel = np.array(
                    [
                        np.mean(mask) if (mask.size > 0) and (np.max(mask) != 0) else 0
                        for mask in extraction
                    ]
                )
                mean_pixel[np.isnan(mean_pixel)] = 0
                all_mean[batch_idx].append(mean_pixel)
                all_boxes[batch_idx].append(boxes)
                all_scores[batch_idx].append(scores)
        
        output_boxes, output_scores = [], []
        for batch_idx in range(len(img_test)):
            boxes = torch.tensor(
                np.concatenate(all_boxes[batch_idx], axis=0).astype(float)
            )
            scores = torch.tensor(
                np.concatenate(all_scores[batch_idx], axis=0).astype(float)
            )
            meanp = torch.tensor(
                np.concatenate(all_mean[batch_idx], axis=0).astype(float)
            )
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            # Stage 1 filtering
            if len(np.unique(meanp.numpy())) >= 10:
                try:
                    num_classes = 4
                    mean_threshold = threshold_multiotsu(meanp.numpy(), classes=num_classes)[0] 
                except ValueError:
                    try:
                        num_classes = 3
                        mean_threshold = threshold_multiotsu(meanp.numpy(), classes=num_classes)[0]
                    except ValueError:
                         mean_threshold = 30
            else:
                mean_threshold = 30
            mean_threshold = mean_threshold if mean_threshold >= 25 else 25
            valid_indices = (area >= self.area) & (meanp > mean_threshold) & (area < self.maxarea) & ~(torch.isnan(boxes).any(dim=1))

            scores = scores[valid_indices]
            boxes = boxes[valid_indices]

            if len(boxes) < 1:
                output_boxes.append(None)
                output_scores.append(None)
            elif len(boxes) >= 1:
                keep = self.NMS(boxes, scores, threshold, GIoU=True)
                output_boxes.append(boxes[keep])
                output_scores.append(scores[keep])
        return output_boxes, output_scores

    def process_patch(self, output, args):
        """
        Helper function to process a single patch of the image.
        """
        global MODEL
        imgs_test = []
        self.imgs_test = imgs_test
        for cell in args:
            x_1, x_2, y_1, y_2 = cell[:4]
            self.imgs_test.append(self.img_cell[x_1:x_2, y_1:y_2])
        MODEL = self.model
        args = np.array(
            [
                args[i]
                for i in range(len(self.imgs_test))
                if np.mean(self.imgs_test[i]) >= 10
            ]
        )
        self.imgs_test = np.array([i for i in self.imgs_test if np.mean(i) >= 10])
        if len(self.imgs_test) >= 1:
            boxes, scores = self.get_boxes_and_scores(
                MODEL, self.imgs_test, self.threshold
            )
            output_boxes, output_scores = [], []
            for idx, cell in enumerate(args):
                if boxes[idx] is not None:
                    idx_box = boxes[idx].clone()
                    x_1, y_1 = cell[0], cell[2]
                    # Adjust the box coordinates to the original image
                    idx_box[:, [0, 2]] += y_1
                    idx_box[:, [1, 3]] += x_1
                    output_boxes.append(idx_box)
                    output_scores.append(scores[idx].clone())
            process = psutil.Process()
            memory_usage = process.memory_info().rss / (1024**3)
            # print(f"Memory usage: {memory_usage:.2f} GB")

            if os.path.exists(os.path.join(output, "raw_boxes.csv")):
                with open(
                    os.path.join(output, "raw_boxes.csv"), "a", newline=""
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    for boxes, scores in zip(output_boxes, output_scores):
                        if boxes is not None:
                            for box, score in zip(boxes.numpy(), scores.numpy()):
                                if box is not None:
                                    writer.writerow(list(box) + [score])
            else:
                with open(
                    os.path.join(output, "raw_boxes.csv"), "w", newline=""
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    for boxes, scores in zip(output_boxes, output_scores):
                        if boxes is not None:
                            for box, score in zip(boxes.numpy(), scores.numpy()):
                                if box is not None:
                                    writer.writerow(list(box) + [score])

    def safe_concatenate(self, list_input, dim=0):
        if len([i for i in list_input if i is not None]) >= 1:
            return torch.cat(list_input, dim=dim)
        else:
            return None

    def process(self, stride, ratio_stride, n_jobs, output):
        """
        Extract bounding boxes from a large image by processing it in smaller patches.
        """
        time_start = time.time()
        edge = ratio_stride * stride
        height, width = self.img_cell.shape[:2]

        if (height >= 512) and (width >= 512):
            x_ref = np.array([min(i, height) for i in range(0, height + stride, stride)])
            y_ref = np.array([min(i, width) for i in range(0, width + stride, stride)])
            tasks = [
                (x_1, x_2, y_1, y_2)
                for x_1, x_2 in zip(x_ref[:-(ratio_stride)], x_ref[ratio_stride:])
                for y_1, y_2 in zip(y_ref[:-(ratio_stride)], y_ref[ratio_stride:])
            ]
        else:
            tasks = [
                (0, width, 0, height)
            ]
        
        if len(tasks) > self.batch_size:
            batched_tasks = [
                tasks[i : i + self.batch_size]
                for i in range(0, len(tasks), self.batch_size)
            ]
        else:
            batched_tasks = [tasks]
        
        
        if not os.path.exists(output):
            os.mkdir(output)
            
        if os.path.exists(os.path.join(output, "raw_boxes.csv")):
            os.remove(os.path.join(output, "raw_boxes.csv"))
        
        time_end = time.time()
        print("Begin extracting boxes after {} seconds.".format(time_end - time_start))

        time_start = time.time()
        _ = Parallel(n_jobs=n_jobs)(
            delayed(self.process_patch)(output, args)
            for args in tqdm(
                batched_tasks, total=len(batched_tasks), desc="Processing patches"
            )
        )
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024**3)
        print(f"Finished box detection using memory {memory_usage:.2f} GB")

        del batched_tasks
        gc.collect()

        results = pd.read_csv(
            os.path.join(output, "raw_boxes.csv"),
            names=np.array(["x1", "y1", "x2", "y2", "scores"]),
        )
        boxes = torch.tensor(np.array(results.iloc[:, :4]))
        scores = torch.tensor(np.array(results.iloc[:, 4]))

        time_end = time.time()
        print("Finish extracting boxes after {} seconds.".format(time_end - time_start))
        
        time_start = time.time()
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024**3)
        print(f"Going to NMS using memory {memory_usage:.2f} GB")

        print("Begin nms for all boxes!")
        keep = self.local_NMS(boxes, scores, self.threshold, GIoU=True)
        boxes = boxes[keep]
        scores = scores[keep]

        time_end = time.time()
        print(
            "Finish nms for all boxes after {} seconds.".format(time_end - time_start)
        )
        print("{} boxes are detected!".format(boxes.shape[0]))
        return boxes, scores

