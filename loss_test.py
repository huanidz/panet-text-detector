import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from time import perf_counter

from src.components.losses.loss import AggregationLoss, DiscriminationLoss, TextDiceLoss, KernelDiceLoss, PANLoss


# def forward(self, pred_similarities, regions_mask, kernels_mask)


if __name__ == "__main__":
     criteria = PANLoss()

     BS = 2

     masks = torch.Tensor([[[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                         [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
                              
                              [[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                         [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]]]]).int()

     masks = torch.cat(BS // 2 * [masks], dim=0)

     masks_region = torch.Tensor([[[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                         [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]],
                                   
                                   [[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                         [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]).int()
    
     masks_region = torch.cat(BS // 2 * [masks_region], dim=0)
     print(f"==>> masks_region.shape: {masks_region.shape}")

     pred_regions = torch.randn_like(masks_region.float())
     
     pred_similarities = torch.randn((BS,4,6,8))
     time = perf_counter()

     loss = criteria(pred_regions, masks_region, pred_regions, masks_region, pred_similarities) # pred_regions, regions_gt, pred_kernels, kernels_gt, pred_similarities
     end = perf_counter()
     print("time:", end - time)

