import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from time import perf_counter

from src.components.losses.loss import AggregationLoss, DiscriminationLoss, TextDiceLoss, KernelDiceLoss


# def forward(self, pred_similarities, regions_mask, kernels_mask)


if __name__ == "__main__":
    criteria = AggregationLoss(sigma_agg=0.5)
    cri_2 = DiscriminationLoss(sigma_dis=3)
    dice_1 = TextDiceLoss(epsilon=1e-6)
    dice_2 = KernelDiceLoss(epsilon=1e-6)
    
    BS = 16
    
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
    
    pred_similarities = torch.randn((BS,4,6,8))
    time = perf_counter()
    loss = criteria(pred_similarities, masks_region, masks)
    loss_2 = cri_2(pred_similarities, masks)
    loss_3 = dice_1(masks_region, masks)
    loss_4 = dice_2(masks_region, masks)
    end = perf_counter()
    print("time:", end - time)
    
