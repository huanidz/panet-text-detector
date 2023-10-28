import torch
import torch.nn as nn
from scipy import ndimage
import itertools
from time import perf_counter
import numpy as np
from tqdm import tqdm
"""
Paper: In Pixel Aggregation, we borrow the idea of clustering
to reconstruct the complete text instances from the kernels.
Let us consider the text instances as clusters. The kernels
of text instances are cluster centers. The text pixels are the
samples to be clustered. Naturally, to aggregate the text
pixels to the corresponding kernels, the distance between
the text pixel and kernel of the same text instance should be
small

---------------------------------------------------

1. Text Instances: Text instances refer to separate text regions or objects present in an image. Each text instance can be seen as a cluster, representing a group of pixels that together form a complete text object.

2. Kernels: Kernels are representative points or centers that represent the text instances. Each kernel is associated with one particular text instance, and it serves as a reference point for aggregating or combining the pixels belonging to that text instance.

3. Clustering: The author suggests using clustering techniques to assign each text pixel to its corresponding kernel or text instance. Clustering is a process of grouping similar data points together. In this case, the text pixels serve as the samples to be clustered, and the cluster centers are the kernels representing the text instances.

4. Distance: To aggregate the text pixels to their respective kernels or text instances, the distance between a text pixel and the kernel of the same text instance should be small. This means that text pixels that are closer to the kernel (in terms of distance) are more likely to be part of the same text instance. By assigning each pixel to the nearest kernel, the complete text instances can be reconstructed or aggregated.
"""

class AggregationLoss(nn.Module):
    def __init__(self, sigma_agg:float = 0.5):
        super().__init__()
        self.sigma_agg = sigma_agg
        
    def forward(self, pred_similarities, regions_mask, kernels_mask, text_mask_ndi_labels, kernel_mask_ndi_labels):
                
        # pred_similarities: (B, S_C, SIZE, SIZE)
        # regions_mask: (B, 1, SIZE, SIZE)
        # kernels_mask: (B, 1, SIZE, SIZE)
        
        batch_size = kernels_mask.shape[0]
        
        # kernel_labels, _ = ndimage.label(kernels_mask.cpu().numpy())
        # region_labels, _ = ndimage.label(regions_mask.cpu().numpy())
        
        kernel_labels = kernel_mask_ndi_labels
        region_labels = text_mask_ndi_labels
        
        # kernel_labels = torch.cat([torch.from_numpy(ndimage.label(kernels_mask[batch])[0]).unsqueeze(dim=0) for batch in range(kernels_mask.shape[0])], dim=0)
        
        # kernel_labels = torch.from_numpy(kernel_labels)
        # region_labels = torch.from_numpy(region_labels)
        
        kernels_mask_cardinality = torch.zeros_like(kernels_mask) # |Ki|        
        regions_mask_cardinality = torch.zeros_like(regions_mask) # |Ti|
        Gk_kernel_similarities = torch.zeros_like(pred_similarities)
        C = pred_similarities.shape[1] # Num channels of similarity vector
        
        
        for batch in range(batch_size):
            num_kernel = kernel_labels[batch].max().int()
            for i in range(1, num_kernel + 1):
                where_ones = (kernel_labels[batch] == i).squeeze(axis=0)
                kernels_mask_cardinality[batch] = kernels_mask_cardinality[batch].masked_fill(where_ones, kernels_mask[batch].masked_select(where_ones).sum())                
                for j in range(C):
                    Gk_kernel_similarities[batch][j][where_ones] = pred_similarities[batch][j][where_ones].sum()
        
        for batch in range(batch_size):
            num_regions = region_labels[batch].max().int()
            for i in range(1, num_regions + 1):
                where_ones = (region_labels[batch] == i).squeeze(axis=0)
                regions_mask_cardinality[batch] = regions_mask_cardinality[batch].masked_fill(where_ones, kernels_mask[batch].masked_select(where_ones).sum())                
        
        Gk_kernel_similarities /= (kernels_mask_cardinality + 1) # Gk / |K| (plus one for handling 0 division)
        
        Fp_similarities = pred_similarities * regions_mask
        
        norm = torch.linalg.norm(Fp_similarities - Gk_kernel_similarities, dim=1) # Compute norm for each pixel.
        
        norm = norm - self.sigma_agg
                
        D_p_K = torch.where(norm > 0.0, norm, torch.full_like(norm, 0.0))
        D_p_K = torch.log(D_p_K**2 + 1) / (regions_mask_cardinality.squeeze(dim=1) + 1) # (plus one for handling 0 division)
        L_agg = (D_p_K / num_kernel).sum(dim=(1,2)).sum()
        return L_agg
 
        
class DiscriminationLoss(nn.Module):
    """
    DiscriminationLoss - L_dis: discrimination loss between text regions
    
    Note: if there's only one text region --> L_dis = 0
    """
    def __init__(self, sigma_dis:float = 3):
        super().__init__()
        self.sigma_dis = sigma_dis
        
    def forward(self, pred_similarities, kernel_mask_ndi_labels):
        # The number of discrimination happen is: N(N-1)/2 where N = number of kernel
        
        # kernels_mask_np = kernels_mask.cpu().numpy()
        
        # kernel_labels = torch.cat([torch.from_numpy(ndimage.label(kernels_mask_np[batch])[0]).unsqueeze(dim=0) for batch in range(kernels_mask.shape[0])], dim=0)
        
        kernel_labels = kernel_mask_ndi_labels
        
        batch_size = pred_similarities.shape[0]
        
        # scale = torch.Tensor([kernel_labels[batch].max() for batch in range(kernel_labels.shape[0])]).cuda()
        # scale = torch.where(scale > 1, 1 / (scale * (scale - 1)), scale)        
        
        C = pred_similarities.shape[1] # Num channels of similarity vector
        
        array_of_Gk_kernel_similarities = []
        for batch in range(batch_size):
            num_kernel = kernel_labels[batch].max().int()
            if num_kernel == 1:
                continue
            elif num_kernel < 1:
                # raise ValueError("Number of kernels in an image should not be 0.!")
                continue
            array_of_Gk_kernel_similarities_for_current_batch = []
            for i in range(1, num_kernel + 1): # Looping through each kernel
                where_ones = (kernel_labels[batch] == i).squeeze(axis=0)
                Gk_kernel_similarities = torch.zeros_like(pred_similarities)
                for j in range(C):
                    Gk_kernel_similarities[batch][j][where_ones] = pred_similarities[batch][j][where_ones].sum()    
                array_of_Gk_kernel_similarities_for_current_batch.append(Gk_kernel_similarities)
            array_of_Gk_kernel_similarities.append(array_of_Gk_kernel_similarities_for_current_batch)
        L_dis = 0.0
        
        for batch_kernel_sim in array_of_Gk_kernel_similarities:
            for pair in (itertools.combinations(batch_kernel_sim, 2)):
                norm = self.sigma_dis - torch.linalg.norm(pair[0] - pair[1], dim=1)
                D_ki_kj = torch.where(norm > 0.0, norm, torch.full_like(norm, 0.0))
                D_ki_kj = torch.log(D_ki_kj**2 + 1)
                L_dis += D_ki_kj
        
        # L_dis *= scale.view(-1, 1, 1)
        L_dis = L_dis.sum(dim=(1,2)).sum(dim=0)
        return L_dis
    
        
    
class TextDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(TextDiceLoss, self).__init__()
        self.eps = epsilon
        
    def forward(self, pred_regions, regions_gt, ohem_masks):
        pred_regions = torch.sigmoid(pred_regions)
        
        ohem_masks = ohem_masks.reshape([ohem_masks.shape[0], -1])
        pred_regions = pred_regions.reshape([pred_regions.shape[0], -1]) * ohem_masks
        regions_gt = regions_gt.reshape([regions_gt.shape[0], -1]) * ohem_masks
        
        intersection = torch.sum(pred_regions * regions_gt)
        
        dice = (2.0 * intersection + self.eps)/(pred_regions.sum() + regions_gt.sum() + self.eps)  
        
        loss = 1 - dice
        
        return loss
    
class KernelDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(KernelDiceLoss, self).__init__()
        self.eps = epsilon
        
    def forward(self, pred_kernels, kernels_gt):
    
        pred_kernels = torch.sigmoid(pred_kernels)
        pred_kernels = pred_kernels.reshape([pred_kernels.shape[0], -1])
        kernels_gt = kernels_gt.reshape([kernels_gt.shape[0], -1])
        
        intersection = torch.sum(pred_kernels * kernels_gt)
        dice = (2.0 * intersection + self.eps)/(pred_kernels.sum() + kernels_gt.sum() + self.eps)  
        
        loss = 1 - dice
        
        return loss
    
class PANLoss(nn.Module):
    def __init__(self, ohem_ratio:int = 3, alpha:float = 0.5, beta:float = 0.25):
        super(PANLoss, self).__init__()
        self.ohem_ratio = ohem_ratio
        
        self.loss_regions = TextDiceLoss()
        self.loss_kernel = KernelDiceLoss()
        self.loss_aggregation = AggregationLoss(sigma_agg=0.5)
        self.loss_discrimination = DiscriminationLoss(sigma_dis=3)
        
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred_regions, regions_gt, pred_kernels, kernels_gt, pred_similarities, text_mask_ndi_labels, kernel_mask_ndi_labels):
        
        # ohem_masks = self.ohem_batch(pred_regions, regions_gt, self.ohem_ratio)
        
        ohem_masks = regions_gt
        
        loss_regions = self.loss_regions(pred_regions, regions_gt, ohem_masks) # pred_regions, regions_gt, ohem_mask
        # print(f"==>> loss_regions: {loss_regions}")
        loss_kernel = self.loss_kernel(pred_kernels, kernels_gt) # pred_kernels, kernels_gt
        # print(f"==>> loss_kernel: {loss_kernel}")
        loss_aggregation = self.loss_aggregation(pred_similarities, regions_gt, kernels_gt, text_mask_ndi_labels, kernel_mask_ndi_labels) # pred_similarities, regions_mask, kernels_mask
        # print(f"==>> loss_aggregation: {loss_aggregation}")
        loss_discrimination = self.loss_discrimination(pred_similarities, kernel_mask_ndi_labels) # pred_similarities, kernels_mask
        # print(f"==>> loss_discrimination: {loss_discrimination}")
        
        loss = loss_regions + self.alpha * loss_kernel + self.beta * (loss_aggregation + loss_discrimination)
        return dict(loss=loss, loss_regions=loss_regions, loss_kernel=loss_kernel, loss_aggregation=loss_aggregation, loss_discrimination=loss_discrimination)
        
    
    def ohem_single(self, pred_regions, regions_gt, ohem_ratio=3):
        pos_num = int(torch.sum((regions_gt > 0.5).to(dtype=torch.float32)))
        
        if pos_num == 0:
            return regions_gt
        
        neg_num = int(torch.sum((regions_gt <= 0.5).to(dtype=torch.float32)))
        neg_num = int(min(pos_num * ohem_ratio, neg_num))
        
        if neg_num == 0:
            return regions_gt
        
        neg_score = torch.masked_select(pred_regions, regions_gt <= 0.5)
        neg_score_sorted = torch.sort(-neg_score).values
        threshold = -neg_score_sorted[neg_num - 1]
        
        selected_mask = torch.logical_or((pred_regions >= threshold), (regions_gt > 0.5))
        selected_mask = selected_mask.reshape_as(regions_gt).to(dtype=torch.float32)
        return selected_mask
    
    def ohem_batch(self, pred_regions, regions_gt, ohem_ratio=3):
        selected_masks = []
        for i in range(pred_regions.shape[0]):
            selected_masks.append(self.ohem_single(pred_regions[i], regions_gt[i], ohem_ratio))

        selected_masks = torch.cat(selected_masks, 0).to(dtype=torch.float32)
        return selected_masks
        