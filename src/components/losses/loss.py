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

class AggregationAndDiscriminationLoss(nn.Module):
    def __init__(self, sigma_agg:float = 0.5, sigma_dis:float = 3.0) -> None:
        super(AggregationAndDiscriminationLoss, self).__init__()
        self.sigma_agg = sigma_agg
        self.sigma_dis = sigma_dis
        
    def forward(self, pred_similarities, text_mask_ndi_labels, kernel_mask_ndi_labels):
        
        batch_size = pred_similarities.shape[0]
        similarities_channel = pred_similarities.shape[1]
        
        pred_similarities = pred_similarities.contiguous().reshape((batch_size, similarities_channel, -1)) # B, Similarity_C, H*W
        text_mask_ndi_labels = text_mask_ndi_labels.contiguous().reshape((batch_size, -1)) # B, H * W
        kernel_mask_ndi_labels = kernel_mask_ndi_labels.contiguous().reshape((batch_size, -1)) # B, H * W

        L_aggregation = []
        L_discrimination = []
        
        # i is the item at the batch_i, we loop through each item in the batch
        # pred_sim_i: predict similarities
        # T_mask_i: text regions mask
        # K_mask_i: kernel regions mask
        # T_mask_labels_i: text regions label (use ndimage)
        # K_mask_labels_i: kernel regions label (use ndimage)

        for pred_sim_i, T_mask_labels_i, K_mask_labels_i in zip(pred_similarities,
                                                                text_mask_ndi_labels,
                                                                kernel_mask_ndi_labels):
            
            num_kernels = K_mask_labels_i.max().int() # Or can use: T_mask_labels_i.max().int()            
            num_text_instances = T_mask_labels_i.max().int()
            
            if num_kernels <= 1:
                L_discrimination.append(torch.tensor(data=0.0, device=pred_similarities.device, dtype=torch.float))
            
            total_kernels_indices = torch.arange(1, num_kernels + 1)
            
            L_agg_single = 0.0
            for i in range(1, num_kernels + 1):
                where_ones_kernel = (K_mask_labels_i == i).unsqueeze(dim=0)
                where_ones_text = (T_mask_labels_i == i).unsqueeze(dim=0)

                kernel_cardinality = where_ones_kernel.sum()
                text_cardinality = where_ones_text.sum()
                
                # Skipping the case where region is being overlap by other region(s)
                # if kernel_cardinality == 0 or text_cardinality == 0:
                #     continue
                
                pred_sim_of_kernel = (pred_sim_i * where_ones_kernel) / (kernel_cardinality + 1)
                pred_sim_of_text = pred_sim_i * where_ones_text

                # The Frobenius norm is the Euclidian norm of a matrix which is what the author mention that he's using Euclidian Norm to calculate distance between similarity vectors.
                norm = torch.norm(pred_sim_of_text - pred_sim_of_kernel, p='fro') - self.sigma_agg
                norm = torch.max(norm, 0)[0].pow(2)
                norm = torch.log(norm + 1)
                norm = norm / (text_cardinality + 1)
                L_agg_single += norm
            
            L_discrimination_single = 0.0
            for K_i_label, K_j_label in itertools.combinations(total_kernels_indices, 2):
                where_ones_K_i = (K_mask_labels_i == K_i_label).unsqueeze(dim=0)
                where_ones_K_j = (K_mask_labels_i == K_j_label).unsqueeze(dim=0)
                
                kernel_cardinality_K_i = where_ones_K_i.sum()
                kernel_cardinality_K_j = where_ones_K_j.sum()
                
                # One of two kernel being overlap (rarely happen)
                # if kernel_cardinality_K_i == 0 or kernel_cardinality_K_j == 0:
                #     continue
                
                pred_sim_of_K_i = (pred_sim_i * where_ones_K_i) / (kernel_cardinality_K_i + 0.001)
                pred_sim_of_K_j = (pred_sim_i * where_ones_K_j) / (kernel_cardinality_K_j + 0.001 )
                norm = self.sigma_dis - torch.norm(pred_sim_of_K_i - pred_sim_of_K_j, p='fro')
                norm = torch.max(norm, 0)[0].pow(2)
                norm = torch.log(norm + 1)
                L_discrimination_single += norm
            
            if num_kernels > 1:
                L_discrimination_single /= (num_kernels * (num_kernels - 1))
                L_discrimination.append(L_discrimination_single)
            
            L_aggregation.append(L_agg_single)
            
        L_aggregation = torch.stack(L_aggregation).sum()
        L_discrimination = torch.stack(L_discrimination).sum()
        
        return L_aggregation, L_discrimination
     
   
class DiscriminationLoss(nn.Module):
    def __init__(self, sigma_dis:float = 3.0) -> None:
        super(DiscriminationLoss, self).__init__()
        self.sigma_dis = sigma_dis
        
    def forward(self, pred_similarities, kernel_mask_ndi_labels):
        
        batch_size = pred_similarities.shape[0]
        similarities_channel = pred_similarities.shape[1]

        pred_similarities = pred_similarities.contiguous().reshape((batch_size, similarities_channel, -1)) # B, Similarity_C, H*W
        kernel_mask_ndi_labels = kernel_mask_ndi_labels.contiguous().reshape((batch_size, -1)) # B, H * W
        
        L_discrimination = []
        
        # i is the item at the batch_i, we loop through each item in the batch
        # pred_sim_i: predict similarities
        # K_mask_i: kernel regions mask
        # K_mask_labels_i: kernel regions label (use ndimage)

        for pred_sim_i, K_mask_labels_i in zip(pred_similarities,
                                                kernel_mask_ndi_labels):
            
            num_kernels = K_mask_labels_i.max().int()
            
            if num_kernels <= 1:
                L_discrimination.append(torch.tensor(data=0.0, device=pred_similarities.device, dtype=torch.float))
                continue
            
            total_kernels_indices = torch.arange(1, num_kernels + 1)
            
            L_discrimination_single = 0
            for K_i_label, K_j_label in itertools.combinations(total_kernels_indices, 2):
                where_ones_K_i = (K_mask_labels_i == K_i_label).unsqueeze(dim=0)
                where_ones_K_j = (K_mask_labels_i == K_j_label).unsqueeze(dim=0)
                
                kernel_cardinality_K_i = where_ones_K_i.sum()
                kernel_cardinality_K_j = where_ones_K_j.sum()
                
                # One of two kernel being overlap (rarely happen)
                if kernel_cardinality_K_i == 0 or kernel_cardinality_K_j == 0:
                    continue
                
                pred_sim_of_K_i = (pred_sim_i * where_ones_K_i) / kernel_cardinality_K_i
                pred_sim_of_K_j = (pred_sim_i * where_ones_K_j) / kernel_cardinality_K_j
                norm = self.sigma_dis - torch.norm(pred_sim_of_K_i - pred_sim_of_K_j, p='fro')
                norm = torch.max(norm, 0)[0].pow(2)
                norm = torch.log(norm + 1)
                L_discrimination_single += norm
            
            L_discrimination_single /= (num_kernels * (num_kernels - 1))
            L_discrimination.append(L_discrimination_single)
        
        L_discrimination = torch.stack(L_discrimination).sum()
        
        return L_discrimination
        
        
class TextDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(TextDiceLoss, self).__init__()
        self.eps = epsilon
        
    def forward(self, pred_regions, regions_gt, ohem_mask):
        pred_regions = torch.sigmoid(pred_regions)
        
        # ohem_masks = ohem_masks.reshape([ohem_masks.shape[0], -1])
        pred_regions = pred_regions.contiguous().reshape([pred_regions.shape[0], -1]) 
        regions_gt = regions_gt.contiguous().reshape([regions_gt.shape[0], -1])
        ohem_mask = ohem_mask.contiguous().reshape([ohem_mask.shape[0], -1])
        
        pred_regions = pred_regions
        regions_gt = regions_gt
        
        intersection = torch.sum(pred_regions * regions_gt, dim=1)
        
        pred_regions_sum = torch.sum(pred_regions * pred_regions, dim=1) + self.eps
        regions_gt_sum = torch.sum(regions_gt * regions_gt, dim=1) + self.eps
        
        dice = (2.0 * intersection + self.eps)/(pred_regions_sum + regions_gt_sum)  
        
        loss = 1 - dice
        loss = loss.sum()
        
        return loss
    
class KernelDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(KernelDiceLoss, self).__init__()
        self.eps = epsilon
        
    def forward(self, pred_kernels, kernels_gt):
    
        pred_kernels = torch.sigmoid(pred_kernels)
        
        pred_kernels = pred_kernels.contiguous().reshape([pred_kernels.shape[0], -1])
        kernels_gt = kernels_gt.contiguous().reshape([kernels_gt.shape[0], -1])
        
        pred_kernels = pred_kernels
        kernels_gt = kernels_gt
        
        intersection = torch.sum(pred_kernels * kernels_gt, dim=1)
        pred_kernels_sum = torch.sum(pred_kernels * pred_kernels, dim=1) + self.eps
        kernels_gt_sum = torch.sum(kernels_gt * kernels_gt, dim=1) + self.eps
        
        dice = (2.0 * intersection + self.eps)/(pred_kernels_sum + kernels_gt_sum)  
        
        loss = 1 - dice
        loss = loss.sum()
        
        return loss
    
class PANLoss(nn.Module):
    def __init__(self, ohem_ratio:int = 3, alpha:float = 0.5, beta:float = 0.25):
        super(PANLoss, self).__init__()
        self.ohem_ratio = ohem_ratio
        
        self.loss_regions = TextDiceLoss(epsilon=1e-5)
        self.loss_kernel = KernelDiceLoss(epsilon=1e-5)
        self.loss_agg_and_dis = AggregationAndDiscriminationLoss(sigma_agg=0.5, sigma_dis=3.0) # Combine because it saves compute source for looping through kernels.
        
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred_regions, regions_gt, pred_kernels, kernels_gt, pred_similarities, text_mask_ndi_labels, kernel_mask_ndi_labels):
        
        ohem_masks = self.ohem_batch(pred_regions, regions_gt, self.ohem_ratio)
        
        loss_regions = self.loss_regions(pred_regions, regions_gt, ohem_masks) # pred_regions, regions_gt, ohem_mask
        loss_kernel = self.loss_kernel(pred_kernels, kernels_gt) # pred_kernels, kernels_gt

        loss_aggregation, loss_discrimination = self.loss_agg_and_dis(pred_similarities, text_mask_ndi_labels, kernel_mask_ndi_labels)

        loss = loss_regions + self.alpha * loss_kernel + self.beta * (loss_aggregation + loss_discrimination)
        return dict(loss=loss, loss_regions=loss_regions, loss_kernel=loss_kernel, loss_aggregation=loss_aggregation, loss_discrimination=loss_discrimination)
        
    
    def ohem_single(self, pred_regions, regions_gt, ohem_ratio=3):
        
        """
        As original, author will skip blurred regions/or they ignore some regions. So they have a training_mask
        
        On this implementation, i'm planning to train on full image.
        This has a biggest downside is the trouble of very small text region (This can be resolved by adjust the shrink ratio to smaller so the kernel don't be vanished).
        
        Conclusion: The training_mask will be all ones. Train on full input image.
        """
        
        training_mask = torch.ones_like(regions_gt)
        
        positive_pixels = int(torch.sum((regions_gt > 0).to(dtype=torch.float32))) # can use == 1 also
        
        if positive_pixels == 0: # Image doesn't contain any text region
            return training_mask
        
        negative_pixels = int(torch.sum((regions_gt == 0).to(dtype=torch.float32)))
        negative_pixels = int(min(positive_pixels * ohem_ratio, negative_pixels))
        
        if negative_pixels == 0: # Training full input image
            return training_mask
        
        """
        Explaination: 
        1. We first use the mask of where the negative (non-text region) to restrict where the negative prediction from the score map.
        
        2. From those negative predictions, the lowest value (lowest confidence) will be the pixel the model is "un-sure" about. Let's call that "UNSURE THRESHOLD"
        
        3. We will use the "Unsure Threshold" to also use for the positive (text-region). That means, if the positive prediction is not bigger than the "Unsure Threshold" then it will also be a "unsure" positive prediciton <-- We want to limit this to happen as much as possible.
        """
        
        pred_of_negative_pixels = torch.masked_select(pred_regions, regions_gt == 0)
        pred_negative_sorted, indices = torch.sort(-pred_of_negative_pixels) # Sort descending. This means very first items of the array will be pixel with highest score, And the very last items will be item with lowest scores
        
        unsure_threshold = -pred_negative_sorted[negative_pixels - 1] # We use "-" sign because we indexing a negative predictions where their values are often < 0. So this will flipped it to "+" value which is make sense value for a threshold.
        
        """
        Now, as what we planned, we gonna pick every pixel in the prediction score map where it's value is bigger than the threshold.
        
        This will do this main thing:
        
        What ever the pixel is, if it's bigger than the unsure_threshold, which mean it's "confidence". Remember that the unsure_threshold has "+" sign which mean it's predicting the pixel is text-region. So, if the ground truth for it is "background", this means it failed, and that is what we want to focus training on.
        
        Of we can interpret the logical_or() line as: Pick every pixel that the predict score map said it sure it's text region (even the groundtruth said no :3) And logical_or with the groundtruth, we then will know there're false pixels.
        """
        selected_mask = torch.logical_or((pred_regions >= unsure_threshold), (regions_gt > 0)).logical_and(training_mask)
        selected_mask = selected_mask.reshape_as(regions_gt).to(dtype=torch.float32)
        return selected_mask
    
    def ohem_batch(self, pred_regions, regions_gt, ohem_ratio=3):
        selected_masks = []
        for i in range(pred_regions.shape[0]):
            selected_masks.append(self.ohem_single(pred_regions[i], regions_gt[i], ohem_ratio))

        selected_masks = torch.cat(selected_masks, 0).to(dtype=torch.float32)
        return selected_masks
        