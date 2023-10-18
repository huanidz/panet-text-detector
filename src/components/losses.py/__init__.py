import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        
    def forward(self, N):
        L = 1.0/N 
        
class DiscriminationLoss(nn.Module):
    def __init__(self, sigma_dis:float = 3):
        super().__init__()
        
    def forward(self, N):
        pass
    
class TextDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(TextDiceLoss, self).__init__()
        self.eps = epsilon
        
    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        
        intersection = torch.sum(pred * gt) # pred should be sigmoied.
        union = torch.sum(pred * gt) + torch.sum(gt) + self.eps
        loss = 1 - (2 * intersection / union)
        return loss
    
class KernelDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(KernelDiceLoss, self).__init__()
        self.eps = epsilon
        
    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        
        intersection = torch.sum(pred * gt) # pred should be sigmoied.
        union = torch.sum(pred * gt) + torch.sum(gt) + self.eps
        loss = 1 - (2 * intersection / union)
        return loss