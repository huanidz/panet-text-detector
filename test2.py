import numpy as np
from scipy import ndimage
import torch

# assuming mask is your input mask
mask = np.array([[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                     [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                     [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.int32)
print(f"==>> mask.shape: {mask.shape}")

# label each cluster
labels, n = ndimage.label(mask)
print(f"==>> n: {n}")
print(f"==>> labels: {labels}")

# find sum of all regions
sums = ndimage.sum(mask, labels, np.arange(n)+1)
print(f"==>> mask.shape: {mask.shape}")
print(f"==>> sums: {sums}")

# map the sums back to the labels using 0 as placeholder for areas not covered by a mask
mapped_sums = np.where(mask>0, sums[labels-1], 0)

print(f"{mapped_sums}")
# if you need integers..
mapped_sums = mapped_sums.astype(int)

# # create a result array where each value in each labeled region is replaced by the sum of that region
# result = sums[labels - 1]

# # this result will be float, so if you need integers..
# result = result.astype(int)
# print(f"==>> result: {result}")

