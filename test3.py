import numpy as np 
from scipy import ndimage

# masks - (N, W, H) batch of masks
# assuming mask is your input mask
masks = np.array([[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                     [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                     [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                 
                 [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                     [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]]], dtype=np.int32)

# Label each mask 
labels = ndimage.label(masks, output=np.int32)[0] 

# Get count of labels for each mask
label_counts = np.bincount(labels.ravel())

# Compute sums for each label 
sums = ndimage.sum(masks, labels, index=np.arange(label_counts.size))

# Map sums to labels
mapped_sums = sums[labels - 1]  

# Set 0 for non-mask areas
mapped_sums = np.where(masks > 0, mapped_sums, 0)

# Optional: convert to int
mapped_sums = mapped_sums.astype(int)
print(f"==>> mapped_sums: {mapped_sums}")