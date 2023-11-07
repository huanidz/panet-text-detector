import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import cv2

from src.datasets.PANData import PANDataset
from src.components.net.PANet import PANet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

images_folder = "./data/test_images/"
labels_folder = "./data/test_labels/"
target_image_size = 640
kernel_shrink_ratio = 0.7

batch_size = 1

model = PANet(channels_scale=128, num_fpem=2, similarity_channels_scale=4).to(device=device)
dataset = PANDataset(images_folder=images_folder, labels_folder=labels_folder, target_image_size=target_image_size, kernel_shrink_ratio=kernel_shrink_ratio, mode='eval')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

checkpoint = torch.load("./checkpoints/epoch_500.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

threshold = 0.5

with torch.no_grad():
    for batch_idx, (original, model_input) in tqdm(enumerate(dataloader)):
        model_input = model_input.to(device)
        original = original.numpy().squeeze(0)
        # text_mask = item['text_mask'].to(device)        
        # kernel_mask = item['kernel_mask'].to(device)
        # text_mask_ndi_labels = item['text_mask_ndi_labels'].to(device)
        # kernel_mask_ndi_labels = item['kernel_mask_ndi_labels'].to(device)
        
        text_regions, text_kernels, similarities = model(model_input)
        
        text_regions = torch.sigmoid(text_regions)
        text_regions = (text_regions > threshold).float().squeeze(0).squeeze(0)
        text_regions_np = text_regions.cpu().numpy() #convert PyTorch tensor to Numpy array.
        text_regions_np *= 255
        text_regions_np = text_regions_np.astype(np.uint8)
        cv2.imwrite(f"./output/original_{batch_idx}.png", original)
        cv2.imwrite(f"./output/output_{batch_idx}.png", text_regions_np)
        