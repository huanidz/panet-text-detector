import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import cv2

from src.datasets.PANData import PANDataset
from src.components.net.PANet import PANet
from src.processes.postprocessing import postprocess

np.set_printoptions(threshold=sys.maxsize)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

images_folder = "./data/train_images/"
labels_folder = "./data/infer_labels/"
target_image_size = 640
kernel_shrink_ratio = 0.7

batch_size = 1

model = PANet(channels_scale=128, num_fpem=2, similarity_channels_scale=4).to(device=device)
dataset = PANDataset(images_folder=images_folder, labels_folder=labels_folder, target_image_size=target_image_size, kernel_shrink_ratio=kernel_shrink_ratio, mode='infer')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

checkpoint = torch.load("./checkpoints/epoch_500.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

threshold = 0.712

results = None
with torch.no_grad():
    for batch_idx, (filename, original, model_input) in tqdm(enumerate(dataloader), desc="Inference and reconstruct masks..."):
        model_input = model_input.to(device)
        original = original.numpy()
        
        text_regions, text_kernels, similarities = model(model_input)
        
        if results is None:
            results = postprocess(filename=filename, original=original, text_predictions=text_regions, kernel_predictions=text_kernels, similarity_predictions=similarities, threshold=threshold) # Array of reconstructed outputs
        else:
            results += postprocess(filename=filename, original=original, text_predictions=text_regions, kernel_predictions=text_kernels, similarity_predictions=similarities, threshold=threshold)
        

for result in tqdm(results, desc="Prepare result's visualization..."):
    filename = result[0][0]
    original_image = result[1].squeeze(0)
    reconstucted_mask = result[2]
    
    original_image = cv2.resize(original_image, dsize=(target_image_size, target_image_size))
    reconstucted_mask = reconstucted_mask.squeeze(0)
    num_regions_in_image = reconstucted_mask.max()
    
    for i in range(1, num_regions_in_image + 1):
        mask_i = (reconstucted_mask == i)
        temp_mask = np.zeros_like(mask_i, dtype=np.uint8)
        temp_mask[mask_i] = 1
        contours, hierarchy = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours = tuple([contour * 4 for contour in contours])

        cv2.drawContours(original_image, contours, -1, (0, 255, 0), 3)
        
    cv2.imwrite(f"./data/infer_result/{filename}", original_image)
        
        