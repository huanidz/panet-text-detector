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

images_folder = "./data/infer/"
labels_folder = "./data/infer_labels/"
target_image_size = 640
kernel_shrink_ratio = 0.7

batch_size = 1

model = PANet(channels_scale=128, num_fpem=2, similarity_channels_scale=4).to(device=device)
dataset = PANDataset(images_folder=images_folder, labels_folder=labels_folder, target_image_size=target_image_size, kernel_shrink_ratio=kernel_shrink_ratio, mode='eval')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

checkpoint = torch.load("./checkpoints/epoch_500.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

threshold = 0.7

with torch.no_grad():
    for batch_idx, (original, model_input) in tqdm(enumerate(dataloader)):
        model_input = model_input.to(device)
        original = original.numpy().squeeze(0)
        
        text_regions, text_kernels, similarities = model(model_input)
        
        results = postprocess(text_predictions=text_regions, kernel_predictions=text_kernels,
                                    similarity_predictions=similarities, threshold=threshold)
        
        
        # text_regions = torch.sigmoid(text_regions)
        # text_regions = (text_regions > threshold).float().squeeze(0).squeeze(0)
        # text_regions_np = text_regions.cpu().numpy() #convert PyTorch tensor to Numpy array.
        # text_regions_np *= 255
        # text_regions_np = text_regions_np.astype(np.uint8)


original_image = cv2.imread("./data/infer/1002.jpg", cv2.IMREAD_COLOR)
print(f"==>> original_image.shape: {original_image.shape}")

original_image = cv2.resize(original_image, (640, 640))


shit = results[0].squeeze(0)
num_shit = shit.max()

addWeighted_alpha = 0.7

for i in range(1, num_shit + 1):
    mask_i = (shit == i)
    zero_nine = np.zeros_like(mask_i, dtype=np.uint8)
    zero_nine[mask_i] = 1
    contours, hierarchy = cv2.findContours(zero_nine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = tuple([contour * 4 for contour in contours])
    
    # if len(contours) > 1:
    #     raise ValueError("SHIT")
    
    
    cv2.drawContours(original_image, contours, -1, (0, 255, 0), 3) 
    
    # coordinates_label_i = np.array(np.argwhere(mask_i)) * 4
    # coordinates_label_i = coordinates_label_i[:, ::-1]
    
    
    # hull = cv2.convexHull(coordinates_label_i, )    
    
    # cv2.polylines(original_image, [hull], True, (0, 255, 0), 3, cv2.LINE_AA)
    
cv2.imwrite("./data/infer_result/heheOrigin.png", original_image)

shit[shit > 0] = 1
shit = shit.astype(np.uint8)
shit *= 255



print(f"==>> shit.shape: {shit.shape}")

cv2.imwrite("./data/infer_result/hehe2.png", shit)
        
        
        