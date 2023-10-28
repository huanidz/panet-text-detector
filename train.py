import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.PANData import PANDataset
from src.components.net.PANet import PANet
from src.components.losses.loss import PANLoss

torch_version = torch.__version__

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
batch_size = 16

images_folder = "./data/train_images/"
labels_folder = "./data/train_labels/"
target_image_size = 320
kernel_shrink_ratio = 0.7

dataset = PANDataset(images_folder=images_folder, labels_folder=labels_folder, target_image_size=target_image_size, kernel_shrink_ratio=kernel_shrink_ratio)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = PANet(channels_scale=128, num_fpem=2, similarity_channels_scale=4).to(device=device)

criteria = PANLoss(ohem_ratio=3, alpha=0.5, beta=0.25)

optimizer = optim.SGD(model.parameters(), lr=0.001)

print("PyTorch version:", torch_version)
print("Enable autograd anomaly detection..")
torch.autograd.set_detect_anomaly(True)
print("Start training...")
# model = torch.compile(model)

for epoch in range(num_epochs):
    print(f"EPOCH {epoch + 1}/{num_epochs}")
    running_loss = 0.0
    running_loss_text = 0.0
    running_loss_kernel = 0.0
    running_loss_agg = 0.0
    running_loss_dis = 0.0
    
    for batch_idx, item in tqdm(enumerate(dataloader)):
        
        optimizer.zero_grad()
        
        image = item['image'].to(device)
        text_mask = item['text_mask'].to(device)        
        kernel_mask = item['kernel_mask'].to(device)
        text_mask_ndi_labels = item['text_mask_ndi_labels'].to(device)
        kernel_mask_ndi_labels = item['kernel_mask_ndi_labels'].to(device)
        
        
        text_regions, text_kernels, similarities = model(image)
        all_loss = criteria(text_regions, text_mask, text_kernels, kernel_mask, similarities, text_mask_ndi_labels, kernel_mask_ndi_labels) # pred_regions, regions_gt, pred_kernels, kernels_gt, pred_similarities
        loss = all_loss['loss']
        
        loss.backward()
        optimizer.step()

        # loss=loss, loss_regions=loss_regions, loss_kernel=loss_kernel, loss_aggregation=loss_aggregation, loss_discrimination=loss_discrimination

        running_loss += loss.item()
        running_loss_text += all_loss['loss_regions']
        running_loss_kernel += all_loss['loss_kernel']
        running_loss_agg += all_loss['loss_aggregation']
        running_loss_dis += all_loss['loss_discrimination']
        
        if (batch_idx + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {running_loss / 20:.4f}, Loss_text:  {running_loss_text/20:.4f}, Loss_kernel: {running_loss_kernel/20:.4f}, Loss_agg: {running_loss_agg/20:.4f}, Loss_dis: {running_loss_dis/20:.4f}')
            running_loss = 0.0
            running_loss_text = 0.0
            running_loss_kernel = 0.0
            running_loss_agg = 0.0
            running_loss_dis = 0.0
        
        
        
        


