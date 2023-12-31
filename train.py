import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import time
from src.datasets.PANData import PANDataset
from src.components.net.PANet import PANet
from src.components.losses.loss import PANLoss

torch_version = torch.__version__

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 600
batch_size = 8
save_checkpoint_each_N_epoch = 20

images_folder = "./data/train_images/"
labels_folder = "./data/train_labels/"

# images_folder = "./data/bad_sample_image/"
# labels_folder = "./data/bad_sample_label/"

target_image_size = 640
kernel_shrink_ratio = 0.7
np.set_printoptions(threshold=sys.maxsize)
dataset = PANDataset(images_folder=images_folder, labels_folder=labels_folder, target_image_size=target_image_size, kernel_shrink_ratio=kernel_shrink_ratio)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

model = PANet(channels_scale=128, num_fpem=2, similarity_channels_scale=4).to(device=device)

criteria = PANLoss(ohem_ratio=3, alpha=0.5, beta=0.25)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99, weight_decay=5e-4)

print("PyTorch version:", torch_version)
print("Training info:")
print("Num epochs:", num_epochs)
print("Batch size:", batch_size)
print("Enable autograd anomaly detection..")
torch.autograd.set_detect_anomaly(True)
print("Start training...")
# model = torch.compile(model)

# checkpoint = torch.load("./model_checkpoint_epoch_60.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.train()

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

        running_loss += loss.item()
        running_loss_text += all_loss['loss_regions']
        running_loss_kernel += all_loss['loss_kernel']
        running_loss_agg += all_loss['loss_aggregation']
        running_loss_dis += all_loss['loss_discrimination']
        
        if (batch_idx + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {running_loss / 20:.6f}, Loss_text:  {running_loss_text/20:.6f}, Loss_kernel: {running_loss_kernel/20:.6f}, Loss_agg: {running_loss_agg/20:.6f}, Loss_dis: {running_loss_dis/20:.6f}')
            running_loss = 0.0
            running_loss_text = 0.0
            running_loss_kernel = 0.0
            running_loss_agg = 0.0
            running_loss_dis = 0.0
        
    if (epoch + 1) % save_checkpoint_each_N_epoch == 0 or (epoch + 1) == num_epochs:
        # Saving the model after each N epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f"./checkpoints/epoch_{epoch + 1}.pth")
        
        
        
        


