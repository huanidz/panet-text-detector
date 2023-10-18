import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .ResNet import ResNet18

class ResNet18PreTrainedImageNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.resbase = ResNet18(img_channels=3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.resbase(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x
