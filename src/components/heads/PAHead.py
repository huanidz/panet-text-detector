import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kersize, stride, padding, is_activation:bool = True) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kersize, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.is_activation = is_activation
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.is_activation: 
            x = self.relu(x)
        return x
    
class PAHead(nn.Module):
    def __init__(self, channels_scale:int = 128, similarity_channels:int = 32) -> None:
        super(PAHead, self).__init__()
        self.feature_map_channels = 4 * channels_scale
        self.conv_text_regions = ConvBlock(in_channels=self.feature_map_channels, out_channels=1, kersize=3, stride=1, padding=1)
        self.conv_text_kernels = ConvBlock(in_channels=self.feature_map_channels, out_channels=1, kersize=3, stride=1, padding=1)    
        self.conv_similarity = ConvBlock(in_channels=self.feature_map_channels, out_channels=similarity_channels, kersize=3, stride=1, padding=1)
        
    def forward(self, x):
        text_regions = self.conv_text_regions(x)
        text_kernels = self.conv_text_kernels(x)
        similarity = self.conv_similarity(x)
        return text_regions, text_kernels, similarity
        