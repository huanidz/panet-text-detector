import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kersize, stride, padding, is_activation:bool = True) -> None:
        super().__init__()
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
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.conv_1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_2 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        identify = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x + identify

class ResBlock50(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels) -> None:
        super(ResBlock50, self).__init__()
        self.conv_1x1a = ConvBlock(in_channels, middle_channels, kersize=1, stride=1, padding=1)
        self.conv_3x3 = ConvBlock(middle_channels, middle_channels, kersize=3, stride=1, padding=1)
        self.conv_1x1b = ConvBlock(middle_channels, out_channels, kersize=1, stride=1, padding=1)
        
    def forward(self, x):
        identity = x
        x = self.conv_1x1a(x)
        x = self.conv_3x3(x)
        x = self.conv_1x1b(x)
        return x + identity

class ResNet18(nn.Module):
    def __init__(self, img_channels=3, channels_scale:int = 128):
        super().__init__()
        self.conv_1 = ConvBlock(in_channels=img_channels, out_channels=64, kersize=7, stride=2, padding=3)
        self.conv_1_half = ConvBlock(in_channels=64, out_channels=64, kersize=3, stride=2, padding=1)
        
        # ResBlock 1
        self.conv2_x = nn.Sequential(
            ResBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ResBlock(64, 64, kernel_size=3, stride=1, padding=1)
        )
        
        # Bridging (used to half the shape and converting the channels)
        self.conv2_3a = ConvBlock(in_channels=64, out_channels=128, kersize=3, stride=2, padding=1)
        self.conv2_3b = ConvBlock(in_channels=128, out_channels=128, kersize=3, stride=1, padding=1)
        
        # ResBlock 2
        self.conv3_x = nn.Sequential(
            ResBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ResBlock(128, 128, kernel_size=3, stride=1, padding=1)
        )
        
        self.conv3_4a = ConvBlock(in_channels=128, out_channels=256, kersize=3, stride=2, padding=1)
        self.conv3_4b = ConvBlock(in_channels=256, out_channels=256, kersize=3, stride=1, padding=1)
        
        # ResBlock 2
        self.conv4_x = nn.Sequential(
            ResBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ResBlock(256, 256, kernel_size=3, stride=1, padding=1)
        )
        
        self.conv4_5a = ConvBlock(in_channels=256, out_channels=512, kersize=3, stride=2, padding=1)
        self.conv4_5b = ConvBlock(in_channels=512, out_channels=512, kersize=3, stride=1, padding=1)
        
        # ResBlock 2
        self.conv5_x = nn.Sequential(
            ResBlock(512, 512, kernel_size=3, stride=1, padding=1),
            ResBlock(512, 512, kernel_size=3, stride=1, padding=1)
        )
        
        self.conv_fr_4 = ConvBlock(in_channels=64, out_channels=channels_scale, kersize=3, stride=1, padding=1)
        self.conv_fr_8 = ConvBlock(in_channels=128, out_channels=channels_scale, kersize=3, stride=1, padding=1)
        self.conv_fr_16 = ConvBlock(in_channels=256, out_channels=channels_scale, kersize=3, stride=1, padding=1)
        self.conv_fr_32 = ConvBlock(in_channels=512, out_channels=channels_scale, kersize=3, stride=1, padding=1)
        
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_1_half(x) # Replace for Maxpool because "Max-pool is shit - Geoffrey Hinton"
        fr_4 = x
        
        x = self.conv2_x(x)
        x = self.conv2_3a(x)
        x = self.conv2_3b(x)
        x = self.conv3_x(x)
        fr_8 = x
        
        
        x = self.conv3_4a(x)
        x = self.conv3_4b(x)
        x = self.conv4_x(x)
        fr_16 = x
        
        
        x = self.conv4_5a(x)
        x = self.conv4_5b(x)
        x = self.conv5_x(x)
        fr_32 = x
        
        return fr_4, fr_8, fr_16, fr_32
    
class ResNet50(nn.Module):
    def __init__(self, img_channels:int=3) -> None:
        super().__init__()
        self.conv_1 = ConvBlock(in_channels=img_channels, out_channels=64, kersize=7, stride=2, padding=3)
        self.conv_1_half = ConvBlock(in_channels=64, out_channels=64, kersize=3, stride=2, padding=1)
        
        # NEED BRIDGE
        
        # ResBlock 1
        self.conv2_x = nn.Sequential(
            ResBlock50(in_channels=64, middle_channels=64, out_channels=256),
            ResBlock50(in_channels=256, middle_channels=64, out_channels=256),
            ResBlock50(in_channels=256, middle_channels=64, out_channels=256)
        )
        
        # Bridging (used to half the shape and converting the channels)
        self.conv2_3a = ConvBlock(in_channels=64, out_channels=128, kersize=3, stride=2, padding=1)
        self.conv2_3b = ConvBlock(in_channels=128, out_channels=128, kersize=3, stride=1, padding=1)
        
        # ResBlock 2
        self.conv3_x = nn.Sequential(
            ResBlock50(in_channels=64, middle_channels=128, out_channels=512),
            ResBlock50(in_channels=512, middle_channels=128, out_channels=512),
            ResBlock50(in_channels=512, middle_channels=128, out_channels=512),
            ResBlock50(in_channels=512, middle_channels=128, out_channels=512)
        )
        
        # NEED BRIDGE
        
        # ResBlock 3
        self.conv4_x = nn.Sequential(
            ResBlock50(in_channels=64, middle_channels=256, out_channels=1024),
            ResBlock50(in_channels=1024, middle_channels=256, out_channels=1024),
            ResBlock50(in_channels=1024, middle_channels=256, out_channels=1024),
            ResBlock50(in_channels=1024, middle_channels=256, out_channels=1024),
            ResBlock50(in_channels=1024, middle_channels=256, out_channels=1024),
            ResBlock50(in_channels=1024, middle_channels=256, out_channels=1024)
        )
        
        # NEED BRIDGE
        
        self.conv5_x = nn.Sequential(
            ResBlock50(in_channels=64, middle_channels=512, out_channels=2048),
            ResBlock50(in_channels=2048, middle_channels=512, out_channels=2048),
            ResBlock50(in_channels=2048, middle_channels=512, out_channels=2048)
        )