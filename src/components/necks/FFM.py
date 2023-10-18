import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(DeConvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return x
    
class FFM(nn.Module):
    
    """
    The target size is (Original//4 x Original//4)
    """
    
    def __init__(self, channels_scale:int = 128):
        super().__init__()
        self.upsample_1 = DeConvBlock(in_channels=channels_scale, out_channels=channels_scale, kernel_size=2, stride=2, padding=0)
        self.upsample_21 = DeConvBlock(in_channels=channels_scale, out_channels=channels_scale, kernel_size=2, stride=2, padding=0)
        self.upsample_22 = DeConvBlock(in_channels=channels_scale, out_channels=channels_scale, kernel_size=2, stride=2, padding=0)
        self.upsample_31 = DeConvBlock(in_channels=channels_scale, out_channels=channels_scale, kernel_size=2, stride=2, padding=0)
        self.upsample_32 = DeConvBlock(in_channels=channels_scale, out_channels=channels_scale, kernel_size=2, stride=2, padding=0)
        self.upsample_33 = DeConvBlock(in_channels=channels_scale, out_channels=channels_scale, kernel_size=2, stride=2, padding=0)
        
    def forward(self, f_4, f_8, f_16, f_32): # Expect f_# is the already added (already fusion)
        f_8 = self.upsample_1(f_8)
        f_16 = self.upsample_22(self.upsample_21(f_16))
        f_32 = self.upsample_33(self.upsample_32(self.upsample_31(f_32)))
        
        return torch.cat([f_4, f_8, f_16, f_32], dim=1) # Concat via channel dim
        