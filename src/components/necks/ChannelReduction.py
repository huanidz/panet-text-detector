import torch.nn as nn

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

class ChannelReducer(nn.Module):
    def __init__(self, target_channels: int = 128):
        super().__init__()
        self.target_channels = target_channels

        self.conv_fr_4 = ConvBlock(in_channels=64, out_channels=self.target_channels, kersize=3, stride=1, padding=1, is_activation=False)
        self.conv_fr_8 = ConvBlock(in_channels=128, out_channels=self.target_channels, kersize=3, stride=1, padding=1, is_activation=False)
        self.conv_fr_16 = ConvBlock(in_channels=256, out_channels=self.target_channels, kersize=3, stride=1, padding=1, is_activation=False)
        self.conv_fr_32 = ConvBlock(in_channels=512, out_channels=self.target_channels, kersize=3, stride=1, padding=1, is_activation=False)
        
    def forward(self, fr_4, fr_8, fr_16, fr_32):
        fr_4_out = self.conv_fr_4(fr_4)
        fr_8_out = self.conv_fr_8(fr_8)
        fr_16_out = self.conv_fr_16(fr_16)
        fr_32_out = self.conv_fr_32(fr_32)
        return fr_4_out, fr_8_out, fr_16_out, fr_32_out
    
        