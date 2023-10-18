import torch.nn as nn

# Depthwise separable convolution
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

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
    
class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(DeConvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return x

class UpScaleAggregator(nn.Module):
    def __init__(self, channels_scale:int = 128):
        super(UpScaleAggregator, self).__init__()
        self.upsample_conv = DeConvBlock(in_channels=channels_scale, out_channels=channels_scale, kernel_size=2, stride=2, padding=0)
        self.dw_conv = DWConv(in_channels=channels_scale, out_channels=channels_scale, stride=1)
        self.last_conv = ConvBlock(in_channels=channels_scale, out_channels=channels_scale, kersize=1, stride=1, padding=0)
        
    def forward(self, f_size_A, f_size_2A):
        upsample = self.upsample_conv(f_size_A)
        out = upsample + f_size_2A
        out = self.dw_conv(out)
        out = self.last_conv(out)
        return upsample, out

class DownScaleAggregator(nn.Module):
    def __init__(self, channels_scale:int = 128):
        super(DownScaleAggregator, self).__init__()
        self.upsample_conv = DeConvBlock(in_channels=channels_scale, out_channels=channels_scale, kernel_size=2, stride=2, padding=0)
        self.dw_conv = DWConv(in_channels=channels_scale, out_channels=channels_scale, stride=2)
        self.last_conv = ConvBlock(in_channels=channels_scale, out_channels=channels_scale, kersize=1, stride=1, padding=0)
        
    def forward(self, upsample_skip_size_A, f_size_2A):
        upsample_brigde = self.upsample_conv(upsample_skip_size_A) # Size 2A
        out = upsample_brigde + f_size_2A # Size 2A
        out = self.dw_conv(out)
        out = self.last_conv(out) # Size A
        return out
        
class FPEM(nn.Module):
    def __init__(self, channels_scale:int = 128):
        super().__init__()
        self.upscale_aggregator_1 = UpScaleAggregator(channels_scale=channels_scale)
        self.upscale_aggregator_2 = UpScaleAggregator(channels_scale=channels_scale)
        self.upscale_aggregator_3 = UpScaleAggregator(channels_scale=channels_scale)
        
        self.downscale_aggregator_1 = DownScaleAggregator(channels_scale=channels_scale)
        self.downscale_aggregator_2 = DownScaleAggregator(channels_scale=channels_scale)
        self.downscale_aggregator_3 = DownScaleAggregator(channels_scale=channels_scale)
        
    def forward(self, f_4, f_8, f_16, f_32):
        # First to third is in order of left-to-right from the paper figure
        f_32_identity_to_f_32 = first_skip_out = f_32
        f_32_upscale_to_f_16, second_skip_out = self.upscale_aggregator_1(f_32_identity_to_f_32, f_16)
        f_16_upscale_to_f_8, third_skip_out = self.upscale_aggregator_2(f_32_upscale_to_f_16, f_8)
        f_8_upscale_to_f_4, _ = self.upscale_aggregator_3(f_16_upscale_to_f_8, f_4)
        
        # First to third is in order of right-to-left from the paper figure
        f_4_downscale_to_f_8 = self.downscale_aggregator_1(third_skip_out, f_8_upscale_to_f_4)
        f_8_downscale_to_f_16 = self.downscale_aggregator_2(second_skip_out, f_4_downscale_to_f_8)
        f_16_downscale_to_f_32 = self.downscale_aggregator_3(first_skip_out, f_8_downscale_to_f_16)
        
        return f_8_upscale_to_f_4, f_4_downscale_to_f_8, f_8_downscale_to_f_16, f_16_downscale_to_f_32
        