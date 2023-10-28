import torch
import torch.nn as nn

class FFM(nn.Module):
    
    """
    The target size is (Original//4 x Original//4)
    """
    
    def __init__(self, channels_scale:int = 128):
        super().__init__()
        self.upsamle_2x = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamle_4x = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamle_8x = nn.Upsample(scale_factor=8, mode='bilinear')
     
    def forward(self, f_4, f_8, f_16, f_32): # Expect f_# is the already added (already fusion)

        f_8 = self.upsamle_2x(f_8)
        f_16 = self.upsamle_4x(f_16)
        f_32 = self.upsamle_8x(f_32)
        
        return torch.cat([f_4, f_8, f_16, f_32], dim=1) # Concat via channel dim
        