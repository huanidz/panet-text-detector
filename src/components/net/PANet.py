import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..backbones.ResNet import ResNet18
from ..necks.ChannelReduction import ChannelReducer
from ..necks.FPEM import FPEM
from ..necks.FFM import FFM

class PANet(nn.Module):
    def __init__(self, channels_scale:int = 128, num_fpem:int = 2):
        super(PANet, self).__init__()
        self.backbone = ResNet18(img_channels=3, channels_scale=channels_scale)
        self.channel_reducer = ChannelReducer(target_channels=channels_scale)
        self.fpems = nn.ModuleList()
        for _ in range(num_fpem):
            self.fpems.append(FPEM(channels_scale=channels_scale))
        self.ffm = FFM(channels_scale=channels_scale)
        
    def forward(self, x):
        f_4, f_8, f_16, f_32 = self.backbone(x)
        f_4, f_8, f_16, f_32 = self.channel_reducer(f_4, f_8, f_16, f_32)
        
        fusion_f4 = torch.zeros_like(f_4)
        fusion_f_8 = torch.zeros_like(f_8)
        fusion_f_16 = torch.zeros_like(f_16)
        fusion_f_32 = torch.zeros_like(f_32)
        
        for fpem in self.fpems:
            f_4, f_8, f_16, f_32 = fpem(f_4, f_8, f_16, f_32)
            fusion_f4.add_(f_4)
            fusion_f_8.add_(f_8)
            fusion_f_16.add_(f_16)
            fusion_f_32.add_(f_32)
        
        output = self.ffm(fusion_f4, fusion_f_8, fusion_f_16, fusion_f_32)
        return output