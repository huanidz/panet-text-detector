from src.components.backbones.ResNet import ResNet18
from src.components.necks.ChannelReduction import ChannelReducer
from src.utils.common import count_parameters

import torch

if __name__ == "__main__":    
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ResNet18()
    
    print("num params: ", count_parameters(model))
    
    
    input = torch.randn((1, 3, 512, 512))
    fr_4, fr_8, fr_16, fr_32 = model(input)
    
    channelReducer = ChannelReducer(target_channels=128)
    fr_4, fr_8, fr_16, fr_32 = channelReducer(fr_4, fr_8, fr_16, fr_32)
    print(f"==>> fr_4.shape: {fr_4.shape}")
    print(f"==>> fr_8.shape: {fr_8.shape}")
    print(f"==>> fr_16.shape: {fr_16.shape}")
    print(f"==>> fr_32.shape: {fr_32.shape}")
    