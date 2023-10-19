import torch
from src.components.net.PANet import PANet
from src.utils.common import count_parameters


if __name__ == "__main__":
    
    model = PANet(channels_scale=128, num_fpem=2, similarity_channels_scale=8)
    print(f"==>> model: {count_parameters(model)}")

    input = torch.randn((1, 3, 512, 512))
    text_regions, text_kernels, similarity = model(input)
    print(f"==>> text_regions.shape: {text_regions.shape}")
    print(f"==>> text_kernels.shape: {text_kernels.shape}")
    print(f"==>> similarity.shape: {similarity.shape}")
    