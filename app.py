import torch
from src.components.net.PANet import PANet
from src.utils.common import count_parameters


if __name__ == "__main__":
    
    model = PANet(channels_scale=128, num_fpem=2)
    print(f"==>> model: {count_parameters(model)}")

    input = torch.randn((1, 3, 512, 512))
    output = model(input)
    print(f"==>> output.shape: {output.shape}")
    