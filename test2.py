import torch

A = torch.Tensor([[1,2,3],
                  [4,5,6]])

B = torch.Tensor([[2,2,2],
                  [2,2,2]])

C = torch.linalg.norm(A - B)
print(f"==>> C: {C}")

