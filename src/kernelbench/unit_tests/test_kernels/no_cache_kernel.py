import torch
import torch.nn as nn

###########################################
# Control for cached_output_kernel.py: the same matmul without a cache.
# It launches GPU work on every call and should pass.
###########################################

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)
