import torch
import torch.nn as nn
import tk_kernels

class ModelNew(nn.Module):
    """
    ThunderKittens-accelerated elementwise addition (C = A + B)
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, N = A.shape
        assert A.shape == B.shape, "Input tensors must have the same shape"
        
        C = torch.zeros((M, N), device=A.device, dtype=torch.float32).contiguous()
        
        # Call into TK pybind wrapper
        tk_kernels.dispatch_add(A, B, C, int(M), int(N))
        
        return C
