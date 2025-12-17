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
        
        # ThunderKittens uses bfloat16 for tile operations
        A_bf16 = A.to(torch.bfloat16).contiguous()
        B_bf16 = B.to(torch.bfloat16).contiguous()
        C = torch.zeros((M, N), device=A.device, dtype=torch.bfloat16).contiguous()
        
        # Call into TK pybind wrapper
        tk_kernels.dispatch_add(A_bf16, B_bf16, C, int(M), int(N))
        
        return C
