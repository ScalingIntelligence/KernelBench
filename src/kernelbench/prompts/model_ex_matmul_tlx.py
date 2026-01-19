import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.matmul(a, b)

def get_inputs():
    # randomly generate input tensors for a matmul operation
    # Using sizes compatible with the TLX kernel logic (e.g. divisible by block sizes ideally, though the kernel handles remainders)
    # The kernel has BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64
    # Let's use standard sizes.
    M = 4096
    N = 4096
    K = 4096
    a = torch.randn(M, K).cuda().to(torch.float16)
    b = torch.randn(K, N).cuda().to(torch.float16)
    return [a, b]

def get_init_inputs():
    return []
