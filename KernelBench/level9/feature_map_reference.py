import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def flatten_diag_outer_product_off1(x, y):
    z = torch.einsum("...i,...j->...ij", x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N, 1)
    indices2 = torch.arange(0, N)
    return z[..., indicies[0], indicies[1]], z[..., indices2, indices2]

class Model(nn.Module):
    """
    Reference implementation of the Taylor Feature Map used in Linear Attention.
    This feature map approximates the softmax kernel using a second-order Taylor expansion.
    """
    def __init__(self, head_dim: int):
        super(Model, self).__init__()
        self.head_dim = head_dim
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(self.head_dim)
        self.rrd = math.sqrt(self.rd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
        Returns:
            torch.Tensor: Feature-mapped tensor of shape [batch_size, seq_len, num_heads, 1 + D + D*(D+1)/2]
        """
        # Second-order Taylor expansion components
        # 1 + x + 0.5 * outer(x, x)
        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        
        # Concatenate components: [1, x, diag(x^2)/sqrt(2), off_diag(x^2)]
        return torch.cat([
            torch.ones_like(x[..., 0:1]), 
            x / self.rrd, 
            x2_2 / (self.rd * self.r2), 
            x2_1 / self.rd
        ], dim=-1)

# Kernelbench Parameters
batch_size = 4
seq_len = 512
num_heads = 8
head_dim = 64

def get_inputs():
    x = torch.randn(batch_size, seq_len, num_heads, head_dim)
    return [x]

def get_init_inputs():
    return [head_dim]
