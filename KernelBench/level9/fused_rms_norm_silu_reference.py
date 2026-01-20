import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation for Fused RMSNorm + SiLU Gating.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            gate (torch.Tensor): Gate tensor of shape [batch_size, seq_len, hidden_size]
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # RMSNorm
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        x_normed = x_normed * self.weight
        
        # SiLU Gating
        return x_normed * F.silu(gate)

# Kernelbench Parameters
batch_size = 8
seq_len = 1024
hidden_size = 2048

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    gate = torch.randn(batch_size, seq_len, hidden_size)
    return [x, gate]

def get_init_inputs():
    return [hidden_size]
