import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_x + self.eps) * self.weight

class Model(nn.Module):
    """
    Reference implementation of Generalized Fused Normalization and Gating.
    Supports both LayerNorm and RMSNorm with optional residual connection and SiLU/Sigmoid gating.
    """
    def __init__(self, hidden_size: int, norm_type: str = 'rms', activation: str = 'silu', eps: float = 1e-5):
        super(Model, self).__init__()
        self.norm_type = norm_type
        self.activation = activation
        if norm_type == 'rms':
            self.norm = RMSNorm(hidden_size, eps=eps)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [B, T, H]
            gate (torch.Tensor): Gate tensor [B, T, H]
            residual (torch.Tensor, optional): Residual tensor [B, T, H]
        Returns:
            torch.Tensor: Gated normalized output [B, T, H]
        """
        if residual is not None:
            x = x + residual
        
        x_normed = self.norm(x)
        
        if self.activation == 'silu':
            gated = x_normed * F.silu(gate)
        elif self.activation == 'sigmoid':
            gated = x_normed * torch.sigmoid(gate)
        else:
            gated = x_normed * gate
            
        return gated

# Kernelbench Parameters
batch_size = 8
seq_len = 1024
hidden_size = 2048

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    gate = torch.randn(batch_size, seq_len, hidden_size)
    residual = torch.randn(batch_size, seq_len, hidden_size)
    return [x, gate, residual]

def get_init_inputs():
    return [hidden_size]
