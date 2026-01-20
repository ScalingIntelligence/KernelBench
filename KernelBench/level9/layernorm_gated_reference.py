import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation for Gated Layer/RMS Normalization.
    Supports gated normalization where the gate 'z' is applied either before or after the normalization.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5, is_rms_norm: bool = True, norm_before_gate: bool = True):
        super(Model, self).__init__()
        self.is_rms_norm = is_rms_norm
        self.norm_before_gate = norm_before_gate
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if not is_rms_norm else None

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [B, T, H]
            z (torch.Tensor): Gate tensor [B, T, H]
        Returns:
            torch.Tensor: Gated normalized output [B, T, H]
        """
        if not self.norm_before_gate:
            x = x * F.silu(z)
            
        if self.is_rms_norm:
            # RMSNorm
            rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            x = x * rstd * self.weight
        else:
            # LayerNorm
            x = F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)
            
        if self.norm_before_gate:
            x = x * F.silu(z)
            
        return x

# Kernelbench Parameters
batch_size = 4
seq_len = 2048
hidden_size = 4096

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    z = torch.randn(batch_size, seq_len, hidden_size)
    return [x, z]

def get_init_inputs():
    return [hidden_size]
