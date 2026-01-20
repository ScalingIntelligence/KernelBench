import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation for Layer/RMS Normalization with Residual Connection.
    Commonly used in Transformers (Prenorm or Postnorm).
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5, is_rms_norm: bool = True, prenorm: bool = False):
        super(Model, self).__init__()
        self.is_rms_norm = is_rms_norm
        self.prenorm = prenorm
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if not is_rms_norm else None

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [B, T, H]
            residual (torch.Tensor, optional): Residual connection [B, T, H]
        Returns:
            torch.Tensor: Normalized output (or tuple of output and residual if prenorm=True)
                          For Kernelbench, we return just the final output tensor.
        """
        if residual is not None:
            if self.prenorm:
                # In prenorm, x is actually the hidden state that expects residual + some_sublayer(norm(x))
                # But typically prenorm modules in this repo add residual first.
                x_with_res = x + residual
            else:
                x_with_res = x + residual
        else:
            x_with_res = x
            
        if self.is_rms_norm:
            # RMSNorm
            rstd = torch.rsqrt(x_with_res.pow(2).mean(-1, keepdim=True) + self.eps)
            out = x_with_res * rstd * self.weight
            if self.bias is not None:
                out = out + self.bias
        else:
            # LayerNorm
            out = F.layer_norm(x_with_res, x_with_res.shape[-1:], self.weight, self.bias, self.eps)
            
        return out

# Kernelbench Parameters
batch_size = 8
seq_len = 1024
hidden_size = 2048

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    residual = torch.randn(batch_size, seq_len, hidden_size)
    return [x, residual]

def get_init_inputs():
    return [hidden_size]
