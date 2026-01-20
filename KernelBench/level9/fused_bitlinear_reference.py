import torch
import torch.nn as nn
import torch.nn.functional as F

def activation_quant(x):
    """Per-token quantization to 8 bits."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w):
    """Per-tensor quantization to 1.58 bits."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return x_normed * self.weight

class Model(nn.Module):
    """
    Reference implementation of BitLinear (1.58-bit Linear Layer).
    Includes RMSNorm and Straight-Through Estimator (STE) for quantization.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(Model, self).__init__()
        self.norm = RMSNorm(in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, in_features]
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, out_features]
        """
        # 1. Apply RMS normalization
        x_norm = self.norm(x)
        
        # 2. Quantize activations (with STE)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        
        # 3. Quantize weights (with STE)
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
        
        # 4. Linear operation
        return F.linear(x_quant, w_quant, self.bias)

# Kernelbench Parameters
batch_size = 8
seq_len = 512
in_features = 1024
out_features = 2048

def get_inputs():
    x = torch.randn(batch_size, seq_len, in_features)
    return [x]

def get_init_inputs():
    return [in_features, out_features]
