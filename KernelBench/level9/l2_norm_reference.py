import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation for L2 Normalization.
    Normalizes the input tensor by its L2 norm along the last dimension.
    """
    def __init__(self, eps: float = 1e-6):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
        Returns:
            torch.Tensor: L2-normalized tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Compute L2 norm along the last dimension
        # norm = sqrt(sum(x^2))
        return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + self.eps)

# Kernelbench Parameters
batch_size = 16
seq_len = 512
hidden_size = 1024

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [1e-6] # eps
