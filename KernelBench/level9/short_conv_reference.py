import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation for Short Causal 1D Convolution.
    """
    def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        B, T, D = x.shape
        # Conv1d expects [B, C, T]
        x = x.transpose(1, 2)
        # Apply convolution and trim the end to maintain causality
        x = self.conv(x)[..., :T]
        # Transpose back
        return x.transpose(1, 2)

# Kernelbench Parameters
batch_size = 16
seq_len = 512
hidden_size = 1024
kernel_size = 4

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size, kernel_size]
