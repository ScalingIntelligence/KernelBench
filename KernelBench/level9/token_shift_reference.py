import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation for Token Shift (as used in RWKV).
    """
    def __init__(self, hidden_size: int):
        super(Model, self).__init__()
        # Learnable mixing coefficient
        self.mu = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # x_shifted is x[t-1]
        x_shifted = F.pad(x, (0, 0, 1, -1))
        # Linear interpolation between current and previous token
        mu = torch.sigmoid(self.mu) # Often used with sigmoid to keep in [0, 1]
        return x * (1 - mu) + x_shifted * mu

# Kernelbench Parameters
batch_size = 32
seq_len = 512
hidden_size = 4096

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size]
