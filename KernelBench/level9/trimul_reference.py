import torch
import torch.nn as nn
from torch import einsum


class Model(nn.Module):
    """
    Triangle Multiplicative Module (TriMul) - commonly used in protein structure prediction.
    Performs gated projections followed by an einsum contraction over a shared dimension.
    
    Based on: https://github.com/lucidrains/triangle-multiplicative-module
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Performs the triangle multiplicative update.

        Args:
            x: Input tensor of shape (batch_size, seq_len, seq_len, dim).
            mask: Mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, seq_len, dim).
        """
        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        # Core computation: contract over the k dimension
        # out[b, i, j, d] = sum_k left[b, i, k, d] * right[b, j, k, d]
        out = einsum('... i k d, ... j k d -> ... i j d', left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


# Problem dimensions
batch_size = 4
seq_len = 64
dim = 128
hidden_dim = 64


def get_inputs():
    x = torch.randn(batch_size, seq_len, seq_len, dim)
    mask = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.float32)
    return [x, mask]


def get_init_inputs():
    return [dim, hidden_dim]

