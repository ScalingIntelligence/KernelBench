import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Inlined helper functions from fla/ops/log_linear_attn/naive.py
def segsum(x):
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool))
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def construct_level_mask(level, L):
    T = L.size(-1)
    if level == 0:
        return torch.diag_embed(L[..., level, :])
    indices = torch.cartesian_prod(torch.arange(T), torch.arange(T)).to(L.device)
    mask = torch.where(
        torch.logical_and(
            torch.logical_and(
                indices[:, 0] % (1 << level) >= (1 << (level - 1)),
                indices[:, 1] + (1 << (level - 1))
                >= indices[:, 0] - (indices[:, 0] % (1 << (level - 1))),
            ),
            indices[:, 1] < indices[:, 0] - (indices[:, 0] % (1 << (level - 1))),
        ).view(T, T),
        L[..., level, :].unsqueeze(-1).expand(*([-1] * (len(L.shape) - 2)), T, T),
        0,
    ).to(L.dtype)
    return mask

def construct_H_matrix(a, L):
    T = a.size(-1)
    A = torch.exp(segsum(a))
    H = torch.zeros_like(A, dtype=a.dtype)
    for level in range(math.ceil(math.log2(T)) + 1):
        mask = construct_level_mask(level, L)
        H += A * mask
    return H

class Model(nn.Module):
    """
    Naive implementation of Log Linear Attention.
    """
    def __init__(self, n_heads: int = 4, seq_len: int = 64):
        super(Model, self).__init__()
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.n_levels = int(math.ceil(math.log2(seq_len))) + 1

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, level_scales: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: [batch_size, seq_len, n_heads, head_dim]
            k: [batch_size, seq_len, n_heads, head_dim]
            v: [batch_size, seq_len, n_heads, head_dim]
            g: [batch_size, seq_len, n_heads]
            level_scales: [batch_size, n_heads, n_levels, seq_len]
        Returns:
            o: [batch_size, seq_len, n_heads, head_dim]
        """
        # H calculation requires [batch_size, n_heads, seq_len] for g 
        # and [batch_size, n_heads, n_levels, seq_len] for level_scales
        # q, k, v are [batch_size, seq_len, n_heads, d]
        
        # g: [b, t, h] -> [b, h, t]
        g_transposed = g.transpose(1, 2)
        
        # level_scales is already [b, h, n_levels, t]
        
        # Compute H matrix [batch_size, n_heads, seq_len, seq_len]
        H = construct_H_matrix(g_transposed, level_scales)
        
        # Attention computation
        # H: [b, h, l, c] (batch, head, seq_len_q, seq_len_k)
        # q: [b, l, h, n] (batch, seq_len_q, head, head_dim)
        # k: [b, c, h, n] (batch, seq_len_k, head, head_dim)
        # v: [b, c, h, p] (batch, seq_len_k, head, head_dim)
        
        # M = H * (q @ k.T) -> but in log-linear it's specialized
        # Based on naive implementation:
        # M = torch.einsum("bhlc,blhn,bchn->bhlc", H, q, k)
        M = torch.einsum("bhlc,blhn,bchn->bhlc", H, q, k)
        # o = torch.einsum("bhlc,bchp->blhp", M, v)
        o = torch.einsum("bhlc,bchp->blhp", M, v)
        
        return o

# Kernelbench Parameters
batch_size = 2
seq_len = 64
n_heads = 4
head_dim = 32
n_levels = int(math.ceil(math.log2(seq_len))) + 1

def get_inputs():
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)
    v = torch.randn(batch_size, seq_len, n_heads, head_dim)
    g = torch.randn(batch_size, seq_len, n_heads)
    level_scales = torch.randn(batch_size, n_heads, n_levels, seq_len).abs()
    return [q, k, v, g, level_scales]

def get_init_inputs():
    return [n_heads, seq_len]
