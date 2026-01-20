import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Reference implementation for Rotary Positional Embeddings (RoPE).
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super(Model, self).__init__()
        self.dim = dim
        self.base = base
        # Generate inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Rotated Q and K
        """
        B, T, H, D = q.shape
        t = torch.arange(T, device=q.device, dtype=torch.float32)
        # freqs: [T, D/2]
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        
        # Standard RoPE: rotate pairs of (0, D/2), (1, D/2+1), ...
        # Here we use the GPT-NeoX style: 
        #   x1 = x[..., :D/2], x2 = x[..., D/2:]
        #   o = [x1*cos - x2*sin, x1*sin + x2*cos]
        
        cos = freqs.cos().view(1, T, 1, D // 2).to(q.dtype)
        sin = freqs.sin().view(1, T, 1, D // 2).to(q.dtype)
        
        def apply_rotary(x):
            x1 = x[..., :D // 2]
            x2 = x[..., D // 2:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        q_out = apply_rotary(q)
        k_out = apply_rotary(k)
        
        # Kernelbench requires a single output tensor
        return torch.cat([q_out, k_out], dim=-1)

# Kernelbench Parameters
batch_size = 4
seq_len = 2048
num_heads = 32
head_dim = 128

def get_inputs():
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    return [q, k]

def get_init_inputs():
    return [head_dim]
