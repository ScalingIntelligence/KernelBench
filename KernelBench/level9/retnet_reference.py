import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb: [T, D]
        # x: [B, H, T, D]
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
            
        return x * cos + rotate_half(x) * sin

class Model(nn.Module):
    """
    Reference implementation of RetNet (MultiScale Retention).
    """
    def __init__(self, hidden_size: int = 1024, num_heads: int = 8, head_dim: int = 64, v_head_dim: int = 128):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * v_head_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_heads * v_head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
        
        self.rotary = RotaryEmbedding(head_dim)
        self.norm = nn.GroupNorm(num_heads, num_heads * v_head_dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, hidden_size]
        Returns:
            torch.Tensor: [batch_size, seq_len, hidden_size]
        """
        B, T, _ = x.shape
        H, D, V = self.num_heads, self.head_dim, self.v_head_dim
        
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2) # [B, H, T, D]
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2) # [B, H, T, D]
        v = self.v_proj(x).view(B, T, H, V).transpose(1, 2) # [B, H, T, V]
        g = self.g_proj(x) # [B, T, H*V]
        
        # Apply Rotary Position Embeddings
        q = self.rotary(q, T)
        k = self.rotary(k, T)
        
        # Parallel Retention computation
        q, k, v = q.float(), k.float(), v.float()
        
        # Head-specific decay rates: gamma = 1 - 2^(-5-h)
        gamma = 1.0 - torch.pow(2.0, -5.0 - torch.arange(H, device=x.device, dtype=torch.float32))
        s = gamma.log2()
        
        # Decay matrix: D[i, j] = gamma^(i-j) for i >= j else 0
        n_idx = torch.arange(T, device=x.device, dtype=torch.float32)
        decay = torch.exp2((n_idx.unsqueeze(-1) - n_idx) * s.view(-1, 1, 1))
        mask = n_idx.unsqueeze(-1) >= n_idx
        decay = decay.masked_fill(~mask, 0)
        
        # Scaled dot-product attention with decay
        # scores: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-1, -2)) * (D ** -0.5)
        scores = scores * decay
        
        # Aggregate values
        o = torch.matmul(scores, v) # [B, H, T, V]
        
        # GroupNorm followed by gating
        o = o.transpose(1, 2).reshape(B * T, H * V)
        o = self.norm(o).view(B, T, H * V)
        o = o * F.silu(g) # Multi-scale gating
        
        return self.o_proj(o.to(x.dtype))

# Kernelbench Parameters
batch_size = 2
seq_len = 128
hidden_size = 1024
num_heads = 8
head_dim = 64
v_head_dim = 128

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size, num_heads, head_dim, v_head_dim]
