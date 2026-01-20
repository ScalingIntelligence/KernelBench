import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Reference implementation of Parallel Forgetting Attention.
    """
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, v_head_dim: int):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, scale: float = None) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): [batch_size, seq_len, num_kv_heads, head_dim]
            v (torch.Tensor): [batch_size, seq_len, num_kv_heads, v_head_dim]
            g (torch.Tensor): [batch_size, seq_len, num_heads] - log decay factors
            scale (float): optional scale factor
        Returns:
            o (torch.Tensor): [batch_size, seq_len, num_heads, v_head_dim]
        """
        if scale is None:
            scale = q.shape[-1] ** -0.5
            
        B, T, H, D = q.shape
        _, _, HKV, V = v.shape
        G = H // HKV
        
        # Compute cumsum of g for forgetting factors
        g_cumsum = torch.cumsum(g, dim=1) # [B, T, H]
        
        # Reshape/transpose for attention
        q = q.transpose(1, 2) # [B, H, T, D]
        k = k.transpose(1, 2) # [B, HKV, T, D]
        v = v.transpose(1, 2) # [B, HKV, T, V]
        g_cumsum = g_cumsum.transpose(1, 2) # [B, H, T]
        
        # Handle GQA by repeating k and v
        if G > 1:
            k = k.repeat_interleave(G, dim=1)
            v = v.repeat_interleave(G, dim=1)
            
        # Compute attention scores
        # scores: [B, H, T, T]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        # Add forgetting factors (decay)
        # score[i, j] += cumsum(g)_i - cumsum(g)_j
        attn_scores += g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Output
        o = torch.matmul(attn_weights, v) # [B, H, T, V]
        
        return o.transpose(1, 2)

# Kernelbench Parameters
batch_size = 2
seq_len = 128
num_heads = 8
num_kv_heads = 4
head_dim = 64
v_head_dim = 64

def get_inputs():
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_kv_heads, v_head_dim)
    g = torch.randn(batch_size, seq_len, num_heads)
    return [q, k, v, g]

def get_init_inputs():
    return [num_heads, num_kv_heads, head_dim, v_head_dim]
