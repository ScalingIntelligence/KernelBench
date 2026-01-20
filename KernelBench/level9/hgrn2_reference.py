import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # standard rms norm implementation
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class Model(nn.Module):
    """
    HGRN2 (Gated Linear RNN with State Expansion) Reference Implementation.
    This model implements the core linear attention mechanism of HGRN2.
    """
    def __init__(self, hidden_size: int = 2048, num_heads: int = 16, expand_ratio: int = 128):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = expand_ratio # State expansion dimension (head_f_dim in FLA)
        self.v_head_dim = hidden_size // num_heads # Value dimension per head (head_i_dim in FLA)
        
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.f_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.i_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.g_norm = RMSNorm(hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the HGRN2 linear attention forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_proj(x)
        f = self.f_proj(x)
        i = self.i_proj(x)
        
        # Apply HGRN2 gating logic
        q = swish(q)
        g = F.logsigmoid(f)
        k = 1 - g.exp()
        
        # Rearrange tensors for multi-head attention
        # Shape: [batch, heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, heads, seq_len, v_head_dim]
        v = i.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        
        # Naive recurrent Linear Attention (GLA) logic
        # Implementation of: h_t = exp(g_t) * h_{t-1} + k_t @ v_t^T
        #                   o_t = q_t @ h_t
        
        # Use float32 for stable recurrence
        dtype = q.dtype
        q, k, v, g = q.float(), k.float(), v.float(), g.float()
        
        scale = self.head_dim ** -0.5
        # state: [batch, heads, head_dim, v_head_dim]
        h = torch.zeros(batch_size, self.num_heads, self.head_dim, self.v_head_dim, device=x.device, dtype=torch.float32)
        o = torch.zeros(batch_size, self.num_heads, seq_len, self.v_head_dim, device=x.device, dtype=torch.float32)
        
        for t in range(seq_len):
            q_t = q[:, :, t] * scale
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            g_t = g[:, :, t].exp()
            
            # Update RNN state
            # k_t: [batch, heads, head_dim], v_t: [batch, heads, v_head_dim] -> kv_t: [batch, heads, head_dim, v_head_dim]
            kv_t = torch.einsum('b h d, b h v -> b h d v', k_t, v_t)
            h = h * g_t.unsqueeze(-1) + kv_t
            
            # Output
            o[:, :, t] = torch.einsum('b h d, b h d v -> b h v', q_t, h)
            
        o = o.to(dtype)
        
        # Final projection and normalization
        # [batch, heads, seq_len, v_head_dim] -> [batch, seq_len, hidden_size]
        res = o.transpose(1, 2).reshape(batch_size, seq_len, -1)
        res = self.g_norm(res)
        res = self.o_proj(res)
        
        return res

# Dimensions for testing
batch_size = 2
seq_len = 1024
hidden_size = 2048
num_heads = 16
expand_ratio = 128

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size, num_heads, expand_ratio]
