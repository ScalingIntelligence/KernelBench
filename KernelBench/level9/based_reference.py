import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Inlined helper from fla/modules/feature_map.py
def flatten_diag_outer_product_off1(x, y):
    z = torch.einsum("...i,...j->...ij", x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N, 1)
    indices2 = torch.arange(0, N)
    return z[..., indicies[0], indicies[1]], z[..., indices2, indices2]

class TaylorFeatureMap(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(self.head_dim)
        self.rrd = math.sqrt(self.rd)

    def forward(self, x: torch.Tensor):
        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        return torch.cat([torch.ones_like(x[..., 0:1]), x / self.rrd, x2_2 / (self.rd * self.r2), x2_1 / self.rd], dim=-1)

class Model(nn.Module):
    """
    Reference implementation of Based Linear Attention.
    """
    def __init__(
        self, 
        hidden_size: int = 1024, 
        feature_dim: int = 16, 
        num_heads: int = 16, 
        causal: bool = True, 
        eps: float = 1e-12
    ):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.eps = eps

        self.q_proj = nn.Linear(hidden_size, feature_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, feature_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.feature_map = TaylorFeatureMap(feature_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        b, t, h = hidden_states.size()
        
        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to heads
        q = q.view(b, t, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Taylor feature map
        # q, k: [b, h, t, d_feature] -> [b, h, t, d_expanded]
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # Linear attention computation
        # q, k: [b, h, t, d_expanded]
        # v: [b, h, t, d_head]
        
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        # q: [b, h, t, 1, d_expanded]
        # k: [b, h, t, 1, d_expanded]
        # v: [b, h, t, d_head, 1]

        if self.causal:
            # kv: [b, h, t, d_head, d_expanded]
            kv = (k * v).cumsum(2)
            y = (q * kv).sum(-1) # [b, h, t, d_head]
            z = k.cumsum(2)
            denom = (q * z).sum(-1) + self.eps
            y = y / denom
        else:
            kv = (k * v).sum(2, keepdim=True)
            y = (q * kv).sum(-1)
            z = k.sum(2, keepdim=True)
            denom = (q * z).sum(-1) + self.eps
            y = y / denom

        # y: [b, h, t, d_head] -> [b, t, h * d_head]
        y = y.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        return self.o_proj(y)

# Kernelbench Parameters
batch_size = 4
seq_len = 1024
hidden_size = 1024
feature_dim = 16
num_heads = 16

def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    return [hidden_states]

def get_init_inputs():
    return [hidden_size, feature_dim, num_heads, True, 1e-12]
