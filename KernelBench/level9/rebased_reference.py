import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def flatten_diag_outer_product_off1(x, y):
    z = torch.einsum("...i,...j->...ij", x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N, 1)
    indices2 = torch.arange(0, N)
    return z[..., indicies[0], indicies[1]], z[..., indices2, indices2]

class RebasedFeatureMap(nn.Module):
    def __init__(self, head_dim: int, use_gamma: bool = True, use_beta: bool = True, normalize: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.normalize = normalize
        if use_gamma:
            self.gamma = nn.Parameter(torch.ones(head_dim))
        else:
            self.gamma = None
        if use_beta:
            self.beta = nn.Parameter(torch.zeros(head_dim))
        else:
            self.beta = None

    def forward(self, x: torch.Tensor):
        if self.normalize:
            x = F.layer_norm(x, (self.head_dim,), self.gamma, self.beta)
        elif self.use_gamma and self.use_beta:
            x = x * self.gamma + self.beta
        elif self.use_gamma:
            x = x * self.gamma
            
        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        # rebased use learnable parameters to approximate any quadratic function
        return torch.cat([x2_2 * self.head_dim ** -0.5, x2_1 * (2 / self.head_dim) ** 0.5], dim=-1)

class Model(nn.Module):
    """
    Reference implementation of ReBased Linear Attention.
    """
    def __init__(self, hidden_size: int, feature_dim: int = 16, num_heads: int = 16, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = eps
        
        self.feature_map = RebasedFeatureMap(feature_dim)
        self.q_proj = nn.Linear(hidden_size, feature_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, feature_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        b, t, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.feature_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.num_heads, self.feature_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # Linear attention
        q, k = self.feature_map(q), self.feature_map(k)
        
        # q, k: [b, h, t, m]
        # v: [b, h, t, d]
        
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        # q: [b, h, t, 1, m]
        # k: [b, h, t, 1, m]
        # v: [b, h, t, d, 1]

        # Compute attention: (q * (k * v).cumsum) / (q * k.cumsum)
        # (k * v) is [b, h, t, d, m]
        kv_cum = (k * v).cumsum(2)
        num = (q * kv_cum).sum(-1) # [b, h, t, d]
        
        k_cum = k.cumsum(2)
        den = (q * k_cum).sum(-1) + self.eps # [b, h, t, 1]
        
        y = num / den
        y = y.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(y.to(hidden_states.dtype))

# Kernelbench Parameters
batch_size = 2
seq_len = 128
hidden_size = 1024
feature_dim = 16
num_heads = 16

def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    return [hidden_states]

def get_init_inputs():
    return [hidden_size, feature_dim, num_heads]
