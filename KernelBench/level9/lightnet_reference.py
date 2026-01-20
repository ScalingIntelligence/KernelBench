import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return x_normed * self.weight

class Model(nn.Module):
    """
    Reference implementation of LightNet (YOSO: You Only Scan Once).
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        expand_ratio: int = 128,
        use_short_conv: bool = False,
        conv_size: int = 4,
        gate_low_rank_dim: int = 128,
    ):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.key_dim = num_heads * expand_ratio
        self.value_dim = hidden_size
        self.head_f_dim = expand_ratio
        self.head_i_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, padding=conv_size-1)
            self.k_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, padding=conv_size-1)
            self.v_conv1d = nn.Conv1d(self.value_dim, self.value_dim, conv_size, groups=self.value_dim, padding=conv_size-1)
            
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, hidden_size, bias=False),
        )
        self.g_norm = RMSNorm(hidden_size)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): [B, T, H_in]
        Returns:
            torch.Tensor: [B, T, H_out]
        """
        B, T, _ = hidden_states.shape
        H, KF, DF = self.num_heads, self.key_dim, self.head_f_dim
        DI = self.head_i_dim
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        if self.use_short_conv:
            q = self.q_conv1d(q.transpose(1, 2))[..., :T].transpose(1, 2)
            k = self.k_conv1d(k.transpose(1, 2))[..., :T].transpose(1, 2)
            v = self.v_conv1d(v.transpose(1, 2))[..., :T].transpose(1, 2)
            
        q = F.silu(q).view(B, T, H, DF)
        k = k.view(B, T, H, DF)
        v = v.view(B, T, H, DI)
        
        # YOSO Gate and Normalization
        # z = logcumsumexp(k)
        # k_new = exp(k - z)
        # g = z_{t-1} - z_t
        
        z = torch.logcumsumexp(k.float(), dim=1) # [B, T, H, DF]
        k_new = torch.exp(k.float() - z).to(k.dtype)
        
        # g = shift(z) - z
        z_shifted = torch.cat([z[:, :1], z[:, :-1]], dim=1)
        gk = (z_shifted - z).to(k.dtype)
        
        # Recurrence
        # S_t = S_{t-1} * exp(gk_t) + k_new_t * v_t^T
        # o_t = q_t * S_t
        
        scale = DF ** -0.5
        q, k_new, v, gk = q.float(), k_new.float(), v.float(), gk.float()
        
        S = torch.zeros(B, H, DF, DI, device=hidden_states.device, dtype=torch.float32)
        o = torch.zeros(B, T, H, DI, device=hidden_states.device, dtype=torch.float32)
        
        for t in range(T):
            q_t = q[:, t] * scale # [B, H, DF]
            k_t = k_new[:, t] # [B, H, DF]
            v_t = v[:, t] # [B, H, DI]
            gk_t = gk[:, t].exp() # [B, H, DF]
            
            # S_t = S_{t-1} * gate + outer(k, v)
            # gk_t is per-dimension of DF
            S = S * gk_t.unsqueeze(-1) + torch.einsum('b h f, b h i -> b h f i', k_t, v_t)
            
            # o_t = q_t @ S_t
            o[:, t] = torch.einsum('b h f, b h f i -> b h i', q_t, S)
            
        o = o.view(B, T, -1)
        
        # Output Gating and Norm
        gate = self.g_proj(hidden_states)
        o = self.g_norm(o) * F.silu(gate)
        o = self.o_proj(o)
        
        return o

# Kernelbench Parameters
batch_size = 2
seq_len = 128
hidden_size = 512
num_heads = 4
expand_ratio = 128

def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    return [hidden_states]

def get_init_inputs():
    return [hidden_size, num_heads, expand_ratio]
