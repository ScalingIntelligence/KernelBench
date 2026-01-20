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
    Reference implementation of Gated Linear Attention (GLA).
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: int = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        use_output_gate: bool = True,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        norm_eps: float = 1e-5,
    ):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.use_short_conv = use_short_conv
        self.use_output_gate = use_output_gate
        self.gate_logit_normalizer = gate_logit_normalizer
        
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim // self.num_kv_groups, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim // self.num_kv_groups, bias=False)
        
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            
        if use_short_conv:
            self.q_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, padding=conv_size-1)
            self.k_conv1d = nn.Conv1d(self.key_dim // self.num_kv_groups, self.key_dim // self.num_kv_groups, conv_size, groups=self.key_dim // self.num_kv_groups, padding=conv_size-1)
            self.v_conv1d = nn.Conv1d(self.value_dim // self.num_kv_groups, self.value_dim // self.num_kv_groups, conv_size, groups=self.value_dim // self.num_kv_groups, padding=conv_size-1)
            
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim // self.num_kv_groups, bias=True)
        )
        
        self.g_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): [B, T, H]
        Returns:
            torch.Tensor: [B, T, H]
        """
        B, T, _ = hidden_states.shape
        H, HK, HV = self.num_heads, self.head_k_dim, self.head_v_dim
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)
        
        if self.use_short_conv:
            # Conv1d expects [B, C, T]
            q = F.silu(self.q_conv1d(q.transpose(1, 2))[..., :T].transpose(1, 2))
            k = F.silu(self.k_conv1d(k.transpose(1, 2))[..., :T].transpose(1, 2))
            v = F.silu(self.v_conv1d(v.transpose(1, 2))[..., :T].transpose(1, 2))
        
        # Reshape and handle GQA
        q = q.view(B, T, H, HK)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=-2).view(B, T, H, HK)
            v = v.repeat_interleave(self.num_kv_groups, dim=-2).view(B, T, H, HV)
            gk = gk.repeat_interleave(self.num_kv_groups, dim=-2).view(B, T, H, HK)
        else:
            k = k.view(B, T, H, HK)
            v = v.view(B, T, H, HV)
            gk = gk.view(B, T, H, HK)
            
        # Gate pre-processing
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        
        # Core Recurrence
        # S_t = S_{t-1} * exp(gk_t) + k_t @ v_t^T
        # o_t = q_t @ S_t
        
        q, k, v, gk = q.float(), k.float(), v.float(), gk.float()
        scale = HK ** -0.5
        
        S = torch.zeros(B, H, HK, HV, device=hidden_states.device, dtype=torch.float32)
        o = torch.zeros(B, T, H, HV, device=hidden_states.device, dtype=torch.float32)
        
        for t in range(T):
            q_t = q[:, t] * scale # [B, H, HK]
            k_t = k[:, t] # [B, H, HK]
            v_t = v[:, t] # [B, H, HV]
            gk_t = gk[:, t].exp() # [B, H, HK]
            
            # S_t = S_{t-1} * decay + outer_product(k, v)
            S = S * gk_t.unsqueeze(-1) + torch.einsum('b h k, b h v -> b h k v', k_t, v_t)
            
            # o_t = q_t @ S_t
            o[:, t] = torch.einsum('b h k, b h k v -> b h v', q_t, S)
            
        # Output processing
        o = self.g_norm(o)
        o = o.view(B, T, -1)
        
        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            o = o * F.silu(g) # swish is silu
            
        return self.o_proj(o.to(hidden_states.dtype))

# Kernelbench Parameters
batch_size = 2
seq_len = 128
hidden_size = 512
num_heads = 4

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size]
