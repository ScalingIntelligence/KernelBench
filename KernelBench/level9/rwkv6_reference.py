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

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, weight=True, bias=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        if weight:
            self.weight = nn.Parameter(torch.ones(num_channels))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        x = x.view(B, T, self.num_groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        x = x.view(B, T, C)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

class LoRA(nn.Module):
    def __init__(self, input_dim, output_dim, low_rank_dim, bias=True):
        super().__init__()
        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            nn.Tanh(),
            nn.Linear(low_rank_dim, output_dim, bias=bias),
        )
        # Initialization
        nn.init.zeros_(self.lora[0].weight)
        nn.init.orthogonal_(self.lora[2].weight, gain=0.1)
        if bias:
            nn.init.zeros_(self.lora[2].bias)

    def forward(self, x):
        return self.lora(x)

class LerpLinear(nn.Module):
    def __init__(self, input_dim, output_dim, low_rank_dim=None):
        super().__init__()
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim)
        self.mu = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, delta):
        return self.linear(x + delta * self.mu)

class DDLerpLinear(nn.Module):
    def __init__(self, input_dim, output_dim, low_rank_dim=None):
        super().__init__()
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim)

    def forward(self, x, mu, delta):
        return self.linear(x + delta * mu)

class Model(nn.Module):
    """
    Reference implementation of RWKV-6 Attention.
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
    ):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        
        # Time-mix projections
        self.x_proj_lora = LoRA(hidden_size, proj_low_rank_dim * 5, proj_low_rank_dim)
        self.x_proj_mu = nn.Parameter(torch.zeros(hidden_size))
        self.x_proj_out = nn.Linear(proj_low_rank_dim * 5, hidden_size, bias=False)
        self.x_bias = nn.Parameter(torch.zeros(5, hidden_size))
        
        self.r_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.w_proj = DDLerpLinear(hidden_size, self.key_dim, low_rank_dim=gate_low_rank_dim)
        self.k_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.v_proj = DDLerpLinear(hidden_size, self.value_dim)
        self.g_proj = DDLerpLinear(hidden_size, self.value_dim)
        
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_k_dim))
        
        self.g_norm = GroupNorm(num_heads, self.value_dim)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): [B, T, H]
        Returns:
            torch.Tensor: [B, T, H]
        """
        B, T, H = hidden_states.shape
        
        # 1. Time Shift
        shifted = F.pad(hidden_states, (0, 0, 1, -1))
        # delta is (x_{t-1} - x_t) in FLA's RWKV6 implementation
        delta = shifted - hidden_states
        
        # 2. Extract r, w, k, v, g parameters
        # x_proj logic
        x_lerp = hidden_states + delta * self.x_proj_mu
        x_lora = self.x_proj_lora(x_lerp) # [B, T, 5 * R]
        # x_proj[1] is Tanh, x_proj[2] is Linear
        x_lora = torch.tanh(x_lora)
        x_params = self.x_proj_out(x_lora) # [B, T, H]
        # Reshape and add bias for the 5 parameters
        # In FLA: x = x_proj[2](x_proj[1](x_proj[0](h, delta)))
        # Here x_params has shape [B, T, H]. We need to expand it to 5 parameters or use the specific head logic.
        # Actually, in RWKV-6, x_proj[0] projects to 5*R, then Tanh, then x_proj[2] projects to H.
        # Wait, the FLA code does:
        # x = self.x_proj[0](hidden_states, delta).view(B, T, -1, self.proj_low_rank_dim)
        # x = torch.einsum('b t n r, h n r-> b t n h', self.x_proj[1](x), self.x_proj[2].weight.view(hidden_size, 5, -1))
        # r, w, k, v, g = x.add_(self.x_bias).unbind(-2)
        
        # Let's re-implement that exactly:
        # projected: [B, T, 5, R]
        x_0 = self.x_proj_lora(x_lerp).view(B, T, 5, self.proj_low_rank_dim)
        x_1 = torch.tanh(x_0)
        # x_proj_out weight: [H, 5 * R] -> [H, 5, R]
        # FLA uses x_proj[2].weight which is [hidden_size, 5*proj_low_rank_dim]
        # In our case x_proj_out is nn.Linear(5*R, H)
        x_2 = torch.einsum('b t n r, h n r -> b t n h', x_1, self.x_proj_out.weight.view(H, 5, -1))
        # Adding bias and unbinding
        r_mu, w_mu, k_mu, v_mu, g_mu = (x_2 + self.x_bias).unbind(-2)
        
        r = self.r_proj(hidden_states, r_mu, delta)
        w = self.w_proj(hidden_states, w_mu, delta)
        k = self.k_proj(hidden_states, k_mu, delta)
        v = self.v_proj(hidden_states, v_mu, delta)
        g = self.g_proj(hidden_states, g_mu, delta)
        
        # 3. Recurrence (RWKV-6)
        # r, w, k, v are [B, T, D]
        # Reshape to heads
        NH = self.num_heads
        DK = self.head_k_dim
        DV = self.head_v_dim
        
        r = r.view(B, T, NH, DK)
        k = k.view(B, T, NH, DK)
        v = v.view(B, T, NH, DV)
        w = -torch.exp(w.view(B, T, NH, DK)) # Time-decay is negative exp
        u = self.bonus # [NH, DK]
        
        # Functional recurrence
        # S_t = S_{t-1} * exp(w_t) + k_t @ v_t^T
        # o_t = r_t @ (S_{t-1} + u * k_t @ v_t^T)
        
        S = torch.zeros(B, NH, DK, DV, device=hidden_states.device, dtype=torch.float32)
        o = torch.zeros(B, T, NH, DV, device=hidden_states.device, dtype=torch.float32)
        
        r, k, v, w = r.float(), k.float(), v.float(), w.float()
        
        for t in range(T):
            r_t = r[:, t] # [B, NH, DK]
            k_t = k[:, t] # [B, NH, DK]
            v_t = v[:, t] # [B, NH, DV]
            w_t = w[:, t].exp() # [B, NH, DK]
            
            # kv = k_t^T @ v_t
            kv = torch.einsum('b h k, b h v -> b h k v', k_t, v_t)
            
            # Output computation
            # o = r @ (S + u * kv)
            # u is [NH, DK].
            
            # S is [DK, DV]. r is [DK]. o is [DV].
            rS = torch.einsum('b h k, b h k v -> b h v', r_t, S)
            # r @ (u * kv) -> (r_t * u * k_t) @ v_t
            ruk = r_t * u.view(1, NH, DK) * k_t
            rukv = torch.einsum('b h k, b h v -> b h v', ruk, v_t)
            
            o[:, t] = rS + rukv
            
            # Update state
            # S = S * w + kv
            S = S * w_t.unsqueeze(-1) + kv
            
        # 4. Final output
        o = o.view(B, T, -1)
        o = self.g_norm(o) * F.silu(g)
        return self.o_proj(o)

# Kernelbench Parameters
batch_size = 2
seq_len = 128
hidden_size = 512
num_heads = 4
expand_k = 0.5
expand_v = 1.0

def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    return [hidden_states]

def get_init_inputs():
    return [hidden_size, expand_k, expand_v, num_heads]
