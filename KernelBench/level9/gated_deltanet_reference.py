import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ShortConvolution(nn.Module):
    def __init__(self, hidden_size, kernel_size, bias=False, activation='silu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=bias,
        )

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.conv(x)[:, :, :T]  # Causal convolution
        if self.activation == 'silu':
            x = F.silu(x)
        return x.transpose(1, 2)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, g):
        # RMSNorm with gating (often used in DeltaNet and Mamba2)
        # x: [..., D], g: [..., D]
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * self.weight
        return x * F.silu(g)

class Model(nn.Module):
    """
    Reference implementation of Gated DeltaNet.
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2.0,
        head_dim: int = 256,
        num_heads: int = 6,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_heads # Simplified for reference
        
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * head_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        
        # Initialization for 'g' logic
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size)
        
        self.use_gate = use_gate
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = GatedRMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, C = hidden_states.shape
        H = self.num_heads
        K = self.head_dim
        V = self.head_v_dim
        
        if self.use_short_conv:
            q = self.q_conv1d(self.q_proj(hidden_states))
            k = self.k_conv1d(self.k_proj(hidden_states))
            v = self.v_conv1d(self.v_proj(hidden_states))
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))
            
        q = q.view(B, T, H, K)
        k = k.view(B, T, H, K)
        v = v.view(B, T, H, V)
        
        beta = self.b_proj(hidden_states).sigmoid() # [B, T, H]
        g = -self.A_log.exp() * F.softplus(self.a_proj(hidden_states) + self.dt_bias) # [B, T, H]
        
        # Delta Rule Recurrence
        # Normalization
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-6)
        k = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
        
        out = torch.zeros(B, T, H, V, device=hidden_states.device, dtype=hidden_states.dtype)
        state = torch.zeros(B, H, K, V, device=hidden_states.device, dtype=hidden_states.dtype)
        
        for t in range(T):
            q_t = q[:, t] # [B, H, K]
            k_t = k[:, t] # [B, H, K]
            v_t = v[:, t] # [B, H, V]
            beta_t = beta[:, t] # [B, H]
            g_t = g[:, t] # [B, H]
            
            # Decay state
            state = state * torch.exp(g_t).view(B, H, 1, 1)
            
            # Delta update using decayed state
            # H_t = H'_t + beta * (v_t - H'_t @ k_t) @ k_t^T
            # (H @ k_t): [B, H, K, V] @ [B, H, K, 1] -> [B, H, V, 1] -> [B, H, V]
            # Actually state is [K, V], k_t is [K]. So state.T @ k_t is [V]. 
            # In our setup: state is [K, V], k_t is [K]. state^T @ k_t -> [V]
            # Let's use einsum for clarity
            kv = torch.einsum('b h k v, b h k -> b h v', state, k_t)
            
            # dv = beta * (v_t - kv)
            dv = beta_t.view(B, H, 1) * (v_t - kv)
            
            # state = state + k_t @ dv^T
            state = state + torch.einsum('b h k, b h v -> b h k v', k_t, dv)
            
            # out_t = q_t @ state -> [B, H, V]
            out[:, t] = torch.einsum('b h k, b h k v -> b h v', q_t, state)
            
        if self.use_gate:
            gate = self.g_proj(hidden_states).view(B, T, H, V)
            o = self.o_norm(out, gate)
        else:
            o = self.o_norm(out)
            
        o = o.reshape(B, T, -1)
        return self.o_proj(o)

# KernelBench utility functions
def get_inputs():
    B, T, C = 4, 32, 2048 # Keeping sequence length small for reference loop
    hidden_states = torch.randn(B, T, C)
    return [hidden_states]

def get_init_inputs():
    return [2048, 2.0, 256, 6] # hidden_size, expand_v, head_dim, num_heads
