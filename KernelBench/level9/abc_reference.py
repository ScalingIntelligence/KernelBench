
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Simplified Modules ---

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)

    def forward(self, x, g):
        # In RMSNormGated, it typically normalizes x and then gates with g
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x * F.silu(g)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k):
        # q, k: [B, T, H, D]
        T = q.shape[1]
        device = q.device
        t = torch.arange(T, device=device, dtype=q.dtype)
        freqs = torch.outer(t, self.inv_freq) # [T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1) # [T, D]
        cos = emb.cos()[None, :, None, :] # [1, T, 1, D]
        sin = emb.sin()[None, :, None, :] # [1, T, 1, D]
        
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k

class ShortConvolution(nn.Module):
    def __init__(self, hidden_size, kernel_size, bias=False, activation='silu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=0,
            bias=bias
        )
        self.activation = activation

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        # Causal padding
        x_padded = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0)) # [B, D, T + K-1]
        x = self.conv(x_padded) # [B, D, T]
        if self.activation == 'silu':
            x = F.silu(x)
        return x.transpose(1, 2)

# --- Core Attention Mechanism ---

def chunk_abc_ref(q, k, v, s):
    # Functional reference implementation of ABC attention
    # q, k, v, s: [B, T, H, D] (or D_v for v, M for s)
    
    B, T, H, K = q.shape
    V = v.shape[-1]
    M = s.shape[-1]
    
    # Transpose to [B, H, T, ...]
    q = q.transpose(1, 2).float()
    k = k.transpose(1, 2).float()
    v = v.transpose(1, 2).float()
    s = s.transpose(1, 2).float()
    
    scale = K ** -0.5
    
    # 1. Compute gating and normalization tokens
    # Using logcumsumexp for numerical stability
    z = s.logcumsumexp(2)
    # g factor for the state recurrence: exp(z_{i-1} - z_i)
    z_prev = torch.cat((torch.zeros_like(z[:, :, :1]), z[:, :, :-1]), 2)
    g = (z_prev - z).exp()
    # Normalized slot weights: exp(s - z)
    s_norm = (s - z).exp()
    
    # 2. Sequential Key-Slot update (hk state)
    # hk: [B, H, K, M]
    hk = torch.zeros(B, H, K, M, device=q.device, dtype=torch.float32)
    ok = torch.zeros(B, H, T, M, device=q.device, dtype=torch.float32)
    
    for i in range(T):
        qi = q[:, :, i] * scale
        ki = k[:, :, i]
        si = s_norm[:, :, i]
        gi = g[:, :, i]
        
        # State update: hk = hk * g + k^T * s_norm
        hk = hk * gi[..., None, :] + ki[..., None] * si[..., None, :]
        # Output: query * hk
        ok[:, :, i] = (qi[..., None] * hk).sum(-2)
        
    # 3. Sequential Slot-Value update (hv state)
    # Interaction between slots and values based on ok
    qv = ok.softmax(-1)
    
    hv = torch.zeros(B, H, M, V, device=q.device, dtype=torch.float32)
    ov = torch.zeros(B, H, T, V, device=q.device, dtype=torch.float32)
    
    for i in range(T):
        qvi = qv[:, :, i]
        ki = s_norm[:, :, i]
        vi = v[:, :, i]
        gi = g[:, :, i]
        
        # State update: hv = hv * g + s_norm^T * v
        hv = hv * gi[..., :, None] + ki[..., None] * vi[..., None, :]
        # Output: qv * hv
        ov[:, :, i] = (qvi[..., None] * hv).sum(-2)
        
    # Transpose back to [B, T, H, V]
    return ov.transpose(1, 2).to(q.dtype)

# --- Main Model ---

class Model(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        num_slots: int = 64,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_rope: bool = True,
        use_output_gate: bool = True,
        use_norm: bool = True,
        norm_eps: float = 1e-5,
    ):
        super(Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.s_proj = nn.Linear(hidden_size, num_heads * num_slots, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size)
            
        self.use_rope = use_rope
        if use_rope:
            self.rotary = RotaryEmbedding(self.head_k_dim)
            
        self.use_norm = use_norm
        if use_norm:
            if use_output_gate:
                self.g_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
            else:
                self.g_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 2. Short Convolution
        if self.use_short_conv:
            q = self.q_conv1d(q)
            k = self.k_conv1d(k)
            v = self.v_conv1d(v)
            
        # 3. Reshape and RoPE
        # [B, T, D] -> [B, T, H, d]
        b, t, d = hidden_states.shape
        q = q.view(b, t, self.num_heads, self.head_k_dim)
        k = k.view(b, t, self.num_heads, self.head_k_dim)
        v = v.view(b, t, self.num_heads, self.head_v_dim)
        
        if self.use_rope:
            q, k = self.rotary(q, k)
            
        # 4. Slot projection
        s = self.s_proj(hidden_states).view(b, t, self.num_heads, self.num_slots)
        # Numerical stability clamp as per original
        s = s.clamp(-32, 32)
        
        # 5. Core ABC Attention
        o = chunk_abc_ref(q, k, v, s)
        
        # 6. Gating and Norm
        if self.use_norm:
            if hasattr(self, 'g_proj'):
                g = self.g_proj(hidden_states).view(b, t, self.num_heads, self.head_v_dim)
                o = self.g_norm(o, g)
            else:
                o = self.g_norm(o)
        elif hasattr(self, 'g_proj'):
            g = self.g_proj(hidden_states).view(b, t, self.num_heads, self.head_v_dim)
            o = o * F.silu(g)
            
        # 7. Final Output Projection
        o = o.reshape(b, t, self.value_dim)
        return self.o_proj(o)

# --- Kernelbench API ---

def get_inputs():
    # Batch size 8, Sequence length 128 (reference is slow), Hidden size 1024
    hidden_states = torch.randn(8, 128, 1024)
    return [hidden_states]

def get_init_inputs():
    # [hidden_size, num_heads, num_slots]
    return [1024, 4, 64]
