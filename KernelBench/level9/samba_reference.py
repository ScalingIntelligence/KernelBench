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

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k):
        # q, k: [B, T, H, D]
        seq_len = q.shape[1]
        t = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # freqs: [T, D/2]
        cos = freqs.cos().to(q.dtype)
        sin = freqs.sin().to(q.dtype)
        
        def apply_rotary(x, cos, sin):
            # x: [B, T, H, D]
            # cos, sin: [T, D/2]
            d = x.shape[-1]
            x1 = x[..., :d//2]
            x2 = x[..., d//2:]
            # cos[T, D/2] -> [1, T, 1, D/2]
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
            o1 = x1 * cos - x2 * sin
            o2 = x1 * sin + x2 * cos
            return torch.cat([o1, o2], dim=-1)

        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads=None, qkv_bias=False, window_size=None, rope_theta=10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.window_size = window_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.rotary = RotaryEmbedding(self.head_dim, base=rope_theta)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        
        q, k = self.rotary(q, k)
        
        # Transpose for scaled_dot_product_attention: [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaling is handled by scaled_dot_product_attention if scale=None
        # We need causal mask or window mask
        # Samba uses sliding window attention if window_size is set
        attn_mask = None
        if self.window_size is not None:
             # Create custom mask for sliding window
             mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0)
             if self.window_size < T:
                 mask = torch.triu(mask, diagonal=-(self.window_size-1))
             attn_mask = (mask == 0)
        
        # SDPA
        o = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=0.0, 
            is_causal=(self.window_size is None)
        )
        
        o = o.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(o)

class MambaLayer(nn.Module):
    def __init__(self, hidden_size, state_size=16, conv_kernel=4, intermediate_size=None, time_step_rank=None, use_bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.ssm_state_size = state_size
        self.conv_kernel_size = conv_kernel
        self.intermediate_size = intermediate_size if intermediate_size is not None else 2 * hidden_size
        self.time_step_rank = time_step_rank if time_step_rank is not None else math.ceil(hidden_size / 16)
        
        self.in_proj = nn.Linear(hidden_size, self.intermediate_size * 2, bias=use_bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=True,
            kernel_size=conv_kernel,
            groups=self.intermediate_size,
            padding=conv_kernel - 1,
        )
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32).repeat(self.intermediate_size, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, hidden_size, bias=use_bias)

    def forward(self, x):
        B, T, _ = x.shape
        projected = self.in_proj(x).transpose(1, 2) # [B, 2*intermediate, T]
        x_inner, gate = projected.chunk(2, dim=1)
        
        # Conv
        x_inner = self.conv1d(x_inner)[..., :T]
        x_inner = F.silu(x_inner)
        
        # SSM parameters
        ssm_params = self.x_proj(x_inner.transpose(1, 2)) # [B, T, rank + 2*state]
        dt, B_ssm, C_ssm = torch.split(ssm_params, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)).transpose(1, 2) # [B, intermediate, T]
        A = -torch.exp(self.A_log.float()) # [intermediate, state]
        
        # Scan (reference implementation)
        # s_t = dA * s_{t-1} + dB * x_t
        # dA = exp(dt * A)
        # dB = dt * B
        # y_t = C * s_t
        
        results = []
        ssm_state = torch.zeros(B, self.intermediate_size, self.ssm_state_size, device=x.device, dtype=x.dtype)
        
        for i in range(T):
            dt_i = dt[:, :, i].unsqueeze(-1) # [B, intermediate, 1]
            B_i = B_ssm[:, i, :].unsqueeze(1) # [B, 1, state]
            C_i = C_ssm[:, i, :].unsqueeze(-1) # [B, state, 1]
            x_i = x_inner[:, :, i].unsqueeze(-1) # [B, intermediate, 1]
            
            dA = torch.exp(A.unsqueeze(0) * dt_i) # [B, intermediate, state]
            dB = dt_i * B_i # [B, intermediate, state]
            
            ssm_state = dA * ssm_state + dB * x_i
            y_i = torch.matmul(ssm_state, C_i).squeeze(-1) # [B, intermediate]
            results.append(y_i)
        
        y = torch.stack(results, dim=2) # [B, intermediate, T]
        y = y + x_inner * self.D.view(1, -1, 1)
        y = y * F.silu(gate)
        
        return self.out_proj(y.transpose(1, 2))

class GatedMLP(nn.Module):
    def __init__(self, hidden_size, hidden_ratio=4):
        super().__init__()
        intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
        intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class SambaBlock(nn.Module):
    def __init__(self, hidden_size, layer_idx, config_attn):
        super().__init__()
        self.mixer_norm = RMSNorm(hidden_size)
        if config_attn is not None and layer_idx in config_attn['layers']:
            self.mixer = Attention(
                hidden_size=hidden_size,
                num_heads=config_attn['num_heads'],
                num_kv_heads=config_attn['num_kv_heads'],
                qkv_bias=config_attn['qkv_bias'],
                window_size=config_attn['window_size'],
                rope_theta=config_attn['rope_theta']
            )
        else:
            self.mixer = MambaLayer(hidden_size=hidden_size)
        
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = GatedMLP(hidden_size=hidden_size)

    def forward(self, x):
        residual = x
        x = self.mixer_norm(x)
        x = self.mixer(x)
        x = x + residual
        
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + residual
        return x

class Model(nn.Module):
    """
    Samba Model Backbone.
    Hybrid architecture alternating between Mamba and sliding window Attention.
    """
    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 4,
        attn_layers: tuple = (1, 3),
        num_heads: int = 8,
        num_kv_heads: int = 8,
        window_size: int = 256,
    ):
        super().__init__()
        config_attn = {
            'layers': attn_layers,
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'qkv_bias': False,
            'window_size': window_size,
            'rope_theta': 10000.0
        }
        self.layers = nn.ModuleList([
            SambaBlock(hidden_size, i, config_attn) for i in range(num_hidden_layers)
        ])
        self.norm_f = RMSNorm(hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return x

# Kernelbench Parameters
batch_size = 2
seq_len = 128
hidden_size = 512
num_hidden_layers = 2
attn_layers = (1,)
num_heads = 8
num_kv_heads = 8

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size, num_hidden_layers, attn_layers, num_heads, num_kv_heads]
