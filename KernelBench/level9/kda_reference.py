import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Reference implementation of Kimi Delta Attention (KDA).
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        expand_v: float = 1.0,
        head_dim: int = 64,
        num_heads: int = 16,
        num_v_heads: int = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        norm_eps: float = 1e-5,
    ):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * head_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.norm_eps = norm_eps
        
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, padding=conv_size-1)
            self.k_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, padding=conv_size-1)
            self.v_conv1d = nn.Conv1d(self.value_dim, self.value_dim, conv_size, groups=self.value_dim, padding=conv_size-1)

        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim))
        
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (B, T, H_in).
        Returns:
            torch.Tensor: Output tensor of shape (B, T, H_in).
        """
        B, T, _ = hidden_states.shape
        H, HK, HV = self.num_heads, self.head_dim, self.head_v_dim
        NV = self.num_v_heads
        
        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        if self.use_short_conv:
            # Conv1d expects [B, C, T]
            q = F.silu(self.q_conv1d(q.transpose(1, 2))[..., :T].transpose(1, 2))
            k = F.silu(self.k_conv1d(k.transpose(1, 2))[..., :T].transpose(1, 2))
            v = F.silu(self.v_conv1d(v.transpose(1, 2))[..., :T].transpose(1, 2))
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
            
        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states).sigmoid()
        
        # Reshape for multi-head
        q = q.view(B, T, H, HK)
        k = k.view(B, T, H, HK)
        g = g.view(B, T, H, HK)
        v = v.view(B, T, NV, HV)
        
        # Expand if GVA
        if NV > H:
            groups = NV // H
            q = q.repeat_interleave(groups, dim=2)
            k = k.repeat_interleave(groups, dim=2)
            g = g.repeat_interleave(groups, dim=2)
            beta = beta.repeat_interleave(groups, dim=2)
            H_eff = NV
        else:
            H_eff = H
            
        # Post-process g and scale q
        A = self.A_log.exp()
        if NV > H: A = A.repeat_interleave(NV // H)
        
        dt_bias = self.dt_bias.view(H, HK)
        if NV > H: dt_bias = dt_bias.repeat_interleave(NV // H, dim=0)
        dt_bias = dt_bias.view(1, 1, H_eff, HK)
        
        # g = -exp(A) * softplus(g + dt_bias)
        g = -A.view(1, 1, H_eff, 1) * F.softplus(g + dt_bias)
        
        scale = HK ** -0.5
        
        # Reference recurrence loop
        # S: [B, H_eff, HK, HV]
        S = torch.zeros(B, H_eff, HK, HV, device=hidden_states.device, dtype=torch.float32)
        o = torch.zeros(B, T, H_eff, HV, device=hidden_states.device, dtype=torch.float32)
        
        q, k, v, g, beta = q.float(), k.float(), v.float(), g.float(), beta.float()
        
        for t in range(T):
            q_t = q[:, t] # [B, H_eff, HK]
            k_t = k[:, t] # [B, H_eff, HK]
            v_t = v[:, t] # [B, H_eff, HV]
            g_t = g[:, t] # [B, H_eff, HK]
            beta_t = beta[:, t] # [B, H_eff]
            
            # QK L2 Norm
            q_t = q_t / (torch.norm(q_t, p=2, dim=-1, keepdim=True) + 1e-6)
            k_t = k_t / (torch.norm(k_t, p=2, dim=-1, keepdim=True) + 1e-6)
            q_t = q_t * scale
            
            # S_t = S_{t-1} * exp(g_t)
            S = S * g_t.exp().unsqueeze(-1)
            
            # Delta Update: v_update = beta_t * (v_t - k_t^T * S_t)
            # k_t: [B, H_eff, HK], S: [B, H_eff, HK, HV]
            # kS: [B, H_eff, HV]
            kS = torch.einsum('b h k, b h k v -> b h v', k_t, S)
            v_update = beta_t.unsqueeze(-1) * (v_t - kS)
            
            # S_t = S_t + k_t * v_update^T
            S = S + torch.einsum('b h k, b h v -> b h k v', k_t, v_update)
            
            # o_t = S_t^T * q_t
            o[:, t] = torch.einsum('b h k, b h k v -> b h v', q_t, S)
            
        # Output Norm and Gating
        # g_proj for final gating
        gate = self.g_proj(hidden_states).view(B, T, NV, HV)
        
        # RMSNorm per head
        # o: [B, T, NV, HV]
        o_norm = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        o_norm = o_norm * self.o_norm_weight.view(1, 1, 1, HV)
        
        # Apply gate
        o = o_norm * gate.sigmoid()
        
        # Project back
        o = o.view(B, T, NV * HV)
        return self.o_proj(o.to(hidden_states.dtype))

# Kernelbench Parameters
batch_size = 4
seq_len = 128 # Smaller for reference loop
hidden_size = 1024
expand_v = 1.0
head_dim = 64
num_heads = 16

def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    return [hidden_states]

def get_init_inputs():
    return [hidden_size, expand_v, head_dim, num_heads]
