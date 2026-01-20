import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Reference implementation of Gated Slot Attention (GSA).
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        num_kv_heads: int = 4,
        head_k_dim: int = 64,
        head_v_dim: int = 64,
        num_slots: int = 64,
        norm_eps: float = 1e-5,
    ):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.num_slots = num_slots
        self.norm_eps = norm_eps
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_k_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_v_dim, bias=False)
        self.f_proj = nn.Linear(hidden_size, num_kv_heads * num_slots, bias=False)
        
        self.g_norm_weight = nn.Parameter(torch.ones(num_heads * head_v_dim))
        self.o_proj = nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, hidden_size]
        Returns:
            torch.Tensor: [batch_size, seq_len, hidden_size]
        """
        B, T, _ = x.shape
        H, HK, HV, M = self.num_heads, self.head_k_dim, self.head_v_dim, self.num_slots
        HKV = self.num_kv_heads
        NG = H // HKV
        
        # Projections
        q = self.q_proj(x).view(B, T, H, HK)
        k = self.k_proj(x).view(B, T, HKV, HK)
        v = self.v_proj(x).view(B, T, HKV, HV)
        f = self.f_proj(x).view(B, T, HKV, M)
        
        # Swish feature map for q, k and silu for v
        q = q * q.sigmoid()
        k = k * k.sigmoid()
        v = v * v.sigmoid()
        
        # Gating: f is log-decay, s is the complementary weight
        f = F.logsigmoid(f) / 8.0 # gate_logit_normalizer=8
        s = (1.0 - f.exp())
        
        # Grouped Query Attention: tile k, v, f, s if num_heads > num_kv_heads
        if NG > 1:
            k = k.repeat_interleave(NG, dim=2)
            v = v.repeat_interleave(NG, dim=2)
            f = f.repeat_interleave(NG, dim=2)
            s = s.repeat_interleave(NG, dim=2)
            
        q, k, v, f, s = q.float(), k.float(), v.float(), f.float(), s.float()
        
        # First recurrence: compute soft slots assignment ok
        hk = torch.zeros(B, H, HK, M, device=x.device, dtype=torch.float32)
        ok = torch.zeros(B, T, H, M, device=x.device, dtype=torch.float32)
        scale = HK ** -0.5
        
        for i in range(T):
            q_i = q[:, i] * scale
            k_i = k[:, i]
            v_i = s[:, i]
            g_i = f[:, i].exp()
            # hk state update: hk = hk * decay + k @ s.T
            hk = hk * g_i.unsqueeze(-2) + torch.einsum('b h k, b h m -> b h k m', k_i, v_i)
            # Read from hk: ok = q.T @ hk
            ok[:, i] = torch.einsum('b h k, b h k m -> b h m', q_i, hk)
            
        # Global softmax over slots
        qv = F.softmax(ok, dim=-1)
        
        # Second recurrence: compute output ov based on soft slots
        hv = torch.zeros(B, H, M, HV, device=x.device, dtype=torch.float32)
        ov = torch.zeros(B, T, H, HV, device=x.device, dtype=torch.float32)
        
        for i in range(T):
            q_i = qv[:, i]
            k_i = s[:, i]
            v_i = v[:, i]
            g_i = f[:, i].exp()
            # hv state update: hv = hv * decay + s @ v.T
            hv = hv * g_i.unsqueeze(-1) + torch.einsum('b h m, b h v -> b h m v', k_i, v_i)
            # Read from hv: ov = qv.T @ hv
            ov[:, i] = torch.einsum('b h m, b h m v -> b h v', q_i, hv)
            
        # Final output processing
        o = ov.reshape(B, T, -1)
        o = F.silu(o)
        
        # RMSNorm
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        o = o * self.g_norm_weight
        
        return self.o_proj(o.to(x.dtype))

# Kernelbench Parameters
batch_size = 2
seq_len = 64
hidden_size = 1024
num_heads = 4
num_kv_heads = 4
head_k_dim = 64
head_v_dim = 64
num_slots = 64

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size, num_heads, num_kv_heads, head_k_dim, head_v_dim, num_slots]
