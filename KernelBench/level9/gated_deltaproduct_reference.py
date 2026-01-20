import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Gated Delta Product - Reference implementation.
    
    GatedDeltaProduct is a generalized version that supports arbitrary number of
    Householder transformations. It applies multiple Householder reflections
    sequentially to transform the state.
    
    The core recurrence:
        For each time step t:
            1. Apply forget gate: h = h * exp(g[t])
            2. Apply num_householder Householder transformations sequentially:
               For each j in num_householder:
                   h = h + (v[t,j] - (h @ k[t,j])) * k[t,j] * beta[t,j]
            3. Readout: o[t] = h @ q[t]
    
    Each Householder transformation is: H = I + beta * outer(k, v - h@k)
    This is a rank-1 update that reflects the state.
    
    Based on: Generalized GatedDoubleDeltaNet with multiple Householder transformations.
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2.0,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        use_output_gate: bool = True,
        use_forget_gate: bool = True,
        allow_neg_eigval: bool = True,
        num_householder: int = 2,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * head_dim
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        
        self.use_output_gate = use_output_gate
        self.use_forget_gate = use_forget_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.num_householder = num_householder
        
        # Projections
        # Q is normal size
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        
        # K and V are num_householder times larger (for multiple transformations)
        self.k_proj = nn.Linear(hidden_size, self.key_dim * num_householder, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim * num_householder, bias=False)
        
        # Beta is also num_householder times larger
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads * num_householder, bias=False)
        
        # Forget gate (optional)
        if use_forget_gate:
            self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
            
            # Learnable decay parameters (A_log and dt_bias, like Mamba)
            A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            
            dt_min, dt_max = 0.001, 0.1
            dt = torch.exp(torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt = torch.clamp(dt, min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
        
        # Output gate and projection
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        
        # RMSNorm weight
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask (unused in this reference)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, T, key_dim]
        k = self.k_proj(x)  # [B, T, key_dim * num_householder]
        v = self.v_proj(x)  # [B, T, value_dim * num_householder]
        
        # Apply SiLU activation (simulating short convolution effect)
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)
        
        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        
        # Reshape K and V: split into num_householder chunks
        k = k.view(batch_size, seq_len, self.num_householder, self.num_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len * self.num_householder, self.num_heads, self.head_k_dim)
        
        v = v.view(batch_size, seq_len, self.num_householder, self.num_v_heads, self.head_v_dim)
        v = v.view(batch_size, seq_len * self.num_householder, self.num_v_heads, self.head_v_dim)
        
        # Compute beta
        beta = torch.sigmoid(self.b_proj(x))  # [B, T, num_v_heads * num_householder]
        if self.allow_neg_eigval:
            beta = beta * 2.0  # Allow range [0, 2] for negative eigenvalues
        beta = beta.view(batch_size, seq_len, self.num_householder, self.num_v_heads)
        beta = beta.view(batch_size, seq_len * self.num_householder, self.num_v_heads)
        
        # Compute forget gate (optional)
        if self.use_forget_gate:
            g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)
            # g is [B, T, num_v_heads], but we need it for each time step
        else:
            g = None
        
        # Expand Q and K for GVA if needed
        if self.num_v_heads > self.num_heads:
            expand_ratio = self.num_v_heads // self.num_heads
            q = q.repeat_interleave(expand_ratio, dim=2)
            k = k.repeat_interleave(expand_ratio, dim=2)
        
        # ============================================
        # Gated Delta Product with Multiple Householder
        # ============================================
        o = self._gated_delta_product(q, k, v, g, beta)
        
        # Output normalization
        if self.use_output_gate:
            gate = self.g_proj(x).view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
            o = self._gated_rms_norm(o, gate, self.o_norm_weight)
        else:
            o = self._rms_norm(o, self.o_norm_weight)
        
        # Reshape and project output
        o = o.view(batch_size, seq_len, self.value_dim)
        o = self.o_proj(o)
        
        return o
    
    def _gated_delta_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Gated Delta Product with multiple Householder transformations.
        
        For each time step t:
            1. Apply forget gate: h = h * exp(g[t])
            2. Apply num_householder Householder transformations:
               For each j: h = h + (v[t,j] - h@k[t,j]) * k[t,j] * beta[t,j]
            3. Readout: o[t] = h @ q[t]
        
        Args:
            q: [B, T, num_v_heads, head_k_dim]
            k: [B, T*num_householder, num_v_heads, head_k_dim]
            v: [B, T*num_householder, num_v_heads, head_v_dim]
            g: [B, T, num_v_heads] or None - forget gate
            beta: [B, T*num_householder, num_v_heads] - Householder strength
            
        Returns:
            o: [B, T, num_v_heads, head_v_dim]
        """
        B, T, H, K = q.shape
        V = v.shape[-1]
        
        # Work in float32 for stability
        q = q.float()
        k = k.float()
        v = v.float()
        beta = beta.float()
        if g is not None:
            g = g.float()
        
        scale = K ** -0.5
        
        # Initialize state: [B, H, K, V]
        h = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
        
        outputs = []
        
        for t in range(T):
            q_t = q[:, t, :, :]  # [B, H, K]
            
            # Apply forget gate if provided
            if g is not None:
                g_t = g[:, t, :]  # [B, H]
                decay = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
                h = h * decay
            
            # Apply num_householder Householder transformations sequentially
            for j in range(self.num_householder):
                idx = t * self.num_householder + j
                k_tj = k[:, idx, :, :]  # [B, H, K]
                v_tj = v[:, idx, :, :]  # [B, H, V]
                beta_tj = beta[:, idx, :]  # [B, H]
                
                # L2 normalize k (as done in kernel)
                k_tj = F.normalize(k_tj, p=2, dim=-1)
                
                # Householder transformation:
                # prediction = h @ k_tj: [B, H, V] = einsum('bhkv,bhk->bhv', h, k_tj)
                prediction = torch.einsum('bhkv,bhk->bhv', h, k_tj)  # [B, H, V]
                
                # delta = v_tj - prediction: [B, H, V]
                delta = v_tj - prediction
                
                # Update: h = h + outer(k_tj, delta) * beta_tj
                # h[b,h] += beta_tj[b,h] * outer(k_tj[b,h], delta[b,h])
                update = torch.einsum('bhk,bhv->bhkv', k_tj, delta)  # [B, H, K, V]
                beta_expanded = beta_tj.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
                h = h + update * beta_expanded
            
            # Readout: o[t] = h @ q_t
            # For each batch and head: o[b,h] = h[b,h] @ q_t[b,h]
            # h[b,h] is [K, V], q_t[b,h] is [K]
            q_t_scaled = q_t * scale
            o_t = torch.einsum('bhkv,bhk->bhv', h, q_t_scaled)  # [B, H, V]
            
            outputs.append(o_t)
        
        # Stack outputs: [T, B, H, V]
        outputs = torch.stack(outputs, dim=0)
        
        # Transpose to [B, T, H, V]
        outputs = outputs.transpose(0, 1)
        
        return outputs
    
    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm implementation."""
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-5)
        return (x.float() / rms * weight).to(x.dtype)
    
    def _gated_rms_norm(self, x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Gated RMSNorm: RMSNorm(x) * sigmoid(g)."""
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-5)
        x_norm = (x.float() / rms) * weight
        return (x_norm * torch.sigmoid(g.float())).to(x.dtype)


# Problem dimensions
batch_size = 4
seq_len = 512
hidden_size = 2048
expand_v = 2.0
head_dim = 256
num_heads = 6
num_v_heads = 6
use_output_gate = True
use_forget_gate = True
allow_neg_eigval = True
num_householder = 2


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]


def get_init_inputs():
    return [hidden_size, expand_v, head_dim, num_heads, num_v_heads, 
            use_output_gate, use_forget_gate, allow_neg_eigval, num_householder]

