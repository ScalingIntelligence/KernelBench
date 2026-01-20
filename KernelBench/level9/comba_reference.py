import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Comba (Closed-loop Control Bilinear RNN) Attention - Reference implementation.
    
    Comba improves upon bilinear RNNs (like Delta Rule) with "closed-loop control":
    - Uses an auxiliary key `p` (decayed version of `k`) for the prediction step
    - Uses the regular key `k` for the state update step
    - This separation allows better control over the memory dynamics
    
    The core recurrence:
        # Prediction using auxiliary key p (closed-loop feedback)
        v_new = v[t] - h @ p[t]  # Subtract current state's prediction using p
        
        # State decay
        h = h * exp(g[t])
        
        # Scale by beta
        v_new = v_new * beta[t]
        
        # Update using regular key k
        h = h + outer(k[t], v_new)
        
        # Output
        o[t] = h @ q[t]
    
    The key difference from standard Delta Rule:
    - Delta Rule: uses same k for both prediction (v - h @ k) and update (h + k @ v)
    - Comba: uses p for prediction (v - h @ p) and k for update (h + k @ v)
    
    This "closed-loop" structure (using different keys) provides better control.
    
    Based on: "Comba: Improving Bilinear RNNs with Closed-loop Control"
    https://arxiv.org/abs/2506.02475
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2.0,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        use_output_gate: bool = True,
        use_output_correction: bool = True,
        use_inner_decay: bool = True,
        correction_factor: float = 1.0,
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
        self.use_output_correction = use_output_correction
        self.use_inner_decay = use_inner_decay
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        
        # Gate projections for decay
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)  # For computing g
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)  # For computing beta
        
        # Inner decay parameter (for computing auxiliary key p)
        if use_inner_decay:
            self.decay = nn.Parameter(torch.ones(num_heads))
        
        # Output correction parameter D
        if use_output_correction:
            self.D = nn.Parameter(torch.ones(num_heads) * correction_factor)
        
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
        k = self.k_proj(x)  # [B, T, key_dim]
        v = self.v_proj(x)  # [B, T, value_dim]
        
        # Apply SiLU activation (simulating short convolution effect)
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)
        
        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        
        # Compute auxiliary key p (decayed version of k)
        if self.use_inner_decay:
            decay_factor = torch.sigmoid(self.decay).view(1, 1, self.num_heads, 1)
            p = k * decay_factor  # [B, T, num_heads, head_k_dim]
        else:
            p = k
        
        # Output correction: q = q - D * p
        if self.use_output_correction:
            D = self.D.view(1, 1, self.num_heads, 1)
            q = q - D * p
        
        # Expand Q, K, P for GVA (Grouped Value Attention) if needed
        if self.num_v_heads > self.num_heads:
            expand_ratio = self.num_v_heads // self.num_heads
            q = q.repeat_interleave(expand_ratio, dim=2)
            k = k.repeat_interleave(expand_ratio, dim=2)
            p = p.repeat_interleave(expand_ratio, dim=2)
        
        # Compute beta (sigmoid) and g (decay)
        beta = torch.sigmoid(self.b_proj(x))  # [B, T, num_v_heads]
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)  # [B, T, num_v_heads]
        
        # ============================================
        # Comba Recurrence (Closed-loop Control)
        # ============================================
        o = self._comba_recurrence(q, k, p, v, g, beta)
        
        # Output normalization and gating
        if self.use_output_gate:
            gate = self.g_proj(x).view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
            o = self._gated_rms_norm(o, gate, self.o_norm_weight)
        else:
            o = self._rms_norm(o, self.o_norm_weight)
        
        # Reshape and project output
        o = o.view(batch_size, seq_len, self.value_dim)
        o = self.o_proj(o)
        
        return o
    
    def _comba_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        p: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Comba recurrence with closed-loop control.
        
        Key difference from Delta Rule:
        - Uses auxiliary key `p` for prediction: v_new = v - h @ p
        - Uses regular key `k` for update: h = h + outer(k, v_new)
        
        Args:
            q: [B, T, num_v_heads, head_k_dim] - queries
            k: [B, T, num_v_heads, head_k_dim] - keys (for state update)
            p: [B, T, num_v_heads, head_k_dim] - auxiliary keys (for prediction)
            v: [B, T, num_v_heads, head_v_dim] - values
            g: [B, T, num_v_heads] - decay gate (negative log values)
            beta: [B, T, num_v_heads] - scaling factor
            
        Returns:
            o: [B, T, num_v_heads, head_v_dim] - output
        """
        B, T, H, K = q.shape
        V = v.shape[-1]
        
        scale = K ** -0.5
        
        # Work in float32 for stability
        q = q.float()
        k = k.float()
        p = p.float()
        v = v.float()
        g = g.float()
        beta = beta.float()
        
        outputs = []
        
        for b in range(B):
            batch_outputs = []
            
            for h in range(H):
                # Initialize state: h_state is [head_k_dim, head_v_dim]
                h_state = torch.zeros(K, V, device=q.device, dtype=torch.float32)
                
                head_outputs = []
                
                for t in range(T):
                    # Get current vectors
                    q_t = q[b, t, h]  # [K]
                    k_t = k[b, t, h]  # [K]
                    p_t = p[b, t, h]  # [K] - auxiliary key
                    v_t = v[b, t, h]  # [V]
                    g_t = g[b, t, h]  # scalar
                    beta_t = beta[b, t, h]  # scalar
                    
                    # L2 normalize q, k, p
                    q_t = F.normalize(q_t, p=2, dim=-1)
                    k_t = F.normalize(k_t, p=2, dim=-1)
                    p_t = F.normalize(p_t, p=2, dim=-1)
                    
                    # Scale query
                    q_t = q_t * scale
                    
                    # ===== CLOSED-LOOP CONTROL =====
                    # Prediction using auxiliary key p (NOT k)
                    # This is the key difference from standard Delta Rule
                    prediction = h_state.T @ p_t  # [V] = h_state^T @ p
                    v_new = v_t - prediction  # Subtract prediction from value
                    
                    # Decay the state
                    h_state = h_state * torch.exp(g_t)
                    
                    # Scale by beta
                    v_new = v_new * beta_t
                    
                    # Update state using regular key k (NOT p)
                    # h = h + outer(k, v_new)
                    h_state = h_state + torch.outer(k_t, v_new)
                    
                    # Output: o = h @ q
                    o_t = h_state.T @ q_t  # [V]
                    
                    head_outputs.append(o_t)
                
                # Stack outputs for this head: [T, V]
                head_outputs = torch.stack(head_outputs, dim=0)
                batch_outputs.append(head_outputs)
            
            # Stack outputs for this batch: [T, H, V]
            batch_outputs = torch.stack(batch_outputs, dim=1)
            outputs.append(batch_outputs)
        
        # Stack all batches: [B, T, H, V]
        outputs = torch.stack(outputs, dim=0)
        
        return outputs
    
    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm implementation."""
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x.float() / rms * weight).to(x.dtype)
    
    def _gated_rms_norm(self, x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Gated RMSNorm: RMSNorm(x) * sigmoid(g)."""
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
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
use_output_correction = True
use_inner_decay = True
correction_factor = 1.0


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]


def get_init_inputs():
    return [hidden_size, expand_v, head_dim, num_heads, num_v_heads, 
            use_output_gate, use_output_correction, use_inner_decay, correction_factor]

