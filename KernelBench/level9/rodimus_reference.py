import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Rodimus Attention - Reference implementation.
    
    Rodimus combines:
    1. Gated Linear Attention (GLA) with learnable gates
    2. Special gate computation: g_gate, tau_gate, it_gate, rt_gate
    3. Input gating via i_gate_proj
    4. Residual connection with learnable weight
    
    The core GLA recurrence:
        h[t] = h[t-1] * exp(rt_gate[t]) + outer(k[t], v[t])
        o[t] = sum(h[t] * q[t], dim=-2)
    
    Key features:
    - k is normalized and scaled by it_gate (input gate for keys)
    - rt_gate is computed as: -g_gate * tau_gate (forget gate)
    - v is gated by i_gate_proj (input gate for values)
    - Residual connection with learnable weight
    
    Based on the Rodimus architecture.
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        expand_ratio: int = 64,
        input_gate_low_rank: int = 16,
        use_short_conv: bool = True,
        conv_size: int = 4,
        residual_in_fp32: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.d_inner = int(hidden_size * 2)  # Expanded dimension
        self.expand_ratio = expand_ratio
        self.mem_size = expand_ratio  # Memory size (K and V dimension)
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.residual_in_fp32 = residual_in_fp32
        self.input_gate_low_rank = input_gate_low_rank
        
        # Main projections (MLP-like structure)
        self.gate_proj = nn.Linear(hidden_size, self.d_inner, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.d_inner, bias=False)
        self.down_proj = nn.Linear(self.d_inner, hidden_size, bias=False)
        
        # Gated activation norm
        self.activation_norm_weight = nn.Parameter(torch.ones(self.d_inner))
        
        # Residual weight (learnable)
        self.residual_weight = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))
        
        # Attention projections
        self.q_proj = nn.Linear(self.d_inner, self.mem_size, bias=False)
        self.k_proj = nn.Linear(self.d_inner, self.mem_size, bias=False)
        
        # Gate projections
        self.g_gate_proj = nn.Linear(self.d_inner, self.mem_size, bias=True)
        self.tau_gate_proj = nn.Linear(self.d_inner, self.mem_size, bias=True)
        
        # Input gate for values (low-rank)
        self.i_gate_proj = nn.Sequential(
            nn.Linear(self.d_inner, input_gate_low_rank, bias=False),
            nn.Linear(input_gate_low_rank, self.d_inner, bias=True),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask (unused in this reference)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Up projection and gate projection
        hidden_states = self.up_proj(x)  # [B, T, d_inner]
        final_gate = self.gate_proj(x)  # [B, T, d_inner]
        
        # Short convolution (simulated with SiLU)
        if self.use_short_conv:
            shift_hidden_states = F.silu(hidden_states)
        else:
            shift_hidden_states = hidden_states
        
        # Project to Q, K
        q = self.q_proj(shift_hidden_states)  # [B, T, mem_size]
        k = self.k_proj(shift_hidden_states)  # [B, T, mem_size]
        
        # Input gate for values
        v = self.i_gate_proj(hidden_states) * hidden_states  # [B, T, d_inner]
        
        # Compute gates
        g_gate = F.linear(shift_hidden_states, self.g_gate_proj.weight) + self.g_gate_proj.bias.float()
        tau_gate = F.linear(shift_hidden_states, self.tau_gate_proj.weight) + self.tau_gate_proj.bias.float()
        
        # Process gates
        g_gate = F.softplus(g_gate)  # [B, T, mem_size]
        tau_gate = torch.sigmoid(tau_gate)  # [B, T, mem_size]
        
        # Input gate for keys: it_gate = g_gate^tau_gate
        it_gate = g_gate ** tau_gate  # [B, T, mem_size]
        
        # Forget gate (for state decay): rt_gate_log = -g_gate * tau_gate
        rt_gate_log = -g_gate * tau_gate  # [B, T, mem_size]
        
        # Normalize and scale k by it_gate
        k = F.normalize(k.float(), dim=-1, eps=1e-12) * it_gate  # [B, T, mem_size]
        
        # Reshape for attention: [B, 1, T, mem_size] -> [B, 1, mem_size, T]
        q = q.unsqueeze(1).transpose(1, 2)  # [B, 1, T, mem_size]
        k = k.unsqueeze(1).transpose(1, 2)  # [B, 1, T, mem_size]
        v = v.unsqueeze(1).transpose(1, 2)  # [B, 1, T, d_inner]
        rt_gate_log = rt_gate_log.unsqueeze(1).transpose(1, 2)  # [B, 1, T, mem_size]
        
        # ============================================
        # GLA (Gated Linear Attention)
        # ============================================
        o = self._gla_attention(q, k, v, rt_gate_log)  # [B, 1, T, d_inner]
        
        # Reshape back: [B, 1, T, d_inner] -> [B, T, d_inner]
        o = o.transpose(1, 2).squeeze(1)  # [B, T, d_inner]
        
        # Residual connection with learnable weight
        if self.residual_in_fp32:
            residual = shift_hidden_states.float() * self.residual_weight
        else:
            residual = shift_hidden_states * self.residual_weight
        o = (o + residual).to(o.dtype)
        
        # Gated activation norm
        o = self._gated_rms_norm(o, final_gate, self.activation_norm_weight)
        
        # Down projection
        o = self.down_proj(o)
        
        return o
    
    def _gla_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor
    ) -> torch.Tensor:
        """
        Gated Linear Attention (GLA) recurrence.
        
        The recurrence:
            h[t] = h[t-1] * exp(gk[t]) + outer(k[t], v[t])
            o[t] = sum(h[t] * q[t], dim=-2)
        
        Args:
            q: [B, 1, T, mem_size] - queries
            k: [B, 1, T, mem_size] - keys
            v: [B, 1, T, d_inner] - values
            gk: [B, 1, T, mem_size] - forget gates (in log space)
            
        Returns:
            o: [B, 1, T, d_inner] - output
        """
        B, H, T, K = q.shape
        V = v.shape[-1]
        
        # Work in float32 for stability
        q = q.float()
        k = k.float()
        v = v.float()
        gk = gk.float()
        
        scale = K ** -0.5
        
        # Initialize state: [B, H, K, V]
        h = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
        
        outputs = []
        
        for t in range(T):
            q_t = q[:, :, t, :]  # [B, H, K]
            k_t = k[:, :, t, :]  # [B, H, K]
            v_t = v[:, :, t, :]  # [B, H, V]
            gk_t = gk[:, :, t, :]  # [B, H, K]
            
            # Scale query
            q_t = q_t * scale
            
            # Decay state: h = h * exp(gk)
            # gk is negative (forget gate), so exp(gk) < 1
            decay = torch.exp(gk_t).unsqueeze(-1)  # [B, H, K, 1]
            h = h * decay
            
            # Update state: h = h + outer(k, v)
            # For each batch and head: h[b,h] += outer(k_t[b,h], v_t[b,h])
            # h[b,h] is [K, V], k_t[b,h] is [K], v_t[b,h] is [V]
            kv_outer = torch.einsum('bhk,bhv->bhkv', k_t, v_t)  # [B, H, K, V]
            h = h + kv_outer
            
            # Output: o = sum(h * q, dim=-2)
            # For each batch and head: o[b,h] = sum(h[b,h] * q_t[b,h], dim=0)
            # h[b,h] is [K, V], q_t[b,h] is [K]
            o_t = torch.einsum('bhk,bhkv->bhv', q_t, h)  # [B, H, V]
            
            outputs.append(o_t)
        
        # Stack outputs: [T, B, H, V]
        outputs = torch.stack(outputs, dim=0)
        
        # Transpose to [B, H, T, V]
        outputs = outputs.transpose(0, 1).transpose(1, 2)
        
        return outputs
    
    def _gated_rms_norm(self, x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Gated RMSNorm: RMSNorm(x) * sigmoid(g)
        
        Args:
            x: [B, T, D]
            g: [B, T, D] - gate
            weight: [D] - norm weight
            
        Returns:
            output: [B, T, D]
        """
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-5)
        x_norm = (x.float() / rms) * weight
        return (x_norm * torch.sigmoid(g.float())).to(x.dtype)


# Problem dimensions
batch_size = 4
seq_len = 512
hidden_size = 1024
expand_ratio = 64
input_gate_low_rank = 16
use_short_conv = True
conv_size = 4
residual_in_fp32 = True


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]


def get_init_inputs():
    return [hidden_size, expand_ratio, input_gate_low_rank, use_short_conv, conv_size, residual_in_fp32]

