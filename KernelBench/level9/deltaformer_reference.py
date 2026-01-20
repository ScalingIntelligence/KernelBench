import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    DeltaFormer Attention - A reference implementation using pure PyTorch.
    
    DeltaFormer is a two-stage attention mechanism:
    1. Delta Update: Compute u[i] = v[i] - beta[i] * sum_{j<i} softmax(q[i] @ k[:i]^T) @ u[:i]
       This creates "delta-updated" values through a recurrence.
    2. Causal Attention: Apply standard causal attention o = softmax(Q @ K^T) @ U
    
    Based on: "Understanding Transformer from the Perspective of Associative Memory"
    https://arxiv.org/pdf/2505.19488
    """
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int = None, head_dim: int = None):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        
        if head_dim is None:
            self.head_dim = hidden_size // num_heads
        else:
            self.head_dim = head_dim
        
        self.kv_dim = self.num_kv_heads * self.head_dim
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)  # Beta projection for delta updates
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # RMSNorm for Q and K
        self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        
        # Rotary embedding parameters
        self.rope_theta = 10000.0

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs DeltaFormer attention computation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            mask: Optional mask tensor (unused in this reference, kept for API compatibility).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, hidden_size]
        k = self.k_proj(x)  # [batch_size, seq_len, kv_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, kv_dim]
        beta = self.b_proj(x)  # [batch_size, seq_len, num_heads]
        
        # Reshape to multi-head format: [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RMSNorm to Q and K
        q = self._rms_norm(q, self.q_norm_weight)
        k = self._rms_norm(k, self.k_norm_weight)
        
        # Apply rotary position embeddings
        q, k = self._apply_rotary_emb(q, k, seq_len)
        
        # Expand K and V for grouped query attention if needed
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)  # [batch_size, num_heads, seq_len]
        
        # ============================================
        # DeltaFormer Attention Core
        # ============================================
        
        # Stage 1: Compute delta-updated values U
        # u[i] = v[i] - beta[i] * sum_{j<i} softmax(q[i] @ k[:i]^T) @ u[:i]
        u = self._compute_delta_values(q, k, v, beta)
        
        # Stage 2: Standard causal attention with (Q, K, U)
        # o = softmax(Q @ K^T / sqrt(d)) @ U
        o = self._causal_attention(q, k, u)
        
        # Reshape back: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, hidden_size]
        o = o.transpose(1, 2).contiguous()
        o = o.view(batch_size, seq_len, -1)
        
        # Final output projection
        o = self.o_proj(o)
        
        return o
    
    def _compute_delta_values(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Compute delta-updated values through the recurrence:
        u[i] = v[i] - beta[i] * sum_{j<i} softmax(q[i] @ k[:i]^T) @ u[:i]
        
        For position 0, there are no previous positions, so u[0] = v[0].
        For position i > 0, we attend to positions j < i (strictly causal) and
        subtract a beta-scaled weighted combination of previous u values.
        
        Args:
            q: [B, H, T, D]
            k: [B, H, T, D]
            v: [B, H, T, D]
            beta: [B, H, T]
        
        Returns:
            u: [B, H, T, D]
        """
        B, H, T, D = q.shape
        qk_scale = 1.0 / math.sqrt(D)
        
        # Compute all Q @ K^T scores at once
        scores = torch.matmul(q, k.transpose(-2, -1)) * qk_scale  # [B, H, T, T]
        
        # Apply strictly causal softmax (j < i, not j <= i)
        probs = self._tril_softmax(scores, strict=True)  # [B, H, T, T]
        
        # Sequential computation of u (the recurrence cannot be parallelized naively)
        u_list = []
        for t in range(T):
            if t == 0:
                # No previous positions to attend to
                u_t = v[:, :, t, :]  # [B, H, D]
            else:
                # Attention weights for position t attending to positions 0..t-1
                w = probs[:, :, t, :t]  # [B, H, t]
                # Stack all previous u values
                u_prev = torch.stack(u_list, dim=-2)  # [B, H, t, D]
                # Weighted sum of previous u values
                weighted_sum = torch.matmul(w.unsqueeze(-2), u_prev).squeeze(-2)  # [B, H, D]
                # Delta update: u[t] = v[t] - beta[t] * weighted_sum
                u_t = v[:, :, t, :] - beta[:, :, t].unsqueeze(-1) * weighted_sum
            u_list.append(u_t)
        
        u = torch.stack(u_list, dim=2)  # [B, H, T, D]
        return u
    
    def _tril_softmax(self, scores: torch.Tensor, strict: bool = True) -> torch.Tensor:
        """
        Row-wise causal softmax over strictly lower-triangular (j < i) positions.
        
        Args:
            scores: [B, H, T, T] raw attention scores
            strict: if True, mask out diagonal (j < i). If False, include diagonal (j <= i).
        
        Returns:
            probs: [B, H, T, T] with probabilities on valid positions, zeros elsewhere
        """
        T = scores.size(-1)
        device = scores.device
        
        i_idx = torch.arange(T, device=device).view(1, 1, T, 1)
        j_idx = torch.arange(T, device=device).view(1, 1, 1, T)
        
        if strict:
            mask = (j_idx < i_idx)  # strictly lower triangular
        else:
            mask = (j_idx <= i_idx)  # lower triangular including diagonal
        
        # Masked softmax with numerical stability
        masked_scores = scores.masked_fill(~mask, float('-inf'))
        max_per_row = masked_scores.max(dim=-1, keepdim=True).values
        max_per_row = torch.where(max_per_row == float('-inf'), torch.zeros_like(max_per_row), max_per_row)
        
        exp_scores = (masked_scores - max_per_row).exp()
        exp_scores = exp_scores.masked_fill(~mask, 0.0)
        
        denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        probs = exp_scores / denom
        
        return probs
    
    def _causal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Standard causal attention: o = softmax(Q @ K^T / sqrt(d), causal_mask) @ V
        
        Args:
            q: [B, H, T, D]
            k: [B, H, T, D]
            v: [B, H, T, D]
        
        Returns:
            o: [B, H, T, D]
        """
        B, H, T, D = q.shape
        qk_scale = 1.0 / math.sqrt(D)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * qk_scale  # [B, H, T, T]
        
        # Standard causal mask (j <= i, including diagonal)
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
        o = torch.matmul(attn_weights, v)  # [B, H, T, D]
        
        return o
    
    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm implementation."""
        # x: [batch_size, seq_len, num_heads, head_dim]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x / rms) * weight
    
    def _apply_rotary_emb(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to Q and K.
        
        Note: Q has num_heads dimensions, K has num_kv_heads dimensions (may differ in GQA).
        """
        device = q.device
        dtype = q.dtype
        
        num_q_heads = q.shape[2]
        num_k_heads = k.shape[2]
        
        # Create position indices: [1, seq_len, 1]
        positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        
        # Create frequency matrix
        dim_t = torch.arange(self.head_dim // 2, device=device, dtype=dtype)
        dim_t = self.rope_theta ** (2 * dim_t / self.head_dim)
        dim_t = dim_t.unsqueeze(0).unsqueeze(0)  # [1, 1, head_dim//2]
        
        # Compute angles: [1, seq_len, head_dim//2]
        angles = positions / dim_t
        
        # Create rotation matrices
        cos_base = torch.cos(angles)
        sin_base = torch.sin(angles)
        
        # Expand for Q and K heads separately
        cos_q = cos_base.unsqueeze(2).expand(-1, -1, num_q_heads, -1)
        sin_q = sin_base.unsqueeze(2).expand(-1, -1, num_q_heads, -1)
        cos_k = cos_base.unsqueeze(2).expand(-1, -1, num_k_heads, -1)
        sin_k = sin_base.unsqueeze(2).expand(-1, -1, num_k_heads, -1)
        
        # Apply rotation to Q
        q1, q2 = q.chunk(2, dim=-1)
        q_rot = torch.cat([q1 * cos_q - q2 * sin_q, q1 * sin_q + q2 * cos_q], dim=-1)
        
        # Apply rotation to K
        k1, k2 = k.chunk(2, dim=-1)
        k_rot = torch.cat([k1 * cos_k - k2 * sin_k, k1 * sin_k + k2 * cos_k], dim=-1)
        
        return q_rot, k_rot


# Problem dimensions
batch_size = 4
seq_len = 512
hidden_size = 2048
num_heads = 32
num_kv_heads = 8  # Grouped query attention
head_dim = hidden_size // num_heads


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]


def get_init_inputs():
    return [hidden_size, num_heads, num_kv_heads, head_dim]

