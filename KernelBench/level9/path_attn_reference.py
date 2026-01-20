import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    PaTH (Path-dependent Transformer with Householder) Attention - Reference implementation.
    
    PaTH Attention applies cumulative Householder transformations to both Q and K,
    making the attention "path-dependent" - earlier tokens affect the representation
    of later tokens through these transformations.
    
    Core idea:
    1. Householder reflection: x_new = x - beta * (x · w) * w, where w is L2-normalized
    2. These reflections are applied cumulatively: each position i sees the effect of
       all Householder reflections from positions j < i
    3. Transformed Q and K are then used in standard softmax attention
    4. Optional forget gate (log-sigmoid) adds exponential decay
    
    The transformation can be understood as:
    - For each position i, transform k[i] by applying Householder reflections from all j < i
    - Similarly transform q[i] 
    - Then compute standard causal softmax attention
    
    Note: This is a simplified reference. The actual kernel uses chunked computation
    with matrix T = solve_tril(beta * w @ w.T) for efficiency.
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int = None,
        use_forget_gate: bool = False,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        
        self.use_forget_gate = use_forget_gate
        self.use_qk_norm = use_qk_norm
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        
        # W projection for Householder vectors (low-rank for parameter efficiency)
        self.w_proj = nn.Sequential(
            nn.Linear(hidden_size, 32, bias=False),
            nn.Linear(32, self.kv_dim, bias=False),
        )
        
        # Beta projection: controls Householder reflection strength
        # sigmoid * 2 allows range [0, 2] for potentially negative eigenvalues
        self.bt_proj = nn.Linear(hidden_size, self.num_kv_heads, bias=True)
        
        # Optional forget gate
        if use_forget_gate:
            self.g_proj = nn.Linear(hidden_size, num_heads, bias=True)
        
        # Optional QK norm
        if use_qk_norm:
            self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim))
            self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask (unused in this reference, kept for API compatibility)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V, W
        q = self.q_proj(x)  # [B, T, hidden_size]
        k = self.k_proj(x)  # [B, T, kv_dim]
        v = self.v_proj(x)  # [B, T, kv_dim]
        w = self.w_proj(x)  # [B, T, kv_dim]
        
        # Beta: controls Householder reflection strength
        # Range [0, 2] allows negative eigenvalues (reflection can flip direction)
        beta = torch.sigmoid(self.bt_proj(x).float()) * 2  # [B, T, num_kv_heads]
        
        # Optional forget gate
        if self.use_forget_gate:
            g = F.logsigmoid(self.g_proj(x).float())  # [B, T, num_heads]
        else:
            g = None
        
        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        w = w.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Optional QK norm
        if self.use_qk_norm:
            q = self._rms_norm(q, self.q_norm_weight)
            k = self._rms_norm(k, self.k_norm_weight)
        
        # L2 normalize W (critical for Householder reflections)
        w = F.normalize(w.float(), p=2, dim=-1)  # [B, T, num_kv_heads, head_dim]
        
        # Apply SiLU activation to W (as done in short conv path)
        w = F.silu(w)
        w = F.normalize(w, p=2, dim=-1)  # Re-normalize after activation
        
        # ============================================
        # PaTH Attention Core
        # ============================================
        
        # Apply cumulative Householder transformations
        q_transformed, k_transformed = self._apply_cumulative_householder(q, k, w, beta)
        
        # Apply causal softmax attention with optional forget gate
        o = self._causal_attention_with_gate(q_transformed, k_transformed, v, g)
        
        # Reshape and project output
        o = o.reshape(batch_size, seq_len, -1)
        o = self.o_proj(o)
        
        return o
    
    def _apply_cumulative_householder(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        w: torch.Tensor, 
        beta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cumulative Householder transformations to Q and K.
        
        For each position i, the transformation is:
        - k_new[i] = k[i] - sum_{j<i} H_j @ k[i]  where H_j is a Householder matrix
        - q_new[i] = q[i] - sum_{j<=i} q[i] @ H_j (applied from the other direction)
        
        Householder reflection: H = beta * w @ w.T
        
        Args:
            q: [B, T, num_heads, head_dim]
            k: [B, T, num_kv_heads, head_dim]
            w: [B, T, num_kv_heads, head_dim] - L2 normalized Householder vectors
            beta: [B, T, num_kv_heads]
            
        Returns:
            q_transformed: [B, T, num_heads, head_dim]
            k_transformed: [B, T, num_kv_heads, head_dim]
        """
        B, T, num_kv_heads, head_dim = k.shape
        num_heads = q.shape[2]
        num_groups = num_heads // num_kv_heads  # For GQA
        
        # Work in float32 for stability
        q = q.float()
        k = k.float()
        w = w.float()
        beta = beta.float()
        
        # Initialize transformed tensors
        q_new = torch.zeros_like(q)
        k_new = torch.zeros_like(k)
        
        # Cumulative Householder state: H_cumsum[i] = sum_{j<i} H_j
        # H_j = beta_j * outer(w_j, w_j)
        # For efficiency, we track this as a [head_dim, head_dim] matrix per head per batch
        
        for b in range(B):
            for h in range(num_kv_heads):
                # Cumulative Householder matrix for this head
                H_cumsum = torch.zeros(head_dim, head_dim, device=k.device, dtype=torch.float32)
                
                for t in range(T):
                    # Get current vectors
                    k_t = k[b, t, h]  # [head_dim]
                    w_t = w[b, t, h]  # [head_dim]
                    beta_t = beta[b, t, h]  # scalar
                    
                    # Transform k[t] using cumulative Householder from positions < t
                    # k_new[t] = k[t] - H_cumsum @ k[t]
                    k_new[b, t, h] = k_t - H_cumsum @ k_t
                    
                    # Update cumulative Householder: H_cumsum += beta * outer(w, w)
                    H_cumsum = H_cumsum + beta_t * torch.outer(w_t, w_t)
                
                # For Q transformation, we need to go the other direction
                # q_new[t] = q[t] - q[t] @ H_cumsum (where H_cumsum includes position t)
                H_cumsum = torch.zeros(head_dim, head_dim, device=k.device, dtype=torch.float32)
                
                # Handle GQA: multiple Q heads share the same KV head
                for g in range(num_groups):
                    q_head_idx = h * num_groups + g
                    H_cumsum_q = torch.zeros(head_dim, head_dim, device=k.device, dtype=torch.float32)
                    
                    for t in range(T):
                        q_t = q[b, t, q_head_idx]  # [head_dim]
                        w_t = w[b, t, h]  # [head_dim]
                        beta_t = beta[b, t, h]  # scalar
                        
                        # Update cumulative Householder first (includes current position)
                        H_cumsum_q = H_cumsum_q + beta_t * torch.outer(w_t, w_t)
                        
                        # Transform q[t]: q_new[t] = q[t] - q[t] @ H_cumsum
                        q_new[b, t, q_head_idx] = q_t - q_t @ H_cumsum_q
        
        return q_new, k_new
    
    def _causal_attention_with_gate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Causal softmax attention with optional forget gate.
        
        If g is provided, attention scores are modified:
        score[i,j] = q[i] @ k[j] + g_cumsum[i] - g_cumsum[j]
        
        where g_cumsum is the cumulative sum of log-sigmoid gate values.
        
        Args:
            q: [B, T, num_heads, head_dim]
            k: [B, T, num_kv_heads, head_dim]
            v: [B, T, num_kv_heads, head_dim]
            g: [B, T, num_heads] optional forget gate (log-sigmoid values)
            
        Returns:
            o: [B, T, num_heads, head_dim]
        """
        B, T, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[2]
        num_groups = num_heads // num_kv_heads
        
        scale = head_dim ** -0.5
        
        # Expand K and V for GQA if needed
        if num_groups > 1:
            k = k.repeat_interleave(num_groups, dim=2)
            v = v.repeat_interleave(num_groups, dim=2)
        
        # Transpose for attention: [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, T, T]
        
        # Apply forget gate if provided
        if g is not None:
            # g_cumsum: cumulative sum of log-sigmoid values
            g_cumsum = torch.cumsum(g, dim=1)  # [B, T, num_heads]
            g_cumsum = g_cumsum.transpose(1, 2)  # [B, num_heads, T]
            
            # Modify scores: score[i,j] += g_cumsum[i] - g_cumsum[j]
            scores = scores + g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        o = torch.matmul(attn_weights, v)  # [B, num_heads, T, head_dim]
        
        # Transpose back: [B, T, num_heads, head_dim]
        o = o.transpose(1, 2)
        
        return o
    
    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm implementation."""
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x.float() / rms * weight).to(x.dtype)


# Problem dimensions
batch_size = 4
seq_len = 512
hidden_size = 2048
num_heads = 32
num_kv_heads = 8  # GQA
use_forget_gate = True
use_qk_norm = False


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]


def get_init_inputs():
    return [hidden_size, num_heads, num_kv_heads, use_forget_gate, use_qk_norm]

