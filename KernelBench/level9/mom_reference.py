import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    MoM (Mixture-of-Memories) Attention - A reference implementation using pure PyTorch.
    
    MoM combines:
    1. Top-k routing: Each token selects top-k memory "experts"
    2. Gated Delta Rule: A linear attention variant with exponential decay and delta updates
    3. Transform/Reconstruct: Reorganize tokens by memory slot, process, then scatter back
    
    The Gated Delta Rule recurrence for each memory:
        h[t] = h[t-1] * exp(g[t]) + k[t] @ (beta[t] * (v[t] - h[t-1].T @ k[t])).T
        o[t] = h[t].T @ q[t]
    
    Where h is a [K, V] outer-product state matrix.
    
    Based on: "MoM: Linear Sequence Modeling with Mixture-of-Memories"
    https://arxiv.org/abs/2502.13685
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 256,
        num_heads: int = 4,
        expand_v: float = 2.0,
        num_memories: int = 8,
        topk: int = 2,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.expand_v = expand_v
        self.num_memories = num_memories
        self.topk = topk
        
        self.key_dim = num_heads * head_dim
        self.value_dim = int(self.key_dim * expand_v)
        self.head_v_dim = int(head_dim * expand_v)
        
        # Router gate
        self.gate = nn.Linear(hidden_size, num_memories, bias=False)
        
        # Projections (shared across memories for simplicity)
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)  # beta projection
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)  # gate projection
        
        # Output projections
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)  # output gate
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        
        # Learnable decay parameters (A_log and dt_bias)
        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        
        # RMSNorm for output
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch_size, seq_len), 1=valid, 0=padding
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # ============================================
        # Step 1: Top-k Routing
        # ============================================
        router_logits = self.gate(x)  # [B, T, num_memories]
        scores = F.softmax(router_logits, dim=-1)
        routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)  # [B, T, topk]
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)  # Normalize
        
        # Create routing mask: [B, T, num_memories]
        routing_mask = torch.zeros(batch_size, seq_len, self.num_memories, device=x.device, dtype=torch.bool)
        routing_mask.scatter_(-1, selected_memories, True)
        
        if attention_mask is not None:
            # Mask out padding tokens
            routing_mask = routing_mask & attention_mask.unsqueeze(-1).bool()
        
        # ============================================
        # Step 2: Transform - Reorganize tokens by memory
        # ============================================
        # For each memory, gather the tokens routed to it
        memory_outputs = []
        memory_indices = []  # Track which tokens went to which memory
        
        for mem_idx in range(self.num_memories):
            # Find tokens routed to this memory
            # For simplicity, process per-batch
            batch_outputs = []
            batch_indices = []
            
            for b in range(batch_size):
                # Get mask for this batch and memory
                mem_mask = routing_mask[b, :, mem_idx]  # [T]
                token_indices = torch.where(mem_mask)[0]  # Indices of tokens routed to this memory
                
                if len(token_indices) == 0:
                    batch_outputs.append(torch.zeros(0, self.value_dim, device=x.device, dtype=x.dtype))
                    batch_indices.append(token_indices)
                    continue
                
                # Gather tokens for this memory
                tokens = x[b, token_indices]  # [num_tokens, hidden_size]
                
                # ============================================
                # Step 3: Gated Delta Rule for this memory
                # ============================================
                mem_output = self._gated_delta_rule(tokens)  # [num_tokens, value_dim]
                
                batch_outputs.append(mem_output)
                batch_indices.append(token_indices)
            
            memory_outputs.append(batch_outputs)
            memory_indices.append(batch_indices)
        
        # ============================================
        # Step 4: Reconstruct - Scatter back and mix
        # ============================================
        output = torch.zeros(batch_size, seq_len, self.value_dim, device=x.device, dtype=x.dtype)
        
        for mem_idx in range(self.num_memories):
            for b in range(batch_size):
                indices = memory_indices[mem_idx][b]
                if len(indices) == 0:
                    continue
                    
                mem_out = memory_outputs[mem_idx][b]  # [num_tokens, value_dim]
                
                # Get routing weights for these tokens to this memory
                # Find which topk slot this memory corresponds to
                mem_weights = torch.zeros(len(indices), device=x.device, dtype=x.dtype)
                for i, idx in enumerate(indices):
                    # Find the weight for this memory in the topk selection
                    for k in range(self.topk):
                        if selected_memories[b, idx, k] == mem_idx:
                            mem_weights[i] = routing_weights[b, idx, k]
                            break
                
                # Weighted scatter-add
                output[b, indices] += mem_out * mem_weights.unsqueeze(-1)
        
        # ============================================
        # Step 5: Output projection with gating
        # ============================================
        # Reshape for head-wise processing
        output = output.view(batch_size, seq_len, self.num_heads, self.head_v_dim)
        
        # Output gate
        g = self.g_proj(x).view(batch_size, seq_len, self.num_heads, self.head_v_dim)
        
        # Gated RMSNorm
        output = self._gated_rms_norm(output, g, self.o_norm_weight)
        
        # Final projection
        output = output.view(batch_size, seq_len, self.value_dim)
        output = self.o_proj(output)
        
        return output
    
    def _gated_delta_rule(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply Gated Delta Rule attention to a sequence of tokens.
        
        The recurrence:
            h[t] = h[t-1] * exp(g[t]) + k[t] @ (beta[t] * (v[t] - h[t-1].T @ k[t])).T
            o[t] = h[t].T @ q[t]
        
        Args:
            tokens: [num_tokens, hidden_size]
            
        Returns:
            outputs: [num_tokens, value_dim]
        """
        num_tokens = tokens.shape[0]
        if num_tokens == 0:
            return tokens.new_zeros(0, self.value_dim)
        
        # Project to Q, K, V
        q = self.q_proj(tokens)  # [T, key_dim]
        k = self.k_proj(tokens)  # [T, key_dim]
        v = self.v_proj(tokens)  # [T, value_dim]
        
        # Compute beta (sigmoid-scaled) and gate g
        beta = self.b_proj(tokens).sigmoid()  # [T, num_heads]
        a = self.a_proj(tokens)  # [T, num_heads]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [T, num_heads]
        
        # Apply SiLU activation (as in the original implementation with conv)
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)
        
        # Reshape to multi-head: [T, num_heads, head_dim]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_heads, self.head_dim)
        v = v.view(num_tokens, self.num_heads, self.head_v_dim)
        
        # L2 normalize Q and K (as used in the kernel)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # Scale Q
        scale = self.head_dim ** -0.5
        q = q * scale
        
        # Initialize state: h is [num_heads, head_dim, head_v_dim]
        h = torch.zeros(self.num_heads, self.head_dim, self.head_v_dim, 
                       device=tokens.device, dtype=torch.float32)
        
        outputs = []
        
        for t in range(num_tokens):
            q_t = q[t]  # [num_heads, head_dim]
            k_t = k[t]  # [num_heads, head_dim]
            v_t = v[t]  # [num_heads, head_v_dim]
            beta_t = beta[t]  # [num_heads]
            g_t = g[t]  # [num_heads]
            
            # Apply decay: h = h * exp(g)
            # g is negative (decay), so exp(g) < 1
            decay = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # [num_heads, 1, 1]
            h = h * decay
            
            # Delta rule update:
            # prediction = h.T @ k  =>  [num_heads, head_v_dim]
            # For each head: prediction[h] = h[h].T @ k_t[h] = einsum('kv,k->v', h[h], k_t[h])
            prediction = torch.einsum('hkv,hk->hv', h, k_t)  # [num_heads, head_v_dim]
            
            # v_new = beta * (v - prediction)
            v_new = beta_t.unsqueeze(-1) * (v_t.float() - prediction)  # [num_heads, head_v_dim]
            
            # Update state: h = h + outer(k, v_new)
            # For each head: h[h] += k_t[h][:, None] @ v_new[h][None, :]
            h = h + torch.einsum('hk,hv->hkv', k_t.float(), v_new)  # [num_heads, head_dim, head_v_dim]
            
            # Output: o = h.T @ q => [num_heads, head_v_dim]
            o_t = torch.einsum('hkv,hk->hv', h, q_t)  # [num_heads, head_v_dim]
            
            outputs.append(o_t)
        
        # Stack outputs: [T, num_heads, head_v_dim]
        outputs = torch.stack(outputs, dim=0)
        
        # Reshape to [T, value_dim]
        outputs = outputs.view(num_tokens, self.value_dim).to(tokens.dtype)
        
        return outputs
    
    def _gated_rms_norm(self, x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Gated RMSNorm: RMSNorm(x) * sigmoid(g)
        
        Args:
            x: [B, T, H, D]
            g: [B, T, H, D] gate
            weight: [D] norm weight
            
        Returns:
            output: [B, T, H, D]
        """
        # RMSNorm
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        x_norm = (x.float() / rms) * weight
        
        # Gate
        output = x_norm * torch.sigmoid(g.float())
        
        return output.to(x.dtype)


# Problem dimensions
batch_size = 4
seq_len = 512
hidden_size = 2048
head_dim = 256
num_heads = 4
expand_v = 2.0
num_memories = 8
topk = 2


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]


def get_init_inputs():
    return [hidden_size, head_dim, num_heads, expand_v, num_memories, topk]

