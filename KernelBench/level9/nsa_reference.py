import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Reference implementation of Native Sparse Attention (NSA).
    Combines sliding window, compressed, and selected attention.
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 32,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        block_size: int = 64,
        window_size: int = 512,
        num_blocks: int = 16,
    ):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.window_size = window_size
        self.num_blocks = num_blocks
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_heads * 3, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, hidden_size]
        Returns:
            torch.Tensor: [batch_size, seq_len, hidden_size]
        """
        B, T, _ = x.shape
        H, HKV, D = self.num_heads, self.num_kv_heads, self.head_dim
        G = H // HKV
        
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2) # [B, H, T, D]
        k = self.k_proj(x).view(B, T, HKV, D).transpose(1, 2) # [B, HKV, T, D]
        v = self.v_proj(x).view(B, T, HKV, D).transpose(1, 2) # [B, HKV, T, D]
        g = self.g_proj(x).view(B, T, H, 3).sigmoid()
        g_cmp, g_slc, g_swa = g[..., 0], g[..., 1], g[..., 2]
        
        if G > 1:
            k = k.repeat_interleave(G, dim=1)
            v = v.repeat_interleave(G, dim=1)
            
        scale = D ** -0.5
        
        # 1. Sliding Window Attention (SWA)
        # Full attention scores with causal and window mask
        scores_swa = torch.matmul(q, k.transpose(-1, -2)) * scale
        mask = torch.tril(torch.ones(T, T, device=x.device))
        if self.window_size > 0:
            window_mask = torch.arange(T, device=x.device).unsqueeze(-1) - torch.arange(T, device=x.device) < self.window_size
            mask = mask * window_mask
        scores_swa = scores_swa.masked_fill(mask == 0, float('-inf'))
        o_swa = torch.matmul(F.softmax(scores_swa, dim=-1), v)
        
        # 2. Compressed Attention (CMP)
        # Pooling K and V in blocks
        TC = (T + self.block_size - 1) // self.block_size
        k_pad = F.pad(k, (0, 0, 0, TC * self.block_size - T))
        v_pad = F.pad(v, (0, 0, 0, TC * self.block_size - T))
        
        # [B, H, TC, BS, D] -> [B, H, TC, D]
        k_cmp = k_pad.view(B, H, TC, self.block_size, D).mean(dim=-2)
        v_cmp = v_pad.view(B, H, TC, self.block_size, D).mean(dim=-2)
        
        scores_cmp = torch.matmul(q, k_cmp.transpose(-1, -2)) * scale
        # Causal mask for compression: token t can see compressed block c if c*BS < t
        cmp_mask = torch.arange(T, device=x.device).unsqueeze(-1) >= (torch.arange(TC, device=x.device) * self.block_size + self.block_size - 1)
        scores_cmp = scores_cmp.masked_fill(cmp_mask == 0, float('-inf'))
        o_cmp = torch.matmul(F.softmax(scores_cmp, dim=-1), v_cmp)
        
        # 3. Selected Attention (SLC) - Top-k blocks selection
        # We use the compressed scores to select the top-k most important blocks for each query
        # and then perform attention over the raw tokens in those blocks.
        # This is the "Sparse" part of Native Sparse Attention.
        
        # For the reference implementation, we'll implement a simplified selected attention:
        # For each query token, pick the top num_blocks blocks from scores_cmp.
        # Then for those blocks, do attention over original tokens.
        
        # scores_cmp: [B, H, T, TC]
        _, top_block_indices = scores_cmp.topk(min(self.num_blocks, TC), dim=-1) # [B, H, T, S]
        
        o_slc = torch.zeros_like(o_swa)
        # Loop over batches and heads for selected part to keep it simple and correct in reference
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    # Selected blocks for query t
                    blocks = top_block_indices[b, h, t]
                    indices = []
                    for blk in blocks:
                        start = blk.item() * self.block_size
                        end = min(start + self.block_size, T)
                        # Only include tokens <= t
                        if start <= t:
                            indices.extend(range(start, min(end, t + 1)))
                    
                    if not indices:
                        continue
                        
                    indices = torch.tensor(indices, device=x.device)
                    q_t = q[b, h, t] * scale # [D]
                    k_t = k[b, h, indices] # [N, D]
                    v_t = v[b, h, indices] # [N, D]
                    
                    attn = F.softmax(torch.matmul(k_t, q_t), dim=0) # [N]
                    o_slc[b, h, t] = torch.matmul(attn, v_t)
                    
        # Final combination using gating coefficients
        # g: [B, T, H]
        o = (o_swa * g_swa.transpose(1, 2).unsqueeze(-1) + 
             o_cmp * g_cmp.transpose(1, 2).unsqueeze(-1) + 
             o_slc * g_slc.transpose(1, 2).unsqueeze(-1))
        
        o = o.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(o.to(x.dtype))

# Kernelbench Parameters
batch_size = 1 # Keep it small for the selection loop
seq_len = 128
hidden_size = 512
num_heads = 8
num_kv_heads = 2
head_dim = 64
block_size = 32
window_size = 64
num_blocks = 4

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size, num_heads, num_kv_heads, head_dim, block_size, window_size, num_blocks]
