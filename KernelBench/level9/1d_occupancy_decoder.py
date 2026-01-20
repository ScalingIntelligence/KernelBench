import math
import torch
import torch.nn as nn


def init_linear(module, embed_dim: int):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=math.sqrt(1.0 / embed_dim))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class MLPEmbedder(nn.Module):
    """MLP with SiLU activation for query embedding."""
    
    def __init__(self, in_dim: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, embed_dim, bias=bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.apply(lambda m: init_linear(m, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class LayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor):
        y = super().forward(input.float())
        return y.type_as(input)


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim=None,
        kv_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        q_dim = q_dim or embed_dim
        kv_dim = kv_dim or embed_dim

        self.c_q = nn.Linear(q_dim, embed_dim, bias=bias)
        self.c_k = nn.Linear(kv_dim, embed_dim, bias=bias)
        self.c_v = nn.Linear(kv_dim, embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.num_heads = num_heads

    def forward(self, x, c, attn_mask=None, is_causal: bool = False):
        q, k = self.c_q(x), self.c_k(c)
        v = self.c_v(c)

        b, l, d = q.shape
        s = k.shape[1]

        q = q.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(b, s, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(b, s, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=(attn_mask is not None) and is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(b, l, d)
        y = self.c_proj(y)
        return y


class OneDOccupancyDecoder(nn.Module):
    """
    Simplified 1DOccupancyDecoder forward pass.
    - 250k queries attending to 1k KV tokens
    - MLP with SiLU activation for query projection
    - Cross-attention with LayerNorm
    - Output projection
    
    Args:
        q_in_dim: Input dimension for queries
        width: The width of the intermediate layers.
        num_heads: The number of attention heads for the cross-attention layer.
        out_features: Output dimension
        eps: Epsilon for layer normalization
    """
    
    def __init__(
        self,
        q_in_dim: int = 3,
        width: int = 768,
        num_heads: int = 12,
        out_features: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()
        
        self.query_in = MLPEmbedder(q_in_dim, width)
        self.attn = CrossAttention(
            embed_dim=width,
            num_heads=num_heads,
            bias=True,
        )
        self.ln = LayerNorm(width, elementwise_affine=False, eps=eps)
        self.out_proj = nn.Linear(width, out_features)

    def forward(self, queries: torch.Tensor, latents: torch.Tensor):
        """
        Forward pass.
        
        Args:
            queries: Input queries of shape [batch_size, num_queries, q_in_dim]
            latents: Input latents of shape [batch_size, num_latents, width]
            
        Returns:
            Output tensor of shape [batch_size, num_queries, out_features]
        """
        q = self.query_in(queries)
        x = self.attn(q, latents)
        x = self.out_proj(self.ln(x))
        return x


class Model(nn.Module):
    """Reference implementation that wraps `OneDOccupancyDecoder`."""

    def __init__(self, q_in_dim: int, width: int, num_heads: int) -> None:
        super().__init__()
        self.decoder = OneDOccupancyDecoder(
            q_in_dim=q_in_dim,
            width=width,
            num_heads=num_heads,
            out_features=1,
        )

    def forward(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(queries, latents)


# Problem configuration
batch_size = 1
num_queries = 8192
num_latents = 1024
width = 768
num_heads = 12
q_in_dim = 3


def get_inputs():
    queries = torch.randn(batch_size, num_queries, q_in_dim)
    latents = torch.randn(batch_size, num_latents, width)
    return [queries, latents]


def get_init_inputs():
    return [q_in_dim, width, num_heads]

