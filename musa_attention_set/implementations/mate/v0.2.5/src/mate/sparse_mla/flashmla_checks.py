from __future__ import annotations

from typing import Optional

import torch

MODEL1_KV_NOPE_ROPE_BYTES = 576
MODEL1_KV_SCALE_BYTES = 8
MODEL1_KV_BYTES_PER_TOKEN = MODEL1_KV_NOPE_ROPE_BYTES + MODEL1_KV_SCALE_BYTES


def _check_last_dim_contiguous_unless_singleton(
    name: str,
    tensor: torch.Tensor,
) -> None:
    if tensor.is_contiguous() or tensor.shape[-1] == 1:
        return
    assert tensor.stride(-1) == 1, f"{name} last dimension must be contiguous"


def normalize_sparse_decode_indices(
    indices: torch.Tensor,
    batch_size: int,
    seq_len_q: int,
    num_heads_k: int,
    name: str,
) -> torch.Tensor:
    assert indices.dtype == torch.int32, f"{name} must be int32"
    _check_last_dim_contiguous_unless_singleton(name, indices)
    if indices.dim() == 3:
        assert num_heads_k == 1, f"{name} without h_k dimension requires h_k == 1"
        assert indices.shape[:2] == (batch_size, seq_len_q)
        return indices[:, :, None, :]
    assert indices.dim() == 4, f"{name} must have shape [B,S,topk] or [B,S,H_k,topk]"
    assert indices.shape[:3] == (batch_size, seq_len_q, num_heads_k)
    return indices


def require_batch_topk_length(
    topk_length: Optional[torch.Tensor],
    batch_size: int,
    name: str,
) -> Optional[torch.Tensor]:
    if topk_length is None:
        return None
    assert topk_length.dtype == torch.int32, f"{name} must be int32"
    assert topk_length.shape == (batch_size,), (
        f"{name} must have official FlashMLA shape [B], got {tuple(topk_length.shape)}"
    )
    _check_last_dim_contiguous_unless_singleton(name, topk_length)
    return topk_length


def check_model1_k_cache(k_cache: torch.Tensor, name: str = "k_cache") -> None:
    assert k_cache.dim() == 4, f"{name} must have shape [blocks, block, H_k, 584]"
    assert k_cache.shape[-1] == MODEL1_KV_BYTES_PER_TOKEN, (
        f"{name} last dimension must be {MODEL1_KV_BYTES_PER_TOKEN}"
    )
    assert k_cache.shape[2] == 1, "MODEL1 sparse decode currently supports H_k == 1"
    assert k_cache.element_size() == 1, f"{name} must be a byte-addressed FP8 cache"
    assert k_cache.stride(-1) == 1, f"{name} last dimension must be contiguous"


def model1_cache_page_views(k_cache: torch.Tensor, *, check: bool = True):
    """Return zero-copy flat MODEL1 cache views over each padded page row."""
    if check:
        check_model1_k_cache(k_cache)
    num_blocks = k_cache.shape[0]
    logical_block_bytes = k_cache.shape[1] * MODEL1_KV_BYTES_PER_TOKEN
    if num_blocks == 1:
        # A singleton page never advances stride(0), so preserve the wider load path.
        page_stride_bytes = (logical_block_bytes + 15) // 16 * 16
    else:
        page_stride_bytes = k_cache.view(torch.uint8).stride(0)
        assert page_stride_bytes > 0, (
            f"k_cache stride(0) must be positive, got {page_stride_bytes}"
        )
        assert page_stride_bytes >= logical_block_bytes, (
            "k_cache stride(0) must cover a page, "
            f"got {page_stride_bytes} < {logical_block_bytes}"
        )
    assert page_stride_bytes % 8 == 0, (
        f"k_cache page stride must be divisible by 8 bytes, got {page_stride_bytes}"
    )
    cache_bytes = torch.as_strided(
        k_cache.view(torch.uint8),
        (num_blocks, logical_block_bytes),
        (page_stride_bytes, 1),
    )
    return (
        cache_bytes.view(torch.float8_e4m3fn),
        cache_bytes.view(torch.bfloat16),
        cache_bytes,
    )
