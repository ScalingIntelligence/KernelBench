"""Unified TileLang sparse decode entrypoint.

FlashMLA exposes sparse decode as one operation.  MODEL1 and V3.2 differ in
kv byte layout and optional extra-kv support, so the heavy TileLang factories
remain layout-specialized.  This module keeps the wrapper-facing path unified:
shape/default handling, metadata initialization, and variant dispatch live here.
"""

# ruff: noqa
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch

from ..flashmla_checks import (
    MODEL1_KV_BYTES_PER_TOKEN,
    check_model1_k_cache,
    model1_cache_page_views,
    normalize_sparse_decode_indices,
    require_batch_topk_length,
)
from .sparse_mla_v32_decode_fwd_scheduled import (
    tilelang_flashmla_interface as _decode_v32,
)
from .sparse_mla_model1_decode_fwd_scheduled import (
    sparse_mla_decode_fwd_scheduled_interface_model1 as _decode_model1,
)


MetadataGetter = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


def _model1_decode_schedule(num_heads_q: int) -> Tuple[int, int, int, int, int, int]:
    """Choose MODEL1 decode tiling and role sizes from the visible q shape."""
    if os.getenv("MATE_MODEL1_DECODE_FORCE_BM64", "0").lower() in ("1", "true", "yes"):
        return 64, 64, 256, 256, 128, 640
    if num_heads_q <= 16:
        return 16, 64, 256, 256, 128, 640
    return 64, 64, 256, 256, 128, 640


def _byte_view_k_cache(k_cache: torch.Tensor, name: str) -> torch.Tensor:
    if k_cache.dtype == torch.uint8:
        return k_cache
    if k_cache.dtype == torch.int8:
        return k_cache.view(torch.uint8)
    if k_cache.dtype == torch.float8_e4m3fn:
        return k_cache.view(torch.uint8)
    raise AssertionError(f"{name} must be uint8 or float8_e4m3fn")


def sparse_mla_decode_fwd(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    tile_scheduler_metadata: Optional[torch.Tensor],
    num_splits: Optional[torch.Tensor],
    metadata_getter: MetadataGetter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch sparse decode by compile-time layout variant."""

    batch_size, seq_len_q, num_heads_q, head_dim_q = q.shape
    _, _, num_heads_k, dim_bytes = k_cache.shape
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    k_cache = _byte_view_k_cache(k_cache, "k_cache")
    if extra_k_cache is not None:
        extra_k_cache = _byte_view_k_cache(extra_k_cache, "extra_k_cache")
    assert indices.dtype == torch.int32, "indices must be int32"
    if head_dim_q == 576:
        return _sparse_decode_v32(
            q=q,
            k_cache=k_cache,
            indices=indices,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            dim_bytes=dim_bytes,
            head_dim_v=head_dim_v,
            softmax_scale=softmax_scale,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
        )

    if head_dim_q == 512:
        return _sparse_decode_model1(
            q=q,
            k_cache=k_cache,
            indices=indices,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            dim_bytes=dim_bytes,
            head_dim_v=head_dim_v,
            softmax_scale=softmax_scale,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            metadata_getter=metadata_getter,
        )

    raise AssertionError(f"Unsupported sparse decode q.shape[-1]: {head_dim_q}")


def _sparse_decode_v32(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seq_len_q: int,
    num_heads_q: int,
    num_heads_k: int,
    dim_bytes: int,
    head_dim_v: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    tile_scheduler_metadata: Optional[torch.Tensor],
    num_splits: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    del num_heads_q
    assert extra_k_cache is None, (
        "extra_k_cache is not supported for V3.2 sparse decode"
    )
    assert extra_indices_in_kvcache is None, (
        "extra_indices_in_kvcache is not supported for V3.2 sparse decode"
    )
    assert extra_topk_length is None, (
        "extra_topk_length is not supported for V3.2 sparse decode"
    )
    assert head_dim_v == 512
    assert dim_bytes == 656
    assert num_heads_k == 1

    indices = normalize_sparse_decode_indices(
        indices, batch_size, seq_len_q, num_heads_k, "indices"
    )
    assert tile_scheduler_metadata is not None
    assert num_splits is not None

    # Zero-copy flatten from official paged cache shape to the current V3.2 kernel ABI.
    return _decode_v32(
        q,
        k_cache.view(-1, num_heads_k, dim_bytes),
        indices,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )


def _sparse_decode_model1(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seq_len_q: int,
    num_heads_q: int,
    num_heads_k: int,
    dim_bytes: int,
    head_dim_v: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    tile_scheduler_metadata: Optional[torch.Tensor],
    num_splits: Optional[torch.Tensor],
    metadata_getter: MetadataGetter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim_v == 512, "MODEL1 sparse decode only supports head_dim_v == 512"
    assert q.shape[-1] == 512, "MODEL1 sparse decode requires q.shape[-1] == 512"
    check_model1_k_cache(k_cache)
    assert dim_bytes == MODEL1_KV_BYTES_PER_TOKEN

    indices = normalize_sparse_decode_indices(
        indices, batch_size, seq_len_q, num_heads_k, "indices"
    )
    topk = indices.shape[-1]
    assert topk % 64 == 0, "MODEL1 sparse decode requires topk to be a multiple of 64"

    k_cache_nope, k_cache_rope, k_cache_scales = model1_cache_page_views(
        k_cache, check=False
    )
    extra_indices = None
    extra_k_cache_nope = None
    extra_k_cache_rope = None
    extra_k_cache_scales = None
    if extra_k_cache is not None:
        assert extra_indices_in_kvcache is not None, (
            "extra_indices_in_kvcache must be provided with extra_k_cache"
        )
        check_model1_k_cache(extra_k_cache, "extra_k_cache")
        assert extra_k_cache.shape[2] == num_heads_k
        extra_indices = normalize_sparse_decode_indices(
            extra_indices_in_kvcache,
            batch_size,
            seq_len_q,
            num_heads_k,
            "extra_indices_in_kvcache",
        )
        extra_topk = extra_indices.shape[-1]
        assert extra_topk % 64 == 0, (
            "MODEL1 sparse decode requires extra_topk to be a multiple of 64"
        )
        extra_k_cache_nope, extra_k_cache_rope, extra_k_cache_scales = (
            model1_cache_page_views(extra_k_cache, check=False)
        )
    else:
        assert extra_indices_in_kvcache is None, (
            "extra_indices_in_kvcache requires extra_k_cache"
        )
        assert extra_topk_length is None, "extra_topk_length requires extra_k_cache"

    if tile_scheduler_metadata is None or num_splits is None:
        assert tile_scheduler_metadata is None and num_splits is None, (
            "tile_scheduler_metadata and num_splits must be provided together"
        )
        metadata_topk_length = require_batch_topk_length(
            topk_length, batch_size, "topk_length"
        )
        metadata_extra_topk_length = require_batch_topk_length(
            extra_topk_length, batch_size, "extra_topk_length"
        )
        tile_scheduler_metadata, num_splits = metadata_getter(
            cache_seqlens=None,
            num_q_tokens_per_head_k=seq_len_q * num_heads_q // num_heads_k,
            num_heads_k=num_heads_k,
            num_heads_q=num_heads_q,
            is_fp8_kvcache=True,
            topk=topk,
            extra_topk=extra_indices.shape[-1] if extra_indices is not None else None,
            q=q,
            bs=batch_size,
            topk_length=metadata_topk_length,
            extra_topk_length=metadata_extra_topk_length,
        )

    (
        block_m,
        block_i,
        consumer0_threads,
        consumer1_threads,
        producer_threads,
        threads,
    ) = _model1_decode_schedule(num_heads_q)

    return _decode_model1(
        q,
        k_cache_nope,
        k_cache_rope,
        k_cache_scales,
        indices,
        tile_scheduler_metadata,
        num_splits,
        extra_kv_nope=extra_k_cache_nope,
        extra_kv_rope=extra_k_cache_rope,
        extra_kv_scales=extra_k_cache_scales,
        extra_indices=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        sm_scale=softmax_scale,
        attn_sink=attn_sink,
        d_v=head_dim_v,
        block_m=block_m,
        block_i=block_i,
        threads=threads,
        consumer0_threads=consumer0_threads,
        consumer1_threads=consumer1_threads,
        producer_threads=producer_threads,
        page_block_size=k_cache.shape[1],
        extra_page_block_size=(
            extra_k_cache.shape[1] if extra_k_cache is not None else None
        ),
        _cache_strides_prepared=True,
    )
