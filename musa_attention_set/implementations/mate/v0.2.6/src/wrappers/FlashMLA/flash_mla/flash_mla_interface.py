from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from mate.flashmla import (
    flash_mla_sparse_fwd as mate_flash_mla_sparse_fwd,
    flash_mla_sparse_fwd_pack8 as mate_flash_mla_sparse_fwd_pack8,
    flash_mla_with_kvcache as mate_flash_mla_with_kvcache,
    get_mla_metadata as mate_get_mla_metadata,
)


@dataclass
class FlashMLASchedMeta:
    """Tile scheduler metadata container compatible with FlashMLA."""

    @dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int
        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]
        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False
    config: Optional[Config] = None
    tile_scheduler_metadata: Optional[torch.Tensor] = None
    num_splits: Optional[torch.Tensor] = None
    metadata_topk: Optional[int] = None
    metadata_extra_topk: Optional[int] = None
    metadata_has_topk_length: bool = False
    metadata_has_extra_topk_length: bool = False


def get_mla_metadata(*args, **kwargs) -> Tuple[FlashMLASchedMeta, None]:
    sched_meta = FlashMLASchedMeta()
    if args or kwargs:
        tile_scheduler_metadata, num_splits = mate_get_mla_metadata(*args, **kwargs)
        sched_meta.tile_scheduler_metadata = tile_scheduler_metadata
        sched_meta.num_splits = num_splits
        sched_meta.metadata_topk = kwargs.get(
            "topk", args[5] if len(args) > 5 else None
        )
        sched_meta.metadata_extra_topk = kwargs.get(
            "extra_topk", args[6] if len(args) > 6 else None
        )
        topk_length = kwargs.get("topk_length", args[9] if len(args) > 9 else None)
        extra_topk_length = kwargs.get(
            "extra_topk_length", args[10] if len(args) > 10 else None
        )
        sched_meta.metadata_has_topk_length = topk_length is not None
        sched_meta.metadata_has_extra_topk_length = extra_topk_length is not None
    return sched_meta, None


def _check_sched_meta(
    sched_meta: FlashMLASchedMeta,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    causal: bool,
    is_fp8_kvcache: bool,
    topk: Optional[int],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
) -> None:
    extra_k_page_block_size = (
        extra_k_cache.shape[1] if extra_k_cache is not None else None
    )
    extra_topk = (
        extra_indices_in_kvcache.shape[-1]
        if extra_indices_in_kvcache is not None
        else None
    )

    if not sched_meta.have_initialized:
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k_cache.shape[1],
            k_cache.shape[2],
            causal,
            is_fp8_kvcache,
            topk,
            extra_k_page_block_size,
            extra_topk,
        )
        return

    helper_msg = (
        " Your input arguments are inconsistent with sched_meta. Please make sure the input "
        "arguments are consistent across different invocations of flash_mla_with_kvcache on "
        "the same sched_meta."
    )
    assert sched_meta.config is not None
    assert sched_meta.config.b == q.shape[0], (
        "sched_meta.config.b must equal batch_size." + helper_msg
    )
    assert sched_meta.config.s_q == q.shape[1], (
        "sched_meta.config.s_q must equal seq_len_q." + helper_msg
    )
    assert sched_meta.config.h_q == q.shape[2], (
        "sched_meta.config.h_q must equal num_heads_q." + helper_msg
    )
    assert sched_meta.config.page_block_size == k_cache.shape[1], (
        "sched_meta.config.page_block_size must equal page_block_size." + helper_msg
    )
    assert sched_meta.config.h_k == k_cache.shape[2], (
        "sched_meta.config.h_k must equal num_heads_k." + helper_msg
    )
    assert sched_meta.config.causal == causal, (
        "sched_meta.config.causal must equal causal." + helper_msg
    )
    assert sched_meta.config.is_fp8_kvcache == is_fp8_kvcache, (
        "sched_meta.config.is_fp8_kvcache must equal is_fp8_kvcache." + helper_msg
    )
    assert sched_meta.config.topk == topk, (
        "sched_meta.config.topk must equal the last dim of indices." + helper_msg
    )
    assert sched_meta.config.extra_page_block_size == extra_k_page_block_size, (
        "sched_meta.config.extra_page_block_size must equal the page_block_size of extra_k_cache."
        + helper_msg
    )
    assert sched_meta.config.extra_topk == extra_topk, (
        "sched_meta.config.extra_topk must equal the last dim of extra_indices_in_kvcache."
        + helper_msg
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sched_meta = tile_scheduler_metadata
    if not isinstance(sched_meta, FlashMLASchedMeta):
        raise AssertionError(
            "tile_scheduler_metadata must be of type FlashMLASchedMeta"
        )
    if num_splits is not None:
        raise AssertionError("num_splits must be None")

    topk = indices.shape[-1] if indices is not None else None
    extra_topk = (
        extra_indices_in_kvcache.shape[-1]
        if extra_indices_in_kvcache is not None
        else None
    )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    _check_sched_meta(
        sched_meta,
        q,
        k_cache,
        causal,
        is_fp8_kvcache,
        topk,
        extra_k_cache,
        extra_indices_in_kvcache,
    )

    metadata_needs_refresh = (
        sched_meta.tile_scheduler_metadata is None
        or sched_meta.num_splits is None
        or sched_meta.metadata_topk != topk
        or sched_meta.metadata_extra_topk != extra_topk
        or sched_meta.metadata_has_topk_length != (topk_length is not None)
        or sched_meta.metadata_has_extra_topk_length != (extra_topk_length is not None)
    )

    if metadata_needs_refresh:
        num_q_tokens_per_head_k = q.shape[1] * q.shape[2] // k_cache.shape[2]
        new_tile_scheduler_metadata, new_num_splits = mate_get_mla_metadata(
            cache_seqlens=cache_seqlens,
            num_q_tokens_per_head_k=num_q_tokens_per_head_k,
            num_heads_k=k_cache.shape[2],
            num_heads_q=q.shape[2],
            is_fp8_kvcache=is_fp8_kvcache,
            topk=topk,
            extra_topk=extra_topk,
            q=q,
            bs=q.shape[0],
            topk_length=topk_length.contiguous() if topk_length is not None else None,
            extra_topk_length=extra_topk_length.contiguous()
            if extra_topk_length is not None
            else None,
        )
        sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata
        sched_meta.num_splits = new_num_splits
        sched_meta.metadata_topk = topk
        sched_meta.metadata_extra_topk = extra_topk
        sched_meta.metadata_has_topk_length = topk_length is not None
        sched_meta.metadata_has_extra_topk_length = extra_topk_length is not None

    return mate_flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=sched_meta.tile_scheduler_metadata,
        num_splits=sched_meta.num_splits,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return mate_flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )


def flash_mla_sparse_fwd_pack8(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    row_masks: Optional[torch.Tensor] = None,
    causal_window: int = 0,
    compressed_kv_len: int = 0,
    compress_ratio: int = 1,
    pack_metadata: Optional[object] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return mate_flash_mla_sparse_fwd_pack8(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        row_masks=row_masks,
        causal_window=causal_window,
        compressed_kv_len=compressed_kv_len,
        compress_ratio=compress_ratio,
        pack_metadata=pack_metadata,
    )
