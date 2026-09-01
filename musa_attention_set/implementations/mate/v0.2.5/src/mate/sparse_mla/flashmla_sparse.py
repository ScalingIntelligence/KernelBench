from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from .tilelang.sparse_mla_decode import sparse_mla_decode_fwd
from .tilelang.sparse_mla_model1_fwd_pack import (
    sparse_mla_fwd_interface_model1_pack,
)
from .tilelang.sparse_mla_prefill import sparse_mla_prefill_fwd


MetadataGetter = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


def _try_model1_seq_pack_sparse_prefill(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Use private MODEL1 seq-pack kernels when the public shape is exact."""
    if d_v != 512 or topk_length is None:
        return None
    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        return None
    seq_len, heads, head_dim = q.shape
    seq_len_kv, kv_group, kv_dim = kv.shape
    if heads == 8:
        token_pack = 8
    elif heads == 16:
        token_pack = 4
    elif heads == 32:
        token_pack = 2
    else:
        return None

    if (
        q.dtype != torch.bfloat16
        or kv.dtype != torch.bfloat16
        or indices.dtype != torch.int32
        or topk_length.dtype != torch.int32
        or not q.is_contiguous()
        or not kv.is_contiguous()
        or head_dim != 512
        or kv_group != 1
        or kv_dim != 512
        or seq_len == 0
        or seq_len % token_pack != 0
        or indices.shape[0] != seq_len
        or indices.shape[1] != 1
        or indices.shape[-1] % 64 != 0
        or topk_length.shape != (seq_len,)
    ):
        return None

    if seq_len_kv == seq_len:
        compressed_kv_len = 0
        compress_ratio = 1
    elif seq_len_kv > seq_len:
        compressed_kv_len = seq_len_kv - seq_len
        if compressed_kv_len <= 0 or seq_len % compressed_kv_len != 0:
            return None
        compress_ratio = seq_len // compressed_kv_len
        if compress_ratio != 128:
            return None
    else:
        return None

    packed_indices = indices[token_pack - 1 :: token_pack].contiguous()
    packed_topk_length = topk_length[token_pack - 1 :: token_pack].contiguous()
    return sparse_mla_fwd_interface_model1_pack(
        q=q,
        kv=kv,
        indices=packed_indices,
        topk_length=packed_topk_length,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        d_v=d_v,
        return_max_logits=True,
        causal_window=128,
        compressed_kv_len=compressed_kv_len,
        compress_ratio=compress_ratio,
        token_pack=token_pack,
    )


def flashmla_sparse_prefill(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single FlashMLA sparse-prefill dispatcher for MODEL1 and V3.2."""
    pack_out = _try_model1_seq_pack_sparse_prefill(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    if pack_out is not None:
        return pack_out

    return sparse_mla_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )


def flashmla_sparse_decode(
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
    """Single FlashMLA sparse-decode dispatcher for MODEL1 and V3.2."""
    return sparse_mla_decode_fwd(
        q=q,
        k_cache=k_cache,
        indices=indices,
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
