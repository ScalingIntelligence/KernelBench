# ruff: noqa
"""Unified TileLang sparse prefill entrypoint."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .sparse_mla_v32_fwd_pipelined import (
    tilelang_sparse_mla_prefill_fwd_interface as _prefill_v32,
)
from .sparse_mla_model1_fwd_pipelined import (
    sparse_mla_fwd_interface_model1 as _prefill_model1,
)


_MODEL1_PERSISTENT_MAX_QUERY_TOKENS = 65535


def _use_model1_persistence(
    seq_len_q: int,
    num_heads_q: int,
    topk_length: Optional[torch.Tensor],
) -> bool:
    # Long H64 variable-work loops can stall while carrying barrier phases.
    return not (
        num_heads_q == 64
        and seq_len_q > _MODEL1_PERSISTENT_MAX_QUERY_TOKENS
        and topk_length is not None
    )


def sparse_mla_prefill_fwd(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch sparse prefill by compile-time layout variant."""

    assert d_v == 512, "sparse prefill only supports d_v == 512"
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert kv.dtype == torch.bfloat16, "kv must be bfloat16"
    assert indices.dtype == torch.int32, "indices must be int32"
    seq_len_q, num_heads_q, head_dim_q = q.shape
    _, h_k, _ = kv.shape
    assert head_dim_q == kv.shape[-1]
    assert seq_len_q == indices.shape[0]
    assert h_k == indices.shape[1]
    assert h_k == 1

    if head_dim_q == 512:
        topk = indices.shape[-1]
        assert topk % 64 == 0, (
            "MODEL1 sparse prefill requires topk to be a multiple of 64"
        )
        return _prefill_model1(
            q,
            kv,
            indices,
            topk_length=topk_length,
            sm_scale=sm_scale,
            attn_sink=attn_sink,
            d_v=d_v,
            return_max_logits=True,
            is_persistence=_use_model1_persistence(
                seq_len_q,
                num_heads_q,
                topk_length,
            ),
        )

    if head_dim_q == 576:
        return _prefill_v32(
            q,
            kv,
            indices,
            sm_scale,
            topk_length=topk_length,
            attn_sink=attn_sink,
            return_max_logits=True,
        )

    raise AssertionError(f"Unsupported sparse prefill q.shape[-1]: {head_dim_q}")
