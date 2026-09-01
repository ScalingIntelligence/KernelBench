from typing import List, Optional, Tuple, Union

import torch
from mate.mha_interface import (
    flash_attn_combine as mate_flash_attn_combine,
    flash_attn_varlen_func as mate_flash_attn_varlen_func,
    flash_attn_with_kvcache as mate_flash_attn_with_kvcache,
    get_scheduler_metadata as mate_get_scheduler_metadata,
)


def _resolve_sink(
    sinks: Optional[torch.Tensor] = None, s_aux: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    return sinks if sinks is not None else s_aux


def _resolve_return_softmax_lse(
    return_attn_probs: bool = False, return_softmax_lse: bool = False
) -> bool:
    return return_attn_probs or return_softmax_lse


def _resolve_block_table(
    page_table: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    return page_table if page_table is not None else block_table


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    return mate_flash_attn_combine(
        out_partial,
        lse_partial,
        out=out,
        out_dtype=out_dtype,
    )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Tuple = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 0,
    pack_gqa=None,
    deterministic: bool = False,
    sm_margin: int = 0,
    return_attn_probs: bool = False,
    return_softmax_lse: bool = False,
    sinks: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
):
    resolved_sink = _resolve_sink(sinks=sinks, s_aux=s_aux)
    resolved_softmax_lse = _resolve_return_softmax_lse(
        return_attn_probs=return_attn_probs,
        return_softmax_lse=return_softmax_lse,
    )

    return mate_flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        seqused_q=None,
        seqused_k=None,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        learnable_sink=resolved_sink,
        attention_chunk=attention_chunk,
        softcap=softcap,
        scheduler_metadata=None,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        sm_margin=sm_margin,
        return_softmax_lse=resolved_softmax_lse,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Union[Tuple, List, None] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    pack_gqa=None,
    deterministic: bool = False,
    sm_margin: int = 0,
    return_attn_probs: bool = False,
    return_softmax_lse: bool = False,
    sinks: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    resolved_sink = _resolve_sink(sinks=sinks, s_aux=s_aux)
    resolved_softmax_lse = _resolve_return_softmax_lse(
        return_attn_probs=return_attn_probs,
        return_softmax_lse=return_softmax_lse,
    )
    resolved_page_table = _resolve_block_table(
        page_table=page_table, block_table=block_table
    )
    resolved_softmax_lse = _resolve_return_softmax_lse(
        return_attn_probs=return_attn_probs,
        return_softmax_lse=return_softmax_lse,
    )

    return mate_flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=resolved_page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        learnable_sink=resolved_sink,
        attention_chunk=attention_chunk,
        softcap=softcap,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        sm_margin=sm_margin,
        return_softmax_lse=resolved_softmax_lse,
        out=out,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk: Optional[int] = 0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
    only_qv: bool = False,
):
    resolved_sink = _resolve_sink(sinks=sinks, s_aux=s_aux)

    return mate_flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        qv=qv,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        max_seqlen_q=max_seqlen_q,
        rotary_seqlens=rotary_seqlens,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        learnable_sink=resolved_sink,
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
        only_qv=only_qv,
    )


def get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    headdim,
    cache_seqlens,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size=None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    has_softcap=False,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    has_qv=False,
):
    return mate_get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        headdim=headdim,
        seqused_q=None,
        seqused_k=cache_seqlens,
        qkv_dtype=qkv_dtype,
        headdim_v=headdim_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        cu_seqlens_k_new=cu_seqlens_k_new,
        cache_leftpad=cache_leftpad,
        page_size=page_size,
        max_seqlen_k_new=max_seqlen_k_new,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,
        has_softcap=has_softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        has_qv=has_qv,
        mp_margin=sm_margin,
    )
