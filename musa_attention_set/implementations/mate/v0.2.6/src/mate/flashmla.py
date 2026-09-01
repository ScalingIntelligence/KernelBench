import functools
from typing import Optional, Tuple

import torch

from mate.api_logging import mate_api
from mate.jit.mla_ops import get_mla_ops_module
from mate.jit.mubin.flash_mla import flash_mla_asm_mubin
from mate.jit.runtime import ffi_to_torch
from .sparse_mla.flashmla_sparse import (
    flashmla_sparse_decode,
    flashmla_sparse_prefill,
)
from .sparse_mla.tilelang.sparse_mla_model1_fwd_pack import (
    sparse_mla_fwd_interface_model1_pack,
)
from .execution_context import raise_complete_if_dry_run


@functools.cache
def _get_module():
    return get_mla_ops_module()


def _allocate_flashmla_outputs(
    q: torch.Tensor, head_dim_v: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(
        (*q.shape[:-1], head_dim_v),
        dtype=q.dtype,
        device=q.device,
    )
    softmax_lse = torch.empty(
        q.shape[:-1],
        dtype=torch.float32,
        device=q.device,
    )
    return out, softmax_lse


def _prepare_mla_query_input(
    x: torch.Tensor, *, require_seq_dense: bool
) -> torch.Tensor:
    # Keep Python-side materialization aligned with the actual FFI layout contract.
    if x.stride(-1) != 1:
        return x.contiguous()
    if require_seq_dense and x.dim() == 4:
        if x.stride(1) != x.shape[-2] * x.stride(2):
            return x.contiguous()
    return x


def _dispatch_mla_metadata(
    cache_seqlens: Optional[torch.Tensor],
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
    extra_topk: Optional[int] = None,
    q: Optional[torch.Tensor] = None,
    bs: Optional[int] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    *,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch MLA metadata into new or caller-owned output tensors."""
    if (tile_scheduler_metadata is None) != (num_splits is None):
        raise ValueError(
            "tile_scheduler_metadata and num_splits must be provided together"
        )

    func = _get_module().get_function("get_mla_decoding_metadata")
    raise_complete_if_dry_run()
    return ffi_to_torch(
        func(
            cache_seqlens,
            num_q_tokens_per_head_k,
            num_heads_k,
            num_heads_q,
            is_fp8_kvcache,
            topk,
            tile_scheduler_metadata,
            num_splits,
            q,
            bs,
            topk_length,
            extra_topk_length,
            extra_topk,
        )
    )


@mate_api
def get_mla_metadata(
    cache_seqlens: Optional[torch.Tensor],
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
    extra_topk: Optional[int] = None,
    q: Optional[torch.Tensor] = None,
    bs: Optional[int] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Get metadata for MLA decoding.

    Parameters
    ----------
    cache_seqlens: Tensor
        The sequence lengths of the KV cache with shape ``(batch_size)``
    num_q_tokens_per_head_k: int
        Equals to num_q_tokens_per_q_seq * num_heads_q // num_heads_k.
    num_heads_k: int
        The number of k heads.
    num_heads_q: Optional[int]
        The number of q heads. This argument is optional when sparse attention is not enabled
    is_fp8_kvcache: bool
        Whether the k_cache and v_cache are in fp8 format.
    topk: Optional[int]
        If not None, sparse attention will be enabled, and only tokens in the `indices` array passed to `flash_mla_with_kvcache_sm90` will be attended to.
    extra_topk: Optional[int]
        Optional sparse extra-KV topk. This mirrors FlashMLA sparse decode
        metadata semantics without folding extra work into `topk`.
    topk_length: Optional[Tensor]
        Optional per-batch sparse workload lengths for MODEL1 scheduled decode metadata.
        If extra_topk_length is provided, both lengths are summed in the C++ metadata kernel.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple of two tensors:

        * tile_scheduler_metadata, shape ``(num_sm_parts, TileSchedulerMetaDataSize)``
        * num_splits, shape ``(batch_size + 1)``
    """
    return _dispatch_mla_metadata(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q,
        is_fp8_kvcache,
        topk,
        extra_topk,
        q,
        bs,
        topk_length,
        extra_topk_length,
    )


@mate_api
def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by exp(lse) / (exp(lse) + exp(attn_sink)).
            +-inf in attn_sink will be handled normally (i.e., -inf has no effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32. If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]], ignoring later k/v tokens (even if provided in indices).
            In extremely rare cases (topk_length provided, there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token containing NaN), operator output will contain NaN, so please avoid this situation.
    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    assert d_v == 512, "sprase prefill only support d_v 512"
    seq_len_q, _, head_dim_q = q.shape
    _, h_k, _ = kv.shape
    _, _, topk = indices.shape
    assert head_dim_q == kv.shape[-1]
    assert seq_len_q == indices.shape[0]
    assert h_k == indices.shape[1]
    assert h_k == 1

    return flashmla_sparse_prefill(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )


@mate_api
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
    """Experimental MODEL1 H=8 pack_s=8 bf16 sparse-prefill entrypoint.

    This API is intentionally separate from the generic sparse prefill wrapper so
    callers can opt into the H8 pack experiment without changing generic sparse
    prefill semantics.
    """
    assert d_v == 512, "pack8 sparse prefill only supports d_v == 512"
    return sparse_mla_fwd_interface_model1_pack(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=topk_length,
        row_masks=row_masks,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        d_v=d_v,
        return_max_logits=True,
        causal_window=causal_window,
        compressed_kv_len=compressed_kv_len,
        compress_ratio=compress_ratio,
        pack_metadata=pack_metadata,
        token_pack=8,
    )


@mate_api
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
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
    """
    Mla forward with kv cache

    Parameters
    ----------
    q : Tensor
        The query tensor with shape ``(batch_size, seq_len_q, num_heads_q, head_dim)``.
    k_cache: Tensor
        The compressed kv cache tensor with shape ``(num_blocks, page_block_size, num_heads_k, head_dim)``.
    block_table: Tensor
        The page table with shape ``(batch_size, max_num_blocks_per_seq)``.
    cache_seqlens: Tensor
        The sequence lengths of the ckv cache wtih shape ``(batch_size)``.
    head_dim_v: int
        Head dimension of v.
    tile_scheduler_metadata: Tensor
        The scheduler metadata with shape ``(num_sm_parts, TileSchedulerMetaDataSize)``, returned by get_mla_metadata.
    num_splits: Tensor
        The num_splits tensor with shape ``(batch_size + 1)``, returned by get_mla_metadata.
    softmax_scale: Optional[float]
        The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
    causal: bool
        Whether to apply causal attention mask.
    is_fp8_kvcache: bool
        Whether the k_cache and v_cache are in fp8 format. For the format of FP8 KV cache, please refer to README.md
    indices: Optinal[Tensor]
        The token indices tensor with shape ``(batch_size, seq_len_q, topk)``.
        If not None, sparse attention will be enabled, and only tokens in the `indices` array will be attended to.
        Invalid indices should be set to -1 or numbers >= total_seq_len_kv. For details about how to set up `indices`, please refer to README.md.
    Notes
    -----
    For DeepSeek V3, DeepSeek V3.1, and DeepSeek V3.2, ``head_dim`` should be
    576 while ``head_dim_v`` should be 512.

    In FP8 + sparse mode, each token's KV cache is 656 bytes. ``k_cache`` has
    shape ``(num_blocks, page_block_size, num_heads_k, head_dim)``, and
    ``num_heads_k`` must be 1. The first 512 bytes contain the quantized NoPE
    part with 512 ``float8_e4m3`` values. The next 16 bytes contain four
    ``float32`` scale factors, one for each group of 128
    ``float8_e4m3`` values. The final 128 bytes contain the RoPE part with 64
    ``bfloat16`` values; this part is left unquantized for accuracy.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple of two tensors:

        * out, shape ``(batch_size, seq_len_q, num_heads_q, head_dim_v)``.
        * softmax_lse, shape ``(batch_size, num_heads_q, seq_len_q)``.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if indices is not None:
        assert not causal, "causal must be `false` if sparse attention is enabled."
        assert is_fp8_kvcache, "sparse mla decode only support fp8 kvcache"
        dsa_backend = "tilelang"
        if dsa_backend == "tilelang":
            # decode fp8 kv cache
            return flashmla_sparse_decode(
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
                metadata_getter=get_mla_metadata,
            )
        else:
            assert False, "Unsupported DSA backend"

    assert attn_sink is None, "attn_sink is only supported for sparse MODEL1 decode"
    assert extra_k_cache is None, (
        "extra_k_cache is only supported for sparse MODEL1 decode"
    )
    assert extra_indices_in_kvcache is None, (
        "extra_indices_in_kvcache is only supported for sparse MODEL1 decode"
    )
    assert topk_length is None, "topk_length is only supported for sparse MODEL1 decode"
    assert extra_topk_length is None, (
        "extra_topk_length is only supported for sparse MODEL1 decode"
    )

    assert head_dim_v == 512

    q_nope = q[:, :, :, :head_dim_v]
    q_pe = q[:, :, :, head_dim_v:]
    should_run_with_asm = q.shape[-2] == 128

    q_nope = _prepare_mla_query_input(q_nope, require_seq_dense=not should_run_with_asm)
    q_pe = _prepare_mla_query_input(q_pe, require_seq_dense=not should_run_with_asm)

    out, softmax_lse = _allocate_flashmla_outputs(q, head_dim_v)

    if should_run_with_asm:
        raise_complete_if_dry_run()
        flash_mla_asm_mubin(
            q_nope,
            q_pe,
            k_cache[:, :, :, :head_dim_v],
            k_cache[:, :, :, head_dim_v:],
            cache_seqlens,
            block_table,
            tile_scheduler_metadata,
            num_splits,
            out,
            softmax_lse,
            softmax_scale,
            causal,
            None,
            None,
        )
    else:
        func = _get_module().get_function("mla_with_kvcache")
        raise_complete_if_dry_run()
        func(
            q_nope,
            q_pe,
            k_cache[:, :, :, :head_dim_v],
            k_cache[:, :, :, head_dim_v:],
            cache_seqlens,
            None,
            None,
            block_table,
            tile_scheduler_metadata,
            num_splits,
            out,
            softmax_lse,
            softmax_scale,
            causal,
        )
    return out, softmax_lse
