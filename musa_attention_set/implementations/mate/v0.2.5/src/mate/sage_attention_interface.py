"""
Low-level quantized SageAttention interface for MATE.

This module exposes only the pre-quantized entrypoints that map directly to the
low-level quantized attention interface. High-level recipe selection, input quantization, and
SageAttention-compatible packaging live in the wrapper layer.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional, Tuple, Union

import torch

from mate.api_logging import mate_api
from mate.jit.mubin.sage_attention import (
    sage_attn_quantized_mubin,
    sage_attn_quantized_with_kvcache_mubin,
)
from mate.jit.utils import maybe_contiguous


QuantRecipe = Tuple[int, int, int, int]
SageAttentionQuantizedOutput = Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
SageAttentionQuantizedKVCacheOutput = Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
_DEFAULT_DENSE_RECIPE: QuantRecipe = (128, 128, -1, 1)


class _SageAttentionAsmQuantMode(IntEnum):
    PER_TENSOR = 0
    PER_BLOCK = 2
    PER_THREAD = 6
    PER_BLOCK_V = 7


_ASM_QUANT_MODE_BY_RECIPE = {
    (-1, -1, -1, -1): _SageAttentionAsmQuantMode.PER_TENSOR,
    (128, 128, -1, 1): _SageAttentionAsmQuantMode.PER_BLOCK,
    (128, 16, -1, 1): _SageAttentionAsmQuantMode.PER_THREAD,
    (128, 128, -1, 128): _SageAttentionAsmQuantMode.PER_BLOCK_V,
}

_DENSE_ASM_RECIPES = set(_ASM_QUANT_MODE_BY_RECIPE)
_KVCACHE_ASM_RECIPES = {
    (-1, -1, -1, -1),
    (128, 128, -1, 1),
    (128, 16, -1, 1),
}


def _format_supported_quant_recipes(recipes: set[QuantRecipe]) -> str:
    return ", ".join(str(item) for item in sorted(recipes))


def _resolve_supported_asm_quant_mode(
    quant_recipe: QuantRecipe,
    *,
    is_kv_cache: bool,
) -> int:
    supported = _KVCACHE_ASM_RECIPES if is_kv_cache else _DENSE_ASM_RECIPES
    if quant_recipe not in supported:
        kind = "KV-cache" if is_kv_cache else "dense"
        supported_str = _format_supported_quant_recipes(supported)
        raise NotImplementedError(
            f"SageAttention {kind} ASM path does not support quant_recipe={quant_recipe}. "
            f"Supported values: {supported_str}."
        )
    return int(_ASM_QUANT_MODE_BY_RECIPE[quant_recipe])


@mate_api
def sage_attn_quantized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    quant_recipe: Optional[QuantRecipe] = None,
    return_lse: bool = False,
    fp8_output: bool = False,
    backend: str = "mubin",
) -> SageAttentionQuantizedOutput:
    r"""
    Low-level pre-quantized dense SageAttention forward pass.

    This entrypoint targets the low-level dense quantized attention interface
    and does not perform any quantization internally. Use it when ``q``, ``k``,
    ``v``, and their scale tensors have already been prepared for the selected
    ``quant_recipe``.

    Parameters
    ----------
    q : Tensor
        Pre-quantized query tensor with shape
        ``(batch, seqlen_q, num_q_heads, head_dim_qk)``. The dense quantized
        path accepts FP8 query tensors and also supports INT8 queries when
        paired with INT8 keys and FP8 values.
    k : Tensor
        Pre-quantized key tensor with shape
        ``(batch, seqlen_kv, num_kv_heads, head_dim_qk)``. ``num_q_heads`` must
        be divisible by ``num_kv_heads``.
    v : Tensor
        Pre-quantized value tensor with shape
        ``(batch, seqlen_kv, num_kv_heads, head_dim_v)``. Dense quantized
        attention expects FP8 values.
    q_scale : Optional[Tensor]
        Query scale tensor matching the backend layout for ``q`` and
        ``quant_recipe``. For example, per-tensor quantization uses shape
        ``(1, 1, 1, 1)``, while the sequence-block dense recipes use
        ``(batch, q_seq_scale_num, num_q_heads, 1)``.
    k_scale : Optional[Tensor]
        Key scale tensor matching the backend layout for ``k`` and
        ``quant_recipe``. The supported dense non-tensor recipes use
        ``(batch, k_seq_scale_num, num_kv_heads, 1)``.
    v_scale : Optional[Tensor]
        Value scale tensor matching the backend layout for ``v`` and
        ``quant_recipe``. For block-V quantization ``(128, 128, -1, 128)`` the
        expected shape is ``(batch, ceil_div(seqlen_kv, 128), num_kv_heads, 1)``;
        other supported dense non-tensor recipes expect
        ``(batch, 1, num_kv_heads, head_dim_v)``.
    softmax_scale : Optional[float]
        Scale factor applied to ``QK^T`` before the softmax. Defaults to
        ``1 / sqrt(head_dim_qk)`` when not provided.
    causal : bool, optional
        Whether to apply a causal mask. Default is ``False``.
    quant_recipe : Optional[QuantRecipe]
        Quantization block-size recipe. The 4-tuple is interpreted as
        ``(seqlen_q, seqlen_kv, headdim_qk, headdim_vo)``, where each entry
        describes the quantization block size on the corresponding axis.

        The dense path currently supports the following tuples:
        ``(-1, -1, -1, -1)``, ``(128, 128, -1, 1)``, ``(128, 16, -1, 1)``,
        and ``(128, 128, -1, 128)``. ``-1`` means the
        corresponding axis is not split into smaller quantization blocks.
        Defaults to ``(128, 128, -1, 1)``.
    return_lse : bool, optional
        Whether to also return the log-sum-exp tensor produced by the kernel.
        Default is ``False``.
    fp8_output : bool, optional
        Whether to quantize the attention output to FP8. If ``False``, the
        output tensor uses ``torch.bfloat16``. If ``True``, the output tensor
        uses the same FP8 dtype as ``v`` and the function also returns an
        ``out_scale`` tensor with shape ``(batch, seqlen_q, num_q_heads, 1)``
        and dtype ``torch.float32``.
    backend : str, optional
        Backend selector. Only ``"mubin"`` is supported for this low-level
        quantized path.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
        The output tensor has shape
        ``(batch, seqlen_q, num_q_heads, head_dim_v)``. When
        ``fp8_output=False``, returns ``out`` or ``(out, lse)`` depending on
        ``return_lse``, where ``out`` has dtype ``torch.bfloat16``. When
        ``fp8_output=True``, returns ``(out_fp8, out_scale)`` or
        ``(out_fp8, out_scale, lse)``, where ``out_fp8.dtype == v.dtype`` and
        ``out_scale.dtype == torch.float32``. The ``lse`` tensor has shape
        ``(batch, num_q_heads, seqlen_q)`` and dtype ``torch.float32``.
    """
    resolved_quant_recipe = quant_recipe or _DEFAULT_DENSE_RECIPE
    if backend != "mubin":
        raise NotImplementedError(
            f"Unsupported SageAttention quantized backend: {backend}. "
            "mutlass does not support this path."
        )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    quant_mode = _resolve_supported_asm_quant_mode(
        resolved_quant_recipe,
        is_kv_cache=False,
    )

    batch, seqlen_q, nheads, _ = q.shape
    headdim_v = v.shape[-1]

    out_dtype = v.dtype if fp8_output else torch.bfloat16
    out = torch.empty(
        (batch, seqlen_q, nheads, headdim_v), dtype=out_dtype, device=q.device
    )
    out_scale = (
        torch.empty((batch, seqlen_q, nheads, 1), dtype=torch.float32, device=q.device)
        if fp8_output
        else None
    )
    lse = torch.empty(
        (batch, nheads, seqlen_q),
        dtype=torch.float32,
        device=q.device,
    )

    sage_attn_quantized_mubin(
        out,
        maybe_contiguous(out_scale),
        lse,
        maybe_contiguous(q),
        maybe_contiguous(k),
        maybe_contiguous(v),
        softmax_scale,
        maybe_contiguous(q_scale),
        maybe_contiguous(k_scale),
        maybe_contiguous(v_scale),
        causal,
        quant_mode,
    )

    if fp8_output and return_lse:
        return out, out_scale, lse
    if fp8_output:
        return out, out_scale
    return (out, lse) if return_lse else out


@mate_api
def sage_attn_quantized_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    quant_recipe: Optional[QuantRecipe] = None,
    return_lse: bool = False,
    fp8_output: bool = False,
    backend: str = "mubin",
) -> SageAttentionQuantizedKVCacheOutput:
    r"""
    Low-level pre-quantized SageAttention forward pass with paged KV cache.

    This entrypoint targets the low-level quantized attention interface for
    paged KV cache and does not quantize inputs internally. The current
    KV-cache path expects FP8 query, key-cache, and value-cache tensors
    together with scale tensors that follow the selected ``quant_recipe``.

    Parameters
    ----------
    q : Tensor
        Pre-quantized query tensor with shape
        ``(batch, seqlen_q, num_q_heads, head_dim_qk)``. The current KV-cache
        path expects FP8 query tensors.
    k_cache : Tensor
        Pre-quantized paged key cache with shape
        ``(num_blocks, page_block_size, num_kv_heads, head_dim_qk)``.
    v_cache : Tensor
        Pre-quantized paged value cache with shape
        ``(num_blocks, page_block_size, num_kv_heads, head_dim_v)``.
    page_table : Tensor
        Int32 page table with shape ``(batch, max_num_pages_per_seq)`` mapping
        each sequence to the physical cache blocks used by ``k_cache`` and
        ``v_cache``.
    cache_seqlens : Tensor
        Int32 tensor with shape ``(batch,)`` containing the valid KV length of
        each sequence in the paged cache.
    q_scale : Optional[Tensor]
        Query scale tensor matching the backend layout for ``q`` and
        ``quant_recipe``. Per-tensor quantization uses shape ``(1, 1, 1, 1)``,
        and the supported non-tensor recipes use
        ``(batch, q_seq_scale_num, num_q_heads, 1)``.
    k_scale : Optional[Tensor]
        Key-cache scale tensor matching the backend layout for ``k_cache`` and
        ``quant_recipe``. The supported non-tensor KV-cache recipes expect shape
        ``(num_blocks, k_seq_scale_num_per_block, num_kv_heads, 1)``.
    v_scale : Optional[Tensor]
        Value-cache scale tensor matching the backend layout for ``v_cache`` and
        ``quant_recipe``. The supported non-tensor KV-cache recipes expect shape
        ``(num_blocks, 1, num_kv_heads, head_dim_v)``.
    softmax_scale : Optional[float]
        Scale factor applied to ``QK^T`` before the softmax. Defaults to
        ``1 / sqrt(head_dim_qk)`` when not provided.
    causal : bool, optional
        Whether to apply a causal mask. Default is ``False``.
    quant_recipe : Optional[QuantRecipe]
        Quantization block-size recipe. The 4-tuple is interpreted as
        ``(seqlen_q, seqlen_kv, headdim_qk, headdim_vo)``, where each entry
        describes the quantization block size on the corresponding axis.

        The KV-cache path currently supports ``(-1, -1, -1, -1)``,
        ``(128, 128, -1, 1)``, and ``(128, 16, -1, 1)``. ``-1`` means the
        corresponding axis is not split into smaller quantization blocks.
        Defaults to ``(128, 128, -1, 1)``.
    return_lse : bool, optional
        Whether to also return the log-sum-exp tensor produced by the kernel.
        Default is ``False``.
    fp8_output : bool, optional
        Whether to quantize the attention output to FP8. If ``False``, the
        output tensor uses ``torch.bfloat16``. If ``True``, the output tensor
        uses the same FP8 dtype as ``v_cache`` and the function also returns an
        ``out_scale`` tensor with shape ``(batch, seqlen_q, num_q_heads, 1)``
        and dtype ``torch.float32``.
    backend : str, optional
        Backend selector. Only ``"mubin"`` is supported for this low-level
        quantized path.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
        The output tensor has shape
        ``(batch, seqlen_q, num_q_heads, head_dim_v)``. When
        ``fp8_output=False``, returns ``out`` or ``(out, lse)`` depending on
        ``return_lse``, where ``out`` has dtype ``torch.bfloat16``. When
        ``fp8_output=True``, returns ``(out_fp8, out_scale)`` or
        ``(out_fp8, out_scale, lse)``, where
        ``out_fp8.dtype == v_cache.dtype`` and
        ``out_scale.dtype == torch.float32``. The ``lse`` tensor has shape
        ``(batch, num_q_heads, seqlen_q)`` and dtype ``torch.float32``.
    """
    resolved_quant_recipe = quant_recipe or _DEFAULT_DENSE_RECIPE
    if backend != "mubin":
        raise NotImplementedError(
            f"Unsupported SageAttention quantized KV-cache backend: {backend}. "
            "mutlass does not support this path."
        )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    quant_mode = _resolve_supported_asm_quant_mode(
        resolved_quant_recipe,
        is_kv_cache=True,
    )

    batch, seqlen_q, nheads, _ = q.shape
    headdim_v = v_cache.shape[-1]

    out_dtype = v_cache.dtype if fp8_output else torch.bfloat16
    out = torch.empty(
        (batch, seqlen_q, nheads, headdim_v), dtype=out_dtype, device=q.device
    )
    out_scale = (
        torch.empty((batch, seqlen_q, nheads, 1), dtype=torch.float32, device=q.device)
        if fp8_output
        else None
    )
    lse = torch.empty(
        (batch, nheads, seqlen_q),
        dtype=torch.float32,
        device=q.device,
    )

    sage_attn_quantized_with_kvcache_mubin(
        out,
        maybe_contiguous(out_scale),
        lse,
        maybe_contiguous(q),
        maybe_contiguous(k_cache),
        maybe_contiguous(v_cache),
        maybe_contiguous(page_table),
        maybe_contiguous(cache_seqlens),
        maybe_contiguous(q_scale),
        maybe_contiguous(k_scale),
        maybe_contiguous(v_scale),
        softmax_scale,
        causal,
        quant_mode,
    )

    if fp8_output and return_lse:
        return out, out_scale, lse
    if fp8_output:
        return out, out_scale
    return (out, lse) if return_lse else out


__all__ = [
    "sage_attn_quantized",
    "sage_attn_quantized_with_kvcache",
]
