"""
SageAttention compatibility interface built on top of MATE.

This wrapper exposes the currently supported high-level SageAttention API
and routes execution to MATE kernels on MUSA.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Tuple, Union, cast, overload

import torch
from mate.jit.utils import maybe_contiguous
from mate.sage_attention_interface import sage_attn_quantized
from mate.testing import quantize_sage_attention_tensor


QuantRecipe = Tuple[int, int, int, int]
_DEFAULT_THREAD_RECIPE: QuantRecipe = (128, 16, -1, 1)
_DEFAULT_QK_QUANT_DTYPE = "int8"
_SUPPORTED_DENSE_RECIPES = {
    (-1, -1, -1, -1),
    (128, 128, -1, 1),
    (128, 16, -1, 1),
    (128, 128, -1, 128),
}
_SUPPORTED_QK_QUANT_DTYPES = {"int8", "fp8"}
SageAttentionOutput = Tuple[torch.Tensor, torch.Tensor]
SageAttentionFp8Output = Tuple[torch.Tensor, torch.Tensor]
SageAttentionFp8LseOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _validate_tensor_layout(tensor_layout: str) -> str:
    if tensor_layout not in {"HND", "NHD"}:
        raise ValueError(
            f"Unsupported tensor_layout: {tensor_layout}. Expected 'HND' or 'NHD'."
        )
    return tensor_layout


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    supported_dtypes = {torch.float16, torch.bfloat16}
    if (
        q.dtype not in supported_dtypes
        or k.dtype not in supported_dtypes
        or v.dtype not in supported_dtypes
    ):
        raise TypeError("q, k, and v must have dtype torch.float16 or torch.bfloat16.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device.")
    if q.device.type != "musa":
        raise ValueError("Input tensors must be on a MUSA device.")


def _format_supported_quant_recipes() -> str:
    return ", ".join(str(item) for item in sorted(_SUPPORTED_DENSE_RECIPES))


def _validate_supported_quant_recipe(quant_recipe: QuantRecipe) -> QuantRecipe:
    if quant_recipe not in _SUPPORTED_DENSE_RECIPES:
        raise NotImplementedError(
            f"Unsupported quant_recipe={quant_recipe}. Supported values: "
            f"{_format_supported_quant_recipes()}."
        )
    return quant_recipe


def _validate_qk_quant_dtype(qk_quant_dtype: str) -> str:
    if qk_quant_dtype not in _SUPPORTED_QK_QUANT_DTYPES:
        supported = ", ".join(sorted(_SUPPORTED_QK_QUANT_DTYPES))
        raise ValueError(
            f"Unsupported qk_quant_dtype={qk_quant_dtype!r}. "
            f"Supported values: {supported}."
        )
    return qk_quant_dtype


def _resolve_quant_recipe(
    quant_recipe: Optional[QuantRecipe] = None,
    qk_quant_gran: Optional[str] = "per_thread",
) -> QuantRecipe:
    if quant_recipe is not None:
        return _validate_supported_quant_recipe(quant_recipe)
    if qk_quant_gran in (None, "per_thread"):
        return _DEFAULT_THREAD_RECIPE
    raise NotImplementedError(
        "Only qk_quant_gran='per_thread' is supported. "
        "Pass quant_recipe to select another quantization granularity."
    )


def _to_bnhd(x: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    _validate_tensor_layout(tensor_layout)
    return x.transpose(1, 2) if tensor_layout == "HND" else x


def _from_bnhd(x: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    _validate_tensor_layout(tensor_layout)
    return x.transpose(1, 2) if tensor_layout == "HND" else x


def _compute_lse_correction(
    q: torch.Tensor,
    k: torch.Tensor,
    tensor_layout: str,
) -> torch.Tensor:
    seq_dim = 1 if tensor_layout == "NHD" else 2
    nh_dim = 2 if tensor_layout == "NHD" else 1

    k_mean = k.mean(dim=seq_dim, keepdim=True)
    q_per_kv_heads = q.size(nh_dim) // k.size(nh_dim)
    k_mean_broadcast = (
        torch.repeat_interleave(k_mean, q_per_kv_heads, dim=nh_dim)
        if q_per_kv_heads > 1
        else k_mean
    )

    if tensor_layout == "NHD":
        return (
            torch.matmul(
                q.transpose(1, 2),
                k_mean_broadcast.transpose(1, 2).transpose(2, 3),
            )
            .squeeze(-1)
            .to(torch.float32)
        )
    return (
        torch.matmul(q, k_mean_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    )


def _finalize_output(
    result: Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    tensor_layout: str,
    output_dtype: torch.dtype,
    return_lse: bool,
    sm_scale: float,
    lse_correction: Optional[torch.Tensor] = None,
) -> Union[
    torch.Tensor, SageAttentionOutput, SageAttentionFp8Output, SageAttentionFp8LseOutput
]:
    if not return_lse:
        if isinstance(result, tuple):
            out, out_scale = cast(SageAttentionFp8Output, result)
            return _from_bnhd(out, tensor_layout).to(output_dtype), _from_bnhd(
                out_scale, tensor_layout
            ).to(torch.float32)
        return _from_bnhd(result, tensor_layout).to(output_dtype)

    result_tuple = cast(Union[SageAttentionOutput, SageAttentionFp8LseOutput], result)
    if len(result_tuple) == 3:
        out, out_scale, lse = cast(SageAttentionFp8LseOutput, result_tuple)
        out = _from_bnhd(out, tensor_layout).to(output_dtype)
        out_scale = _from_bnhd(out_scale, tensor_layout).to(torch.float32)
        if lse_correction is not None:
            lse = lse + lse_correction * sm_scale
        return out, out_scale, lse

    out, lse = cast(SageAttentionOutput, result_tuple)
    out = _from_bnhd(out, tensor_layout).to(output_dtype)
    if lse_correction is not None:
        lse = lse + lse_correction * sm_scale
    return out, lse


def _resolve_qk_quant_dtype(qk_quant_dtype: str) -> torch.dtype:
    return torch.int8 if qk_quant_dtype == "int8" else torch.float8_e4m3fn


def _run_quantized_sage_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str,
    is_causal: bool,
    sm_scale: Optional[float],
    quant_recipe: QuantRecipe,
    qk_quant_dtype: str,
    return_lse: bool,
    smooth_k: bool,
    fp8_output: bool = False,
) -> Union[
    torch.Tensor, SageAttentionOutput, SageAttentionFp8Output, SageAttentionFp8LseOutput
]:
    tensor_layout = _validate_tensor_layout(tensor_layout)
    _validate_inputs(q, k, v)
    resolved_quant_recipe = _validate_supported_quant_recipe(quant_recipe)
    resolved_qk_quant_dtype = _validate_qk_quant_dtype(qk_quant_dtype)
    resolved_sm_scale = (
        sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    )
    output_dtype = torch.float8_e4m3fn if fp8_output else q.dtype
    qk_quant_torch_dtype = _resolve_qk_quant_dtype(resolved_qk_quant_dtype)

    q_bnhd = maybe_contiguous(_to_bnhd(q, tensor_layout).to(torch.bfloat16))
    k_bnhd = maybe_contiguous(_to_bnhd(k, tensor_layout).to(torch.bfloat16))
    v_bnhd = maybe_contiguous(_to_bnhd(v, tensor_layout).to(torch.bfloat16))

    lse_correction = (
        _compute_lse_correction(q, k, tensor_layout)
        if return_lse and smooth_k
        else None
    )

    q_quant, q_scale = quantize_sage_attention_tensor(
        q_bnhd,
        operand="q",
        quant_recipe=resolved_quant_recipe,
        quant_dtype=qk_quant_torch_dtype,
    )
    k_quant, k_scale = quantize_sage_attention_tensor(
        k_bnhd,
        operand="k",
        quant_recipe=resolved_quant_recipe,
        quant_dtype=qk_quant_torch_dtype,
        smooth_k=smooth_k,
    )
    v_quant, v_scale = quantize_sage_attention_tensor(
        v_bnhd,
        operand="v",
        quant_recipe=resolved_quant_recipe,
        quant_dtype=torch.float8_e4m3fn,
    )

    result = sage_attn_quantized(
        q=q_quant,
        k=k_quant,
        v=v_quant,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        softmax_scale=resolved_sm_scale,
        causal=is_causal,
        quant_recipe=resolved_quant_recipe,
        return_lse=return_lse,
        fp8_output=fp8_output,
    )
    return _finalize_output(
        result,
        tensor_layout,
        output_dtype,
        return_lse,
        resolved_sm_scale,
        lse_correction=lse_correction,
    )


@overload
def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    qk_quant_dtype: str = _DEFAULT_QK_QUANT_DTYPE,
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: Literal[False] = False,
    fp8_output: Literal[False] = False,
    quant_recipe: Optional[QuantRecipe] = None,
    **kwargs: Any,
) -> torch.Tensor: ...


@overload
def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    qk_quant_dtype: str = _DEFAULT_QK_QUANT_DTYPE,
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: Literal[True] = True,
    fp8_output: Literal[False] = False,
    quant_recipe: Optional[QuantRecipe] = None,
    **kwargs: Any,
) -> SageAttentionOutput: ...


@overload
def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    qk_quant_dtype: str = _DEFAULT_QK_QUANT_DTYPE,
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    fp8_output: bool = False,
    quant_recipe: Optional[QuantRecipe] = None,
    **kwargs: Any,
) -> Union[
    torch.Tensor, SageAttentionOutput, SageAttentionFp8Output, SageAttentionFp8LseOutput
]: ...


def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    qk_quant_dtype: str = _DEFAULT_QK_QUANT_DTYPE,
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    fp8_output: bool = False,
    quant_recipe: Optional[QuantRecipe] = None,
    **kwargs: Any,
) -> Union[
    torch.Tensor, SageAttentionOutput, SageAttentionFp8Output, SageAttentionFp8LseOutput
]:
    r"""
    SageAttention wrapper with configurable QK quantization dtype on MUSA.

    When ``quant_recipe`` is omitted, the wrapper uses ``(128, 16, -1, 1)``.
    If ``quant_recipe`` is explicitly provided, it overrides ``qk_quant_gran``.
    ``qk_quant_dtype`` selects whether QK is quantized to ``int8`` or ``fp8``.
    ``quant_recipe`` describes the quantization granularity. Only
    ``qk_quant_gran='per_thread'`` is supported as the default granularity
    shortcut. If ``fp8_output=True``, returns ``(out_fp8, out_scale)`` or
    ``(out_fp8, out_scale, lse)`` when ``return_lse=True``. The ``out_scale``
    tensor uses dtype ``torch.float32`` and follows the requested public tensor
    layout.
    """
    del pv_accum_dtype, kwargs
    resolved_quant_recipe = _resolve_quant_recipe(
        quant_recipe=quant_recipe,
        qk_quant_gran=qk_quant_gran,
    )
    return _run_quantized_sage_attention(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        quant_recipe=resolved_quant_recipe,
        qk_quant_dtype=qk_quant_dtype,
        return_lse=return_lse,
        smooth_k=smooth_k,
        fp8_output=fp8_output,
    )


@overload
def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: Literal[False] = False,
    qk_quant_gran: str = "per_thread",
    qk_quant_dtype: str = _DEFAULT_QK_QUANT_DTYPE,
    quant_recipe: Optional[QuantRecipe] = None,
    smooth_k: bool = True,
    fp8_output: Literal[False] = False,
    **kwargs: Any,
) -> torch.Tensor: ...


@overload
def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: Literal[True] = True,
    qk_quant_gran: str = "per_thread",
    qk_quant_dtype: str = _DEFAULT_QK_QUANT_DTYPE,
    quant_recipe: Optional[QuantRecipe] = None,
    smooth_k: bool = True,
    fp8_output: Literal[False] = False,
    **kwargs: Any,
) -> SageAttentionOutput: ...


@overload
def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    qk_quant_gran: str = "per_thread",
    qk_quant_dtype: str = _DEFAULT_QK_QUANT_DTYPE,
    quant_recipe: Optional[QuantRecipe] = None,
    smooth_k: bool = True,
    fp8_output: bool = False,
    **kwargs: Any,
) -> Union[
    torch.Tensor, SageAttentionOutput, SageAttentionFp8Output, SageAttentionFp8LseOutput
]: ...


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    qk_quant_gran: str = "per_thread",
    qk_quant_dtype: str = _DEFAULT_QK_QUANT_DTYPE,
    quant_recipe: Optional[QuantRecipe] = None,
    smooth_k: bool = True,
    fp8_output: bool = False,
    **kwargs: Any,
) -> Union[
    torch.Tensor, SageAttentionOutput, SageAttentionFp8Output, SageAttentionFp8LseOutput
]:
    r"""
    Automatically selects the supported SageAttention implementation on MUSA.

    By default this forwards to ``sageattn_qk_int8_pv_fp8_cuda_sm90`` with the
    thread recipe ``(128, 16, -1, 1)`` and ``qk_quant_dtype="int8"``.
    ``quant_recipe`` controls quantization granularity, while
    ``qk_quant_dtype`` controls whether QK is quantized to INT8 or FP8. If
    ``fp8_output=True``, returns ``(out_fp8, out_scale)`` or
    ``(out_fp8, out_scale, lse)`` when ``return_lse=True``. Without FP8 output,
    returns ``out`` or ``(out, lse)``.
    """
    return sageattn_qk_int8_pv_fp8_cuda_sm90(
        q=q,
        k=k,
        v=v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        qk_quant_dtype=qk_quant_dtype,
        sm_scale=sm_scale,
        smooth_k=smooth_k,
        return_lse=return_lse,
        fp8_output=fp8_output,
        quant_recipe=quant_recipe,
        **kwargs,
    )


__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_cuda_sm90"]
