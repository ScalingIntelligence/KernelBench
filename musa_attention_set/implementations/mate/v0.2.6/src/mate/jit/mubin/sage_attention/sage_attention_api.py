from __future__ import annotations

import functools
from typing import Optional

import torch

from mate.artifacts import ensure_mubin_module_artifacts

from ..launcher import get_mubin_launch_function, render_mubin_launcher
from .dispatch import (
    SageAttentionMubinDispatcher,
    get_sage_attention_mubin_dispatcher,
)


@functools.cache
def _get_sage_attention_mubin_runtime() -> SageAttentionMubinDispatcher:
    module_dir = ensure_mubin_module_artifacts("sage_attention")
    return get_sage_attention_mubin_dispatcher(module_dir / "kernel_map.json")


@functools.cache
def _get_sage_attention_mubin_launch_plan(
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    is_causal: bool,
    is_kv_cache: bool,
    headdim_qk: int,
    quant_mode: int,
    fp8_output: bool,
):
    dispatcher = _get_sage_attention_mubin_runtime()
    asm_id = dispatcher.get_asm_id(
        q_dtype,
        k_dtype,
        v_dtype,
        is_causal,
        is_kv_cache,
        headdim_qk,
        quant_mode,
        fp8_output,
    )
    entry = dispatcher.resolve_kernel_entry(asm_id)
    kernel_path = dispatcher.resolve_kernel_path(asm_id)
    launch = get_mubin_launch_function(
        "sage_attention",
        entry.kernel_name,
        render_mubin_launcher(
            "sage_attention", func_name=entry.kernel_name, spec=asm_id
        ),
    )
    return str(kernel_path), launch


def sage_attn_quantized_mubin(
    out: torch.Tensor,
    out_scale: Optional[torch.Tensor],
    out_lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_causal: bool,
    quant_mode: int,
):
    fp8_output = out_scale is not None
    qk_int8_path = (
        q.dtype == torch.int8
        and k.dtype == torch.int8
        and v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    )
    all_fp8_path = q.dtype == k.dtype == v.dtype and q.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if not (qk_int8_path or all_fp8_path):
        raise ValueError("q, k, and v must be all fp8 or q/k int8 with fp8 v")
    if quant_mode not in (0, 2, 6, 7):
        raise ValueError(f"Unsupported dense SageAttention quant_mode: {quant_mode}")

    kernel_path, launch = _get_sage_attention_mubin_launch_plan(
        q.dtype,
        k.dtype,
        v.dtype,
        is_causal,
        False,
        128,
        quant_mode,
        fp8_output,
    )
    launch(
        kernel_path,
        out,
        out_scale,
        out_lse,
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        softmax_scale,
    )
    return None


def sage_attn_quantized_with_kvcache_mubin(
    out: torch.Tensor,
    out_scale: Optional[torch.Tensor],
    out_lse: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    quant_mode: int,
):
    fp8_output = out_scale is not None
    if q.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("q must be fp8")
    if k_cache.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("k_cache must be fp8")
    if v_cache.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("v_cache must be fp8")
    if quant_mode not in (0, 2, 6):
        raise ValueError(f"Unsupported KV-cache SageAttention quant_mode: {quant_mode}")

    kernel_path, launch = _get_sage_attention_mubin_launch_plan(
        q.dtype,
        k_cache.dtype,
        v_cache.dtype,
        is_causal,
        True,
        128,
        quant_mode,
        fp8_output,
    )
    launch(
        kernel_path,
        out,
        out_scale,
        out_lse,
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        q_scale,
        k_scale,
        v_scale,
        softmax_scale,
    )
    return None
