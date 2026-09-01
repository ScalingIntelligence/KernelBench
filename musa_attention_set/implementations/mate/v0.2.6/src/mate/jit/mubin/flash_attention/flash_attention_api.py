from __future__ import annotations

import functools
from typing import Optional

import torch

from mate.artifacts import ensure_mubin_module_artifacts

from ..common import get_asm_dtype_from_torch_dtype
from ..launcher import get_mubin_launch_function, render_mubin_launcher
from .dispatch import (
    FlashAttentionMubinDispatcher,
    get_flash_attention_mubin_dispatcher,
)


@functools.cache
def _get_flash_attention_mubin_runtime() -> FlashAttentionMubinDispatcher:
    module_dir = ensure_mubin_module_artifacts("flash_attention")
    return get_flash_attention_mubin_dispatcher(module_dir / "kernel_map.json")


@functools.cache
def _get_flash_attention_mubin_launch_plan(
    dtype: torch.dtype,
    is_causal: bool,
    is_varlen: bool,
    headdim_qk: int,
):
    dispatcher = _get_flash_attention_mubin_runtime()
    asm_id = dispatcher.get_asm_id(dtype, is_causal, is_varlen, headdim_qk)
    entry = dispatcher.resolve_kernel_entry(asm_id)
    kernel_path = dispatcher.resolve_kernel_path(asm_id)
    launch = get_mubin_launch_function(
        "flash_attention",
        entry.kernel_name,
        render_mubin_launcher(
            "flash_attention",
            func_name=entry.kernel_name,
            spec=asm_id,
            asm_dtype=get_asm_dtype_from_torch_dtype(asm_id.dtype),
        ),
    )
    return str(kernel_path), launch


def flash_atten_varlen_asm_mubin(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor,
    out_lse: torch.Tensor,
    is_causal: bool,
    input_cu_seqlen_q: Optional[torch.Tensor] = None,
    input_cu_seqlen_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
):
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("flash_atten_varlen_asm_mubin only supports fp16 and bf16")
    if q.ndim not in (3, 4):
        raise ValueError("q must be a 3D or 4D tensor")

    headdim_qk = q.shape[-1]
    if headdim_qk <= 0 or (headdim_qk > 128 and headdim_qk != 192):
        raise ValueError("HeadDim unsupported")
    headdim_v = v.shape[-1]
    if headdim_qk % 2 != 0 or headdim_v % 2 != 0:
        raise ValueError(
            "FP16/BF16 Q/K and V head dimensions must be divisible by 2, "
            f"got Q/K={headdim_qk} and V={headdim_v}"
        )

    is_varlen = input_cu_seqlen_q is not None or input_cu_seqlen_k is not None

    dispatch_is_varlen = is_varlen
    # A single varlen batch uses the fixed kernel
    use_not_varlen = (
        is_varlen
        and input_cu_seqlen_q is not None
        and input_cu_seqlen_k is not None
        and input_cu_seqlen_q.ndim == 1
        and input_cu_seqlen_k.ndim == 1
        and input_cu_seqlen_q.shape[0] == 2
        and input_cu_seqlen_k.shape[0] == 2
        and q.ndim == 3
        and k.ndim == 3
        and v.ndim == 3
        and out.ndim == 3
        and out_lse.ndim == 2
    )
    if use_not_varlen:
        dispatch_is_varlen = False

    kernel_path, launch = _get_flash_attention_mubin_launch_plan(
        q.dtype, is_causal, dispatch_is_varlen, headdim_qk
    )
    launch(
        kernel_path,
        q,
        k,
        v,
        softmax_scale,
        out,
        out_lse,
        input_cu_seqlen_q,
        input_cu_seqlen_k,
        max_seqlen_q,
        max_seqlen_kv,
    )
    return None
