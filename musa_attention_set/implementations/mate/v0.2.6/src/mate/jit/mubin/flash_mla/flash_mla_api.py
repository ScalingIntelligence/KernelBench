from __future__ import annotations

import functools
from typing import Optional

import torch

from mate.artifacts import ensure_mubin_module_artifacts

from ..common import get_asm_dtype_from_torch_dtype
from ..launcher import get_mubin_launch_function, render_mubin_launcher
from .dispatch import FlashMLAMubinDispatcher, get_flash_mla_mubin_dispatcher


@functools.cache
def _get_flash_mla_mubin_runtime() -> FlashMLAMubinDispatcher:
    module_dir = ensure_mubin_module_artifacts("flash_mla")
    return get_flash_mla_mubin_dispatcher(module_dir / "kernel_map.json")


@functools.cache
def _get_flash_mla_mubin_launch_plan(
    dtype: torch.dtype,
    is_causal: bool,
    is_varlen_q: bool,
):
    dispatcher = _get_flash_mla_mubin_runtime()
    asm_id = dispatcher.get_asm_id(dtype, is_causal, is_varlen_q)
    entry = dispatcher.resolve_kernel_entry(asm_id)
    kernel_path = dispatcher.resolve_kernel_path(asm_id)
    launch = get_mubin_launch_function(
        "flash_mla",
        entry.kernel_name,
        render_mubin_launcher(
            "flash_mla",
            func_name=entry.kernel_name,
            spec=asm_id,
            asm_dtype=get_asm_dtype_from_torch_dtype(asm_id.dtype),
        ),
    )
    return str(kernel_path), launch


def flash_mla_asm_mubin(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    out: torch.Tensor,
    out_lse: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
):
    if q_nope.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("flash_mla_asm_mubin only supports fp16 and bf16")

    is_varlen_q = cu_seqlens_q is not None
    kernel_path, launch = _get_flash_mla_mubin_launch_plan(
        q_nope.dtype, is_causal, is_varlen_q
    )
    launch(
        kernel_path,
        q_nope,
        q_pe,
        ckv,
        kpe,
        seqlens_k,
        block_table,
        tile_scheduler_metadata,
        num_splits,
        out,
        out_lse,
        softmax_scale,
        cu_seqlens_q,
        max_seqlen_q,
    )
    return None
