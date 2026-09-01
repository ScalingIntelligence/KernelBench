# ruff: noqa
"""Shared sparse prefill host/JIT plumbing for MODEL1 and V3.2."""

from __future__ import annotations

from typing import Optional

import torch
import tilelang


SPARSE_PREFILL_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
}


SPARSE_PREFILL_COMPILE_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


def validate_token_lengths(
    lengths: Optional[torch.Tensor],
    seq_len: int,
    name: str,
) -> Optional[torch.Tensor]:
    if lengths is None:
        return None
    assert lengths.dtype == torch.int32, f"{name} must be int32"
    assert lengths.shape == (seq_len,), f"{name} must have shape [S_q]"
    if lengths.shape[-1] != 1:
        assert lengths.stride(-1) == 1, f"{name} last dimension must be contiguous"
    return lengths.contiguous()


def validate_prefill_attn_sink(
    attn_sink: Optional[torch.Tensor],
    heads: int,
) -> Optional[torch.Tensor]:
    if attn_sink is None:
        return None
    assert attn_sink.dtype == torch.float32, "attn_sink must be float32"
    assert attn_sink.shape == (heads,), "attn_sink must have shape [H_q]"
    if attn_sink.shape[-1] != 1:
        assert attn_sink.stride(-1) == 1, "attn_sink last dimension must be contiguous"
    return attn_sink.contiguous()


def prepare_sparse_mla_strided_tensor(
    name: str, tensor: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Validate a runtime-strided input once and normalize singleton strides."""

    shape = tensor.shape
    strides = tensor.stride()
    normalized = None
    if strides[-1] != 1:
        assert shape[-1] == 1, f"{name} last dimension must be contiguous"
        normalized = list(strides)
        normalized[-1] = 1
    for dim in range(len(strides) - 1):
        stride = strides[dim]
        if stride > 0 and stride % multiple == 0:
            continue
        if shape[dim] != 1:
            assert stride > 0, f"{name} stride({dim}) must be positive, got {stride}"
            raise AssertionError(
                f"{name} stride({dim}) must be divisible by {multiple}, got {stride}"
            )
        if normalized is None:
            normalized = list(strides)
        normalized[dim] = multiple
    if normalized is None:
        return tensor, shape
    return (
        torch.as_strided(tensor, shape, normalized, tensor.storage_offset()),
        shape,
    )
