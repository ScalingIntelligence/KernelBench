from __future__ import annotations

from typing import Optional

import torch

from mate.artifacts import ensure_mubin_module_artifacts

from ..common import (
    check_contiguous,
    check_musa,
    check_shape,
    check_tensor_same_device,
    check_type,
)
from ..launcher import get_mubin_launch_function, render_mubin_launcher
from .dispatch import get_flash_attention_mubin_dispatcher


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
    q_ffi = q
    k_ffi = k
    v_ffi = v
    out_ffi = out
    out_lse_ffi = out_lse
    input_cu_seqlen_q_ffi = input_cu_seqlen_q
    input_cu_seqlen_k_ffi = input_cu_seqlen_k

    tensors = [q_ffi, k_ffi, v_ffi, out_ffi, out_lse_ffi]
    if input_cu_seqlen_q_ffi is not None:
        tensors.append(input_cu_seqlen_q_ffi)
    if input_cu_seqlen_k_ffi is not None:
        tensors.append(input_cu_seqlen_k_ffi)

    check_musa(q_ffi)
    check_tensor_same_device(tensors)
    check_contiguous(q_ffi, dim=-1)
    check_contiguous(k_ffi, dim=-1)
    check_contiguous(v_ffi, dim=-1)
    check_contiguous(out_ffi)
    check_contiguous(out_lse_ffi)
    check_type(out_lse_ffi, torch.float32)
    if q_ffi.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("flash_atten_varlen_asm_mubin only supports fp16 and bf16")
    for tensor in (k_ffi, v_ffi, out_ffi):
        if tensor.dtype != q_ffi.dtype:
            raise ValueError("q, k, v, and out must have the same dtype")

    is_varlen = input_cu_seqlen_q_ffi is not None or input_cu_seqlen_k_ffi is not None
    if is_varlen:
        if input_cu_seqlen_q_ffi is None:
            raise ValueError(
                "input_cu_seqlen_q must be provided for varlen flash attention"
            )
        if input_cu_seqlen_k_ffi is None:
            raise ValueError(
                "input_cu_seqlen_k must be provided for varlen flash attention"
            )
        if max_seqlen_q is None:
            raise ValueError("max_seqlen_q must be provided for varlen flash attention")
        if max_seqlen_kv is None:
            raise ValueError(
                "max_seqlen_kv must be provided for varlen flash attention"
            )
        check_contiguous(input_cu_seqlen_q_ffi)
        check_contiguous(input_cu_seqlen_k_ffi)
        check_type(input_cu_seqlen_q_ffi, torch.int32)
        check_type(input_cu_seqlen_k_ffi, torch.int32)
        if input_cu_seqlen_k_ffi.shape[0] != input_cu_seqlen_q_ffi.shape[0]:
            raise ValueError(
                "input_cu_seqlen_q and input_cu_seqlen_k must have the same size"
            )
        if input_cu_seqlen_k_ffi.shape[0] <= 0:
            raise ValueError("input_cu_seqlen_k must be non-empty")
        if max_seqlen_q <= 0:
            raise ValueError("max_seqlen_q must be positive")
        if max_seqlen_kv <= 0:
            raise ValueError("max_seqlen_kv must be positive")

        q_shape = tuple(q_ffi.shape)
        k_shape = tuple(k_ffi.shape)
        v_shape = tuple(v_ffi.shape)
        batch = input_cu_seqlen_q_ffi.shape[0] - 1
        total_seqlen_q, num_heads, headdim_qk = q_shape
        total_seqlen_kv, num_heads_kv, k_headdim_qk = k_shape
        v_total_seqlen_kv, v_num_heads_kv, headdim_v = v_shape
        check_shape(q_ffi, (total_seqlen_q, num_heads, headdim_qk))
        check_shape(k_ffi, (total_seqlen_kv, num_heads_kv, headdim_qk))
        check_shape(v_ffi, (total_seqlen_kv, num_heads_kv, headdim_v))
        check_shape(out_ffi, (total_seqlen_q, num_heads, headdim_v))
        check_shape(out_lse_ffi, (num_heads, total_seqlen_q))
        if k_headdim_qk != headdim_qk:
            raise ValueError("k must have the same head dimension as q")
        if v_total_seqlen_kv != total_seqlen_kv or v_num_heads_kv != num_heads_kv:
            raise ValueError("v must match k total sequence and head dimensions")
    else:
        q_shape = tuple(q_ffi.shape)
        k_shape = tuple(k_ffi.shape)
        v_shape = tuple(v_ffi.shape)
        batch, seqlen_q, num_heads, headdim_qk = q_shape
        k_batch, seqlen_kv, num_heads_kv, k_headdim_qk = k_shape
        v_batch, v_seqlen_kv, v_num_heads_kv, headdim_v = v_shape
        check_shape(q_ffi, (batch, seqlen_q, num_heads, headdim_qk))
        check_shape(k_ffi, (batch, seqlen_kv, num_heads_kv, headdim_qk))
        check_shape(v_ffi, (batch, seqlen_kv, num_heads_kv, headdim_v))
        check_shape(out_ffi, (batch, seqlen_q, num_heads, headdim_v))
        check_shape(out_lse_ffi, (batch, num_heads, seqlen_q))
        if k_batch != batch or v_batch != batch:
            raise ValueError("q, k, and v must have the same batch size")
        if k_headdim_qk != headdim_qk:
            raise ValueError("k must have the same head dimension as q")
        if v_seqlen_kv != seqlen_kv or v_num_heads_kv != num_heads_kv:
            raise ValueError("v must match k sequence and head dimensions")

    is_192_128 = headdim_qk == 192 and headdim_v == 128
    is_128_128_or_less = headdim_qk == headdim_v and headdim_qk <= 128
    if not (is_192_128 or is_128_128_or_less):
        raise ValueError("HeadDim unsupported")
    if num_heads_kv <= 0:
        raise ValueError("nr_heads_kv must be positive")
    if num_heads < num_heads_kv:
        raise ValueError("nr_heads must be >= nr_heads_kv")
    if num_heads % num_heads_kv != 0:
        raise ValueError("nr_heads must be divisible by nr_heads_kv")

    dispatch_is_varlen = is_varlen
    q_launch = q
    k_launch = k
    v_launch = v
    out_launch = out
    out_lse_launch = out_lse
    cu_q_launch = input_cu_seqlen_q
    cu_k_launch = input_cu_seqlen_k
    max_q_launch = max_seqlen_q
    max_kv_launch = max_seqlen_kv
    if is_varlen and batch == 1:
        dispatch_is_varlen = False
        q_launch = q.as_strided(
            (1, q.shape[0], q.shape[1], q.shape[2]),
            (q.shape[0] * q.stride(0), q.stride(0), q.stride(1), q.stride(2)),
        )
        k_launch = k.as_strided(
            (1, k.shape[0], k.shape[1], k.shape[2]),
            (k.shape[0] * k.stride(0), k.stride(0), k.stride(1), k.stride(2)),
        )
        v_launch = v.as_strided(
            (1, v.shape[0], v.shape[1], v.shape[2]),
            (v.shape[0] * v.stride(0), v.stride(0), v.stride(1), v.stride(2)),
        )
        out_launch = out.as_strided(
            (1, out.shape[0], out.shape[1], out.shape[2]),
            (
                out.shape[0] * out.stride(0),
                out.stride(0),
                out.stride(1),
                out.stride(2),
            ),
        )
        out_lse_launch = out_lse.as_strided(
            (1, out_lse.shape[0], out_lse.shape[1]),
            (
                out_lse.shape[0] * out_lse.stride(0),
                out_lse.stride(0),
                out_lse.stride(1),
            ),
        )
        cu_q_launch = None
        cu_k_launch = None
        max_q_launch = None
        max_kv_launch = None

    module_dir = ensure_mubin_module_artifacts("flash_attention")
    dispatcher = get_flash_attention_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id = dispatcher.get_asm_id(
        q_ffi.dtype, is_causal, dispatch_is_varlen, headdim_qk
    )
    entry = dispatcher.resolve_kernel_entry(asm_id)
    kernel_path = dispatcher.resolve_kernel_path(asm_id, module_dir / "mubin")
    launcher_source = render_mubin_launcher(
        "flash_attention", func_name=entry.kernel_name, spec=asm_id
    )
    launch = get_mubin_launch_function(
        "flash_attention", entry.kernel_name, launcher_source
    )
    launch(
        str(kernel_path),
        q_launch,
        k_launch,
        v_launch,
        softmax_scale,
        out_launch,
        out_lse_launch,
        cu_q_launch,
        cu_k_launch,
        max_q_launch,
        max_kv_launch,
    )
    return None
