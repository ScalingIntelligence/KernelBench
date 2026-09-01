from __future__ import annotations

from typing import Optional

import torch

from mate.artifacts import ensure_mubin_module_artifacts
from mate.utils import ceil_div

from ..common import (
    check_contiguous,
    check_musa,
    check_shape,
    check_tensor_same_device,
    check_type,
)
from ..launcher import get_mubin_launch_function, render_mubin_launcher
from .dispatch import get_sage_attention_mubin_dispatcher


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
    out_ffi = out
    out_scale_ffi = out_scale
    out_lse_ffi = out_lse
    q_ffi = q
    k_ffi = k
    v_ffi = v
    q_scale_ffi = q_scale
    k_scale_ffi = k_scale
    v_scale_ffi = v_scale

    tensors = [
        out_ffi,
        out_lse_ffi,
        q_ffi,
        k_ffi,
        v_ffi,
        q_scale_ffi,
        k_scale_ffi,
        v_scale_ffi,
    ]
    if out_scale_ffi is not None:
        tensors.append(out_scale_ffi)

    check_musa(q_ffi)
    check_tensor_same_device(tensors)
    check_contiguous(q_ffi, dim=-1)
    check_contiguous(k_ffi, dim=-1)
    check_contiguous(v_ffi, dim=-1)
    check_contiguous(out_ffi)
    check_contiguous(out_lse_ffi)
    check_type(out_lse_ffi, torch.float32)

    fp8_output = out_scale_ffi is not None
    qk_int8_path = (
        q_ffi.dtype == torch.int8
        and k_ffi.dtype == torch.int8
        and v_ffi.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    )
    if qk_int8_path:
        pass
    elif q_ffi.dtype == k_ffi.dtype == v_ffi.dtype and q_ffi.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        pass
    else:
        raise ValueError("q, k, and v must be all fp8 or q/k int8 with fp8 v")
    if fp8_output:
        if out_ffi.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("out must be fp8 when fp8_output is enabled")
        if out_ffi.dtype != v_ffi.dtype:
            raise ValueError(
                "out must have the same dtype as v when fp8_output is enabled"
            )
        check_type(out_scale_ffi, torch.float32)
    elif out_ffi.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("out must be fp16 or bf16 when fp8_output is disabled")

    batch, seqlen_q, num_heads, headdim_qk = tuple(q_ffi.shape)
    k_batch, seqlen_kv, num_heads_kv, k_headdim_qk = tuple(k_ffi.shape)
    v_batch, v_seqlen_kv, v_num_heads_kv, headdim_v = tuple(v_ffi.shape)
    check_shape(q_ffi, (batch, seqlen_q, num_heads, headdim_qk))
    check_shape(k_ffi, (batch, seqlen_kv, num_heads_kv, headdim_qk))
    check_shape(v_ffi, (batch, seqlen_kv, num_heads_kv, headdim_v))
    check_shape(out_ffi, (batch, seqlen_q, num_heads, headdim_v))
    check_shape(out_lse_ffi, (batch, num_heads, seqlen_q))
    if fp8_output:
        check_shape(out_scale_ffi, (batch, seqlen_q, num_heads, 1))
    if k_batch != batch or v_batch != batch:
        raise ValueError("q, k, and v must have the same batch size")
    if k_headdim_qk != headdim_qk:
        raise ValueError("k must have the same head dimension as q")
    if v_seqlen_kv != seqlen_kv or v_num_heads_kv != num_heads_kv:
        raise ValueError("v must match k sequence and head dimensions")
    if num_heads_kv <= 0:
        raise ValueError("num_heads_kv must be positive")
    if num_heads % num_heads_kv != 0:
        raise ValueError("num_heads must be divisible by num_heads_kv")
    if headdim_qk <= 0 or headdim_qk > 128:
        raise ValueError("headdim_qk must be in range (0, 128]")
    if headdim_v <= 0 or headdim_v > 128:
        raise ValueError("headdim_v must be in range (0, 128]")

    q_seq_scale_num = ceil_div(seqlen_q, 128)
    k_seq_scale_num = ceil_div(seqlen_kv, 128)
    v_seq_scale_num = 1
    if quant_mode == 0:
        q_seq_scale_num = 1
        k_seq_scale_num = 1
    elif quant_mode == 1:
        q_seq_scale_num = seqlen_q
        k_seq_scale_num = seqlen_kv
        v_seq_scale_num = seqlen_kv
    elif quant_mode == 2:
        pass
    elif quant_mode == 6:
        k_seq_scale_num = ceil_div(seqlen_kv, 128) * 128 // 16
    elif quant_mode == 7:
        v_seq_scale_num = ceil_div(seqlen_kv, 128)
    else:
        raise ValueError(f"Unsupported dense SageAttention quant_mode: {quant_mode}")

    if quant_mode == 0:
        q_scale_shape = (1, 1, 1, 1)
        k_scale_shape = (1, 1, 1, 1)
        v_scale_shape = (1, 1, 1, 1)
    else:
        q_scale_shape = (batch, q_seq_scale_num, num_heads, 1)
        k_scale_shape = (batch, k_seq_scale_num, num_heads_kv, 1)
        if quant_mode == 7:
            v_scale_shape = (batch, v_seq_scale_num, num_heads_kv, 1)
        elif quant_mode == 1:
            v_scale_shape = (batch, v_seq_scale_num, num_heads_kv, headdim_v)
        else:
            v_scale_shape = (batch, 1, num_heads_kv, headdim_v)
    for scale_tensor, scale_shape in (
        (q_scale_ffi, q_scale_shape),
        (k_scale_ffi, k_scale_shape),
        (v_scale_ffi, v_scale_shape),
    ):
        check_contiguous(scale_tensor, dim=-1)
        check_type(scale_tensor, torch.float32)
        check_shape(scale_tensor, scale_shape)

    module_dir = ensure_mubin_module_artifacts("sage_attention")
    dispatcher = get_sage_attention_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id = dispatcher.get_asm_id(
        q_ffi.dtype,
        k_ffi.dtype,
        v_ffi.dtype,
        is_causal,
        False,
        headdim_qk,
        quant_mode,
        fp8_output,
    )
    entry = dispatcher.resolve_kernel_entry(asm_id)
    kernel_path = dispatcher.resolve_kernel_path(asm_id, module_dir / "mubin")
    launcher_source = render_mubin_launcher(
        "sage_attention", func_name=entry.kernel_name, spec=asm_id
    )
    launch = get_mubin_launch_function(
        "sage_attention", entry.kernel_name, launcher_source
    )
    launch(
        str(kernel_path),
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
    out_ffi = out
    out_scale_ffi = out_scale
    out_lse_ffi = out_lse
    q_ffi = q
    k_cache_ffi = k_cache
    v_cache_ffi = v_cache
    page_table_ffi = page_table
    cache_seqlens_ffi = cache_seqlens
    q_scale_ffi = q_scale
    k_scale_ffi = k_scale
    v_scale_ffi = v_scale

    tensors = [
        out_ffi,
        out_lse_ffi,
        q_ffi,
        k_cache_ffi,
        v_cache_ffi,
        page_table_ffi,
        cache_seqlens_ffi,
        q_scale_ffi,
        k_scale_ffi,
        v_scale_ffi,
    ]
    if out_scale_ffi is not None:
        tensors.append(out_scale_ffi)

    check_musa(q_ffi)
    check_tensor_same_device(tensors)
    check_contiguous(q_ffi, dim=-1)
    check_contiguous(k_cache_ffi, dim=-1)
    check_contiguous(v_cache_ffi, dim=-1)
    check_contiguous(q_scale_ffi, dim=-1)
    check_contiguous(k_scale_ffi, dim=-1)
    check_contiguous(v_scale_ffi, dim=-1)
    check_contiguous(out_ffi)
    check_contiguous(out_lse_ffi)
    check_contiguous(page_table_ffi)
    check_contiguous(cache_seqlens_ffi)
    check_type(out_lse_ffi, torch.float32)
    check_type(page_table_ffi, torch.int32)
    check_type(cache_seqlens_ffi, torch.int32)

    fp8_output = out_scale_ffi is not None
    if q_ffi.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("q must be fp8")
    if k_cache_ffi.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("k_cache must be fp8")
    if v_cache_ffi.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("v_cache must be fp8")
    if fp8_output:
        if out_ffi.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("out must be fp8 when fp8_output is enabled")
        if out_ffi.dtype != v_cache_ffi.dtype:
            raise ValueError(
                "out must have the same dtype as v_cache when fp8_output is enabled"
            )
        check_type(out_scale_ffi, torch.float32)
    elif out_ffi.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("out must be fp16 or bf16 when fp8_output is disabled")

    batch, seqlen_q, num_heads, headdim_qk = tuple(q_ffi.shape)
    num_blocks, page_block_size, num_heads_kv, k_headdim_qk = tuple(k_cache_ffi.shape)
    v_num_blocks, v_page_block_size, v_num_heads_kv, headdim_v = tuple(
        v_cache_ffi.shape
    )
    check_shape(q_ffi, (batch, seqlen_q, num_heads, headdim_qk))
    check_shape(k_cache_ffi, (num_blocks, page_block_size, num_heads_kv, headdim_qk))
    check_shape(v_cache_ffi, (num_blocks, page_block_size, num_heads_kv, headdim_v))
    check_shape(out_ffi, (batch, seqlen_q, num_heads, headdim_v))
    check_shape(out_lse_ffi, (batch, num_heads, seqlen_q))
    check_shape(page_table_ffi, (batch, page_table_ffi.shape[1]))
    check_shape(cache_seqlens_ffi, (batch,))
    if fp8_output:
        check_shape(out_scale_ffi, (batch, seqlen_q, num_heads, 1))
    if (
        v_num_blocks != num_blocks
        or v_page_block_size != page_block_size
        or v_num_heads_kv != num_heads_kv
    ):
        raise ValueError("v_cache must match k_cache block/page/head dimensions")
    if k_headdim_qk != headdim_qk:
        raise ValueError("k_cache must have the same head dimension as q")
    if num_heads_kv <= 0:
        raise ValueError("num_heads_kv must be positive")
    if num_heads % num_heads_kv != 0:
        raise ValueError("num_heads must be divisible by num_heads_kv")
    if headdim_qk <= 0 or headdim_qk > 128:
        raise ValueError("headdim_qk must be in range (0, 128]")
    if headdim_v <= 0 or headdim_v > 128:
        raise ValueError("headdim_v must be in range (0, 128]")
    if page_block_size not in (64, 128):
        raise ValueError("page_block_size must be 64 or 128")

    q_seq_scale_num = ceil_div(seqlen_q, 128)
    k_scale_per_block = ceil_div(page_block_size, 128)
    k_scale_per_thread = ceil_div(page_block_size, 16)
    k_seq_scale_num = num_blocks * k_scale_per_block
    if quant_mode == 0:
        q_seq_scale_num = 1
        k_seq_scale_num = 1
    elif quant_mode == 1:
        q_seq_scale_num = seqlen_q
        k_seq_scale_num = num_blocks * page_block_size
    elif quant_mode == 2:
        pass
    elif quant_mode == 6:
        k_seq_scale_num = num_blocks * k_scale_per_thread
    else:
        raise ValueError(f"Unsupported KV-cache SageAttention quant_mode: {quant_mode}")

    if quant_mode == 0:
        q_scale_shape = (1, 1, 1, 1)
        k_scale_shape = (1, 1, 1, 1)
        v_scale_shape = (1, 1, 1, 1)
    else:
        q_scale_shape = (batch, q_seq_scale_num, num_heads, 1)
        k_scale_shape = (num_blocks, k_seq_scale_num // num_blocks, num_heads_kv, 1)
        if quant_mode == 1:
            v_scale_shape = (num_blocks, page_block_size, num_heads_kv, headdim_v)
        else:
            v_scale_shape = (num_blocks, 1, num_heads_kv, headdim_v)
    for scale_tensor, scale_shape in (
        (q_scale_ffi, q_scale_shape),
        (k_scale_ffi, k_scale_shape),
        (v_scale_ffi, v_scale_shape),
    ):
        check_contiguous(scale_tensor, dim=-1)
        check_type(scale_tensor, torch.float32)
        check_shape(scale_tensor, scale_shape)

    module_dir = ensure_mubin_module_artifacts("sage_attention")
    dispatcher = get_sage_attention_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id = dispatcher.get_asm_id(
        q_ffi.dtype,
        k_cache_ffi.dtype,
        v_cache_ffi.dtype,
        is_causal,
        True,
        headdim_qk,
        quant_mode,
        fp8_output,
    )
    entry = dispatcher.resolve_kernel_entry(asm_id)
    kernel_path = dispatcher.resolve_kernel_path(asm_id, module_dir / "mubin")
    launcher_source = render_mubin_launcher(
        "sage_attention", func_name=entry.kernel_name, spec=asm_id
    )
    launch = get_mubin_launch_function(
        "sage_attention", entry.kernel_name, launcher_source
    )
    launch(
        str(kernel_path),
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
