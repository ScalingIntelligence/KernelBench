from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from mate.artifacts import ensure_mubin_module_artifacts

from ..common import (
    check_contiguous,
    check_musa,
    check_shape,
    check_tensor_same_device,
    check_type,
    get_asm_dtype_from_torch_dtype,
)
from ..launcher import get_mubin_launch_function, render_mubin_launcher
from .dispatch import get_flash_mla_mubin_dispatcher


_FLASH_MLA_METADATA_SIZE = 8


def _check_same_dtype(tensors: Tuple, expected_dtype) -> None:
    for tensor in tensors:
        if tensor.dtype != expected_dtype:
            raise ValueError("q_nope, q_pe, ckv, and kpe must have the same dtype")


def _check_rank(tensor, rank: int, name: str) -> None:
    if len(tuple(tensor.shape)) != rank:
        raise ValueError(f"{name} must be {rank}D")


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
    q_nope_torch = q_nope
    q_pe_torch = q_pe
    ckv_torch = ckv
    kpe_torch = kpe
    seqlens_k_torch = seqlens_k
    block_table_torch = block_table
    tile_scheduler_metadata_torch = tile_scheduler_metadata
    num_splits_torch = num_splits
    out_torch = out
    out_lse_torch = out_lse
    cu_seqlens_q_torch = cu_seqlens_q

    q_nope_ffi = q_nope_torch
    q_pe_ffi = q_pe_torch
    ckv_ffi = ckv_torch
    kpe_ffi = kpe_torch
    seqlens_k_ffi = seqlens_k_torch
    block_table_ffi = block_table_torch
    tile_scheduler_metadata_ffi = tile_scheduler_metadata_torch
    num_splits_ffi = num_splits_torch
    out_ffi = out_torch
    out_lse_ffi = out_lse_torch
    cu_seqlens_q_ffi = cu_seqlens_q_torch

    tensors = [
        q_nope_ffi,
        q_pe_ffi,
        ckv_ffi,
        kpe_ffi,
        seqlens_k_ffi,
        block_table_ffi,
        tile_scheduler_metadata_ffi,
        num_splits_ffi,
        out_ffi,
        out_lse_ffi,
    ]
    if cu_seqlens_q_ffi is not None:
        tensors.append(cu_seqlens_q_ffi)

    check_musa(q_nope_ffi)
    check_tensor_same_device(tensors)
    check_contiguous(q_nope_ffi, dim=-1)
    check_contiguous(q_pe_ffi, dim=-1)
    check_contiguous(ckv_ffi, dim=-1)
    check_contiguous(kpe_ffi, dim=-1)
    check_contiguous(tile_scheduler_metadata_ffi)
    check_contiguous(num_splits_ffi)
    check_contiguous(seqlens_k_ffi)
    check_contiguous(out_ffi, dim=-1)
    check_contiguous(out_lse_ffi, dim=-1)
    if block_table_ffi.stride()[-1] != 1:
        raise ValueError("block_table must have contiguous last dimension")

    check_type(tile_scheduler_metadata_ffi, torch.int32)
    check_type(num_splits_ffi, torch.int32)
    check_type(seqlens_k_ffi, torch.int32)
    check_type(block_table_ffi, torch.int32)
    check_type(out_lse_ffi, torch.float32)
    if q_nope_ffi.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("flash_mla_asm_mubin only supports fp16 and bf16")
    _check_same_dtype((q_pe_ffi, ckv_ffi, kpe_ffi), q_nope_ffi.dtype)
    check_type(out_ffi, q_nope_ffi.dtype)

    is_varlen_q = cu_seqlens_q_ffi is not None
    if is_varlen_q:
        if max_seqlen_q is None:
            raise ValueError("max_seqlen_q must be provided when cu_seqlens_q is set")
        check_contiguous(cu_seqlens_q_ffi)
        check_type(cu_seqlens_q_ffi, torch.int32)
        _check_rank(q_nope_ffi, 3, "q_nope")
        _check_rank(q_pe_ffi, 3, "q_pe")
        batch = cu_seqlens_q_ffi.shape[0] - 1
        seqlen_q = max_seqlen_q
        total_q = q_nope_ffi.shape[0]
    else:
        _check_rank(q_nope_ffi, 4, "q_nope")
        _check_rank(q_pe_ffi, 4, "q_pe")
        batch = q_nope_ffi.shape[0]
        seqlen_q = q_nope_ffi.shape[1]
        total_q = batch * seqlen_q

    num_heads = q_nope_ffi.shape[-2]
    head_dim_latent = q_nope_ffi.shape[-1]
    head_dim_rope = q_pe_ffi.shape[-1]
    head_dim_v = ckv_ffi.shape[-1]
    num_blocks = ckv_ffi.shape[0]
    page_block_size = ckv_ffi.shape[1]
    if head_dim_latent != 512:
        raise ValueError("q_nope last dim must be 512")
    if head_dim_rope != 64:
        raise ValueError("q_pe last dim must be 64")
    if head_dim_v != 512:
        raise ValueError("ckv last dim must be 512")
    if page_block_size != 64:
        raise ValueError("page block size must be 64")
    if q_nope_ffi.shape[-2] != q_pe_ffi.shape[-2]:
        raise ValueError("q_nope and q_pe head counts must match")
    if not is_varlen_q and (q_nope_ffi.shape[:2] != q_pe_ffi.shape[:2]):
        raise ValueError("q_nope and q_pe batch/sequence dims must match")
    if is_varlen_q and q_nope_ffi.shape[0] != q_pe_ffi.shape[0]:
        raise ValueError("q_nope and q_pe total_q dims must match")

    if len(ckv_ffi.shape) == 4:
        check_shape(ckv_ffi, (num_blocks, page_block_size, 1, 512))
        check_shape(kpe_ffi, (num_blocks, page_block_size, 1, 64))
        if ckv_ffi.stride()[-3] % 576 != 0 or kpe_ffi.stride()[-3] % 576 != 0:
            raise ValueError("ckv/kpe head strides must be multiples of 576")
    elif len(ckv_ffi.shape) == 3:
        check_shape(ckv_ffi, (num_blocks, page_block_size, 512))
        check_shape(kpe_ffi, (num_blocks, page_block_size, 64))
        if ckv_ffi.stride()[-2] % 576 != 0 or kpe_ffi.stride()[-2] % 576 != 0:
            raise ValueError("ckv/kpe page strides must be multiples of 576")
    else:
        raise ValueError("ckv must be 3D or 4D")
    if ckv_ffi.stride()[0] % 576 != 0 or kpe_ffi.stride()[0] % 576 != 0:
        raise ValueError("ckv/kpe block strides must be multiples of 576")

    check_shape(seqlens_k_ffi, (batch,))
    if len(block_table_ffi.shape) != 2 or block_table_ffi.shape[0] != batch:
        raise ValueError("block_table must have shape (batch, max_num_blocks_per_seq)")
    if len(tile_scheduler_metadata_ffi.shape) != 2:
        raise ValueError("tile_scheduler_metadata must be 2D")
    if tile_scheduler_metadata_ffi.shape[1] != _FLASH_MLA_METADATA_SIZE:
        raise ValueError("tile_scheduler_metadata has unexpected second dimension")
    check_shape(num_splits_ffi, (batch + 1,))

    if not is_varlen_q:
        check_shape(out_ffi, (batch, seqlen_q, num_heads, 512))
        check_shape(out_lse_ffi, (batch, seqlen_q, num_heads))
        if out_ffi.stride()[1] != out_ffi.shape[-2] * out_ffi.stride()[2]:
            raise ValueError("out sequence layout is not representable")
        if out_lse_ffi.stride()[1] != out_lse_ffi.shape[-1] * out_lse_ffi.stride()[2]:
            raise ValueError("out_lse sequence layout is not representable")
    else:
        check_shape(out_ffi, (total_q, num_heads, 512))
        check_shape(out_lse_ffi, (num_heads, total_q))

    module_dir = ensure_mubin_module_artifacts("flash_mla")
    dispatcher = get_flash_mla_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id = dispatcher.get_asm_id(q_nope_ffi.dtype, is_causal, is_varlen_q)
    entry = dispatcher.resolve_kernel_entry(asm_id)
    kernel_path = dispatcher.resolve_kernel_path(asm_id, module_dir / "mubin")
    launcher_spec = SimpleNamespace(
        dtype=get_asm_dtype_from_torch_dtype(asm_id.dtype),
        is_varlen_q=asm_id.is_varlen_q,
    )
    launcher_source = render_mubin_launcher(
        "flash_mla", func_name=entry.kernel_name, spec=launcher_spec
    )
    launch = get_mubin_launch_function("flash_mla", entry.kernel_name, launcher_source)
    launch(
        str(kernel_path),
        q_nope_torch,
        q_pe_torch,
        ckv_torch,
        kpe_torch,
        seqlens_k_torch,
        block_table_torch,
        tile_scheduler_metadata_torch,
        num_splits_torch,
        out_torch,
        out_lse_torch,
        softmax_scale,
        cu_seqlens_q_torch,
        max_seqlen_q,
    )
    return None
