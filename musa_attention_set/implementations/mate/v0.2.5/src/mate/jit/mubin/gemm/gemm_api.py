from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from mate.artifacts import ensure_mubin_module_artifacts
from mate.mate_runtime import get_physical_num_mps, resolve_num_mps
from mate.utils import ceil_div

from ..common import (
    check_contiguous,
    check_musa,
    check_shape,
    check_tensor_same_device,
    check_type,
)
from ..launcher import get_mubin_launch_function, render_gemm_mubin_launcher
from .dispatch import get_moe_gemm_mubin_dispatcher
from .types import MoeGemmArgs, MxScaleMode, TensorMajor, TensorQuantMode


def _resolve_mp_counts(
    device: torch.device, num_mp: Optional[int] = None
) -> tuple[int, int]:
    return get_physical_num_mps(device), resolve_num_mps(device, num_mp)


def get_gemm_quant_axis_size(logical_size: int, quant_block: int) -> int:
    if quant_block == -1:
        return 1
    if quant_block <= 0:
        raise ValueError("quant_recipe entries must be positive or -1")
    return ceil_div(logical_size, quant_block)


def get_gemm_quant_mode_from_recipe(quant_recipe: Tuple[int, int]) -> TensorQuantMode:
    outer_block, inner_block = quant_recipe
    if outer_block == -1 and inner_block == -1:
        return TensorQuantMode.TENSOR
    if outer_block == 1 and inner_block == -1:
        return TensorQuantMode.CHANNEL
    if outer_block == 1 or inner_block == -1:
        return TensorQuantMode.GROUP
    return TensorQuantMode.BLOCK


def _quant_tile_from_recipes(
    a_quant_recipe: Tuple[int, int], b_quant_recipe: Tuple[int, int]
) -> int:
    a_k_quant_block = a_quant_recipe[1]
    b_k_quant_block = b_quant_recipe[1]
    if a_k_quant_block > 0 and b_k_quant_block > 0:
        if a_k_quant_block != b_k_quant_block:
            raise ValueError(
                "a_quant_recipe and b_quant_recipe must use the same k block size"
            )
        return a_k_quant_block
    if a_k_quant_block > 0:
        return a_k_quant_block
    if b_k_quant_block > 0:
        return b_k_quant_block
    return 1


def ragged_moe_gemm_8bit_mubin(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    ragged_tokens_info: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int],
    out: torch.Tensor,
    alignment_m: int,
    num_mp: Optional[int] = None,
):
    total_mp_count, target_mp_count = _resolve_mp_counts(input_a[0].device, num_mp)
    ragged_tokens_info_torch = ragged_tokens_info
    out_torch = out

    a = input_a[0]
    scale_a = input_a[1]
    b = input_b[0]
    scale_b = input_b[1]
    tensors = (a, b, scale_a, scale_b, ragged_tokens_info, out)

    check_musa(a)
    check_tensor_same_device(tensors)

    m, k = a.shape
    num_expert, n, _ = b.shape
    _, scale_granularity_n, scale_granularity_k = scale_granularity_mnk
    if scale_granularity_n != 128 or scale_granularity_k != 128:
        raise ValueError("scale_granularity_mnk must have n and k granularity 128")
    scale_k = ceil_div(k, scale_granularity_k)
    scale_n = ceil_div(n, scale_granularity_n)

    check_shape(a, (m, k))
    check_shape(b, (num_expert, n, k))
    check_shape(scale_a, (m, scale_k))
    check_shape(scale_b, (num_expert, scale_n, scale_k))
    check_shape(ragged_tokens_info, (m,))
    check_shape(out, (m, n))

    check_type(scale_a, torch.float32)
    check_type(scale_b, torch.float32)
    check_type(ragged_tokens_info, torch.int32)
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("a must be fp8")
    if b.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("b must be fp8")
    if out.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("out must be bf16 or fp16")

    check_contiguous(a, dim=-1)
    check_contiguous(b, dim=-1)
    check_contiguous(scale_a)
    check_contiguous(scale_b)
    check_contiguous(ragged_tokens_info)
    check_contiguous(out)

    if alignment_m not in (128, 256):
        raise ValueError("alignment_m must be 128 or 256")

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        m=m,
        n=n,
        k=k,
        num_expert=num_expert,
        quant_tile=scale_granularity_k,
        alignment_m=alignment_m,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=a.stride()[0],
        stride_batch_a=a.stride()[0] * a.shape[0],
        stride_n_b=b.stride()[1],
        stride_batch_b=b.stride()[0],
        stride_m_out=out.stride()[0],
        stride_batch_out=out.stride()[0] * out.shape[0],
        scale_a_m=scale_a.shape[0],
        scale_a_k=scale_a.shape[1],
        scale_a_nr_elem=scale_a.numel(),
        scale_b_n=scale_b.shape[1],
        scale_b_k=scale_b.shape[2],
        scale_b_nr_elem=scale_b.numel(),
        p_a=a,
        p_b=b,
        p_d=out,
        p_scale_a=scale_a,
        p_scale_b=scale_b,
        p_m_indices=ragged_tokens_info,
        p_signal=None,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_ragged_asm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        input_a[0],
        input_b[0],
        out_torch,
        input_a[1],
        input_b[1],
        None,
        ragged_tokens_info_torch,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        args.m,
        None,
        args.target_mp_count,
        None,
    )


def masked_moe_gemm_8bit_mubin(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    masked_tokens_info: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int],
    out: torch.Tensor,
    expect_tokens: Optional[int],
    signal: Optional[torch.Tensor],
    num_mp: Optional[int] = None,
):
    total_mp_count, target_mp_count = _resolve_mp_counts(input_a[0].device, num_mp)
    torch_out = out
    masked_tokens_info_torch = masked_tokens_info
    signal_torch = signal
    if expect_tokens is None:
        expect_tokens = 0

    a = input_a[0]
    scale_a = input_a[1]
    b = input_b[0]
    scale_b = input_b[1]
    signal_view = signal
    tensors = [a, b, scale_a, scale_b, masked_tokens_info, out]
    if signal_view is not None:
        tensors.append(signal_view)

    check_musa(a)
    check_tensor_same_device(tensors)

    num_expert, max_m, k = a.shape
    b_num_expert, n, b_k = b.shape
    _, scale_granularity_n, scale_granularity_k = scale_granularity_mnk
    if scale_granularity_n != 128 or scale_granularity_k != 128:
        raise ValueError("scale_granularity_mnk must have n and k granularity 128")
    scale_k = ceil_div(k, scale_granularity_k)
    scale_n = ceil_div(n, scale_granularity_n)

    check_shape(a, (num_expert, max_m, k))
    check_shape(b, (num_expert, n, k))
    check_shape(scale_a, (num_expert, max_m, scale_k))
    check_shape(scale_b, (num_expert, scale_n, scale_k))
    check_shape(masked_tokens_info, (num_expert,))
    check_shape(out, (num_expert, max_m, n))
    if b_num_expert != num_expert or b_k != k:
        raise ValueError("b must have shape (num_expert, n, k)")

    check_type(scale_a, torch.float32)
    check_type(scale_b, torch.float32)
    check_type(masked_tokens_info, torch.int32)
    if signal_view is not None:
        check_type(signal_view, torch.int32)
        tile_signal = 64
        check_shape(signal_view, (num_expert * ceil_div(max_m, tile_signal),))
        check_contiguous(signal_view)
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("a must be fp8")
    if b.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("b must be fp8")
    if out.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("out must be bf16 or fp16")

    check_contiguous(a, dim=-1)
    check_contiguous(b, dim=-1)
    if scale_a.stride()[-1] != 1 and scale_a.stride()[-2] != 1:
        raise ValueError("scale_a must be contiguous")
    check_contiguous(scale_b)
    check_contiguous(masked_tokens_info)
    check_contiguous(out)

    if max_m == 0 or n == 0:
        return None
    if k == 0:
        torch_out.zero_()
        return None

    major_scale_a = TensorMajor.MN if scale_a.stride()[-1] != 1 else TensorMajor.K
    if major_scale_a == TensorMajor.MN and scale_a.stride()[-1] != max_m:
        raise ValueError(
            "uncontiguous scale_a only supports max_m_per_group == scale_a.stride(-1)"
        )

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        m=max_m * num_expert,
        n=n,
        k=k,
        num_expert=num_expert,
        quant_tile=scale_granularity_k,
        expected_m=expect_tokens,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=a.stride()[1],
        stride_batch_a=a.stride()[0] * a.shape[0],
        stride_n_b=b.stride()[1],
        stride_batch_b=b.stride()[0],
        stride_m_out=out.stride()[1],
        stride_batch_out=out.stride()[0] * out.shape[0],
        scale_a_m=(
            max_m
            if major_scale_a == TensorMajor.MN
            else scale_a.shape[0] * scale_a.shape[1]
        ),
        scale_a_k=scale_a.shape[2],
        scale_a_nr_elem=scale_a.numel(),
        scale_b_n=scale_b.shape[1],
        scale_b_k=scale_b.shape[2],
        scale_b_nr_elem=scale_b.numel(),
        major_scale_a=major_scale_a,
        p_a=a,
        p_b=b,
        p_d=out,
        p_scale_a=scale_a,
        p_scale_b=scale_b,
        p_m_indices=masked_tokens_info,
        p_signal=signal_view,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_masked_asm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        input_a[0],
        input_b[0],
        torch_out,
        input_a[1],
        input_b[1],
        None,
        masked_tokens_info_torch,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        args.num_expert,
        signal_torch,
        args.target_mp_count,
        None,
    )

    if signal_view is not None:
        select_m = 128 if expect_tokens <= 128 else 256
        block_m = select_m
        block_n = 256
        return block_m, ceil_div(n, block_n)
    return None


def ragged_moe_gemm_w4a8_mubin(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    alignment_m: int,
    a_quant_recipe: Tuple[int, int],
    b_quant_recipe: Tuple[int, int],
    num_mp: Optional[int] = None,
    *,
    mixed_dtype: str = "s4fp8",
):
    total_mp_count, target_mp_count = _resolve_mp_counts(input_a[0].device, num_mp)
    if mixed_dtype not in ("s4fp8", "fp4fp8"):
        raise ValueError("mixed_dtype must be 's4fp8' or 'fp4fp8'")
    is_fp4 = mixed_dtype == "fp4fp8"
    expected_b_recipe = (1, 32) if is_fp4 else (1, 128)
    if (
        not isinstance(a_quant_recipe, tuple)
        or len(a_quant_recipe) != 2
        or any(type(item) is not int for item in a_quant_recipe)
    ):
        raise TypeError("a_quant_recipe must be a tuple of two int values")
    if (
        not isinstance(b_quant_recipe, tuple)
        or len(b_quant_recipe) != 2
        or any(type(item) is not int for item in b_quant_recipe)
    ):
        raise TypeError("b_quant_recipe must be a tuple of two int values")
    if a_quant_recipe != (1, -1):
        raise ValueError("a_quant_recipe must be (1, -1)")
    if b_quant_recipe != expected_b_recipe:
        raise ValueError(f"b_quant_recipe must be {expected_b_recipe}")

    a, scale_a = input_a
    packed_b, b_scales = input_b
    if is_fp4:
        if not isinstance(b_scales, tuple) or len(b_scales) != 2:
            raise TypeError("FP4FP8 input_b requires residual and epilogue scales")
        scale_b, scale_b1 = b_scales
    else:
        if not isinstance(b_scales, torch.Tensor):
            raise TypeError("S4FP8 input_b requires one scale tensor")
        scale_b = b_scales
        scale_b1 = None
    ragged_tokens_info_torch = ragged_tokens_info
    out_torch = out

    check_musa(a)
    tensors = [a, packed_b, scale_a, scale_b, out]
    if scale_b1 is not None:
        tensors.append(scale_b1)
    check_tensor_same_device(tensors)
    check_tensor_same_device((a, ragged_tokens_info))

    allowed_a_dtypes = (
        (torch.float8_e4m3fn,) if is_fp4 else (torch.float8_e4m3fn, torch.float8_e5m2)
    )
    if a.dtype not in allowed_a_dtypes:
        raise ValueError("a must be fp8")
    check_type(packed_b, torch.int8)
    check_type(scale_a, torch.float32)
    check_type(scale_b, torch.uint8 if is_fp4 else torch.bfloat16)
    if scale_b1 is not None:
        check_type(scale_b1, torch.float32)
    allowed_out_dtypes = (
        (torch.bfloat16,) if is_fp4 else (torch.bfloat16, torch.float16)
    )
    if out.dtype not in allowed_out_dtypes:
        raise ValueError("out must be bf16 or fp16")

    check_contiguous(a, dim=-1)
    check_contiguous(packed_b)
    check_contiguous(scale_b)
    if scale_b1 is not None:
        check_contiguous(scale_b1)
    check_contiguous(out)

    m, k = a.shape
    num_expert, n, packed_k = packed_b.shape
    scale_a_m = get_gemm_quant_axis_size(m, a_quant_recipe[0])
    scale_a_k = get_gemm_quant_axis_size(k, a_quant_recipe[1])
    scale_b_n = get_gemm_quant_axis_size(n, b_quant_recipe[0])
    scale_b_k = get_gemm_quant_axis_size(k, b_quant_recipe[1])

    check_shape(a, (m, k))
    check_shape(scale_a, (scale_a_m, scale_a_k))
    check_shape(packed_b, (num_expert, n, packed_k))
    check_shape(scale_b, (num_expert, scale_b_n, scale_b_k))
    if scale_b1 is not None:
        check_shape(scale_b1, (num_expert, n))
    check_shape(ragged_tokens_info, (m,))
    check_shape(out, (m, n))
    if packed_k * 2 != k:
        raise ValueError("packed_b must have logical k matching a")

    check_type(ragged_tokens_info, torch.int32)
    check_contiguous(scale_a)
    check_contiguous(ragged_tokens_info)

    if alignment_m not in ((256,) if is_fp4 else (128, 256)):
        raise ValueError("alignment_m must be 256 for FP4FP8, or 128/256 for S4FP8")
    if m == 0 or n == 0:
        return None
    if k == 0:
        out_torch.zero_()
        return None

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=packed_b.dtype,
        type_d=out.dtype,
        m=m,
        n=n,
        k=k,
        num_expert=num_expert,
        type_scale_a=scale_a.dtype,
        type_scale_b=scale_b.dtype,
        b_pack_bits=4,
        mx_scale=MxScaleMode.FP4_E8M0 if is_fp4 else MxScaleMode.NONE,
        n_split=True,
        quant_tile=_quant_tile_from_recipes(a_quant_recipe, b_quant_recipe),
        alignment_m=alignment_m,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=a.stride()[0],
        stride_batch_a=a.stride()[0] * a.shape[0],
        stride_n_b=packed_b.stride()[1],
        stride_batch_b=packed_b.stride()[0],
        stride_m_out=out.stride()[0],
        stride_batch_out=out.stride()[0] * out.shape[0],
        major_a=TensorMajor.K,
        major_b=TensorMajor.K,
        quant_mode_a=get_gemm_quant_mode_from_recipe(a_quant_recipe),
        quant_mode_b=get_gemm_quant_mode_from_recipe(b_quant_recipe),
        major_scale_a=TensorMajor.K,
        major_scale_b=TensorMajor.K,
        scale_a_m=scale_a_m,
        scale_a_k=scale_a_k,
        scale_a_nr_elem=scale_a.numel(),
        scale_b_n=scale_b_n,
        scale_b_k=scale_b_k,
        scale_b_nr_elem=scale_b.numel(),
        p_a=a,
        p_b=packed_b,
        p_d=out,
        p_scale_a=scale_a,
        p_scale_b=scale_b,
        p_m_indices=ragged_tokens_info,
        p_signal=None,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_w4a8_ragged_asm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        a,
        packed_b,
        out_torch,
        scale_a,
        scale_b,
        scale_b1,
        ragged_tokens_info_torch,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        args.m,
        None,
        args.target_mp_count,
        None,
    )
    return None


def masked_moe_gemm_w4a8_mubin(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    masked_tokens_info: torch.Tensor,
    out: torch.Tensor,
    expect_tokens: Optional[int],
    signal: Optional[torch.Tensor],
    a_quant_recipe: Tuple[int, int],
    b_quant_recipe: Tuple[int, int],
    num_mp: Optional[int] = None,
    *,
    mixed_dtype: str = "s4fp8",
):
    total_mp_count, target_mp_count = _resolve_mp_counts(input_a[0].device, num_mp)
    if expect_tokens is None:
        expect_tokens = 0
    if mixed_dtype not in ("s4fp8", "fp4fp8"):
        raise ValueError("mixed_dtype must be 's4fp8' or 'fp4fp8'")
    is_fp4 = mixed_dtype == "fp4fp8"
    expected_b_recipe = (1, 32) if is_fp4 else (1, 128)
    if (
        not isinstance(a_quant_recipe, tuple)
        or len(a_quant_recipe) != 2
        or any(type(item) is not int for item in a_quant_recipe)
    ):
        raise TypeError("a_quant_recipe must be a tuple of two int values")
    if (
        not isinstance(b_quant_recipe, tuple)
        or len(b_quant_recipe) != 2
        or any(type(item) is not int for item in b_quant_recipe)
    ):
        raise TypeError("b_quant_recipe must be a tuple of two int values")
    if a_quant_recipe != (1, -1):
        raise ValueError("a_quant_recipe must be (1, -1)")
    if b_quant_recipe != expected_b_recipe:
        raise ValueError(f"b_quant_recipe must be {expected_b_recipe}")

    a, scale_a = input_a
    packed_b, b_scales = input_b
    if is_fp4:
        if not isinstance(b_scales, tuple) or len(b_scales) != 2:
            raise TypeError("FP4FP8 input_b requires residual and epilogue scales")
        scale_b, scale_b1 = b_scales
    else:
        if not isinstance(b_scales, torch.Tensor):
            raise TypeError("S4FP8 input_b requires one scale tensor")
        scale_b = b_scales
        scale_b1 = None
    masked_tokens_info_torch = masked_tokens_info
    signal_torch = signal
    out_torch = out

    check_musa(a)
    input_tensors = [a, packed_b, scale_a, scale_b, out]
    if scale_b1 is not None:
        input_tensors.append(scale_b1)
    check_tensor_same_device(input_tensors)
    tensors = [a, masked_tokens_info]
    if signal is not None:
        tensors.append(signal)
    check_tensor_same_device(tensors)

    allowed_a_dtypes = (
        (torch.float8_e4m3fn,) if is_fp4 else (torch.float8_e4m3fn, torch.float8_e5m2)
    )
    if a.dtype not in allowed_a_dtypes:
        raise ValueError("a must be fp8")
    check_type(packed_b, torch.int8)
    check_type(scale_a, torch.float32)
    check_type(scale_b, torch.uint8 if is_fp4 else torch.bfloat16)
    if scale_b1 is not None:
        check_type(scale_b1, torch.float32)
    allowed_out_dtypes = (
        (torch.bfloat16,) if is_fp4 else (torch.bfloat16, torch.float16)
    )
    if out.dtype not in allowed_out_dtypes:
        raise ValueError("out must be bf16 or fp16")

    check_contiguous(a, dim=-1)
    check_contiguous(packed_b)
    check_contiguous(scale_b)
    if scale_b1 is not None:
        check_contiguous(scale_b1)
    check_contiguous(out)

    num_expert, max_m, k = a.shape
    b_num_expert, n, packed_k = packed_b.shape
    scale_a_m_per_expert = get_gemm_quant_axis_size(max_m, a_quant_recipe[0])
    scale_a_k = get_gemm_quant_axis_size(k, a_quant_recipe[1])
    scale_b_n = get_gemm_quant_axis_size(n, b_quant_recipe[0])
    scale_b_k = get_gemm_quant_axis_size(k, b_quant_recipe[1])

    check_shape(a, (num_expert, max_m, k))
    check_shape(scale_a, (num_expert, scale_a_m_per_expert, scale_a_k))
    check_shape(packed_b, (b_num_expert, n, packed_k))
    check_shape(scale_b, (b_num_expert, scale_b_n, scale_b_k))
    if scale_b1 is not None:
        check_shape(scale_b1, (b_num_expert, n))
    check_shape(masked_tokens_info, (num_expert,))
    check_shape(out, (num_expert, max_m, n))
    if b_num_expert != num_expert:
        raise ValueError("packed_b must have shape (num_expert, n, packed_k)")
    if packed_k * 2 != k:
        raise ValueError("packed_b must have logical k matching a")

    check_type(masked_tokens_info, torch.int32)
    if scale_a.stride()[-1] != 1 and scale_a.stride()[-2] != 1:
        raise ValueError("scale_a must be contiguous")
    check_contiguous(masked_tokens_info)
    if signal is not None:
        check_type(signal, torch.int32)
        tile_signal = 64
        check_shape(signal, (num_expert * ceil_div(max_m, tile_signal),))
        check_contiguous(signal)

    if max_m == 0 or n == 0:
        return None
    if k == 0:
        out_torch.zero_()
        return None

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=packed_b.dtype,
        type_d=out.dtype,
        m=max_m * num_expert,
        n=n,
        k=k,
        num_expert=num_expert,
        type_scale_a=scale_a.dtype,
        type_scale_b=scale_b.dtype,
        b_pack_bits=4,
        mx_scale=MxScaleMode.FP4_E8M0 if is_fp4 else MxScaleMode.NONE,
        n_split=True,
        quant_tile=_quant_tile_from_recipes(a_quant_recipe, b_quant_recipe),
        alignment_m=256 if is_fp4 else 0,
        expected_m=expect_tokens,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=a.stride()[1],
        stride_batch_a=a.stride()[0] * a.shape[0],
        stride_n_b=packed_b.stride()[1],
        stride_batch_b=packed_b.stride()[0],
        stride_m_out=out.stride()[1],
        stride_batch_out=out.stride()[0] * out.shape[0],
        major_a=TensorMajor.K,
        major_b=TensorMajor.K,
        quant_mode_a=get_gemm_quant_mode_from_recipe(a_quant_recipe),
        quant_mode_b=get_gemm_quant_mode_from_recipe(b_quant_recipe),
        major_scale_a=TensorMajor.K,
        major_scale_b=TensorMajor.K,
        scale_a_m=num_expert * scale_a_m_per_expert,
        scale_a_k=scale_a_k,
        scale_a_nr_elem=scale_a.numel(),
        scale_b_n=scale_b_n,
        scale_b_k=scale_b_k,
        scale_b_nr_elem=scale_b.numel(),
        p_a=a,
        p_b=packed_b,
        p_d=out,
        p_scale_a=scale_a,
        p_scale_b=scale_b,
        p_m_indices=masked_tokens_info,
        p_signal=signal,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_w4a8_masked_asm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        a,
        packed_b,
        out_torch,
        scale_a,
        scale_b,
        scale_b1,
        masked_tokens_info_torch,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        args.num_expert,
        signal_torch,
        args.target_mp_count,
        None,
    )

    if signal is not None:
        block_m = 256 if is_fp4 else (128 if expect_tokens <= 128 else 256)
        block_n = 256
        return block_m, ceil_div(n, block_n)
    return None


def ragged_k_moe_gemm_8bit_mubin(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    ragged_tokens_info: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int],
    out: torch.Tensor,
    num_mp: Optional[int] = None,
):
    total_mp_count, target_mp_count = _resolve_mp_counts(input_a[0].device, num_mp)
    ragged_tokens_info_torch = ragged_tokens_info
    out_torch = out

    a = input_a[0]
    scale_a = input_a[1]
    b = input_b[0]
    scale_b = input_b[1]
    tensors = (a, b, scale_a, scale_b, ragged_tokens_info, out)

    check_musa(a)
    check_tensor_same_device(tensors)

    k, m = a.shape
    b_k, n = b.shape
    scale_a_k, scale_a_m = scale_a.shape
    scale_b_k, scale_b_n = scale_b.shape
    num_expert = ragged_tokens_info.shape[0]
    if scale_granularity_mnk != (1, 1, 128):
        raise ValueError("ragged_k_moe_gemm_8bit_mubin only supports (1, 1, 128)")

    check_shape(a, (k, m))
    check_shape(b, (k, n))
    check_shape(scale_a, (scale_a_k, m))
    check_shape(scale_b, (scale_b_k, n))
    check_shape(ragged_tokens_info, (num_expert,))
    check_shape(out, (num_expert, m, n))
    if b_k != k:
        raise ValueError("b must have shape (k, n)")
    if scale_a_m != m:
        raise ValueError("scale_a must have shape (scale_k, m)")
    if scale_b_n != n:
        raise ValueError("scale_b must have shape (scale_k, n)")
    if scale_a_k != scale_b_k:
        raise ValueError("scale_a and scale_b must have matching scale_k")

    check_type(scale_a, torch.float32)
    check_type(scale_b, torch.float32)
    check_type(ragged_tokens_info, torch.int32)
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("a must be fp8")
    if b.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("b must be fp8")
    check_type(out, torch.float32)

    check_contiguous(a, dim=-1)
    check_contiguous(b, dim=-1)
    if scale_a.stride()[-1] != 1 and scale_a.stride()[-2] != 1:
        raise ValueError("scale_a must be contiguous")
    check_contiguous(scale_b)
    check_contiguous(ragged_tokens_info)
    check_contiguous(out)

    if k == 0:
        return None

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        m=m,
        n=n,
        k=k,
        num_expert=num_expert,
        quant_tile=128,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=a.stride()[1],
        stride_k_a=a.stride()[0],
        stride_batch_a=m * k,
        stride_n_b=b.stride()[1],
        stride_k_b=b.stride()[0],
        stride_batch_b=n * k,
        stride_m_out=out.stride()[1],
        stride_batch_out=out.stride()[0],
        major_a=TensorMajor.MN,
        major_b=TensorMajor.MN,
        quant_mode_a=TensorQuantMode.GROUP,
        quant_mode_b=TensorQuantMode.GROUP,
        major_scale_a=TensorMajor.MN,
        major_scale_b=TensorMajor.MN,
        scale_a_m=scale_a.shape[1],
        scale_a_k=scale_a.shape[0],
        scale_a_nr_elem=scale_a.numel(),
        scale_b_n=scale_b.shape[1],
        scale_b_k=scale_b.shape[0],
        scale_b_nr_elem=scale_b.numel(),
        p_a=a,
        p_b=b,
        p_d=out,
        p_scale_a=scale_a,
        p_scale_b=scale_b,
        p_m_indices=ragged_tokens_info,
        p_signal=None,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_k_contig_asm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        input_a[0],
        input_b[0],
        out_torch,
        input_a[1],
        input_b[1],
        None,
        ragged_tokens_info_torch,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        args.num_expert,
        None,
        args.target_mp_count,
        None,
    )
    return None


def m_grouped_contig_gemm_8bit_mubin(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    group_m_idx: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int],
    out: torch.Tensor,
    major_a_mode: str,
    major_b_mode: str,
    num_mp: Optional[int] = None,
):
    total_mp_count, target_mp_count = _resolve_mp_counts(input_a[0].device, num_mp)
    if major_a_mode != "K":
        raise ValueError("m_grouped_contig_gemm_8bit_mubin only supports K-major A")
    if major_b_mode not in ("K", "N"):
        raise ValueError("major_b_mode must be K or N")

    torch_out = out
    group_m_idx_torch = group_m_idx
    a = input_a[0]
    scale_a = input_a[1]
    b = input_b[0]
    scale_b = input_b[1]
    tensors = (a, b, scale_a, scale_b, group_m_idx, out)

    check_musa(a)
    check_tensor_same_device(tensors)

    m, k = a.shape
    num_expert = group_m_idx.shape[0]
    n = b.shape[-2] if major_b_mode == "K" else b.shape[-1]
    _, scale_granularity_n, scale_granularity_k = scale_granularity_mnk
    if scale_granularity_n != 128 or scale_granularity_k != 128:
        raise ValueError("scale_granularity_mnk must have n and k granularity 128")
    scale_k = ceil_div(k, scale_granularity_k)
    scale_n = ceil_div(n, scale_granularity_n)

    check_shape(a, (m, k))
    if major_b_mode == "K":
        check_shape(b, (num_expert, n, k))
        check_shape(scale_b, (num_expert, scale_n, scale_k))
    else:
        check_shape(b, (num_expert, k, n))
        check_shape(scale_b, (num_expert, scale_k, scale_n))
    check_shape(scale_a, (m, scale_k))
    check_shape(group_m_idx, (num_expert,))
    check_shape(out, (m, n))

    check_type(scale_a, torch.float32)
    check_type(scale_b, torch.float32)
    check_type(group_m_idx, torch.int32)
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("a must be fp8")
    if b.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("b must be fp8")
    if out.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("out must be bf16 or fp16")

    check_contiguous(a, dim=-1)
    check_contiguous(b, dim=-1)
    if scale_a.stride()[-1] != 1 and scale_a.stride()[-2] != 1:
        raise ValueError("scale_a must be contiguous")
    check_contiguous(scale_b)
    check_contiguous(group_m_idx)
    check_contiguous(out)

    if m == 0 or n == 0:
        return None
    if k == 0:
        torch_out.zero_()
        return None

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        m=m,
        n=n,
        k=k,
        num_expert=num_expert,
        quant_tile=scale_granularity_k,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=k,
        stride_k_a=m,
        stride_batch_a=m * k,
        stride_n_b=k,
        stride_k_b=n,
        stride_batch_b=n * k,
        stride_m_out=n,
        stride_batch_out=m * n,
        major_a=TensorMajor.K,
        major_b=TensorMajor.K if major_b_mode == "K" else TensorMajor.MN,
        quant_mode_a=TensorQuantMode.GROUP,
        quant_mode_b=TensorQuantMode.BLOCK,
        major_scale_a=TensorMajor.K,
        major_scale_b=TensorMajor.K if major_b_mode == "K" else TensorMajor.MN,
        scale_a_m=scale_a.shape[0],
        scale_a_k=scale_a.shape[-1],
        scale_a_nr_elem=scale_a.numel(),
        scale_b_n=scale_b.shape[-2] if major_b_mode == "K" else scale_b.shape[-1],
        scale_b_k=scale_b.shape[-1] if major_b_mode == "K" else scale_b.shape[-2],
        scale_b_nr_elem=scale_b.numel(),
        p_a=a,
        p_b=b,
        p_d=out,
        p_scale_a=scale_a,
        p_scale_b=scale_b,
        p_m_indices=group_m_idx,
        p_signal=None,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_ragged_expert_layout_asm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        input_a[0],
        input_b[0],
        torch_out,
        input_a[1],
        input_b[1],
        None,
        group_m_idx_torch,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        args.num_expert,
        None,
        args.target_mp_count,
        None,
    )
    return None


def ragged_moe_gemm_16bit_mubin(
    a: torch.Tensor,
    b: torch.Tensor,
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    use_psum_layout: bool,
    expected_m_for_psum_layout: Optional[int],
    alignment_m: int,
    num_mp: Optional[int] = None,
):
    total_mp_count, target_mp_count = _resolve_mp_counts(a.device, num_mp)
    torch_out = out
    a_torch = a
    b_torch = b
    ragged_tokens_info_torch = ragged_tokens_info
    tensors = (a, b, ragged_tokens_info, out)

    check_musa(a)
    check_tensor_same_device(tensors)

    m, k = a.shape
    num_expert, n, b_k = b.shape
    check_shape(a, (m, k))
    check_shape(b, (num_expert, n, k))
    check_shape(out, (m, n))
    if b_k != k:
        raise ValueError("b must have shape (num_expert, n, k)")
    if use_psum_layout:
        if expected_m_for_psum_layout is None:
            raise ValueError("expected_m_for_psum_layout must be set")
        raise ValueError("use_psum_layout must be false")
    check_shape(ragged_tokens_info, (m,))

    check_type(ragged_tokens_info, torch.int32)
    if a.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("a must be bf16 or fp16")
    if b.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("b must be bf16 or fp16")
    if out.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("out must be bf16 or fp16")
    if a.dtype != b.dtype or a.dtype != out.dtype:
        raise ValueError("a, b, out must have the same dtype")

    check_contiguous(a, dim=-1)
    check_contiguous(b, dim=-1)
    check_contiguous(ragged_tokens_info)
    check_contiguous(out)

    if m == 0 or n == 0:
        return None
    if k == 0:
        torch_out.zero_()
        return None

    if alignment_m not in (128, 256):
        raise ValueError("alignment_m must be 128 or 256")

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        m=m,
        n=n,
        k=k,
        num_expert=num_expert,
        quant_tile=128,
        alignment_m=alignment_m,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=a.stride()[0],
        stride_batch_a=a.stride()[0] * a.shape[0],
        stride_n_b=b.stride()[1],
        stride_batch_b=b.stride()[0],
        stride_m_out=out.stride()[0],
        stride_batch_out=out.stride()[0] * out.shape[0],
        quant_mode_a=TensorQuantMode.NO_QUANT,
        quant_mode_b=TensorQuantMode.NO_QUANT,
        p_a=a,
        p_b=b,
        p_d=out,
        p_m_indices=ragged_tokens_info,
        p_signal=None,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_ragged_asm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        a_torch,
        b_torch,
        torch_out,
        None,
        None,
        None,
        ragged_tokens_info_torch,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        args.m,
        None,
        args.target_mp_count,
        None,
    )


def masked_moe_gemm_16bit_mubin(
    a: torch.Tensor,
    b: torch.Tensor,
    masked_tokens_info: torch.Tensor,
    out: torch.Tensor,
    expect_tokens: Optional[int],
    signal: Optional[torch.Tensor],
    num_mp: Optional[int] = None,
):
    total_mp_count, target_mp_count = _resolve_mp_counts(a.device, num_mp)
    torch_out = out
    a_torch = a
    b_torch = b
    masked_tokens_info_torch = masked_tokens_info
    signal_torch = signal
    if expect_tokens is None:
        expect_tokens = 0

    signal_view = signal
    tensors = [a, b, masked_tokens_info, out]
    if signal_view is not None:
        tensors.append(signal_view)

    check_musa(a)
    check_tensor_same_device(tensors)

    num_expert, max_m, k = a.shape
    b_num_expert, n, b_k = b.shape
    check_shape(a, (num_expert, max_m, k))
    check_shape(b, (num_expert, n, k))
    check_shape(masked_tokens_info, (num_expert,))
    check_shape(out, (num_expert, max_m, n))
    if b_num_expert != num_expert or b_k != k:
        raise ValueError("b must have shape (num_expert, n, k)")

    check_type(masked_tokens_info, torch.int32)
    if signal_view is not None:
        check_type(signal_view, torch.int32)
        tile_signal = 64
        check_shape(signal_view, (num_expert * ceil_div(max_m, tile_signal),))
        check_contiguous(signal_view)
    if a.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("a must be bf16 or fp16")
    if b.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("b must be bf16 or fp16")
    if out.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("out must be bf16 or fp16")
    if a.dtype != b.dtype or a.dtype != out.dtype:
        raise ValueError("a, b, out must have the same dtype")

    check_contiguous(a, dim=-1)
    check_contiguous(b, dim=-1)
    check_contiguous(masked_tokens_info)
    check_contiguous(out)

    if max_m == 0 or n == 0:
        return None
    if k == 0:
        torch_out.zero_()
        return None

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        m=max_m * num_expert,
        n=n,
        k=k,
        num_expert=num_expert,
        quant_tile=128,
        expected_m=expect_tokens,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=a.stride()[1],
        stride_batch_a=a.stride()[0] * a.shape[0],
        stride_n_b=b.stride()[1],
        stride_batch_b=b.stride()[0],
        stride_m_out=out.stride()[1],
        stride_batch_out=out.stride()[0] * out.shape[0],
        quant_mode_a=TensorQuantMode.NO_QUANT,
        quant_mode_b=TensorQuantMode.NO_QUANT,
        p_a=a,
        p_b=b,
        p_d=out,
        p_m_indices=masked_tokens_info,
        p_signal=signal_view,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_masked_asm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        a_torch,
        b_torch,
        torch_out,
        None,
        None,
        None,
        masked_tokens_info_torch,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        args.num_expert,
        signal_torch,
        args.target_mp_count,
        None,
    )

    if signal_view is not None:
        select_m = 128 if expect_tokens <= 128 else 256
        block_m = select_m
        block_n = 256
        return block_m, ceil_div(n, block_n)
    return None


def ragged_k_moe_gemm_16bit_mubin(
    a: torch.Tensor,
    b: torch.Tensor,
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    num_mp: Optional[int] = None,
):
    total_mp_count, target_mp_count = _resolve_mp_counts(a.device, num_mp)
    a_torch = a
    b_torch = b
    ragged_tokens_info_torch = ragged_tokens_info
    out_torch = out

    tensors = (a, b, ragged_tokens_info, out)

    check_musa(a)
    check_tensor_same_device(tensors)

    k, m = a.shape
    b_k, n = b.shape
    num_expert = ragged_tokens_info.shape[0]
    check_shape(a, (k, m))
    check_shape(b, (k, n))
    check_shape(ragged_tokens_info, (num_expert,))
    check_shape(out, (num_expert, m, n))
    if b_k != k:
        raise ValueError("b must have shape (k, n)")

    check_type(ragged_tokens_info, torch.int32)
    if a.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("a must be bf16 or fp16")
    if b.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("b must be bf16 or fp16")
    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")
    if out.dtype == torch.bfloat16:
        if a.dtype != torch.bfloat16:
            raise ValueError("bf16 output requires bf16 a and b")
    elif out.dtype != torch.float32:
        raise ValueError("out must be fp32, or bf16 when a and b are bf16")

    check_contiguous(a, dim=-1)
    check_contiguous(b, dim=-1)
    check_contiguous(ragged_tokens_info)
    check_contiguous(out)

    if k == 0:
        return None

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        m=m,
        n=n,
        k=k,
        num_expert=num_expert,
        quant_tile=128,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=a.stride()[1],
        stride_k_a=a.stride()[0],
        stride_batch_a=m * k,
        stride_n_b=b.stride()[1],
        stride_k_b=b.stride()[0],
        stride_batch_b=n * k,
        stride_m_out=out.stride()[1],
        stride_batch_out=out.stride()[0],
        major_a=TensorMajor.MN,
        major_b=TensorMajor.MN,
        quant_mode_a=TensorQuantMode.NO_QUANT,
        quant_mode_b=TensorQuantMode.NO_QUANT,
        major_scale_a=TensorMajor.MN,
        major_scale_b=TensorMajor.MN,
        p_a=a,
        p_b=b,
        p_d=out,
        p_m_indices=ragged_tokens_info,
        p_signal=None,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_k_contig_asm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        a_torch,
        b_torch,
        out_torch,
        None,
        None,
        None,
        ragged_tokens_info_torch,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        args.num_expert,
        None,
        args.target_mp_count,
        None,
    )
    return None


def m_grouped_contig_gemm_16bit_mubin(
    a: torch.Tensor,
    b: torch.Tensor,
    group_m_idx: torch.Tensor,
    out: torch.Tensor,
    major_a_mode: str,
    major_b_mode: str,
    num_mp: Optional[int] = None,
):
    total_mp_count, target_mp_count = _resolve_mp_counts(a.device, num_mp)
    if major_a_mode != "K":
        raise ValueError("m_grouped_contig_gemm_16bit_mubin only supports K-major A")
    if major_b_mode not in ("K", "N"):
        raise ValueError("major_b_mode must be K or N")

    torch_out = out
    a_torch = a
    b_torch = b
    group_m_idx_torch = group_m_idx
    tensors = (a, b, group_m_idx, out)

    check_musa(a)
    check_tensor_same_device(tensors)

    m, k = a.shape
    num_expert = group_m_idx.shape[0]
    n = b.shape[-2] if major_b_mode == "K" else b.shape[-1]
    check_shape(a, (m, k))
    if major_b_mode == "K":
        check_shape(b, (num_expert, n, k))
    else:
        check_shape(b, (num_expert, k, n))
    check_shape(group_m_idx, (num_expert,))
    check_shape(out, (m, n))

    check_type(group_m_idx, torch.int32)
    if a.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("a must be bf16 or fp16")
    if b.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("b must be bf16 or fp16")
    if out.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("out must be bf16 or fp16")
    if a.dtype != b.dtype or a.dtype != out.dtype:
        raise ValueError("a, b, out must have the same dtype")

    check_contiguous(group_m_idx)
    check_contiguous(a, dim=-1)
    check_contiguous(b, dim=-1)
    check_contiguous(out)

    if m == 0 or n == 0:
        return None
    if k == 0:
        torch_out.zero_()
        return None

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        m=m,
        n=n,
        k=k,
        num_expert=num_expert,
        quant_tile=128,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=k,
        stride_k_a=m,
        stride_batch_a=m * k,
        stride_n_b=k,
        stride_k_b=n,
        stride_batch_b=n * k,
        stride_m_out=n,
        stride_batch_out=m * n,
        major_a=TensorMajor.K,
        major_b=TensorMajor.K if major_b_mode == "K" else TensorMajor.MN,
        quant_mode_a=TensorQuantMode.NO_QUANT,
        quant_mode_b=TensorQuantMode.NO_QUANT,
        major_scale_a=TensorMajor.K,
        major_scale_b=TensorMajor.K if major_b_mode == "K" else TensorMajor.MN,
        p_a=a,
        p_b=b,
        p_d=out,
        p_m_indices=group_m_idx,
        p_signal=None,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_ragged_expert_layout_asm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        a_torch,
        b_torch,
        torch_out,
        None,
        None,
        None,
        group_m_idx_torch,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        args.num_expert,
        None,
        args.target_mp_count,
        None,
    )
    return None


def groupwise_gemm_8bit_fp8output_mubin(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    scale_granularity_mnk: Tuple[int, int, int],
    out: torch.Tensor,
    output_scale: torch.Tensor,
    major_a_mode: str,
    major_b_mode: str,
    num_mp: Optional[int] = None,
):
    total_mp_count, target_mp_count = _resolve_mp_counts(input_a[0].device, num_mp)
    if major_a_mode not in ("K", "MN"):
        raise ValueError("major_a_mode must be K or MN")
    if major_b_mode not in ("K", "MN"):
        raise ValueError("major_b_mode must be K or MN")
    if scale_granularity_mnk != (1, 128, 128):
        raise ValueError(
            "groupwise_gemm_8bit_fp8output_mubin only supports (1, 128, 128)"
        )

    out_torch = out
    output_scale_torch = output_scale

    a = input_a[0]
    scale_a = input_a[1]
    b = input_b[0]
    scale_b = input_b[1]
    tensors = (a, b, scale_a, scale_b, out, output_scale)

    check_musa(a)
    check_tensor_same_device(tensors)

    m = a.shape[-2] if major_a_mode == "K" else a.shape[-1]
    k = a.shape[-1] if major_a_mode == "K" else a.shape[-2]
    n = b.shape[-2] if major_b_mode == "K" else b.shape[-1]
    b_k = b.shape[-1] if major_b_mode == "K" else b.shape[-2]
    scale_k = ceil_div(k, 128)
    scale_n = ceil_div(n, 128)

    if major_a_mode == "K":
        check_shape(a, (m, k))
        check_shape(scale_a, (m, scale_k))
    else:
        check_shape(a, (k, m))
        check_shape(scale_a, (scale_k, m))
    if major_b_mode == "K":
        check_shape(b, (n, k))
        check_shape(scale_b, (scale_n, scale_k))
    else:
        check_shape(b, (k, n))
        check_shape(scale_b, (scale_k, scale_n))
    check_shape(out, (m, n))
    check_shape(output_scale, (m, scale_n))
    if b_k != k:
        raise ValueError("a and b must have matching k")

    check_type(scale_a, torch.float32)
    check_type(scale_b, torch.float32)
    check_type(output_scale, torch.float32)
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("a must be fp8")
    if b.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("b must be fp8")
    check_type(out, torch.float8_e4m3fn)

    check_contiguous(a, dim=-1)
    check_contiguous(b, dim=-1)
    if scale_a.stride()[-1] != 1 and scale_a.stride()[-2] != 1:
        raise ValueError("scale_a must be contiguous")
    check_contiguous(scale_b)
    check_contiguous(out)
    check_contiguous(output_scale)

    if m == 0 or n == 0 or k == 0:
        return None

    args = MoeGemmArgs(
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        m=m,
        n=n,
        k=k,
        num_expert=1,
        quant_tile=128,
        total_mp_count=total_mp_count,
        target_mp_count=target_mp_count,
        stride_m_a=k,
        stride_k_a=m,
        stride_batch_a=m * k,
        stride_n_b=k,
        stride_k_b=n,
        stride_batch_b=n * k,
        stride_m_out=n,
        stride_batch_out=m * n,
        major_a=TensorMajor.K if major_a_mode == "K" else TensorMajor.MN,
        major_b=TensorMajor.K if major_b_mode == "K" else TensorMajor.MN,
        quant_mode_a=TensorQuantMode.GROUP,
        quant_mode_b=TensorQuantMode.BLOCK,
        major_scale_a=TensorMajor.K if major_a_mode == "K" else TensorMajor.MN,
        major_scale_b=TensorMajor.K if major_b_mode == "K" else TensorMajor.MN,
        scale_a_m=scale_a.shape[-2] if major_a_mode == "K" else scale_a.shape[-1],
        scale_a_k=scale_a.shape[-1] if major_a_mode == "K" else scale_a.shape[-2],
        scale_a_nr_elem=scale_a.numel(),
        scale_b_n=scale_b.shape[-2] if major_b_mode == "K" else scale_b.shape[-1],
        scale_b_k=scale_b.shape[-1] if major_b_mode == "K" else scale_b.shape[-2],
        scale_b_nr_elem=scale_b.numel(),
        p_a=a,
        p_b=b,
        p_d=out,
        p_scale_a=scale_a,
        p_scale_b=scale_b,
        p_scale_out=output_scale,
        p_m_indices=None,
        p_signal=None,
    )

    module_dir = ensure_mubin_module_artifacts("gemm")
    dispatcher = get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")
    asm_id, entry, kernel_path = dispatcher.resolve_kernel(
        args,
        dispatcher.get_normal_gemm_id,
        module_dir / "mubin",
    )
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    launch(
        str(kernel_path),
        input_a[0],
        input_b[0],
        out_torch,
        input_a[1],
        input_b[1],
        None,
        None,
        args.m,
        args.n,
        args.k,
        args.num_expert,
        args.batch,
        args.quant_tile,
        args.total_mp_count,
        args.stride_m_a,
        args.stride_k_a,
        args.stride_batch_a,
        args.stride_n_b,
        args.stride_k_b,
        args.stride_batch_b,
        args.stride_m_out,
        args.stride_batch_out,
        args.scale_a_m,
        args.scale_a_k,
        args.scale_a_nr_elem,
        args.scale_b_n,
        args.scale_b_k,
        args.scale_b_nr_elem,
        0,
        None,
        args.target_mp_count,
        output_scale_torch,
    )
    return None
