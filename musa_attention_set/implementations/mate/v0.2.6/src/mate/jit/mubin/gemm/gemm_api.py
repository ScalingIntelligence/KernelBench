from __future__ import annotations

import functools
from typing import Optional, Tuple, Union, cast

import torch

from mate.artifacts import ensure_mubin_module_artifacts
from mate.mate_runtime import get_physical_num_mps
from mate.utils import ceil_div

from ..launcher import get_mubin_launch_function, render_gemm_mubin_launcher
from .dispatch import (
    RAGGED_EXPERT_LAYOUT_BLOCK_CANDIDATES,
    MoeGemmMubinDispatcher,
    get_moe_gemm_mubin_dispatcher,
)
from .types import (
    MxScaleMode,
    TensorMajor,
    TensorQuantMode,
)


@functools.cache
def _get_gemm_mubin_runtime() -> MoeGemmMubinDispatcher:
    module_dir = ensure_mubin_module_artifacts("gemm")
    return get_moe_gemm_mubin_dispatcher(module_dir / "kernel_map.json")


@functools.cache
def _get_gemm_mubin_launch_plan(
    get_asm_id: str,
    **dispatch_args,
):
    dispatcher = _get_gemm_mubin_runtime()
    asm_id = getattr(dispatcher, get_asm_id)(**dispatch_args)
    entry, kernel_path = dispatcher.resolve_kernel_artifact(asm_id)
    launch = get_mubin_launch_function(
        "gemm",
        entry.kernel_name,
        render_gemm_mubin_launcher(asm_id, entry.kernel_name),
    )
    return str(kernel_path), launch


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


@functools.lru_cache(maxsize=256)
def _estimate_optimal_block(
    m: int,
    n: int,
    num_expert: int,
    num_mp: int,
) -> int:
    dispatcher = _get_gemm_mubin_runtime()
    best_score = -1.0
    best_candidate_id = 0
    for candidate_id, (block, base_score) in enumerate(
        RAGGED_EXPERT_LAYOUT_BLOCK_CANDIDATES
    ):
        score = dispatcher.estimate_block_score(
            block,
            m,
            n,
            num_expert,
            num_mp,
            base_score,
        )

        if score > best_score:
            best_score = score
            best_candidate_id = candidate_id

    return best_candidate_id


def ragged_moe_gemm_8bit_mubin(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    ragged_tokens_info: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int],
    out: torch.Tensor,
    alignment_m: int,
    num_mp: Optional[int] = None,
):
    a, scale_a = input_a
    b, scale_b = input_b

    m = a.shape[0]
    num_expert = b.shape[0]
    _, scale_granularity_n, scale_granularity_k = scale_granularity_mnk
    if scale_granularity_n != 128 or scale_granularity_k != 128:
        raise ValueError("scale_granularity_mnk must have n and k granularity 128")
    if alignment_m not in (128, 256):
        raise ValueError("alignment_m must be 128 or 256")

    tme_cache_hint_b = num_expert > 0 and m / num_expert <= 192
    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_ragged_asm_id",
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        tme_cache_hint_b=tme_cache_hint_b,
        quant_tile=scale_granularity_k,
        alignment_m=alignment_m,
        quant_mode_a=TensorQuantMode.GROUP,
        quant_mode_b=TensorQuantMode.BLOCK,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        b,  # b
        out,  # out
        scale_a,  # scale_a
        scale_b,  # scale_b
        None,  # scale_b1
        ragged_tokens_info,  # token_info
        None,  # signal
        None,  # output_scale
        num_mp,  # num_target_mp
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
    if expect_tokens is None:
        expect_tokens = 0

    a, scale_a = input_a
    b, scale_b = input_b

    _, n, _ = b.shape
    _, scale_granularity_n, scale_granularity_k = scale_granularity_mnk
    if scale_granularity_n != 128 or scale_granularity_k != 128:
        raise ValueError("scale_granularity_mnk must have n and k granularity 128")
    major_scale_a = TensorMajor.MN if scale_a.stride()[-1] != 1 else TensorMajor.K
    alignment_m = 128 if expect_tokens <= 128 else 256
    tme_cache_hint_b = alignment_m == 128 or (
        alignment_m == 256 and expect_tokens <= 256
    )

    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_masked_asm_id",
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        quant_tile=scale_granularity_k,
        alignment_m=alignment_m,
        tme_cache_hint_b=tme_cache_hint_b,
        major_scale_a=major_scale_a,
        quant_mode_a=TensorQuantMode.GROUP,
        quant_mode_b=TensorQuantMode.BLOCK,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        b,  # b
        out,  # out
        scale_a,  # scale_a
        scale_b,  # scale_b
        None,  # scale_b1
        masked_tokens_info,  # token_info
        signal,  # signal
        None,  # output_scale
        num_mp,  # num_target_mp
    )

    if signal is not None:
        block_m = alignment_m
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
    if mixed_dtype not in ("s4fp8", "fp4fp8"):
        raise ValueError("mixed_dtype must be 's4fp8' or 'fp4fp8'")
    is_fp4 = mixed_dtype == "fp4fp8"
    expected_b_recipe = (1, 32) if is_fp4 else (1, 128)
    if a_quant_recipe != (1, -1):
        raise ValueError("a_quant_recipe must be (1, -1)")
    if b_quant_recipe != expected_b_recipe:
        raise ValueError(f"b_quant_recipe must be {expected_b_recipe}")

    a, scale_a = input_a
    packed_b, b_scales = input_b
    if is_fp4:
        scale_b, scale_b1 = cast(Tuple[torch.Tensor, torch.Tensor], b_scales)
    else:
        scale_b = cast(torch.Tensor, b_scales)
        scale_b1 = None
    m = a.shape[0]
    num_expert = packed_b.shape[0]

    if alignment_m not in ((256,) if is_fp4 else (128, 256)):
        raise ValueError("alignment_m must be 256 for FP4FP8, or 128/256 for S4FP8")

    quant_tile = _quant_tile_from_recipes(a_quant_recipe, b_quant_recipe)
    quant_mode_a = get_gemm_quant_mode_from_recipe(a_quant_recipe)
    quant_mode_b = get_gemm_quant_mode_from_recipe(b_quant_recipe)
    mx_scale = MxScaleMode.FP4_E8M0 if is_fp4 else MxScaleMode.NONE
    tme_cache_hint_b = num_expert > 0 and m / num_expert <= 192
    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_w4a8_ragged_asm_id",
        type_a=a.dtype,
        type_d=out.dtype,
        type_scale_b=scale_b.dtype,
        tme_cache_hint_b=tme_cache_hint_b,
        mx_scale=mx_scale,
        quant_tile=quant_tile,
        alignment_m=alignment_m,
        quant_mode_a=quant_mode_a,
        quant_mode_b=quant_mode_b,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        packed_b,  # b
        out,  # out
        scale_a,  # scale_a
        scale_b,  # scale_b
        scale_b1,  # scale_b1
        ragged_tokens_info,  # token_info
        None,  # signal
        None,  # output_scale
        num_mp,  # num_target_mp
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
    if expect_tokens is None:
        expect_tokens = 0
    if mixed_dtype not in ("s4fp8", "fp4fp8"):
        raise ValueError("mixed_dtype must be 's4fp8' or 'fp4fp8'")
    is_fp4 = mixed_dtype == "fp4fp8"
    expected_b_recipe = (1, 32) if is_fp4 else (1, 128)
    if a_quant_recipe != (1, -1):
        raise ValueError("a_quant_recipe must be (1, -1)")
    if b_quant_recipe != expected_b_recipe:
        raise ValueError(f"b_quant_recipe must be {expected_b_recipe}")

    a, scale_a = input_a
    packed_b, b_scales = input_b
    if is_fp4:
        scale_b, scale_b1 = cast(Tuple[torch.Tensor, torch.Tensor], b_scales)
    else:
        scale_b = cast(torch.Tensor, b_scales)
        scale_b1 = None
    _, n, _ = packed_b.shape
    quant_tile = _quant_tile_from_recipes(a_quant_recipe, b_quant_recipe)
    quant_mode_a = get_gemm_quant_mode_from_recipe(a_quant_recipe)
    quant_mode_b = get_gemm_quant_mode_from_recipe(b_quant_recipe)
    mx_scale = MxScaleMode.FP4_E8M0 if is_fp4 else MxScaleMode.NONE
    alignment_m = 256 if is_fp4 else (128 if expect_tokens <= 128 else 256)
    tme_cache_hint_b = alignment_m == 128 or (
        alignment_m == 256 and expect_tokens <= 256
    )
    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_w4a8_masked_asm_id",
        type_a=a.dtype,
        type_d=out.dtype,
        type_scale_b=scale_b.dtype,
        mx_scale=mx_scale,
        quant_tile=quant_tile,
        alignment_m=alignment_m,
        tme_cache_hint_b=tme_cache_hint_b,
        quant_mode_a=quant_mode_a,
        quant_mode_b=quant_mode_b,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        packed_b,  # b
        out,  # out
        scale_a,  # scale_a
        scale_b,  # scale_b
        scale_b1,  # scale_b1
        masked_tokens_info,  # token_info
        signal,  # signal
        None,  # output_scale
        num_mp,  # num_target_mp
    )

    if signal is not None:
        block_m = alignment_m
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
    a, scale_a = input_a
    b, scale_b = input_b
    if scale_granularity_mnk != (1, 1, 128):
        raise ValueError("ragged_k_moe_gemm_8bit_mubin only supports (1, 1, 128)")

    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_k_contig_asm_id",
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        quant_tile=128,
        quant_mode_a=TensorQuantMode.GROUP,
        quant_mode_b=TensorQuantMode.GROUP,
        major_scale_a=TensorMajor.MN,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        b,  # b
        out,  # out
        scale_a,  # scale_a
        scale_b,  # scale_b
        None,  # scale_b1
        ragged_tokens_info,  # token_info
        None,  # signal
        None,  # output_scale
        num_mp,  # num_target_mp
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
    dispatch_num_mp = get_physical_num_mps(input_a[0].device)
    if major_a_mode != "K":
        raise ValueError("m_grouped_contig_gemm_8bit_mubin only supports K-major A")
    if major_b_mode not in ("K", "N"):
        raise ValueError("major_b_mode must be K or N")

    a, scale_a = input_a
    b, scale_b = input_b

    m = a.shape[0]
    num_expert = group_m_idx.shape[0]
    n = b.shape[-2] if major_b_mode == "K" else b.shape[-1]
    _, scale_granularity_n, scale_granularity_k = scale_granularity_mnk
    if scale_granularity_n != 128 or scale_granularity_k != 128:
        raise ValueError("scale_granularity_mnk must have n and k granularity 128")
    major_b = TensorMajor.K if major_b_mode == "K" else TensorMajor.MN
    candidate_block_id = _estimate_optimal_block(m, n, num_expert, dispatch_num_mp)
    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_ragged_expert_layout_asm_id",
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        candidate_block_id=candidate_block_id,
        quant_tile=scale_granularity_k,
        major_b=major_b,
        quant_mode_a=TensorQuantMode.GROUP,
        quant_mode_b=TensorQuantMode.BLOCK,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        b,  # b
        out,  # out
        scale_a,  # scale_a
        scale_b,  # scale_b
        None,  # scale_b1
        group_m_idx,  # token_info
        None,  # signal
        None,  # output_scale
        num_mp,  # num_target_mp
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
    m = a.shape[0]
    num_expert = b.shape[0]
    if use_psum_layout:
        if expected_m_for_psum_layout is None:
            raise ValueError("expected_m_for_psum_layout must be set")
        raise ValueError("use_psum_layout must be false")

    if alignment_m not in (128, 256):
        raise ValueError("alignment_m must be 128 or 256")

    tme_cache_hint_b = num_expert > 0 and m / num_expert <= 192
    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_ragged_asm_id",
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        tme_cache_hint_b=tme_cache_hint_b,
        quant_tile=128,
        alignment_m=alignment_m,
        quant_mode_a=TensorQuantMode.NO_QUANT,
        quant_mode_b=TensorQuantMode.NO_QUANT,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        b,  # b
        out,  # out
        None,  # scale_a
        None,  # scale_b
        None,  # scale_b1
        ragged_tokens_info,  # token_info
        None,  # signal
        None,  # output_scale
        num_mp,  # num_target_mp
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
    if expect_tokens is None:
        expect_tokens = 0

    _, n, _ = b.shape
    alignment_m = 128 if expect_tokens <= 128 else 256
    tme_cache_hint_b = alignment_m == 128 or (
        alignment_m == 256 and expect_tokens <= 256
    )

    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_masked_asm_id",
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        quant_tile=128,
        alignment_m=alignment_m,
        tme_cache_hint_b=tme_cache_hint_b,
        quant_mode_a=TensorQuantMode.NO_QUANT,
        quant_mode_b=TensorQuantMode.NO_QUANT,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        b,  # b
        out,  # out
        None,  # scale_a
        None,  # scale_b
        None,  # scale_b1
        masked_tokens_info,  # token_info
        signal,  # signal
        None,  # output_scale
        num_mp,  # num_target_mp
    )

    if signal is not None:
        block_m = alignment_m
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
    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_k_contig_asm_id",
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        quant_tile=128,
        quant_mode_a=TensorQuantMode.NO_QUANT,
        quant_mode_b=TensorQuantMode.NO_QUANT,
        major_scale_a=TensorMajor.MN,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        b,  # b
        out,  # out
        None,  # scale_a
        None,  # scale_b
        None,  # scale_b1
        ragged_tokens_info,  # token_info
        None,  # signal
        None,  # output_scale
        num_mp,  # num_target_mp
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
    dispatch_num_mp = get_physical_num_mps(a.device)
    if major_a_mode != "K":
        raise ValueError("m_grouped_contig_gemm_16bit_mubin only supports K-major A")
    if major_b_mode not in ("K", "N"):
        raise ValueError("major_b_mode must be K or N")

    m = a.shape[0]
    num_expert = group_m_idx.shape[0]
    n = b.shape[-2] if major_b_mode == "K" else b.shape[-1]

    major_b = TensorMajor.K if major_b_mode == "K" else TensorMajor.MN
    candidate_block_id = _estimate_optimal_block(m, n, num_expert, dispatch_num_mp)
    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_ragged_expert_layout_asm_id",
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        candidate_block_id=candidate_block_id,
        quant_tile=128,
        major_b=major_b,
        quant_mode_a=TensorQuantMode.NO_QUANT,
        quant_mode_b=TensorQuantMode.NO_QUANT,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        b,  # b
        out,  # out
        None,  # scale_a
        None,  # scale_b
        None,  # scale_b1
        group_m_idx,  # token_info
        None,  # signal
        None,  # output_scale
        num_mp,  # num_target_mp
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
    if major_a_mode not in ("K", "MN"):
        raise ValueError("major_a_mode must be K or MN")
    if major_b_mode not in ("K", "MN"):
        raise ValueError("major_b_mode must be K or MN")
    if scale_granularity_mnk != (1, 128, 128):
        raise ValueError(
            "groupwise_gemm_8bit_fp8output_mubin only supports (1, 128, 128)"
        )

    a, scale_a = input_a
    b, scale_b = input_b
    major_a = TensorMajor.K if major_a_mode == "K" else TensorMajor.MN
    major_b = TensorMajor.K if major_b_mode == "K" else TensorMajor.MN
    kernel_path, launch = _get_gemm_mubin_launch_plan(
        "get_normal_gemm_id",
        type_a=a.dtype,
        type_b=b.dtype,
        type_d=out.dtype,
        quant_tile=128,
        major_a=major_a,
        major_b=major_b,
    )
    launch(
        kernel_path,  # object_path
        a,  # a
        b,  # b
        out,  # out
        scale_a,  # scale_a
        scale_b,  # scale_b
        None,  # scale_b1
        None,  # token_info
        None,  # signal
        output_scale,  # output_scale
        num_mp,  # num_target_mp
    )
    return None
