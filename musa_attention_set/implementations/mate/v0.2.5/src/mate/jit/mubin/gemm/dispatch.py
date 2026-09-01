from dataclasses import asdict, replace
import functools
import hashlib
import json
from pathlib import Path
from typing import Callable

import torch

from mate.artifacts import ensure_mubin_kernel_artifact, load_kernel_map
from mate.utils import ceil_div

from ..common import (
    MP31_ARCH,
    get_asm_dtype_from_torch_dtype,
)
from .types import (
    AsmDType,
    GemmKernelBlock,
    GemmMubinId,
    MoeGemmArgs,
    MoeGemmMode,
    MxScaleMode,
    TensorMajor,
)


RAGGED_BLOCK_BY_ALIGNMENT_M = {
    128: GemmKernelBlock(num_thread=256, tile_m=128, num_buffer=4, num_squad_m=1),
    256: GemmKernelBlock(num_thread=512, tile_m=256, num_buffer=2, num_squad_m=2),
}


MASKED_BLOCK_BY_SELECT_M = {
    128: GemmKernelBlock(num_thread=256, tile_m=128, num_buffer=4, num_squad_m=1),
    256: GemmKernelBlock(num_thread=512, tile_m=256, num_buffer=2, num_squad_m=2),
}


PACKED_4BIT_BLOCK_BY_SELECT_M = {
    128: GemmKernelBlock(
        num_thread=256,
        tile_m=128,
        tile_n=256,
        tile_k=128,
        num_buffer=2,
        num_squad_m=1,
        num_squad_n=2,
    ),
    256: GemmKernelBlock(
        num_thread=512,
        tile_m=256,
        tile_n=256,
        tile_k=128,
        num_buffer=2,
        num_squad_m=2,
        num_squad_n=2,
    ),
}


RAGGED_EXPERT_LAYOUT_BLOCK_CANDIDATES = [
    (
        GemmKernelBlock(
            num_thread=512,
            tile_m=256,
            tile_n=256,
            tile_k=128,
            num_buffer=2,
            num_squad_m=2,
            num_squad_n=2,
        ),
        1.0,
    ),
    (
        GemmKernelBlock(
            num_thread=256,
            tile_m=128,
            tile_n=256,
            tile_k=128,
            num_buffer=4,
            num_squad_m=1,
            num_squad_n=2,
        ),
        0.8,
    ),
    (
        GemmKernelBlock(
            num_thread=256,
            tile_m=256,
            tile_n=128,
            tile_k=128,
            num_buffer=4,
            num_squad_m=2,
            num_squad_n=1,
        ),
        0.8,
    ),
]


@functools.cache
def get_gemm_mubin_id_hash(asm_id: GemmMubinId) -> str:
    payload = asdict(asm_id)
    for enum_key in (
        "group_mode",
        "quant_mode_a",
        "quant_mode_b",
        "major_a",
        "major_b",
        "mx_scale",
    ):
        payload[enum_key] = getattr(asm_id, enum_key).value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_n_spilt_gemm_kernel(block: GemmKernelBlock, args: MoeGemmArgs) -> bool:
    return (
        args.n_split
        and block.tile_m == 256
        and block.tile_n == 256
        and block.num_thread == 512
    )


class MoeGemmMubinDispatcher:
    def __init__(self, kernel_map_path: Path):
        self.kernel_map_path = Path(kernel_map_path)
        self._kernel_hash_map = {
            entry.dispatch_hash: entry
            for entry in load_kernel_map(self.kernel_map_path)
        }

    def _estimate_block_score(
        self, block: GemmKernelBlock, args: MoeGemmArgs, base_score: float
    ) -> float:
        if base_score == 0:
            return base_score

        average_group_m = args.m // args.num_expert
        upper_average_group_m = int(average_group_m * 1.04)
        lower_average_group_m = int(average_group_m * 0.96)
        deviate_group_count = args.num_expert // 3
        lower_average_group_m_tile = ceil_div(lower_average_group_m, block.tile_m)
        upper_average_group_m_tile = ceil_div(upper_average_group_m, block.tile_m)
        average_group_m_tile = ceil_div(upper_average_group_m, block.tile_m)
        m_tile = (
            lower_average_group_m_tile * deviate_group_count
            + upper_average_group_m_tile * deviate_group_count
            + average_group_m_tile * (args.num_expert - deviate_group_count * 2)
        )
        n_tile = ceil_div(args.n, block.tile_n)
        wave = ceil_div(m_tile * n_tile, args.total_mp_count)
        wave_ratio = (
            args.m * args.n / (block.tile_m * block.tile_n * wave * args.total_mp_count)
        )
        return base_score * wave_ratio

    def get_ragged_asm_id(self, args: MoeGemmArgs) -> GemmMubinId:
        try:
            kernel_block = RAGGED_BLOCK_BY_ALIGNMENT_M[args.alignment_m]
        except KeyError as err:
            raise ValueError("alignment_m must be 128 or 256") from err
        tme_cache_hint_b = False

        m_per_group = args.m / args.num_expert
        if m_per_group <= 192:
            tme_cache_hint_b = True

        fixed_scale_layout_a = (
            args.type_a in (torch.float8_e4m3fn, torch.float8_e5m2)
            and args.major_scale_a == TensorMajor.MN
        )

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(args.type_a),
            b_type=get_asm_dtype_from_torch_dtype(args.type_b),
            d_type=get_asm_dtype_from_torch_dtype(args.type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.RAGGED,
            quant_mode_a=args.quant_mode_a,
            quant_mode_b=args.quant_mode_b,
            quant_tile=args.quant_tile,
            kernel_block=kernel_block,
            major_a=args.major_a,
            major_b=args.major_b,
            tme_cache_hint_b=tme_cache_hint_b,
            fixed_scale_layout_a=fixed_scale_layout_a,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_w4a8_ragged_asm_id(self, args: MoeGemmArgs) -> GemmMubinId:
        if args.b_pack_bits != 4:
            raise ValueError("W4A8 GEMM requires b_pack_bits=4")
        try:
            kernel_block = PACKED_4BIT_BLOCK_BY_SELECT_M[args.alignment_m]
        except KeyError as err:
            raise ValueError("alignment_m must be 128 or 256") from err
        if is_n_spilt_gemm_kernel(kernel_block, args):
            kernel_block = replace(kernel_block, num_squad_m=1, num_squad_n=4)

        tme_cache_hint_b = False
        m_per_group = args.m / args.num_expert
        if m_per_group <= 192:
            tme_cache_hint_b = True

        is_fp4 = args.mx_scale == MxScaleMode.FP4_E8M0
        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(args.type_a),
            b_type=(AsmDType.FP4_E2M1.value if is_fp4 else AsmDType.INT4.value),
            d_type=get_asm_dtype_from_torch_dtype(args.type_d),
            dtype_bias_scale=(
                AsmDType.FP8_E8M0.value
                if is_fp4
                else get_asm_dtype_from_torch_dtype(args.type_scale_b)
            ),
            group_mode=MoeGemmMode.RAGGED,
            quant_mode_a=args.quant_mode_a,
            quant_mode_b=args.quant_mode_b,
            quant_tile=args.quant_tile,
            kernel_block=kernel_block,
            major_a=args.major_a,
            major_b=args.major_b,
            tme_cache_hint_b=tme_cache_hint_b,
            enable_tme_split=True,
            enable_persistence=True,
            b_pack_bits=args.b_pack_bits,
            mx_scale=args.mx_scale,
            n_split=is_n_spilt_gemm_kernel(kernel_block, args),
        )

    def get_masked_asm_id(self, args: MoeGemmArgs) -> GemmMubinId:
        if args.alignment_m == 0:
            select_m = 128 if args.expected_m <= 128 else 256
        else:
            select_m = args.alignment_m

        try:
            kernel_block = MASKED_BLOCK_BY_SELECT_M[select_m]
        except KeyError as err:
            raise ValueError("masked alignment_m must be 0, 128, or 256") from err

        tme_cache_hint_b = select_m == 128 or (
            select_m == 256 and args.expected_m <= 256
        )
        fixed_scale_layout_a = (
            args.type_a in (torch.float8_e4m3fn, torch.float8_e5m2)
            and args.major_scale_a == TensorMajor.MN
        )

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(args.type_a),
            b_type=get_asm_dtype_from_torch_dtype(args.type_b),
            d_type=get_asm_dtype_from_torch_dtype(args.type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.MASKED,
            quant_mode_a=args.quant_mode_a,
            quant_mode_b=args.quant_mode_b,
            quant_tile=args.quant_tile,
            kernel_block=kernel_block,
            major_a=args.major_a,
            major_b=args.major_b,
            tme_cache_hint_b=tme_cache_hint_b,
            fixed_scale_layout_a=fixed_scale_layout_a,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_w4a8_masked_asm_id(self, args: MoeGemmArgs) -> GemmMubinId:
        if args.b_pack_bits != 4:
            raise ValueError("W4A8 GEMM requires b_pack_bits=4")
        if args.alignment_m == 0:
            select_m = 128 if args.expected_m <= 128 else 256
        else:
            select_m = args.alignment_m

        try:
            kernel_block = PACKED_4BIT_BLOCK_BY_SELECT_M[select_m]
        except KeyError as err:
            raise ValueError("masked alignment_m must be 0, 128, or 256") from err
        if is_n_spilt_gemm_kernel(kernel_block, args):
            kernel_block = replace(kernel_block, num_squad_m=1, num_squad_n=4)

        tme_cache_hint_b = select_m == 128 or (
            select_m == 256 and args.expected_m <= 256
        )

        is_fp4 = args.mx_scale == MxScaleMode.FP4_E8M0
        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(args.type_a),
            b_type=(AsmDType.FP4_E2M1.value if is_fp4 else AsmDType.INT4.value),
            d_type=get_asm_dtype_from_torch_dtype(args.type_d),
            dtype_bias_scale=(
                AsmDType.FP8_E8M0.value
                if is_fp4
                else get_asm_dtype_from_torch_dtype(args.type_scale_b)
            ),
            group_mode=MoeGemmMode.MASKED,
            quant_mode_a=args.quant_mode_a,
            quant_mode_b=args.quant_mode_b,
            quant_tile=args.quant_tile,
            kernel_block=kernel_block,
            major_a=args.major_a,
            major_b=args.major_b,
            tme_cache_hint_b=tme_cache_hint_b,
            enable_tme_split=True,
            enable_persistence=True,
            b_pack_bits=args.b_pack_bits,
            mx_scale=args.mx_scale,
            n_split=is_n_spilt_gemm_kernel(kernel_block, args),
        )

    def get_k_contig_asm_id(self, args: MoeGemmArgs) -> GemmMubinId:
        fixed_scale_layout_a = (
            args.type_a in (torch.float8_e4m3fn, torch.float8_e5m2)
            and args.major_scale_a == TensorMajor.MN
        )

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(args.type_a),
            b_type=get_asm_dtype_from_torch_dtype(args.type_b),
            d_type=get_asm_dtype_from_torch_dtype(args.type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.K_CONTIG,
            quant_mode_a=args.quant_mode_a,
            quant_mode_b=args.quant_mode_b,
            quant_tile=args.quant_tile,
            kernel_block=GemmKernelBlock(),
            major_a=args.major_a,
            major_b=args.major_b,
            fixed_scale_layout_a=fixed_scale_layout_a,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_ragged_expert_layout_asm_id(self, args: MoeGemmArgs) -> GemmMubinId:
        if args.total_mp_count <= 0:
            raise ValueError("total_mp_count must be greater than 0")

        best_score = -1.0
        kernel_block = None
        for candidate_block, base_score in RAGGED_EXPERT_LAYOUT_BLOCK_CANDIDATES:
            score = self._estimate_block_score(candidate_block, args, base_score)
            if score > best_score:
                best_score = score
                kernel_block = candidate_block

        if kernel_block is None:
            raise ValueError("No RAGGED_EXPERT_LAYOUT GEMM kernel block found")

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(args.type_a),
            b_type=get_asm_dtype_from_torch_dtype(args.type_b),
            d_type=get_asm_dtype_from_torch_dtype(args.type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.RAGGED_EXPERT_LAYOUT,
            quant_mode_a=args.quant_mode_a,
            quant_mode_b=args.quant_mode_b,
            quant_tile=args.quant_tile,
            kernel_block=kernel_block,
            major_a=args.major_a,
            major_b=args.major_b,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_normal_gemm_id(self, args: MoeGemmArgs) -> GemmMubinId:
        kernel_block = GemmKernelBlock()
        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(args.type_a),
            b_type=get_asm_dtype_from_torch_dtype(args.type_b),
            d_type=get_asm_dtype_from_torch_dtype(args.type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.NO_GROUP,
            quant_mode_a=args.quant_mode_a,
            quant_mode_b=args.quant_mode_b,
            quant_tile=args.quant_tile,
            kernel_block=kernel_block,
            major_a=args.major_a,
            major_b=args.major_b,
            fixed_scale_layout_a=args.major_scale_a == TensorMajor.MN,
            fixed_scale_layout_b=False,
            enable_tme_split=True,
            enable_persistence=True,
        )

    @functools.cache
    def resolve_kernel_entry(self, asm_id: GemmMubinId):
        asm_id_hash = get_gemm_mubin_id_hash(asm_id)
        entry = self._kernel_hash_map.get(asm_id_hash)
        if entry is None:
            raise ValueError(f"No MoE GEMM mubin kernel found for hash {asm_id_hash}")
        return entry

    def resolve_kernel(
        self,
        args: MoeGemmArgs,
        get_asm_id: Callable[[MoeGemmArgs], GemmMubinId],
        mubin_dir: Path,
    ):
        asm_id = get_asm_id(args)
        entry = self.resolve_kernel_entry(asm_id)
        path = ensure_mubin_kernel_artifact("gemm", Path(mubin_dir).parent, entry)
        return asm_id, entry, path


@functools.cache
def get_moe_gemm_mubin_dispatcher(kernel_map_path: Path):
    return MoeGemmMubinDispatcher(kernel_map_path)
