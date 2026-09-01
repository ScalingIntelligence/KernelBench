from dataclasses import asdict
import functools
import hashlib
import json
import math
from pathlib import Path

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
    MoeGemmMode,
    MxScaleMode,
    TensorMajor,
    TensorQuantMode,
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
        num_squad_m=1,
        num_squad_n=4,
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


class MoeGemmMubinDispatcher:
    def __init__(self, kernel_map_path: Path):
        self.kernel_map_path = Path(kernel_map_path)
        self.module_dir = self.kernel_map_path.parent
        self._kernel_hash_map = {
            entry.dispatch_hash: entry
            for entry in load_kernel_map(self.kernel_map_path)
        }

    def estimate_block_score(
        self,
        block: GemmKernelBlock,
        m: int,
        n: int,
        num_expert: int,
        num_mp: int,
        base_score: float,
    ) -> float:
        if num_mp <= 0:
            raise ValueError("num_mp must be greater than 0")
        if base_score == 0 or m == 0 or n == 0 or num_expert == 0:
            return base_score

        average_group_m = m / num_expert
        upper_average_group_m = average_group_m * 1.04
        lower_average_group_m = average_group_m * 0.96
        deviate_group_count = num_expert // 3
        lower_average_group_m_tile = math.ceil(lower_average_group_m / block.tile_m)
        upper_average_group_m_tile = math.ceil(upper_average_group_m / block.tile_m)
        average_group_m_tile = math.ceil(upper_average_group_m / block.tile_m)
        m_tile = (
            lower_average_group_m_tile * deviate_group_count
            + upper_average_group_m_tile * deviate_group_count
            + average_group_m_tile * (num_expert - deviate_group_count * 2)
        )
        n_tile = ceil_div(n, block.tile_n)
        wave = ceil_div(m_tile * n_tile, num_mp)
        wave_ratio = m * n / (block.tile_m * block.tile_n * wave * num_mp)
        return base_score * wave_ratio

    def get_ragged_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_b: torch.dtype,
        type_d: torch.dtype,
        tme_cache_hint_b: bool,
        alignment_m: int,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        major_a: TensorMajor = TensorMajor.K,
        major_b: TensorMajor = TensorMajor.K,
        major_scale_a: TensorMajor = TensorMajor.K,
    ) -> GemmMubinId:
        try:
            kernel_block = RAGGED_BLOCK_BY_ALIGNMENT_M[alignment_m]
        except KeyError as err:
            raise ValueError("alignment_m must be 128 or 256") from err
        fixed_scale_layout_a = (
            type_a in (torch.float8_e4m3fn, torch.float8_e5m2)
            and major_scale_a == TensorMajor.MN
        )

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=get_asm_dtype_from_torch_dtype(type_b),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.RAGGED,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=major_a,
            major_b=major_b,
            tme_cache_hint_b=tme_cache_hint_b,
            fixed_scale_layout_a=fixed_scale_layout_a,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_w4a8_ragged_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_d: torch.dtype,
        type_scale_b: torch.dtype,
        tme_cache_hint_b: bool,
        alignment_m: int,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        mx_scale: MxScaleMode,
    ) -> GemmMubinId:
        try:
            kernel_block = PACKED_4BIT_BLOCK_BY_SELECT_M[alignment_m]
        except KeyError as err:
            raise ValueError("alignment_m must be 128 or 256") from err
        is_fp4 = mx_scale == MxScaleMode.FP4_E8M0
        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=(AsmDType.FP4_E2M1.value if is_fp4 else AsmDType.INT4.value),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=(
                AsmDType.FP8_E8M0.value
                if is_fp4
                else get_asm_dtype_from_torch_dtype(type_scale_b)
            ),
            group_mode=MoeGemmMode.RAGGED,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=TensorMajor.K,
            major_b=TensorMajor.K,
            tme_cache_hint_b=tme_cache_hint_b,
            enable_tme_split=True,
            enable_persistence=True,
            b_pack_bits=4,
            mx_scale=mx_scale,
            n_split=alignment_m == 256,
        )

    def get_masked_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_b: torch.dtype,
        type_d: torch.dtype,
        alignment_m: int,
        tme_cache_hint_b: bool,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        major_scale_a: TensorMajor = TensorMajor.K,
    ) -> GemmMubinId:
        try:
            kernel_block = MASKED_BLOCK_BY_SELECT_M[alignment_m]
        except KeyError as err:
            raise ValueError("masked alignment_m must be 128 or 256") from err

        fixed_scale_layout_a = (
            type_a in (torch.float8_e4m3fn, torch.float8_e5m2)
            and major_scale_a == TensorMajor.MN
        )

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=get_asm_dtype_from_torch_dtype(type_b),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.MASKED,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=TensorMajor.K,
            major_b=TensorMajor.K,
            tme_cache_hint_b=tme_cache_hint_b,
            fixed_scale_layout_a=fixed_scale_layout_a,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_w4a8_masked_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_d: torch.dtype,
        type_scale_b: torch.dtype,
        alignment_m: int,
        tme_cache_hint_b: bool,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        mx_scale: MxScaleMode,
    ) -> GemmMubinId:
        try:
            kernel_block = PACKED_4BIT_BLOCK_BY_SELECT_M[alignment_m]
        except KeyError as err:
            raise ValueError("masked alignment_m must be 128 or 256") from err
        is_fp4 = mx_scale == MxScaleMode.FP4_E8M0
        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=(AsmDType.FP4_E2M1.value if is_fp4 else AsmDType.INT4.value),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=(
                AsmDType.FP8_E8M0.value
                if is_fp4
                else get_asm_dtype_from_torch_dtype(type_scale_b)
            ),
            group_mode=MoeGemmMode.MASKED,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=TensorMajor.K,
            major_b=TensorMajor.K,
            tme_cache_hint_b=tme_cache_hint_b,
            enable_tme_split=True,
            enable_persistence=True,
            b_pack_bits=4,
            mx_scale=mx_scale,
            n_split=alignment_m == 256,
        )

    def get_k_contig_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_b: torch.dtype,
        type_d: torch.dtype,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        major_scale_a: TensorMajor,
    ) -> GemmMubinId:
        fixed_scale_layout_a = (
            type_a in (torch.float8_e4m3fn, torch.float8_e5m2)
            and major_scale_a == TensorMajor.MN
        )

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=get_asm_dtype_from_torch_dtype(type_b),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.K_CONTIG,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=GemmKernelBlock(),
            major_a=TensorMajor.MN,
            major_b=TensorMajor.MN,
            fixed_scale_layout_a=fixed_scale_layout_a,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_ragged_expert_layout_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_b: torch.dtype,
        type_d: torch.dtype,
        candidate_block_id: int,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        major_b: TensorMajor,
    ) -> GemmMubinId:
        kernel_block = RAGGED_EXPERT_LAYOUT_BLOCK_CANDIDATES[candidate_block_id][0]

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=get_asm_dtype_from_torch_dtype(type_b),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.RAGGED_EXPERT_LAYOUT,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=TensorMajor.K,
            major_b=major_b,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_normal_gemm_id(
        self,
        *,
        type_a: torch.dtype,
        type_b: torch.dtype,
        type_d: torch.dtype,
        quant_tile: int,
        major_a: TensorMajor,
        major_b: TensorMajor,
    ) -> GemmMubinId:
        kernel_block = GemmKernelBlock()
        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=get_asm_dtype_from_torch_dtype(type_b),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.NO_GROUP,
            quant_mode_a=TensorQuantMode.GROUP,
            quant_mode_b=TensorQuantMode.BLOCK,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=major_a,
            major_b=major_b,
            fixed_scale_layout_a=major_a == TensorMajor.MN,
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

    @functools.cache
    def resolve_kernel_artifact(self, asm_id: GemmMubinId):
        entry = self.resolve_kernel_entry(asm_id)
        path = ensure_mubin_kernel_artifact("gemm", self.module_dir, entry)
        return entry, path


@functools.cache
def get_moe_gemm_mubin_dispatcher(kernel_map_path: Path):
    return MoeGemmMubinDispatcher(kernel_map_path)
