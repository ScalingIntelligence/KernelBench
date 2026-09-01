from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from ..common import Arch


class MoeGemmMode(Enum):
    NO_GROUP = 0
    RAGGED_EXPERT_LAYOUT = 1
    RESERVE1 = 2
    RAGGED = 3
    MASKED = 4
    BYTE_ML_RAGGED = 5
    K_CONTIG = 6


class TensorMajor(Enum):
    K = "K"
    MN = "MN"


class TensorQuantMode(Enum):
    NO_QUANT = 0
    TENSOR = 1
    CHANNEL = 2
    BLOCK = 3
    GROUP = 4


class AsmDType(str, Enum):
    INT4 = "int4"
    FP4_E2M1 = "fp4_e2m1"
    FP8_E8M0 = "fp8_e8m0"


class MxScaleMode(Enum):
    NONE = 0
    E8M0 = 1
    E4M3 = 2
    FP4_E8M0 = 3


@dataclass
class MoeGemmArgs:
    type_a: torch.dtype
    type_b: torch.dtype
    type_d: torch.dtype

    m: int
    n: int
    k: int
    num_expert: int
    batch: int = 1

    type_scale_a: torch.dtype = torch.float32
    type_scale_b: torch.dtype = torch.float32
    b_pack_bits: int = 8
    mx_scale: MxScaleMode = MxScaleMode.NONE
    n_split: bool = False

    quant_tile: int = 128
    alignment_m: int = 0
    expected_m: int = 0
    total_mp_count: int = 0
    target_mp_count: int = 0

    stride_m_a: int = 0
    stride_k_a: int = 1
    stride_batch_a: int = 0
    stride_n_b: int = 0
    stride_k_b: int = 1
    stride_batch_b: int = 0
    stride_m_out: int = 0
    stride_batch_out: int = 0

    major_a: TensorMajor = TensorMajor.K
    major_b: TensorMajor = TensorMajor.K
    quant_mode_a: TensorQuantMode = TensorQuantMode.GROUP
    quant_mode_b: TensorQuantMode = TensorQuantMode.BLOCK
    major_scale_a: TensorMajor = TensorMajor.K
    major_scale_b: TensorMajor = TensorMajor.K

    scale_a_m: int = 0
    scale_a_k: int = 0
    scale_a_nr_elem: int = 0
    scale_b_n: int = 0
    scale_b_k: int = 0
    scale_b_nr_elem: int = 0

    p_a: Optional[torch.Tensor] = None
    p_b: Optional[torch.Tensor] = None
    p_d: Optional[torch.Tensor] = None
    p_scale_a: Optional[torch.Tensor] = None
    p_scale_b: Optional[torch.Tensor] = None
    p_scale_out: Optional[torch.Tensor] = None
    p_m_indices: Optional[torch.Tensor] = None
    p_signal: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class GemmKernelBlock:
    num_thread: int = 512
    tile_m: int = 256
    tile_n: int = 256
    tile_k: int = 128
    num_buffer: int = 2
    blk_per_mp: int = 1
    num_squad_m: int = 2
    num_squad_n: int = 2
    macro_tile_x: int = 2
    switch_swizzle: int = 0


@dataclass(frozen=True)
class GemmMubinId:
    arch: Arch
    a_type: str
    b_type: str
    d_type: str
    dtype_bias_scale: str
    group_mode: MoeGemmMode
    quant_mode_a: TensorQuantMode
    quant_mode_b: TensorQuantMode
    quant_tile: int
    kernel_block: GemmKernelBlock
    major_a: TensorMajor
    major_b: TensorMajor
    tme_cache_hint_a: bool = False
    tme_cache_hint_b: bool = False
    fixed_scale_layout_a: bool = False
    fixed_scale_layout_b: bool = False
    enable_tme_split: bool = False
    enable_persistence: bool = False
    b_pack_bits: int = 8
    mx_scale: MxScaleMode = MxScaleMode.NONE
    n_split: bool = False
