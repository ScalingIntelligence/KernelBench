from dataclasses import dataclass

import torch

from ..common import Arch


@dataclass(frozen=True)
class SageAttentionMubinId:
    arch: Arch
    q_dtype: torch.dtype
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    is_causal: bool
    is_kv_cache: bool
    is_varlen: bool
    headdim_qk: int
    quant_mode: int
    is_qk_int8: bool
    fp8_output: bool
