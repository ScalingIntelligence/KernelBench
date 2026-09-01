from dataclasses import dataclass

import torch

from ..common import Arch


@dataclass(frozen=True)
class FlashAttentionMubinId:
    arch: Arch
    dtype: torch.dtype
    is_causal: bool
    is_varlen: bool
    headdim_qk: int
