from dataclasses import dataclass

import torch

from ..common import Arch


@dataclass(frozen=True)
class FlashMLAMubinId:
    arch: Arch
    dtype: torch.dtype
    is_causal: bool
    is_varlen_q: bool
