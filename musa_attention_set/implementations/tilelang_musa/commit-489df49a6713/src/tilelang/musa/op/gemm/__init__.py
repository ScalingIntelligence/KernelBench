from __future__ import annotations

from tilelang.musa.target import target_is_mp31
from tilelang.tileop.gemm.registry import register_gemm_impl

from .mp31_sqmma import GEMM_INST_SQMMA, GemmMP31SQMMA


def _match_mp31_sqmma(target) -> bool:
    return target_is_mp31(target)


register_gemm_impl("musa.mp31.sqmma", GEMM_INST_SQMMA, _match_mp31_sqmma, GemmMP31SQMMA)
