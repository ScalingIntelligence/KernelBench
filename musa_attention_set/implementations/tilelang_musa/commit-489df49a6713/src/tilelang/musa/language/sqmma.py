from __future__ import annotations

from tvm import tirx

from tilelang.language.gemm_op import GemmWarpPolicy, _gemm_impl
from tilelang.language.utils import BufferLikeType


def sqmma_gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
) -> tirx.Call:
    """Perform an explicit MP31 SQMMA operation."""
    return _gemm_impl(
        "tl.tileop.sqmma_gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        wg_wait=0,
    )


__all__ = ["sqmma_gemm"]
