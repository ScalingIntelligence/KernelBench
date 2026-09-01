"""MP31 SQMMA GEMM lowering."""

from __future__ import annotations

from tilelang import _ffi_api
from tilelang import language as T
from tilelang import tvm as tvm
from tilelang.musa.layout import make_mp31_sqmma_fragment_c, make_mp31_sqmma_shared_ab
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_shared
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_SQMMA = "musa.sqmma"

_DTYPE_TO_SQMMA_TAG = {
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "custom[tfloat32]": "tf32",
    "tfloat32": "tf32",
    "int8": "s8",
    "int32": "s32",
    "uint8": "u8",
    "float8_e4m3": "e4m3",
    "float8_e4m3fn": "e4m3",
    "float8_e5m2": "e5m2",
}


def _as_int(value, name: str) -> int:
    if isinstance(value, int):
        return value
    if hasattr(value, "value"):
        return int(value.value)
    return int(value)


def _sqmma_tag(dtype) -> str:
    name = str(dtype)
    if name in _DTYPE_TO_SQMMA_TAG:
        return _DTYPE_TO_SQMMA_TAG[name]
    raise ValueError(f"Unsupported MP31 SQMMA dtype: {name}")


class GemmMP31SQMMA(GemmBase):
    def __post_init__(self) -> None:
        # MP31 supports asymmetric FP16/BF16 with S8 combinations that the
        # generic GEMM dtype validator rejects. The C++ capability table is the
        # single source of truth for exact A/B/C combinations.
        pass

    def _warp_partition(self, target: Target, block_size: int) -> tuple[int, int]:
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, block_size, target, GEMM_INST_SQMMA)
        return int(m_warp), int(n_warp)

    def _get_inst_shape(self, target: Target, block_size: int, m_warp: int, n_warp: int) -> tuple[int, int, int]:
        inst_m, inst_n, inst_k = _ffi_api.get_mp31_sqmma_inst_shape(
            self.gemm_node,
            int(block_size),
            int(m_warp),
            int(n_warp),
            target,
        )
        return int(inst_m), int(inst_n), int(inst_k)

    def infer_layout(self, target: Target, thread_nums: int):
        if not self.is_gemm_ss():
            raise ValueError(f"Unsupported SQMMA combination, A: {self.A.scope()}, B: {self.B.scope()}")
        thread_nums = _as_int(thread_nums, "thread_nums")
        m_warp, n_warp = self._warp_partition(target, thread_nums)
        inst_m, inst_n, inst_k = self._get_inst_shape(target, thread_nums, m_warp, n_warp)
        a_shape = self.A.shape
        b_shape = self.B.shape
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError("GemmMP31SQMMA expects 2D A/B buffers")

        a_layout = make_mp31_sqmma_shared_ab(
            self.A,
            k_major=not bool(self.trans_A),
        )
        if int(a_shape[-2]) == 32 and int(a_shape[-1]) == 32:
            a_bits = int(tvm.DataType(self.A.dtype).bits)
            a_layout = a_layout.repeat(1, 32 // a_bits)

        b_layout = make_mp31_sqmma_shared_ab(
            self.B,
            k_major=bool(self.trans_B),
        )
        if int(b_shape[-2]) == 32 and int(b_shape[-1]) == 32:
            b_bits = int(tvm.DataType(self.B.dtype).bits)
            b_layout = b_layout.repeat(1, 32 // b_bits)

        return {
            self.A: a_layout,
            self.B: b_layout,
            self.C: make_mp31_sqmma_fragment_c(self.C, m_warp, n_warp, (inst_m, inst_n, inst_k)),
        }

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        del layout_map, thread_index, mbar_phase_expr
        if not self.is_gemm_ss():
            raise ValueError(f"Unsupported SQMMA combination, A: {self.A.scope()}, B: {self.B.scope()}")

        block_size = _as_int(thread_bounds.extent, "thread_bounds.extent")
        m_warp, n_warp = self._warp_partition(target, block_size)
        inst_m, inst_n, inst_k = self._get_inst_shape(target, block_size, m_warp, n_warp)
        a_tag = _sqmma_tag(self.a_dtype)
        b_tag = _sqmma_tag(self.b_dtype)
        c_tag = _sqmma_tag(self.accum_dtype)
        func = (
            f"tl::sqmma_ss<{self.M}, {self.N}, {self.K}, {m_warp}, {n_warp}, "
            f"{inst_m}, {inst_n}, {inst_k}, tl::sqmma::{a_tag}, tl::sqmma::{b_tag}, tl::sqmma::{c_tag}>"
        )
        a_col_major = int(bool(self.trans_A))
        b_col_major = int(bool(self.trans_B))
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        @T.prim_func
        def _gemm_ss() -> None:
            T.call_extern(
                "handle",
                func,
                T.access_ptr(A_region, "r", ignore_last_ndim=2),
                T.access_ptr(B_region, "r", ignore_last_ndim=2),
                T.access_ptr(C_region, "rw", ignore_last_ndim=2),
                self.stride_A,
                self.stride_B,
                a_col_major,
                b_col_major,
                self.clear_accum,
            )

        return _Simplify(_gemm_ss, inline_let=True)

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)
