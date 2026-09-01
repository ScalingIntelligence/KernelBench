import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

_BLOCK_N = 128
_N = 256
_M2D = 128
_N2D = 128
_BLOCK_M2D = 64
_BLOCK_N2D = 64
_REGION_START = 16
_REGION_EXTENT = 64


def _torch_dtype(tilelang_dtype):
    return {
        T.float32: torch.float32,
        T.float16: torch.float16,
        T.bfloat16: torch.bfloat16,
        T.int32: torch.int32,
        T.int64: torch.int64,
    }[tilelang_dtype]


def _make_scan_kernel(op, dtype=T.float32, reverse=False, scope="shared"):
    @T.prim_func
    def kernel(
        A: T.Tensor((_N,), dtype),
        B: T.Tensor((_N,), dtype),
    ):
        with T.Kernel(T.ceildiv(_N, _BLOCK_N), threads=_BLOCK_N) as bx:
            A_shared = T.alloc_shared((_BLOCK_N,), dtype)
            T.copy(A[bx * _BLOCK_N], A_shared)

            if scope == "fragment":
                A_fragment = T.alloc_fragment((_BLOCK_N,), dtype)
                T.copy(A_shared, A_fragment)
                if op == "cumsum":
                    T.cumsum(src=A_fragment, dim=0, reverse=reverse)
                else:
                    T.cummax(src=A_fragment, dim=0, reverse=reverse)
                T.copy(A_fragment, B[bx * _BLOCK_N])
            else:
                if op == "cumsum":
                    T.cumsum(src=A_shared, dim=0, reverse=reverse)
                else:
                    T.cummax(src=A_shared, dim=0, reverse=reverse)
                T.copy(A_shared, B[bx * _BLOCK_N])

    return kernel


def _make_scan_2d_kernel(op, dim=0, reverse=False, scope="shared", out_of_place=False):
    @T.prim_func
    def kernel(
        A: T.Tensor((_M2D, _N2D), "float32"),
        B: T.Tensor((_M2D, _N2D), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(_N2D, _BLOCK_N2D),
            T.ceildiv(_M2D, _BLOCK_M2D),
            threads=256,
        ) as (bx, by):
            A_shared = T.alloc_shared((_BLOCK_M2D, _BLOCK_N2D), "float32")
            T.copy(A[by * _BLOCK_M2D, bx * _BLOCK_N2D], A_shared)

            if scope == "fragment":
                A_fragment = T.alloc_fragment((_BLOCK_M2D, _BLOCK_N2D), "float32")
                T.copy(A_shared, A_fragment)
                if out_of_place:
                    B_fragment = T.alloc_fragment((_BLOCK_M2D, _BLOCK_N2D), "float32")
                    if op == "cumsum":
                        T.cumsum(src=A_fragment, dst=B_fragment, dim=dim, reverse=reverse)
                    else:
                        T.cummax(src=A_fragment, dst=B_fragment, dim=dim, reverse=reverse)
                    T.copy(B_fragment, B[by * _BLOCK_M2D, bx * _BLOCK_N2D])
                else:
                    if op == "cumsum":
                        T.cumsum(src=A_fragment, dim=dim, reverse=reverse)
                    else:
                        T.cummax(src=A_fragment, dim=dim, reverse=reverse)
                    T.copy(A_fragment, B[by * _BLOCK_M2D, bx * _BLOCK_N2D])
            elif out_of_place:
                B_shared = T.alloc_shared((_BLOCK_M2D, _BLOCK_N2D), "float32")
                if op == "cumsum":
                    T.cumsum(src=A_shared, dst=B_shared, dim=dim, reverse=reverse)
                else:
                    T.cummax(src=A_shared, dst=B_shared, dim=dim, reverse=reverse)
                T.copy(B_shared, B[by * _BLOCK_M2D, bx * _BLOCK_N2D])
            else:
                if op == "cumsum":
                    T.cumsum(src=A_shared, dim=dim, reverse=reverse)
                else:
                    T.cummax(src=A_shared, dim=dim, reverse=reverse)
                T.copy(A_shared, B[by * _BLOCK_M2D, bx * _BLOCK_N2D])

    return kernel


def _make_region_scan_kernel(op, reverse=False):
    @T.prim_func
    def kernel(
        A: T.Tensor((_N,), "float32"),
        B: T.Tensor((_N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(_N, _BLOCK_N), threads=_BLOCK_N) as bx:
            A_shared = T.alloc_shared((_BLOCK_N,), "float32")
            T.copy(A[bx * _BLOCK_N], A_shared)
            if op == "cumsum":
                T.cumsum(
                    src=A_shared[_REGION_START : _REGION_START + _REGION_EXTENT],
                    dim=0,
                    reverse=reverse,
                )
            else:
                T.cummax(
                    src=A_shared[_REGION_START : _REGION_START + _REGION_EXTENT],
                    dim=0,
                    reverse=reverse,
                )
            T.copy(A_shared, B[bx * _BLOCK_N])

    return kernel


def _scan_reference(chunk, op, dim, reverse):
    if reverse:
        flipped = torch.flip(chunk, dims=[dim])
        if op == "cumsum":
            return torch.flip(flipped.cumsum(dim=dim), dims=[dim])
        return torch.flip(flipped.cummax(dim=dim).values, dims=[dim])
    if op == "cumsum":
        return chunk.cumsum(dim=dim)
    return chunk.cummax(dim=dim).values


def _blockwise_reference(a_cpu, op, reverse):
    ref = torch.empty_like(a_cpu)
    for start in range(0, _N, _BLOCK_N):
        chunk = a_cpu[start : start + _BLOCK_N]
        scanned = _scan_reference(chunk, op, 0, reverse)
        ref[start : start + _BLOCK_N] = scanned
    return ref


def _blockwise_2d_reference(a_cpu, op, dim, reverse):
    ref = torch.empty_like(a_cpu)
    for start_m in range(0, _M2D, _BLOCK_M2D):
        for start_n in range(0, _N2D, _BLOCK_N2D):
            tile = a_cpu[
                start_m : start_m + _BLOCK_M2D,
                start_n : start_n + _BLOCK_N2D,
            ]
            ref[
                start_m : start_m + _BLOCK_M2D,
                start_n : start_n + _BLOCK_N2D,
            ] = _scan_reference(tile, op, dim, reverse)
    return ref


def _region_reference(a_cpu, op, reverse):
    ref = a_cpu.clone()
    for start in range(0, _N, _BLOCK_N):
        region = ref[
            start + _REGION_START : start + _REGION_START + _REGION_EXTENT
        ]
        ref[
            start + _REGION_START : start + _REGION_START + _REGION_EXTENT
        ] = _scan_reference(region, op, 0, reverse)
    return ref


@pytest.mark.parametrize(
    ("op", "symbol"),
    [
        ("cumsum", "tl::CumSum1D"),
        ("cummax", "tl::CumMax1D"),
    ],
)
@pytest.mark.parametrize("dtype", [T.float32, T.float16, T.bfloat16, T.int32, T.int64])
@tilelang.testing.requires_musa
def test_musa_scan_template_codegen(op, symbol, dtype):
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _make_scan_kernel(op, dtype),
            target=target,
            enable_device_compile=False,
        )

    assert artifact.kernel_source is not None
    assert "#include <tl_templates/musa/common/scan.h>" in artifact.kernel_source
    assert symbol in artifact.kernel_source


@pytest.mark.parametrize("op", ["cumsum", "cummax"])
@pytest.mark.parametrize("dtype", [T.float32, T.float16, T.bfloat16, T.int32, T.int64])
@pytest.mark.parametrize("reverse", [False, True])
@tilelang.testing.requires_musa
def test_musa_shared_scan_runtime_values(op, dtype, reverse):
    kernel = tilelang.compile(
        _make_scan_kernel(op, dtype, reverse),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    a_cpu = (torch.arange(_N, dtype=torch.float32) % 17 - 8).to(_torch_dtype(dtype))
    out = kernel(a_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(
        out.cpu(),
        _blockwise_reference(a_cpu, op, reverse),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("op", ["cumsum", "cummax"])
@pytest.mark.parametrize("dtype", [T.float32, T.float16, T.bfloat16, T.int32, T.int64])
@pytest.mark.parametrize("reverse", [False, True])
@tilelang.testing.requires_musa
def test_musa_fragment_scan_runtime_values(op, dtype, reverse):
    kernel = tilelang.compile(
        _make_scan_kernel(op, dtype, reverse, scope="fragment"),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    a_cpu = (torch.arange(_N, dtype=torch.float32) % 17 - 8).to(_torch_dtype(dtype))
    out = kernel(a_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(
        out.cpu(),
        _blockwise_reference(a_cpu, op, reverse),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("op", ["cumsum", "cummax"])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("scope", ["shared", "fragment"])
@pytest.mark.parametrize("reverse", [False, True])
@tilelang.testing.requires_musa
def test_musa_scan_2d_runtime_values(op, dim, scope, reverse):
    kernel = tilelang.compile(
        _make_scan_2d_kernel(op, dim, reverse, scope=scope),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    a_cpu = torch.arange(_M2D * _N2D, dtype=torch.float32).reshape(_M2D, _N2D) % 17 - 8
    out = kernel(a_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(
        out.cpu(),
        _blockwise_2d_reference(a_cpu, op, dim, reverse),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("op", ["cumsum", "cummax"])
@pytest.mark.parametrize("scope", ["shared", "fragment"])
@tilelang.testing.requires_musa
def test_musa_scan_out_of_place_runtime_values(op, scope):
    kernel = tilelang.compile(
        _make_scan_2d_kernel(op, dim=1, reverse=True, scope=scope, out_of_place=True),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    a_cpu = torch.arange(_M2D * _N2D, dtype=torch.float32).reshape(_M2D, _N2D) % 17 - 8
    out = kernel(a_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(
        out.cpu(),
        _blockwise_2d_reference(a_cpu, op, dim=1, reverse=True),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("op", ["cumsum", "cummax"])
@pytest.mark.parametrize("reverse", [False, True])
@tilelang.testing.requires_musa
def test_musa_scan_region_runtime_values(op, reverse):
    kernel = tilelang.compile(
        _make_region_scan_kernel(op, reverse),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    a_cpu = torch.arange(_N, dtype=torch.float32) % 17 - 8
    out = kernel(a_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(
        out.cpu(),
        _region_reference(a_cpu, op, reverse),
        atol=0,
        rtol=0,
    )
