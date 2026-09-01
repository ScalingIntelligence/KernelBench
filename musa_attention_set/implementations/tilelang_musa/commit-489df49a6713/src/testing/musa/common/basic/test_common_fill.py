import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

_N = 256
_M = 8
_K = 128
_REGION_START = 32
_REGION_EXTENT = 96


def _torch_dtype(tilelang_dtype):
    return {
        T.bool: torch.bool,
        T.float64: torch.float64,
        T.float32: torch.float32,
        T.float16: torch.float16,
        T.bfloat16: torch.bfloat16,
        T.float8_e4m3fn: torch.float8_e4m3fn,
        T.int64: torch.int64,
        T.int32: torch.int32,
        T.int16: torch.int16,
        T.int8: torch.int8,
        T.uint64: torch.uint64,
        T.uint32: torch.uint32,
        T.uint16: torch.uint16,
    }[tilelang_dtype]


def _make_shared_fill_kernel(dtype, value):
    @T.prim_func
    def kernel(out: T.Tensor((_N,), dtype)):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((_N,), dtype)
            T.fill(smem, value)
            T.copy(smem, out)

    return kernel


def _make_fragment_fill_kernel(dtype, value):
    @T.prim_func
    def kernel(out: T.Tensor((_N,), dtype)):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((_N,), dtype)
            T.fill(frag, value)
            T.copy(frag, out)

    return kernel


@T.prim_func
def _global_region_fill_kernel(out: T.Tensor((_N,), "float32")):
    with T.Kernel(1, threads=128):
        T.fill(out[_REGION_START : _REGION_START + _REGION_EXTENT], 3.5)


@T.prim_func
def _clear_shared_kernel(out: T.Tensor((_M, _K), "int32")):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((_M, _K), "int32")
        T.fill(smem, -7)
        T.clear(smem)
        T.copy(smem, out)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (T.bool, True),
        (T.float64, 5.0),
        (T.float32, 1.25),
        (T.float16, -2.0),
        (T.bfloat16, 3.0),
        (T.float8_e4m3fn, 1.25),
        (T.int64, -17),
        (T.int32, -17),
        (T.int16, -17),
        (T.int8, -7),
        (T.uint64, 17),
        (T.uint32, 17),
        (T.uint16, 17),
    ],
)
@tilelang.testing.requires_musa
def test_musa_shared_fill_runtime_values(dtype, value):
    kernel = tilelang.compile(
        _make_shared_fill_kernel(dtype, value),
        out_idx=[0],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    out = kernel()
    torch.musa.synchronize()

    ref = torch.full((_N,), value, dtype=_torch_dtype(dtype))
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (T.bool, True),
        (T.float64, 5.0),
        (T.float32, 1.25),
        (T.float16, -2.0),
        (T.bfloat16, 3.0),
        (T.float8_e4m3fn, 1.25),
        (T.int64, -17),
        (T.int32, -17),
        (T.int16, -17),
        (T.int8, -7),
        (T.uint64, 17),
        (T.uint32, 17),
        (T.uint16, 17),
    ],
)
@tilelang.testing.requires_musa
def test_musa_fragment_fill_runtime_values(dtype, value):
    kernel = tilelang.compile(
        _make_fragment_fill_kernel(dtype, value),
        out_idx=[0],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    out = kernel()
    torch.musa.synchronize()

    ref = torch.full((_N,), value, dtype=_torch_dtype(dtype))
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_global_region_fill_runtime_values():
    kernel = tilelang.compile(
        _global_region_fill_kernel,
        out_idx=[],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    out = torch.zeros((_N,), dtype=torch.float32, device="musa")
    kernel(out)
    torch.musa.synchronize()

    ref = torch.zeros((_N,), dtype=torch.float32)
    ref[_REGION_START : _REGION_START + _REGION_EXTENT] = 3.5
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_clear_shared_runtime_values():
    kernel = tilelang.compile(
        _clear_shared_kernel,
        out_idx=[0],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    out = kernel()
    torch.musa.synchronize()

    ref = torch.zeros((_M, _K), dtype=torch.int32)
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)
