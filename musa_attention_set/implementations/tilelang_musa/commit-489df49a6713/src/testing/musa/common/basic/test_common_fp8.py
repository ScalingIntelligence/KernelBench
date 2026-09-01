import torch
import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


_N = 32

_VECTOR_CAST_DTYPES = [
    (T.float32, T.float8_e4m3fn),
    (T.float32, T.float8_e5m2),
    (T.float32, T.float8_e8m0fnu),
    (T.float64, T.float8_e4m3fn),
    (T.float64, T.float8_e5m2),
    (T.float64, T.float8_e8m0fnu),
    (T.float8_e4m3fn, T.float32),
    (T.float8_e5m2, T.float32),
    (T.float8_e4m3fn, T.bfloat16),
    (T.float8_e5m2, T.bfloat16),
    (T.float8_e8m0fnu, T.bfloat16),
    (T.bfloat16, T.float8_e4m3fn),
    (T.bfloat16, T.float8_e5m2),
    (T.bfloat16, T.float8_e8m0fnu),
]

_TORCH_DTYPES = {
    T.float32: torch.float32,
    T.float64: torch.float64,
    T.bfloat16: torch.bfloat16,
    T.float8_e4m3fn: torch.float8_e4m3fn,
    T.float8_e5m2: torch.float8_e5m2,
    T.float8_e8m0fnu: torch.float8_e8m0fnu,
}


def _make_fp8_cast_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((_N,), T.float32),
        dst: T.Tensor((_N,), T.float32),
    ):
        with T.Kernel(1, threads=_N):
            for i in T.Parallel(_N):
                dst[i] = T.cast(T.cast(src[i], dtype), T.float32)

    return kernel


def _make_fp8_vector_copy_kernel(dtype_a, dtype_b):
    @T.prim_func
    def kernel(
        src: T.Tensor((256,), dtype_a),
        dst: T.Tensor((256,), dtype_b),
    ):
        with T.Kernel(1, threads=128):
            T.copy(src, dst)

    return kernel


def _make_e8m0_vector_copy_roundtrip_kernel(dst_dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((256,), T.float32),
        dst: T.Tensor((256,), dst_dtype),
    ):
        with T.Kernel(1, threads=128):
            tmp = T.alloc_fragment((256,), T.float8_e8m0fnu)
            T.copy(src, tmp)
            T.copy(tmp, dst)

    return kernel


@tilelang.testing.requires_musa
def test_musa_common_fp8_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        sources = []
        for dtype in (T.float8_e4m3fn, T.float8_e5m2, T.float8_e8m0fnu):
            artifact = tilelang.lower(
                _make_fp8_cast_kernel(dtype),
                target=target,
                enable_device_compile=False,
            )
            sources.append(artifact.kernel_source)

    for source in sources:
        assert source is not None
        assert "#include <musa_fp8.h>" in source
        assert "__mt_fp8" in source


@tilelang.testing.requires_musa
@pytest.mark.parametrize("dtype", [T.float8_e4m3fn, T.float8_e5m2, T.float8_e8m0fnu])
def test_musa_common_fp8_runtime(dtype):
    kernel = tilelang.compile(
        _make_fp8_cast_kernel(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    if dtype == T.float8_e8m0fnu:
        src = torch.tensor(
            [2.0 ** ((i % 7) - 3) for i in range(_N)], dtype=torch.float32
        )
        atol = rtol = 0.0
    else:
        src = torch.linspace(-1.0, 1.0, _N, dtype=torch.float32)
        atol = rtol = 0.02 if dtype == T.float8_e4m3fn else 0.08
    out = kernel(src.to("musa"))
    torch.musa.synchronize()

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.cpu(), src, atol=atol, rtol=rtol)


@tilelang.testing.requires_musa
@pytest.mark.parametrize("dtype_a,dtype_b", _VECTOR_CAST_DTYPES)
def test_musa_common_fp8_vector_cast_codegen(dtype_a, dtype_b):
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _make_fp8_vector_copy_kernel(dtype_a, dtype_b),
            target=target,
            enable_device_compile=False,
        )

    source = artifact.kernel_source
    assert source is not None
    if dtype_a is T.float32 or dtype_b is T.float32:
        assert "musa_fp8.h" in source


@tilelang.testing.requires_musa
@pytest.mark.parametrize("dtype_a,dtype_b", _VECTOR_CAST_DTYPES)
def test_musa_common_fp8_vector_cast_runtime(dtype_a, dtype_b):
    values = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32).repeat(64)
    if T.float8_e8m0fnu in (dtype_a, dtype_b):
        out_dtype = T.bfloat16 if dtype_a is T.float8_e8m0fnu else T.float32
        kernel = tilelang.compile(
            _make_e8m0_vector_copy_roundtrip_kernel(out_dtype),
            out_idx=[1],
            target={"kind": "musa"},
            execution_backend="tvm_ffi",
        )
        out = kernel(values.to("musa"))
        torch.musa.synchronize()
        expected = values.to(torch.float8_e8m0fnu).to(_TORCH_DTYPES[out_dtype])
        torch.testing.assert_close(out.cpu(), expected)
        return

    kernel = tilelang.compile(
        _make_fp8_vector_copy_kernel(dtype_a, dtype_b),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    src = values.to(_TORCH_DTYPES[dtype_a])
    out = kernel(src.to("musa"))
    torch.musa.synchronize()

    expected = src.to(_TORCH_DTYPES[dtype_b]).float()
    torch.testing.assert_close(out.float().cpu(), expected)


if __name__ == "__main__":
    tilelang.testing.main()
