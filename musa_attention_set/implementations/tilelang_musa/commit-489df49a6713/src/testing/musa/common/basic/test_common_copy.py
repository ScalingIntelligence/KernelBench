import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

_M = 32
_N = 32


def _torch_dtype(tilelang_dtype):
    return {
        T.float32: torch.float32,
        T.float16: torch.float16,
        T.bfloat16: torch.bfloat16,
        T.float8_e4m3fn: torch.float8_e4m3fn,
        T.int32: torch.int32,
    }[tilelang_dtype]


def _make_copy_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((_M, _N), dtype),
        dst: T.Tensor((_M, _N), dtype),
    ):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((_M, _N), dtype)
            T.copy(src, smem)
            T.copy(smem, dst)

    return kernel


@pytest.mark.parametrize(
    "dtype", [T.float32, T.float16, T.bfloat16, T.float8_e4m3fn, T.int32]
)
@tilelang.testing.requires_musa
def test_musa_copy_codegen(dtype):
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _make_copy_kernel(dtype),
            target=target,
            enable_device_compile=False,
        )

    assert artifact.kernel_source is not None
    assert "__global__" in artifact.kernel_source


@pytest.mark.parametrize(
    "dtype", [T.float32, T.float16, T.bfloat16, T.float8_e4m3fn, T.int32]
)
@tilelang.testing.requires_musa
def test_musa_copy_runtime_values(dtype):
    kernel = tilelang.compile(
        _make_copy_kernel(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    src_cpu = torch.arange(_M * _N, dtype=torch.float32).reshape(_M, _N)
    if dtype is T.float8_e4m3fn:
        src_cpu = (src_cpu % 8).to(torch.float8_e4m3fn)
    else:
        src_cpu = src_cpu.to(_torch_dtype(dtype))
    dst = kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(dst.cpu(), src_cpu, atol=0, rtol=0)
