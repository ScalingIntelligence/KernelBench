import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

_M = 1024


def _async_copy_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((_M,), dtype),
        dst: T.Tensor((_M,), dtype),
    ):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((_M,), dtype)
            T.async_copy(src, smem)
            T.ptx_wait_group(0)
            T.sync_threads()
            T.copy(smem, dst)

    return kernel


@tilelang.testing.requires_musa
@pytest.mark.parametrize("dtype", [T.float32, T.float8_e4m3fn])
def test_musa_async_copy_codegen(dtype):
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _async_copy_kernel(dtype),
            target=target,
            enable_device_compile=False,
        )

    source = artifact.kernel_source
    assert "#include <tl_templates/musa/common/copy.h>" in source
    expected_bytes = 8 if dtype is T.float8_e4m3fn else 16
    assert f"tl::cp_async_gs<{expected_bytes}>" in source
    assert "tl::cp_async_commit()" in source
    assert "tl::cp_async_wait<0>()" in source


@tilelang.testing.requires_musa
@pytest.mark.parametrize("dtype", [T.float32, T.float8_e4m3fn])
def test_musa_async_copy_runtime_values(dtype):
    kernel = tilelang.compile(
        _async_copy_kernel(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    src_cpu = torch.arange(_M, dtype=torch.float32)
    if dtype is T.float8_e4m3fn:
        src_cpu = (src_cpu % 8).to(torch.float8_e4m3fn)
    dst = kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(dst.cpu(), src_cpu, atol=0, rtol=0)
