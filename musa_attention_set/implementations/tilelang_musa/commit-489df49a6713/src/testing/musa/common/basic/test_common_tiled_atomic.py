import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

_M = 16
_N = 16
_NUM_THREADS = 128


@T.prim_func
def _tiled_atomic_kernel(
    src: T.Tensor((_M, _N), "float32"),
    add_out: T.Tensor((_M, _N), "float32"),
    max_out: T.Tensor((_M, _N), "float32"),
    min_out: T.Tensor((_M, _N), "float32"),
):
    with T.Kernel(1, threads=_NUM_THREADS):
        shared = T.alloc_shared((_M, _N), "float32")
        T.copy(src, shared)

        for i, j in T.Parallel(_M, _N):
            add_out[i, j] = 0.0
            max_out[i, j] = -100.0
            min_out[i, j] = 100.0
        T.sync_threads()

        T.atomic_add(add_out, shared)
        T.atomic_max(max_out, shared)
        T.atomic_min(min_out, shared)


@T.prim_func
def _tiled_atomic_scalar_kernel(out: T.Tensor((_M, _N), "float32")):
    with T.Kernel(1, threads=_NUM_THREADS):
        for i, j in T.Parallel(_M, _N):
            out[i, j] = 0.0
        T.sync_threads()
        T.atomic_add(out, 1.0)


@tilelang.testing.requires_musa
def test_musa_tiled_atomic_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _tiled_atomic_kernel,
            target=target,
            enable_device_compile=False,
        )
        scalar_artifact = tilelang.lower(
            _tiled_atomic_scalar_kernel,
            target=target,
            enable_device_compile=False,
        )

    source = artifact.kernel_source
    assert source is not None
    assert "tl::AtomicAddx2(" in source or "tl::AtomicAddx4(" in source
    assert "tl::AtomicMax(" in source
    assert "tl::AtomicMin(" in source

    scalar_source = scalar_artifact.kernel_source
    assert scalar_source is not None
    assert "tl::AtomicAddx2(" in scalar_source or "tl::AtomicAddx4(" in scalar_source


@tilelang.testing.requires_musa
def test_musa_tiled_atomic_runtime_values():
    kernel = tilelang.compile(
        _tiled_atomic_kernel,
        out_idx=[1, 2, 3],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    src_cpu = torch.linspace(-4.0, 4.0, _M * _N, dtype=torch.float32).reshape(_M, _N)
    add_out, max_out, min_out = kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(add_out.cpu(), src_cpu, atol=0, rtol=0)
    torch.testing.assert_close(max_out.cpu(), src_cpu, atol=0, rtol=0)
    torch.testing.assert_close(min_out.cpu(), src_cpu, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_tiled_atomic_scalar_runtime_values():
    kernel = tilelang.compile(
        _tiled_atomic_scalar_kernel,
        out_idx=[0],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    out = kernel()
    torch.musa.synchronize()
    torch.testing.assert_close(out.cpu(), torch.ones((_M, _N), dtype=torch.float32))
