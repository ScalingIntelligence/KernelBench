import torch
import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

_NUM_THREADS = 128


def _torch_dtype(tilelang_dtype):
    return {
        T.int32: torch.int32,
        T.float32: torch.float32,
        T.float16: torch.float16,
        T.bfloat16: torch.bfloat16,
    }[tilelang_dtype]


def _common_atomic_kernel(dtype):
    @T.prim_func
    def kernel(A: T.Tensor((_NUM_THREADS,), dtype), B: T.Tensor((8,), dtype)):
        with T.Kernel(1, threads=_NUM_THREADS):
            tx = T.get_thread_binding()

            if tx == 0:
                B[0] = T.cast(0, dtype)
                B[1] = T.cast(-1, dtype)
                B[2] = T.cast(2, dtype)
                B[3] = T.cast(0, dtype)
                T.atomic_store(B[4], T.cast(7, dtype), memory_order="release")
            T.sync_threads()

            T.atomic_add(B[0], A[tx])
            T.atomic_max(B[1], A[tx])
            T.atomic_min(B[2], A[tx])
            if dtype == T.int32:
                T.atomic_or(B[3], A[tx])
            T.sync_threads()

            if tx == 0:
                B[5] = T.atomic_load(B[4], memory_order="acquire")
                B[6] = T.atomic_add(B[0], T.cast(1, dtype), return_prev=True)
                B[7] = T.atomic_max(B[1], T.cast(3, dtype), return_prev=True)

    return kernel


@pytest.mark.parametrize("dtype", [T.int32, T.float32, T.float16, T.bfloat16])
@tilelang.testing.requires_musa
def test_musa_common_atomic_codegen(dtype):
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _common_atomic_kernel(dtype),
            target=target,
            enable_device_compile=False,
        )

    assert artifact.kernel_source is not None
    assert "#include <tl_templates/musa/common/atomic.h>" in artifact.kernel_source
    assert "tl::AtomicAdd(" in artifact.kernel_source
    assert "tl::AtomicMax(" in artifact.kernel_source
    assert "tl::AtomicMin(" in artifact.kernel_source
    assert "tl::AtomicLoad(" in artifact.kernel_source
    assert "tl::AtomicStore(" in artifact.kernel_source
    if dtype == T.int32:
        assert "tl::AtomicOr(" in artifact.kernel_source


@pytest.mark.parametrize("dtype", [T.int32, T.float32, T.float16, T.bfloat16])
@tilelang.testing.requires_musa
def test_musa_common_atomic_runtime_values(dtype):
    kernel = tilelang.compile(
        _common_atomic_kernel(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    torch_dtype = _torch_dtype(dtype)
    a_cpu = torch.ones((_NUM_THREADS,), dtype=torch_dtype)
    out = kernel(a_cpu.to("musa"))
    torch.musa.synchronize()

    out_cpu = out.cpu()
    expected_or = 1 if dtype == T.int32 else 0
    expected = torch.tensor(
        [_NUM_THREADS + 1, 3, 1, expected_or, 7, 7, _NUM_THREADS, 1],
        dtype=torch_dtype,
    )
    torch.testing.assert_close(out_cpu, expected, atol=0, rtol=0)
