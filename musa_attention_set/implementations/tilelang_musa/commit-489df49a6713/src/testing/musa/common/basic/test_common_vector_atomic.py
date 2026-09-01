import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _vector_atomic_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((8,), dtype),
        dst: T.Tensor((8,), dtype),
    ):
        with T.Kernel(1, threads=1):
            for i in T.serial(8):
                dst[i] = T.cast(0, dtype)
            T.atomic_addx2(dst[0], src[0])
            T.atomic_addx4(dst[4], src[4])

    return kernel


def _vector_atomic_return_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((8,), dtype),
        dst: T.Tensor((8,), dtype),
        prev: T.Tensor((8,), dtype),
    ):
        with T.Kernel(1, threads=1):
            for i in T.serial(8):
                dst[i] = T.cast(0, dtype)
                prev[i] = T.cast(-1, dtype)
            prev[0:2] = T.atomic_addx2(dst[0:2], src[0:2], return_prev=True)
            prev[4:8] = T.atomic_addx4(dst[4:8], src[4:8], return_prev=True)

    return kernel


@T.prim_func
def _tiled_vector_atomic_kernel(dst: T.Tensor((4,), "float32")):
    with T.Kernel(1, threads=1):
        local = T.alloc_fragment((4,), "float32")
        for i in T.serial(4):
            dst[i] = 0.0
            local[i] = T.cast(i + 1, T.float32)
        for i in T.Parallel(4):
            T.atomic_add(dst[i], local[i])


@pytest.mark.parametrize("dtype", [T.float32, T.float16, T.bfloat16])
@tilelang.testing.requires_musa
def test_musa_common_vector_atomic_codegen(dtype):
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _vector_atomic_kernel(dtype),
            target=target,
            enable_device_compile=False,
        )
        return_artifact = tilelang.lower(
            _vector_atomic_return_kernel(dtype),
            target=target,
            enable_device_compile=False,
        )

    source = artifact.kernel_source
    assert source is not None
    assert "#include <tl_templates/musa/common/atomic.h>" in source
    assert "tl::AtomicAddx2(" in source
    assert "tl::AtomicAddx4(" in source

    return_source = return_artifact.kernel_source
    assert return_source is not None
    assert "tl::AtomicAddx2Ret(" in return_source
    assert "tl::AtomicAddx4Ret(" in return_source


@pytest.mark.parametrize(
    ("dtype", "torch_dtype"),
    [
        (T.float32, torch.float32),
        (T.float16, torch.float16),
        (T.bfloat16, torch.bfloat16),
    ],
)
@tilelang.testing.requires_musa
def test_musa_common_vector_atomic_runtime(dtype, torch_dtype):
    kernel = tilelang.compile(
        _vector_atomic_kernel(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    src_cpu = torch.arange(1, 9, dtype=torch_dtype)
    out = kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    expected = torch.tensor([1, 2, 0, 0, 5, 6, 7, 8], dtype=torch_dtype)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)

    return_kernel = tilelang.compile(
        _vector_atomic_return_kernel(dtype),
        out_idx=[1, 2],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    return_out, prev = return_kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(return_out.cpu(), expected, atol=0, rtol=0)
    expected_prev = torch.tensor([0, 0, -1, -1, 0, 0, 0, 0], dtype=torch_dtype)
    torch.testing.assert_close(prev.cpu(), expected_prev, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_tiled_atomic_add_vectorization():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _tiled_vector_atomic_kernel,
            target=target,
            enable_device_compile=False,
        )

    source = artifact.kernel_source
    assert source is not None
    assert "tl::AtomicAddx4(" in source

    kernel = tilelang.compile(
        _tiled_vector_atomic_kernel,
        out_idx=[0],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    out = kernel()
    torch.musa.synchronize()
    torch.testing.assert_close(
        out.cpu(), torch.arange(1, 5, dtype=torch.float32), atol=0, rtol=0
    )
