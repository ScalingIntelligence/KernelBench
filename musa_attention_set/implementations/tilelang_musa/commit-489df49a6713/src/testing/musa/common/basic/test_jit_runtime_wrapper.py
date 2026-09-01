import torch

import tilelang.testing
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.jit.adapter import TVMFFIKernelAdapter
from tilelang.backend.runtime_device import resolve_runtime_device


@T.prim_func
def _copy_one(A: T.Tensor((1,), "float32"), B: T.Tensor((1,), "float32")):
    with T.Kernel(1, threads=32):
        B[0] = A[0]


def test_musa_runtime_device_is_backend_owned():
    runtime_device = resolve_runtime_device(tvm.target.Target({"kind": "musa"}))

    assert runtime_device.target_kind == "musa"
    assert runtime_device.current_device.__module__ == "tilelang.musa.runtime"


@tilelang.testing.requires_musa
def test_musa_tvm_ffi_jit_wrapper():
    kernel = tilelang.compile(
        _copy_one,
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    assert kernel.execution_backend == "tvm_ffi"
    assert isinstance(kernel.adapter, TVMFFIKernelAdapter)
    assert kernel.adapter.target.kind.name == "musa"
    assert "__global__" in kernel.get_kernel_source()

    a_cpu = torch.tensor([3.25], dtype=torch.float32)
    a = a_cpu.to("musa")
    b = kernel(a)
    torch.musa.synchronize()

    assert b.device.type == "musa"
    torch.testing.assert_close(b.cpu(), a_cpu)
