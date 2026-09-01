import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


_NUM_ELEMS = 8


def _make_isfinite_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((_NUM_ELEMS,), dtype),
        dst: T.Tensor((_NUM_ELEMS,), T.int32),
    ):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(_NUM_ELEMS):
                dst[i] = T.if_then_else(T.isfinite(src[i]), 1, 0)

    return kernel


@pytest.mark.parametrize("dtype", [T.float32, T.float64])
@tilelang.testing.requires_musa
def test_musa_common_isfinite_codegen(dtype):
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _make_isfinite_kernel(dtype),
            target=target,
            enable_device_compile=False,
        )

    assert "isfinite(" in artifact.kernel_source
    assert "tirx.isfinite" not in artifact.kernel_source


@pytest.mark.parametrize(
    ("dtype", "torch_dtype"),
    [(T.float32, torch.float32), (T.float64, torch.float64)],
)
@tilelang.testing.requires_musa
def test_musa_common_isfinite_runtime(dtype, torch_dtype):
    kernel = tilelang.compile(
        _make_isfinite_kernel(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    src_cpu = torch.tensor(
        [0.0, -0.0, 1.0, -2.5, float("inf"), float("-inf"), float("nan"), 3.25],
        dtype=torch_dtype,
    )
    out = kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(out.cpu(), torch.isfinite(src_cpu).to(torch.int32))


if __name__ == "__main__":
    tilelang.testing.main()
