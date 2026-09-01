import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


_GRID_X = 9
_GRID_Y = 7
_PANEL_SIZE = 3


def _make_swizzle_kernel(order):
    @T.prim_func
    def kernel(out: T.Tensor((_GRID_Y, _GRID_X), T.int32)):
        with T.Kernel(_GRID_X, _GRID_Y, threads=1) as (bx, by):
            T.use_swizzle(panel_size=_PANEL_SIZE, order=order)
            out[by, bx] = by * _GRID_X + bx

    return kernel


@pytest.mark.parametrize(
    "order,helper",
    [
        ("row", "rasterization2DRow"),
        ("column", "rasterization2DColumn"),
    ],
)
@tilelang.testing.requires_musa
def test_musa_common_threadblock_swizzle_codegen(order, helper):
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _make_swizzle_kernel(order),
            target=target,
            enable_device_compile=False,
        )

    source = artifact.kernel_source
    assert source is not None
    assert "#include <tl_templates/musa/common/threadblock_swizzle.h>" in source
    assert f"tl::{helper}<{_PANEL_SIZE}>()" in source


@pytest.mark.parametrize("order", ["row", "column"])
@tilelang.testing.requires_musa
def test_musa_common_threadblock_swizzle_runtime(order):
    kernel = tilelang.compile(
        _make_swizzle_kernel(order),
        out_idx=[0],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    out = kernel()
    torch.musa.synchronize()

    expected = torch.arange(_GRID_X * _GRID_Y, dtype=torch.int32).reshape(
        _GRID_Y, _GRID_X
    )
    torch.testing.assert_close(out.cpu(), expected)


if __name__ == "__main__":
    tilelang.testing.main()
