import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@T.prim_func
def _dp4a_kernel(
    a: T.Tensor((4,), T.int8),
    b: T.Tensor((4,), T.int8),
    c: T.Tensor((1,), T.int32),
):
    with T.Kernel(1, threads=1):
        c[0] = 7
        T.dp4a(a, b, c)


@tilelang.testing.requires_musa
def test_musa_common_dp4a_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _dp4a_kernel,
            target=target,
            enable_device_compile=False,
        )

    source = artifact.kernel_source
    assert source is not None
    assert "#include <tl_templates/musa/common/dp4a.h>" in source
    assert "tl::DP4A(" in source


@pytest.mark.parametrize(
    "a,b",
    [
        ([1, 2, 3, 4], [5, 6, 7, 8]),
        ([-1, 2, -3, 4], [5, -6, 7, -8]),
        ([127, -128, 1, -1], [-1, 1, 127, -128]),
    ],
)
@tilelang.testing.requires_musa
def test_musa_common_dp4a_runtime(a, b):
    kernel = tilelang.compile(
        _dp4a_kernel,
        out_idx=[2],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    a_cpu = torch.tensor(a, dtype=torch.int8)
    b_cpu = torch.tensor(b, dtype=torch.int8)
    out = kernel(a_cpu.to("musa"), b_cpu.to("musa"))
    torch.musa.synchronize()

    expected = 7 + sum(x * y for x, y in zip(a, b))
    torch.testing.assert_close(out.cpu(), torch.tensor([expected], dtype=torch.int32))


if __name__ == "__main__":
    tilelang.testing.main()
