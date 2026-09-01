import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


_N = 128


@T.prim_func
def _fast_divmod_kernel(
    x: T.Tensor((_N,), T.int32),
    q: T.Tensor((_N,), T.int32),
    r: T.Tensor((_N,), T.int32),
):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(_N):
            q[i], r[i] = T.fast_divmod(x[i], 7)


@T.prim_func
def _fast_div_kernel(
    x: T.Tensor((_N,), T.int32),
    q: T.Tensor((_N,), T.int32),
):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(_N):
            q[i] = T.fast_div(x[i], 7)


@T.prim_func
def _fast_mod_kernel(
    x: T.Tensor((_N,), T.int32),
    r: T.Tensor((_N,), T.int32),
):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(_N):
            r[i] = T.fast_mod(x[i], 7)


@T.prim_func
def _fast_div_dynamic_kernel(
    x: T.Tensor((_N,), T.int32),
    q: T.Tensor((_N,), T.int32),
    divisor: T.int32,
):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(_N):
            q[i] = T.fast_div(x[i], divisor)


@tilelang.testing.requires_musa
def test_musa_common_fast_divmod_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _fast_divmod_kernel,
            target=target,
            enable_device_compile=False,
        )

    src = artifact.kernel_source
    assert src is not None
    assert "#include <tl_templates/musa/common/fast_divmod.h>" in src
    assert "tl::fast_div(" in src
    assert "tl::fast_mod(" not in src
    # The code generator may spell the loop index as threadIdx.x and assign
    # the quotient/remainder through temporary variables.
    assert " - (" in src and " * 7" in src


@tilelang.testing.requires_musa
def test_musa_common_fast_divmod_single_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        div_artifact = tilelang.lower(
            _fast_div_kernel,
            target=target,
            enable_device_compile=False,
        )
        mod_artifact = tilelang.lower(
            _fast_mod_kernel,
            target=target,
            enable_device_compile=False,
        )

    div_src = div_artifact.kernel_source
    mod_src = mod_artifact.kernel_source
    assert div_src is not None and mod_src is not None
    assert "#include <tl_templates/musa/common/fast_divmod.h>" in div_src
    assert "#include <tl_templates/musa/common/fast_divmod.h>" in mod_src
    assert "tl::fast_div(" in div_src
    assert "tl::fast_mod(" in mod_src


@tilelang.testing.requires_musa
def test_musa_common_fast_divmod_dynamic_divisor_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _fast_div_dynamic_kernel,
            target=target,
            enable_device_compile=False,
        )

    src = artifact.kernel_source
    assert src is not None
    assert "tl::fast_div(" in src
    assert "musa_fast_div_multiplier" in src

    kernel = tilelang.compile(
        _fast_div_dynamic_kernel,
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    values = torch.arange(_N, dtype=torch.int32) * 11 + 3
    for divisor in (1, 3, 7, 31):
        out = kernel(values.to("musa"), divisor)
        torch.musa.synchronize()
        expected = torch.div(values, divisor, rounding_mode="floor")
        torch.testing.assert_close(out.cpu(), expected)


@tilelang.testing.requires_musa
def test_musa_common_fast_divmod_runtime():
    kernel = tilelang.compile(
        _fast_divmod_kernel,
        out_idx=[1, 2],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    values = torch.arange(_N, dtype=torch.int32) * 3 + 1
    q, r = kernel(values.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(q.cpu(), torch.div(values, 7, rounding_mode="floor"))
    torch.testing.assert_close(r.cpu(), torch.remainder(values, 7))


if __name__ == "__main__":
    tilelang.testing.main()
