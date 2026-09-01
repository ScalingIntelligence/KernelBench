import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.language.tir.op import pow_of_int


_NUM_ELEMS = 64
_NUM_STANDARD_RESULTS = 12
_NUM_FAST_RESULTS = 8


def _torch_dtype(dtype):
    return {
        T.float64: torch.float64,
        T.float32: torch.float32,
        T.float16: torch.float16,
        T.bfloat16: torch.bfloat16,
    }[dtype]


def _make_standard_math_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((_NUM_ELEMS,), dtype),
        dst: T.Tensor((_NUM_ELEMS, _NUM_STANDARD_RESULTS), dtype),
    ):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(_NUM_ELEMS):
                value = src[i]
                dst[i, 0] = T.exp(value)
                dst[i, 1] = T.log(value)
                dst[i, 2] = T.sqrt(value)
                dst[i, 3] = T.rsqrt(value)
                dst[i, 4] = T.sin(value)
                dst[i, 5] = T.cos(value)
                dst[i, 6] = T.tanh(value)
                dst[i, 7] = T.abs(value)
                dst[i, 8] = T.floor(value)
                dst[i, 9] = T.ceil(value)
                dst[i, 10] = T.trunc(value)
                dst[i, 11] = T.round(value)

    return kernel


def _make_fast_math_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((_NUM_ELEMS,), dtype),
        dst: T.Tensor((_NUM_ELEMS, _NUM_FAST_RESULTS), dtype),
    ):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(_NUM_ELEMS):
                value = src[i]
                dst[i, 0] = T.__exp(value)
                dst[i, 1] = T.__exp10(value)
                dst[i, 2] = T.__log(value)
                dst[i, 3] = T.__log2(value)
                dst[i, 4] = T.__log10(value)
                dst[i, 5] = T.__tan(value)
                dst[i, 6] = T.__cos(value)
                dst[i, 7] = T.__sin(value)

    return kernel


@T.prim_func
def _fast_rcp_kernel(
    src: T.Tensor((_NUM_ELEMS,), T.float32),
    dst: T.Tensor((_NUM_ELEMS,), T.float32),
):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(_NUM_ELEMS):
            dst[i] = T.fast_rcp(src[i])


def _standard_reference(src):
    work = src.to(torch.float64 if src.dtype == torch.float64 else torch.float32)
    ref = torch.stack(
        [
            torch.exp(work),
            torch.log(work),
            torch.sqrt(work),
            torch.rsqrt(work),
            torch.sin(work),
            torch.cos(work),
            torch.tanh(work),
            torch.abs(work),
            torch.floor(work),
            torch.ceil(work),
            torch.trunc(work),
            torch.round(work),
        ],
        dim=1,
    )
    return ref.to(src.dtype)


def _fast_reference(src):
    work = src.to(torch.float64 if src.dtype == torch.float64 else torch.float32)
    ref = torch.stack(
        [
            torch.exp(work),
            torch.pow(10.0, work),
            torch.log(work),
            torch.log2(work),
            torch.log10(work),
            torch.tan(work),
            torch.cos(work),
            torch.sin(work),
        ],
        dim=1,
    )
    return ref.to(src.dtype)


def _tolerances(dtype, fast=False):
    if dtype is T.float64:
        return (1e-12, 1e-12)
    if dtype is T.float32:
        return (2e-3, 2e-3) if fast else (1e-5, 1e-5)
    if dtype is T.float16:
        return (2e-2, 2e-2)
    return (5e-2, 5e-2)


@pytest.mark.parametrize("dtype", [T.float64, T.float32, T.float16, T.bfloat16])
@tilelang.testing.requires_musa
def test_musa_standard_math_runtime(dtype):
    kernel = tilelang.compile(
        _make_standard_math_kernel(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    torch_dtype = _torch_dtype(dtype)
    src_cpu = torch.linspace(0.25, 2.0, _NUM_ELEMS, dtype=torch.float32).to(torch_dtype)
    out = kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out.cpu(), _standard_reference(src_cpu), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [T.float64, T.float32, T.float16, T.bfloat16])
@tilelang.testing.requires_musa
def test_musa_fast_math_runtime(dtype):
    kernel = tilelang.compile(
        _make_fast_math_kernel(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    torch_dtype = _torch_dtype(dtype)
    src_cpu = torch.linspace(0.125, 0.75, _NUM_ELEMS, dtype=torch.float32).to(torch_dtype)
    out = kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    atol, rtol = _tolerances(dtype, fast=True)
    torch.testing.assert_close(out.cpu(), _fast_reference(src_cpu), atol=atol, rtol=rtol)


@tilelang.testing.requires_musa
def test_musa_fast_rcp_runtime():
    kernel = tilelang.compile(
        _fast_rcp_kernel,
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    src_cpu = torch.linspace(0.25, 4.0, _NUM_ELEMS, dtype=torch.float32)
    out = kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(out.cpu(), 1.0 / src_cpu, atol=2e-6, rtol=2e-6)


@tilelang.testing.requires_musa
def test_musa_math_template_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _make_fast_math_kernel(T.bfloat16),
            target=target,
            enable_device_compile=False,
        )
        rcp_artifact = tilelang.lower(
            _fast_rcp_kernel,
            target=target,
            enable_device_compile=False,
        )

    assert "#include <tl_templates/musa/common/math.h>" in artifact.kernel_source
    assert "#define TL_MUSA_ENABLE_BF16" in artifact.kernel_source
    assert "hexp(" in artifact.kernel_source
    assert "htan(" in artifact.kernel_source
    assert "tl::fast_rcp(" in rcp_artifact.kernel_source


@T.prim_func
def _infinity_kernel(dst: T.Tensor((_NUM_ELEMS,), T.float32)):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(_NUM_ELEMS):
            dst[i] = T.infinity(T.float32)


@T.prim_func
def _round_ties_away_kernel(
    src: T.Tensor((_NUM_ELEMS,), T.float32),
    dst: T.Tensor((_NUM_ELEMS,), T.float32),
):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(_NUM_ELEMS):
            dst[i] = T.round(src[i], "ties-away-from-zero")


@T.prim_func
def _pow_of_int_kernel(
    src: T.Tensor((_NUM_ELEMS,), T.float32),
    dst: T.Tensor((_NUM_ELEMS,), T.float32),
):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(_NUM_ELEMS):
            dst[i] = pow_of_int(src[i], 3)


@T.prim_func
def _special_value_predicate_kernel(
    src: T.Tensor((_NUM_ELEMS,), T.float32),
    is_inf: T.Tensor((_NUM_ELEMS,), T.int32),
    is_nan: T.Tensor((_NUM_ELEMS,), T.int32),
):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(_NUM_ELEMS):
            is_inf[i] = T.if_then_else(T.isinf(src[i]), 1, 0)
            is_nan[i] = T.if_then_else(T.isnan(src[i]), 1, 0)


@tilelang.testing.requires_musa
def test_musa_infinity_runtime():
    kernel = tilelang.compile(
        _infinity_kernel,
        out_idx=[0],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    out = kernel()
    torch.musa.synchronize()
    out_cpu = out.cpu()
    assert torch.isinf(out_cpu).all() and (out_cpu > 0).all()


@tilelang.testing.requires_musa
def test_musa_round_ties_away_runtime():
    kernel = tilelang.compile(
        _round_ties_away_kernel,
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    half_values = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    src = torch.tensor(
        half_values * (_NUM_ELEMS // len(half_values)),
        dtype=torch.float32,
        device="musa",
    )
    out = kernel(src)
    torch.musa.synchronize()
    src_cpu = src.cpu()
    expected = torch.sign(src_cpu) * torch.floor(torch.abs(src_cpu) + 0.5)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_pow_of_int_runtime():
    kernel = tilelang.compile(
        _pow_of_int_kernel,
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    src = torch.linspace(-2.0, 2.0, _NUM_ELEMS, dtype=torch.float32, device="musa")
    out = kernel(src)
    torch.musa.synchronize()
    torch.testing.assert_close(out.cpu(), src.cpu() ** 3, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_special_value_predicates_runtime():
    kernel = tilelang.compile(
        _special_value_predicate_kernel,
        out_idx=[1, 2],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    base_values = [0.0, 1.0, float("inf"), float("-inf"), float("nan")]
    values = [base_values[i % len(base_values)] for i in range(_NUM_ELEMS)]
    src = torch.tensor(values, dtype=torch.float32, device="musa")
    inf_out, nan_out = kernel(src)
    torch.musa.synchronize()
    expected_inf = torch.isinf(src.cpu()).to(torch.int32)
    expected_nan = torch.isnan(src.cpu()).to(torch.int32)
    torch.testing.assert_close(inf_out.cpu(), expected_inf, atol=0, rtol=0)
    torch.testing.assert_close(nan_out.cpu(), expected_nan, atol=0, rtol=0)
