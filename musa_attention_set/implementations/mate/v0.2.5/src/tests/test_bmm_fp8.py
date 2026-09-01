import torch
import torch_musa  # noqa: F401
import pytest

import mate
import mate.gemm
from mate.utils import ceil_div
from mate.testing.utils import (
    tensor_quantize_fp8,
    group_quantize_fp8,
    group_dequantize_fp8,
)
from mate.testing import supported_musa_compute_capability


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("m", [128, 2048])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 2048])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("scale_granularity_mnk", [(1, -1, -1), (-1, -1, -1)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("backend", ["mudnn", "auto"])
@pytest.mark.parametrize("use_graph", [False, True])
def test_bmm_fp8(
    batch,
    m,
    n,
    k,
    a_fp8_type,
    b_fp8_type,
    scale_granularity_mnk,
    out_dtype,
    backend,
    use_graph,
):
    a = torch.rand((batch, m, k), device="musa", dtype=torch.float)
    b = torch.rand((batch, n, k), device="musa", dtype=torch.float)

    d = torch.empty((batch, m, n), device="musa", dtype=out_dtype)

    scale_granularity_m, scale_granularity_n, scale_granularity_k = (
        scale_granularity_mnk
    )
    scale_granularity_m = m if scale_granularity_m == -1 else scale_granularity_m
    # scale_granularity_n = n if scale_granularity_n == -1 else scale_granularity_n
    scale_granularity_k = k if scale_granularity_k == -1 else scale_granularity_k

    quant_tile_shape_a = (1, scale_granularity_m, scale_granularity_k)
    scale_a_shape = (batch, m // scale_granularity_m, k // scale_granularity_k)

    if scale_granularity_mnk[0] == -1 and scale_granularity_mnk[1] == -1:
        fp8_a, scale_a = tensor_quantize_fp8(a, a_fp8_type)
    else:
        fp8_a, scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
        )
    fp8_b, scale_b = tensor_quantize_fp8(b, b_fp8_type)

    if use_graph:
        g = torch.musa.MUSAGraph()

        # capture
        with torch.musa.graph(g):
            mate.gemm.bmm_fp8(
                fp8_a,
                fp8_b,
                scale_a,
                scale_b,
                out_dtype,
                d,
                backend,
                scale_granularity_mnk,
            )

        a.uniform_(0, 1)
        b.uniform_(0, 1)

        if scale_granularity_mnk[0] == -1 and scale_granularity_mnk[1] == -1:
            new_fp8_a, new_scale_a = tensor_quantize_fp8(a, a_fp8_type)
        else:
            new_fp8_a, new_scale_a = group_quantize_fp8(
                a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
            )
        new_fp8_b, new_scale_b = tensor_quantize_fp8(b, b_fp8_type)

        fp8_a.copy_(new_fp8_a)
        fp8_b.copy_(new_fp8_b)
        scale_a.copy_(new_scale_a)
        scale_b.copy_(new_scale_b)

        ref_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        ref_b = group_dequantize_fp8(fp8_b, scale_b, "K")
        ref_d = torch.bmm(ref_a, ref_b.transpose(-2, -1))

        g.replay()

    else:
        ref_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        ref_b = group_dequantize_fp8(fp8_b, scale_b, "K")
        ref_d = torch.bmm(ref_a, ref_b.transpose(-2, -1))

        mate.gemm.bmm_fp8(
            fp8_a,
            fp8_b,
            scale_a,
            scale_b,
            out_dtype,
            d,
            backend,
            scale_granularity_mnk,
        )

    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)


def _make_groupwise_bmm_inputs(batch, m, n, k, recipe):
    a = torch.rand((batch, m, k), device="musa", dtype=torch.float)
    b = torch.rand((batch, n, k), device="musa", dtype=torch.float)
    _, scale_granularity_n, scale_granularity_k = recipe

    fp8_a, scale_a = group_quantize_fp8(
        a,
        (batch, m, ceil_div(k, scale_granularity_k)),
        (1, 1, scale_granularity_k),
        torch.float8_e4m3fn,
        "K",
    )
    fp8_b, scale_b = group_quantize_fp8(
        b,
        (
            batch,
            ceil_div(n, scale_granularity_n),
            ceil_div(k, scale_granularity_k),
        ),
        (1, scale_granularity_n, scale_granularity_k),
        torch.float8_e4m3fn,
        "K",
    )
    return fp8_a, scale_a, fp8_b, scale_b


def _a_arg_for_major(fp8_a, major_a_mode):
    return fp8_a if major_a_mode == "K" else fp8_a.transpose(-2, -1).contiguous()


def _b_arg_for_major(fp8_b, major_b_mode):
    return fp8_b.transpose(-2, -1).contiguous() if major_b_mode == "N" else fp8_b


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("recipe", [(1, 128, 128), (1, 1, 128)])
@pytest.mark.parametrize("major_a_mode", ["K", "M"])
@pytest.mark.parametrize("major_b_mode", ["N", "K"])
def test_bmm_fp8_groupwise_recipes(recipe, major_a_mode, major_b_mode):
    batch, m, n, k = 2, 128, 256, 384
    fp8_a, scale_a, fp8_b, scale_b = _make_groupwise_bmm_inputs(batch, m, n, k, recipe)

    a_arg = _a_arg_for_major(fp8_a, major_a_mode)
    b_arg = _b_arg_for_major(fp8_b, major_b_mode)

    d = torch.empty((batch, m, n), device="musa", dtype=torch.bfloat16)
    ref_d = torch.bmm(
        group_dequantize_fp8(fp8_a, scale_a, "K"),
        group_dequantize_fp8(fp8_b, scale_b, "K").transpose(-2, -1),
    )

    mate.gemm.bmm_fp8(
        a_arg,
        b_arg,
        scale_a,
        scale_b,
        torch.bfloat16,
        d,
        "mudnn",
        recipe,
        major_a_mode=major_a_mode,
        major_b_mode=major_b_mode,
    )

    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
def test_bmm_fp8_groupwise_fp32_accumulate_with_c():
    batch, m, n, k = 2, 128, 128, 128
    recipe = (1, 1, 128)
    fp8_a, scale_a, fp8_b, scale_b = _make_groupwise_bmm_inputs(batch, m, n, k, recipe)
    c = torch.randn((batch, m, n), device="musa", dtype=torch.float32)
    d = torch.empty_like(c)

    ref_d = c + torch.bmm(
        group_dequantize_fp8(fp8_a, scale_a, "K"),
        group_dequantize_fp8(fp8_b, scale_b, "K").transpose(-2, -1),
    )

    mate.gemm.bmm_fp8(
        fp8_a,
        fp8_b,
        scale_a,
        scale_b,
        torch.float32,
        d,
        "mudnn",
        recipe,
        c=c,
    )

    torch.testing.assert_close(d, ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("m", [128, 2048])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 2048])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("scale_granularity_mnk", [(1, -1, -1), (-1, -1, -1)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("backend", ["mudnn", "auto"])
def test_bmm_fp8_not_contiguous_output(
    batch,
    m,
    n,
    k,
    a_fp8_type,
    b_fp8_type,
    scale_granularity_mnk,
    out_dtype,
    backend,
):
    a = torch.rand((batch, m, k), device="musa", dtype=torch.float)
    b = torch.rand((batch, n, k), device="musa", dtype=torch.float)

    d_factor = 2
    d_shape = (batch, m, n)
    d_stride = (m * n * d_factor * d_factor, n * d_factor, 1)
    d_storage = sum((s - 1) * st for s, st in zip(d_shape, d_stride)) + 1
    d_storage_tensor = torch.empty(d_storage, dtype=out_dtype, device="musa")
    d = torch.as_strided(d_storage_tensor, size=d_shape, stride=d_stride)

    scale_granularity_m, scale_granularity_n, scale_granularity_k = (
        scale_granularity_mnk
    )
    scale_granularity_m = m if scale_granularity_m == -1 else scale_granularity_m
    # scale_granularity_n = n if scale_granularity_n == -1 else scale_granularity_n
    scale_granularity_k = k if scale_granularity_k == -1 else scale_granularity_k

    quant_tile_shape_a = (1, scale_granularity_m, scale_granularity_k)
    scale_a_shape = (batch, m // scale_granularity_m, k // scale_granularity_k)

    if scale_granularity_mnk[0] == -1 and scale_granularity_mnk[1] == -1:
        fp8_a, scale_a = tensor_quantize_fp8(a, a_fp8_type)
    else:
        fp8_a, scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
        )
    fp8_b, scale_b = tensor_quantize_fp8(b, b_fp8_type)

    ref_a = group_dequantize_fp8(fp8_a, scale_a, "K")
    ref_b = group_dequantize_fp8(fp8_b, scale_b, "K")
    ref_d = torch.bmm(ref_a, ref_b.transpose(-2, -1))

    mate.gemm.bmm_fp8(
        fp8_a,
        fp8_b,
        scale_a,
        scale_b,
        out_dtype,
        d,
        backend,
        scale_granularity_mnk,
    )

    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)
