import torch
import torch_musa  # noqa: F401
import torch.nn.functional as F
import pytest

import mate
from mate.utils import ceil_div
from mate.testing.utils import (
    tensor_quantize_fp8,
    group_quantize_fp8,
    group_dequantize_fp8,
)
from mate.testing import supported_musa_compute_capability

_SCALE_GRANULARITY_OUTPUT_CASES = [
    ((1, 128, 128), torch.bfloat16, False),
    ((1, 128, 128), torch.half, False),
    ((1, 1, -1), torch.bfloat16, False),
    ((1, 1, -1), torch.half, False),
    ((1, -1, -1), torch.bfloat16, False),
    ((1, -1, -1), torch.half, False),
    ((1, 128, 128), torch.float8_e4m3fn, True),
]


def _fp8_quantize_1x128(x: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    pad = (128 - x.size(-1) % 128) % 128
    x_padded = torch.nn.functional.pad(x, (0, pad)) if pad else x
    x_blocks = x_padded.reshape(*x_padded.shape[:-1], x_padded.size(-1) // 128, 128)
    amax = x_blocks.abs().amax(dim=-1, keepdim=True)
    scale = finfo.max / amax.clamp(min=1e-12)
    q_padded = (
        (x_blocks * scale)
        .clamp(min=finfo.min, max=finfo.max)
        .to(torch.float8_e4m3fn)
        .reshape_as(x_padded)
    )
    scale_inv = scale.float().reciprocal()
    dequant_padded = (
        q_padded.reshape_as(x_blocks).to(torch.float32) * scale_inv
    ).reshape_as(x_padded)
    q = q_padded[..., : x.size(-1)].contiguous() if pad else q_padded
    dequant = dequant_padded[..., : x.size(-1)].contiguous() if pad else dequant_padded
    return q, scale_inv.squeeze(-1), dequant


def pad_to_multiple(x, dim, multiple=128):
    # for unalign scale
    size = x.shape[dim]
    target = ((size + multiple - 1) // multiple) * multiple
    pad = target - size

    if pad == 0:
        return x

    pads = [0] * (x.dim() * 2)
    pads[-(dim + 1) * 2 + 1] = pad
    return F.pad(x, pads)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("m", [128, 4096, 6360, 14400])
@pytest.mark.parametrize("n", [128, 4096, 11008, 12384])
@pytest.mark.parametrize("k", [128, 4096, 11008, 4608])
@pytest.mark.parametrize(
    "ab_fp8_type",
    [
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e5m2),
    ],
)
@pytest.mark.parametrize(
    ("scale_granularity_mnk", "out_dtype", "output_fp8"),
    _SCALE_GRANULARITY_OUTPUT_CASES,
    ids=[
        f"scale{scale_granularity_mnk}-out-{out_dtype}-output_fp8-{output_fp8}"
        for scale_granularity_mnk, out_dtype, output_fp8 in _SCALE_GRANULARITY_OUTPUT_CASES
    ],
)
@pytest.mark.parametrize("scale_major", ["K", "MN"])
@pytest.mark.parametrize("use_graph", [False, True])
def test_groupwise_fp8_scaled_gemm(
    m,
    n,
    k,
    ab_fp8_type,
    scale_granularity_mnk,
    out_dtype,
    output_fp8,
    scale_major,
    use_graph,
):
    a_fp8_type, b_fp8_type = ab_fp8_type

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((n, k), device="musa", dtype=torch.float)

    if scale_major != "K" and output_fp8:
        return

    d = (
        torch.empty((m, n), device="musa", dtype=torch.float8_e4m3fn)
        if output_fp8
        else torch.empty((m, n), device="musa", dtype=out_dtype)
    )
    out_scale = (
        torch.empty((m, ceil_div(n, 128)), device="musa", dtype=torch.float32)
        if output_fp8
        else None
    )

    scale_granularity_m, scale_granularity_n, scale_granularity_k = (
        scale_granularity_mnk
    )
    scale_granularity_m = m if scale_granularity_m == -1 else scale_granularity_m
    scale_granularity_n = n if scale_granularity_n == -1 else scale_granularity_n
    scale_granularity_k = k if scale_granularity_k == -1 else scale_granularity_k

    if use_graph and (n % scale_granularity_n != 0):
        return

    padding_b = pad_to_multiple(b, 0, scale_granularity_n)

    quant_tile_shape_a = (scale_granularity_m, scale_granularity_k)
    quant_tile_shape_b = (scale_granularity_n, scale_granularity_k)
    if scale_major == "K":
        scale_a_shape = (
            ceil_div(m, scale_granularity_m),
            ceil_div(k, scale_granularity_k),
        )
        scale_b_shape = (
            ceil_div(n, scale_granularity_n),
            ceil_div(k, scale_granularity_k),
        )
    else:
        # MN Major
        scale_a_shape = (
            ceil_div(k, scale_granularity_k),
            ceil_div(m, scale_granularity_m),
        )
        scale_b_shape = (
            ceil_div(k, scale_granularity_k),
            ceil_div(n, scale_granularity_n),
        )

    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, scale_major
    )
    if scale_granularity_mnk[1] == -1 and scale_granularity_mnk[2] == -1:
        fp8_b, scale_b = tensor_quantize_fp8(b, b_fp8_type)
    else:
        fp8_b, scale_b = group_quantize_fp8(
            padding_b, scale_b_shape, quant_tile_shape_b, b_fp8_type, scale_major
        )
    fp8_b_actual = fp8_b[:n, :].contiguous()

    if use_graph:
        g = torch.musa.MUSAGraph()
        # capture
        with torch.musa.graph(g):
            result = mate.gemm.gemm_fp8_nt_groupwise(
                fp8_a,
                fp8_b,
                scale_a,
                scale_b,
                scale_major,
                1,
                scale_granularity_mnk,
                d,
                out_dtype,
                "mudnn" if not output_fp8 else "auto",
                output_scale=out_scale,
            )
            if output_fp8:
                d = result

        a.uniform_(0, 1)
        b.uniform_(0, 1)

        new_fp8_a, new_scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, scale_major
        )
        if scale_granularity_mnk[1] == -1 and scale_granularity_mnk[2] == -1:
            new_fp8_b, new_scale_b = tensor_quantize_fp8(b, b_fp8_type)
        else:
            new_fp8_b, new_scale_b = group_quantize_fp8(
                b, scale_b_shape, quant_tile_shape_b, b_fp8_type, scale_major
            )

        fp8_a.copy_(new_fp8_a)
        fp8_b.copy_(new_fp8_b)
        scale_a.copy_(new_scale_a)
        scale_b.copy_(new_scale_b)

        ref_a = group_dequantize_fp8(fp8_a, scale_a, scale_major)
        ref_b = group_dequantize_fp8(fp8_b, scale_b, scale_major)
        ref_d = torch.matmul(ref_a, ref_b.t())
        if output_fp8:
            _, out_scale_ref, ref_d = _fp8_quantize_1x128(ref_d)

        g.replay()

    else:
        ref_a = group_dequantize_fp8(fp8_a, scale_a, scale_major)
        ref_b = group_dequantize_fp8(fp8_b, scale_b, scale_major)[:n, :]
        ref_d = torch.matmul(ref_a, ref_b.t())
        if output_fp8:
            _, out_scale_ref, ref_d = _fp8_quantize_1x128(ref_d)

        result = mate.gemm.gemm_fp8_nt_groupwise(
            fp8_a,
            fp8_b_actual,
            scale_a,
            scale_b,
            scale_major,
            1,
            scale_granularity_mnk,
            d,
            out_dtype,
            "mudnn" if not output_fp8 else "auto",
            output_scale=out_scale,
        )
        if output_fp8:
            d = result

    if output_fp8:
        torch.testing.assert_close(out_scale, out_scale_ref, rtol=1e-2, atol=1e-2)
        d_padding = pad_to_multiple(d.view(torch.int8), -1, 128).view(
            torch.float8_e4m3fn
        )
        d_dequant = (
            d_padding.float().reshape(*d.shape[:-1], out_scale.size(-1), 128)
            * out_scale.unsqueeze(-1)
        ).reshape_as(d_padding)
        d_dequant = d_dequant[:, :n].contiguous()
        sim = F.cosine_similarity(d_dequant.flatten(), ref_d.flatten(), dim=0)
        assert sim > 0.999
    else:
        torch.testing.assert_close(d.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("out_dtype_arg", [None, torch.float8_e4m3fn])
def test_groupwise_fp8_scaled_gemm_uses_provided_out_dtype(out_dtype, out_dtype_arg):
    m = n = k = 128
    scale_major = "K"
    scale_granularity_mnk = (1, 128, 128)

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((n, k), device="musa", dtype=torch.float)

    fp8_a, scale_a = group_quantize_fp8(
        a,
        (m, 1),
        (1, 128),
        torch.float8_e4m3fn,
        scale_major,
    )
    fp8_b, scale_b = group_quantize_fp8(
        b,
        (1, 1),
        (128, 128),
        torch.float8_e4m3fn,
        scale_major,
    )

    out = torch.empty((m, n), device="musa", dtype=out_dtype)
    ref_d = torch.matmul(
        group_dequantize_fp8(fp8_a, scale_a, scale_major),
        group_dequantize_fp8(fp8_b, scale_b, scale_major).t(),
    )

    result = mate.gemm.gemm_fp8_nt_groupwise(
        fp8_a,
        fp8_b,
        scale_a,
        scale_b,
        scale_major,
        1,
        scale_granularity_mnk,
        out,
        out_dtype_arg,
    )

    assert result is out
    torch.testing.assert_close(out.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("m", [128, 4096])
@pytest.mark.parametrize("n", [128, 4096])
@pytest.mark.parametrize("k", [128, 4096])
@pytest.mark.parametrize(
    "ab_fp8_type",
    [
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e5m2),
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize(
    "scale_granularity_mnk", [(1, 128, 128), (1, 1, -1), (1, -1, -1)]
)
@pytest.mark.parametrize("scale_major", ["K", "MN"])
def test_groupwise_fp8_scaled_gemm_not_contiguous_out(
    m, n, k, ab_fp8_type, out_dtype, scale_granularity_mnk, scale_major
):
    a_fp8_type, b_fp8_type = ab_fp8_type

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((n, k), device="musa", dtype=torch.float)

    d_factor = 2
    d_shape = (m, n)
    d_stride = (n * d_factor, 1)
    d_storage = sum((s - 1) * st for s, st in zip(d_shape, d_stride)) + 1
    d_storage_tensor = torch.empty(d_storage, dtype=out_dtype, device="musa")
    d = torch.as_strided(d_storage_tensor, size=d_shape, stride=d_stride)

    scale_granularity_m, scale_granularity_n, scale_granularity_k = (
        scale_granularity_mnk
    )
    scale_granularity_m = m if scale_granularity_m == -1 else scale_granularity_m
    scale_granularity_n = n if scale_granularity_n == -1 else scale_granularity_n
    scale_granularity_k = k if scale_granularity_k == -1 else scale_granularity_k

    quant_tile_shape_a = (scale_granularity_m, scale_granularity_k)
    quant_tile_shape_b = (scale_granularity_n, scale_granularity_k)
    if scale_major == "K":
        scale_a_shape = (m // scale_granularity_m, k // scale_granularity_k)
        scale_b_shape = (n // scale_granularity_n, k // scale_granularity_k)
    else:
        # MN Major
        scale_a_shape = (k // scale_granularity_k, m // scale_granularity_m)
        scale_b_shape = (k // scale_granularity_k, n // scale_granularity_n)

    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, scale_major
    )
    if scale_granularity_mnk[1] == -1 and scale_granularity_mnk[2] == -1:
        fp8_b, scale_b = tensor_quantize_fp8(b, b_fp8_type)
    else:
        fp8_b, scale_b = group_quantize_fp8(
            b, scale_b_shape, quant_tile_shape_b, b_fp8_type, scale_major
        )

    ref_a = group_dequantize_fp8(fp8_a, scale_a, scale_major)
    ref_b = group_dequantize_fp8(fp8_b, scale_b, scale_major)
    ref_d = torch.matmul(ref_a, ref_b.t())

    mate.gemm.gemm_fp8_nt_groupwise(
        fp8_a,
        fp8_b,
        scale_a,
        scale_b,
        scale_major,
        1,
        scale_granularity_mnk,
        d,
        out_dtype,
        "mudnn",
    )

    torch.testing.assert_close(d.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)
