import pytest
import torch
import torch_musa  # noqa: F401
import torch.nn.functional as F

from flashinfer.gemm import bmm_bf16
from flashinfer.gemm import bmm_fp8
from flashinfer.gemm import gemm_fp8_nt_groupwise
from mate.testing import supported_musa_compute_capability
from mate.testing.utils import (
    group_dequantize_fp8,
    group_quantize_fp8,
    per_token_cast_to_fp8,
    tensor_quantize_fp8,
)
from mate.utils import ceil_div


def _fp8_output_reference(x: torch.Tensor):
    pad = ceil_div(x.size(-1), 128) * 128 - x.size(-1)
    x_padded = F.pad(x, (0, pad))
    x_fp8, scale = per_token_cast_to_fp8(x_padded, torch.float8_e4m3fn)
    dequant = (
        x_fp8.float().reshape(*x_fp8.shape[:-1], scale.size(-1), 128)
        * scale.unsqueeze(-1)
    ).reshape_as(x_fp8)
    return scale, dequant[..., : x.size(-1)].contiguous()


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("m", [128, 2048])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 2048])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("scale_granularity_mnk", [(1, -1, -1), (-1, -1, -1)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
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
    use_graph,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)
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
            bmm_fp8(
                fp8_a,
                fp8_b.transpose(-2, -1),
                scale_a,
                scale_b,
                out_dtype,
                d,
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

        bmm_fp8(
            fp8_a,
            fp8_b.transpose(-2, -1),
            scale_a,
            scale_b,
            out_dtype,
            d,
        )

    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("m", [128, 2048])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 2048])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_bmm_fp16(batch, m, n, k, dtype, strided) -> None:
    torch.manual_seed(666)
    torch.musa.manual_seed(666)
    if strided:
        a = torch.rand((batch, m + 99, k), device="musa", dtype=dtype)[:, :m, :]
        b = torch.rand((batch, n + 111, k), device="musa", dtype=dtype)[:, :n, :]
        d = torch.empty((batch, m + 777, n), device="musa", dtype=dtype)[:, :m, :]
    else:
        a = torch.rand((batch, m, k), device="musa", dtype=dtype)
        b = torch.rand((batch, n, k), device="musa", dtype=dtype)
        d = torch.empty((batch, m, n), device="musa", dtype=dtype)

    ref_d = torch.bmm(a, b.transpose(-2, -1))

    bmm_bf16(a, b.transpose(-2, -1), out=d, out_dtype=dtype)

    torch.testing.assert_close(d, ref_d, rtol=5e-3, atol=5e-3)
    ref_d_fp32 = torch.bmm(a.float(), b.float().transpose(-2, -1))
    if strided:
        d_fp32 = torch.empty((batch, m + 777, n), device="musa", dtype=torch.float32)[
            :, :m, :
        ]
    else:
        d_fp32 = torch.empty((batch, m, n), device="musa", dtype=torch.float32)

    for out_arg in (None, d_fp32):
        out = bmm_bf16(
            a,
            b.transpose(-2, -1),
            out=out_arg,
            out_dtype=torch.float32,
        )
        assert out.dtype == torch.float32
        if out_arg is not None:
            assert out.data_ptr() == out_arg.data_ptr()
        torch.testing.assert_close(out, ref_d_fp32, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("out_preallocated", [False, True])
def test_bmm_fp16_float32_output_v024(dtype, strided, out_preallocated):
    batch, m, n, k = 2, 128, 256, 512
    if strided:
        a = torch.rand((batch, m + 11, k), device="musa", dtype=dtype)[:, :m, :]
        b = torch.rand((batch, n + 13, k), device="musa", dtype=dtype)[:, :n, :]
    else:
        a = torch.rand((batch, m, k), device="musa", dtype=dtype)
        b = torch.rand((batch, n, k), device="musa", dtype=dtype)

    out_arg = None
    if out_preallocated:
        if strided:
            out_arg = torch.empty(
                (batch, m + 17, n), device="musa", dtype=torch.float32
            )[:, :m, :]
        else:
            out_arg = torch.empty((batch, m, n), device="musa", dtype=torch.float32)

    out = bmm_bf16(
        a,
        b.transpose(-2, -1),
        out=out_arg,
        out_dtype=torch.float32,
    )
    ref = torch.bmm(a.float(), b.float().transpose(-2, -1))
    assert out.dtype == torch.float32
    if out_arg is not None:
        assert out.data_ptr() == out_arg.data_ptr()
    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)


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
    "scale_granularity_mnk", [(1, 128, 128), (1, 1, -1), (1, -1, -1)]
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("scale_major", ["K", "MN"])
@pytest.mark.parametrize("use_graph", [False, True])
def test_gemm_fp8_nt_groupwise(
    m,
    n,
    k,
    ab_fp8_type,
    scale_granularity_mnk,
    out_dtype,
    scale_major,
    use_graph,
):
    a_fp8_type, b_fp8_type = ab_fp8_type
    a = torch.rand((m, k), device="musa", dtype=torch.float)
    d = torch.empty((m, n), device="musa", dtype=out_dtype)

    scale_granularity_m, scale_granularity_n, scale_granularity_k = (
        scale_granularity_mnk
    )
    scale_granularity_m = m if scale_granularity_m == -1 else scale_granularity_m
    scale_granularity_n = n if scale_granularity_n == -1 else scale_granularity_n
    scale_granularity_k = k if scale_granularity_k == -1 else scale_granularity_k

    if use_graph and n % scale_granularity_n != 0:
        pytest.skip("graph case requires N to match its scale granularity")

    padded_n = ceil_div(n, scale_granularity_n) * scale_granularity_n
    padded_b = torch.zeros((padded_n, k), device="musa", dtype=torch.float)
    b = padded_b[:n, :]
    b.uniform_(0, 1)
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
            padded_b, scale_b_shape, quant_tile_shape_b, b_fp8_type, scale_major
        )
    fp8_b_actual = fp8_b[:n, :].contiguous()

    if use_graph:
        g = torch.musa.MUSAGraph()
        with torch.musa.graph(g):
            gemm_fp8_nt_groupwise(
                fp8_a,
                fp8_b,
                scale_a,
                scale_b,
                scale_major,
                1,
                scale_granularity_mnk,
                d,
                out_dtype,
            )

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
        g.replay()
    else:
        ref_a = group_dequantize_fp8(fp8_a, scale_a, scale_major)
        ref_b = group_dequantize_fp8(fp8_b, scale_b, scale_major)[:n, :]
        ref_d = torch.matmul(ref_a, ref_b.t())

        gemm_fp8_nt_groupwise(
            fp8_a,
            fp8_b_actual,
            scale_a,
            scale_b,
            scale_major,
            1,
            scale_granularity_mnk,
            d,
            out_dtype,
        )

    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("out_dtype_arg", [None, torch.float8_e4m3fn])
def test_gemm_fp8_nt_groupwise_uses_provided_out_dtype(out_dtype, out_dtype_arg):
    m = n = k = 128
    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((n, k), device="musa", dtype=torch.float)
    fp8_a, scale_a = group_quantize_fp8(a, (m, 1), (1, 128), torch.float8_e4m3fn, "K")
    fp8_b, scale_b = group_quantize_fp8(b, (1, 1), (128, 128), torch.float8_e4m3fn, "K")
    out = torch.empty((m, n), device="musa", dtype=out_dtype)
    ref = torch.matmul(
        group_dequantize_fp8(fp8_a, scale_a, "K"),
        group_dequantize_fp8(fp8_b, scale_b, "K").t(),
    )

    result = gemm_fp8_nt_groupwise(
        fp8_a,
        fp8_b,
        scale_a,
        scale_b,
        "K",
        1,
        (1, 128, 128),
        out,
        out_dtype_arg,
    )
    assert result is out
    torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("noncontiguous_scale", ["a", "b"])
def test_gemm_fp8_nt_groupwise_requires_contiguous_scales(noncontiguous_scale):
    m, n, k = 128, 256, 128
    a = torch.zeros((m, k), device="musa", dtype=torch.float8_e4m3fn)
    b = torch.zeros((n, k), device="musa", dtype=torch.float8_e4m3fn)
    scale_a = torch.ones((m, 1), device="musa", dtype=torch.float32)
    scale_b = torch.ones((n // 128, 1), device="musa", dtype=torch.float32)
    if noncontiguous_scale == "a":
        scale_a = torch.ones((m, 2), device="musa", dtype=torch.float32)[:, ::2]
    else:
        scale_b = torch.ones((n // 128, 2), device="musa", dtype=torch.float32)[:, ::2]

    with pytest.raises(
        ValueError, match=f"{noncontiguous_scale}_scale must be contiguous"
    ):
        gemm_fp8_nt_groupwise(
            a,
            b,
            scale_a,
            scale_b,
            "K",
            1,
            (1, 128, 128),
        )


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
@pytest.mark.parametrize("use_graph", [False, True])
def test_gemm_fp8_nt_groupwise_fp8_output(m, n, k, ab_fp8_type, use_graph):
    if use_graph and n % 128 != 0:
        pytest.skip("graph case requires N to match its scale granularity")

    a_fp8_type, b_fp8_type = ab_fp8_type
    a = torch.rand((m, k), device="musa", dtype=torch.float)
    padded_n = ceil_div(n, 128) * 128
    padded_b = torch.zeros((padded_n, k), device="musa", dtype=torch.float)
    b = padded_b[:n, :]
    b.uniform_(0, 1)

    fp8_a, scale_a = group_quantize_fp8(
        a,
        (m, ceil_div(k, 128)),
        (1, 128),
        a_fp8_type,
        "K",
    )
    fp8_b, scale_b = group_quantize_fp8(
        padded_b,
        (ceil_div(n, 128), ceil_div(k, 128)),
        (128, 128),
        b_fp8_type,
        "K",
    )
    fp8_b_actual = fp8_b[:n, :].contiguous()
    d = torch.empty((m, n), device="musa", dtype=torch.float8_e4m3fn)
    out_scale = torch.empty((m, ceil_div(n, 128)), device="musa", dtype=torch.float32)

    if use_graph:
        g = torch.musa.MUSAGraph()
        with torch.musa.graph(g):
            gemm_fp8_nt_groupwise(
                fp8_a,
                fp8_b,
                scale_a,
                scale_b,
                "K",
                1,
                (1, 128, 128),
                d,
                torch.float8_e4m3fn,
                output_scale=out_scale,
            )

        a.uniform_(0, 1)
        b.uniform_(0, 1)
        new_fp8_a, new_scale_a = group_quantize_fp8(
            a,
            (m, ceil_div(k, 128)),
            (1, 128),
            a_fp8_type,
            "K",
        )
        new_fp8_b, new_scale_b = group_quantize_fp8(
            b,
            (ceil_div(n, 128), ceil_div(k, 128)),
            (128, 128),
            b_fp8_type,
            "K",
        )
        fp8_a.copy_(new_fp8_a)
        fp8_b.copy_(new_fp8_b)
        scale_a.copy_(new_scale_a)
        scale_b.copy_(new_scale_b)
        ref_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        ref_b = group_dequantize_fp8(fp8_b, scale_b, "K")
        ref_d = torch.matmul(ref_a, ref_b.t())
        out_scale_ref, ref_d = _fp8_output_reference(ref_d)
        g.replay()
    else:
        ref_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        ref_b = group_dequantize_fp8(fp8_b, scale_b, "K")[:n, :]
        ref_d = torch.matmul(ref_a, ref_b.t())
        out_scale_ref, ref_d = _fp8_output_reference(ref_d)
        gemm_fp8_nt_groupwise(
            fp8_a,
            fp8_b_actual,
            scale_a,
            scale_b,
            "K",
            1,
            (1, 128, 128),
            d,
            torch.float8_e4m3fn,
            output_scale=out_scale,
        )

    torch.testing.assert_close(out_scale, out_scale_ref, rtol=1e-2, atol=1e-2)
    pad = ceil_div(n, 128) * 128 - n
    d_padding = F.pad(d.view(torch.int8), (0, pad)).view(torch.float8_e4m3fn)
    d_dequant = (
        d_padding.float().reshape(*d.shape[:-1], out_scale.size(-1), 128)
        * out_scale.unsqueeze(-1)
    ).reshape_as(d_padding)
    d_dequant = d_dequant[:, :n].contiguous()
    similarity = F.cosine_similarity(d_dequant.flatten(), ref_d.flatten(), dim=0)
    assert similarity > 0.999


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
def test_gemm_fp8_nt_groupwise_strided_out(
    m, n, k, ab_fp8_type, out_dtype, scale_granularity_mnk, scale_major
):
    a_fp8_type, b_fp8_type = ab_fp8_type
    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((n, k), device="musa", dtype=torch.float)

    d_shape = (m, n)
    d_stride = (n * 2, 1)
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

    ref_d = torch.matmul(
        group_dequantize_fp8(fp8_a, scale_a, scale_major),
        group_dequantize_fp8(fp8_b, scale_b, scale_major).t(),
    )
    gemm_fp8_nt_groupwise(
        fp8_a,
        fp8_b,
        scale_a,
        scale_b,
        scale_major,
        1,
        scale_granularity_mnk,
        d,
        out_dtype,
    )

    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)
