import torch
import torch_musa  # noqa: F401
import pytest

import mate
from mate.utils import ceil_div
from mate.testing.utils import (
    group_quantize_fp8,
    group_dequantize_fp8,
    tensor_quantize_fp8,
    align,
    check_gemm_sbo_signal,
)
from mate.testing import supported_musa_compute_capability


def _pack_int4_k(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8
    assert x.size(-1) % 2 == 0
    x_i16 = x.to(torch.int16)
    low = x_i16[..., 0::2] & 0xF
    high = x_i16[..., 1::2] & 0xF
    return (low | (high << 4)).to(torch.int8).contiguous()


def _quantize_w4a8_a_per_channel(x: torch.Tensor, out_dtype: torch.dtype):
    fp8_amax = torch.tensor(
        torch.finfo(out_dtype).max, device=x.device, dtype=torch.float32
    )
    abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(abs_max / fp8_amax)))
    return (x / (scale + 1e-8)).to(out_dtype), scale.float()


def _dequant_w4a8_a_per_channel(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale


def _quantize_w4a8_a_grouped(x: torch.Tensor, out_dtype: torch.dtype):
    block_k = 128
    k = x.size(-1)
    padded_k = ceil_div(k, block_k) * block_k
    padded = torch.zeros((*x.shape[:-1], padded_k), device=x.device, dtype=x.dtype)
    padded[..., :k].copy_(x)
    blocks = padded.reshape(*x.shape[:-1], padded_k // block_k, block_k)
    fp8_amax = torch.tensor(
        torch.finfo(out_dtype).max, device=x.device, dtype=torch.float32
    )
    abs_max = blocks.abs().amax(dim=-1).clamp(1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(abs_max / fp8_amax)))
    quantized = (blocks / (scale.unsqueeze(-1) + 1e-8)).to(out_dtype)
    return quantized.reshape(*x.shape[:-1], padded_k)[..., :k].contiguous(), scale


def _dequant_w4a8_a_grouped(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    expanded_scale = scale.repeat_interleave(128, dim=-1)[..., : x.size(-1)]
    return x.float() * expanded_scale


def _make_w4a8_b_nk(num_expert: int, n: int, k: int, device):
    b = torch.rand((num_expert, n, k), device=device, dtype=torch.float) * 2 - 1
    padded_k = ceil_div(k, 128) * 128
    padded_b = torch.zeros((num_expert, n, padded_k), device=device, dtype=torch.float)
    padded_b[..., :k].copy_(b)
    b_blocks = padded_b.reshape(num_expert, n, padded_k // 128, 128)
    scale_b = (b_blocks.abs().amax(dim=3).clamp(1e-4) / 7.0).to(torch.bfloat16)
    padded_b_int4 = (
        (b_blocks / scale_b.float().unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    )
    b_int4 = padded_b_int4.reshape(num_expert, n, padded_k)[..., :k].contiguous()
    packed_input = b_int4
    if k % 2:
        packed_input = torch.zeros(
            (num_expert, n, k + 1), device=device, dtype=torch.int8
        )
        packed_input[..., :k].copy_(b_int4)
    return b_int4, _pack_int4_k(packed_input), scale_b


def _dequant_w4a8_b_nk(
    b_int4: torch.Tensor, scale_b: torch.Tensor, expert_id: int, n: int, k: int
) -> torch.Tensor:
    b_scale = scale_b[expert_id].float().repeat_interleave(128, dim=1)[:, :k]
    return b_int4[expert_id].float() * b_scale


def _make_fp4fp8_b_nk(num_expert: int, n: int, k: int, device):
    codes = (
        (
            2
            + 2
            * (
                torch.arange(num_expert * n * (k // 2)).reshape(num_expert, n, k // 2)
                % 2
            )
        )
        .to(torch.uint8)
        .cpu()
    )
    packed_b = (codes | (codes << 4)).view(torch.int8).to(device)
    logical_b = (codes // 2).repeat_interleave(2, dim=-1).float()
    residual_e8m0 = (
        127
        + torch.arange(num_expert * n * ceil_div(k, 32)).reshape(
            num_expert, n, ceil_div(k, 32)
        )
        % 2
    ).to(dtype=torch.uint8)
    residual_f32 = torch.pow(2.0, residual_e8m0.to(torch.int32).float() - 127.0)
    residual_f32 = residual_f32.repeat_interleave(32, dim=-1)[..., :k]
    effective_b = (
        (logical_b * residual_f32).to(device=device, dtype=torch.float8_e4m3fn).float()
    )
    residual_e8m0 = residual_e8m0.to(device)
    epilogue_fp32 = (
        0.5 + (torch.arange(num_expert * n, device=device) % 2).reshape(num_expert, n)
    ).float()
    return effective_b, packed_b, residual_e8m0, epilogue_fp32


def _w4a8_ragged_cases():
    return [
        ([64, 128], 512, 512, 128),
        ([256, 256], 1024, 1024, 256),
    ]


def mixed_dtype_masked_moe_gemm_test_cases():
    return [
        ([64, 128], 512, 512, 128),
        ([256, 256], 1024, 1024, 512),
        ([0, 1, 17, 32], 256, 384, 256),
        ([1, 16, 17, 64], 256, 384, 16),
    ]


def test_moe_gemm_quant_recipe_validation():
    assert mate.gemm._resolve_moe_gemm_quant_recipe(
        "a_quant_recipe",
        (1, -1),
    ) == (1, -1)
    assert mate.gemm._resolve_moe_gemm_quant_recipe(
        "b_quant_recipe",
        (1, 128),
    ) == (1, 128)

    with pytest.raises(
        TypeError, match="a_quant_recipe must be a tuple of two int values"
    ):
        mate.gemm._resolve_moe_gemm_quant_recipe("a_quant_recipe", "fp8_per_channel")

    with pytest.raises(
        TypeError, match="b_quant_recipe must be a tuple of two int values"
    ):
        mate.gemm._resolve_moe_gemm_quant_recipe("b_quant_recipe", (1, 128, 128))


def get_ragged_moe_gemm_16bit_cases():
    return [
        [1, 0, 0],
        [111],
        [111, 222, 333, 444],
        [4096 for _ in range(8)],
        [8192 for _ in range(4)],
    ]


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group", get_ragged_moe_gemm_16bit_cases())
@pytest.mark.parametrize("n", [4096, 6144, 7168])
@pytest.mark.parametrize("k", [2048, 3072, 7168])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("alignment_m", [128, 256])
@pytest.mark.parametrize("psum_layout_param", [(False, 0)])
@pytest.mark.parametrize("use_graph", [False, True])
def test_ragged_moe_gemm_16bit(
    ms_per_group, n, k, data_type, alignment_m, psum_layout_param, use_graph
):
    use_psum_layout, expected_m_for_psum_layout = psum_layout_param

    num_expert = len(ms_per_group)
    aligned_ms = [align(m, alignment_m) for m in ms_per_group]
    m = sum(aligned_ms)

    a = torch.rand((m, k), device="musa", dtype=data_type)
    b = torch.rand((num_expert, n, k), device="musa", dtype=data_type)
    d = torch.empty((m, n), device="musa", dtype=data_type)

    m_indices = torch.full((m,), -1, device="musa", dtype=torch.int32)

    ref_d = torch.zeros((m, n), device="musa", dtype=data_type)

    if use_graph:
        g = torch.musa.MUSAGraph()
        with torch.musa.graph(g):
            mate.gemm.ragged_m_moe_gemm_16bit(
                a,
                b,
                m_indices,
                d,
                gemm_mode="per_token",
                alignment_m=alignment_m,
            )

        a.uniform_(0, 1)
        b.uniform_(0, 1)

        m_base = 0
        for i in range(num_expert):
            r = slice(m_base, m_base + ms_per_group[i])
            aligned_r = slice(m_base, m_base + aligned_ms[i])
            m_indices[r] = i
            ref_d[r] = torch.matmul(a[aligned_r], b[i].t())[: ms_per_group[i]]
            m_base += aligned_ms[i]

        g.replay()
    else:
        m_base = 0
        for i in range(num_expert):
            r = slice(m_base, m_base + ms_per_group[i])
            aligned_r = slice(m_base, m_base + aligned_ms[i])
            m_indices[r] = i
            ref_d[r] = torch.matmul(a[aligned_r], b[i].t())[: ms_per_group[i]]
            m_base += aligned_ms[i]

        mate.gemm.ragged_m_moe_gemm_16bit(
            a,
            b,
            m_indices,
            d,
            gemm_mode="per_token",
            alignment_m=alignment_m,
        )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d, ref_d, rtol=5e-3, atol=5e-3)


def get_masked_moe_gemm_16bit_cases():
    return [
        [1024 for _ in range(6)],
        [192 for _ in range(32)],
        [50 for _ in range(32)],
        [256, 256, 256, 256],
        [333, 444, 555, 666],
    ]


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group", get_masked_moe_gemm_16bit_cases())
@pytest.mark.parametrize("n", [4096, 6144, 7168])
@pytest.mark.parametrize("k", [2048, 3072, 7168])
@pytest.mark.parametrize("expected_m", [None, 128, 256])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("enable_overlap", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
def test_masked_moe_gemm_16bit(
    ms_per_group,
    n,
    k,
    expected_m,
    data_type,
    enable_overlap,
    use_graph,
):
    max_m = max(ms_per_group)
    expected_m = expected_m if expected_m is not None else max_m

    num_expert = len(ms_per_group)

    a = torch.rand((num_expert, max_m, k), device="musa", dtype=data_type)
    b = torch.rand((num_expert, n, k), device="musa", dtype=data_type)
    masked_m = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    d = torch.empty((num_expert, max_m, n), device="musa", dtype=data_type)

    tile_signal = 64
    signal = torch.zeros(
        num_expert * ceil_div(max_m, tile_signal), dtype=torch.int32, device=a.device
    )

    if use_graph:
        g = torch.musa.MUSAGraph()
        with torch.musa.graph(g):
            res = mate.gemm.masked_moe_gemm_16bit(
                a,
                b,
                masked_m,
                d,
                expected_m,
                enable_overlap=enable_overlap,
                signal=signal,
            )
        a.uniform_(0, 1)
        b.uniform_(0, 1)
        ref_d = torch.einsum("bmk,bnk->bmn", a, b).to(data_type)

        g.replay()
    else:
        ref_d = torch.einsum("bmk,bnk->bmn", a, b).to(data_type)

        res = mate.gemm.masked_moe_gemm_16bit(
            a,
            b,
            masked_m,
            d,
            expected_m,
            enable_overlap=enable_overlap,
            signal=signal,
        )

    for i in range(num_expert):
        torch.testing.assert_close(
            d[i, : ms_per_group[i], :],
            ref_d[i, : ms_per_group[i], :],
            rtol=5e-3,
            atol=5e-3,
        )

    if enable_overlap:
        block_m = res[2]
        threshold = res[3]
        check_gemm_sbo_signal(num_expert, max_m, block_m, threshold, signal, masked_m)


def get_ragged_moe_gemm_8bit_cases():
    return [
        [1, 0, 0],
        [111],
        [111, 222, 333, 444],
        [256],
        [4096 for _ in range(8)],
        [8192 for _ in range(4)],
    ]


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group", get_ragged_moe_gemm_8bit_cases())
@pytest.mark.parametrize("n", [4096, 6144, 7168])
@pytest.mark.parametrize("k", [2048, 3072, 7168])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("alignment_m", [128, 256])
@pytest.mark.parametrize("use_graph", [False, True])
def test_ragged_moe_gemm_8bit(
    ms_per_group, n, k, a_fp8_type, b_fp8_type, out_dtype, alignment_m, use_graph
):
    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)

    num_expert = len(ms_per_group)
    aligned_ms = [align(m, alignment_m) for m in ms_per_group]
    m = sum(aligned_ms)

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((num_expert, n, k), device="musa", dtype=torch.float)
    m_indices = torch.full((m,), -1, device="musa", dtype=torch.int32)

    d = torch.empty((m, n), device="musa", dtype=out_dtype)
    ref_d = torch.zeros((m, n), device="musa", dtype=torch.float)

    quant_tile_shape_a = (1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (m, k // quant_tile)
    scale_b_shape = (num_expert, n // quant_tile, k // quant_tile)
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
    )

    if use_graph:
        g = torch.musa.MUSAGraph()
        # capture
        with torch.musa.graph(g):
            mate.gemm.ragged_m_moe_gemm_8bit(
                (fp8_a, scale_a),
                (fp8_b, scale_b),
                m_indices,
                d,
                scale_granularity_mnk=scale_granularity_mnk,
                alignment_m=alignment_m,
            )

        a.uniform_(0, 1)
        b.uniform_(0, 1)

        new_fp8_a, new_scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
        )
        new_fp8_b, new_scale_b = group_quantize_fp8(
            b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
        )

        fp8_a.copy_(new_fp8_a)
        fp8_b.copy_(new_fp8_b)
        scale_a.copy_(new_scale_a)
        scale_b.copy_(new_scale_b)

        dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")

        # calc ref
        m_base = 0
        for i in range(num_expert):
            r = slice(m_base, m_base + ms_per_group[i])
            m_indices[r] = i
            ref_d[r] = torch.matmul(dequant_a[r], dequant_b[i].t())
            m_base += aligned_ms[i]

        g.replay()

    else:
        dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")

        # calc ref
        m_base = 0
        for i in range(num_expert):
            r = slice(m_base, m_base + ms_per_group[i])
            m_indices[r] = i
            ref_d[r] = torch.matmul(dequant_a[r], dequant_b[i].t())
            m_base += aligned_ms[i]

        mate.gemm.ragged_m_moe_gemm_8bit(
            (fp8_a, scale_a),
            (fp8_b, scale_b),
            m_indices,
            d,
            scale_granularity_mnk=scale_granularity_mnk,
            alignment_m=alignment_m,
        )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)


def get_masked_moe_gemm_8bit_cases():
    return [
        [1024 for _ in range(6)],
        [192 for _ in range(32)],
        [50 for _ in range(32)],
        [256, 256, 256, 256],
        [333, 444, 555, 666],
    ]


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group", get_masked_moe_gemm_8bit_cases())
@pytest.mark.parametrize("n", [128, 4096, 7168])
@pytest.mark.parametrize("k", [512, 4096, 7168])
@pytest.mark.parametrize("expected_m", [None, 128, 256])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("enable_overlap", [True])
@pytest.mark.parametrize("use_graph", [False, True])
def test_masked_moe_gemm_8bit(
    ms_per_group,
    n,
    k,
    expected_m,
    a_fp8_type,
    b_fp8_type,
    out_dtype,
    enable_overlap,
    use_graph,
):
    tile_signal = 64
    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)

    max_m = max(ms_per_group)
    expected_m = expected_m if expected_m is not None else max_m

    num_expert = len(ms_per_group)

    a = torch.rand((num_expert, max_m, k), device="musa", dtype=torch.float)
    b = torch.rand((num_expert, n, k), device="musa", dtype=torch.float)
    masked_m = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    d = torch.empty((num_expert, max_m, n), device="musa", dtype=out_dtype)
    signal = torch.zeros(
        num_expert * ceil_div(max_m, tile_signal), dtype=torch.int32, device=a.device
    )

    quant_tile_shape_a = (1, 1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (num_expert, max_m, k // quant_tile)
    scale_b_shape = (num_expert, n // quant_tile, k // quant_tile)
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
    )

    if use_graph:
        g = torch.musa.MUSAGraph()
        # capture
        with torch.musa.graph(g):
            res = mate.gemm.masked_moe_gemm_8bit(
                (fp8_a, scale_a),
                (fp8_b, scale_b),
                masked_m,
                d,
                scale_granularity_mnk,
                expected_m,
                enable_overlap=enable_overlap,
                signal=signal,
            )

        a.uniform_(0, 1)
        b.uniform_(0, 1)

        new_fp8_a, new_scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
        )
        new_fp8_b, new_scale_b = group_quantize_fp8(
            b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
        )

        fp8_a.copy_(new_fp8_a)
        fp8_b.copy_(new_fp8_b)
        scale_a.copy_(new_scale_a)
        scale_b.copy_(new_scale_b)

        dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
        ref_d = torch.einsum("bmk,bnk->bmn", dequant_a, dequant_b).to(torch.float)

        g.replay()
    else:
        dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
        ref_d = torch.einsum("bmk,bnk->bmn", dequant_a, dequant_b).to(torch.float)

        res = mate.gemm.masked_moe_gemm_8bit(
            (fp8_a, scale_a),
            (fp8_b, scale_b),
            masked_m,
            d,
            scale_granularity_mnk,
            expected_m,
            enable_overlap=enable_overlap,
            signal=signal,
        )

    d = d.to(torch.float)
    for i in range(num_expert):
        torch.testing.assert_close(
            d[i, : ms_per_group[i], :],
            ref_d[i, : ms_per_group[i], :],
            rtol=5e-3,
            atol=5e-3,
        )

    if enable_overlap:
        block_m = res[2]
        threshold = res[3]
        check_gemm_sbo_signal(num_expert, max_m, block_m, threshold, signal, masked_m)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group,n,k,alignment_m", _w4a8_ragged_cases())
@pytest.mark.parametrize(
    "mixed_dtype,a_fp8_type,out_dtype",
    [
        (mate.gemm.GemmMixedDType.S4FP8, torch.float8_e4m3fn, torch.bfloat16),
        (mate.gemm.GemmMixedDType.S4FP8, torch.float8_e4m3fn, torch.half),
        (mate.gemm.GemmMixedDType.S4FP8, torch.float8_e5m2, torch.bfloat16),
        (mate.gemm.GemmMixedDType.S4FP8, torch.float8_e5m2, torch.half),
        (mate.gemm.GemmMixedDType.FP4FP8, torch.float8_e4m3fn, torch.bfloat16),
    ],
)
def test_ragged_moe_gemm_mixed_dtype(
    ms_per_group,
    n,
    k,
    alignment_m,
    mixed_dtype,
    a_fp8_type,
    out_dtype,
):
    torch.manual_seed(0)
    if mixed_dtype == mate.gemm.GemmMixedDType.FP4FP8:
        alignment_m = 256
    num_expert = len(ms_per_group)
    aligned_ms = [align(m, alignment_m) for m in ms_per_group]
    m = sum(aligned_ms)
    device = torch.device("musa")

    a = torch.rand((m, k), device=device, dtype=torch.float)
    fp8_a, scale_a = _quantize_w4a8_a_per_channel(a, a_fp8_type)
    if mixed_dtype == mate.gemm.GemmMixedDType.FP4FP8:
        effective_b, packed_b, residual_e8m0, epilogue_fp32 = _make_fp4fp8_b_nk(
            num_expert, n, k, device
        )
        input_b = (packed_b, (residual_e8m0, epilogue_fp32))
        b_quant_recipe = (1, 32)
    else:
        b_int4, packed_b, scale_b = _make_w4a8_b_nk(num_expert, n, k, device)
        input_b = (packed_b, scale_b)
        b_quant_recipe = (1, 128)
    m_indices = torch.full((m,), -1, device=device, dtype=torch.int32)
    out = torch.empty((m, n), device=device, dtype=out_dtype)
    ref = torch.zeros((m, n), device=device, dtype=torch.float)
    dequant_a = _dequant_w4a8_a_per_channel(fp8_a, scale_a)

    m_base = 0
    for expert_id, expert_m in enumerate(ms_per_group):
        rows = slice(m_base, m_base + expert_m)
        m_indices[rows] = expert_id
        if mixed_dtype == mate.gemm.GemmMixedDType.FP4FP8:
            ref[rows] = (dequant_a[rows] @ effective_b[expert_id].t()) * epilogue_fp32[
                expert_id
            ]
        else:
            dequant_b = _dequant_w4a8_b_nk(b_int4, scale_b, expert_id, n, k)
            ref[rows] = dequant_a[rows] @ dequant_b.t()
        m_base += aligned_ms[expert_id]

    mate.gemm.ragged_moe_gemm_mixed_dtype(
        (fp8_a, scale_a),
        input_b,
        m_indices,
        out,
        alignment_m=alignment_m,
        mixed_dtype=mixed_dtype,
        backend="mubin",
        a_quant_recipe=(1, -1),
        b_quant_recipe=b_quant_recipe,
    )

    out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    "ms_per_group,n,k,expected_m",
    mixed_dtype_masked_moe_gemm_test_cases(),
)
@pytest.mark.parametrize(
    "mixed_dtype,a_fp8_type,out_dtype",
    [
        (mate.gemm.GemmMixedDType.S4FP8, torch.float8_e4m3fn, torch.bfloat16),
        (mate.gemm.GemmMixedDType.S4FP8, torch.float8_e4m3fn, torch.half),
        (mate.gemm.GemmMixedDType.S4FP8, torch.float8_e5m2, torch.bfloat16),
        (mate.gemm.GemmMixedDType.S4FP8, torch.float8_e5m2, torch.half),
        (mate.gemm.GemmMixedDType.FP4FP8, torch.float8_e4m3fn, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("enable_overlap", [False, True])
@pytest.mark.parametrize("backend", ["auto", "mubin", "mutlass"])
@pytest.mark.parametrize(
    "a_quant_recipe,scale_a_major",
    [((1, -1), "K"), ((1, 128), "K"), ((1, 128), "M")],
    ids=["Apertoken", "Apergroup-Kmajor", "Apergroup-Mmajor"],
)
def test_masked_moe_gemm_mixed_dtype(
    ms_per_group,
    n,
    k,
    expected_m,
    mixed_dtype,
    a_fp8_type,
    out_dtype,
    enable_overlap,
    backend,
    a_quant_recipe,
    scale_a_major,
):
    if mixed_dtype == mate.gemm.GemmMixedDType.FP4FP8 and a_quant_recipe != (1, -1):
        pytest.skip("FP4FP8 does not support grouped A quantization")
    if mixed_dtype == mate.gemm.GemmMixedDType.FP4FP8 and backend == "mutlass":
        pytest.skip("MUTLASS does not support FP4FP8")
    if a_quant_recipe == (1, 128) and backend == "mubin":
        pytest.skip("MUBIN does not support grouped A quantization")
    if (backend == "mutlass" or a_quant_recipe == (1, 128)) and (
        a_fp8_type != torch.float8_e4m3fn or enable_overlap
    ):
        pytest.skip("MUTLASS does not support this dtype or overlap configuration")

    torch.manual_seed(1)
    num_expert = len(ms_per_group)
    max_m = max(ms_per_group)
    device = torch.device("musa")

    a = torch.rand((num_expert, max_m, k), device=device, dtype=torch.float)
    if a_quant_recipe == (1, -1):
        fp8_a, scale_a = _quantize_w4a8_a_per_channel(a, a_fp8_type)
        dequant_a = _dequant_w4a8_a_per_channel(fp8_a, scale_a)
        if backend == "mutlass":
            scale_a_storage = torch.empty(
                (*scale_a.shape[:-1], 2), device=device, dtype=scale_a.dtype
            )
            scale_a_storage[..., ::2].copy_(scale_a)
            scale_a = scale_a_storage[..., ::2]
            assert scale_a.stride(-1) == 2
    else:
        fp8_a, scale_a = _quantize_w4a8_a_grouped(a, a_fp8_type)
        dequant_a = _dequant_w4a8_a_grouped(fp8_a, scale_a)
        if scale_a_major == "M":
            scale_a = scale_a.transpose(1, 2).contiguous().transpose(1, 2)
            assert scale_a.stride(1) == 1
            assert scale_a.stride(2) > 1

    if mixed_dtype == mate.gemm.GemmMixedDType.FP4FP8:
        effective_b, packed_b, residual_e8m0, epilogue_fp32 = _make_fp4fp8_b_nk(
            num_expert, n, k, device
        )
        input_b = (packed_b, (residual_e8m0, epilogue_fp32))
        b_quant_recipe = (1, 32)
    else:
        b_int4, packed_b, scale_b = _make_w4a8_b_nk(num_expert, n, k, device)
        input_b = (packed_b, scale_b)
        b_quant_recipe = (1, 128)
    masked_m = torch.tensor(ms_per_group, device=device, dtype=torch.int32)
    out = torch.empty((num_expert, max_m, n), device=device, dtype=out_dtype)
    signal = None

    if enable_overlap:
        tile_signal = 64
        signal = torch.zeros(
            num_expert * ceil_div(max_m, tile_signal),
            dtype=torch.int32,
            device=device,
        )

    ref = torch.zeros((num_expert, max_m, n), device=device, dtype=torch.float)
    for expert_id, expert_m in enumerate(ms_per_group):
        if mixed_dtype == mate.gemm.GemmMixedDType.FP4FP8:
            ref[expert_id, :expert_m] = (
                dequant_a[expert_id, :expert_m] @ effective_b[expert_id].t()
            ) * epilogue_fp32[expert_id]
        else:
            dequant_b = _dequant_w4a8_b_nk(b_int4, scale_b, expert_id, n, k)
            ref[expert_id, :expert_m] = dequant_a[expert_id, :expert_m] @ dequant_b.t()

    res = mate.gemm.masked_moe_gemm_mixed_dtype(
        (fp8_a, scale_a),
        input_b,
        masked_m,
        out,
        expect_tokens=expected_m,
        enable_overlap=enable_overlap,
        signal=signal,
        mixed_dtype=mixed_dtype,
        backend=backend,
        a_quant_recipe=a_quant_recipe,
        b_quant_recipe=b_quant_recipe,
    )

    for expert_id, expert_m in enumerate(ms_per_group):
        torch.testing.assert_close(
            out[expert_id, :expert_m].float(),
            ref[expert_id, :expert_m],
            rtol=5e-2,
            atol=5e-2,
        )

    if enable_overlap:
        block_m = res[2]
        threshold = res[3]
        check_gemm_sbo_signal(num_expert, max_m, block_m, threshold, signal, masked_m)


def k_grouped_contig_cases():
    return [
        [32, 32, 32, 32],
        [128, 128],
        [0, 256, 0, 768, 512, 0, 512, 512, 1024, 384],
        [128],
        [256],
        [111],
        [111, 222, 333, 444, 555, 666],
        [1024, 0, 512, 333, 666, 444],
        [0, 0, 0, 0, 111, 0, 0, 0, 0],
    ]


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ks_per_group", k_grouped_contig_cases())
@pytest.mark.parametrize("m", [16, 2048, 4096, 7168])
@pytest.mark.parametrize("n", [32, 2048, 4096, 7168])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.float])
def test_k_grouped_contig_gemm_8bit(
    ks_per_group,
    m,
    n,
    a_fp8_type,
    b_fp8_type,
    out_dtype,
):
    quant_tile = 128

    num_expert = len(ks_per_group)
    a_ = []
    b_ = []
    scale_a_ = []
    scale_b_ = []
    d = torch.rand((num_expert, m, n), device="musa", dtype=out_dtype)
    d_ref = d.clone()

    for i in range(num_expert):
        if ks_per_group[i] == 0:
            continue
        a = torch.rand(
            (m, ceil_div(ks_per_group[i], quant_tile) * quant_tile),
            device="musa",
            dtype=torch.float,
        )
        b = torch.rand(
            (n, ceil_div(ks_per_group[i], quant_tile) * quant_tile),
            device="musa",
            dtype=torch.float,
        )
        quant_tile_shape_a = (1, quant_tile)
        quant_tile_shape_b = (1, quant_tile)
        scale_a_shape = (m, ceil_div(ks_per_group[i], quant_tile))
        scale_b_shape = (n, ceil_div(ks_per_group[i], quant_tile))
        fp8_a, scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
        )
        fp8_b, scale_b = group_quantize_fp8(
            b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
        )
        dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
        d_ref[i] += dequant_a[:, : ks_per_group[i]] @ dequant_b[:, : ks_per_group[i]].T
        a_.append(fp8_a[:, : ks_per_group[i]].transpose(1, 0).contiguous())
        b_.append(fp8_b[:, : ks_per_group[i]].transpose(1, 0).contiguous())
        scale_a_.append(scale_a.transpose(1, 0).contiguous())
        scale_b_.append(scale_b.transpose(1, 0).contiguous())

    group_k_idx = torch.tensor(ks_per_group, device="musa", dtype=torch.int32)

    fp8_a = torch.cat(a_, dim=0).contiguous()
    scale_a = torch.cat(scale_a_, dim=0).contiguous()
    fp8_b = torch.cat(b_, dim=0).contiguous()
    scale_b = torch.cat(scale_b_, dim=0).contiguous()

    mate.gemm.ragged_k_moe_gemm_8bit(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        group_k_idx,
        d,
    )

    d = d.to(torch.float)
    for i in range(num_expert):
        torch.testing.assert_close(
            d[i],
            d_ref[i],
            rtol=5e-3,
            atol=5e-3,
        )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ks_per_group", k_grouped_contig_cases())
@pytest.mark.parametrize("m", [2048, 4096, 7168])
@pytest.mark.parametrize("n", [2048, 4096, 7168])
@pytest.mark.parametrize(
    "a_type,b_type,out_dtype",
    [
        (torch.bfloat16, torch.bfloat16, torch.float),
        (torch.half, torch.half, torch.float),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ],
)
def test_k_grouped_contig_gemm_16bit(
    ks_per_group,
    m,
    n,
    a_type,
    b_type,
    out_dtype,
):
    k = sum(ks_per_group)
    num_expert = len(ks_per_group)

    a_fp32 = torch.rand((k, m), device="musa", dtype=torch.float)
    b_fp32 = torch.rand((k, n), device="musa", dtype=torch.float)
    group_k_idx = torch.tensor(ks_per_group, device="musa", dtype=torch.int32)

    d = torch.zeros((num_expert, m, n), device="musa", dtype=out_dtype)
    d_ref = d.clone()

    a_bf16 = a_fp32.clone().to(a_type)
    b_bf16 = b_fp32.clone().to(b_type)

    start_k = 0
    for i in range(num_expert):
        nr_k = ks_per_group[i]
        a_i = a_fp32[start_k : start_k + nr_k,]
        b_i = b_fp32[start_k : start_k + nr_k,]
        start_k += nr_k
        d_ref[i] += a_i.T @ b_i

    mate.gemm.ragged_k_moe_gemm_16bit(
        a_bf16,
        b_bf16,
        group_k_idx,
        d,
    )
    d = d.to(torch.float)
    d_ref = d_ref.to(torch.float)
    tolerance = 2e-2 if out_dtype == torch.bfloat16 else 5e-3
    for i in range(num_expert):
        torch.testing.assert_close(
            d[i],
            d_ref[i],
            rtol=tolerance,
            atol=tolerance,
        )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group", get_ragged_moe_gemm_8bit_cases())
@pytest.mark.parametrize("n", [4096, 6144, 7168])
@pytest.mark.parametrize("k", [2048, 3072, 7168])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
def test_m_contig_gemm_8bit(
    ms_per_group,
    n,
    k,
    a_fp8_type,
    b_fp8_type,
    out_dtype,
):
    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)
    num_expert = len(ms_per_group)
    m = sum(ms_per_group)

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((num_expert, n, k), device="musa", dtype=torch.float)
    m_indices = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    d = torch.empty((m, n), device="musa", dtype=out_dtype)
    ref_d = torch.zeros((m, n), device="musa", dtype=torch.float)

    quant_tile_shape_a = (1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (m, ceil_div(k, quant_tile))
    scale_b_shape = (num_expert, ceil_div(n, quant_tile), ceil_div(k, quant_tile))
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
    )

    dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
    dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
    # calc ref
    m_base = 0
    for i in range(num_expert):
        r = slice(m_base, m_base + ms_per_group[i])
        ref_d[r] = torch.matmul(dequant_a[r], dequant_b[i].t())
        m_base += ms_per_group[i]
    mate.gemm.ragged_m_moe_gemm_8bit(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        m_indices,
        d,
        gemm_mode="per_expert",
        major_a_mode="K",
        major_b_mode="K",
        scale_granularity_mnk=scale_granularity_mnk,
    )

    torch.testing.assert_close(
        d.to(torch.float), ref_d.to(torch.float), rtol=5e-3, atol=5e-3
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group", get_ragged_moe_gemm_16bit_cases())
@pytest.mark.parametrize("n", [4096, 6144, 7168])
@pytest.mark.parametrize("k", [2048, 3072, 7168])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.half])
def test_m_contig_gemm_16bit(
    ms_per_group,
    n,
    k,
    data_type,
):
    num_expert = len(ms_per_group)
    m = sum(ms_per_group)

    a = torch.rand((m, k), device="musa", dtype=data_type)
    b = torch.rand((num_expert, k, n), device="musa", dtype=data_type)
    d = torch.empty((m, n), device="musa", dtype=data_type)

    m_indices = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    ref_d = torch.zeros((m, n), device="musa", dtype=torch.float)

    m_base = 0
    for i in range(num_expert):
        r = slice(m_base, m_base + ms_per_group[i])
        if ms_per_group[i] > 0:
            ref_d[r] = torch.matmul(a[r].float(), b[i].float())
        m_base += ms_per_group[i]

    mate.gemm.ragged_m_moe_gemm_16bit(
        a,
        b,
        m_indices,
        d,
        gemm_mode="per_expert",
        major_a_mode="K",
        major_b_mode="N",
    )

    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
def test_m_contig_gemm_8bit_zero_k_fills_output():
    num_expert = 2
    m = 8
    n = 256
    scale_granularity_mnk = (1, 128, 128)

    fp8_a = torch.empty((m, 0), device="musa", dtype=torch.float8_e4m3fn)
    scale_a = torch.empty((m, 0), device="musa", dtype=torch.float)
    fp8_b = torch.empty((num_expert, n, 0), device="musa", dtype=torch.float8_e4m3fn)
    scale_b = torch.empty((num_expert, n // 128, 0), device="musa", dtype=torch.float)
    m_indices = torch.tensor([4, 4], device="musa", dtype=torch.int32)
    out = torch.ones((m, n), device="musa", dtype=torch.bfloat16)

    mate.gemm.ragged_m_moe_gemm_8bit(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        m_indices,
        out,
        gemm_mode="per_expert",
        major_a_mode="K",
        major_b_mode="K",
        scale_granularity_mnk=scale_granularity_mnk,
    )

    torch.testing.assert_close(out, torch.zeros_like(out))


@supported_musa_compute_capability([31])
def test_m_contig_gemm_16bit_zero_k_fills_output():
    num_expert = 2
    m = 8
    n = 256

    a = torch.empty((m, 0), device="musa", dtype=torch.bfloat16)
    b = torch.empty((num_expert, 0, n), device="musa", dtype=torch.bfloat16)
    m_indices = torch.tensor([4, 4], device="musa", dtype=torch.int32)
    out = torch.ones((m, n), device="musa", dtype=torch.bfloat16)

    mate.gemm.ragged_m_moe_gemm_16bit(
        a,
        b,
        m_indices,
        out,
        gemm_mode="per_expert",
        major_a_mode="K",
        major_b_mode="N",
    )

    torch.testing.assert_close(out, torch.zeros_like(out))


def _make_groupwise_bmm_inputs(
    batch,
    m,
    n,
    k,
    recipe,
    a_fp8_type=torch.float8_e4m3fn,
    b_fp8_type=torch.float8_e4m3fn,
    scale_major="K",
):
    a = torch.rand((batch, m, k), device="musa", dtype=torch.float)
    b = torch.rand((batch, n, k), device="musa", dtype=torch.float)
    _, scale_granularity_n, scale_granularity_k = recipe

    scale_a_shape = (batch, m, ceil_div(k, scale_granularity_k))
    scale_b_shape = (
        batch,
        ceil_div(n, scale_granularity_n),
        ceil_div(k, scale_granularity_k),
    )
    if scale_major == "MN":
        scale_a_shape = (scale_a_shape[0], scale_a_shape[2], scale_a_shape[1])
        scale_b_shape = (scale_b_shape[0], scale_b_shape[2], scale_b_shape[1])

    fp8_a, scale_a = group_quantize_fp8(
        a,
        scale_a_shape,
        (1, 1, scale_granularity_k),
        a_fp8_type,
        scale_major,
    )
    fp8_b, scale_b = group_quantize_fp8(
        b,
        scale_b_shape,
        (1, scale_granularity_n, scale_granularity_k),
        b_fp8_type,
        scale_major,
    )
    return fp8_a, scale_a, fp8_b, scale_b


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch", [8])
@pytest.mark.parametrize("m", [128, 2048])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 2048])
@pytest.mark.parametrize(
    "a_fp8_type,b_fp8_type",
    [
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e5m2),
    ],
)
@pytest.mark.parametrize("recipe", [(1, 128, 128), (1, 1, 128)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("backend", ["auto"])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize(
    "trans_a,trans_b",
    [(False, True), (False, False), (True, False), (True, True)],
)
def test_bmm_fp8_groupwise_recipes(
    batch,
    m,
    n,
    k,
    a_fp8_type,
    b_fp8_type,
    recipe,
    out_dtype,
    backend,
    use_graph,
    trans_a,
    trans_b,
):
    scale_major = "MN" if trans_a or not trans_b else "K"
    fp8_a, scale_a, fp8_b, scale_b = _make_groupwise_bmm_inputs(
        batch, m, n, k, recipe, a_fp8_type, b_fp8_type, scale_major
    )

    a_arg = fp8_a.transpose(-2, -1).contiguous() if trans_a else fp8_a
    b_arg = fp8_b if trans_b else fp8_b.transpose(-2, -1).contiguous()

    d = torch.empty((batch, m, n), device="musa", dtype=out_dtype)
    if use_graph:
        g = torch.musa.MUSAGraph()
        with torch.musa.graph(g):
            mate.gemm.bmm(
                a_arg,
                b_arg,
                d,
                trans_a=trans_a,
                trans_b=trans_b,
                scale_a=scale_a,
                scale_b=scale_b,
                recipe_a=(recipe[0], recipe[2]),
                recipe_b=(recipe[1], recipe[2]),
                backend=backend,
            )

        new_fp8_a, new_scale_a, new_fp8_b, new_scale_b = _make_groupwise_bmm_inputs(
            batch, m, n, k, recipe, a_fp8_type, b_fp8_type, scale_major
        )
        new_a_arg = new_fp8_a.transpose(-2, -1).contiguous() if trans_a else new_fp8_a
        new_b_arg = new_fp8_b if trans_b else new_fp8_b.transpose(-2, -1).contiguous()
        a_arg.copy_(new_a_arg)
        b_arg.copy_(new_b_arg)
        scale_a.copy_(new_scale_a)
        scale_b.copy_(new_scale_b)
        fp8_a, scale_a = new_fp8_a, new_scale_a
        fp8_b, scale_b = new_fp8_b, new_scale_b
        g.replay()
    else:
        mate.gemm.bmm(
            a_arg,
            b_arg,
            d,
            trans_a=trans_a,
            trans_b=trans_b,
            scale_a=scale_a,
            scale_b=scale_b,
            recipe_a=(recipe[0], recipe[2]),
            recipe_b=(recipe[1], recipe[2]),
            backend=backend,
        )

    ref_d = torch.bmm(
        group_dequantize_fp8(fp8_a, scale_a, scale_major),
        group_dequantize_fp8(fp8_b, scale_b, scale_major).transpose(-2, -1),
    )

    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    "trans_a,trans_b",
    [(False, True), (False, False), (True, False), (True, True)],
)
def test_bmm_fp8_mubin_output(trans_a, trans_b):
    batch, m, n, k = 2, 128, 256, 384
    recipe = (1, 128, 128)
    fp8_a, scale_a, fp8_b, scale_b = _make_groupwise_bmm_inputs(batch, m, n, k, recipe)

    a_arg = fp8_a.transpose(-2, -1).contiguous() if trans_a else fp8_a
    b_arg = fp8_b if trans_b else fp8_b.transpose(-2, -1).contiguous()
    scale_a_arg = scale_a.transpose(-2, -1).contiguous() if trans_a else scale_a
    scale_b_arg = scale_b if trans_b else scale_b.transpose(-2, -1).contiguous()
    scale_out = torch.empty(
        (batch, m, ceil_div(n, 128)), device="musa", dtype=torch.float32
    )

    ref_d = torch.bmm(
        group_dequantize_fp8(fp8_a, scale_a, "K"),
        group_dequantize_fp8(fp8_b, scale_b, "K").transpose(-2, -1),
    )
    result = mate.gemm.bmm(
        a_arg,
        b_arg,
        trans_a=trans_a,
        trans_b=trans_b,
        scale_a=scale_a_arg,
        scale_b=scale_b_arg,
        scale_out=scale_out,
        recipe_a=(recipe[0], recipe[2]),
        recipe_b=(recipe[1], recipe[2]),
        backend="auto",
    )

    assert result.dtype == torch.float8_e4m3fn
    dequant_result = (
        result.float().reshape(batch, m, n // 128, 128) * scale_out.unsqueeze(-1)
    ).reshape(batch, m, n)
    similarity = torch.cosine_similarity(
        dequant_result.flatten(), ref_d.flatten(), dim=0
    )
    assert similarity > 0.999


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch", [8])
@pytest.mark.parametrize("m", [128, 2048])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 2048])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("scale_granularity_mnk", [(1, -1, -1), (-1, -1, -1)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("backend", ["auto", "mudnn"])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize(
    "trans_a,trans_b",
    [(False, True), (False, False), (True, False), (True, True)],
)
def test_bmm_fp8_tensorwise_channelwise(
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
    trans_a,
    trans_b,
):
    a = torch.rand((batch, m, k), device="musa", dtype=torch.float)
    b = torch.rand((batch, n, k), device="musa", dtype=torch.float)
    d = torch.empty((batch, m, n), device="musa", dtype=out_dtype)

    scale_granularity_m, _, scale_granularity_k = scale_granularity_mnk
    scale_granularity_m = m if scale_granularity_m == -1 else scale_granularity_m
    scale_granularity_k = k if scale_granularity_k == -1 else scale_granularity_k
    quant_tile_shape_a = (1, scale_granularity_m, scale_granularity_k)
    scale_a_shape = (batch, m // scale_granularity_m, k // scale_granularity_k)
    scale_major = "MN" if trans_a or not trans_b else "K"
    if scale_major == "MN":
        scale_a_shape = (scale_a_shape[0], scale_a_shape[2], scale_a_shape[1])
    recipe_a = (scale_granularity_mnk[0], scale_granularity_mnk[2])
    recipe_b = (scale_granularity_mnk[1], scale_granularity_mnk[2])

    if scale_granularity_mnk[0] == -1 and scale_granularity_mnk[1] == -1:
        fp8_a, scale_a = tensor_quantize_fp8(a, a_fp8_type)
    else:
        fp8_a, scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, scale_major
        )
    fp8_b, scale_b = tensor_quantize_fp8(b, b_fp8_type)

    a_arg = fp8_a.transpose(-2, -1).contiguous() if trans_a else fp8_a
    b_arg = fp8_b if trans_b else fp8_b.transpose(-2, -1).contiguous()

    if use_graph:
        g = torch.musa.MUSAGraph()
        with torch.musa.graph(g):
            mate.gemm.bmm(
                a_arg,
                b_arg,
                d,
                trans_a=trans_a,
                trans_b=trans_b,
                scale_a=scale_a,
                scale_b=scale_b,
                recipe_a=recipe_a,
                recipe_b=recipe_b,
                backend=backend,
            )

        a.uniform_(0, 1)
        b.uniform_(0, 1)
        if scale_granularity_mnk[0] == -1 and scale_granularity_mnk[1] == -1:
            new_fp8_a, new_scale_a = tensor_quantize_fp8(a, a_fp8_type)
        else:
            new_fp8_a, new_scale_a = group_quantize_fp8(
                a, scale_a_shape, quant_tile_shape_a, a_fp8_type, scale_major
            )
        new_fp8_b, new_scale_b = tensor_quantize_fp8(b, b_fp8_type)
        new_a_arg = new_fp8_a.transpose(-2, -1).contiguous() if trans_a else new_fp8_a
        new_b_arg = new_fp8_b if trans_b else new_fp8_b.transpose(-2, -1).contiguous()
        a_arg.copy_(new_a_arg)
        b_arg.copy_(new_b_arg)
        scale_a.copy_(new_scale_a)
        scale_b.copy_(new_scale_b)
        ref_a = group_dequantize_fp8(new_fp8_a, new_scale_a, scale_major)
        ref_b = group_dequantize_fp8(new_fp8_b, new_scale_b, scale_major)
        ref_d = torch.bmm(ref_a, ref_b.transpose(-2, -1))
        g.replay()
    else:
        ref_a = group_dequantize_fp8(fp8_a, scale_a, scale_major)
        ref_b = group_dequantize_fp8(fp8_b, scale_b, scale_major)
        ref_d = torch.bmm(ref_a, ref_b.transpose(-2, -1))
        mate.gemm.bmm(
            a_arg,
            b_arg,
            d,
            trans_a=trans_a,
            trans_b=trans_b,
            scale_a=scale_a,
            scale_b=scale_b,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            backend=backend,
        )

    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
def test_bmm_fp8_groupwise_fp32_accumulate_with_c():
    batch, m, n, k = 2, 128, 128, 128
    recipe = (1, 1, 128)
    fp8_a, scale_a, fp8_b, scale_b = _make_groupwise_bmm_inputs(batch, m, n, k, recipe)
    c = torch.rand((batch, m, n), device="musa", dtype=torch.float32)
    d = torch.empty_like(c)

    ref_d = c.float() + torch.bmm(
        group_dequantize_fp8(fp8_a, scale_a, "K"),
        group_dequantize_fp8(fp8_b, scale_b, "K").transpose(-2, -1),
    )

    mate.gemm.bmm(
        fp8_a,
        fp8_b,
        d,
        scale_a=scale_a,
        scale_b=scale_b,
        recipe_a=(recipe[0], recipe[2]),
        recipe_b=(recipe[1], recipe[2]),
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
@pytest.mark.parametrize("backend", ["auto", "mudnn"])
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

    mate.gemm.bmm(
        fp8_a,
        fp8_b,
        d,
        scale_a=scale_a,
        scale_b=scale_b,
        recipe_a=(scale_granularity_mnk[0], scale_granularity_mnk[2]),
        recipe_b=(scale_granularity_mnk[1], scale_granularity_mnk[2]),
        backend=backend,
    )

    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("m", [128, 2048])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 2048])
@pytest.mark.parametrize(
    "input_dtype,out_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float16),
        (torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize(
    "trans_a,trans_b",
    [
        (False, True),
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_bmm_bf16_fp16(batch, m, n, k, input_dtype, out_dtype, trans_a, trans_b):
    a = torch.rand((batch, m, k), device="musa", dtype=input_dtype)
    b = torch.rand((batch, n, k), device="musa", dtype=input_dtype)
    ref_d = torch.bmm(a.float(), b.float().transpose(-2, -1))

    a_arg = a.transpose(-2, -1).contiguous() if trans_a else a
    b_arg = b if trans_b else b.transpose(-2, -1).contiguous()
    out = mate.gemm.bmm(
        a_arg,
        b_arg,
        trans_a=trans_a,
        trans_b=trans_b,
        out_dtype=out_dtype,
        backend="mudnn",
    )
    assert out.dtype == out_dtype
    torch.testing.assert_close(out.float(), ref_d, rtol=5e-3, atol=5e-3)
