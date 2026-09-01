# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for MATE's dense low-level SageAttention API.
"""

import pytest
import torch
import torch.nn.functional as F

import mate
from mate.testing import quantize_sage_attention_tensor
from mate.testing import supported_musa_compute_capability
from mate.testing.flash_attn import attention_ref

_SUPPORTED_RECIPES = [
    (-1, -1, -1, -1),
    (128, 128, -1, 1),
    (128, 16, -1, 1),
    (128, 128, -1, 128),
]
_QK_QUANT_DTYPES = ["int8", "fp8"]
ShapeSpec = tuple[int, int, int, int, int, int, int]
_CONBINATION_SHAPE: ShapeSpec = (1, 1024, 1024, 2, 2, 128, 128)
_REFERENCE_SHAPE: ShapeSpec = (2, 25440, 440, 32, 32, 128, 128)
_SMOKE_SHAPES: list[ShapeSpec] = [
    (1, 64, 256, 1, 1, 128, 128),
    (1, 128, 1024, 2, 2, 128, 128),
    (2, 80, 272, 6, 2, 96, 96),
]
_REFERENCE_COMBINATIONS: list[tuple[str, tuple[int, int, int, int]]] = [
    (qk_quant_dtype, quant_recipe)
    for qk_quant_dtype in _QK_QUANT_DTYPES
    for quant_recipe in _SUPPORTED_RECIPES
]
_COMPILE_PARITY_CASES: list[
    tuple[str, tuple[int, int, int, int], torch.dtype, bool]
] = [
    ("q", (-1, -1, -1, -1), torch.int8, False),
    ("q", (128, 128, -1, 128), torch.int8, False),
    ("k", (-1, -1, -1, -1), torch.int8, True),
    ("k", (128, 16, -1, 1), torch.float8_e4m3fn, True),
    ("v", (-1, -1, -1, -1), torch.float8_e4m3fn, False),
    ("v", (128, 128, -1, 1), torch.float8_e4m3fn, False),
    ("v", (128, 128, -1, 128), torch.float8_e4m3fn, False),
]


def _manual_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "musa"):
        torch.musa.manual_seed(seed)


def _shape_id(shape: ShapeSpec) -> str:
    return (
        f"b{shape[0]}-sq{shape[1]}-skv{shape[2]}"
        f"-hq{shape[3]}-hkv{shape[4]}-dqk{shape[5]}-dv{shape[6]}"
    )


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.reshape(-1).to(torch.float32)
    y_flat = y.reshape(-1).to(torch.float32)
    denom = max(torch.linalg.norm(x_flat), 1e-8) * max(torch.linalg.norm(y_flat), 1e-8)
    return torch.dot(x_flat, y_flat).div(denom).item()


def relative_l1_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.reshape(-1).to(torch.float32)
    y_flat = y.reshape(-1).to(torch.float32)
    return (
        (x_flat - y_flat).abs().sum() / (x_flat.abs().sum() + y_flat.abs().sum() + 1e-8)
    ).item()


def rmse(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.reshape(-1).to(torch.float32)
    y_flat = y.reshape(-1).to(torch.float32)
    return torch.sqrt(torch.mean((x_flat - y_flat) ** 2)).item()


def _fp8_quantize_per_channel(
    x: torch.Tensor, *, reduceheaddim: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(torch.float8_e4m3fn)
    reduce_dim = -1 if reduceheaddim else -2
    amax = x.abs().amax(dim=reduce_dim, keepdim=True)
    scale = finfo.max / amax.clamp(min=1e-12)
    q = (x * scale).clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)
    scale_inv = scale.float().reciprocal()
    return q, scale_inv, q.to(torch.float32) * scale_inv


def _resolve_qk_quant_dtype(qk_quant_dtype: str) -> torch.dtype:
    return torch.int8 if qk_quant_dtype == "int8" else torch.float8_e4m3fn


def _build_prequantized_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    quant_recipe: tuple[int, int, int, int],
    qk_quant_dtype: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    q_quant_dtype = _resolve_qk_quant_dtype(qk_quant_dtype)
    q_quant, q_scale, q_ref = quantize_sage_attention_tensor(
        q,
        operand="q",
        quant_recipe=quant_recipe,
        quant_dtype=q_quant_dtype,
        return_dequant=True,
        use_compile=False,
    )
    k_quant, k_scale, k_ref = quantize_sage_attention_tensor(
        k,
        operand="k",
        quant_recipe=quant_recipe,
        quant_dtype=q_quant_dtype,
        return_dequant=True,
        smooth_k=(qk_quant_dtype == "int8" and quant_recipe == (128, 16, -1, 1)),
        use_compile=False,
    )
    v_quant, v_scale, v_ref = quantize_sage_attention_tensor(
        v,
        operand="v",
        quant_recipe=quant_recipe,
        quant_dtype=torch.float8_e4m3fn,
        return_dequant=True,
        use_compile=False,
    )
    return q_quant, k_quant, v_quant, q_scale, k_scale, v_scale, q_ref, k_ref, v_ref


def _make_inputs(shape: ShapeSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        batch,
        seqlen_q,
        seqlen_kv,
        num_q_heads,
        num_kv_heads,
        headdim_qk,
        headdim_v,
    ) = shape
    q = torch.randn(
        batch,
        seqlen_q,
        num_q_heads,
        headdim_qk,
        dtype=torch.bfloat16,
        device="musa",
    )
    k = torch.randn(
        batch,
        seqlen_kv,
        num_kv_heads,
        headdim_qk,
        dtype=torch.bfloat16,
        device="musa",
    )
    v = torch.randn(
        batch,
        seqlen_kv,
        num_kv_heads,
        headdim_v,
        dtype=torch.bfloat16,
        device="musa",
    )
    return q, k, v


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    ("qk_quant_dtype", "quant_recipe"),
    _REFERENCE_COMBINATIONS,
    ids=[
        f"{qk_quant_dtype}-{quant_recipe}"
        for qk_quant_dtype, quant_recipe in _REFERENCE_COMBINATIONS
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_all_recipe_dtype_combinations_match_reference(
    is_causal: bool,
    qk_quant_dtype: str,
    quant_recipe: tuple[int, int, int, int],
):
    _manual_seed(321)

    q, k, v = _make_inputs(_CONBINATION_SHAPE)

    q_quant, k_quant, v_quant, q_scale, k_scale, v_scale, q_ref, k_ref, v_ref = (
        _build_prequantized_inputs(
            q,
            k,
            v,
            quant_recipe=quant_recipe,
            qk_quant_dtype=qk_quant_dtype,
        )
    )
    out_ref, _, scores = attention_ref(
        q_ref.cpu().to(torch.float32),
        k_ref.cpu().to(torch.float32),
        v_ref.cpu().to(torch.float32),
        causal=is_causal,
        upcast=True,
    )
    lse_ref = torch.logsumexp(scores, dim=-1).cpu().to(torch.float32)

    out, lse = mate.sage_attn_quantized(
        q=q_quant,
        k=k_quant,
        v=v_quant,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        causal=is_causal,
        quant_recipe=quant_recipe,
        return_lse=True,
    )

    out = out.cpu().to(torch.float32)
    lse = lse.cpu().to(torch.float32)
    out_ref = out_ref.cpu().to(torch.float32)

    assert F.cosine_similarity(out.flatten(), out_ref.flatten(), dim=0) > 0.998
    assert relative_l1_distance(out, out_ref) < 3e-2
    assert rmse(out, out_ref) < 1.5e-2
    assert F.cosine_similarity(lse.flatten(), lse_ref.flatten(), dim=0) > 0.999


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("shape", _SMOKE_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("qk_quant_dtype", _QK_QUANT_DTYPES)
@pytest.mark.parametrize("quant_recipe", _SUPPORTED_RECIPES)
def test_supported_recipe_dtype_combinations_smoke(
    shape: ShapeSpec,
    qk_quant_dtype: str,
    quant_recipe: tuple[int, int, int, int],
):
    _manual_seed(404)

    q, k, v = _make_inputs(shape)
    batch, seqlen_q, _, num_q_heads, _, _, headdim_v = shape

    q_quant, k_quant, v_quant, q_scale, k_scale, v_scale, *_ = (
        _build_prequantized_inputs(
            q,
            k,
            v,
            quant_recipe=quant_recipe,
            qk_quant_dtype=qk_quant_dtype,
        )
    )

    out, lse = mate.sage_attn_quantized(
        q=q_quant,
        k=k_quant,
        v=v_quant,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_recipe=quant_recipe,
        return_lse=True,
    )

    assert out.shape == (batch, seqlen_q, num_q_heads, headdim_v)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (batch, num_q_heads, seqlen_q)
    assert lse.dtype == torch.float32
    assert torch.isfinite(out).all().item()
    assert torch.isfinite(lse).all().item()


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("qk_quant_dtype", _QK_QUANT_DTYPES)
@pytest.mark.parametrize("quant_recipe", _SUPPORTED_RECIPES)
def test_fp8_output_matches_per_channel_golden(
    qk_quant_dtype: str,
    quant_recipe: tuple[int, int, int, int],
):
    _manual_seed(909)

    q, k, v = _make_inputs(_REFERENCE_SHAPE)

    q_quant, k_quant, v_quant, q_scale, k_scale, v_scale, q_ref, k_ref, v_ref = (
        _build_prequantized_inputs(
            q,
            k,
            v,
            quant_recipe=quant_recipe,
            qk_quant_dtype=qk_quant_dtype,
        )
    )
    out_ref, _, _ = attention_ref(
        q_ref.cpu().to(torch.float32),
        k_ref.cpu().to(torch.float32),
        v_ref.cpu().to(torch.float32),
        causal=False,
        upcast=True,
    )
    out_ref = out_ref.to("musa").to(torch.float32)
    out_ref_fp8, out_scale_ref, out_ref_dequant = _fp8_quantize_per_channel(
        out_ref, reduceheaddim=True
    )

    out_fp8, out_scale = mate.sage_attn_quantized(
        q=q_quant,
        k=k_quant,
        v=v_quant,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_recipe=quant_recipe,
        fp8_output=True,
    )

    assert out_fp8.shape == out_ref_fp8.shape
    assert out_fp8.dtype == torch.float8_e4m3fn
    assert out_scale.shape == out_scale_ref.shape
    assert out_scale.dtype == torch.float32
    torch.testing.assert_close(
        out_scale.cpu(), out_scale_ref.cpu(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        out_scale.cpu(),
        out_scale_ref.cpu(),
        atol=5e-2,
        rtol=5e-2,
    )
    assert (
        F.cosine_similarity(
            (out_fp8.to(torch.float32) * out_scale).flatten(),
            out_ref_dequant.flatten(),
            dim=0,
        )
        > 0.998
    )
    # torch.testing.assert_close(
    #    (out_fp8.to(torch.float32) * out_scale).cpu(),
    #    out_ref_dequant.cpu(),
    #    atol=5e-2,
    #    rtol=5e-2,
    # )


@supported_musa_compute_capability([31])
def test_fp8_output_with_lse_returns_expected_tuple():
    _manual_seed(910)

    q, k, v = _make_inputs(_REFERENCE_SHAPE)
    q_quant, k_quant, v_quant, q_scale, k_scale, v_scale, *_ = (
        _build_prequantized_inputs(
            q,
            k,
            v,
            quant_recipe=(128, 16, -1, 1),
            qk_quant_dtype="int8",
        )
    )

    out_fp8, out_scale, lse = mate.sage_attn_quantized(
        q=q_quant,
        k=k_quant,
        v=v_quant,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_recipe=(128, 16, -1, 1),
        return_lse=True,
        fp8_output=True,
    )

    assert out_fp8.dtype == torch.float8_e4m3fn
    assert out_scale.dtype == torch.float32
    assert lse.dtype == torch.float32
    assert out_scale.shape[-1] == 1


#@supported_musa_compute_capability([31])
#@pytest.mark.parametrize(
#    ("operand", "quant_recipe", "quant_dtype", "smooth_k"),
#    _COMPILE_PARITY_CASES,
#    ids=[
#        f"{operand}-{quant_recipe}-{quant_dtype}-{smooth_k}"
#        for operand, quant_recipe, quant_dtype, smooth_k in _COMPILE_PARITY_CASES
#    ],
#)
#def test_quantize_sage_attention_tensor_compile_matches_eager(
#    operand: str,
#    quant_recipe: tuple[int, int, int, int],
#    quant_dtype: torch.dtype,
#    smooth_k: bool,
#):
#    _manual_seed(505)
#
#    x = torch.randn(1, 128, 2, 128, dtype=torch.bfloat16, device="musa")
#
#    eager = quantize_sage_attention_tensor(
#        x,
#        operand=operand,
#        quant_recipe=quant_recipe,
#        quant_dtype=quant_dtype,
#        return_dequant=True,
#        smooth_k=smooth_k,
#        use_compile=False,
#    )
#    compiled = quantize_sage_attention_tensor(
#        x,
#        operand=operand,
#        quant_recipe=quant_recipe,
#        quant_dtype=quant_dtype,
#        return_dequant=True,
#        smooth_k=smooth_k,
#        use_compile=True,
#    )
#
#    eager_quant, eager_scale, eager_dequant = eager
#    compiled_quant, compiled_scale, compiled_dequant = compiled
#    assert compiled_quant.shape == eager_quant.shape
#    assert compiled_quant.dtype == eager_quant.dtype
#    assert compiled_scale.shape == eager_scale.shape
#    assert compiled_dequant.shape == eager_dequant.shape
#    assert torch.allclose(compiled_scale, eager_scale, atol=5e-3, rtol=5e-3)
#    assert (
#        F.cosine_similarity(
#            compiled_dequant.to(torch.float32).flatten(), eager_dequant.flatten(), dim=0
#        )
#        > 0.998
#    )
#    # assert torch.allclose(compiled_dequant, eager_dequant, atol=3.5e-1, rtol=1e-3)
