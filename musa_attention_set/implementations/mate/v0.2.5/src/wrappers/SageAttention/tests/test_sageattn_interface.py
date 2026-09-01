# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for the supported public SageAttention wrapper API.
"""

import math
import os
import sys

import pytest
import torch

# Add parent directory to path for importing sageattention
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda_sm90
from mate.testing import quantize_sage_attention_tensor
from mate.testing.flash_attn import attention_ref

_SUPPORTED_RECIPES = [
    (-1, -1, -1, -1),
    (128, 128, -1, 1),
    (128, 16, -1, 1),
    (128, 128, -1, 128),
]
_QK_QUANT_DTYPES = ["int8", "fp8"]
_REFERENCE_LAYOUTS = ["HND", "NHD"]
_REFERENCE_COMBINATIONS = [
    (qk_quant_dtype, quant_recipe)
    for qk_quant_dtype in _QK_QUANT_DTYPES
    for quant_recipe in _SUPPORTED_RECIPES
]


def _manual_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "musa"):
        torch.musa.manual_seed(seed)


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.reshape(-1).to(torch.float32)
    y_flat = y.reshape(-1).to(torch.float32)
    denom = torch.linalg.norm(x_flat) * torch.linalg.norm(y_flat) + 1e-8
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


def _to_bnhd(x: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    return x.transpose(1, 2) if tensor_layout == "HND" else x


def _from_bnhd(x: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    return x.transpose(1, 2) if tensor_layout == "HND" else x


def _resolve_qk_quant_dtype(qk_quant_dtype: str) -> torch.dtype:
    return torch.int8 if qk_quant_dtype == "int8" else torch.float8_e4m3fn


def _make_inputs(
    *,
    tensor_layout: str,
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    headdim: int,
    dtype: torch.dtype,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = device or "cpu"
    if tensor_layout == "HND":
        q_shape = (batch, heads, seqlen_q, headdim)
        kv_shape = (batch, heads, seqlen_kv, headdim)
    else:
        q_shape = (batch, seqlen_q, heads, headdim)
        kv_shape = (batch, seqlen_kv, heads, headdim)
    q = torch.randn(*q_shape, dtype=dtype, device=device)
    k = torch.randn(*kv_shape, dtype=dtype, device=device)
    v = torch.randn(*kv_shape, dtype=dtype, device=device)
    return q, k, v


def _build_quantized_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str,
    quant_recipe: tuple[int, int, int, int],
    qk_quant_dtype: str,
    smooth_k: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_bnhd = _to_bnhd(q, tensor_layout).to(torch.bfloat16)
    k_bnhd = _to_bnhd(k, tensor_layout).to(torch.bfloat16)
    v_bnhd = _to_bnhd(v, tensor_layout).to(torch.bfloat16)
    q_quant_dtype = _resolve_qk_quant_dtype(qk_quant_dtype)

    q_ref = quantize_sage_attention_tensor(
        q_bnhd,
        operand="q",
        quant_recipe=quant_recipe,
        quant_dtype=q_quant_dtype,
        return_dequant=True,
        use_compile=False,
    )[2]
    k_ref = quantize_sage_attention_tensor(
        k_bnhd,
        operand="k",
        quant_recipe=quant_recipe,
        quant_dtype=q_quant_dtype,
        return_dequant=True,
        smooth_k=smooth_k,
        use_compile=False,
    )[2]
    v_ref = quantize_sage_attention_tensor(
        v_bnhd,
        operand="v",
        quant_recipe=quant_recipe,
        quant_dtype=torch.float8_e4m3fn,
        return_dequant=True,
        use_compile=False,
    )[2]
    return q_ref, k_ref, v_ref


def _compute_lse_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    scores: torch.Tensor,
    *,
    tensor_layout: str,
    smooth_k: bool,
) -> torch.Tensor:
    lse_ref = torch.logsumexp(scores, dim=-1)
    if not smooth_k:
        return lse_ref

    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    seq_dim = 1 if tensor_layout == "NHD" else 2
    nh_dim = 2 if tensor_layout == "NHD" else 1

    k_mean = k.mean(dim=seq_dim, keepdim=True)
    q_per_kv_heads = q.size(nh_dim) // k.size(nh_dim)
    k_mean_broadcast = (
        torch.repeat_interleave(k_mean, q_per_kv_heads, dim=nh_dim)
        if q_per_kv_heads > 1
        else k_mean
    )

    if tensor_layout == "NHD":
        lse_correction = (
            torch.matmul(
                q.transpose(1, 2),
                k_mean_broadcast.transpose(1, 2).transpose(2, 3),
            )
            .squeeze(-1)
            .to(torch.float32)
        )
    else:
        lse_correction = (
            torch.matmul(q, k_mean_broadcast.transpose(2, 3))
            .squeeze(-1)
            .to(torch.float32)
        )
    return lse_ref + lse_correction * scale


@pytest.mark.parametrize(
    ("qk_quant_dtype", "quant_recipe"),
    _REFERENCE_COMBINATIONS,
    ids=[
        f"{qk_quant_dtype}-{quant_recipe}"
        for qk_quant_dtype, quant_recipe in _REFERENCE_COMBINATIONS
    ],
)
@pytest.mark.parametrize("tensor_layout", _REFERENCE_LAYOUTS)
@pytest.mark.parametrize("is_causal", [False, True])
def test_all_recipe_dtype_combinations_match_reference(
    is_causal: bool,
    tensor_layout: str,
    qk_quant_dtype: str,
    quant_recipe: tuple[int, int, int, int],
):
    _manual_seed(321)

    smooth_k = True
    batch, heads, seqlen_q, seqlen_kv, headdim = 1, 2, 128, 1024, 128

    q, k, v = _make_inputs(
        tensor_layout=tensor_layout,
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        headdim=headdim,
        dtype=torch.float32,
    )

    q_ref, k_ref, v_ref = _build_quantized_reference(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        quant_recipe=quant_recipe,
        qk_quant_dtype=qk_quant_dtype,
        smooth_k=smooth_k,
    )
    out_ref, _, scores = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=is_causal,
        upcast=True,
    )
    lse_ref = _compute_lse_reference(
        q,
        k,
        scores,
        tensor_layout=tensor_layout,
        smooth_k=smooth_k,
    )

    out, lse = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q.to(torch.bfloat16).to("musa"),
        k.to(torch.bfloat16).to("musa"),
        v.to(torch.bfloat16).to("musa"),
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_dtype=qk_quant_dtype,
        quant_recipe=quant_recipe,
        return_lse=True,
    )

    out = out.cpu().to(torch.float32)
    out_ref = _from_bnhd(out_ref.cpu().to(torch.float32), tensor_layout)
    lse = lse.cpu().to(torch.float32)
    lse_ref = lse_ref.cpu().to(torch.float32)

    assert cosine_similarity(out, out_ref) > 0.997
    assert relative_l1_distance(out, out_ref) < 4e-2
    assert rmse(out, out_ref) < 2e-2
    assert cosine_similarity(lse, lse_ref) > 0.999


@pytest.mark.parametrize("qk_quant_dtype", _QK_QUANT_DTYPES)
@pytest.mark.parametrize("quant_recipe", _SUPPORTED_RECIPES)
def test_supported_recipe_dtype_combinations_smoke(
    qk_quant_dtype: str,
    quant_recipe: tuple[int, int, int, int],
):
    _manual_seed(404)

    batch, heads, seqlen_q, seqlen_kv, headdim = 1, 1, 64, 256, 128
    q, k, v = _make_inputs(
        tensor_layout="HND",
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        headdim=headdim,
        dtype=torch.bfloat16,
        device="musa",
    )

    out, lse = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q,
        k,
        v,
        qk_quant_dtype=qk_quant_dtype,
        quant_recipe=quant_recipe,
        return_lse=True,
    )

    assert out.shape == (batch, heads, seqlen_q, headdim)
    assert lse.shape == (batch, heads, seqlen_q)
    assert torch.isfinite(out).all().item()
    assert torch.isfinite(lse).all().item()


def test_quant_recipe_overrides_qk_quant_gran():
    _manual_seed(654)

    q, k, v = _make_inputs(
        tensor_layout="HND",
        batch=1,
        heads=1,
        seqlen_q=64,
        seqlen_kv=256,
        headdim=128,
        dtype=torch.bfloat16,
        device="musa",
    )

    out_override, lse_override = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q,
        k,
        v,
        qk_quant_gran="per-warp",
        qk_quant_dtype="fp8",
        quant_recipe=(128, 128, -1, 1),
        return_lse=True,
    )
    out_explicit, lse_explicit = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q,
        k,
        v,
        qk_quant_dtype="fp8",
        quant_recipe=(128, 128, -1, 1),
        return_lse=True,
    )

    assert torch.equal(out_override, out_explicit)
    assert torch.equal(lse_override, lse_explicit)


def test_smooth_k_toggle_changes_lse():
    _manual_seed(777)

    q, k, v = _make_inputs(
        tensor_layout="HND",
        batch=1,
        heads=1,
        seqlen_q=64,
        seqlen_kv=256,
        headdim=128,
        dtype=torch.bfloat16,
        device="musa",
    )

    _, lse_smooth = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q,
        k,
        v,
        return_lse=True,
        smooth_k=True,
    )
    _, lse_plain = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q,
        k,
        v,
        return_lse=True,
        smooth_k=False,
    )

    assert not torch.allclose(lse_smooth, lse_plain)


def test_sageattn_fp8_output_matches_reference():
    _manual_seed(1001)

    tensor_layout = "HND"
    qk_quant_dtype = "int8"
    quant_recipe = (128, 16, -1, 1)
    q, k, v = _make_inputs(
        tensor_layout=tensor_layout,
        batch=1,
        heads=2,
        seqlen_q=128,
        seqlen_kv=1024,
        headdim=128,
        dtype=torch.float32,
    )

    q_ref, k_ref, v_ref = _build_quantized_reference(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        quant_recipe=quant_recipe,
        qk_quant_dtype=qk_quant_dtype,
        smooth_k=True,
    )
    out_ref, _, _ = attention_ref(q_ref, k_ref, v_ref, causal=False, upcast=True)
    out_ref = _from_bnhd(out_ref, tensor_layout).to(torch.float32)
    out_ref_fp8, out_scale_ref, out_ref_dequant = _fp8_quantize_per_channel(
        out_ref, reduceheaddim=True
    )

    out_fp8, out_scale = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q.to(torch.bfloat16).to("musa"),
        k.to(torch.bfloat16).to("musa"),
        v.to(torch.bfloat16).to("musa"),
        tensor_layout=tensor_layout,
        qk_quant_dtype=qk_quant_dtype,
        quant_recipe=quant_recipe,
        fp8_output=True,
    )

    assert out_fp8.dtype == torch.float8_e4m3fn
    assert out_scale.dtype == torch.float32
    torch.testing.assert_close(
        out_scale.cpu(), out_scale_ref.cpu(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        (out_fp8.to(torch.float32) * out_scale).cpu(),
        out_ref_dequant.cpu(),
        atol=5e-2,
        rtol=5e-2,
    )


def test_sageattn_fp8_output_with_lse_tuple_shape():
    _manual_seed(1002)

    q, k, v = _make_inputs(
        tensor_layout="HND",
        batch=1,
        heads=1,
        seqlen_q=64,
        seqlen_kv=256,
        headdim=128,
        dtype=torch.bfloat16,
        device="musa",
    )

    out_fp8, out_scale, lse = sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        fp8_output=True,
        return_lse=True,
    )

    assert out_fp8.dtype == torch.float8_e4m3fn
    assert out_scale.dtype == torch.float32
    assert out_scale.shape == (1, 1, 64, 1)
    assert lse.shape == (1, 1, 64)


@pytest.mark.parametrize("headdim", [96, 128])
def test_sageattn_nhd_layout_smoke(headdim: int):
    _manual_seed(888)

    batch, seqlen_q, seqlen_kv, heads = 1, 64, 256, 2
    q, k, v = _make_inputs(
        tensor_layout="NHD",
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        headdim=headdim,
        dtype=torch.bfloat16,
        device="musa",
    )

    out = sageattn(q, k, v, tensor_layout="NHD", qk_quant_dtype="fp8")

    assert out.shape == (batch, seqlen_q, heads, headdim)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all().item()


def test_sageattn_noncontiguous_inputs_smoke():
    _manual_seed(2024)

    batch, heads, seqlen_q, seqlen_kv, headdim = 1, 2, 64, 256, 128
    q_base, k_base, v_base = _make_inputs(
        tensor_layout="NHD",
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        headdim=headdim,
        dtype=torch.bfloat16,
        device="musa",
    )

    q = q_base.transpose(1, 2)
    k = k_base.transpose(1, 2)
    v = v_base.transpose(1, 2)

    out = sageattn(q, k, v, tensor_layout="HND", qk_quant_dtype="fp8")

    assert out.shape == (batch, heads, seqlen_q, headdim)
    assert torch.isfinite(out).all().item()
