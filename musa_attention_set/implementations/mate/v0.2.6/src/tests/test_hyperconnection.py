import pytest
import torch

import mate
import mate.deep_gemm as deep_gemm
from mate.testing.utils import calc_diff
from mate.testing import supported_musa_compute_capability


def _truncate_fp32_to_tf32(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    return (x.view(torch.int32) & ~0x1FFF).view(torch.float32)


def _sinkhorn_ref(logits: torch.Tensor, eps: float, repeat: int) -> torch.Tensor:
    row_max = logits.amax(dim=-1, keepdim=True)
    comb = torch.exp(logits - row_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def _mhc_pre_ref(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.reshape(-1, mhc_mult, hidden_size)
    x_flat = residual_flat.float().reshape(-1, mhc_mult * hidden_size)
    rms = torch.rsqrt(
        x_flat.pow(2).sum(dim=-1, keepdim=True) / (mhc_mult * hidden_size) + rms_eps
    )
    mixes = torch.matmul(x_flat, hc_fn.t()) * rms

    pre = (
        torch.sigmoid(mixes[:, :mhc_mult] * hc_scale[0] + hc_base[:mhc_mult])
        + mhc_pre_eps
    )
    post = torch.sigmoid(
        mixes[:, mhc_mult : 2 * mhc_mult] * hc_scale[1]
        + hc_base[mhc_mult : 2 * mhc_mult]
    )
    post = post * mhc_post_mult_value

    comb_logits = mixes[:, 2 * mhc_mult :].reshape(-1, mhc_mult, mhc_mult)
    comb_base = hc_base[2 * mhc_mult :].reshape(mhc_mult, mhc_mult)
    comb = _sinkhorn_ref(
        comb_logits * hc_scale[2] + comb_base,
        mhc_sinkhorn_eps,
        sinkhorn_repeat,
    )
    layer_input = (pre.unsqueeze(-1) * residual_flat.float()).sum(dim=-2).bfloat16()
    outer_shape = residual.shape[:-2]
    return (
        post.reshape(*outer_shape, mhc_mult, 1),
        comb.reshape(*outer_shape, mhc_mult, mhc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def _make_mhc_inputs(
    *,
    num_tokens: int,
    hidden_size: int,
    mhc_mult: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    device = torch.device("musa")
    residual = (
        torch.randn(
            num_tokens,
            mhc_mult,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    hc_fn = (
        torch.randn(
            mhc_mult * (2 + mhc_mult),
            mhc_mult * hidden_size,
            device=device,
            dtype=torch.float32,
        )
        * 0.01
    )
    hc_scale = torch.tensor([0.8, 0.7, 0.5], device=device, dtype=torch.float32)
    hc_base = torch.linspace(
        -0.2,
        0.2,
        mhc_mult * (2 + mhc_mult),
        device=device,
        dtype=torch.float32,
    )
    return residual, hc_fn, hc_scale, hc_base


@supported_musa_compute_capability([31])
@pytest.mark.usefixtures("enable_musa_tf32")
@pytest.mark.parametrize("m", [13, 137, 4096, 8192])
@pytest.mark.parametrize("n,k", [(24, 28672), (24, 7680), (24, 7168)])
@pytest.mark.parametrize("num_splits", [None, 16])
def test_hc_prenorm_gemm(m: int, n: int, k: int, num_splits: int | None) -> None:
    # Needs TF32 precision for PyTorch GEMMs
    a = torch.randn((m, k), dtype=torch.bfloat16, device="musa")
    b = torch.randn((n, k), dtype=torch.float32, device="musa")

    if num_splits is None:
        d = torch.empty((m, n), dtype=torch.float32, device="musa")
        s = torch.empty((m,), dtype=torch.float32, device="musa")
    else:
        d = torch.empty((num_splits, m, n), dtype=torch.float32, device="musa")
        s = torch.empty((num_splits, m), dtype=torch.float32, device="musa")

    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)

    final_d = d if num_splits is None else d.sum(0)
    final_s = s if num_splits is None else s.sum(0)

    ref_d = a.float() @ _truncate_fp32_to_tf32(b).T
    ref_s = a.float().square().sum(-1)

    diff_d = calc_diff(final_d, ref_d)
    diff_s = calc_diff(final_s, ref_s)
    diff = max(diff_d, diff_s)

    assert diff < 1e-7, (
        f"FAILED m={m}, n={n}, k={k}, num_splits={num_splits}: "
        f"diff_d={diff_d:.2e}, diff_s={diff_s:.2e}"
    )


@supported_musa_compute_capability([31])
@pytest.mark.usefixtures("enable_musa_tf32")
@pytest.mark.parametrize("num_tokens,hidden_size,split_k", [(16, 4096, 32)])
def test_mhc_pre_deepgemm_tilelang_matches_torch_reference(
    num_tokens: int,
    hidden_size: int,
    split_k: int,
) -> None:
    pytest.importorskip("tilelang")

    residual, hc_fn, hc_scale, hc_base = _make_mhc_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
    )
    kwargs = dict(
        rms_eps=1e-6,
        mhc_pre_eps=1e-6,
        mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=2.0,
        sinkhorn_repeat=4,
    )

    x_flat = residual.reshape(num_tokens, -1)
    d_part = torch.empty(
        split_k,
        num_tokens,
        hc_fn.shape[0],
        device=residual.device,
        dtype=torch.float32,
    )
    s_part = torch.empty(
        split_k,
        num_tokens,
        device=residual.device,
        dtype=torch.float32,
    )
    deep_gemm.tf32_hc_prenorm_gemm(
        x_flat,
        hc_fn.contiguous(),
        d_part,
        s_part,
        num_splits=split_k,
    )
    post, comb, layer_input = mate.hyperconnection.mhc_pre_big_fuse(
        d_part,
        s_part,
        hc_scale,
        hc_base,
        residual,
        backend="tilelang",
        threads=128,
        hidden_block=512,
        pass_config="safe",
        **kwargs,
    )

    ref_post, ref_comb, ref_layer_input = _mhc_pre_ref(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        **kwargs,
    )
    torch.testing.assert_close(
        post.reshape(num_tokens, 4, 1).float().cpu(),
        ref_post.float().cpu(),
        atol=2e-4,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        comb.reshape(num_tokens, 4, 4).float().cpu(),
        ref_comb.float().cpu(),
        atol=2e-4,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        layer_input.float().cpu(),
        ref_layer_input.float().cpu(),
        atol=2e-2,
        rtol=2e-2,
    )


@supported_musa_compute_capability([31])
def test_mhc_pre_convenience_deepgemm_tilelang_matches_torch_reference() -> None:
    pytest.importorskip("tilelang")
    residual, hc_fn, hc_scale, hc_base = _make_mhc_inputs(
        num_tokens=16,
        hidden_size=4096,
    )
    kwargs = dict(
        rms_eps=1e-6,
        mhc_pre_eps=1e-6,
        mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=2.0,
        sinkhorn_repeat=4,
    )
    post, comb, layer_input = mate.hyperconnection.mhc_pre(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        split_k=32,
        **kwargs,
    )
    ref_post, ref_comb, ref_layer_input = _mhc_pre_ref(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        **kwargs,
    )
    torch.testing.assert_close(
        post.float().cpu(),
        ref_post.float().cpu(),
        atol=2e-4,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        comb.float().cpu(),
        ref_comb.float().cpu(),
        atol=2e-4,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        layer_input.float().cpu(),
        ref_layer_input.float().cpu(),
        atol=2e-2,
        rtol=2e-2,
    )
