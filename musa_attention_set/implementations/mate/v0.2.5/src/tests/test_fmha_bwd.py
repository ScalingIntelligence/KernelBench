# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_musa  # noqa: F401
from mate import flash_attn_varlen_func
from mate.execution_context import (
    MateDryRunComplete,
    is_dry_run_enabled,
    maybe_fake_tensor_mode,
)
from mate.flash_attention.tilelang.flash_attention_varlen_bwd import (
    flashattn_varlen_bwd_interface,
)
from typing import NamedTuple, Optional  # noqa: F401
from mate.testing import supported_musa_compute_capability


USE_FAKE_MODE = is_dry_run_enabled()


def ref_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    cu_kv_lens: torch.Tensor,
    max_query_len: int,
    max_kv_len: int,
    is_causal: bool = True,
    is_varlen: bool = True,
    scale: float = 1.0,
    upcast: bool = True,
    reorder_ops: bool = False,
):
    batch = len(cu_query_lens) - 1 if is_varlen else query.shape[0]
    dtype_og = query.dtype
    # head_size_qk = key_cache.shape[-1]

    outputs: list[torch.Tensor] = []
    lse_outputs: list[torch.Tensor] = []

    for i in range(batch):
        query_len = (
            cu_query_lens[i + 1] - cu_query_lens[i] if is_varlen else query.shape[1]
        )
        kv_len = (
            cu_kv_lens[i + 1] - cu_kv_lens[i] if is_varlen else value_cache.shape[1]
        )
        if is_varlen:
            q = query[cu_query_lens[i] : cu_query_lens[i + 1]]

            k = key_cache[cu_kv_lens[i] : cu_kv_lens[i + 1]]
            v = value_cache[cu_kv_lens[i] : cu_kv_lens[i + 1]]
        else:
            q = query[i]
            k = key_cache[i]
            v = value_cache[i]

        if upcast:
            q, k, v = q.float(), k.float(), v.float()

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        # raw attention score
        if reorder_ops:
            attn = torch.einsum("qhd,khd->hqk", q, k * scale)
        else:
            attn = torch.einsum("qhd,khd->hqk", q * scale, k)
        attn = attn.float()

        # mask
        if is_causal:
            empty_mask = torch.ones(query_len, kv_len, device=query.device)
            mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
            attn.masked_fill_(mask, float("-inf"))

        # LSE (head, query)
        lse = torch.logsumexp(attn, dim=-1)  # shape [H, Q]
        lse = torch.nan_to_num(lse)
        lse_outputs.append(lse)

        # softmax = exp(score - lse)
        attn = torch.exp(attn - lse.unsqueeze(-1)).to(v.dtype)

        # output
        out = torch.einsum("hqk,khd->qhd", attn, v)
        out = torch.nan_to_num(out)
        outputs.append(out.to(dtype=dtype_og))
    if is_varlen:
        return torch.cat(outputs, dim=0), torch.cat(lse_outputs, dim=1)
    else:
        return torch.stack(outputs, dim=0), torch.stack(lse_outputs, dim=0)


def clone_with_grad(t: torch.Tensor) -> torch.Tensor:
    return t.clone().detach().requires_grad_(True)


class FmhaRunResult(NamedTuple):
    out: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


def strided_last_dim_tensor(
    shape: tuple[int, ...],
    *,
    device: str,
    dtype: torch.dtype,
    requires_grad: bool = False,
    pad: int = 8,
) -> torch.Tensor:
    base = torch.randn((*shape[:-1], shape[-1] + pad), device=device, dtype=dtype)
    tensor = base[..., : shape[-1]].detach()
    if requires_grad:
        tensor.requires_grad_(True)
    assert tensor.stride(-1) == 1
    assert not tensor.is_contiguous()
    return tensor


def copy_to_strided_last_dim(tensor: torch.Tensor, pad: int = 8) -> torch.Tensor:
    base = torch.empty(
        (*tensor.shape[:-1], tensor.shape[-1] + pad),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    out = base[..., : tensor.shape[-1]]
    out.copy_(tensor)
    assert out.stride(-1) == 1
    assert not out.is_contiguous()
    return out


def assert_fa_error_within_pt(
    actual: torch.Tensor,
    ref: torch.Tensor,
    pt: torch.Tensor,
    name: str,
    *,
    multiplier: float = 2.0,
) -> None:
    actual_err = (actual - ref).abs().max().item()
    pt_err = (pt - ref).abs().max().item()
    atol_floor = 2 * (ref + 0.3 - 0.3 - ref).abs().max().item()
    assert actual_err <= multiplier * pt_err + atol_floor, (
        f"{name} max error {actual_err} exceeds "
        f"{multiplier}x PyTorch low-precision baseline {pt_err} "
        f"+ numerical floor {atol_floor}"
    )


def fa_error_limit(
    ref: torch.Tensor,
    pt: torch.Tensor,
    *,
    multiplier: float = 2.0,
) -> float:
    pt_err = (pt - ref).abs().max().item()
    atol_floor = 2 * (ref + 0.3 - 0.3 - ref).abs().max().item()
    return multiplier * pt_err + atol_floor


def _detach_result(
    out: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> FmhaRunResult:
    return FmhaRunResult(
        out.detach().clone(),
        dq.detach().clone(),
        dk.detach().clone(),
        dv.detach().clone(),
    )


def _run_fmha_case(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    cu_query_lens: Optional[torch.Tensor],
    cu_kv_lens: Optional[torch.Tensor],
    max_query_len: Optional[int],
    max_kv_len: Optional[int],
    is_causal: bool,
    deterministic: bool,
) -> FmhaRunResult:
    q_fa, k_fa, v_fa = map(clone_with_grad, (query, key_cache, value_cache))
    out_fa = flash_attn_varlen_func(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        causal=is_causal,
        deterministic=deterministic,
    )
    out_fa.backward(gradient=grad_out.detach().clone())
    torch.musa.synchronize()
    return _detach_result(out_fa, q_fa.grad, k_fa.grad, v_fa.grad)


def _assert_bitwise_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str,
    iteration: int,
) -> None:
    if torch.equal(actual, expected):
        return
    max_abs = (actual.float() - expected.float()).abs().max().item()
    raise AssertionError(
        f"{name} changed at FMHA BWD stress iteration {iteration}: "
        f"max_abs_diff={max_abs}"
    )


def _stress_fmha_bwd_case(
    pytestconfig: pytest.Config,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    cu_query_lens: Optional[torch.Tensor],
    cu_kv_lens: Optional[torch.Tensor],
    max_query_len: Optional[int],
    max_kv_len: Optional[int],
    is_causal: bool,
    deterministic: bool,
    ref_result: FmhaRunResult,
    pt_result: FmhaRunResult,
) -> None:
    iterations = pytestconfig.getoption("fmha_bwd_stress_iters")
    if iterations < 0:
        raise pytest.UsageError("--fmha-bwd-stress-iters must be >= 0")
    if iterations <= 0:
        return
    mode = pytestconfig.getoption("fmha_bwd_stress_mode")
    progress_interval = pytestconfig.getoption("fmha_bwd_stress_progress_interval")
    if progress_interval < 0:
        raise pytest.UsageError("--fmha-bwd-stress-progress-interval must be >= 0")
    baseline: Optional[FmhaRunResult] = None

    for iteration in range(1, iterations + 1):
        try:
            current = _run_fmha_case(
                query,
                key_cache,
                value_cache,
                grad_out,
                cu_query_lens=cu_query_lens,
                cu_kv_lens=cu_kv_lens,
                max_query_len=max_query_len,
                max_kv_len=max_kv_len,
                is_causal=is_causal,
                deterministic=deterministic,
            )
        except Exception as exc:
            layout = "varlen" if cu_query_lens is not None else "bshd"
            raise RuntimeError(
                "FMHA BWD stress failed "
                f"at iteration {iteration}/{iterations}, "
                f"mode={mode}, layout={layout}, causal={is_causal}, "
                f"deterministic={deterministic}"
            ) from exc
        if progress_interval and iteration % progress_interval == 0:
            print(
                f"FMHA BWD stress progress: iteration {iteration}/{iterations}, "
                f"mode={mode}, deterministic={deterministic}",
                flush=True,
            )
        if mode == "kernel-only":
            continue

        if not deterministic:
            assert_fa_error_within_pt(
                current.dq,
                ref_result.dq,
                pt_result.dq,
                f"dq stress iter {iteration}",
            )

        if baseline is None:
            baseline = current
            continue

        _assert_bitwise_equal(current.out, baseline.out, "out", iteration)
        _assert_bitwise_equal(current.dk, baseline.dk, "dk", iteration)
        _assert_bitwise_equal(current.dv, baseline.dv, "dv", iteration)
        if deterministic:
            _assert_bitwise_equal(current.dq, baseline.dq, "dq", iteration)


def _dry_run_fmha_forward_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    is_causal: bool,
    deterministic: bool,
) -> None:
    with pytest.raises(MateDryRunComplete):
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=is_causal,
            deterministic=deterministic,
        )

    out = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
    if q.ndim == 3:
        softmax_lse = torch.empty(
            (q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device
        )
    else:
        softmax_lse = torch.empty(
            (q.shape[0], q.shape[2], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    with pytest.raises(MateDryRunComplete):
        flashattn_varlen_bwd_interface(
            q,
            k,
            v,
            out,
            torch.empty_like(out),
            softmax_lse,
            max_seqlen_q,
            max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            is_causal=is_causal,
            smscale=q.shape[-1] ** -0.5,
            dtype=None,
            is_bhsd=False,
            deterministic=deterministic,
        )


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [
            (512, 512),
            (511, 511),
            (384, 384),
            (257, 257),
            (128, 128),
            (65, 65),
            (320, 320),
            (192, 192),
        ]
    ],
)
@pytest.mark.parametrize("num_heads", [(6, 1), (6, 6)])
@pytest.mark.parametrize(
    "head_size",
    [(64, 64), (100, 80), (128, 128), (192, 128), (192, 192), (256, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("is_varlen", [True])
@pytest.mark.parametrize("deterministic", [True, False])
def test_varlen_fast_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: tuple[int, int],
    dtype: torch.dtype,
    is_causal: bool,
    is_varlen: bool,
    deterministic: bool,
    pytestconfig: pytest.Config,
) -> None:
    torch.manual_seed(42)
    torch.musa.manual_seed(42)
    device = torch.device("musa")
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    head_size_qk = head_size[0]
    head_size_v = head_size[1]
    scale = head_size_qk**-0.5

    if is_varlen:
        query = torch.randn(
            (sum(query_lens), num_query_heads, head_size_qk),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        key_cache = torch.randn(
            (sum(kv_lens), num_kv_heads, head_size_qk),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        value_cache = torch.randn(
            (sum(kv_lens), num_kv_heads, head_size_v),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        query.retain_grad()
        key_cache.retain_grad()
        value_cache.retain_grad()
    else:
        return

    q_fa, k_fa, v_fa = map(clone_with_grad, (query, key_cache, value_cache))

    cu_query_lens = torch.tensor(
        [0] + query_lens, device=device, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)

    cu_kv_lens = torch.tensor([0] + kv_lens, device=device, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    if USE_FAKE_MODE:
        _dry_run_fmha_forward_backward(
            q_fa,
            k_fa,
            v_fa,
            cu_seqlens_q=cu_query_lens,
            cu_seqlens_k=cu_kv_lens,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            is_causal=is_causal,
            deterministic=deterministic,
        )
        return

    out_fa = flash_attn_varlen_func(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        causal=is_causal,
        deterministic=deterministic,
    )

    grad_out = torch.randn_like(out_fa)
    q_t, k_t, v_t = map(clone_with_grad, (query, key_cache, value_cache))
    q_pt, k_pt, v_pt = map(clone_with_grad, (query, key_cache, value_cache))

    ref_output, _ = ref_attn(
        query=q_t,
        key_cache=k_t,
        value_cache=v_t,
        cu_query_lens=cu_query_lens,
        cu_kv_lens=cu_kv_lens,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
        upcast=True,
    )
    pt_output, _ = ref_attn(
        query=q_pt,
        key_cache=k_pt,
        value_cache=v_pt,
        cu_query_lens=cu_query_lens,
        cu_kv_lens=cu_kv_lens,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
        upcast=False,
        reorder_ops=True,
    )
    grad_fa = clone_with_grad(grad_out)
    grad_t = clone_with_grad(grad_out)
    grad_pt = clone_with_grad(grad_out)

    out_fa.backward(gradient=grad_fa, retain_graph=True)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    ref_output.backward(gradient=grad_t, retain_graph=True)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    pt_output.backward(gradient=grad_pt, retain_graph=True)
    dq_pt, dk_pt, dv_pt = q_pt.grad, k_pt.grad, v_pt.grad

    assert_fa_error_within_pt(out_fa, ref_output, pt_output, "out")
    assert_fa_error_within_pt(dk_fa, dk_t, dk_pt, "dk")
    assert_fa_error_within_pt(dv_fa, dv_t, dv_pt, "dv")
    assert_fa_error_within_pt(dq_fa, dq_t, dq_pt, "dq")

    _stress_fmha_bwd_case(
        pytestconfig,
        query,
        key_cache,
        value_cache,
        grad_out,
        cu_query_lens=cu_query_lens,
        cu_kv_lens=cu_kv_lens,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        deterministic=deterministic,
        ref_result=_detach_result(ref_output, dq_t, dk_t, dv_t),
        pt_result=_detach_result(pt_output, dq_pt, dk_pt, dv_pt),
    )


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("seq_lens", [[(511, 511)], [(63, 63)]])
@pytest.mark.parametrize("num_heads", [(6, 1), (6, 6)])
@pytest.mark.parametrize(
    "head_size",
    [(64, 64), (100, 80), (128, 128), (192, 128), (192, 192), (256, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("is_varlen", [False])
@pytest.mark.parametrize("deterministic", [True, False])
def test_bshd_fast_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: tuple[int, int],
    dtype: torch.dtype,
    is_causal: bool,
    is_varlen: bool,
    deterministic: bool,
    pytestconfig: pytest.Config,
) -> None:
    torch.manual_seed(42)
    torch.musa.manual_seed(42)
    device = torch.device("musa")
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    head_size_qk = head_size[0]
    head_size_v = head_size[1]
    scale = head_size_qk**-0.5

    batch = len(seq_lens)
    assert len(set(query_lens)) == 1
    assert len(set(kv_lens)) == 1
    query = torch.randn(
        (batch, max_query_len, num_query_heads, head_size_qk),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    key_cache = torch.randn(
        (batch, max_kv_len, num_kv_heads, head_size_qk),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    value_cache = torch.randn(
        (batch, max_kv_len, num_kv_heads, head_size_v),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    q_fa, k_fa, v_fa = map(clone_with_grad, (query, key_cache, value_cache))

    if USE_FAKE_MODE:
        _dry_run_fmha_forward_backward(
            q_fa,
            k_fa,
            v_fa,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            is_causal=is_causal,
            deterministic=deterministic,
        )
        return

    out_fa = flash_attn_varlen_func(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        causal=is_causal,
        deterministic=deterministic,
    )

    grad_out = torch.randn_like(out_fa)
    q_t, k_t, v_t = map(clone_with_grad, (query, key_cache, value_cache))
    q_pt, k_pt, v_pt = map(clone_with_grad, (query, key_cache, value_cache))

    ref_output, _ = ref_attn(
        query=q_t,
        key_cache=k_t,
        value_cache=v_t,
        cu_query_lens=None,
        cu_kv_lens=None,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
        upcast=True,
    )
    pt_output, _ = ref_attn(
        query=q_pt,
        key_cache=k_pt,
        value_cache=v_pt,
        cu_query_lens=None,
        cu_kv_lens=None,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
        upcast=False,
        reorder_ops=True,
    )
    grad_fa = clone_with_grad(grad_out)
    grad_t = clone_with_grad(grad_out)
    grad_pt = clone_with_grad(grad_out)

    out_fa.backward(gradient=grad_fa, retain_graph=True)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    ref_output.backward(gradient=grad_t, retain_graph=True)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    pt_output.backward(gradient=grad_pt, retain_graph=True)
    dq_pt, dk_pt, dv_pt = q_pt.grad, k_pt.grad, v_pt.grad

    assert_fa_error_within_pt(out_fa, ref_output, pt_output, "out")
    assert_fa_error_within_pt(dk_fa, dk_t, dk_pt, "dk")
    assert_fa_error_within_pt(dv_fa, dv_t, dv_pt, "dv")
    assert_fa_error_within_pt(dq_fa, dq_t, dq_pt, "dq")

    _stress_fmha_bwd_case(
        pytestconfig,
        query,
        key_cache,
        value_cache,
        grad_out,
        cu_query_lens=None,
        cu_kv_lens=None,
        max_query_len=None,
        max_kv_len=None,
        is_causal=is_causal,
        deterministic=deterministic,
        ref_result=_detach_result(ref_output, dq_t, dk_t, dv_t),
        pt_result=_detach_result(pt_output, dq_pt, dk_pt, dv_pt),
    )
