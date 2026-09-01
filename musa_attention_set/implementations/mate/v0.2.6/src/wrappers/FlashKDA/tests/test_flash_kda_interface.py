import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flash_kda
from mate.kda import chunk_kda

pytestmark = pytest.mark.skipif(
    os.environ.get("MATE_MUSA_ARCH_LIST") is None,
    reason="Set MATE_MUSA_ARCH_LIST=3.1 to run FlashKDA wrapper tests.",
)


def _get_runtime_device() -> torch.device:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device("musa")
    pytest.skip("MUSA is not available")


def _manual_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "musa") and torch.musa.is_available():
        torch.musa.manual_seed(seed)


def _make_inputs(
    *,
    batch: int,
    seqlen: int,
    heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    head_dim = 128
    q = F.normalize(
        torch.randn(
            (batch, seqlen, heads, head_dim), dtype=torch.float32, device=device
        ),
        p=2.0,
        dim=-1,
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(
            (batch, seqlen, heads, head_dim), dtype=torch.float32, device=device
        ),
        p=2.0,
        dim=-1,
    ).to(torch.bfloat16)
    v = torch.randn(
        (batch, seqlen, heads, head_dim), dtype=torch.bfloat16, device=device
    )
    g = torch.randn(
        (batch, seqlen, heads, head_dim), dtype=torch.bfloat16, device=device
    )
    beta = torch.randn((batch, seqlen, heads), dtype=torch.bfloat16, device=device)
    A_log = torch.rand((heads,), dtype=torch.float32, device=device)
    dt_bias = torch.rand((heads, head_dim), dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(float(head_dim))
    return q, k, v, g, beta, A_log, dt_bias, scale


def _assert_wrapper_matches(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-2,
    atol: float = 4e-3,
) -> None:
    torch.testing.assert_close(
        actual.to(torch.float32),
        expected.to(torch.float32),
        rtol=rtol,
        atol=atol,
    )


def test_get_workspace_size_matches_official_formula() -> None:
    assert flash_kda.get_workspace_size(16, 1) == 0
    assert flash_kda.get_workspace_size(32, 4, 2) == 0


def test_fwd_matches_mate_chunk_kda_batched() -> None:
    device = _get_runtime_device()
    _manual_seed(20260622)

    q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(
        batch=2, seqlen=32, heads=4, device=device
    )
    initial_state = torch.randn((2, 4, 128, 128), dtype=torch.bfloat16, device=device)
    final_state = torch.empty_like(initial_state)
    out = torch.empty_like(v)

    ret = flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        out,
        A_log,
        dt_bias,
        -5.0,
        initial_state=initial_state.clone(),
        final_state=final_state,
    )
    assert ret is None
    if device.type == "musa":
        torch.musa.synchronize()

    expected_out, expected_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state.clone(),
        output_final_state=True,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
        output=torch.empty_like(v),
        final_state=torch.empty_like(initial_state),
    )
    if device.type == "musa":
        torch.musa.synchronize()

    _assert_wrapper_matches(out, expected_out)
    _assert_wrapper_matches(final_state, expected_state, rtol=2e-2, atol=1.5e-2)


def test_fwd_matches_mate_chunk_kda_varlen_int32_cu_seqlens() -> None:
    device = _get_runtime_device()
    _manual_seed(20260623)

    q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(
        batch=1, seqlen=21, heads=2, device=device
    )
    cu_seqlens = torch.tensor([0, 9, 21], dtype=torch.int32, device=device)
    initial_state = torch.randn((2, 2, 128, 128), dtype=torch.float32, device=device)
    final_state = torch.empty_like(initial_state)
    out = torch.empty_like(v)

    flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        out,
        A_log,
        dt_bias,
        -5.0,
        initial_state=initial_state.clone(),
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )
    if device.type == "musa":
        torch.musa.synchronize()

    expected_out, expected_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
        output=torch.empty_like(v),
        final_state=torch.empty_like(initial_state),
    )
    if device.type == "musa":
        torch.musa.synchronize()

    _assert_wrapper_matches(out, expected_out)
    _assert_wrapper_matches(final_state, expected_state, rtol=2e-2, atol=1.5e-2)
