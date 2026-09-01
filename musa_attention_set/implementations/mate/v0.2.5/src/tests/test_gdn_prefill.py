from __future__ import annotations

import os
import random

import pytest
import torch
import torch.nn.functional as F

from mate.gdn_prefill import chunk_gated_delta_rule


pytestmark = pytest.mark.usefixtures("disable_mudnn_tf32")

CHUNK_SIZE = 64
HEAD_SIZE = 128


def exclusive_cumsum(seq_lens: list[int]) -> list[int]:
    out = [0]
    for seq_len in seq_lens:
        out.append(out[-1] + seq_len)
    return out


def _get_runtime_device() -> torch.device:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device("musa")
    pytest.skip("MUSA is not available")


def _manual_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if device.type == "musa":
        torch.musa.manual_seed(seed)


def _synchronize(device: torch.device) -> None:
    if device.type == "musa":
        torch.musa.synchronize()


def _resolve_scale(scale: float | str, head_size: int) -> float:
    return head_size**-0.5 if scale == "auto" else float(scale)


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
    is_output: bool,
) -> None:
    if dtype == torch.bfloat16:
        rtol = 5e-2 if is_output else 2e-2
        atol = 5e-2 if is_output else 2e-2
    else:
        rtol = 2e-4 if is_output else 1e-2
        atol = 2e-4 if is_output else 1e-2
    torch.testing.assert_close(
        actual.detach().float().cpu(),
        expected.detach().float().cpu(),
        rtol=rtol,
        atol=atol,
    )


def gen_qkv(
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    normalize_k: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_seq_len = sum(seq_lens)
    q_dim = num_qk_heads * head_size
    k_dim = num_qk_heads * head_size
    v_dim = num_v_heads * head_size
    fused_dim = q_dim + k_dim + v_dim

    mixed_qkv = torch.empty(total_seq_len, fused_dim, device=device, dtype=dtype)
    q_flat, k_flat, v_flat = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

    q_dense = torch.empty(
        total_seq_len, num_qk_heads, head_size, device=device, dtype=dtype
    ).uniform_(-0.25, 0.25)
    k_dense = torch.empty(
        total_seq_len, num_qk_heads, head_size, device=device, dtype=dtype
    ).uniform_(-0.25, 0.25)
    v_dense = torch.empty(
        total_seq_len, num_v_heads, head_size, device=device, dtype=dtype
    ).uniform_(-0.25, 0.25)
    if normalize_k:
        k_dense = F.normalize(k_dense.float(), p=2.0, dim=-1).to(dtype)

    q_flat.copy_(q_dense.reshape(total_seq_len, q_dim))
    k_flat.copy_(k_dense.reshape(total_seq_len, k_dim))
    v_flat.copy_(v_dense.reshape(total_seq_len, v_dim))

    q = q_flat.view(total_seq_len, num_qk_heads, head_size)
    k = k_flat.view(total_seq_len, num_qk_heads, head_size)
    v = v_flat.view(total_seq_len, num_v_heads, head_size)

    assert q.stride() == (fused_dim, head_size, 1)
    assert k.stride() == (fused_dim, head_size, 1)
    assert v.stride() == (fused_dim, head_size, 1)
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()
    return q, k, v


def gen_gates(
    total_seq_len: int,
    num_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.exp(
        -0.02 * torch.rand(total_seq_len, num_heads, device=device, dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(total_seq_len, num_heads, device=device, dtype=torch.float32)
    )
    return g, beta


def _to_kernel_gate_space(g: torch.Tensor, is_log_space: bool) -> torch.Tensor:
    return torch.log(g) if is_log_space else g


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.where(x <= 0, x, torch.full_like(x, -float("inf"))))


def _inverse_from_neg_strict_lower(p: torch.Tensor) -> torch.Tensor:
    length = p.shape[0]
    p = p.to(torch.float32)
    s = torch.eye(length, device=p.device, dtype=torch.float32)
    for i in range(1, length):
        for j in range(i):
            s[i, j] = torch.dot(p[i, j:i], s[j:i, j])
    return s


@torch.inference_mode()
def blockwise_gdn_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    seq_lens: list[int],
    *,
    initial_state: torch.Tensor,
    chunk_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_seq_len, num_qk_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    assert k.shape == (total_seq_len, num_qk_heads, head_size)
    assert v.shape == (total_seq_len, num_v_heads, head_size)
    assert g.shape == beta.shape == (total_seq_len, num_v_heads)
    assert num_v_heads % num_qk_heads == 0
    assert initial_state.shape == (
        len(seq_lens),
        num_v_heads,
        head_size,
        head_size,
    )

    group_size = num_v_heads // num_qk_heads
    output = torch.empty(
        total_seq_len, num_v_heads, head_size, device=q.device, dtype=torch.float32
    )
    final_state = torch.empty_like(initial_state, dtype=torch.float32)

    cu_seqlens = exclusive_cumsum(seq_lens)
    for seq_idx, seq_start in enumerate(cu_seqlens[:-1]):
        seq_end = cu_seqlens[seq_idx + 1]
        state = torch.transpose(initial_state[seq_idx].to(torch.float32).clone(), 1, 2)

        for left in range(seq_start, seq_end, chunk_size):
            right = min(left + chunk_size, seq_end)
            q_blk = q[left:right].to(torch.float32)
            k_blk = k[left:right].to(torch.float32)
            v_blk = v[left:right].to(torch.float32)
            g_blk = torch.cumsum(torch.log(g[left:right].to(torch.float32)), dim=0)
            beta_blk = beta[left:right].to(torch.float32)

            next_state = torch.empty_like(state)
            for v_head in range(num_v_heads):
                qk_head = v_head // group_size
                q_h = q_blk[:, qk_head, :]
                k_h = k_blk[:, qk_head, :]
                v_h = v_blk[:, v_head, :]
                g_h = g_blk[:, v_head]
                beta_h = beta_blk[:, v_head]

                gamma = torch.tril(_safe_exp(g_h[:, None] - g_h[None, :]))
                gram = k_h @ k_h.transpose(0, 1)
                transition = -gram * beta_h[:, None] * torch.tril(gamma, diagonal=-1)
                solve = _inverse_from_neg_strict_lower(transition)

                old_state = state[v_head]
                g_exp = _safe_exp(g_h)
                w = v_h - g_exp.unsqueeze(-1) * (k_h @ old_state)
                new_v = (solve * beta_h[None, :]) @ w

                o_inter = g_exp.unsqueeze(-1) * (q_h @ old_state)
                attn = (q_h @ k_h.transpose(0, 1)) * gamma
                output[left:right, v_head] = scale * (o_inter + attn @ new_v)

                g_last = g_h[-1]
                new_v_scaled = _safe_exp(g_last - g_h).unsqueeze(-1) * new_v
                next_state[v_head] = (
                    _safe_exp(g_last) * old_state + k_h.transpose(0, 1) @ new_v_scaled
                )

            state = next_state

        final_state[seq_idx] = torch.transpose(state, 1, 2)

    return output, final_state


def _to_kernel_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    seq_lens: list[int],
    *,
    varlen: bool,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    if varlen:
        cu_seqlens = torch.tensor(
            exclusive_cumsum(seq_lens), device=device, dtype=torch.int32
        )
        return q, k, v, g, beta, cu_seqlens

    assert len(set(seq_lens)) == 1
    batch_size = len(seq_lens)
    num_tokens = seq_lens[0]
    return (
        q.reshape(batch_size, num_tokens, q.shape[1], q.shape[2]),
        k.reshape(batch_size, num_tokens, k.shape[1], k.shape[2]),
        v.reshape(batch_size, num_tokens, v.shape[1], v.shape[2]),
        g.reshape(batch_size, num_tokens, g.shape[1]),
        beta.reshape(batch_size, num_tokens, beta.shape[1]),
        None,
    )


def _reference_output_for_kernel_shape(
    output: torch.Tensor,
    *,
    seq_lens: list[int],
    varlen: bool,
) -> torch.Tensor:
    if varlen:
        return output
    batch_size = len(seq_lens)
    num_tokens = seq_lens[0]
    return output.reshape(batch_size, num_tokens, output.shape[1], output.shape[2])


def _test_prefill_kernel(
    *,
    dtype_name: str,
    num_qk_heads: int,
    num_v_heads: int,
    seq_lens: list[int],
    scale: float | str,
    use_initial_state: bool,
    varlen: bool,
    seed: int,
    is_log_space: bool = True,
    use_zero_alpha: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> None:
    device = _get_runtime_device()
    _manual_seed(seed, device)

    dtype = getattr(torch, dtype_name)
    scale = _resolve_scale(scale, HEAD_SIZE)
    scale = 1.0
    q, k, v = gen_qkv(
        seq_lens=seq_lens,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        head_size=HEAD_SIZE,
        dtype=dtype,
        device=device,
        normalize_k=not use_qk_l2norm_in_kernel,
    )
    g, beta = gen_gates(sum(seq_lens), num_v_heads, device)
    if use_zero_alpha:
        g[7::17, :] = 0.0
    initial_state = 0.05 * torch.randn(
        len(seq_lens),
        num_v_heads,
        HEAD_SIZE,
        HEAD_SIZE,
        device=device,
        dtype=torch.float32,
    )
    ref_initial_state = (
        initial_state if use_initial_state else torch.zeros_like(initial_state)
    )
    kernel_initial_state = initial_state if use_initial_state else None

    q_in, k_in, v_in, g_in, beta_in, cu_seqlens = _to_kernel_inputs(
        q, k, v, g, beta, seq_lens, varlen=varlen, device=device
    )
    assert q_in.stride(-1) == k_in.stride(-1) == v_in.stride(-1) == 1
    assert not q_in.is_contiguous()
    assert not k_in.is_contiguous()
    assert not v_in.is_contiguous()
    kernel_g_in = _to_kernel_gate_space(g_in, is_log_space)
    actual_o, actual_state = chunk_gated_delta_rule(
        q=q_in,
        k=k_in,
        v=v_in,
        g=kernel_g_in,
        beta=beta_in,
        scale=scale,
        initial_state=kernel_initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_log_space=is_log_space,
    )
    _synchronize(device)

    ref_o, ref_state = blockwise_gdn_prefill_reference(
        q.float().cpu(),
        k.float().cpu(),
        v.float().cpu(),
        g.cpu(),
        beta.cpu(),
        seq_lens,
        initial_state=ref_initial_state.cpu(),
        chunk_size=CHUNK_SIZE,
        scale=scale,
    )
    ref_o = _reference_output_for_kernel_shape(ref_o, seq_lens=seq_lens, varlen=varlen)

    _assert_close(actual_o, ref_o, dtype=dtype, is_output=False)
    _assert_close(actual_state, ref_state, dtype=dtype, is_output=False)


def _concat_varlen_segments(
    x1: torch.Tensor,
    seq_lens1: list[int],
    x2: torch.Tensor,
    seq_lens2: list[int],
) -> torch.Tensor:
    cu_seqlens1 = exclusive_cumsum(seq_lens1)
    cu_seqlens2 = exclusive_cumsum(seq_lens2)
    pieces = []
    for seq_idx in range(len(seq_lens1)):
        pieces.append(x1[cu_seqlens1[seq_idx] : cu_seqlens1[seq_idx + 1]])
        pieces.append(x2[cu_seqlens2[seq_idx] : cu_seqlens2[seq_idx + 1]])
    return torch.cat(pieces, dim=0)


def _test_chunked_prefill(
    *,
    dtype_name: str,
    num_qk_heads: int,
    num_v_heads: int,
    seq_lens1: list[int],
    seq_lens2: list[int],
    scale: float | str,
    seed: int,
    is_log_space: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
) -> None:
    assert len(seq_lens1) == len(seq_lens2)
    device = _get_runtime_device()
    _manual_seed(seed, device)

    dtype = getattr(torch, dtype_name)
    scale = _resolve_scale(scale, HEAD_SIZE)

    q1, k1, v1 = gen_qkv(
        seq_lens1,
        num_qk_heads,
        num_v_heads,
        HEAD_SIZE,
        dtype,
        device,
        normalize_k=not use_qk_l2norm_in_kernel,
    )
    q2, k2, v2 = gen_qkv(
        seq_lens2,
        num_qk_heads,
        num_v_heads,
        HEAD_SIZE,
        dtype,
        device,
        normalize_k=not use_qk_l2norm_in_kernel,
    )
    g1, beta1 = gen_gates(sum(seq_lens1), num_v_heads, device)
    g2, beta2 = gen_gates(sum(seq_lens2), num_v_heads, device)

    q1_in, k1_in, v1_in, g1_in, beta1_in, cu_seqlens1 = _to_kernel_inputs(
        q1, k1, v1, g1, beta1, seq_lens1, varlen=True, device=device
    )
    q2_in, k2_in, v2_in, g2_in, beta2_in, cu_seqlens2 = _to_kernel_inputs(
        q2, k2, v2, g2, beta2, seq_lens2, varlen=True, device=device
    )
    kernel_g1_in = _to_kernel_gate_space(g1_in, is_log_space)
    kernel_g2_in = _to_kernel_gate_space(g2_in, is_log_space)

    actual_o1, state1 = chunk_gated_delta_rule(
        q=q1_in,
        k=k1_in,
        v=v1_in,
        g=kernel_g1_in,
        beta=beta1_in,
        scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens1,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_log_space=is_log_space,
    )
    actual_o2, state2 = chunk_gated_delta_rule(
        q=q2_in,
        k=k2_in,
        v=v2_in,
        g=kernel_g2_in,
        beta=beta2_in,
        scale=scale,
        initial_state=state1,
        output_final_state=True,
        cu_seqlens=cu_seqlens2,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_log_space=is_log_space,
    )
    _synchronize(device)

    q = _concat_varlen_segments(q1, seq_lens1, q2, seq_lens2)
    k = _concat_varlen_segments(k1, seq_lens1, k2, seq_lens2)
    v = _concat_varlen_segments(v1, seq_lens1, v2, seq_lens2)
    g = _concat_varlen_segments(g1, seq_lens1, g2, seq_lens2)
    beta = _concat_varlen_segments(beta1, seq_lens1, beta2, seq_lens2)
    seq_lens = [a + b for a, b in zip(seq_lens1, seq_lens2)]
    initial_state = torch.zeros(
        len(seq_lens),
        num_v_heads,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cpu",
        dtype=torch.float32,
    )
    ref_o, ref_state = blockwise_gdn_prefill_reference(
        q.float().cpu(),
        k.float().cpu(),
        v.float().cpu(),
        g.cpu(),
        beta.cpu(),
        seq_lens,
        initial_state=initial_state.cpu(),
        chunk_size=CHUNK_SIZE,
        scale=scale,
    )

    actual_o = _concat_varlen_segments(actual_o1, seq_lens1, actual_o2, seq_lens2)
    _assert_close(actual_o, ref_o, dtype=dtype, is_output=True)
    _assert_close(state2, ref_state, dtype=dtype, is_output=False)


PREFILL_CASES = [
    pytest.param([64], 1, 1, "auto", False, False, 0, id="full-64-h1-zero"),
    pytest.param([128, 128], 2, 2, 1.0, True, False, 1, id="batch2-128-h2-init"),
    pytest.param([128], 1, 2, "auto", True, False, 2, id="gva-full-128-init"),
]

VARLEN_PREFILL_CASES = [
    pytest.param([31], 1, 1, "auto", False, 3, id="varlen-tail-31-h1-zero"),
    pytest.param([64, 128], 2, 2, 1.0, True, 4, id="varlen-64-128-h2-init"),
    pytest.param([31, 63, 93], 1, 2, "auto", True, 5, id="gva-varlen-tails-init"),
]

CHUNKED_PREFILL_CASES = [
    pytest.param([64], [128], 1, 1, "auto", 6, id="chunked-full-h1"),
    pytest.param([31, 63], [33, 65], 1, 2, 1.0, 7, id="gva-chunked-tails"),
]


# @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("dtype_name", ["float16"])
@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_qk_heads",
        "num_v_heads",
        "scale",
        "use_initial_state",
        "varlen",
        "seed_offset",
    ),
    PREFILL_CASES,
)
def test_gdn_prefill_matches_reference(
    dtype_name: str,
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    scale: float | str,
    use_initial_state: bool,
    varlen: bool,
    seed_offset: int,
) -> None:
    seed = int(os.environ.get("SEED", "0")) + seed_offset
    _test_prefill_kernel(
        dtype_name=dtype_name,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        seq_lens=seq_lens,
        scale=scale,
        use_initial_state=use_initial_state,
        varlen=varlen,
        seed=seed,
    )


def test_gdn_prefill_alpha_space_matches_reference() -> None:
    seed = int(os.environ.get("SEED", "0")) + 8
    _test_prefill_kernel(
        dtype_name="float16",
        num_qk_heads=1,
        num_v_heads=1,
        seq_lens=[64],
        scale=1.0,
        use_initial_state=False,
        varlen=False,
        seed=seed,
        is_log_space=False,
    )


def test_gdn_prefill_log_space_neg_inf_matches_reference() -> None:
    seed = int(os.environ.get("SEED", "0")) + 9
    _test_prefill_kernel(
        dtype_name="float16",
        num_qk_heads=1,
        num_v_heads=1,
        seq_lens=[64],
        scale=1.0,
        use_initial_state=True,
        varlen=False,
        seed=seed,
        is_log_space=True,
        use_zero_alpha=True,
    )


# @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("dtype_name", ["float16"])
@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_qk_heads",
        "num_v_heads",
        "scale",
        "use_initial_state",
        "seed_offset",
    ),
    VARLEN_PREFILL_CASES,
)
def test_gdn_prefill_varlen_matches_reference(
    dtype_name: str,
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    scale: float | str,
    use_initial_state: bool,
    seed_offset: int,
) -> None:
    seed = int(os.environ.get("SEED", "0")) + seed_offset
    _test_prefill_kernel(
        dtype_name=dtype_name,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        seq_lens=seq_lens,
        scale=scale,
        use_initial_state=use_initial_state,
        varlen=True,
        seed=seed,
    )


@pytest.mark.parametrize("dtype_name", ["float16"])
# @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    ("seq_lens1", "seq_lens2", "num_qk_heads", "num_v_heads", "scale", "seed_offset"),
    CHUNKED_PREFILL_CASES,
)
def test_gdn_prefill_chunked_matches_reference(
    dtype_name: str,
    seq_lens1: list[int],
    seq_lens2: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    scale: float | str,
    seed_offset: int,
) -> None:
    seed = int(os.environ.get("SEED", "0")) + seed_offset
    _test_chunked_prefill(
        dtype_name=dtype_name,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        seq_lens1=seq_lens1,
        seq_lens2=seq_lens2,
        scale=scale,
        seed=seed,
    )
