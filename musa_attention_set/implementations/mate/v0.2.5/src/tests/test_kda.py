from __future__ import annotations

import os
import random

import pytest
import torch
import torch.nn.functional as F

CHUNK_SIZE = 32
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


def _resolve_seed(seed_offset: int) -> int:
    return int(os.environ.get("SEED", "0")) + seed_offset


def _synchronize(device: torch.device) -> None:
    if device.type == "musa":
        torch.musa.synchronize()


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    value_dtype: torch.dtype,
    state_dtype: torch.dtype,
    is_output: bool,
    min_cosine: float | None = None,
) -> None:
    dtype = value_dtype if is_output else state_dtype
    if dtype == torch.bfloat16:
        rtol = 2e-2
        atol = 2e-2 if is_output else 8e-2
    else:
        rtol = 1.5e-2
        atol = 1.25e-2 if is_output else 6e-2
    actual_f = actual.detach().float().cpu()
    expected_f = expected.detach().float().cpu()
    diff = (actual_f - expected_f).abs()
    threshold = atol + rtol * expected_f.abs()
    bad = int((diff > threshold).sum().item())
    max_abs = float(diff.max().item())
    max_rel = float((diff / expected_f.abs().clamp_min(1.0e-12)).max().item())
    cosine = float(
        F.cosine_similarity(actual_f.reshape(1, -1), expected_f.reshape(1, -1)).item()
    )
    print(
        f"{'output' if is_output else 'state'}: "
        f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} bad={bad}/{diff.numel()} cosine={cosine:.9f}"
    )
    if min_cosine is not None:
        assert cosine >= min_cosine, f"cosine {cosine:.9f} is below {min_cosine:.9f}"
    torch.testing.assert_close(
        actual_f,
        expected_f,
        rtol=rtol,
        atol=atol,
    )


def _exp2_ftz_like(x: torch.Tensor) -> torch.Tensor:
    y = torch.exp2(torch.clamp(x, max=127.0))
    return torch.where(x < -126.0, torch.zeros_like(y), y)


def gen_qk(
    total: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    normalize: bool,
) -> torch.Tensor:
    qk = torch.empty(
        (total, num_heads, HEAD_SIZE), device=device, dtype=torch.float32
    ).uniform_(-0.25, 0.25)
    if normalize:
        qk = F.normalize(qk, p=2.0, dim=-1)
    return qk.to(dtype)


def gen_kda_inputs(
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    normalize_qk: bool,
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
    torch.Tensor,
]:
    total = sum(seq_lens)
    q_dim = num_qk_heads * HEAD_SIZE
    k_dim = num_qk_heads * HEAD_SIZE
    v_dim = num_v_heads * HEAD_SIZE
    fused_dim = q_dim + k_dim + v_dim

    mixed_qkv = torch.empty((total, fused_dim), device=device, dtype=dtype)
    q_flat, k_flat, v_flat = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

    q_ref = gen_qk(
        total, num_qk_heads, dtype, torch.device("cpu"), normalize=normalize_qk
    )
    k_ref = gen_qk(
        total, num_qk_heads, dtype, torch.device("cpu"), normalize=normalize_qk
    )
    v_ref = torch.empty(
        (total, num_v_heads, HEAD_SIZE), device="cpu", dtype=dtype
    ).uniform_(-0.25, 0.25)
    g_ref = torch.empty(
        (total, num_v_heads, HEAD_SIZE), device="cpu", dtype=dtype
    ).uniform_(-1.0, 1.0)
    beta_ref = torch.empty((total, num_v_heads), device="cpu", dtype=dtype).uniform_(
        -1.0, 1.0
    )

    q_dense = q_ref.to(device)
    k_dense = k_ref.to(device)
    v_dense = v_ref.to(device)
    g = g_ref.to(device)
    beta = beta_ref.to(device)

    q_flat.copy_(q_dense.reshape(total, q_dim))
    k_flat.copy_(k_dense.reshape(total, k_dim))
    v_flat.copy_(v_dense.reshape(total, v_dim))

    q = q_flat.view(total, num_qk_heads, HEAD_SIZE)
    k = k_flat.view(total, num_qk_heads, HEAD_SIZE)
    v = v_flat.view(total, num_v_heads, HEAD_SIZE)

    assert q.stride() == (fused_dim, HEAD_SIZE, 1)
    assert k.stride() == (fused_dim, HEAD_SIZE, 1)
    assert v.stride() == (fused_dim, HEAD_SIZE, 1)
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()
    return q, k, v, g, beta, q_ref, k_ref, v_ref, g_ref, beta_ref


def gen_gate_params(
    num_v_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    A_log = torch.empty((num_v_heads,), device=device, dtype=torch.float32).uniform_(
        -1.0, 0.5
    )
    dt_bias = torch.empty(
        (num_v_heads, HEAD_SIZE), device=device, dtype=torch.float32
    ).uniform_(-0.5, 0.5)
    return A_log, dt_bias


def gen_initial_state(
    num_seqs: int,
    num_v_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    use_initial_state: bool,
) -> torch.Tensor | None:
    if not use_initial_state:
        return None
    return torch.empty(
        (num_seqs, num_v_heads, HEAD_SIZE, HEAD_SIZE), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)


def _to_kernel_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    seq_lens: list[int],
    *,
    cu_seqlens_dtype: torch.dtype = torch.int64,
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
            exclusive_cumsum(seq_lens), device=device, dtype=cu_seqlens_dtype
        )
        return q, k, v, g, beta, cu_seqlens

    assert len(set(seq_lens)) == 1
    batch_size = len(seq_lens)
    num_tokens = seq_lens[0]
    return (
        q.reshape(batch_size, num_tokens, q.shape[1], q.shape[2]).contiguous(),
        k.reshape(batch_size, num_tokens, k.shape[1], k.shape[2]).contiguous(),
        v.reshape(batch_size, num_tokens, v.shape[1], v.shape[2]).contiguous(),
        g.reshape(batch_size, num_tokens, g.shape[1], g.shape[2]),
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


def _assert_last_dim_contiguous(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        assert tensor.stride(-1) == 1


def _assert_last_dim_contiguous_views(*tensors: torch.Tensor) -> None:
    _assert_last_dim_contiguous(*tensors)
    for tensor in tensors:
        assert not tensor.is_contiguous()


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


def _run_fused_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mate.kda import chunk_kda

    result = chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        final_state=final_state,
    )
    assert isinstance(result, tuple)
    return result


@torch.inference_mode()
def blockwise_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    D = q.shape[-1]
    Hqk = q.shape[-2]
    H = v.shape[-2]
    assert D == HEAD_SIZE
    assert k.shape[-2] == Hqk
    assert g.shape[-2] == H
    assert beta.shape[-1] == H
    assert H % Hqk == 0
    qk_group = H // Hqk

    if cu_seqlens is None and q.ndim == 4:
        batch_size, num_tokens = q.shape[:2]
        flat_q = q.reshape(-1, Hqk, D)
        flat_k = k.reshape(-1, Hqk, D)
        flat_v = v.reshape(-1, H, D)
        flat_g = g.reshape(-1, H, D)
        flat_beta = beta.reshape(-1, H)
        seq_offsets = [i * num_tokens for i in range(batch_size + 1)]
        out_shape = v.shape
    elif cu_seqlens is None:
        if q.ndim != 3:
            raise AssertionError(
                "Dense reference inputs must be [B, T, H, D] or flattened to [B*T, H, D]."
            )
        flat_q = q
        flat_k = k
        flat_v = v
        flat_g = g
        flat_beta = beta
        seq_len = flat_q.shape[0] // initial_state.shape[0]
        seq_offsets = [i * seq_len for i in range(initial_state.shape[0] + 1)]
        out_shape = v.shape
    else:
        flat_q = q
        flat_k = k
        flat_v = v
        flat_g = g
        flat_beta = beta
        seq_offsets = [int(x) for x in cu_seqlens.tolist()]
        out_shape = v.shape

    if use_qk_l2norm_in_kernel:
        qn = F.normalize(flat_q.float(), p=2.0, dim=-1).to(q.dtype).float()
        kn = F.normalize(flat_k.float(), p=2.0, dim=-1).to(k.dtype).float()
    else:
        qn = flat_q.float()
        kn = flat_k.float()
    out = torch.empty_like(flat_v)
    state = initial_state.clone()
    log2e = 1.4426950408889634

    def round_workspace(x: torch.Tensor) -> torch.Tensor:
        return x.to(q.dtype).float()

    for seq_idx, bos in enumerate(seq_offsets[:-1]):
        eos = seq_offsets[seq_idx + 1]
        for h in range(H):
            h_qk = h // qk_group
            s = state[seq_idx, h].float().T
            for start in range(bos, eos, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, eos)
                actual = end - start
                q_blk = torch.zeros(
                    (CHUNK_SIZE, D), device=q.device, dtype=torch.float32
                )
                k_blk = torch.zeros_like(q_blk)
                v_blk = torch.zeros_like(q_blk)
                g_blk = torch.zeros_like(q_blk)
                beta_blk = torch.full(
                    (CHUNK_SIZE,), -80.0, device=q.device, dtype=torch.float32
                )

                q_blk[:actual] = qn[start:end, h_qk]
                k_blk[:actual] = kn[start:end, h_qk]
                v_blk[:actual] = flat_v[start:end, h].float()
                g_blk[:actual] = flat_g[start:end, h].float()
                beta_blk[:actual] = flat_beta[start:end, h].float()

                gate = (
                    lower_bound
                    * log2e
                    * torch.sigmoid(torch.exp(A_log[h]) * (g_blk + dt_bias[h]))
                )
                gate[actual:] = 0.0
                g_cumsum = gate.cumsum(dim=0)
                g_total = g_cumsum[-1]
                shift = 0.5 * (g_cumsum[0] + g_total)
                k_decayed = round_workspace(k_blk * _exp2_ftz_like(g_cumsum))
                q_decayed = round_workspace(q_blk * _exp2_ftz_like(g_cumsum) * scale)
                k_decayed_for_tri = round_workspace(
                    k_blk * _exp2_ftz_like(g_cumsum - shift)
                )
                q_decayed_for_p = round_workspace(
                    q_blk * _exp2_ftz_like(g_cumsum - shift) * scale
                )
                k_inv = round_workspace(k_blk * _exp2_ftz_like(shift - g_cumsum))
                l_mat = round_workspace(
                    (k_decayed_for_tri @ k_inv.T).tril(-1)
                    * torch.sigmoid(beta_blk)[:, None]
                )

                a = torch.zeros(
                    (CHUNK_SIZE, CHUNK_SIZE), device=q.device, dtype=torch.float32
                )
                beta_sigmoid = torch.sigmoid(beta_blk)
                a[0, 0] = beta_sigmoid[0]
                for i in range(1, CHUNK_SIZE):
                    a[i, i] = beta_sigmoid[i]
                    a[i, :i] = -l_mat[i, :i] @ a[:i, :i]
                a = round_workspace(a)
                p = round_workspace((q_decayed_for_p @ k_inv.T).tril())
                s_committed = round_workspace(s)
                tmp = round_workspace(v_blk - k_decayed @ s_committed)
                u = round_workspace(a @ tmp)
                out[start:end, h] = round_workspace(q_decayed @ s_committed + p @ u)[
                    :actual
                ]

                k_restored = round_workspace(
                    k_blk * _exp2_ftz_like(g_total[None, :] - g_cumsum)
                )
                s = s * _exp2_ftz_like(g_total)[:, None] + k_restored.T @ u
            state[seq_idx, h].copy_(s.T.to(initial_state.dtype))

    return out.reshape(out_shape), state


def _test_kda_kernel(
    *,
    dtype_name: str,
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    use_initial_state: bool,
    state_dtype_name: str,
    varlen: bool,
    cu_seqlens_dtype: torch.dtype = torch.int64,
    seed: int,
    lower_bound: float,
    use_qk_l2norm_in_kernel: bool,
    zero_v: bool = False,
    min_output_cosine: float | None = None,
) -> None:
    device = _get_runtime_device()
    _manual_seed(seed, device)

    dtype = getattr(torch, dtype_name)
    state_dtype = getattr(torch, state_dtype_name)
    scale = HEAD_SIZE**-0.5
    q, k, v, g, beta, q_ref, k_ref, v_ref, g_ref, beta_ref = gen_kda_inputs(
        seq_lens,
        num_qk_heads,
        num_v_heads,
        dtype,
        device,
        normalize_qk=not use_qk_l2norm_in_kernel,
    )
    if zero_v:
        v.zero_()
        v_ref.zero_()
    A_log, dt_bias = gen_gate_params(num_v_heads, device)
    initial_state = gen_initial_state(
        len(seq_lens),
        num_v_heads,
        state_dtype,
        device,
        use_initial_state=use_initial_state,
    )
    if initial_state is None:
        ref_initial_state = torch.zeros(
            (len(seq_lens), num_v_heads, HEAD_SIZE, HEAD_SIZE),
            device="cpu",
            dtype=dtype,
        )
    else:
        ref_initial_state = initial_state.cpu().clone()

    q_in, k_in, v_in, g_in, beta_in, cu_seqlens = _to_kernel_inputs(
        q,
        k,
        v,
        g,
        beta,
        seq_lens,
        cu_seqlens_dtype=cu_seqlens_dtype,
        varlen=varlen,
        device=device,
    )
    if varlen:
        _assert_last_dim_contiguous_views(q_in, k_in, v_in)
    else:
        _assert_last_dim_contiguous(q_in, k_in, v_in)

    actual_o, actual_state = _run_fused_kda(
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    _synchronize(device)

    expected_o, expected_state = blockwise_kda_reference(
        q_ref,
        k_ref,
        v_ref,
        g_ref,
        beta_ref,
        scale=scale,
        initial_state=ref_initial_state,
        A_log=A_log.cpu(),
        dt_bias=dt_bias.cpu(),
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens.cpu() if cu_seqlens is not None else None,
    )
    expected_o = _reference_output_for_kernel_shape(
        expected_o, seq_lens=seq_lens, varlen=varlen
    )

    _assert_close(
        actual_o,
        expected_o,
        value_dtype=dtype,
        state_dtype=state_dtype,
        is_output=True,
        min_cosine=min_output_cosine,
    )
    _assert_close(
        actual_state,
        expected_state,
        value_dtype=dtype,
        state_dtype=state_dtype,
        is_output=False,
    )


def _test_chunked_kda(
    *,
    dtype_name: str,
    seq_lens1: list[int],
    seq_lens2: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    state_dtype_name: str,
    seed: int,
    lower_bound: float,
    use_qk_l2norm_in_kernel: bool,
) -> None:
    assert len(seq_lens1) == len(seq_lens2)
    device = _get_runtime_device()
    _manual_seed(seed, device)

    dtype = getattr(torch, dtype_name)
    state_dtype = getattr(torch, state_dtype_name)
    scale = HEAD_SIZE**-0.5
    q1, k1, v1, g1, beta1, q1_ref, k1_ref, v1_ref, g1_ref, beta1_ref = gen_kda_inputs(
        seq_lens1,
        num_qk_heads,
        num_v_heads,
        dtype,
        device,
        normalize_qk=not use_qk_l2norm_in_kernel,
    )
    q2, k2, v2, g2, beta2, q2_ref, k2_ref, v2_ref, g2_ref, beta2_ref = gen_kda_inputs(
        seq_lens2,
        num_qk_heads,
        num_v_heads,
        dtype,
        device,
        normalize_qk=not use_qk_l2norm_in_kernel,
    )
    A_log, dt_bias = gen_gate_params(num_v_heads, device)
    _assert_last_dim_contiguous_views(q1, k1, v1, q2, k2, v2)

    cu_seqlens1 = torch.tensor(
        exclusive_cumsum(seq_lens1), device=device, dtype=torch.int64
    )
    cu_seqlens2 = torch.tensor(
        exclusive_cumsum(seq_lens2), device=device, dtype=torch.int64
    )
    state1_buffer = torch.empty(
        (len(seq_lens1), num_v_heads, HEAD_SIZE, HEAD_SIZE),
        device=device,
        dtype=state_dtype,
    )
    state2_buffer = torch.empty(
        (len(seq_lens2), num_v_heads, HEAD_SIZE, HEAD_SIZE),
        device=device,
        dtype=state_dtype,
    )

    actual_o1, state1 = _run_fused_kda(
        q1,
        k1,
        v1,
        g1,
        beta1,
        scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens1,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        final_state=state1_buffer,
    )
    actual_o2, state2 = _run_fused_kda(
        q2,
        k2,
        v2,
        g2,
        beta2,
        scale=scale,
        initial_state=state1,
        output_final_state=True,
        cu_seqlens=cu_seqlens2,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        final_state=state2_buffer,
    )
    _synchronize(device)

    q = _concat_varlen_segments(q1_ref, seq_lens1, q2_ref, seq_lens2)
    k = _concat_varlen_segments(k1_ref, seq_lens1, k2_ref, seq_lens2)
    v = _concat_varlen_segments(v1_ref, seq_lens1, v2_ref, seq_lens2)
    g = _concat_varlen_segments(g1_ref, seq_lens1, g2_ref, seq_lens2)
    beta = _concat_varlen_segments(beta1_ref, seq_lens1, beta2_ref, seq_lens2)
    seq_lens = [a + b for a, b in zip(seq_lens1, seq_lens2)]
    expected_o, expected_state = blockwise_kda_reference(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=torch.zeros(
            (len(seq_lens), num_v_heads, HEAD_SIZE, HEAD_SIZE),
            device="cpu",
            dtype=state_dtype,
        ),
        A_log=A_log.cpu(),
        dt_bias=dt_bias.cpu(),
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=torch.tensor(
            exclusive_cumsum(seq_lens), device="cpu", dtype=torch.int64
        ),
    )
    actual_o = _concat_varlen_segments(actual_o1, seq_lens1, actual_o2, seq_lens2)

    _assert_close(
        actual_o, expected_o, value_dtype=dtype, state_dtype=state_dtype, is_output=True
    )
    _assert_close(
        state2,
        expected_state,
        value_dtype=dtype,
        state_dtype=state_dtype,
        is_output=False,
    )


DENSE_KDA_CASES = [
    pytest.param([64], 1, 1, False, "bfloat16", -5.0, 0, id="dense-64-h1-no-state"),
    pytest.param([64, 64], 2, 2, True, "bfloat16", -5.0, 1, id="dense-batch2-h2-state"),
    pytest.param([32, 32], 1, 2, True, "float32", -5.0, 2, id="dense-gva-fp32-state"),
]

VARLEN_KDA_CASES = [
    pytest.param([17], 1, 1, False, "bfloat16", -5.0, 3, id="varlen-tail-17"),
    pytest.param([31, 63], 1, 2, True, "bfloat16", -5.0, 4, id="varlen-gva-tails"),
    pytest.param(
        [19, 32, 57], 2, 2, True, "float32", -5.0, 5, id="varlen-multi-fp32-state"
    ),
]

CHUNKED_KDA_CASES = [
    pytest.param([31], [33], 1, 1, "bfloat16", -5.0, 6, id="chunked-single-seq"),
    pytest.param(
        [31, 63], [33, 65], 1, 2, "float32", -5.0, 7, id="chunked-gva-fp32-state"
    ),
]


@pytest.mark.parametrize("dtype_name", ["bfloat16"])
@pytest.mark.parametrize(
    "use_qk_l2norm_in_kernel",
    [False, True],
    ids=["pre-norm", "kernel-l2norm"],
)
@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_qk_heads",
        "num_v_heads",
        "use_initial_state",
        "state_dtype_name",
        "lower_bound",
        "seed_offset",
    ),
    DENSE_KDA_CASES,
)
def test_kda_fused_matches_reference(
    dtype_name: str,
    use_qk_l2norm_in_kernel: bool,
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    use_initial_state: bool,
    state_dtype_name: str,
    lower_bound: float,
    seed_offset: int,
) -> None:
    _test_kda_kernel(
        dtype_name=dtype_name,
        seq_lens=seq_lens,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        use_initial_state=use_initial_state,
        state_dtype_name=state_dtype_name,
        varlen=False,
        cu_seqlens_dtype=torch.int64,
        seed=_resolve_seed(seed_offset),
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@pytest.mark.parametrize("dtype_name", ["bfloat16"])
@pytest.mark.parametrize(
    "use_qk_l2norm_in_kernel",
    [False, True],
    ids=["pre-norm", "kernel-l2norm"],
)
@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_qk_heads",
        "num_v_heads",
        "use_initial_state",
        "state_dtype_name",
        "lower_bound",
        "seed_offset",
    ),
    VARLEN_KDA_CASES,
)
def test_kda_fused_varlen_matches_reference(
    dtype_name: str,
    use_qk_l2norm_in_kernel: bool,
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    use_initial_state: bool,
    state_dtype_name: str,
    lower_bound: float,
    seed_offset: int,
) -> None:
    _test_kda_kernel(
        dtype_name=dtype_name,
        seq_lens=seq_lens,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        use_initial_state=use_initial_state,
        state_dtype_name=state_dtype_name,
        varlen=True,
        cu_seqlens_dtype=torch.int64,
        seed=_resolve_seed(seed_offset),
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def test_kda_fused_varlen_int32_cu_seqlens_matches_reference() -> None:
    _test_kda_kernel(
        dtype_name="bfloat16",
        seq_lens=[19, 32, 57],
        num_qk_heads=2,
        num_v_heads=2,
        use_initial_state=True,
        state_dtype_name="float32",
        varlen=True,
        cu_seqlens_dtype=torch.int32,
        seed=_resolve_seed(105),
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
    )


def test_kda_fused_single_chunk_zero_v_with_initial_state_matches_reference() -> None:
    _test_kda_kernel(
        dtype_name="bfloat16",
        seq_lens=[32],
        num_qk_heads=1,
        num_v_heads=1,
        use_initial_state=True,
        state_dtype_name="float32",
        varlen=False,
        cu_seqlens_dtype=torch.int64,
        seed=_resolve_seed(107),
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
        zero_v=True,
        min_output_cosine=0.999,
    )


def test_kda_fused_varlen_single_sequence_four_chunks_matches_reference() -> None:
    _test_kda_kernel(
        dtype_name="bfloat16",
        seq_lens=[128],
        num_qk_heads=2,
        num_v_heads=2,
        use_initial_state=True,
        state_dtype_name="float32",
        varlen=True,
        cu_seqlens_dtype=torch.int32,
        seed=_resolve_seed(106),
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
    )


@pytest.mark.parametrize("dtype_name", ["bfloat16"])
@pytest.mark.parametrize(
    "use_qk_l2norm_in_kernel",
    [False, True],
    ids=["pre-norm", "kernel-l2norm"],
)
@pytest.mark.parametrize(
    (
        "seq_lens1",
        "seq_lens2",
        "num_qk_heads",
        "num_v_heads",
        "state_dtype_name",
        "lower_bound",
        "seed_offset",
    ),
    CHUNKED_KDA_CASES,
)
def test_kda_fused_chunked_matches_reference(
    dtype_name: str,
    use_qk_l2norm_in_kernel: bool,
    seq_lens1: list[int],
    seq_lens2: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    state_dtype_name: str,
    lower_bound: float,
    seed_offset: int,
) -> None:
    _test_chunked_kda(
        dtype_name=dtype_name,
        seq_lens1=seq_lens1,
        seq_lens2=seq_lens2,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        state_dtype_name=state_dtype_name,
        seed=_resolve_seed(seed_offset),
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
