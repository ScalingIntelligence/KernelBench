import torch

from mate.execution_context import (
    is_dry_run_enabled,
    is_fake_mode,
    maybe_fake_tensor_mode,
)
from mate.kda import gated_delta_rule_decode

TEST_SEED = 20262423
USE_FAKE_MODE = is_dry_run_enabled()


def _exp2_f32_ref(x: torch.Tensor) -> torch.Tensor:
    LOG2E = 1.4426950408889634
    return torch.exp2(x * LOG2E)


def _apply_gate_ref(
    raw_g: torch.Tensor,
    A_log_val: torch.Tensor,
    dt_bias_vec: torch.Tensor,
    *,
    use_gate_in_kernel: bool,
    has_dt_bias: bool,
    use_lower_bound: bool,
    lower_bound: float,
):
    if not use_gate_in_kernel:
        return raw_g.float()

    g_vec = raw_g.float().clone()
    if has_dt_bias:
        g_vec = g_vec + dt_bias_vec.float()

    if use_lower_bound:
        g_vec = (
            lower_bound
            * 1.0
            / (1.0 + _exp2_f32_ref(-1 * _exp2_f32_ref(A_log_val.float()) * g_vec))
        )
    else:
        SOFTPLUS_BETA = 1.0
        SOFTPLUS_THRESHOLD = 20.0
        g_tmp = SOFTPLUS_BETA * g_vec
        softplus_x = torch.where(
            g_tmp <= SOFTPLUS_THRESHOLD,
            (1.0 / SOFTPLUS_BETA) * torch.log(1.0 + _exp2_f32_ref(g_tmp)),
            g_vec,
        )
        g_vec = -_exp2_f32_ref(A_log_val.float()) * softplus_x

    return _exp2_f32_ref(g_vec)


def _apply_beta_ref(
    beta_value: torch.Tensor,
    *,
    apply_beta_sigmoid: bool,
    allow_neg_eigval: bool,
):
    beta_out = beta_value.float().clone()
    if apply_beta_sigmoid:
        beta_out = 1.0 / (1.0 + _exp2_f32_ref(-beta_out))
        if allow_neg_eigval:
            beta_out = beta_out * 2.0
    return beta_out


def _warp_xor_sum_ref(vec: torch.Tensor) -> torch.Tensor:
    values = vec.clone().to(torch.float32)
    for mask in (16, 8, 4, 2, 1):
        shuffled = values[torch.arange(32, device=values.device) ^ mask]
        values = values + shuffled
    return values


def _summarize_diff(name: str, got: torch.Tensor, ref: torch.Tensor):
    got_flat = got.float().reshape(-1).cpu()
    ref_flat = ref.float().reshape(-1).cpu()
    abs_diff = (got_flat - ref_flat).abs()
    rel_diff = abs_diff / torch.clamp(ref_flat.abs(), min=1e-12)
    topk = min(5, rel_diff.numel())
    topk_vals, topk_idx = torch.topk(rel_diff, k=topk)
    max_idx = int(topk_idx[0].item()) if topk > 0 else -1

    print(f"{name} shape: got={tuple(got.shape)} ref={tuple(ref.shape)}")
    print(f"{name} diff summary:")
    if topk > 0:
        print(
            f"  max_abs_diff={abs_diff[max_idx].item()} max_rel_diff={topk_vals[0].item()} flat_idx={max_idx} got={got_flat[max_idx].item()} ref={ref_flat[max_idx].item()}"
        )
    print(f"  top-{topk} {name} relative diffs:")
    for rank, (idx, rel_diff_val) in enumerate(
        zip(topk_idx.tolist(), topk_vals.tolist())
    ):
        print(
            f"    [{rank}] flat_idx={idx} got={got_flat[idx].item()} ref={ref_flat[idx].item()} abs_diff={abs_diff[idx].item()} rel_diff={rel_diff_val}"
        )
    print(f"  first-5 {name} diffs:")
    for idx in range(min(5, abs_diff.numel())):
        print(
            f"    [{idx}] flat_idx={idx} got={got_flat[idx].item()} ref={ref_flat[idx].item()} abs_diff={abs_diff[idx].item()} rel_diff={rel_diff[idx].item()}"
        )


def generate_kda_case(
    *,
    seed: int = TEST_SEED,
    IS_VARLEN=True,
    USE_INITIAL_STATE=True,
    INPLACE_FINAL_STATE=False,
    IS_BETA_HEADWISE=True,
    USE_QK_L2NORM_IN_KERNEL=True,
    IS_CONTINUOUS_BATCHING=False,
    IS_SPEC_DECODING=False,
    STORE_FINAL_STATE=False,
    HAS_DT_BIAS=False,
    USE_GATE_IN_KERNEL=True,
    USE_LOWER_BOUND=False,
    APPLY_BETA_SIGMOID=True,
    ALLOW_NEG_EIGVAL=False,
    STATE_V_FIRST=False,
    B=10,
    T_fixed=4,
    H_qk=16,
    H_v=64,
    head_size=128,
    scale=0.5,
    lower_bound_value=-5.0,
    input_dtype=torch.float16,
    state_dtype=torch.float32,
    gate_batch_dtype=torch.float16,
    dt_bias_dtype=torch.float16,
    output_dtype=torch.float32,
    device="musa",
    NUM_SLOT_STATES=None,
):
    torch.manual_seed(seed)
    N_seq = B
    K = head_size
    V = head_size

    flags = {
        "IS_VARLEN": IS_VARLEN,
        "USE_INITIAL_STATE": USE_INITIAL_STATE,
        "INPLACE_FINAL_STATE": INPLACE_FINAL_STATE,
        "IS_BETA_HEADWISE": IS_BETA_HEADWISE,
        "USE_QK_L2NORM_IN_KERNEL": USE_QK_L2NORM_IN_KERNEL,
        "IS_CONTINUOUS_BATCHING": IS_CONTINUOUS_BATCHING,
        "IS_SPEC_DECODING": IS_SPEC_DECODING,
        "STORE_FINAL_STATE": STORE_FINAL_STATE,
        "HAS_DT_BIAS": HAS_DT_BIAS,
        "USE_GATE_IN_KERNEL": USE_GATE_IN_KERNEL,
        "USE_LOWER_BOUND": USE_LOWER_BOUND,
        "APPLY_BETA_SIGMOID": APPLY_BETA_SIGMOID,
        "ALLOW_NEG_EIGVAL": ALLOW_NEG_EIGVAL,
        "STATE_V_FIRST": STATE_V_FIRST,
    }

    if IS_VARLEN:
        if is_fake_mode():
            # FakeTensor has shape/dtype/device metadata but no scalar values.
            # Use a deterministic host-side pattern so varlen dry runs can
            # resolve all dynamic shapes without data-dependent .item() calls.
            seqlens_host = [1 if seq_idx % 2 == 0 else 4 for seq_idx in range(B)]
            cu_seqlens_host = [0]
            for seq_len in seqlens_host:
                cu_seqlens_host.append(cu_seqlens_host[-1] + seq_len)
            seqlens = torch.tensor(seqlens_host, device=device, dtype=torch.int32)
            cu_seqlens = torch.tensor(cu_seqlens_host, device=device, dtype=torch.int32)
            total_tokens = cu_seqlens_host[-1]
        else:
            choices = torch.tensor([1, 4], device=device, dtype=torch.int32)
            indices = torch.randint(0, choices.numel(), (B,), device=device)
            seqlens = choices[indices]
            cu_seqlens = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.int32),
                    torch.cumsum(seqlens, dim=0),
                ]
            )
            seqlens_host = None
            total_tokens = int(seqlens.sum().item())

        q_shape = (1, total_tokens, H_qk, K)
        k_shape = (1, total_tokens, H_qk, K)
        v_shape = (1, total_tokens, H_v, V)
    else:
        cu_seqlens = None
        seqlens_host = None
        seqlens = torch.full((B,), T_fixed, device=device, dtype=torch.int32)

        total_tokens = B * T_fixed

        q_shape = (B, T_fixed, H_qk, K)
        k_shape = (B, T_fixed, H_qk, K)
        v_shape = (B, T_fixed, H_v, V)

    def _sample_discrete(shape, *, dtype):
        return ((torch.rand(shape, device=device, dtype=torch.float32) * 2.0) - 1.0).to(
            dtype=dtype
        )

    q_c = _sample_discrete(q_shape, dtype=input_dtype)
    k_c = _sample_discrete(k_shape, dtype=input_dtype)
    v_c = _sample_discrete(v_shape, dtype=input_dtype)
    g_c = _sample_discrete((*v_shape[:-1], K), dtype=gate_batch_dtype)

    if IS_BETA_HEADWISE:
        beta_c = _sample_discrete(v_shape, dtype=torch.float32)
    else:
        if IS_VARLEN:
            shape_beta = (1, total_tokens, H_v)
        else:
            shape_beta = (B, T_fixed, H_v)
        beta_c = _sample_discrete(shape_beta, dtype=torch.float32)
    out = _sample_discrete(v_shape, dtype=output_dtype)

    A_log = (
        (_sample_discrete((H_v,), dtype=torch.float32)) if USE_GATE_IN_KERNEL else None
    )

    dt_bias = _sample_discrete((H_v, K), dtype=dt_bias_dtype) if HAS_DT_BIAS else None

    lower_bound = lower_bound_value if USE_LOWER_BOUND else None
    state_matrix_shape = (V, K) if STATE_V_FIRST else (K, V)

    if IS_CONTINUOUS_BATCHING:
        num_slot_states = max(total_tokens * 3, N_seq * 3)
    else:
        num_slot_states = N_seq if NUM_SLOT_STATES is None else NUM_SLOT_STATES
        if num_slot_states < N_seq:
            raise ValueError("NUM_SLOT_STATES must be at least B")
    state_base_shape = (num_slot_states, H_v, *state_matrix_shape)

    initial_state = (
        (_sample_discrete(state_base_shape, dtype=torch.float32) + 3.0).to(
            dtype=state_dtype
        )
        if USE_INITIAL_STATE
        else None
    )

    output_final_state = STORE_FINAL_STATE

    initial_state_ref = None if initial_state is None else initial_state.clone()

    final_state_slots = None
    if output_final_state:
        if IS_CONTINUOUS_BATCHING:
            if INPLACE_FINAL_STATE:
                final_state_slots = num_slot_states
            else:
                final_state_slots = total_tokens
        else:
            final_state_slots = B

    if INPLACE_FINAL_STATE:
        assert initial_state is not None
        final_state = initial_state
    elif output_final_state:
        if STATE_V_FIRST:
            final_state = q_c.new_empty(final_state_slots, H_v, V, K, dtype=state_dtype)
        else:
            final_state = q_c.new_empty(final_state_slots, H_v, K, V, dtype=state_dtype)
    else:
        final_state = None

    if IS_CONTINUOUS_BATCHING:
        if IS_VARLEN:
            max_T = (
                max(seqlens_host)
                if seqlens_host is not None
                else int(seqlens.max().item())
            )
            ssm_state_indices = torch.empty(
                (B, max_T), device=device, dtype=torch.int32
            )

            cursor = 0
            for b in range(B):
                L = (
                    seqlens_host[b]
                    if seqlens_host is not None
                    else int(seqlens[b].item())
                )
                ssm_state_indices[b, :L] = torch.arange(
                    cursor, cursor + L, device=device, dtype=torch.int32
                )
                if L < max_T:
                    ssm_state_indices[b, L:] = 0
                cursor += L
        else:
            if T_fixed == 1:
                ssm_state_indices = torch.randperm(num_slot_states, device=device)[
                    :B
                ].to(torch.int32)
            else:
                ssm_state_indices = torch.arange(
                    B * T_fixed, device=device, dtype=torch.int32
                ).reshape(B, T_fixed)

        if IS_SPEC_DECODING:
            if IS_VARLEN:
                num_accepted_tokens = torch.randint(
                    1,
                    int(seqlens.max().item()) + 1,
                    (B,),
                    device=device,
                    dtype=torch.int32,
                )
                num_accepted_tokens = torch.minimum(num_accepted_tokens, seqlens)
            else:
                num_accepted_tokens = torch.randint(
                    1, T_fixed + 1, (B,), device=device, dtype=torch.int32
                )
        else:
            num_accepted_tokens = None
    else:
        ssm_state_indices = None
        num_accepted_tokens = None
    meta = {
        "B": B,
        "T_fixed": T_fixed,
        "N_seq": N_seq,
        "total_tokens": total_tokens,
        "H_qk": H_qk,
        "H_v": H_v,
        "K": K,
        "V": V,
        "scale": scale,
        "state_matrix_shape": state_matrix_shape,
        "state_base_shape": state_base_shape,
        "num_slot_states": num_slot_states,
        "lower_bound": lower_bound,
        "output_final_state": output_final_state,
        "device": device,
        "input_dtype": input_dtype,
        "state_dtype": state_dtype,
        "gate_batch_dtype": gate_batch_dtype,
        "dt_bias_dtype": dt_bias_dtype,
        "output_dtype": output_dtype,
    }

    tensors = {
        "seqlens": seqlens,
        "cu_seqlens": cu_seqlens,
        "q_c / q": q_c,
        "k_c / k": k_c,
        "v_c / v": v_c,
        "g_c / g": g_c,
        "beta_c / beta": beta_c,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "out / o": out,
        "initial_state / h0": initial_state,
        "initial_state_ref": initial_state_ref,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "final_state": final_state,
    }

    return {
        "flags": flags,
        "meta": meta,
        "tensors": tensors,
    }


def kda_test(case: dict):
    flags = case["flags"]
    meta = case["meta"]
    tensors = case["tensors"]

    out, final_state = gated_delta_rule_decode(
        q=tensors["q_c / q"],
        k=tensors["k_c / k"],
        v=tensors["v_c / v"],
        state=tensors["initial_state / h0"],
        state_indices=tensors["ssm_state_indices"],
        A_log=tensors["A_log"],
        a=tensors["g_c / g"],
        dt_bias=tensors["dt_bias"],
        b=tensors["beta_c / beta"],
        cu_seqlens=tensors["cu_seqlens"],
        num_accepted_tokens=tensors["num_accepted_tokens"],
        output=tensors["out / o"],
        scale=meta["scale"],
        lower_bound=meta["lower_bound"] if meta["lower_bound"] is not None else 0.0,
        use_qk_l2norm=flags["USE_QK_L2NORM_IN_KERNEL"],
        is_varlen=flags["IS_VARLEN"],
        inplace_final_state=flags["INPLACE_FINAL_STATE"],
        store_final_state=flags["STORE_FINAL_STATE"],
        use_gate_in_kernel=flags["USE_GATE_IN_KERNEL"],
        use_lower_bound=flags["USE_LOWER_BOUND"],
        apply_beta_sigmoid=flags["APPLY_BETA_SIGMOID"],
        allow_neg_eigval=flags["ALLOW_NEG_EIGVAL"],
        state_v_first=flags["STATE_V_FIRST"],
        use_initial_state=flags["USE_INITIAL_STATE"],
        state_dtype=meta["state_dtype"],
    )
    return out, final_state


def _kda_reference_from_case(case: dict):
    flags = case["flags"]
    meta = case["meta"]
    tensors = case["tensors"]

    q = tensors["q_c / q"].float()
    k = tensors["k_c / k"].float()
    v = tensors["v_c / v"].float()
    g = tensors["g_c / g"].float()
    beta = tensors["beta_c / beta"].float()

    A_log = tensors["A_log"]
    if A_log is None:
        A_log = torch.zeros(
            meta["H_v"],
            device=meta["device"],
            dtype=torch.float32,
        )
    else:
        A_log = A_log.float()

    dt_bias = tensors["dt_bias"]
    if dt_bias is None:
        dt_bias = torch.zeros(
            meta["H_v"],
            meta["K"],
            device=meta["device"],
            dtype=torch.float32,
        )
    else:
        dt_bias = dt_bias.float()

    initial_state = tensors["initial_state_ref"]
    final_state = tensors["final_state"]
    cu_seqlens = tensors["cu_seqlens"]
    ssm_state_indices = tensors["ssm_state_indices"]
    num_accepted_tokens = tensors["num_accepted_tokens"]
    seqlens = tensors["seqlens"]

    if initial_state is None:
        base_state = torch.zeros(
            meta["state_base_shape"],
            device=meta["device"],
            dtype=torch.float32,
        )
    else:
        base_state = initial_state.float().clone()

    state_pool = base_state
    ref_final_state = None if final_state is None else torch.zeros_like(final_state)

    if flags["INPLACE_FINAL_STATE"]:
        assert initial_state is not None
        ref_final_state = initial_state

    ref_out = torch.zeros(
        (1, meta["total_tokens"], meta["H_v"], meta["V"]),
        device=meta["device"],
        dtype=torch.float32,
    )

    q_flat = q.reshape(1, meta["total_tokens"], meta["H_qk"], meta["K"])
    k_flat = k.reshape(1, meta["total_tokens"], meta["H_qk"], meta["K"])
    v_flat = v.reshape(1, meta["total_tokens"], meta["H_v"], meta["V"])
    g_flat = g.reshape(1, meta["total_tokens"], meta["H_v"], meta["K"])
    if flags["IS_BETA_HEADWISE"]:
        beta_flat = beta.reshape(1, meta["total_tokens"], meta["H_v"], meta["V"])
    else:
        beta_flat = beta.reshape(1, meta["total_tokens"], meta["H_v"])

    H_qk = meta["H_qk"]
    H_v = meta["H_v"]
    head_group_size = H_v // H_qk
    vec_size = meta["K"] // 32
    for seq_idx in range(meta["B"]):
        if flags["IS_VARLEN"]:
            bos = int(cu_seqlens[seq_idx].item())
            eos = int(cu_seqlens[seq_idx + 1].item())
            T_seq = eos - bos
        else:
            T_seq = int(seqlens[seq_idx].item())
            bos = seq_idx * T_seq

        if T_seq <= 0:
            continue

        if flags["IS_CONTINUOUS_BATCHING"]:
            if flags["IS_SPEC_DECODING"]:
                accepted = int(num_accepted_tokens[seq_idx].item())
                read_t = accepted - 1
            else:
                read_t = 0

            if ssm_state_indices.dim() == 1:
                state_slot = int(ssm_state_indices[seq_idx].item())
            else:
                state_slot = int(ssm_state_indices[seq_idx, read_t].item())
        else:
            state_slot = seq_idx

        if state_slot < 0:
            continue

        for local_t in range(T_seq):
            token_idx = bos + local_t

            q_heads = (
                q_flat[0, token_idx]
                .clone()
                .to(torch.float32)
                .reshape(H_qk, 32, vec_size)
            )
            k_heads = (
                k_flat[0, token_idx]
                .clone()
                .to(torch.float32)
                .reshape(H_qk, 32, vec_size)
            )

            if flags["USE_QK_L2NORM_IN_KERNEL"]:
                sum_q_lane = torch.sum(q_heads * q_heads, dim=2)
                sum_k_lane = torch.sum(k_heads * k_heads, dim=2)
                sum_q_full = torch.stack(
                    [_warp_xor_sum_ref(sum_q_lane[h]) for h in range(H_qk)], dim=0
                )
                sum_k_full = torch.stack(
                    [_warp_xor_sum_ref(sum_k_lane[h]) for h in range(H_qk)], dim=0
                )
                inv_norm_q = torch.rsqrt(sum_q_full + 1e-6)
                inv_norm_k = torch.rsqrt(sum_k_full + 1e-6)
                q_heads = q_heads * inv_norm_q[:, :, None]
                k_heads = k_heads * inv_norm_k[:, :, None]

            q_heads = q_heads * meta["scale"]

            v_token = v_flat[0, token_idx].clone().to(torch.float32)
            g_token = g_flat[0, token_idx].clone().to(torch.float32)
            h_token = state_pool[state_slot].clone().to(torch.float32)

            decay_token = torch.stack(
                [
                    _apply_gate_ref(
                        raw_g=g_token[hv_idx],
                        A_log_val=A_log[hv_idx],
                        dt_bias_vec=dt_bias[hv_idx],
                        use_gate_in_kernel=flags["USE_GATE_IN_KERNEL"],
                        has_dt_bias=flags["HAS_DT_BIAS"],
                        use_lower_bound=flags["USE_LOWER_BOUND"],
                        lower_bound=meta["lower_bound"]
                        if meta["lower_bound"] is not None
                        else 0.0,
                    )
                    for hv_idx in range(H_v)
                ],
                dim=0,
            )

            beta_token = torch.stack(
                [
                    _apply_beta_ref(
                        beta_flat[0, token_idx, hv_idx],
                        apply_beta_sigmoid=flags["APPLY_BETA_SIGMOID"],
                        allow_neg_eigval=flags["ALLOW_NEG_EIGVAL"],
                    )
                    for hv_idx in range(H_v)
                ],
                dim=0,
            )

            for qk_hid in range(H_qk):
                hv_start = qk_hid * head_group_size
                hv_end = hv_start + head_group_size
                q_lanes = q_heads[qk_hid]
                k_lanes = k_heads[qk_hid]
                if flags["STATE_V_FIRST"]:
                    h_group = (
                        h_token[hv_start:hv_end] * decay_token[hv_start:hv_end, None, :]
                    )
                    h_group_lanes = h_group.reshape(
                        head_group_size, meta["V"], 32, vec_size
                    )

                    sum_hk_lane = torch.sum(
                        h_group_lanes * k_lanes[None, None, :, :], dim=3
                    )
                    sum_hk_full = torch.stack(
                        [
                            torch.stack(
                                [
                                    _warp_xor_sum_ref(sum_hk_lane[hv_local, row])
                                    for row in range(meta["V"])
                                ],
                                dim=0,
                            )
                            for hv_local in range(head_group_size)
                        ],
                        dim=0,
                    )
                    if flags["IS_BETA_HEADWISE"]:
                        beta_group = beta_token[hv_start:hv_end]
                    else:
                        beta_group = beta_token[hv_start:hv_end][:, None]
                    v_new = (
                        v_token[hv_start:hv_end] - sum_hk_full[:, :, 0]
                    ) * beta_group
                    h_group_lanes = (
                        h_group_lanes
                        + k_lanes[None, None, :, :] * v_new[:, :, None, None]
                    )

                    sum_hq_lane = torch.sum(
                        h_group_lanes * q_lanes[None, None, :, :], dim=3
                    )
                    sum_hq_full = torch.stack(
                        [
                            torch.stack(
                                [
                                    _warp_xor_sum_ref(sum_hq_lane[hv_local, row])
                                    for row in range(meta["V"])
                                ],
                                dim=0,
                            )
                            for hv_local in range(head_group_size)
                        ],
                        dim=0,
                    )
                    o_group = sum_hq_full[:, :, 0]
                    h_token[hv_start:hv_end] = h_group_lanes.reshape(
                        head_group_size, meta["V"], meta["K"]
                    )
                    ref_out[0, token_idx, hv_start:hv_end] = o_group
                else:
                    h_group = (
                        h_token[hv_start:hv_end] * decay_token[hv_start:hv_end, :, None]
                    )
                    h_group_lanes = h_group.permute(0, 2, 1).reshape(
                        head_group_size, meta["V"], 32, vec_size
                    )

                    sum_hk_lane = torch.sum(
                        h_group_lanes * k_lanes[None, None, :, :], dim=3
                    )
                    sum_hk_full = torch.stack(
                        [
                            torch.stack(
                                [
                                    _warp_xor_sum_ref(sum_hk_lane[hv_local, col])
                                    for col in range(meta["V"])
                                ],
                                dim=0,
                            )
                            for hv_local in range(head_group_size)
                        ],
                        dim=0,
                    )
                    if flags["IS_BETA_HEADWISE"]:
                        beta_group = beta_token[hv_start:hv_end]
                    else:
                        beta_group = beta_token[hv_start:hv_end][:, None]
                    v_new = (
                        v_token[hv_start:hv_end] - sum_hk_full[:, :, 0]
                    ) * beta_group
                    h_group_lanes = (
                        h_group_lanes
                        + k_lanes[None, None, :, :] * v_new[:, :, None, None]
                    )

                    sum_hq_lane = torch.sum(
                        h_group_lanes * q_lanes[None, None, :, :], dim=3
                    )
                    sum_hq_full = torch.stack(
                        [
                            torch.stack(
                                [
                                    _warp_xor_sum_ref(sum_hq_lane[hv_local, col])
                                    for col in range(meta["V"])
                                ],
                                dim=0,
                            )
                            for hv_local in range(head_group_size)
                        ],
                        dim=0,
                    )
                    o_group = sum_hq_full[:, :, 0]
                    h_token[hv_start:hv_end] = h_group_lanes.reshape(
                        head_group_size, meta["V"], meta["K"]
                    ).permute(0, 2, 1)
                    ref_out[0, token_idx, hv_start:hv_end] = o_group

            h_token_stored = h_token.to(meta["state_dtype"]).float()
            state_pool[state_slot] = h_token_stored

            if ref_final_state is not None:
                do_store_final_this_token = False
                write_slot = seq_idx
                if flags["IS_CONTINUOUS_BATCHING"]:
                    do_store_final_this_token = True
                    if flags["INPLACE_FINAL_STATE"]:
                        if ssm_state_indices.dim() == 1:
                            write_slot = int(ssm_state_indices[seq_idx].item())
                        else:
                            write_slot = int(ssm_state_indices[seq_idx, local_t].item())
                    else:
                        write_slot = bos + local_t
                elif flags["STORE_FINAL_STATE"]:
                    if local_t == T_seq - 1:
                        do_store_final_this_token = True

                if do_store_final_this_token:
                    ref_final_state[write_slot] = h_token_stored

    if not flags["IS_VARLEN"]:
        ref_out = ref_out.reshape(meta["B"], meta["T_fixed"], meta["H_v"], meta["V"])

    return ref_out, ref_final_state


def kda_test_with_reference(case: dict, *, rtol: float = 1e-2, atol: float = 1e-2):
    out, final_state = kda_test(case)
    ref_out, ref_final_state = _kda_reference_from_case(case)

    meta = case["meta"]
    flags = case["flags"]
    print("=" * 80)
    print("KDA TEST CONFIG")
    print("flags:")
    for key in sorted(flags.keys()):
        print(f"  {key}={flags[key]}")
    print("meta:")
    for key in sorted(meta.keys()):
        print(f"  {key}={meta[key]}")

    _summarize_diff("output", out, ref_out)

    if final_state is not None and ref_final_state is not None:
        _summarize_diff("final_state", final_state, ref_final_state)

    if out.shape != ref_out.shape:
        ref_out = ref_out.reshape(out.shape)

    torch.testing.assert_close(out.float(), ref_out, rtol=rtol, atol=atol)
    print("output pass")
    if final_state is not None and ref_final_state is not None:
        torch.testing.assert_close(
            final_state.float(), ref_final_state.float(), rtol=rtol, atol=atol
        )
        print("final_state pass")

    return out, final_state, ref_out, ref_final_state


@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
def test_kda_decode_bf16_state():
    smoke_cases = (
        {
            "IS_VARLEN": False,
            "USE_INITIAL_STATE": True,
            "INPLACE_FINAL_STATE": False,
            "IS_BETA_HEADWISE": False,
            "IS_CONTINUOUS_BATCHING": False,
            "STORE_FINAL_STATE": True,
            "STATE_V_FIRST": True,
            "B": 8,
            "T_fixed": 1,
        },
        {
            "IS_VARLEN": False,
            "USE_INITIAL_STATE": True,
            "INPLACE_FINAL_STATE": True,
            "IS_BETA_HEADWISE": False,
            "IS_CONTINUOUS_BATCHING": True,
            "STORE_FINAL_STATE": True,
            "STATE_V_FIRST": True,
            "B": 4,
            "T_fixed": 1,
        },
        {
            "IS_VARLEN": False,
            "USE_INITIAL_STATE": True,
            "INPLACE_FINAL_STATE": False,
            "IS_BETA_HEADWISE": True,
            "IS_CONTINUOUS_BATCHING": False,
            "STORE_FINAL_STATE": True,
            "STATE_V_FIRST": False,
            "T_fixed": 2,
        },
        {
            "IS_VARLEN": True,
            "USE_INITIAL_STATE": True,
            "INPLACE_FINAL_STATE": True,
            "IS_BETA_HEADWISE": False,
            "IS_CONTINUOUS_BATCHING": True,
            "STORE_FINAL_STATE": True,
            "STATE_V_FIRST": True,
            "T_fixed": 4,
        },
        {
            "IS_VARLEN": True,
            "USE_INITIAL_STATE": True,
            "INPLACE_FINAL_STATE": False,
            "IS_BETA_HEADWISE": True,
            "IS_CONTINUOUS_BATCHING": False,
            "STORE_FINAL_STATE": True,
            "STATE_V_FIRST": True,
            "B": 17,
            "T_fixed": 4,
        },
        {
            "IS_VARLEN": False,
            "USE_INITIAL_STATE": False,
            "INPLACE_FINAL_STATE": False,
            "IS_BETA_HEADWISE": False,
            "IS_CONTINUOUS_BATCHING": False,
            "STORE_FINAL_STATE": True,
            "STATE_V_FIRST": True,
            "T_fixed": 2,
        },
        {
            "IS_VARLEN": False,
            "USE_INITIAL_STATE": True,
            "INPLACE_FINAL_STATE": True,
            "IS_BETA_HEADWISE": False,
            "IS_CONTINUOUS_BATCHING": False,
            "STORE_FINAL_STATE": True,
            "STATE_V_FIRST": True,
            "B": 2,
            "T_fixed": 1,
            # The state pool capacity is independent of logical batch size.
            # This guards old_batch shape inference in the TileLang launch.
            "NUM_SLOT_STATES": 5,
        },
    )

    for overrides in smoke_cases:
        case_options = overrides.copy()
        batch = case_options.pop("B", 2)
        case = generate_kda_case(
            B=batch,
            H_qk=1,
            H_v=1,
            head_size=128,
            input_dtype=torch.float16,
            state_dtype=torch.bfloat16,
            gate_batch_dtype=torch.float16,
            output_dtype=torch.float32,
            device="musa",
            **case_options,
        )
        tail_before = None
        if "NUM_SLOT_STATES" in overrides:
            tail_before = case["tensors"]["initial_state / h0"][batch:].clone()
        _, final_state, _, _ = kda_test_with_reference(case, rtol=2e-2, atol=8e-2)
        assert final_state is not None
        assert final_state.dtype == torch.bfloat16
        if tail_before is not None:
            torch.testing.assert_close(final_state[batch:], tail_before)

    noncontiguous_case = generate_kda_case(
        IS_VARLEN=False,
        USE_INITIAL_STATE=True,
        INPLACE_FINAL_STATE=True,
        IS_BETA_HEADWISE=False,
        IS_CONTINUOUS_BATCHING=False,
        STORE_FINAL_STATE=True,
        STATE_V_FIRST=True,
        B=2,
        T_fixed=1,
        H_qk=1,
        H_v=1,
        head_size=128,
        input_dtype=torch.float16,
        state_dtype=torch.bfloat16,
        gate_batch_dtype=torch.float16,
        output_dtype=torch.float32,
        device="musa",
    )
    tensors = noncontiguous_case["tensors"]
    state = tensors["initial_state / h0"].transpose(-1, -2)
    assert not state.is_contiguous()
    tensors["initial_state / h0"] = state
    tensors["initial_state_ref"] = state.clone()
    tensors["final_state"] = state
    _, final_state, _, _ = kda_test_with_reference(
        noncontiguous_case, rtol=2e-2, atol=8e-2
    )
    assert final_state is state


@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
def test_kda_decode_fp32_varlen_large_batch_config():
    case = generate_kda_case(
        IS_VARLEN=True,
        USE_INITIAL_STATE=True,
        INPLACE_FINAL_STATE=False,
        IS_BETA_HEADWISE=False,
        IS_CONTINUOUS_BATCHING=False,
        STORE_FINAL_STATE=True,
        STATE_V_FIRST=True,
        B=17,
        T_fixed=4,
        H_qk=1,
        H_v=1,
        head_size=128,
        input_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        gate_batch_dtype=torch.float32,
        output_dtype=torch.float32,
        device="musa",
    )
    _, final_state, _, _ = kda_test_with_reference(case, rtol=1e-2, atol=1e-2)
    assert final_state is not None
    assert final_state.dtype == torch.float32


def test_kda_decode_bf16_autotune_configs():
    from mate.kda_kernels.tilelang.kda_decode import (
        _resolve_autotuned_kernel_config,
    )

    expected_v_first = {
        1: (16, 8),
        2: (16, 8),
        4: (16, 4),
        8: (16, 2),
        16: (16, 2),
        32: (16, 4),
        64: (16, 4),
        65: (16, 2),
        512: (16, 2),
    }
    for batch, expected in expected_v_first.items():
        config = _resolve_autotuned_kernel_config(batch, "bfloat16", True)
        assert (config["v_tile"], config["num_blocks_per_state"]) == expected

    expected_v_first_fallback = {
        (1, 64): (8, 8),
        (4, 32): (8, 4),
        (64, 32): (8, 4),
        (512, 64): (8, 2),
    }
    for (batch, dim_v), expected in expected_v_first_fallback.items():
        config = _resolve_autotuned_kernel_config(batch, "bfloat16", True, dim_v)
        assert (config["v_tile"], config["num_blocks_per_state"]) == expected

    expected_k_first = {
        2: (8, 8),
        4: (8, 4),
        16: (8, 2),
        64: (8, 4),
        512: (8, 4),
    }
    for batch, expected in expected_k_first.items():
        config = _resolve_autotuned_kernel_config(batch, "bfloat16", False)
        assert (config["v_tile"], config["num_blocks_per_state"]) == expected

    expected_fp32 = {
        2: (8, 8),
        4: (8, 4),
        16: (8, 2),
        17: (4, 4),
        512: (4, 4),
    }
    for batch, expected in expected_fp32.items():
        config = _resolve_autotuned_kernel_config(batch, "float32", True)
        assert (config["v_tile"], config["num_blocks_per_state"]) == expected

    expected_fp32_varlen = {
        16: (8, 2),
        17: (16, 8),
        64: (16, 8),
        512: (16, 8),
    }
    for batch, expected in expected_fp32_varlen.items():
        config = _resolve_autotuned_kernel_config(batch, "float32", True, 128, True)
        assert (config["v_tile"], config["num_blocks_per_state"]) == expected

    fp32_varlen_fallbacks = (
        _resolve_autotuned_kernel_config(512, "float32", False, 128, True),
        _resolve_autotuned_kernel_config(512, "float32", True, 64, True),
    )
    for config in fp32_varlen_fallbacks:
        assert (config["v_tile"], config["num_blocks_per_state"]) == (4, 8)

    expected_bf16_varlen = {
        16: (16, 2),
        17: (32, 4),
        64: (32, 4),
        512: (32, 4),
    }
    for batch, expected in expected_bf16_varlen.items():
        config = _resolve_autotuned_kernel_config(batch, "bfloat16", True, 128, True)
        assert (config["v_tile"], config["num_blocks_per_state"]) == expected

    bf16_varlen_fallbacks = (
        _resolve_autotuned_kernel_config(512, "bfloat16", False, 128, True),
        _resolve_autotuned_kernel_config(512, "bfloat16", True, 64, True),
    )
    assert (
        bf16_varlen_fallbacks[0]["v_tile"],
        bf16_varlen_fallbacks[0]["num_blocks_per_state"],
    ) == (8, 4)
    assert (
        bf16_varlen_fallbacks[1]["v_tile"],
        bf16_varlen_fallbacks[1]["num_blocks_per_state"],
    ) == (8, 2)


if __name__ == "__main__":
    test_kda_decode_bf16_state()
    qkv_dtype_options = [torch.float16, torch.bfloat16]
    gate_dtype_options = [torch.float16, torch.bfloat16, torch.float32]
    output_dtype_options = [torch.float16, torch.bfloat16, torch.float32]
    dt_bias_dtype_options = [torch.float32, torch.bfloat16]

    for B in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        for USE_QK_L2NORM_IN_KERNEL in [False, True]:
            for HAS_DT_BIAS in [False, True]:
                for USE_GATE_IN_KERNEL in [False, True]:
                    for IS_VARLEN in [False, True]:
                        for INPLACE_FINAL_STATE in [False, True]:
                            for STORE_FINAL_STATE in [False, True]:
                                for STATE_V_FIRST in [True, False]:
                                    for IS_CONTINUOUS_BATCHING in [False, True]:
                                        for input_dtype in qkv_dtype_options:
                                            for gate_batch_dtype in gate_dtype_options:
                                                for (
                                                    output_dtype
                                                ) in output_dtype_options:
                                                    dt_bias_dtype_candidates = (
                                                        dt_bias_dtype_options
                                                        if HAS_DT_BIAS
                                                        else [torch.float32]
                                                    )
                                                    for (
                                                        dt_bias_dtype
                                                    ) in dt_bias_dtype_candidates:
                                                        case = generate_kda_case(
                                                            IS_VARLEN=IS_VARLEN,
                                                            USE_INITIAL_STATE=True,
                                                            INPLACE_FINAL_STATE=INPLACE_FINAL_STATE,
                                                            IS_BETA_HEADWISE=False,
                                                            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
                                                            IS_CONTINUOUS_BATCHING=IS_CONTINUOUS_BATCHING,
                                                            IS_SPEC_DECODING=False,
                                                            STORE_FINAL_STATE=STORE_FINAL_STATE,
                                                            HAS_DT_BIAS=HAS_DT_BIAS,
                                                            USE_GATE_IN_KERNEL=USE_GATE_IN_KERNEL,
                                                            USE_LOWER_BOUND=False,
                                                            APPLY_BETA_SIGMOID=False,
                                                            ALLOW_NEG_EIGVAL=False,
                                                            STATE_V_FIRST=STATE_V_FIRST,
                                                            B=B,
                                                            T_fixed=4,
                                                            H_qk=16,
                                                            H_v=64,
                                                            head_size=128,
                                                            scale=0.5,
                                                            lower_bound_value=10.0,
                                                            input_dtype=input_dtype,
                                                            gate_batch_dtype=gate_batch_dtype,
                                                            dt_bias_dtype=dt_bias_dtype,
                                                            output_dtype=output_dtype,
                                                            device="musa",
                                                        )
                                                        kda_test_with_reference(case)
