from __future__ import annotations

import argparse

import torch

from mate.kda_kernels.tilelang.kda_decode import run_gated_delta_rule_decode_vk_fp32
from mate.testing.utils import bench_kineto

HEAD_CONFIGS = [
    # (h_qk, h_v, d)
    (2, 8, 128),
    (4, 16, 128),
    (8, 32, 128),
    (16, 64, 128),
    (16, 32, 128),
    (16, 48, 128),
    (16, 16, 128),
    (32, 32, 128),
]

DECODE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DEFAULT_INPUT_DTYPES = ("fp16",)
DEFAULT_STATE_DTYPES = ("fp32",)
DEFAULT_SEQLEN_MODE = "fixed"
DEFAULT_FIXED_T = 1
KERNEL_NAME = "gated_deltanet_kda_decode"


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _parse_dtype(spec: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    try:
        return mapping[spec]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype spec {spec!r}.") from exc


def _str2bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value {value!r}. Use true/false, 1/0, yes/no, on/off."
    )


def _kda_decode_flops(
    *,
    total_tokens: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
) -> int:
    num_o_heads = max(num_q_heads, num_v_heads)
    return 6 * total_tokens * num_o_heads * head_size * head_size


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _kda_decode_bytes(
    *,
    tensors: dict[str, object],
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    state_dtype: torch.dtype = torch.float32,
    gate_batch_dtype: torch.dtype,
    dt_bias_dtype: torch.dtype,
    b_dtype: torch.dtype,
    output_dtype: torch.dtype,
    has_dt_bias: bool,
    use_initial_state: bool,
    store_final_state: bool,
    inplace_final_state: bool,
    is_beta_headwise: bool,
    is_varlen: bool,
    is_continuous_batching: bool,
    is_spec_decoding: bool,
    state_v_first: bool,
) -> int:
    total_tokens = int(tensors["total_tokens"])
    num_sequences = int(tensors["num_sequences"])
    elem_size = _dtype_size(input_dtype)

    q_bytes = total_tokens * num_q_heads * head_size * elem_size
    k_bytes = total_tokens * num_q_heads * head_size * elem_size
    v_bytes = total_tokens * num_v_heads * head_size * elem_size
    a_bytes = total_tokens * num_v_heads * head_size * _dtype_size(gate_batch_dtype)
    b_width = head_size if is_beta_headwise else 1
    b_bytes = total_tokens * num_v_heads * b_width * _dtype_size(b_dtype)
    o_bytes = total_tokens * num_v_heads * head_size * _dtype_size(output_dtype)

    state_slot_bytes = num_v_heads * head_size * head_size * _dtype_size(state_dtype)
    if use_initial_state:
        state_bytes = num_sequences * state_slot_bytes
    else:
        state_bytes = 0

    if is_continuous_batching:
        # Continuous decode persists every token state, independent of the
        # store_final_state flag and whether the destination is in-place.
        final_state_bytes = total_tokens * state_slot_bytes
    elif store_final_state:
        touched_final_slots = num_sequences
        final_state_bytes = touched_final_slots * state_slot_bytes
    else:
        final_state_bytes = 0

    A_log_bytes = num_v_heads * _dtype_size(torch.float32)
    dt_bias_bytes = (
        num_v_heads * head_size * _dtype_size(dt_bias_dtype) if has_dt_bias else 0
    )
    cu_seqlens_bytes = (
        (num_sequences + 1) * _dtype_size(torch.int32) if is_varlen else 0
    )
    state_indices_bytes = 0
    if is_continuous_batching:
        if is_spec_decoding:
            state_indices_bytes = total_tokens * _dtype_size(torch.int32)
        elif is_varlen:
            state_indices_bytes = total_tokens * _dtype_size(torch.int32)
        else:
            state_indices_bytes = num_sequences * _dtype_size(torch.int32)
    num_accepted_tokens_bytes = (
        num_sequences * _dtype_size(torch.int32)
        if tensors["num_accepted_tokens"] is not None
        else 0
    )

    return (
        q_bytes
        + k_bytes
        + v_bytes
        + a_bytes
        + b_bytes
        + o_bytes
        + state_bytes
        + final_state_bytes
        + A_log_bytes
        + dt_bias_bytes
        + cu_seqlens_bytes
        + state_indices_bytes
        + num_accepted_tokens_bytes
    )


def _build_varlen_seqlens(
    batch_size: int,
    *,
    device: torch.device,
    randomize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    choices = torch.tensor([1, 4], device=device, dtype=torch.int32)
    if randomize:
        indices = torch.randint(0, choices.numel(), (batch_size,), device=device)
    else:
        indices = torch.arange(batch_size, device=device) % choices.numel()
    seqlens = choices[indices]
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(seqlens, dim=0),
        ]
    )
    return seqlens, cu_seqlens


def _sample_discrete(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return ((torch.rand(shape, device=device, dtype=torch.float32) * 2.0) - 1.0).to(
        dtype=dtype
    )


def _make_case(
    *,
    batch_size: int,
    time_steps: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    state_dtype: torch.dtype = torch.float32,
    gate_batch_dtype: torch.dtype,
    dt_bias_dtype: torch.dtype,
    output_dtype: torch.dtype,
    is_varlen: bool,
    has_dt_bias: bool,
    use_initial_state: bool,
    store_final_state: bool,
    inplace_final_state: bool,
    is_beta_headwise: bool,
    is_continuous_batching: bool,
    is_spec_decoding: bool,
    state_v_first: bool,
    randomize_varlen: bool,
    seed: int,
) -> dict[str, object]:
    device = torch.device("musa")
    torch.manual_seed(seed)

    if is_varlen:
        seqlens, cu_seqlens = _build_varlen_seqlens(
            batch_size,
            device=device,
            randomize=randomize_varlen,
        )
        total_tokens = int(seqlens.sum().item())
        max_tokens = int(seqlens.max().item())
        q_shape = (1, total_tokens, num_q_heads, head_size)
        k_shape = (1, total_tokens, num_q_heads, head_size)
        v_shape = (1, total_tokens, num_v_heads, head_size)
        beta_shape = (
            (1, total_tokens, num_v_heads, head_size)
            if is_beta_headwise
            else (1, total_tokens, num_v_heads)
        )
        logical_time_steps = total_tokens
    else:
        seqlens = torch.full(
            (batch_size,), time_steps, device=device, dtype=torch.int32
        )
        cu_seqlens = None
        total_tokens = batch_size * time_steps
        max_tokens = time_steps
        q_shape = (batch_size, time_steps, num_q_heads, head_size)
        k_shape = (batch_size, time_steps, num_q_heads, head_size)
        v_shape = (batch_size, time_steps, num_v_heads, head_size)
        beta_shape = (
            (batch_size, time_steps, num_v_heads, head_size)
            if is_beta_headwise
            else (batch_size, time_steps, num_v_heads)
        )
        logical_time_steps = time_steps

    q = _sample_discrete(q_shape, dtype=input_dtype, device=device)
    k = _sample_discrete(k_shape, dtype=input_dtype, device=device)
    v = _sample_discrete(v_shape, dtype=input_dtype, device=device)
    a = _sample_discrete(
        (*v_shape[:-1], head_size), dtype=gate_batch_dtype, device=device
    )
    b = _sample_discrete(beta_shape, dtype=output_dtype, device=device)
    A_log = _sample_discrete((num_v_heads,), dtype=torch.float32, device=device)
    dt_bias = (
        _sample_discrete((num_v_heads, head_size), dtype=dt_bias_dtype, device=device)
        if has_dt_bias
        else None
    )

    state_matrix_shape = (
        (head_size, head_size) if state_v_first else (head_size, head_size)
    )

    if is_continuous_batching:
        num_slot_states = max(total_tokens * 3, batch_size * 3)
        state_shape = (num_slot_states, num_v_heads, *state_matrix_shape)
    else:
        num_slot_states = batch_size
        state_shape = (batch_size, num_v_heads, *state_matrix_shape)

    state = (
        (_sample_discrete(state_shape, dtype=torch.float32, device=device) + 3.0).to(
            dtype=state_dtype
        )
        if use_initial_state
        else torch.empty(state_shape, dtype=state_dtype, device=device)
    )

    state_indices = None
    num_accepted_tokens = None
    if is_continuous_batching:
        if is_varlen:
            state_indices = torch.empty(
                (batch_size, max_tokens), device=device, dtype=torch.int32
            )
            cursor = 0
            for batch_idx, seq_len in enumerate(seqlens.tolist()):
                state_indices[batch_idx, :seq_len] = torch.arange(
                    cursor, cursor + seq_len, device=device, dtype=torch.int32
                )
                if seq_len < max_tokens:
                    state_indices[batch_idx, seq_len:] = 0
                cursor += seq_len
        else:
            if time_steps == 1:
                state_indices = torch.randperm(num_slot_states, device=device)[
                    :batch_size
                ].to(torch.int32)[:, None]
            else:
                state_indices = torch.arange(
                    batch_size * time_steps, device=device, dtype=torch.int32
                ).reshape(batch_size, time_steps)

        if is_spec_decoding:
            if is_varlen:
                num_accepted_tokens = torch.randint(
                    1,
                    max_tokens + 1,
                    (batch_size,),
                    device=device,
                    dtype=torch.int32,
                )
                num_accepted_tokens = torch.minimum(num_accepted_tokens, seqlens)
            else:
                num_accepted_tokens = torch.randint(
                    1, time_steps + 1, (batch_size,), device=device, dtype=torch.int32
                )

    final_state = None
    if store_final_state:
        if is_continuous_batching:
            if inplace_final_state:
                final_state = state
            else:
                final_state_shape = (total_tokens, num_v_heads, *state_matrix_shape)
                final_state = torch.empty(
                    final_state_shape, dtype=state_dtype, device=device
                )
        else:
            if inplace_final_state:
                final_state = state
            else:
                final_state_shape = (batch_size, num_v_heads, *state_matrix_shape)
                final_state = torch.empty(
                    final_state_shape, dtype=state_dtype, device=device
                )

    output = torch.empty(v_shape, dtype=output_dtype, device=device)

    return {
        "q": q,
        "k": k,
        "v": v,
        "state": state,
        "final_state": final_state,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
        "cu_seqlens": cu_seqlens,
        "state_indices": state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "output": output,
        "seqlens": seqlens,
        "logical_time_steps": logical_time_steps,
        "total_tokens": total_tokens,
        "num_sequences": batch_size,
    }


def _bench_one_case(
    *,
    batch_size: int,
    time_steps: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    state_dtype: torch.dtype = torch.float32,
    gate_batch_dtype: torch.dtype,
    dt_bias_dtype: torch.dtype,
    output_dtype: torch.dtype,
    num_tests: int,
    use_qk_l2norm: bool,
    has_dt_bias: bool,
    use_gate_in_kernel: bool,
    is_varlen: bool,
    use_initial_state: bool,
    inplace_final_state: bool,
    is_beta_headwise: bool,
    is_continuous_batching: bool,
    is_spec_decoding: bool,
    store_final_state: bool,
    state_v_first: bool,
    use_lower_bound: bool,
    apply_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    randomize_varlen: bool,
    seed: int,
    verbose_dispatch: bool,
) -> tuple[float, dict[str, object]]:
    if inplace_final_state and not store_final_state:
        raise ValueError("inplace_final_state=True requires store_final_state=True.")
    if use_lower_bound and not use_gate_in_kernel:
        raise ValueError("use_lower_bound=True requires use_gate_in_kernel=True.")
    if allow_neg_eigval and not apply_beta_sigmoid:
        raise ValueError("allow_neg_eigval=True requires apply_beta_sigmoid=True.")
    if is_spec_decoding and not is_continuous_batching:
        raise ValueError("is_spec_decoding=True requires is_continuous_batching=True.")

    tensors = _make_case(
        batch_size=batch_size,
        time_steps=time_steps,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        input_dtype=input_dtype,
        state_dtype=state_dtype,
        gate_batch_dtype=gate_batch_dtype,
        dt_bias_dtype=dt_bias_dtype,
        output_dtype=output_dtype,
        is_varlen=is_varlen,
        has_dt_bias=has_dt_bias,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        inplace_final_state=inplace_final_state,
        is_beta_headwise=is_beta_headwise,
        is_continuous_batching=is_continuous_batching,
        is_spec_decoding=is_spec_decoding,
        state_v_first=state_v_first,
        randomize_varlen=randomize_varlen,
        seed=seed,
    )

    if verbose_dispatch:
        print(
            "[dispatch] direct tilelang kda decode "
            f"B={batch_size} T={time_steps} Hq={num_q_heads} Hv={num_v_heads} D={head_size} "
            f"dtype={input_dtype} state_dtype={state_dtype} varlen={is_varlen} "
            f"total_tokens={tensors['total_tokens']} "
            f"dt_bias={has_dt_bias} init_state={use_initial_state} store_final_state={store_final_state} "
            f"inplace={inplace_final_state} beta_headwise={is_beta_headwise} cb={is_continuous_batching} "
            f"spec={is_spec_decoding} l2={use_qk_l2norm} gate={use_gate_in_kernel} lower={use_lower_bound} "
            f"beta_sigmoid={apply_beta_sigmoid} neg_eig={allow_neg_eigval} svf={state_v_first}"
        )

    def _runner():
        run_gated_delta_rule_decode_vk_fp32(
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            state=tensors["state"],
            state_indices=tensors["state_indices"],
            A_log=tensors["A_log"],
            g=tensors["a"],
            dt_bias=tensors["dt_bias"],
            b=tensors["b"],
            cu_seqlens=tensors["cu_seqlens"],
            num_accepted_tokens=tensors["num_accepted_tokens"],
            output=tensors["output"],
            scale=head_size**-0.5,
            lower_bound=0.0,
            use_qk_l2norm=use_qk_l2norm,
            is_varlen=is_varlen,
            inplace_final_state=inplace_final_state,
            is_beta_headwise=is_beta_headwise,
            is_continuous_batching=is_continuous_batching,
            is_spec_decoding=is_spec_decoding,
            store_final_state=store_final_state,
            has_dt_bias=has_dt_bias,
            use_gate_in_kernel=use_gate_in_kernel,
            use_lower_bound=use_lower_bound,
            apply_beta_sigmoid=apply_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            state_v_first=state_v_first,
            use_initial_state=use_initial_state,
        )

    seconds = bench_kineto(
        _runner,
        kernel_names=KERNEL_NAME,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if seconds <= 0:
        raise RuntimeError(f"Failed to capture kernel time for {KERNEL_NAME}.")
    return float(seconds), tensors


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MATE KDA decode kernel with broader shape coverage."
    )
    parser.add_argument(
        "--input-dtypes",
        nargs="+",
        choices=["fp16", "bf16", "fp32"],
        default=list(DEFAULT_INPUT_DTYPES),
        help="Input dtypes to benchmark.",
    )
    parser.add_argument(
        "--state-dtype",
        "--state-dtypes",
        dest="state_dtypes",
        nargs="+",
        choices=["bf16", "fp32"],
        default=list(DEFAULT_STATE_DTYPES),
        help="State storage dtypes to benchmark.",
    )
    parser.add_argument("--num-tests", type=int, default=10)
    parser.add_argument(
        "--seqlen-mode",
        choices=["fixed", "varlen", "both"],
        default=DEFAULT_SEQLEN_MODE,
        help="Benchmark fixed decode T or test_kda-style varlen shapes, or both.",
    )
    parser.add_argument(
        "--fixed-t",
        type=int,
        default=DEFAULT_FIXED_T,
        help="Fixed decode time steps when --seqlen-mode includes fixed.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DECODE_BATCH_SIZES),
        help="Batch sizes to benchmark for each head config.",
    )
    parser.add_argument(
        "--randomize-varlen",
        action="store_true",
        help="Use randomized [1,4] varlen pattern like test_kda instead of deterministic alternation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20262423,
        help="Random seed used for test_kda-style input construction.",
    )
    parser.add_argument(
        "--use-qk-l2norm",
        type=_str2bool,
        default=True,
        help="Whether to enable q/k l2norm inside kernel.",
    )
    parser.add_argument(
        "--has-dt-bias",
        type=_str2bool,
        default=True,
        help="Whether dt_bias tensor is present.",
    )
    parser.add_argument(
        "--use-gate-in-kernel",
        type=_str2bool,
        default=True,
        help="Whether to apply gate transformation inside kernel.",
    )
    parser.add_argument(
        "--use-initial-state",
        type=_str2bool,
        default=True,
        help="Whether to read initial state.",
    )
    parser.add_argument(
        "--store-final-state",
        type=_str2bool,
        default=True,
        help="Whether to write final state.",
    )
    parser.add_argument(
        "--inplace-final-state",
        type=_str2bool,
        default=False,
        help="Whether final state is written in place into the state buffer.",
    )
    parser.add_argument(
        "--is-beta-headwise",
        type=_str2bool,
        default=False,
        help="Whether beta uses per-value-dimension shape.",
    )
    parser.add_argument(
        "--is-continuous-batching",
        type=_str2bool,
        default=False,
        help="Whether to use continuous batching mode.",
    )
    parser.add_argument(
        "--is-spec-decoding",
        type=_str2bool,
        default=False,
        help="Whether to use speculative decoding mode.",
    )
    parser.add_argument(
        "--state-v-first",
        type=_str2bool,
        default=True,
        help="Whether state layout is V-first.",
    )
    parser.add_argument(
        "--use-lower-bound",
        type=_str2bool,
        default=False,
        help="Whether to enable lower-bound gate branch.",
    )
    parser.add_argument(
        "--apply-beta-sigmoid",
        type=_str2bool,
        default=False,
        help="Whether to apply sigmoid to beta.",
    )
    parser.add_argument(
        "--allow-neg-eigval",
        type=_str2bool,
        default=False,
        help="Whether beta sigmoid branch allows negative eigenvalues.",
    )
    parser.add_argument(
        "--verbose-dispatch",
        action="store_true",
        help="Print one-line dispatch diagnostics before each benchmarked shape.",
    )
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise RuntimeError("MUSA device is not available.")
    if args.fixed_t <= 0:
        raise ValueError("--fixed-t must be positive.")

    input_dtypes = [_parse_dtype(spec) for spec in args.input_dtypes]
    state_dtypes = [_parse_dtype(spec) for spec in args.state_dtypes]
    selected_head_configs = list(HEAD_CONFIGS)

    seqlen_modes = (
        ("fixed", "varlen") if args.seqlen_mode == "both" else (args.seqlen_mode,)
    )

    print(f"\nMUSA: {torch.musa.get_device_name(0)}")
    print(f"Kernel: {KERNEL_NAME}")
    print(
        "Dispatch target: mate.kda_kernels.tilelang.kda_decode.run_gated_delta_rule_decode_vk_fp32"
    )
    print(
        f"Config: input_dtypes={tuple(args.input_dtypes)}, state_dtypes={tuple(args.state_dtypes)}, "
        f"seqlen_mode={args.seqlen_mode}, "
        f"fixed_t={args.fixed_t}, batches={tuple(args.batch_sizes)}, seed={args.seed}, "
        f"l2={args.use_qk_l2norm}, dt_bias={args.has_dt_bias}, gate={args.use_gate_in_kernel}, "
        f"init_state={args.use_initial_state}, store_state={args.store_final_state}, "
        f"inplace_state={args.inplace_final_state}, beta_headwise={args.is_beta_headwise}, "
        f"cb={args.is_continuous_batching}, spec={args.is_spec_decoding}, svf={args.state_v_first}, "
        f"lower_bound={args.use_lower_bound}, beta_sigmoid={args.apply_beta_sigmoid}, "
        f"neg_eig={args.allow_neg_eigval}"
    )
    print()

    header = (
        f"{'Hq':>4s}  {'Hv':>4s}  {'D':>4s}  {'mode':<6s}  {'B':>4s}  {'T':>4s}  "
        f"{'Tok':>5s}  {'qkv':>5s}  {'state':>5s}  {'Latency(us)':>12s}  {'TFLOPS':>8s}  {'GB/s':>8s}"
    )
    print(header)
    print("-" * len(header))

    for num_q_heads, num_v_heads, head_size in selected_head_configs:
        for batch_size in args.batch_sizes:
            for seqlen_mode in seqlen_modes:
                is_varlen = seqlen_mode == "varlen"
                time_steps = args.fixed_t
                for input_dtype in input_dtypes:
                    for state_dtype in state_dtypes:
                        seconds, tensors = _bench_one_case(
                            batch_size=batch_size,
                            time_steps=time_steps,
                            num_q_heads=num_q_heads,
                            num_v_heads=num_v_heads,
                            head_size=head_size,
                            input_dtype=input_dtype,
                            state_dtype=state_dtype,
                            gate_batch_dtype=torch.float32,
                            dt_bias_dtype=torch.float32,
                            output_dtype=torch.float32,
                            num_tests=args.num_tests,
                            use_qk_l2norm=args.use_qk_l2norm,
                            has_dt_bias=args.has_dt_bias,
                            use_gate_in_kernel=args.use_gate_in_kernel,
                            is_varlen=is_varlen,
                            use_initial_state=args.use_initial_state,
                            inplace_final_state=args.inplace_final_state,
                            is_beta_headwise=args.is_beta_headwise,
                            is_continuous_batching=args.is_continuous_batching,
                            is_spec_decoding=args.is_spec_decoding,
                            store_final_state=args.store_final_state,
                            state_v_first=args.state_v_first,
                            use_lower_bound=args.use_lower_bound,
                            apply_beta_sigmoid=args.apply_beta_sigmoid,
                            allow_neg_eigval=args.allow_neg_eigval,
                            randomize_varlen=args.randomize_varlen,
                            seed=args.seed,
                            verbose_dispatch=args.verbose_dispatch,
                        )
                        flops = _kda_decode_flops(
                            total_tokens=int(tensors["total_tokens"]),
                            num_q_heads=num_q_heads,
                            num_v_heads=num_v_heads,
                            head_size=head_size,
                        )
                        io_bytes = _kda_decode_bytes(
                            tensors=tensors,
                            num_q_heads=num_q_heads,
                            num_v_heads=num_v_heads,
                            head_size=head_size,
                            input_dtype=input_dtype,
                            state_dtype=state_dtype,
                            gate_batch_dtype=torch.float32,
                            dt_bias_dtype=torch.float32,
                            b_dtype=torch.float32,
                            output_dtype=torch.float32,
                            has_dt_bias=args.has_dt_bias,
                            use_initial_state=args.use_initial_state,
                            store_final_state=args.store_final_state,
                            inplace_final_state=args.inplace_final_state,
                            is_beta_headwise=args.is_beta_headwise,
                            is_varlen=is_varlen,
                            is_continuous_batching=args.is_continuous_batching,
                            is_spec_decoding=args.is_spec_decoding,
                            state_v_first=args.state_v_first,
                        )
                        latency_us = seconds * 1e6
                        tflops = flops / seconds / 1e12
                        bandwidth = io_bytes / seconds / 1e9
                        dtype_label = str(input_dtype).split(".")[-1]
                        state_dtype_label = str(state_dtype).split(".")[-1]
                        print(
                            f"{num_q_heads:>4d}  {num_v_heads:>4d}  {head_size:>4d}  {seqlen_mode:<6s}  "
                            f"{batch_size:>4d}  {int(tensors['logical_time_steps']):>4d}  {int(tensors['total_tokens']):>5d}  "
                            f"{dtype_label:>5s}  {state_dtype_label:>5s}  {latency_us:>12.3f}  "
                            f"{tflops:>8.3f}  {bandwidth:>8.3f}"
                        )
                        torch.musa.empty_cache()
        print()


if __name__ == "__main__":
    main()
