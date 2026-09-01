from __future__ import annotations

import argparse

import torch

from mate.gdn_kernels.tilelang import gdn_mtp as mtp_backend
from mate.testing.utils import bench_kineto


DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DEFAULT_SEQ_LENS = (2, 3)
DEFAULT_HEAD_CONFIGS = (
    "2,8",
    "4,8",
    "16,16",
    "16,32",
    "16,48",
    "16,64",
)
KERNEL_NAME = "gated_deltanet_mtp_fp32_vk_smem"


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _parse_head_config(spec: str) -> tuple[int, int]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid head config {spec!r}. Expected 'Hq,Hv', for example '16,32'."
        )
    num_q_heads, num_v_heads = (int(part) for part in parts)
    if num_q_heads <= 0 or num_v_heads <= 0:
        raise ValueError(f"Head counts must be positive, got {spec!r}.")
    return num_q_heads, num_v_heads


def _gdn_mtp_flops(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
) -> int:
    num_o_heads = max(num_q_heads, num_v_heads)
    return 6 * batch_size * seq_len * num_o_heads * head_size * head_size


def _gdn_mtp_bytes(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    state_dtype: torch.dtype,
    *,
    disable_state_update: bool,
    cache_intermediate_states: bool,
) -> int:
    num_o_heads = max(num_q_heads, num_v_heads)
    elem_size = _dtype_size(input_dtype)
    state_elem_size = _dtype_size(state_dtype)

    q_bytes = batch_size * seq_len * num_q_heads * head_size * elem_size
    k_bytes = batch_size * seq_len * num_q_heads * head_size * elem_size
    v_bytes = batch_size * seq_len * num_v_heads * head_size * elem_size
    o_bytes = batch_size * seq_len * num_o_heads * head_size * _dtype_size(output_dtype)
    state_read_bytes = (
        batch_size * num_v_heads * head_size * head_size * state_elem_size
    )
    state_write_bytes = 0 if disable_state_update else state_read_bytes
    intermediate_bytes = (
        batch_size * seq_len * num_v_heads * head_size * head_size * state_elem_size
        if cache_intermediate_states
        else 0
    )
    A_log_bytes = num_v_heads * 4
    dt_bias_bytes = num_v_heads * elem_size
    a_bytes = batch_size * seq_len * num_v_heads * elem_size
    b_bytes = batch_size * seq_len * num_v_heads * elem_size

    return (
        q_bytes
        + k_bytes
        + v_bytes
        + o_bytes
        + state_read_bytes
        + state_write_bytes
        + intermediate_bytes
        + A_log_bytes
        + dt_bias_bytes
        + a_bytes
        + b_bytes
    )


def _make_inputs(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    *,
    cache_intermediate_states: bool,
) -> tuple[torch.Tensor, ...]:
    device = torch.device("musa")
    q = torch.randn(
        (batch_size, seq_len, num_q_heads, head_size), dtype=dtype, device=device
    )
    k = torch.randn(
        (batch_size, seq_len, num_q_heads, head_size), dtype=dtype, device=device
    )
    v = torch.randn(
        (batch_size, seq_len, num_v_heads, head_size), dtype=dtype, device=device
    )
    state = torch.randn(
        (batch_size, num_v_heads, head_size, head_size),
        dtype=state_dtype,
        device=device,
    )
    state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    A_log = torch.randn((num_v_heads,), dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn((num_v_heads,), dtype=dtype, device=device) * 0.1
    a = (
        torch.randn((batch_size, seq_len, num_v_heads), dtype=dtype, device=device)
        * 0.1
    )
    b = (
        torch.randn((batch_size, seq_len, num_v_heads), dtype=dtype, device=device)
        * 0.1
    )
    output = torch.empty(
        (batch_size, seq_len, num_v_heads, head_size), dtype=dtype, device=device
    )
    intermediate = (
        torch.empty(
            (batch_size, seq_len, num_v_heads, head_size, head_size),
            dtype=state_dtype,
            device=device,
        )
        if cache_intermediate_states
        else None
    )
    return q, k, v, state, state_indices, A_log, a, dt_bias, b, output, intermediate


def _make_runner(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    *,
    cache_intermediate_states: bool,
    disable_state_update: bool,
    tile_v_override: int | None,
    ilp_rows_override: int | None,
):
    q, k, v, state, state_indices, A_log, a, dt_bias, b, output, intermediate = (
        _make_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            num_q_heads=num_q_heads,
            num_v_heads=num_v_heads,
            head_size=head_size,
            dtype=dtype,
            state_dtype=state_dtype,
            cache_intermediate_states=cache_intermediate_states,
        )
    )

    default_tile_v, _, default_ilp_rows = mtp_backend._get_mtp_config(
        batch_size=batch_size,
        seq_len=seq_len,
        num_v_heads=num_v_heads,
        v_dim=head_size,
        cache_intermediate_states=cache_intermediate_states,
        state_dtype=state_dtype,
    )
    tile_v = default_tile_v if tile_v_override is None else tile_v_override
    ilp_rows = default_ilp_rows if ilp_rows_override is None else ilp_rows_override
    intermediate_arg = (
        intermediate
        if intermediate is not None
        else torch.empty((1, 1, 1, 1, 1), dtype=state_dtype, device=q.device)
    )
    common_kwargs = dict(
        seq_len=seq_len,
        qk_head=num_q_heads,
        head=num_v_heads,
        dim_k=head_size,
        dim_v=head_size,
        input_dtype=str(dtype).split(".")[-1],
        output_dtype=str(dtype).split(".")[-1],
        dt_bias_dtype=str(dt_bias.dtype).split(".")[-1],
        state_dtype=str(state_dtype).split(".")[-1],
        use_qk_l2norm=True,
        disable_state_update=disable_state_update,
        use_identity_state_indices=False,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
        disable_index_type_promotion=mtp_backend._should_disable_index_type_promotion(
            batch_size=batch_size,
            seq_len=seq_len,
            num_v_heads=num_v_heads,
            state_dtype=state_dtype,
            cache_intermediate_states=cache_intermediate_states,
            disable_state_update=disable_state_update,
        ),
    )
    kernel_fn = mtp_backend._get_mtp_fp32_vk_smem_kernel(
        cache_intermediate_states=cache_intermediate_states,
        **common_kwargs,
    )
    scale = head_size**-0.5

    def _runner():
        kernel_fn(
            q,
            k,
            v,
            A_log,
            a,
            dt_bias,
            b,
            float(scale),
            state,
            state_indices,
            intermediate_arg,
            output,
        )

    return _runner


def _bench_one_shape(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    num_tests: int,
    *,
    cache_intermediate_states: bool,
    disable_state_update: bool,
    tile_v_override: int | None,
    ilp_rows_override: int | None,
) -> float:
    runner = _make_runner(
        batch_size=batch_size,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        dtype=dtype,
        state_dtype=state_dtype,
        cache_intermediate_states=cache_intermediate_states,
        disable_state_update=disable_state_update,
        tile_v_override=tile_v_override,
        ilp_rows_override=ilp_rows_override,
    )
    seconds = bench_kineto(
        runner,
        kernel_names=KERNEL_NAME,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if seconds <= 0:
        raise RuntimeError(f"Failed to capture kernel time for {KERNEL_NAME}.")
    return float(seconds)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MATE GDN MTP kernels.")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--state-dtype",
        choices=["fp32", "bf16"],
        default="bf16",
        help="State and intermediate-state buffer dtype.",
    )
    parser.add_argument("--num-tests", type=int, default=10)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to benchmark. Default matches FlashInfer MTP tests.",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEQ_LENS),
        help="MTP sequence lengths. Default matches FlashInfer MTP tests.",
    )
    parser.add_argument(
        "--head-configs",
        nargs="+",
        default=list(DEFAULT_HEAD_CONFIGS),
        help="Head configs in 'Hq,Hv' form. Default is FlashInfer MTP 16,32.",
    )
    parser.add_argument(
        "--cache-intermediate-states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache intermediate states.",
    )
    parser.add_argument(
        "--update-state",
        action="store_true",
        help="Update final state. Default is read-only verify mode like FlashInfer tests.",
    )
    parser.add_argument(
        "--tile-v",
        type=int,
        default=None,
        help="Override the V tile size selected by the MTP config helper.",
    )
    parser.add_argument(
        "--ilp-rows",
        type=int,
        choices=(1, 2, 4, 8),
        default=None,
        help="Override the ILP rows selected by the MTP config helper.",
    )
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise RuntimeError("MUSA device is not available.")
    if args.tile_v is not None and args.tile_v <= 0:
        parser.error("--tile-v must be a positive integer.")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    state_dtype = torch.float32 if args.state_dtype == "fp32" else torch.bfloat16
    head_configs = [_parse_head_config(spec) for spec in args.head_configs]
    cache_intermediate_states = args.cache_intermediate_states
    disable_state_update = not args.update_state

    print(f"\nMUSA: {torch.musa.get_device_name(0)}")
    print(f"Kernel: {KERNEL_NAME}")
    print(
        f"Config: D={args.head_size}, dtype={args.dtype}, "
        f"state_dtype={args.state_dtype}, "
        f"head_configs={head_configs}, batches={tuple(args.batch_sizes)}, "
        f"seq_lens={tuple(args.seq_lens)}, cache_intermediate={cache_intermediate_states}, "
        f"update_state={not disable_state_update}, tile_v_override={args.tile_v}, "
        f"ilp_rows_override={args.ilp_rows}"
    )
    print()

    header = (
        f"{'Hq':>4s}  {'Hv':>4s}  {'B':>4s}  {'T':>3s}  "
        f"{'Latency(us)':>12s}  {'TFLOPS':>8s}  {'GB/s':>8s}"
    )
    print(header)
    print("-" * len(header))

    for num_q_heads, num_v_heads in head_configs:
        for seq_len in args.seq_lens:
            for batch_size in args.batch_sizes:
                seconds = _bench_one_shape(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_q_heads=num_q_heads,
                    num_v_heads=num_v_heads,
                    head_size=args.head_size,
                    dtype=dtype,
                    state_dtype=state_dtype,
                    num_tests=args.num_tests,
                    cache_intermediate_states=cache_intermediate_states,
                    disable_state_update=disable_state_update,
                    tile_v_override=args.tile_v,
                    ilp_rows_override=args.ilp_rows,
                )
                flops = _gdn_mtp_flops(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_q_heads=num_q_heads,
                    num_v_heads=num_v_heads,
                    head_size=args.head_size,
                )
                io_bytes = _gdn_mtp_bytes(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_q_heads=num_q_heads,
                    num_v_heads=num_v_heads,
                    head_size=args.head_size,
                    input_dtype=dtype,
                    output_dtype=dtype,
                    state_dtype=state_dtype,
                    disable_state_update=disable_state_update,
                    cache_intermediate_states=cache_intermediate_states,
                )
                latency_us = seconds * 1e6
                tflops = flops / seconds / 1e12
                bandwidth = io_bytes / seconds / 1e9
                print(
                    f"{num_q_heads:>4d}  {num_v_heads:>4d}  "
                    f"{batch_size:>4d}  {seq_len:>3d}  {latency_us:>12.3f}  "
                    f"{tflops:>8.3f}  {bandwidth:>8.3f}"
                )
                torch.musa.empty_cache()


if __name__ == "__main__":
    main()
