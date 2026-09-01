from __future__ import annotations

import argparse

import torch

import mate

from mate.testing.utils import bench_kineto


DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DEFAULT_HEAD_CONFIGS = ("2,8", "4,8", "16, 16", "16,32", "16, 48", "16,64")
KERNEL_NAME = "gated_deltanet_decode_fp32_vk"


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


def _gdn_decode_flops(
    batch_size: int, num_q_heads: int, num_v_heads: int, head_size: int
) -> int:
    num_o_heads = max(num_q_heads, num_v_heads)
    return 6 * batch_size * num_o_heads * head_size * head_size


def _gdn_decode_bytes(
    batch_size: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    use_cache_indices: bool,
) -> int:
    num_o_heads = max(num_q_heads, num_v_heads)
    elem_size = _dtype_size(input_dtype)

    q_bytes = batch_size * num_q_heads * head_size * elem_size
    k_bytes = batch_size * num_q_heads * head_size * elem_size
    v_bytes = batch_size * num_v_heads * head_size * elem_size
    o_bytes = batch_size * num_o_heads * head_size * _dtype_size(output_dtype)
    state_bytes = 2 * batch_size * num_v_heads * head_size * head_size * 4
    A_log_bytes = num_v_heads * 4
    dt_bias_bytes = num_v_heads * 4
    a_bytes = batch_size * num_v_heads * elem_size
    b_bytes = batch_size * num_v_heads * elem_size
    cache_indices_bytes = batch_size * 4 if use_cache_indices else 0

    return (
        q_bytes
        + k_bytes
        + v_bytes
        + o_bytes
        + state_bytes
        + A_log_bytes
        + dt_bias_bytes
        + a_bytes
        + b_bytes
        + cache_indices_bytes
    )


def _make_inputs(
    batch_size: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_cache_indices: bool,
) -> tuple[torch.Tensor, ...]:
    device = torch.device("musa")
    q = torch.randn((batch_size, 1, num_q_heads, head_size), dtype=dtype, device=device)
    k = torch.randn((batch_size, 1, num_q_heads, head_size), dtype=dtype, device=device)
    v = torch.randn((batch_size, 1, num_v_heads, head_size), dtype=dtype, device=device)
    state = torch.randn(
        (batch_size, num_v_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )
    A_log = torch.randn((num_v_heads,), dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn((num_v_heads,), dtype=torch.float32, device=device) * 0.1
    a = torch.randn((batch_size, 1, num_v_heads), dtype=dtype, device=device) * 0.1
    b = torch.randn((batch_size, 1, num_v_heads), dtype=dtype, device=device)
    state_indices = (
        torch.arange(batch_size, dtype=torch.int32, device=device)
        if use_cache_indices
        else None
    )
    return q, k, v, state, A_log, a, dt_bias, b, state_indices


def _bench_one_shape(
    batch_size: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    num_tests: int,
    use_cache_indices: bool,
) -> float:
    q, k, v, state, A_log, a, dt_bias, b, state_indices = _make_inputs(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        dtype=dtype,
        use_cache_indices=use_cache_indices,
    )

    def _runner():
        mate.gated_delta_rule_decode(
            q=q,
            k=k,
            v=v,
            state=state,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            state_layout="VK",
            state_indices=state_indices,
            scale=None,
            use_qk_l2norm=True,
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
    return float(seconds)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MATE GDN decode kernel.")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--num-tests", type=int, default=10)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to benchmark. Default follows FlashInfer/SGLang decode sweeps.",
    )
    parser.add_argument(
        "--head-configs",
        nargs="+",
        default=list(DEFAULT_HEAD_CONFIGS),
        help="Head configs in 'Hq,Hv' form. Default covers 8,16 / 16,32 / 16,64.",
    )
    parser.add_argument(
        "--cache-indices",
        choices=["none", "identity", "both"],
        default="none",
        help="Benchmark without cache_indices, with identity cache_indices, or both.",
    )
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise RuntimeError("MUSA device is not available.")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    head_configs = [_parse_head_config(spec) for spec in args.head_configs]
    cache_index_modes = (
        ("none", "identity") if args.cache_indices == "both" else (args.cache_indices,)
    )

    print(f"\nMUSA: {torch.musa.get_device_name(0)}")
    print(f"Kernel: {KERNEL_NAME}")
    print(
        f"Config: D={args.head_size}, dtype={args.dtype}, "
        f"head_configs={head_configs}, batches={tuple(args.batch_sizes)}, "
        f"cache_indices={args.cache_indices}"
    )
    print()

    header = (
        f"{'Hq':>4s}  {'Hv':>4s}  {'B':>4s}  {'cache':>8s}  "
        f"{'Latency(us)':>12s}  {'TFLOPS':>8s}  {'GB/s':>8s}"
    )
    print(header)
    print("-" * len(header))

    for num_q_heads, num_v_heads in head_configs:
        for batch_size in args.batch_sizes:
            for cache_mode in cache_index_modes:
                use_cache_indices = cache_mode == "identity"
                seconds = _bench_one_shape(
                    batch_size=batch_size,
                    num_q_heads=num_q_heads,
                    num_v_heads=num_v_heads,
                    head_size=args.head_size,
                    dtype=dtype,
                    num_tests=args.num_tests,
                    use_cache_indices=use_cache_indices,
                )
                flops = _gdn_decode_flops(
                    batch_size=batch_size,
                    num_q_heads=num_q_heads,
                    num_v_heads=num_v_heads,
                    head_size=args.head_size,
                )
                io_bytes = _gdn_decode_bytes(
                    batch_size=batch_size,
                    num_q_heads=num_q_heads,
                    num_v_heads=num_v_heads,
                    head_size=args.head_size,
                    input_dtype=dtype,
                    output_dtype=dtype,
                    use_cache_indices=use_cache_indices,
                )
                latency_us = seconds * 1e6
                tflops = flops / seconds / 1e12
                bandwidth = io_bytes / seconds / 1e9
                print(
                    f"{num_q_heads:>4d}  {num_v_heads:>4d}  {batch_size:>4d}  "
                    f"{cache_mode:>8s}  {latency_us:>12.3f}  "
                    f"{tflops:>8.3f}  {bandwidth:>8.3f}"
                )
                torch.musa.empty_cache()


if __name__ == "__main__":
    main()
