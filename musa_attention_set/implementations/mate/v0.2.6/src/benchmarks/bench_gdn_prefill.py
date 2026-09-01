import argparse

import numpy as np
import torch

from mate.gdn_prefill import chunk_gated_delta_rule
from mate.testing.utils import bench_kineto

HEAD_CONFIGS = [
    # (h_qk, h_v, d, label)
    # Qwen3.5-397B and 122B (h_k=16, h_v=64, d=128) under different TP
    (2, 8, 128, "397B/122B TP8"),
    (4, 16, 128, "397B/122B TP4"),
    (8, 32, 128, "397B/122B TP2"),
    (16, 64, 128, "397B/122B TP1"),
    # Qwen3.5-35B, 9B and 4B (h_k=16, h_v=32, d=128)
    (16, 32, 128, "35B/9B/4B TP1"),
    # Qwen3.5-27B (h_k=16, h_v=48, d=128)
    (16, 48, 128, "27B TP1"),
    # Qwen3.5-2B and 0.8B (h_k=16, h_v=16, d=128)
    (16, 16, 128, "2B/0.8B TP1"),
    # Symmetric heads
    (32, 32, 128, "Sym h32"),
]

SEQ_CONFIGS = [
    # (cu_seqlen_endpoints, label)
    # endpoints are cumulative positions (leading 0 added automatically)
    ((8192,), "1x8192"),
    ((4096,), "1x4096"),
    ((2048,), "1x2048"),
    ((1024 * 6, 8192), "6144+2048"),
    ((1024 * 4, 8192), "4096+4096"),
    ((1024 * 2, 8192), "2048+6144"),
    ((1024 * 1, 8192), "1024+7168"),
    ((2048, 2048 * 2, 2048 * 3, 8192), "2048x4"),
    (tuple(1024 * (i + 1) for i in range(8)), "1024x8"),
]


def _gdn_tflops(total_tokens: int, h_o: int, d: int, time_ms: float) -> float:
    # 2 GEMMs (kv outer product + q@state), MAC counted as 2 FLOPs.
    flops = 2 * 2 * total_tokens * h_o * d * d
    return flops / time_ms / 1e9


def bench_mate(
    endpoints: tuple[int, ...],
    h_qk: int,
    h_v: int,
    d: int,
    dtype: torch.dtype,
    num_tests: int,
) -> float:
    device = "musa"
    num_seqs = len(endpoints)
    total_tokens = endpoints[-1]
    head_o = max(h_qk, h_v)

    cu_seqlens = torch.tensor([0] + list(endpoints), dtype=torch.int32, device=device)

    mixed_qkv = torch.randn(
        (1, total_tokens, (h_qk + h_qk + h_v) * d),
        dtype=dtype,
        device=device,
    )
    q, k, v = torch.split(mixed_qkv, [h_qk * d, h_qk * d, h_v * d], dim=-1)
    q = q.view(1, total_tokens, h_qk, d)
    k = k.view(1, total_tokens, h_qk, d)
    v = v.view(1, total_tokens, h_v, d)
    log_alpha = -torch.rand(1, total_tokens, head_o, dtype=torch.float32, device=device)
    beta = torch.sigmoid(
        torch.randn(1, total_tokens, head_o, dtype=torch.float32, device=device)
    )
    h0 = torch.randn((num_seqs, head_o, d, d), dtype=torch.float32, device=device)

    run = lambda: chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=log_alpha,
        beta=beta,
        scale=None,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    kernel_names = (
        "tilelang_gdn_l2norm_kernel",
        "tilelang_kkt_solve_kernel",
        "tilelang_fused_chunk_gdn_prefill_kernel_",
    )

    kernel_times = bench_kineto(
        run,
        kernel_names=kernel_names,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=True,
        with_multiple_kernels=True,
    )
    return float(np.sum(np.asarray(kernel_times, dtype=np.float64))) * 1e3


def main():
    parser = argparse.ArgumentParser(description="Benchmark mate.gdn_prefill")
    parser.add_argument(
        "--num-tests",
        type=int,
        default=10,
        help="Number of timing iterations",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Input dtype for q/k/v",
    )
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise RuntimeError("MUSA device is not available.")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print(f"\nMUSA: {torch.musa.get_device_name(0)}")
    print("Kernel: mate.gdn_prefill.chunk_gated_delta_rule")
    print(f"dtype={args.dtype}, split_qkv=strided, qk_l2norm=standalone_inplace")
    print()

    header = (
        f"{'Heads':<15s}  {'Seqlens':<16s}  {'h_qk':>4s} {'h_v':>4s}"
        f"  {'MATE (MUSA)':>12s}  {'TFLOPS':>7s}"
    )
    print(header)
    print("-" * len(header))

    for h_qk, h_v, d, h_label in HEAD_CONFIGS:
        for endpoints, s_label in SEQ_CONFIGS:
            total_tokens = endpoints[-1]
            mate_ms = bench_mate(
                endpoints=endpoints,
                h_qk=h_qk,
                h_v=h_v,
                d=d,
                dtype=dtype,
                num_tests=args.num_tests,
            )
            tflops = _gdn_tflops(total_tokens, max(h_qk, h_v), d, mate_ms)
            print(
                f"{h_label:<15s}  {s_label:<16s}  {h_qk:>4d} {h_v:>4d}"
                f"  {mate_ms:>11.3f}ms  {tflops:>6.1f}"
            )
        print()


if __name__ == "__main__":
    main()
