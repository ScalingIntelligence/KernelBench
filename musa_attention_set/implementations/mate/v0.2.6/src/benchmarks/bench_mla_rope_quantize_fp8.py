from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mate.sparse_mla_interface import mla_rope_quantize_fp8  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _measure_us(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()

    starts = [torch.musa.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.musa.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.musa.synchronize()
    samples = [start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends)]
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark native MATE MLA RoPE plus FP8 quantization."
    )
    parser.add_argument("--tokens", type=_positive_int, default=4096)
    parser.add_argument("--heads", type=_positive_int, default=128)
    parser.add_argument("--warmup", type=_nonnegative_int, default=10)
    parser.add_argument("--repeat", type=_positive_int, default=50)
    args = parser.parse_args()

    device = "musa"
    torch.manual_seed(20260817)
    q_rope = torch.randn(
        args.tokens, args.heads, 64, device=device, dtype=torch.bfloat16
    )
    q_nope = torch.randn(
        args.tokens, args.heads, 512, device=device, dtype=torch.bfloat16
    )
    k_rope = torch.randn(args.tokens, 64, device=device, dtype=torch.bfloat16)
    k_nope = torch.randn(args.tokens, 512, device=device, dtype=torch.bfloat16)
    angles = torch.randn(args.tokens, 32, device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    pos_ids = torch.arange(args.tokens, device=device, dtype=torch.int32)
    q_out = torch.empty(
        args.tokens, args.heads, 576, device=device, dtype=torch.float8_e4m3fn
    )
    k_out = torch.empty(args.tokens, 576, device=device, dtype=torch.float8_e4m3fn)

    def quant() -> None:
        mla_rope_quantize_fp8(
            q_rope,
            k_rope,
            q_nope,
            k_nope,
            cos_sin_cache,
            pos_ids,
            quant_scale_q=0.625,
            quant_scale_kv=1.75,
            q_rope_out=q_out[..., 512:],
            k_rope_out=k_out[..., 512:],
            q_nope_out=q_out[..., :512],
            k_nope_out=k_out[..., :512],
        )

    values = args.tokens * (args.heads + 1) * 576
    quant_payload_bytes = values * 3
    copy_elements = (quant_payload_bytes + 1) // 2
    copy_src = torch.empty(copy_elements, device=device, dtype=torch.uint8)
    copy_dst = torch.empty_like(copy_src)

    quant_us = _measure_us(quant, args.warmup, args.repeat)
    copy_us = _measure_us(lambda: copy_dst.copy_(copy_src), args.warmup, args.repeat)
    quant_gbs = quant_payload_bytes / quant_us / 1e3
    copy_payload_bytes = copy_elements * 2
    copy_gbs = copy_payload_bytes / copy_us / 1e3
    ratio = quant_gbs / copy_gbs

    print("benchmark=mla-rope-quantize-fp8 backend=mate")
    print(
        f"tokens={args.tokens} heads={args.heads} warmup={args.warmup} repeat={args.repeat}"
    )
    print(
        f"quant_us={quant_us:.3f} payload_bytes={quant_payload_bytes} "
        f"effective_gbs={quant_gbs:.3f}"
    )
    print(
        f"d2d_us={copy_us:.3f} payload_bytes={copy_payload_bytes} "
        f"effective_gbs={copy_gbs:.3f}"
    )
    print(f"quant_to_d2d={ratio:.4f} target=0.8500")


if __name__ == "__main__":
    main()
