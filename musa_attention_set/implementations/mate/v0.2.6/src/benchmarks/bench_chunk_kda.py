from __future__ import annotations

import argparse
import math
import os
from typing import Callable


FIXED_CASES = [
    [8192],
]

VARLEN_CASES = [
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
]


def _parse_seq_lens(value: str) -> list[int]:
    try:
        seq_lens = [int(x) for x in value.replace(":", ",").split(",") if x]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seq_lens: {value}") from exc
    if not seq_lens or any(x <= 0 for x in seq_lens):
        raise argparse.ArgumentTypeError("seq_lens must contain positive integers")
    return seq_lens


def _configure_runtime(device: str | None, allow_device0: bool) -> None:
    if device is not None:
        os.environ["MUSA_VISIBLE_DEVICES"] = device
    elif "MUSA_VISIBLE_DEVICES" not in os.environ:
        os.environ["MUSA_VISIBLE_DEVICES"] = "1"

    first = os.environ.get("MUSA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if first == "0" and not allow_device0:
        raise RuntimeError(
            "Refusing to run on physical MUSA device 0. "
            "Use --device 1 or set MUSA_VISIBLE_DEVICES to a non-zero device."
        )

    os.environ.setdefault("MATE_MUSA_ARCH_LIST", "3.1")


def _summarize_ms(values: list[float]) -> tuple[float, float, float, float]:
    xs = sorted(float(x) for x in values)
    n = len(xs)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan, nan
    mn = xs[0]
    mx = xs[-1]
    mean = sum(xs) / n
    mid = n // 2
    median = xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])
    return mn, mx, mean, median


def bench_fn(
    fn: Callable[[], object],
    *,
    warmup: int | None,
    iters: int | None,
    repeats: int,
    dry_run_time_ms: int,
    repeat_time_ms: int,
    l2_flush: bool,
    l2_flush_size_mb: int,
) -> tuple[float, float, float, float, int]:
    from mate.testing.utils import bench_gpu_time

    dry_run_iters = warmup
    repeat_iters = None if iters is None else iters * repeats
    times = bench_gpu_time(
        fn,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        dry_run_time_ms=dry_run_time_ms,
        repeat_time_ms=repeat_time_ms,
        l2_flush=l2_flush,
        l2_flush_size_mb=l2_flush_size_mb,
        l2_flush_device="musa",
    )
    mn, mx, mean, median = _summarize_ms(times)
    return mn, mx, mean, median, len(times)


def _state_name(dtype) -> str:
    name = str(dtype).rsplit(".", maxsplit=1)[-1]
    if name == "bfloat16":
        return "bf16"
    if name == "float16":
        return "fp16"
    if name == "float32":
        return "fp32"
    return name


def run_case(
    torch,
    F,
    mate_kda,
    seq_lens: list[int],
    H: int,
    D: int,
    dtype,
    warmup: int | None,
    iters: int | None,
    repeats: int,
    dry_run_time_ms: int,
    repeat_time_ms: int,
    l2_flush: bool,
    l2_flush_size_mb: int,
    seed: int,
    state_mode: str,
    varlen: bool,
    cu_seqlens_dtype,
) -> None:
    device = torch.device("musa")
    lower_bound = -5.0
    scale = 1.0 / math.sqrt(D)
    if not varlen and len(seq_lens) != 1:
        raise ValueError("fixed benchmark mode requires exactly one sequence length")

    total = sum(seq_lens)
    nseq = len(seq_lens)

    torch.manual_seed(seed)
    torch.musa.manual_seed(seed)

    if varlen:
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=cu_seqlens_dtype,
            device=device,
        )
        bench_desc = (
            f"warmup={warmup} iters={iters} repeats={repeats}"
            if iters is not None
            else f"dry_run_time_ms={dry_run_time_ms} repeat_time_ms={repeat_time_ms}"
        )
        print(
            f"varlen shape=[{total},{H},{D}] seq_lens={seq_lens} "
            f"{bench_desc} l2_flush={l2_flush}"
        )
    else:
        cu_seqlens = None
        bench_desc = (
            f"warmup={warmup} iters={iters} repeats={repeats}"
            if iters is not None
            else f"dry_run_time_ms={dry_run_time_ms} repeat_time_ms={repeat_time_ms}"
        )
        print(f"shape=[{total},{H},{D}] {bench_desc} l2_flush={l2_flush}")

    q = F.normalize(
        torch.randn((1, total, H, D), dtype=torch.float32, device=device),
        p=2,
        dim=-1,
    ).to(dtype)
    k = F.normalize(
        torch.randn((1, total, H, D), dtype=torch.float32, device=device),
        p=2,
        dim=-1,
    ).to(dtype)
    v = torch.randn((1, total, H, D), dtype=dtype, device=device)
    g = torch.randn((1, total, H, D), dtype=dtype, device=device)
    beta = torch.randn((1, total, H), dtype=dtype, device=device)
    a_log = torch.rand(H, dtype=torch.float32, device=device)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)
    output = torch.zeros_like(v)

    state_elems = nseq * H * D * D
    initial_state = (
        torch.arange(state_elems, dtype=torch.float32, device=device)
        .reshape(nseq, H, D, D)
        .to(dtype)
    )
    final_state = torch.zeros_like(initial_state)

    initial_state_fp32 = initial_state.float()
    final_state_fp32 = torch.zeros_like(initial_state_fp32)

    def run_fused_kda(state, state_out) -> object:
        return mate_kda.chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=state,
            output_final_state=state_out is not None,
            cu_seqlens=cu_seqlens,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            use_qk_l2norm_in_kernel=True,
            output=output,
            final_state=state_out,
        )

    def bench_one(label: str, fn: Callable[[], object]) -> None:
        mn, mx, mean, median, n = bench_fn(
            fn,
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            l2_flush=l2_flush,
            l2_flush_size_mb=l2_flush_size_mb,
        )
        print(
            f"  {label:<24s}: min={mn:.4f} ms, max={mx:.4f} ms, "
            f"mean={mean:.4f} ms, median={median:.4f} ms, n={n}"
        )

    run_bf16_state = state_mode in ("bf16", "both", "all")
    run_fp32_state = state_mode in ("fp32", "both", "all")
    run_no_state = state_mode in ("none", "all")

    if run_bf16_state:
        bench_one(
            f"fused ({_state_name(dtype)} state)",
            lambda: run_fused_kda(initial_state, final_state),
        )
    if run_no_state:
        bench_one("fused (no state)", lambda: run_fused_kda(None, None))
    if run_fp32_state:
        bench_one(
            "fused (fp32 state)",
            lambda: run_fused_kda(initial_state_fp32, final_state_fp32),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MATE chunk KDA with the same cases as FlashKDA's bench_fwd.py."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Fixed dry-run iterations. Defaults to time based.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Fixed measured iterations per repeat. Defaults to time based.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Multiplier for --iters when fixed measured iterations are used.",
    )
    parser.add_argument("--dry-run-time-ms", type=int, default=100)
    parser.add_argument("--repeat-time-ms", type=int, default=1000)
    parser.add_argument("--no-l2-flush", action="store_true")
    parser.add_argument("--l2-flush-size-mb", type=int, default=512)
    parser.add_argument(
        "--mode",
        choices=["fixed", "varlen", "all"],
        default="all",
        help=(
            "Select default cases and, with --seq-lens, force fixed/varlen layout; "
            "'all' infers varlen when multiple lengths are supplied."
        ),
    )
    parser.add_argument("--H", type=int, default=96)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-lens", action="append", type=_parse_seq_lens)
    parser.add_argument(
        "--cu-seqlens-dtype",
        choices=["int32", "int64"],
        default="int64",
        help="Cumulative sequence-length dtype used in varlen mode.",
    )
    parser.add_argument(
        "--state-mode", choices=["bf16", "fp32", "both", "none", "all"], default="bf16"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Physical MUSA device id to expose, e.g. 1.",
    )
    parser.add_argument(
        "--allow-device0", action="store_true", help="Allow physical MUSA device 0."
    )
    args = parser.parse_args()

    if args.D != 128:
        raise ValueError("chunk_kda currently requires --D 128.")
    if args.warmup is not None and args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.iters is not None and args.iters <= 0:
        raise ValueError("--iters must be > 0.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0.")
    if args.dry_run_time_ms <= 0 or args.repeat_time_ms <= 0:
        raise ValueError("--dry-run-time-ms and --repeat-time-ms must be > 0.")

    _configure_runtime(args.device, args.allow_device0)

    import torch
    import torch.nn.functional as F

    import mate.kda as mate_kda

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise RuntimeError("MUSA device is not available.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cu_seqlens_dtype = torch.int32 if args.cu_seqlens_dtype == "int32" else torch.int64

    print(f"MUSA_VISIBLE_DEVICES={os.environ.get('MUSA_VISIBLE_DEVICES')}")
    print(f"MUSA: {torch.musa.get_device_name(0)}")
    print("Kernel: fused")

    cases: list[tuple[list[int], bool]] = []
    if args.seq_lens:
        if args.mode == "fixed":
            cases.extend((seq_lens, False) for seq_lens in args.seq_lens)
        elif args.mode == "varlen":
            cases.extend((seq_lens, True) for seq_lens in args.seq_lens)
        else:
            cases.extend((seq_lens, len(seq_lens) > 1) for seq_lens in args.seq_lens)
    else:
        if args.mode in ("fixed", "all"):
            cases.extend((seq_lens, False) for seq_lens in FIXED_CASES)
        if args.mode in ("varlen", "all"):
            cases.extend((seq_lens, True) for seq_lens in VARLEN_CASES)

    for seq_lens, varlen in cases:
        run_case(
            torch,
            F,
            mate_kda,
            seq_lens,
            args.H,
            args.D,
            dtype,
            args.warmup,
            args.iters,
            args.repeats,
            args.dry_run_time_ms,
            args.repeat_time_ms,
            not args.no_l2_flush,
            args.l2_flush_size_mb,
            args.seed,
            args.state_mode,
            varlen,
            cu_seqlens_dtype,
        )


if __name__ == "__main__":
    main()
