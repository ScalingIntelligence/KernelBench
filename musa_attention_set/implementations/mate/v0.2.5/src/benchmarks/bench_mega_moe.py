from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Callable

os.environ["NVSHMEM_IBGDA_NIC_HANDLER"] = "cpu"

import torch
import torch.distributed as dist

from mate import deep_gemm
from mate.jit.mega_moe import get_block_m_for_mega_moe
from mate.testing.utils import per_block_cast_to_fp8, per_token_cast_to_fp8


DEFAULT_TOKEN_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
_LOCAL_RANK: int | None = None


@dataclass(frozen=True)
class StageStats:
    flops: int
    hbm_bytes: int


@dataclass(frozen=True)
class LocalBenchResult:
    rank: int
    tokens: int
    received_tokens: int
    padded_rows: int
    touched_experts: int
    stage1: StageStats
    stage2: StageStats
    stage1_seconds: float
    stage2_seconds: float


def _parse_tokens(text: str) -> list[int]:
    tokens = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not tokens:
        raise ValueError("--tokens must contain at least one token count")
    if any(token <= 0 for token in tokens):
        raise ValueError("--tokens only accepts positive token counts")
    return tokens


def _safe_div(lhs: float, rhs: float) -> float:
    return float("nan") if rhs == 0 else lhs / rhs


def _init_dist(local_rank: int, num_local_ranks: int):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    global _LOCAL_RANK
    _LOCAL_RANK = local_rank

    dist.init_process_group(
        backend="mccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank,
        device_id=local_rank,
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("musa")
    torch.musa.set_device(local_rank)
    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(num_local_ranks * num_nodes)), backend="mccl"),
    )


def _rank0_print(message: str = "") -> None:
    if dist.is_initialized():
        should_print = dist.get_rank() == 0
    else:
        should_print = _LOCAL_RANK == 0
    if should_print:
        print(message, flush=True)


def _profiler_event_time_range_us(event) -> tuple[float, float] | None:
    time_range = getattr(event, "time_range", None)
    if time_range is not None:
        start = getattr(time_range, "start", None)
        if start is not None:
            elapsed = (
                time_range.elapsed_us()
                if callable(getattr(time_range, "elapsed_us", None))
                else getattr(time_range, "elapsed_us", None)
            )
            if elapsed is not None:
                return float(start), float(elapsed)

    start = getattr(event, "start_us", None)
    duration = getattr(event, "duration_us", None)
    if start is not None and duration is not None:
        return float(start), float(duration)
    return None


def _extract_profiler_kernel_events(prof, kernel_name: str) -> list[dict[str, float]]:
    kernel_events = []
    for event in prof.events():
        if kernel_name not in getattr(event, "name", ""):
            continue
        time_range = _profiler_event_time_range_us(event)
        if time_range is None:
            continue
        start_us, duration_us = time_range
        kernel_events.append(
            {
                "start_us": start_us,
                "duration_us": duration_us,
                "end_us": start_us + duration_us,
            }
        )
    return sorted(kernel_events, key=lambda item: item["start_us"])


def _average_aligned_kernel_duration(
    kernel_events: list[dict[str, float]], group: dist.ProcessGroup
) -> float:
    rank_kernel_events = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(rank_kernel_events, kernel_events, group=group)
    assert all(rank_kernel_events)
    num_iters = min(len(events) for events in rank_kernel_events)
    assert num_iters > 0

    aligned_duration_us = 0.0
    for iter_idx in range(num_iters):
        aligned_start_us = max(
            events[iter_idx]["start_us"] for events in rank_kernel_events
        )
        aligned_end_us = min(
            events[iter_idx]["end_us"] for events in rank_kernel_events
        )
        aligned_iter_us = aligned_end_us - aligned_start_us
        if aligned_iter_us <= 0:
            aligned_iter_us = min(
                events[iter_idx]["duration_us"] for events in rank_kernel_events
            )
        aligned_duration_us += aligned_iter_us
    return aligned_duration_us / num_iters / 1e6


def _bench_kineto(
    fn: Callable[[], object],
    kernel_name: str,
    *,
    num_tests: int,
    suppress_kineto_output: bool,
    barrier: Callable[[], object],
    aligned_kernel_duration: bool,
    group: dist.ProcessGroup,
    flush_l2_bytes: int,
    barrier_comm_profiling: bool,
) -> float:
    class empty_suppress:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class suppress_stdout_stderr:
        def __enter__(self):
            self.outnull_file = open(os.devnull, "w")
            self.errnull_file = open(os.devnull, "w")
            self.old_stdout_fileno_undup = os.dup(1)
            self.old_stderr_fileno_undup = os.dup(2)
            os.dup2(self.outnull_file.fileno(), 1)
            os.dup2(self.errnull_file.fileno(), 2)
            return self

        def __exit__(self, *_):
            os.dup2(self.old_stdout_fileno_undup, 1)
            os.dup2(self.old_stderr_fileno_undup, 2)
            os.close(self.old_stdout_fileno_undup)
            os.close(self.old_stderr_fileno_undup)
            self.outnull_file.close()
            self.errnull_file.close()

    flush_l2_elems = flush_l2_bytes // 4
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.MUSA], schedule=schedule
        ) as prof:
            for _ in range(2):
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device="musa")
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device="musa")
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device="musa"))
                for _ in range(num_tests):
                    if flush_l2_elems > 0:
                        torch.empty(
                            flush_l2_elems, dtype=torch.int, device="musa"
                        ).zero_()
                    barrier()
                    fn()
                prof.step()

    prof_lines = (
        prof.key_averages()
        .table(sort_by="count", max_name_column_width=100)
        .split("\n")
    )
    units = {"ms": 1e3, "us": 1e6}
    selected_kernel_name = None
    for line in prof_lines:
        if kernel_name not in line:
            continue
        time_str = line.split()[-2]
        for unit, scale in units.items():
            if unit in time_str:
                selected_kernel_name = kernel_name
                kernel_duration = float(time_str.replace(unit, "")) / scale
                break
        if selected_kernel_name is not None:
            break
    else:
        event_names = sorted(
            {
                getattr(event, "name", "")
                for event in prof.events()
                if getattr(event, "name", "")
            }
        )
        raise RuntimeError(
            f"Profiler did not find kernel name: {kernel_name}. "
            f"Available events: {event_names[:64]}"
        )

    if aligned_kernel_duration:
        events = _extract_profiler_kernel_events(prof, selected_kernel_name)
        return _average_aligned_kernel_duration(events, group)
    return kernel_duration


def _get_local_expert_recv_count_ref(
    topk_idx: torch.Tensor,
    num_tokens: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_ranks: int,
    rank_idx: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    num_topk = topk_idx.size(1)
    padded_topk_idx = torch.full(
        (num_max_tokens_per_rank, num_topk),
        -1,
        dtype=topk_idx.dtype,
        device=topk_idx.device,
    )
    padded_topk_idx[:num_tokens].copy_(topk_idx[:num_tokens])
    all_topk_idx = torch.empty(
        (num_ranks, num_max_tokens_per_rank, num_topk),
        dtype=topk_idx.dtype,
        device=topk_idx.device,
    )
    dist.all_gather_into_tensor(all_topk_idx, padded_topk_idx, group=group)

    num_experts_per_rank = num_experts // num_ranks
    local_expert_ids = torch.arange(
        rank_idx * num_experts_per_rank,
        (rank_idx + 1) * num_experts_per_rank,
        dtype=topk_idx.dtype,
        device=topk_idx.device,
    )
    return (
        all_topk_idx.unsqueeze(-1)
        .eq(local_expert_ids)
        .sum(dim=(0, 1, 2))
        .to(torch.int64)
    )


def _make_transformed_weights(
    num_experts_per_rank: int,
    hidden: int,
    intermediate_hidden: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    l1_weight_fp8 = []
    l1_weight_sf = []
    for _ in range(num_experts_per_rank):
        w_bf16 = (
            torch.rand((intermediate_hidden * 2, hidden), device="musa") * 2 - 1
        ).to(torch.bfloat16)
        w_fp8, w_sf = per_block_cast_to_fp8(w_bf16, torch.float8_e4m3fn)
        l1_weight_fp8.append(w_fp8)
        l1_weight_sf.append(w_sf.contiguous())

    l2_weight_fp8 = []
    l2_weight_sf = []
    for _ in range(num_experts_per_rank):
        w_bf16 = (torch.rand((hidden, intermediate_hidden), device="musa") * 2 - 1).to(
            torch.bfloat16
        )
        w_fp8, w_sf = per_block_cast_to_fp8(w_bf16, torch.float8_e4m3fn)
        l2_weight_fp8.append(w_fp8)
        l2_weight_sf.append(w_sf.contiguous())

    l1_weights = (
        torch.stack(l1_weight_fp8).contiguous(),
        torch.stack(l1_weight_sf).contiguous(),
    )
    l2_weights = (
        torch.stack(l2_weight_fp8).contiguous(),
        torch.stack(l2_weight_sf).contiguous(),
    )
    return deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)


def _make_stage_stats(
    *,
    total_pool_rows: int,
    touched_experts: int,
    hidden: int,
    intermediate_hidden: int,
) -> tuple[StageStats, StageStats]:
    stage1_flops = 2 * total_pool_rows * hidden * (intermediate_hidden * 2)
    stage1_hbm_bytes = (
        total_pool_rows * hidden
        + total_pool_rows * ((hidden + 127) // 128) * 4
        + touched_experts * intermediate_hidden * 2 * hidden
        + touched_experts
        * ((intermediate_hidden * 2) // 64)
        * ((hidden + 127) // 128)
        * 4
        + total_pool_rows * intermediate_hidden
        + total_pool_rows * (intermediate_hidden // 128) * 4
    )
    stage2_flops = 2 * total_pool_rows * intermediate_hidden * hidden
    stage2_hbm_bytes = (
        total_pool_rows * intermediate_hidden
        + total_pool_rows * (intermediate_hidden // 128) * 4
        + touched_experts * hidden * intermediate_hidden
        + touched_experts * (hidden // 128) * (intermediate_hidden // 128) * 4
        + total_pool_rows * hidden * 2
    )
    return (
        StageStats(flops=stage1_flops, hbm_bytes=stage1_hbm_bytes),
        StageStats(flops=stage2_flops, hbm_bytes=stage2_hbm_bytes),
    )


def _print_result_header() -> None:
    _rank0_print(
        f"{'tokens':>6} {'recv_tokens':>11} {'padded_rows':>11} "
        f"{'experts':>7} {'stage1_us':>10} {'stage1_TFLOPS':>13} "
        f"{'stage1_GB/s':>12} {'stage2_us':>10} {'stage2_TFLOPS':>13} "
        f"{'stage2_GB/s':>12}"
    )


def _print_aggregated_result(
    result: LocalBenchResult, group: dist.ProcessGroup
) -> None:
    gathered: list[LocalBenchResult | None] = [
        None for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather_object(gathered, result, group=group)
    if dist.get_rank(group) != 0:
        return

    rows = [row for row in gathered if row is not None]
    rank0 = next(row for row in rows if row.rank == 0)
    print(
        f"{rank0.tokens:6d} {rank0.received_tokens:11d} "
        f"{rank0.padded_rows:11d} {rank0.touched_experts:7d} "
        f"{rank0.stage1_seconds * 1e6:10.2f} "
        f"{_safe_div(rank0.stage1.flops / 1e12, rank0.stage1_seconds):13.2f} "
        f"{_safe_div(rank0.stage1.hbm_bytes / 1e9, rank0.stage1_seconds):12.2f} "
        f"{rank0.stage2_seconds * 1e6:10.2f} "
        f"{_safe_div(rank0.stage2.flops / 1e12, rank0.stage2_seconds):13.2f} "
        f"{_safe_div(rank0.stage2.hbm_bytes / 1e9, rank0.stage2_seconds):12.2f}",
        flush=True,
    )


def _bench_one_token_count(
    *,
    num_tokens: int,
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    args: argparse.Namespace,
    x_fp8_all: torch.Tensor,
    x_sf_all: torch.Tensor,
    topk_idx_all: torch.Tensor,
    topk_weights_all: torch.Tensor,
    transformed_l1_weights: tuple[torch.Tensor, torch.Tensor],
    transformed_l2_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer,
) -> LocalBenchResult:
    block_m = get_block_m_for_mega_moe(
        num_ranks, args.num_experts, args.num_max_tokens_per_rank, args.num_topk
    )
    expert_counts = _get_local_expert_recv_count_ref(
        topk_idx_all,
        num_tokens,
        args.num_max_tokens_per_rank,
        args.num_experts,
        num_ranks,
        rank_idx,
        group,
    )
    total_pool_rows = 0
    touched_experts = 0
    for count_tensor in expert_counts.tolist():
        count = int(count_tensor)
        padded_count = ((count + block_m - 1) // block_m) * block_m if count > 0 else 0
        total_pool_rows += padded_count
        touched_experts += int(padded_count > 0)
    received_tokens = int(expert_counts.sum().item())
    stage1_stats, stage2_stats = _make_stage_stats(
        total_pool_rows=total_pool_rows,
        touched_experts=touched_experts,
        hidden=args.hidden,
        intermediate_hidden=args.intermediate_hidden,
    )

    def prepare_fused_inputs() -> None:
        sym_buffer.x[:num_tokens].copy_(x_fp8_all[:num_tokens])
        sym_buffer.x_sf[:num_tokens].copy_(x_sf_all[:num_tokens])
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx_all[:num_tokens])
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights_all[:num_tokens])

    def run_fused() -> torch.Tensor:
        prepare_fused_inputs()
        y = torch.empty((num_tokens, args.hidden), dtype=torch.bfloat16, device="musa")
        deep_gemm.fp8_fp8_mega_moe(
            y=y,
            l1_weights=transformed_l1_weights,
            l2_weights=transformed_l2_weights,
            sym_buffer=sym_buffer,
            cumulative_local_expert_recv_stats=None,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )
        torch.musa.synchronize()
        dist.barrier(group=group)
        return y

    run_fused()
    torch.musa.synchronize()
    dist.barrier(group=group)

    stage1_seconds = _bench_kineto(
        run_fused,
        "fp8_fp8_mega_moe_stage1_impl",
        num_tests=args.num_tests,
        suppress_kineto_output=not args.show_kineto_output,
        barrier=lambda: dist.barrier(group=group),
        aligned_kernel_duration=True,
        group=group,
        flush_l2_bytes=args.flush_l2_bytes,
        barrier_comm_profiling=args.stage1_barrier_comm_profiling,
    )
    stage2_seconds = _bench_kineto(
        run_fused,
        "fp8_fp8_mega_moe_stage2_impl",
        num_tests=args.num_tests,
        suppress_kineto_output=not args.show_kineto_output,
        barrier=lambda: dist.barrier(group=group),
        aligned_kernel_duration=True,
        group=group,
        flush_l2_bytes=args.flush_l2_bytes,
        barrier_comm_profiling=False,
    )

    return LocalBenchResult(
        rank=rank_idx,
        tokens=num_tokens,
        received_tokens=received_tokens,
        padded_rows=total_pool_rows,
        touched_experts=touched_experts,
        stage1=stage1_stats,
        stage2=stage2_stats,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
    )


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = _init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed + rank_idx)
    random.seed(args.seed + rank_idx)
    sym_buffer = None

    try:
        assert args.hidden % 512 == 0
        assert args.intermediate_hidden % 256 == 0
        assert args.num_experts % num_ranks == 0

        num_experts_per_rank = args.num_experts // num_ranks
        max_tokens = max(args.token_values)
        x_bf16 = torch.rand(
            (max_tokens, args.hidden), dtype=torch.bfloat16, device="musa"
        )
        x_fp8_all, x_sf_all = per_token_cast_to_fp8(
            x_bf16.contiguous(), torch.float8_e4m3fn
        )
        x_fp8_all = x_fp8_all.contiguous()
        x_sf_all = x_sf_all.contiguous()

        scores = torch.randn(
            (max_tokens, args.num_experts), dtype=torch.float32, device="musa"
        )
        topk_weights_all, topk_idx_all = torch.topk(
            scores, args.num_topk, dim=-1, largest=True, sorted=False
        )

        transformed_l1_weights, transformed_l2_weights = _make_transformed_weights(
            num_experts_per_rank,
            args.hidden,
            args.intermediate_hidden,
        )

        sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            args.num_experts,
            args.num_max_tokens_per_rank,
            args.num_topk,
            args.hidden,
            args.intermediate_hidden,
        )
        torch.musa.synchronize()
        dist.barrier(group=group)

        _rank0_print("MATE MegaMoE benchmark")
        _rank0_print(
            f"ranks={num_ranks}, max_tokens_per_rank={args.num_max_tokens_per_rank}, "
            f"hidden={args.hidden}, intermediate={args.intermediate_hidden}, "
            f"experts={args.num_experts}, topk={args.num_topk}, "
            f"buffer={sym_buffer.buffer.nbytes / 2**30:.3f} GiB"
        )
        _print_result_header()

        for num_tokens in args.token_values:
            result = _bench_one_token_count(
                num_tokens=num_tokens,
                rank_idx=rank_idx,
                num_ranks=num_ranks,
                group=group,
                args=args,
                x_fp8_all=x_fp8_all,
                x_sf_all=x_sf_all,
                topk_idx_all=topk_idx_all,
                topk_weights_all=topk_weights_all,
                transformed_l1_weights=transformed_l1_weights,
                transformed_l2_weights=transformed_l2_weights,
                sym_buffer=sym_buffer,
            )
            _print_aggregated_result(result, group)

    finally:
        if dist.is_initialized():
            torch.musa.synchronize()
            dist.barrier(group=group)
            if sym_buffer is not None:
                sym_buffer.destroy()
            dist.destroy_process_group(group)
            dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MATE MegaMoE fused kernels."
    )
    parser.add_argument(
        "--tokens",
        default=",".join(str(token) for token in DEFAULT_TOKEN_SWEEP),
        help="Comma-separated input token counts per rank.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=int(os.getenv("MATE_MEGA_MOE_BENCH_NUM_PROCESSES", "8")),
    )
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--activation-clamp", type=float, default=10)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--num-tests", type=int, default=5)
    parser.add_argument("--flush-l2-bytes", type=int, default=int(8e9))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--show-kineto-output", action="store_true")
    parser.add_argument("--stage1-barrier-comm-profiling", action="store_true")
    args = parser.parse_args()

    token_values = _parse_tokens(args.tokens)
    if max(token_values) > args.num_max_tokens_per_rank:
        raise ValueError(
            f"max token count {max(token_values)} exceeds "
            f"--num-max-tokens-per-rank={args.num_max_tokens_per_rank}"
        )
    if args.num_tests <= 0:
        raise ValueError("--num-tests must be positive")
    if args.flush_l2_bytes < 0:
        raise ValueError("--flush-l2-bytes must be non-negative")
    args.token_values = token_values
    return args


def main() -> None:
    args = _parse_args()
    torch.multiprocessing.spawn(
        _worker, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()
