#!/usr/bin/env python3
"""Sparse MLA parity benchmark for the FlashMLA-compatible wrapper.

Measures two views for each case:
- end-to-end wrapper latency with MUSA events
- graph-replayed kernel-path latency with MUSA/CUDA events
- implementation-kernel-only profiler time, filtered to sparse split/combine kernels
- optional per-kernel profiler breakdowns

Run with GPU 2, for example:
    MUSA_VISIBLE_DEVICES=2 python benchmarks/bench_sparse_mla.py --mode both --quick
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import statistics
import sys
from pathlib import Path
from typing import Callable, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FLASHMLA_ROOT = REPO_ROOT / "wrappers" / "FlashMLA"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FLASHMLA_ROOT))

import flash_mla  # noqa: E402
from mate.mate_runtime import resolve_num_mps  # noqa: E402
from mate.sparse_mla_interface import (  # noqa: E402
    get_batch_decode_metadata_mla,
    sparse_mla_fp8_decode,
)
from mate.testing.sparse_mla import quantize_sparse_mla_cache  # noqa: E402


DV = 512
_GRAPH_KEEPALIVE: list[object] = []
MAIN_KERNEL_PATTERNS = (
    "flash_fwd_splitkv_mla_fp8_sparse_kernel",
    "sparse_attn_fwd",
    "dsa_prefill",
    "dsa_decode",
)
UNEXPECTED_MAIN_KERNEL_PATTERNS = ("main_kernel", "main_split")
COMBINE_KERNEL_PATTERNS = (
    "flash_fwd_mla_combine_kernel",
    "dsa_combine",
    "combine",
)


@dataclasses.dataclass(frozen=True)
class PrefillCase:
    name: str
    d_qk: int
    num_heads: int
    seq_len_q: int
    seq_len_kv: int
    topk: int
    attn_sink: bool = True
    topk_length: bool = True
    extra_seq_len_kv: int = 0
    extra_topk: int = 0
    direct_tilelang: bool = False
    index_pattern: str = "strided"
    is_persistence: bool = True
    persistent_blocks: int | None = None


@dataclasses.dataclass(frozen=True)
class DecodeCase:
    name: str
    d_qk: int
    num_heads: int
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    topk: int
    page_size: int
    extra_seq_len_kv: int = 0
    extra_topk: int = 0
    extra_page_size: int = 64
    attn_sink: bool = True
    topk_length: bool = True
    extra_topk_length: bool = True
    direct_tilelang: bool = False
    temp_metadata: bool = False
    mp_count: int = 56
    tile_m: int = 64
    native_fp8: bool = False


@dataclasses.dataclass(frozen=True)
class WorkloadStats:
    flop: float
    mem_bytes: float

    @property
    def compute_memory_ratio(self) -> float:
        return self.flop / self.mem_bytes if self.mem_bytes > 0 else 0.0

    def tflops(self, elapsed_us: float) -> float:
        return self.flop / (elapsed_us * 1e-6) / 1e12 if elapsed_us > 0 else 0.0

    def gbps(self, elapsed_us: float) -> float:
        return self.mem_bytes / (elapsed_us * 1e-6) / 1e9 if elapsed_us > 0 else 0.0


def _gpu_backend():
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.musa
    if torch.cuda.is_available():
        return torch.cuda
    raise RuntimeError("Neither MUSA nor CUDA is available")


def _device() -> str:
    backend = _gpu_backend()
    return "musa" if hasattr(torch, "musa") and backend is torch.musa else "cuda"


def _sync() -> None:
    _gpu_backend().synchronize()


def _profiler_activity():
    if hasattr(torch.profiler.ProfilerActivity, "MUSA"):
        return torch.profiler.ProfilerActivity.MUSA
    return torch.profiler.ProfilerActivity.CUDA


def _new_event():
    return _gpu_backend().Event(enable_timing=True)


def _new_graph():
    backend = _gpu_backend()
    if hasattr(backend, "MUSAGraph"):
        return backend.MUSAGraph()
    return backend.CUDAGraph()


def _bench_event_us(fn: Callable[[], object], repeat: int) -> float:
    # Event timing keeps launch overhead visible for wrapper e2e numbers without
    # relying on tilelang.profiler.do_bench, which can be unstable on large cases.
    repeat = max(3, repeat)
    for _ in range(max(3, repeat // 4)):
        fn()
    _sync()

    start_events = [_new_event() for _ in range(repeat)]
    end_events = [_new_event() for _ in range(repeat)]
    for idx in range(repeat):
        start_events[idx].record()
        fn()
        end_events[idx].record()
    _sync()
    times_ms = [
        start_events[idx].elapsed_time(end_events[idx]) for idx in range(repeat)
    ]
    return statistics.median(times_ms) * 1000.0


def _bench_graph_us(
    fn: Callable[[], object],
    repeat: int,
    num_iters_within_graph: int,
) -> float:
    backend = _gpu_backend()
    num_iters_within_graph = max(1, num_iters_within_graph)

    captured_outputs = []
    for _ in range(3):
        captured_outputs.append(fn())
    _sync()

    graph = _new_graph()
    with backend.graph(graph):
        for _ in range(num_iters_within_graph):
            captured_outputs.append(fn())
    _sync()

    for _ in range(3):
        graph.replay()
    _sync()

    repeat = max(3, repeat)
    start_events = [_new_event() for _ in range(repeat)]
    end_events = [_new_event() for _ in range(repeat)]
    for idx in range(repeat):
        start_events[idx].record()
        graph.replay()
        end_events[idx].record()
    _sync()
    times_ms = [
        start_events[idx].elapsed_time(end_events[idx]) / num_iters_within_graph
        for idx in range(repeat)
    ]
    _GRAPH_KEEPALIVE.append((graph, captured_outputs))
    return statistics.median(times_ms) * 1000.0


def _bench_kernel_path_us(
    fn: Callable[[], object],
    repeat: int,
    *,
    use_graph: bool,
    graph_iters: int,
    graph_fn: Callable[[], object] | None = None,
) -> tuple[float, bool]:
    if use_graph and graph_fn is not None:
        return _bench_graph_us(graph_fn, repeat, graph_iters), True
    return _bench_event_us(fn, repeat), False


def _kernel_role(kernel_name: str) -> str | None:
    name = kernel_name.lower()
    if any(pattern in name for pattern in COMBINE_KERNEL_PATTERNS):
        return "combine"
    # TileLang lowers the second T.Kernel in a prim_func to *_kernel_1.
    # In scheduled decode that second launch is the DSA combine stage.
    if name.endswith("_kernel_1") and "dsa_decode" in name:
        return "combine"
    if any(pattern in name for pattern in MAIN_KERNEL_PATTERNS):
        return "split"
    if any(pattern in name for pattern in UNEXPECTED_MAIN_KERNEL_PATTERNS):
        return "unexpected-main"
    return None


def _bench_kernel_us(
    fn: Callable[[], object],
    repeat: int,
    *,
    roles: set[str] | None = None,
) -> tuple[float, list[tuple[str, str, float]]]:
    fn()
    _sync()
    with torch.profiler.profile(activities=[_profiler_activity()]) as prof:
        for _ in range(repeat):
            fn()
            prof.step()
    rows: list[tuple[str, str, float]] = []
    for evt in prof.key_averages():
        role = _kernel_role(evt.key)
        if role is None:
            continue
        if roles is not None and role not in roles:
            continue
        total_us = float(
            getattr(evt, "device_time_total", 0.0)
            or getattr(evt, "cuda_time_total", 0.0)
            or 0.0
        )
        if total_us > 0:
            rows.append((role, evt.key, total_us / repeat))
    rows.sort(key=lambda item: item[2], reverse=True)
    return sum(time_us for _, _, time_us in rows), rows


def _bench_impl_kernel_us(
    fn: Callable[[], object],
    repeat: int,
) -> tuple[float, list[tuple[str, str, float]]]:
    # Device-body-only implementation time: only the sparse MLA split/combine
    # kernels, excluding graph replay helpers and framework fill kernels.
    return _bench_kernel_us(fn, max(2, repeat), roles={"split", "combine"})


def _make_indices(rows: int, topk: int, seq_len_kv: int, device: str) -> torch.Tensor:
    indices = torch.full((rows, 1, topk), -1, dtype=torch.int32, device=device)
    valid = min(topk, seq_len_kv)
    base = torch.arange(valid, dtype=torch.int32, device=device)
    for row in range(rows):
        offset = (row * 131) % max(seq_len_kv, 1)
        indices[row, 0, :valid] = (base + offset) % seq_len_kv
    return indices


def _make_temp_prefill_indices(
    seq_len_q: int,
    topk: int,
    seq_len_kv: int,
    device: str,
) -> torch.Tensor:
    indices = torch.full((seq_len_q, 1, topk), -1, dtype=torch.int32, device=device)
    for row in range(seq_len_q):
        valid = torch.randperm(max(1, min(row, seq_len_kv)), device=device)[:topk]
        indices[row, 0, : valid.numel()] = valid
    return indices


def _make_temp_decode_indices(
    batch_size: int,
    seq_len_q: int,
    topk: int,
    cache_seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    device: str,
) -> torch.Tensor:
    indices = torch.full(
        (batch_size, seq_len_q, 1, topk), -1, dtype=torch.int32, device=device
    )
    for batch in range(batch_size):
        cur_len = int(cache_seqlens[batch].item())
        base = int(cu_seqlens[batch].item())
        for row in range(seq_len_q):
            valid = torch.randperm(cur_len, device=device)[:topk] + base
            indices[batch, row, 0, : valid.numel()] = valid
    return indices


def _mask_by_topk_length(
    indices: torch.Tensor, topk_length: torch.Tensor | None
) -> torch.Tensor:
    if topk_length is None:
        return indices
    topk = indices.shape[-1]
    masked = indices.clone()
    if indices.dim() == 3:
        pos = torch.arange(topk, device=indices.device).view(1, 1, topk)
        mask = pos >= topk_length.view(-1, 1, 1)
    elif indices.dim() == 4:
        pos = torch.arange(topk, device=indices.device).view(1, 1, 1, topk)
        mask = pos >= topk_length.view(-1, 1, 1, 1)
    else:
        raise AssertionError(f"unsupported indices ndim: {indices.dim()}")
    masked = torch.where(mask, masked.new_full((), -1), masked)
    return masked


def _format_perf(
    stats: WorkloadStats,
    e2e_us: float,
    kernel_path_us: float,
    impl_kernel_us: float | None = None,
) -> str:
    perf = (
        f"compute/mem={stats.compute_memory_ratio:.2f}, "
        f"e2e={stats.tflops(e2e_us):.1f} TFLOPS/{stats.gbps(e2e_us):.0f} GB/s, "
        f"kernel_path={stats.tflops(kernel_path_us):.1f} TFLOPS/"
        f"{stats.gbps(kernel_path_us):.0f} GB/s"
    )
    if impl_kernel_us is not None:
        perf += (
            f", impl_kernel={stats.tflops(impl_kernel_us):.1f} TFLOPS/"
            f"{stats.gbps(impl_kernel_us):.0f} GB/s"
        )
    return perf


def _kernel_path_name(used_graph: bool) -> str:
    return "kernel_path(graph)" if used_graph else "kernel_path"


def _compile_with_explicit_outputs(jit_func, *args, **kwargs):
    """Compile a TileLang prim_func variant whose outputs are caller-owned."""
    import tilelang

    prim_func = jit_func.get_tir(*args, **kwargs)
    return tilelang.compile(
        prim_func,
        out_idx=[],
        execution_backend=jit_func.execution_backend,
        target=jit_func.target,
        target_host=jit_func.target_host,
        verbose=False,
        pass_configs=jit_func.pass_configs,
        instruments=jit_func.instruments,
        compile_flags=jit_func.compile_flags,
    )


def _parse_int_list(value: str) -> list[int]:
    vals = [int(item) for item in value.split(",") if item.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    if any(val <= 0 for val in vals):
        raise argparse.ArgumentTypeError("all values must be positive")
    return vals


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _prefill_stats(
    case: PrefillCase,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
) -> WorkloadStats:
    # Mirrors FlashMLA/tests/lib.py::count_flop_and_mem_vol.
    total_topk = (
        case.seq_len_q * case.topk
        if topk_length is None
        else int(topk_length.sum().item())
    )
    valid_mask = (indices >= 0) & (indices < case.seq_len_kv)
    if topk_length is not None:
        topk = indices.shape[-1]
        pos = torch.arange(topk, device=indices.device).view(1, 1, topk)
        valid_mask = valid_mask & (pos < topk_length.view(case.seq_len_q, 1, 1))
    num_valid_indices = int(valid_mask.sum().item())

    extra_topk = case.extra_topk if case.extra_topk > 0 else 0
    if extra_topk > 0:
        total_topk += case.seq_len_q * extra_topk
        num_valid_indices += case.seq_len_q * min(extra_topk, case.extra_seq_len_kv)

    flop = 2 * total_topk * case.num_heads * (case.d_qk + DV)
    mem_bytes = (
        num_valid_indices * case.d_qk * 2
        + case.seq_len_q * case.num_heads * (case.d_qk + DV) * 2
    )
    return WorkloadStats(float(flop), float(mem_bytes))


def _decode_stats(
    case: DecodeCase,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_topk_length: torch.Tensor | None,
) -> WorkloadStats:
    # Mirrors FlashMLA/tests/lib.py::count_flop_and_mem_vol_for_decode.
    def attended_tokens(topk: int, lengths: torch.Tensor | None) -> int:
        if lengths is None:
            return case.batch_size * case.seq_len_q * topk
        return int(lengths.sum().item()) * case.seq_len_q

    def retrieved_tokens(
        cur_indices: torch.Tensor, lengths: torch.Tensor | None
    ) -> int:
        masked = _mask_by_topk_length(cur_indices, lengths)
        # Official FlashMLA counts unique gathered entries and includes the
        # invalid sentinel if present; keep that convention for comparable stats.
        return int(masked.unique().numel())

    num_attended = attended_tokens(case.topk, topk_length)
    num_retrieved = retrieved_tokens(indices, topk_length)
    if extra_indices is not None:
        num_attended += attended_tokens(case.extra_topk, extra_topk_length)
        num_retrieved += retrieved_tokens(extra_indices, extra_topk_length)

    flop = 2 * case.num_heads * num_attended * (case.d_qk + DV)
    if case.native_fp8:
        kv_token_size = 576
    elif case.d_qk == 512:
        kv_token_size = 584
    else:
        kv_token_size = 656
    q_element_size = 1 if case.native_fp8 else 2
    mem_bytes = sum(
        [
            q_element_size
            * case.batch_size
            * case.seq_len_q
            * case.num_heads
            * case.d_qk,
            num_retrieved * kv_token_size,
            2 * case.batch_size * case.seq_len_q * case.num_heads * DV,
        ]
    )
    return WorkloadStats(float(flop), float(mem_bytes))


def _make_v32_prefill_graph_runner(
    case: PrefillCase,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
) -> Callable[[], object]:
    from mate.sparse_mla.tilelang.sparse_mla_v32_fwd_pipelined import (
        sparse_attention_fwd_kernel,
    )

    seq_len, heads, dim_plus_tail_dim = q.shape
    dim = DV
    tail_dim = dim_plus_tail_dim - dim
    _, kv_group, _ = kv.shape
    topk = indices.shape[-1]
    kernel = _compile_with_explicit_outputs(
        sparse_attention_fwd_kernel,
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=kv_group,
        sm_scale=None,
        is_causal=False,
        threads=640,
        has_attn_sink=attn_sink is not None,
        has_topk_length=topk_length is not None,
        is_persistence=case.is_persistence,
        persistent_blocks=case.persistent_blocks,
    )
    out = torch.empty((seq_len, heads, dim), dtype=q.dtype, device=q.device)
    max_logits = torch.empty((seq_len, heads), dtype=torch.float32, device=q.device)
    lse = torch.empty((seq_len, heads), dtype=torch.float32, device=q.device)

    def run_graph():
        args = [q, kv, indices]
        args.extend((out, max_logits, lse))
        if topk_length is not None:
            args.append(topk_length)
        if attn_sink is not None:
            args.append(attn_sink)
        kernel(*args)
        return out, max_logits, lse

    return run_graph


def _make_model1_prefill_graph_runner(
    case: PrefillCase,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_kv: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_topk_length: torch.Tensor | None,
) -> Callable[[], object]:
    from mate.sparse_mla.tilelang.sparse_mla_model1_fwd_pipelined import (
        sparse_attention_fwd_kernel_model1,
    )

    seq_len, heads, dim = q.shape
    _, kv_group, _ = kv.shape
    assert extra_kv is None and extra_indices is None and extra_topk_length is None, (
        "MODEL1 prefill graph runner has no extra-KV ABI"
    )

    kernel_kwargs = {
        "kv_group": kv_group,
        "sm_scale": case.d_qk**-0.5,
        "is_causal": True,
        "threads": 640,
        "has_attn_sink": attn_sink is not None,
        "has_topk_length": topk_length is not None,
        "is_persistence": case.is_persistence,
        "persistent_blocks": case.persistent_blocks,
    }
    kernel = _compile_with_explicit_outputs(
        sparse_attention_fwd_kernel_model1,
        heads,
        dim,
        **kernel_kwargs,
    )
    out = torch.empty((seq_len, heads, dim), dtype=q.dtype, device=q.device)
    max_logits = torch.empty((seq_len, heads), dtype=torch.float32, device=q.device)
    lse = torch.empty((seq_len, heads), dtype=torch.float32, device=q.device)

    def run_graph():
        args = [q, kv, indices]
        args.extend((out, max_logits, lse))
        if topk_length is not None:
            args.append(topk_length)
        if attn_sink is not None:
            args.append(attn_sink)
        kernel(*args)
        return out, max_logits, lse

    return run_graph


def _run_prefill_case(
    case: PrefillCase,
    repeat: int,
    profile_kernels: bool,
    use_graph: bool,
    graph_iters: int,
) -> None:
    device = _device()
    torch.manual_seed(20260502 + case.d_qk + case.seq_len_kv)
    q = torch.randn(
        (case.seq_len_q, case.num_heads, case.d_qk),
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.randn(
        (case.seq_len_kv, 1, case.d_qk), dtype=torch.bfloat16, device=device
    )
    if case.index_pattern == "temp_causal":
        indices = _make_temp_prefill_indices(
            case.seq_len_q, case.topk, case.seq_len_kv, device
        )
    else:
        indices = _make_indices(case.seq_len_q, case.topk, case.seq_len_kv, device)
    topk_length = None
    if case.topk_length:
        topk_length = torch.full(
            (case.seq_len_q,), case.topk, dtype=torch.int32, device=device
        )
        topk_length[::17] = max(1, case.topk // 2)
    attn_sink = (
        torch.randn((case.num_heads,), dtype=torch.float32, device=device)
        if case.attn_sink
        else None
    )

    extra_kv = None
    extra_indices = None
    extra_topk_length = None
    if case.extra_topk > 0:
        extra_kv = torch.randn(
            (case.extra_seq_len_kv, 1, case.d_qk), dtype=torch.bfloat16, device=device
        )
        extra_indices = _make_indices(
            case.seq_len_q, case.extra_topk, case.extra_seq_len_kv, device
        )

    if case.direct_tilelang:
        if case.d_qk == 512:
            if extra_kv is not None:
                raise AssertionError(
                    "MODEL1 direct prefill no longer has an extra-KV ABI"
                )
            from mate.sparse_mla.tilelang.sparse_mla_model1_fwd_pipelined import (
                sparse_mla_fwd_interface_model1,
            )

            def run():
                return sparse_mla_fwd_interface_model1(
                    q=q,
                    kv=kv,
                    indices=indices,
                    topk_length=topk_length,
                    sm_scale=case.d_qk**-0.5,
                    attn_sink=attn_sink,
                    d_v=DV,
                    is_persistence=case.is_persistence,
                    persistent_blocks=case.persistent_blocks,
                )
        elif case.d_qk == 576:
            if extra_kv is not None:
                raise AssertionError("V3.2 direct prefill has no extra-KV ABI")
            from mate.sparse_mla.tilelang.sparse_mla_v32_fwd_pipelined import (
                tilelang_sparse_mla_prefill_fwd_interface,
            )

            def run():
                return tilelang_sparse_mla_prefill_fwd_interface(
                    q,
                    kv,
                    indices,
                    sm_scale=None,
                    topk_length=topk_length,
                    attn_sink=attn_sink,
                    d_v=DV,
                    return_max_logits=True,
                    is_persistence=case.is_persistence,
                    persistent_blocks=case.persistent_blocks,
                )
        else:
            raise AssertionError(
                f"unsupported direct TileLang prefill d_qk={case.d_qk}"
            )
    else:
        if case.extra_topk > 0:
            raise AssertionError(
                "FlashMLA wrapper sparse prefill has no extra-KV ABI; use direct_tilelang"
            )

        def run():
            return flash_mla.flash_mla_sparse_fwd(
                q=q,
                kv=kv,
                indices=indices,
                sm_scale=case.d_qk**-0.5,
                d_v=DV,
                attn_sink=attn_sink,
                topk_length=topk_length,
            )

    stats = _prefill_stats(case, indices, topk_length)
    e2e_us = _bench_event_us(run, repeat)
    run_graph = None
    if use_graph and case.d_qk == 576:
        run_graph = _make_v32_prefill_graph_runner(
            case, q, kv, indices, topk_length, attn_sink
        )
    elif use_graph and case.d_qk == 512:
        run_graph = _make_model1_prefill_graph_runner(
            case,
            q,
            kv,
            indices,
            topk_length,
            attn_sink,
            extra_kv,
            extra_indices,
            extra_topk_length,
        )
    impl_kernel_us, impl_kernel_rows = _bench_impl_kernel_us(
        run_graph if run_graph is not None else run, repeat
    )
    kernel_path_us, used_graph = _bench_kernel_path_us(
        run,
        repeat,
        use_graph=use_graph,
        graph_iters=graph_iters,
        graph_fn=run_graph,
    )
    label = "prefill-direct" if case.direct_tilelang else "prefill"
    if case.d_qk in (512, 576) and case.is_persistence:
        label += f"-persist{case.persistent_blocks}"
    print(
        f"{label} {case.name}: e2e={e2e_us:.1f} us, "
        f"{_kernel_path_name(used_graph)}={kernel_path_us:.1f} us, "
        f"impl_kernel_us={impl_kernel_us:.1f} us, "
        f"{_format_perf(stats, e2e_us, kernel_path_us, impl_kernel_us)}"
    )
    if profile_kernels:
        print(f"  impl kernels={impl_kernel_us:.1f} us")
        for role, name, time_us in impl_kernel_rows:
            print(f"  {time_us:8.1f} us  [{role}] {name}")
        kernel_us, kernel_rows = _bench_kernel_us(run, max(2, repeat // 2))
        print(f"  profiled kernels={kernel_us:.1f} us")
        for role, name, time_us in kernel_rows:
            print(f"  {time_us:8.1f} us  [{role}] {name}")


def _make_decode_inputs(case: DecodeCase):
    device = _device()
    torch.manual_seed(20260503 + case.d_qk + case.batch_size)
    variant = "v4" if case.d_qk == 512 else "v32"
    q = torch.randn(
        (case.batch_size, case.seq_len_q, case.num_heads, case.d_qk),
        dtype=torch.bfloat16,
        device=device,
    )
    num_pages = (case.seq_len_kv + case.page_size - 1) // case.page_size
    kv = torch.randn(
        (num_pages, case.page_size, 1, case.d_qk), dtype=torch.bfloat16, device=device
    )
    k_cache = quantize_sparse_mla_cache(kv, variant)
    indices = _make_indices(
        case.batch_size * case.seq_len_q, case.topk, case.seq_len_kv, device
    ).view(case.batch_size, case.seq_len_q, 1, case.topk)
    topk_length = None
    if case.topk_length:
        topk_length = torch.full(
            (case.batch_size,), case.topk, dtype=torch.int32, device=device
        )
        topk_length[::5] = max(1, case.topk // 2)
    attn_sink = (
        torch.randn((case.num_heads,), dtype=torch.float32, device=device)
        if case.attn_sink
        else None
    )

    extra_k_cache = None
    extra_indices = None
    extra_topk_length = None
    if case.extra_topk > 0:
        extra_pages = (
            case.extra_seq_len_kv + case.extra_page_size - 1
        ) // case.extra_page_size
        extra_kv = torch.randn(
            (extra_pages, case.extra_page_size, 1, case.d_qk),
            dtype=torch.bfloat16,
            device=device,
        )
        extra_k_cache = quantize_sparse_mla_cache(extra_kv, variant)
        extra_indices = _make_indices(
            case.batch_size * case.seq_len_q,
            case.extra_topk,
            case.extra_seq_len_kv,
            device,
        ).view(case.batch_size, case.seq_len_q, 1, case.extra_topk)
        if case.extra_topk_length:
            extra_topk_length = torch.full(
                (case.batch_size,), case.extra_topk, dtype=torch.int32, device=device
            )
            extra_topk_length[::7] = max(1, case.extra_topk // 2)

    return (
        q,
        k_cache,
        indices,
        topk_length,
        attn_sink,
        extra_k_cache,
        extra_indices,
        extra_topk_length,
    )


def _make_native_fp8_decode_inputs(case: DecodeCase, use_graph: bool):
    if case.d_qk != 576:
        raise AssertionError("native FP8 decode requires d_qk=576")
    if case.extra_topk > 0 or case.attn_sink:
        raise AssertionError("native FP8 decode does not support extra KV or sinks")

    device = _device()
    torch.manual_seed(20260722 + case.batch_size + case.topk)
    q = (
        torch.randn(
            (case.batch_size, case.seq_len_q, case.num_heads, case.d_qk),
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    num_pages = (case.seq_len_kv + case.page_size - 1) // case.page_size
    kv_cache = (
        torch.randn(
            (num_pages, case.page_size, case.d_qk),
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    kv_cache = kv_cache.unsqueeze(1)
    indices = _make_indices(
        case.batch_size * case.seq_len_q,
        case.topk,
        case.seq_len_kv,
        device,
    ).view(case.batch_size, case.seq_len_q, case.topk)
    # The native FP8 API requires seq_lens metadata even when every row uses
    # the full sparse topk.  Default cases perturb selected rows; full-topk
    # cases keep every entry equal to topk.
    topk_length = torch.full(
        (case.batch_size,), case.topk, dtype=torch.int32, device=device
    )
    if case.topk_length:
        topk_length[::5] = max(1, case.topk // 2)
    out = torch.empty(
        (case.batch_size, case.seq_len_q, case.num_heads, DV),
        dtype=torch.bfloat16,
        device=device,
    )
    return q, kv_cache, indices, topk_length, out


def _make_v32_temp_decode_inputs(case: DecodeCase):
    if case.d_qk != 576:
        raise AssertionError("temp-aligned direct decode is V3.2-only")
    if case.extra_topk > 0 or case.attn_sink or case.topk_length:
        raise AssertionError(
            "temp-aligned V3.2 decode disables extra/topk_length/attn_sink"
        )

    device = _device()
    torch.manual_seed(20260503 + case.d_qk + case.batch_size)
    q = torch.randn(
        (case.batch_size, case.seq_len_q, case.num_heads, case.d_qk),
        dtype=torch.bfloat16,
        device=device,
    )
    cache_seqlens = torch.tensor(
        [case.seq_len_kv - 4 * i for i in range(case.batch_size)],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens = torch.tensor(
        [0] + [case.seq_len_kv - 4 * i for i in range(case.batch_size)],
        dtype=torch.int32,
        device=device,
    ).cumsum(dim=0, dtype=torch.int32)
    total_seqlens = int(cache_seqlens.sum().item())
    kv = torch.randn((total_seqlens, 1, case.d_qk), dtype=torch.bfloat16, device=device)
    indices = _make_temp_decode_indices(
        case.batch_size,
        case.seq_len_q,
        case.topk,
        cache_seqlens,
        cu_seqlens,
        device,
    )

    quant_scales = torch.tensor(
        [0.6, 0.7, 0.8, 0.9], dtype=torch.float32, device=device
    ).view(1, 1, 4)
    quant_scales = quant_scales.repeat_interleave(total_seqlens, dim=0)
    k_latent_fp8 = kv[..., :DV].to(torch.float8_e4m3fn).contiguous()
    k_pe = kv[..., DV:].to(torch.bfloat16).contiguous()
    k_cache = torch.cat(
        [
            k_latent_fp8.view(torch.uint8),
            quant_scales.view(torch.uint8),
            k_pe.view(torch.uint8),
        ],
        dim=-1,
    ).contiguous()

    from mate.sparse_mla.tilelang.temp.sparse_mla_v32_decode_fwd_scheduled import (
        get_mla_metadata_pytorch,
    )

    sched_meta, num_splits = get_mla_metadata_pytorch(
        cache_seqlens,
        num_q_tokens_per_head_k=case.seq_len_q * case.num_heads,
        num_heads_k=1,
        num_heads_q=case.num_heads,
        topk=case.topk,
        mp_count=case.mp_count,
        TILE_M=case.tile_m,
    )
    topk_length = None
    return q, k_cache, indices, sched_meta, num_splits, topk_length


def _make_v32_decode_graph_runner(
    case: DecodeCase,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    sched_meta: torch.Tensor,
    num_splits: torch.Tensor,
) -> Callable[[], object]:
    from mate.sparse_mla.tilelang.sparse_mla_v32_decode_fwd_scheduled import (
        sparse_attention_fwd_kernel,
    )
    from mate.sparse_mla.tilelang.sparse_mla_decode_scheduled_common import (
        prepare_scheduled_decode_runtime,
    )

    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    dim = DV
    tail_dim = dim_plus_tail_dim - dim
    kv_flat = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
    kv_group = kv_flat.shape[1]
    support_split = int(sched_meta.shape[0]) != 1
    runtime = prepare_scheduled_decode_runtime(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        topk_length=topk_length,
        attn_sink=attn_sink,
        tile_scheduler_metadata=sched_meta,
        num_splits=num_splits,
        out_dtype=q.dtype,
        device=q.device,
        variant_name="V3.2",
        support_split=support_split,
    )
    kernel = sparse_attention_fwd_kernel(
        heads,
        dim,
        tail_dim,
        indices.shape[-1],
        kv_group=kv_group,
        sm_scale=case.d_qk**-0.5,
        threads=640,
        max_nums_splits=runtime.max_nums_splits,
        has_attn_sink=runtime.has_attn_sink,
        has_topk_length=runtime.topk_length is not None,
        support_split=support_split,
    )
    kv_latent_f8 = kv_flat.view(torch.float8_e4m3fn)
    k_rope = kv_flat.view(torch.bfloat16)
    scales = kv_flat.view(torch.float32)

    def run_graph():
        args = [q, kv_latent_f8, k_rope, scales, indices]
        if runtime.topk_length is not None:
            args.append(runtime.topk_length)
        if runtime.attn_sink is not None:
            args.append(runtime.attn_sink)
        args.extend((sched_meta, num_splits))
        if support_split:
            assert runtime.glse is not None and runtime.out_partial is not None
            args.extend((runtime.glse, runtime.out_partial))
        args.extend((runtime.out, runtime.lse))
        kernel(*args)
        return runtime.out, runtime.lse

    return run_graph


def _make_model1_decode_graph_runner(
    case: DecodeCase,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_k_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_topk_length: torch.Tensor | None,
    sched_meta: torch.Tensor,
    num_splits: torch.Tensor,
) -> Callable[[], object]:
    from mate.sparse_mla.flashmla_checks import model1_cache_page_views
    from mate.sparse_mla.tilelang.sparse_mla_decode_scheduled_common import (
        prepare_scheduled_decode_runtime,
    )
    from mate.sparse_mla.tilelang.sparse_mla_model1_decode_fwd_scheduled import (
        sparse_attention_decode_fwd_scheduled_kernel_model1,
    )

    batch, seq_len, heads, _ = q.shape
    kv_nope, kv_rope, kv_scales = model1_cache_page_views(k_cache)
    has_extra = extra_k_cache is not None
    assert (extra_indices is not None) == has_extra
    assert has_extra or extra_topk_length is None
    if has_extra:
        extra_kv_nope, extra_kv_rope, extra_kv_scales = model1_cache_page_views(
            extra_k_cache
        )
    else:
        extra_kv_nope = None
        extra_kv_rope = None
        extra_kv_scales = None
    support_split = int(sched_meta.shape[0]) != 1
    runtime = prepare_scheduled_decode_runtime(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=DV,
        topk_length=topk_length,
        attn_sink=attn_sink,
        tile_scheduler_metadata=sched_meta,
        num_splits=num_splits,
        out_dtype=q.dtype,
        device=q.device,
        variant_name="MODEL1",
        support_split=support_split,
    )
    kernel = sparse_attention_decode_fwd_scheduled_kernel_model1(
        heads,
        DV,
        has_extra=has_extra,
        kv_group=1,
        sm_scale=case.d_qk**-0.5,
        threads=640,
        max_nums_splits=runtime.max_nums_splits,
        has_attn_sink=runtime.has_attn_sink,
        has_topk_length=runtime.topk_length is not None,
        has_extra_topk_length=extra_topk_length is not None,
        page_block_size=k_cache.shape[1],
        extra_page_block_size=(
            extra_k_cache.shape[1] if extra_k_cache is not None else None
        ),
        use_8byte_kv_loads=(
            kv_rope.stride(0) % 8 != 0
            or (extra_kv_rope is not None and extra_kv_rope.stride(0) % 8 != 0)
        ),
        support_split=support_split,
    )

    def run_graph():
        args = [q, kv_nope, kv_rope, kv_scales, indices]
        if runtime.topk_length is not None:
            args.append(runtime.topk_length)
        if has_extra:
            args.extend((extra_kv_nope, extra_kv_rope, extra_kv_scales, extra_indices))
            if extra_topk_length is not None:
                args.append(extra_topk_length)
        if runtime.attn_sink is not None:
            args.append(runtime.attn_sink)
        args.extend((sched_meta, num_splits))
        if support_split:
            assert runtime.glse is not None and runtime.out_partial is not None
            args.extend((runtime.glse, runtime.out_partial))
        args.extend((runtime.out, runtime.lse))
        kernel(*args)
        return runtime.out, runtime.lse

    return run_graph


def _run_decode_case(
    case: DecodeCase,
    repeat: int,
    profile_kernels: bool,
    use_graph: bool,
    graph_iters: int,
) -> None:
    if case.native_fp8:
        q, kv_cache, indices, topk_length, out = _make_native_fp8_decode_inputs(
            case, use_graph
        )
        stats = _decode_stats(case, indices.unsqueeze(2), topk_length, None, None)

        def run_native():
            return sparse_mla_fp8_decode(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=None,
                qk_nope_head_dim=128,
                kv_lora_rank=DV,
                qk_rope_head_dim=64,
                block_tables=indices,
                seq_lens=topk_length,
                max_seq_len=case.seq_len_kv,
                sparse_mla_top_k=case.topk,
                out=out,
                bmm1_scale=case.d_qk**-0.5,
            )

        metadata = get_batch_decode_metadata_mla(
            q,
            topk_length,
            case.topk,
        )

        def run_native_reusing_metadata():
            return sparse_mla_fp8_decode(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=None,
                qk_nope_head_dim=128,
                kv_lora_rank=DV,
                qk_rope_head_dim=64,
                block_tables=indices,
                seq_lens=topk_length,
                max_seq_len=case.seq_len_kv,
                sparse_mla_top_k=case.topk,
                out=out,
                bmm1_scale=case.d_qk**-0.5,
                metadata=metadata,
            )

        run_native()
        run_native_reusing_metadata()
        _sync()
        auto_e2e_us = _bench_event_us(run_native, repeat)
        reused_e2e_us = _bench_event_us(run_native_reusing_metadata, repeat)
        impl_kernel_us, impl_kernel_rows = _bench_impl_kernel_us(
            run_native_reusing_metadata, repeat
        )
        kernel_path_us, used_graph = _bench_kernel_path_us(
            run_native_reusing_metadata,
            repeat,
            use_graph=use_graph,
            graph_iters=graph_iters,
            graph_fn=run_native_reusing_metadata if use_graph else None,
        )
        print(
            f"decode-native-fp8 {case.name}: "
            f"e2e(auto-metadata)={auto_e2e_us:.1f} us, "
            f"e2e(reused-metadata)={reused_e2e_us:.1f} us, "
            f"{_kernel_path_name(used_graph)}(reused-metadata)="
            f"{kernel_path_us:.1f} us, "
            f"impl_kernel_us={impl_kernel_us:.1f} us, "
            f"{_format_perf(stats, reused_e2e_us, kernel_path_us, impl_kernel_us)}"
        )
        if profile_kernels:
            print(f"  impl kernels={impl_kernel_us:.1f} us")
            for role, name, time_us in impl_kernel_rows:
                print(f"  {time_us:8.1f} us  [{role}] {name}")
            kernel_us, kernel_rows = _bench_kernel_us(
                run_native_reusing_metadata, max(2, repeat // 2)
            )
            print(f"  profiled kernels={kernel_us:.1f} us")
            for role, name, time_us in kernel_rows:
                print(f"  {time_us:8.1f} us  [{role}] {name}")
        return

    if case.direct_tilelang:
        q, k_cache, indices, sched_meta, num_splits, topk_length = (
            _make_v32_temp_decode_inputs(case)
        )
        stats = _decode_stats(case, indices, topk_length, None, None)

        from mate.sparse_mla.tilelang.sparse_mla_v32_decode_fwd_scheduled import (
            tilelang_flashmla_interface,
        )

        def run_direct_kernel_path():
            return tilelang_flashmla_interface(
                q,
                k_cache,
                indices,
                sched_meta,
                num_splits,
                sm_scale=None,
                topk_length=topk_length,
                attn_sink=None,
                d_v=DV,
                threads=512,
            )

        run_graph = (
            _make_v32_decode_graph_runner(
                case, q, k_cache, indices, topk_length, None, sched_meta, num_splits
            )
            if use_graph
            else None
        )
        run_direct_kernel_path()
        _sync()
        impl_kernel_us, impl_kernel_rows = _bench_impl_kernel_us(
            run_graph if run_graph is not None else run_direct_kernel_path, repeat
        )
        kernel_path_us, used_graph = _bench_kernel_path_us(
            run_direct_kernel_path,
            repeat,
            use_graph=use_graph,
            graph_iters=graph_iters,
            graph_fn=run_graph,
        )
        max_splits = int(torch.diff(num_splits).max().item())
        print(
            f"decode-direct {case.name}: {_kernel_path_name(used_graph)}={kernel_path_us:.1f} us, "
            f"impl_kernel_us={impl_kernel_us:.1f} us, "
            f"mp_parts={sched_meta.shape[0]}, max_splits={max_splits}, "
            f"{_format_perf(stats, kernel_path_us, kernel_path_us, impl_kernel_us)}"
        )
        if profile_kernels:
            print(f"  impl kernels={impl_kernel_us:.1f} us")
            for role, name, time_us in impl_kernel_rows:
                print(f"  {time_us:8.1f} us  [{role}] {name}")
            kernel_us, kernel_rows = _bench_kernel_us(
                run_direct_kernel_path, max(2, repeat // 2)
            )
            print(f"  profiled kernels={kernel_us:.1f} us")
            for role, name, time_us in kernel_rows:
                print(f"  {time_us:8.1f} us  [{role}] {name}")
        return

    inputs = _make_decode_inputs(case)
    (
        q,
        k_cache,
        indices,
        topk_length,
        attn_sink,
        extra_k_cache,
        extra_indices,
        extra_topk_length,
    ) = inputs
    stats = _decode_stats(case, indices, topk_length, extra_indices, extra_topk_length)

    def run_with_new_metadata():
        sched_meta, num_splits = flash_mla.get_mla_metadata()
        return flash_mla.flash_mla_with_kvcache(
            q=q,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=DV,
            tile_scheduler_metadata=sched_meta,
            num_splits=num_splits,
            softmax_scale=case.d_qk**-0.5,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
        )

    sched_meta, num_splits = flash_mla.get_mla_metadata()

    def run_kernel_path():
        return flash_mla.flash_mla_with_kvcache(
            q=q,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=DV,
            tile_scheduler_metadata=sched_meta,
            num_splits=num_splits,
            softmax_scale=case.d_qk**-0.5,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
        )

    run_kernel_path()
    _sync()
    e2e_us = _bench_event_us(run_with_new_metadata, repeat)
    reused_metadata_us = _bench_event_us(run_kernel_path, repeat)
    run_graph = None
    if use_graph and case.d_qk == 512:
        run_graph = _make_model1_decode_graph_runner(
            case,
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
        )
    elif use_graph:
        run_graph = _make_v32_decode_graph_runner(
            case,
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
        )
    impl_kernel_us, impl_kernel_rows = _bench_impl_kernel_us(
        run_graph if run_graph is not None else run_kernel_path, repeat
    )
    kernel_path_us, used_graph = _bench_kernel_path_us(
        run_kernel_path,
        repeat,
        use_graph=use_graph,
        graph_iters=graph_iters,
        graph_fn=run_graph,
    )
    print(
        f"decode  {case.name}: e2e={e2e_us:.1f} us, "
        f"e2e(reused-metadata)={reused_metadata_us:.1f} us, "
        f"{_kernel_path_name(used_graph)}={kernel_path_us:.1f} us, "
        f"impl_kernel_us={impl_kernel_us:.1f} us, "
        f"{_format_perf(stats, e2e_us, kernel_path_us, impl_kernel_us)}"
    )
    if profile_kernels:
        print(f"  impl kernels={impl_kernel_us:.1f} us")
        for role, name, time_us in impl_kernel_rows:
            print(f"  {time_us:8.1f} us  [{role}] {name}")
        kernel_us, kernel_rows = _bench_kernel_us(run_kernel_path, max(2, repeat // 2))
        print(f"  profiled kernels={kernel_us:.1f} us")
        for role, name, time_us in kernel_rows:
            print(f"  {time_us:8.1f} us  [{role}] {name}")


def _prefill_cases(quick: bool) -> Iterable[PrefillCase]:
    skvs = [8192]
    templates = [(512, 64, 512), (512, 64, 2048), (576, 64, 512), (576, 64, 2048)]
    if quick:
        templates = [template for template in templates if template[-1] == 512]
    for d_qk, heads, topk in templates:
        for skv in skvs:
            yield PrefillCase(
                f"d{d_qk}_h{heads}_skv{skv}_topk{topk}", d_qk, heads, 4096, skv, topk
            )


def _custom_prefill_cases(
    *,
    d_qk: int,
    heads: int,
    sqs: Iterable[int],
    skv: int,
    topks: Iterable[int],
    full_topk: bool,
    attn_sink: bool,
) -> Iterable[PrefillCase]:
    for sq in sqs:
        for topk in topks:
            yield PrefillCase(
                f"d{d_qk}_h{heads}_sq{sq}_skv{skv}_topk{topk}",
                d_qk,
                heads,
                sq,
                skv,
                topk,
                topk_length=not full_topk,
                attn_sink=attn_sink,
            )


def _decode_cases(
    quick: bool,
    *,
    batches: Iterable[int] | None = None,
    heads: Iterable[int] | None = None,
    topks: Iterable[int] | None = None,
    full_topk: bool = False,
) -> Iterable[DecodeCase]:
    batches = list(batches) if batches is not None else [128]
    heads = list(heads) if heads is not None else [64]
    topks = list(topks) if topks is not None else ([512] if quick else [2048, 512])
    for bsz in batches:
        for num_heads in heads:
            for topk in topks:
                yield DecodeCase(
                    f"b{bsz}_d512_heads{num_heads}_topk{topk}",
                    512,
                    num_heads,
                    bsz,
                    1,
                    8192,
                    topk,
                    64,
                    topk_length=not full_topk,
                )
                yield DecodeCase(
                    f"b{bsz}_d576_heads{num_heads}_topk{topk}",
                    576,
                    num_heads,
                    bsz,
                    1,
                    8192,
                    topk,
                    64,
                    topk_length=not full_topk,
                    attn_sink=False,
                )
                yield DecodeCase(
                    f"b{bsz}_d576_native_fp8_heads{num_heads}_topk{topk}",
                    576,
                    num_heads,
                    bsz,
                    1,
                    8192,
                    topk,
                    64,
                    topk_length=not full_topk,
                    attn_sink=False,
                    native_fp8=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument("--repeat", type=_positive_int, default=20)
    parser.add_argument(
        "--quick", action="store_true", help="Run one size per family for smoke checks"
    )
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="Also print per-kernel profiler names/times; impl_kernel_us is always collected",
    )
    parser.add_argument(
        "--prefill-sqs",
        type=_parse_int_list,
        default=None,
        help="Comma-separated prefill seq_len_q override, e.g. 1,128,256,2048",
    )
    parser.add_argument(
        "--prefill-topks",
        type=_parse_int_list,
        default=None,
        help="Comma-separated prefill topk override, e.g. 64,512,2048",
    )
    parser.add_argument(
        "--prefill-d-qk",
        type=int,
        default=576,
        help="d_qk for --prefill-sqs/--prefill-topks custom prefill sweep",
    )
    parser.add_argument(
        "--prefill-heads",
        type=int,
        default=64,
        help="num_heads for --prefill-sqs/--prefill-topks custom prefill sweep",
    )
    parser.add_argument(
        "--prefill-skv",
        type=int,
        default=8192,
        help="seq_len_kv for --prefill-sqs/--prefill-topks custom prefill sweep",
    )
    parser.add_argument(
        "--prefill-full-topk",
        action="store_true",
        help="Use full topk length for every prefill row instead of the default variable-length perturbation",
    )
    parser.add_argument(
        "--prefill-attn-sink",
        action="store_true",
        help="Enable attention sink for custom prefill cases",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable graph replay for kernel-path timing",
    )
    parser.add_argument(
        "--decode-layout",
        choices=("all", "quantized-cache", "native-fp8", "v32-compare"),
        default="all",
        help=(
            "Filter decode cases by input layout; v32-compare runs the 656-byte "
            "cache and native FP8 V3.2 paths"
        ),
    )
    parser.add_argument(
        "--decode-batches",
        type=_parse_int_list,
        default=None,
        help="Comma-separated decode batch sizes; default: 128",
    )
    parser.add_argument(
        "--decode-heads",
        type=_parse_int_list,
        default=None,
        help="Comma-separated decode head counts; default: 64",
    )
    parser.add_argument(
        "--decode-topks",
        type=_parse_int_list,
        default=None,
        help="Comma-separated decode topk values; default: 2048,512",
    )
    parser.add_argument(
        "--decode-full-topk",
        action="store_true",
        help="Use the full decode topk for every batch row",
    )
    parser.add_argument(
        "--graph-iters",
        type=_positive_int,
        default=4,
        help="Function calls captured inside one graph replay",
    )
    parser.add_argument(
        "--model1-prefill-persistent-blocks",
        type=int,
        default=(
            int(os.environ["MATE_MODEL1_PREFILL_PERSISTENT_BLOCKS"])
            if "MATE_MODEL1_PREFILL_PERSISTENT_BLOCKS" in os.environ
            else None
        ),
        help="Override physical CTA count for MODEL1 persistent prefill paths",
    )
    parser.add_argument(
        "--no-model1-prefill-persistence",
        action="store_true",
        help="Disable MODEL1 persistent prefill in benchmark direct/graph paths",
    )
    args = parser.parse_args()

    device = _device()
    torch.set_default_device(device)
    torch.set_float32_matmul_precision("high")
    use_graph = not args.no_graph
    model1_prefill_persistent_blocks = resolve_num_mps(
        device, args.model1_prefill_persistent_blocks
    )
    print(
        f"device={device}, repeat={args.repeat}, quick={args.quick}, "
        f"graph={use_graph}, graph_iters={args.graph_iters}, "
        f"model1_prefill_persistence={not args.no_model1_prefill_persistence}, "
        f"model1_prefill_persistent_blocks={model1_prefill_persistent_blocks}"
    )

    if args.mode in ("prefill", "both"):
        custom_prefill = args.prefill_sqs is not None or args.prefill_topks is not None
        if custom_prefill:
            prefill_cases = _custom_prefill_cases(
                d_qk=args.prefill_d_qk,
                heads=args.prefill_heads,
                sqs=args.prefill_sqs or [4096],
                skv=args.prefill_skv,
                topks=args.prefill_topks or [512, 2048],
                full_topk=args.prefill_full_topk,
                attn_sink=args.prefill_attn_sink,
            )
        else:
            prefill_cases = _prefill_cases(args.quick)
        for prefill_case in prefill_cases:
            if prefill_case.d_qk == 512:
                prefill_case = dataclasses.replace(
                    prefill_case,
                    direct_tilelang=True,
                    is_persistence=not args.no_model1_prefill_persistence,
                    persistent_blocks=model1_prefill_persistent_blocks,
                )
            elif prefill_case.d_qk == 576:
                prefill_case = dataclasses.replace(
                    prefill_case,
                    is_persistence=not args.no_model1_prefill_persistence,
                    persistent_blocks=model1_prefill_persistent_blocks,
                )
            _run_prefill_case(
                prefill_case,
                args.repeat,
                args.profile_kernels,
                use_graph,
                args.graph_iters,
            )
    if args.mode in ("decode", "both"):
        for decode_case in _decode_cases(
            args.quick,
            batches=args.decode_batches,
            heads=args.decode_heads,
            topks=args.decode_topks,
            full_topk=args.decode_full_topk,
        ):
            if args.decode_layout == "quantized-cache" and decode_case.native_fp8:
                continue
            if args.decode_layout == "native-fp8" and not decode_case.native_fp8:
                continue
            if args.decode_layout == "v32-compare" and decode_case.d_qk != 576:
                continue
            _run_decode_case(
                decode_case,
                args.repeat,
                args.profile_kernels,
                use_graph,
                args.graph_iters,
            )
    _sync()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import os
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    # MUSA graph / TileLang objects can race during interpreter teardown in this
    # benchmark-only path. Flush output and let the OS reclaim graph resources.
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
