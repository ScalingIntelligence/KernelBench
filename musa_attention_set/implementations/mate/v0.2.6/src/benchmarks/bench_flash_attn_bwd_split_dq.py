#!/usr/bin/env python3
"""Focused correctness and latency check for the split-DQ backward kernel."""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch_musa  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mate import flash_attn_varlen_func  # noqa: E402
from mate.flash_attention.tilelang._flash_attention_bwd_post import (  # noqa: E402
    compute_delta_ws,
    to_tilelang_dtype,
)
from mate.flash_attention.tilelang._flash_attention_bwd_split_dq import (  # noqa: E402
    flashattn_bwd_ws_split_dq,
)


_GRAPH_KEEPALIVE: list[object] = []


@dataclasses.dataclass(frozen=True)
class DQCase:
    batch: int
    heads: int
    total_seq_q: int
    total_seq_kv: int
    dim: int
    dtype: torch.dtype
    causal: bool
    block_m: int
    block_n: int

    @property
    def max_seq_q(self) -> int:
        return self.total_seq_q // self.batch

    @property
    def max_seq_kv(self) -> int:
        return self.total_seq_kv // self.batch

    @property
    def dq_flops(self) -> float:
        factor = 3 if self.causal else 6
        return (
            self.total_seq_q
            * self.total_seq_kv
            / self.batch
            * self.heads
            * factor
            * self.dim
        )


def _sync() -> None:
    torch.musa.synchronize()


def _new_event() -> torch.musa.Event:
    return torch.musa.Event(enable_timing=True)


def _new_graph() -> torch.musa.MUSAGraph:
    return torch.musa.MUSAGraph()


def _dtype_from_name(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _make_cu_seqlens(total_seq: int, batch: int, device: str) -> torch.Tensor:
    if total_seq % batch != 0:
        raise ValueError("this benchmark expects total_seq divisible by batch")
    step = total_seq // batch
    return torch.arange(batch + 1, device=device, dtype=torch.int32) * step


def _ref_flashattn_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    causal: bool,
    smscale: float | None = None,
) -> torch.Tensor:
    _, heads_q, dim = q.shape
    heads_kv = k.shape[1]
    head_groups = heads_q // heads_kv
    scale = smscale if smscale is not None else dim**-0.5
    dq = torch.zeros_like(q, dtype=torch.float32)

    for batch_idx in range(cu_q.numel() - 1):
        q0 = int(cu_q[batch_idx].item())
        q1 = int(cu_q[batch_idx + 1].item())
        k0 = int(cu_k[batch_idx].item())
        k1 = int(cu_k[batch_idx + 1].item())
        q_batch = q[q0:q1].float()
        k_batch = k[k0:k1].float()
        v_batch = v[k0:k1].float()
        do_batch = do[q0:q1].float()
        q_len = q1 - q0
        kv_len = k1 - k0
        if causal:
            q_idx = torch.arange(q_len, device=q.device)[:, None]
            k_idx = torch.arange(kv_len, device=q.device)[None, :]
            causal_mask = q_idx + (kv_len - q_len) >= k_idx
        for head_q in range(heads_q):
            head_kv = head_q // head_groups
            scores = q_batch[:, head_q, :] @ k_batch[:, head_kv, :].T
            scores = scores * scale
            if causal:
                scores = scores.masked_fill(~causal_mask, float("-inf"))
            prob = torch.exp(scores - lse[head_q, q0:q1].unsqueeze(-1))
            dp = do_batch[:, head_q, :] @ v_batch[:, head_kv, :].T
            ds = prob * (dp - delta[q0:q1, head_q].unsqueeze(-1)) * scale
            dq[q0:q1, head_q, :] += ds @ k_batch[:, head_kv, :]
    return dq.to(q.dtype)


def _make_inputs(case: DQCase, device: str, seed: int):
    torch.manual_seed(seed)
    torch.musa.manual_seed(seed)
    q = (
        torch.randn(
            (case.total_seq_q, case.heads, case.dim),
            device=device,
            dtype=case.dtype,
        )
        / 10
    ).contiguous()
    k = (
        torch.randn(
            (case.total_seq_kv, case.heads, case.dim),
            device=device,
            dtype=case.dtype,
        )
        / 10
    ).contiguous()
    v = (
        torch.randn(
            (case.total_seq_kv, case.heads, case.dim),
            device=device,
            dtype=case.dtype,
        )
        / 10
    ).contiguous()
    do = (torch.randn_like(q) / 10).contiguous()
    cu_q = _make_cu_seqlens(case.total_seq_q, case.batch, device)
    cu_k = _make_cu_seqlens(case.total_seq_kv, case.batch, device)
    out, lse = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=case.max_seq_q,
        max_seqlen_k=case.max_seq_kv,
        causal=case.causal,
        return_softmax_lse=True,
        backend="mutlass",
    )
    return q, k, v, do, out.contiguous(), lse.contiguous(), cu_q, cu_k


class DQKernelPath:
    def __init__(self, case: DQCase, device: str, seed: int) -> None:
        self.case = case
        self.device = device
        (
            self.q,
            self.k,
            self.v,
            self.do,
            self.out,
            self.lse,
            self.cu_q,
            self.cu_k,
        ) = _make_inputs(case, device, seed)
        self.kernel_dtype = to_tilelang_dtype(case.dtype)
        self.delta = torch.empty(
            (case.total_seq_q, case.heads), device=device, dtype=torch.float32
        )
        self.dq = torch.empty_like(self.q)
        self.delta_kernel = compute_delta_ws(
            case.dim,
            is_varlen=True,
            dtype=self.kernel_dtype,
            block_M=case.block_m,
            threads=640,
        )
        self.dq_kernel = flashattn_bwd_ws_split_dq(
            dim=case.dim,
            is_causal=case.causal,
            is_varlen=True,
            heads_q_eq_heads_kv=True,
            block_M=case.block_m,
            block_N=case.block_n,
            smscale=None,
            dtype=self.kernel_dtype,
        )
        self.delta_kernel(self.out, self.do, self.delta)
        _sync()

    def run(self) -> torch.Tensor:
        self.dq_kernel(
            self.q,
            self.k,
            self.v,
            self.dq,
            self.do,
            self.cu_q,
            self.cu_k,
            self.lse,
            self.delta,
            self.case.max_seq_q,
        )
        return self.dq


def _bench_event_us(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start_events = [_new_event() for _ in range(repeat)]
    end_events = [_new_event() for _ in range(repeat)]
    for idx in range(repeat):
        start_events[idx].record()
        fn()
        end_events[idx].record()
    _sync()
    return statistics.median(
        start_events[idx].elapsed_time(end_events[idx]) * 1000.0
        for idx in range(repeat)
    )


def _bench_graph_us(
    fn: Callable[[], object],
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    graph = _new_graph()
    with torch.musa.graph(graph):
        for _ in range(graph_iters):
            fn()
    _sync()
    for _ in range(2):
        graph.replay()
    _sync()
    start_events = [_new_event() for _ in range(repeat)]
    end_events = [_new_event() for _ in range(repeat)]
    for idx in range(repeat):
        start_events[idx].record()
        graph.replay()
        end_events[idx].record()
    _sync()
    _GRAPH_KEEPALIVE.append(graph)
    return statistics.median(
        start_events[idx].elapsed_time(end_events[idx]) * 1000.0 / graph_iters
        for idx in range(repeat)
    )


def _validate_correctness(args: argparse.Namespace) -> None:
    case = DQCase(
        batch=1,
        heads=args.correctness_heads,
        total_seq_q=args.correctness_seq_q,
        total_seq_kv=args.correctness_seq_kv,
        dim=args.dim,
        dtype=_dtype_from_name(args.dtype),
        causal=args.causal,
        block_m=args.block_m,
        block_n=args.block_n,
    )
    path = DQKernelPath(case, args.device, args.seed)
    dq = path.run()
    _sync()
    if not bool(torch.isfinite(dq).all().item()):
        raise AssertionError("split-DQ output contains non-finite values")
    ref_dq = _ref_flashattn_dq(
        path.q,
        path.k,
        path.v,
        path.do,
        path.lse,
        path.delta,
        path.cu_q,
        path.cu_k,
        causal=case.causal,
    )
    _sync()
    err = (dq.float() - ref_dq.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f"[correctness] dQ max={max_err:.6f} mean={mean_err:.6f}")
    if max_err > args.atol or mean_err > args.mean_atol:
        raise AssertionError(
            f"dQ error exceeds tolerance: max={max_err} mean={mean_err}"
        )
    print("[correctness] pass")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--total-seq-q", type=int, default=8192)
    parser.add_argument("--total-seq-kv", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--graph-iters", type=int, default=10)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--correctness-heads", type=int, default=4)
    parser.add_argument("--correctness-seq-q", type=int, default=255)
    parser.add_argument("--correctness-seq-kv", type=int, default=255)
    parser.add_argument("--atol", type=float, default=2.5e-2)
    parser.add_argument("--mean-atol", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", default="musa")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is required for this benchmark")
    if args.dim not in (128, 256):
        raise ValueError("split-DQ kernel currently supports dim 128 or 256")
    torch.set_default_device(args.device)

    if not args.skip_correctness:
        _validate_correctness(args)

    case = DQCase(
        batch=args.batch,
        heads=args.heads,
        total_seq_q=args.total_seq_q,
        total_seq_kv=args.total_seq_kv,
        dim=args.dim,
        dtype=_dtype_from_name(args.dtype),
        causal=args.causal,
        block_m=args.block_m,
        block_n=args.block_n,
    )
    path = DQKernelPath(case, args.device, args.seed)
    path.run()
    _sync()
    event_us = _bench_event_us(path.run, args.warmup, args.repeat)
    graph_us = _bench_graph_us(path.run, args.warmup, args.repeat, args.graph_iters)
    print(
        "[bench] "
        f"batch={case.batch} heads={case.heads} seq_q={case.total_seq_q} "
        f"seq_kv={case.total_seq_kv} dim={case.dim} block_m={case.block_m} "
        f"block_n={case.block_n} dtype={args.dtype} causal={case.causal}"
    )
    print(f"[bench] event_dq={event_us:.3f} us")
    print(
        f"[bench] graph_dq={graph_us:.3f} us "
        f"TFLOPS={case.dq_flops / (graph_us * 1e-6) / 1e12:.3f}"
    )


if __name__ == "__main__":
    main()
