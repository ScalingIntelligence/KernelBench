from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import torch


def _sync() -> None:
    if hasattr(torch, "musa") and torch.musa.is_available():
        torch.musa.synchronize()
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_gpu_time(
    fn: Callable,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    dry_run_iters: Optional[int] = None,
    repeat_iters: Optional[int] = None,
    cold_l2_cache: bool = True,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
) -> list[float]:
    """MUSA-friendly compatibility timer returning per-call milliseconds."""

    del cold_l2_cache
    if input_kwargs is None:
        input_kwargs = {}

    def call_fn():
        return fn(*input_args, **input_kwargs)

    _sync()
    start = time.perf_counter()
    call_fn()
    _sync()
    estimate_ms = max((time.perf_counter() - start) * 1000.0, 1.0e-6)
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimate_ms))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimate_ms))

    for _ in range(dry_run_iters):
        call_fn()
    _sync()

    results = []
    for _ in range(repeat_iters):
        _sync()
        start = time.perf_counter()
        call_fn()
        _sync()
        results.append((time.perf_counter() - start) * 1000.0)
    return results


def attention_tflops(
    qo_lens,
    kv_lens,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
    time_ms,
) -> float:
    if isinstance(qo_lens, torch.Tensor):
        qo_lens = qo_lens.cpu().tolist()
    if isinstance(kv_lens, torch.Tensor):
        kv_lens = kv_lens.cpu().tolist()

    total_flops = 0.0
    for q_len, kv_len in zip(qo_lens, kv_lens):
        if causal:
            qk_flops = q_len * kv_len - q_len * (q_len - 1) / 2
        else:
            qk_flops = q_len * kv_len
        total_flops += num_qo_heads * qk_flops * (head_dim_qk + head_dim_vo) * 2
    return total_flops / (time_ms * 1.0e-3) / 1.0e12
