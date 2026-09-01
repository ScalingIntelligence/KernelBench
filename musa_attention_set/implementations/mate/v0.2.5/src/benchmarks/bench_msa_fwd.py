#!/usr/bin/env python3
"""Benchmark the MSA forward path against dense attention.

The sparse E2E path is exactly:

    proxy max-score -> top-k block selection -> MSA forward

It intentionally has no q2k transpose, CSR build, partial output, or combine.
The runtime head ratio is 8 or 16 with D=128, page/block=128, and top-k=16.
Prefill uses q_len=kv_len; decode uses q_len=1.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mate.msa_interface as msa  # noqa: E402
from mate.jit.msa_fwd import (  # noqa: E402
    _msa_fwd,
)
from mate.mha_interface import (  # noqa: E402
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)


HEAD_DIM = 128
PAGE_SIZE = 128
TOPK = 16
FP8 = getattr(torch, "float8_e4m3fn", None)
DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
if FP8 is not None:
    DTYPES["fp8e4m3"] = FP8
DEFAULT_DTYPE = "fp8e4m3" if FP8 is not None else "bf16"


def _device(dtype: torch.dtype) -> torch.device:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("this benchmark requires a local MUSA device")
    if dtype == FP8 and FP8 is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable")
    return torch.device("musa")


def _rand_input(
    shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return (torch.randn(shape, dtype=torch.float16, device=device) * 0.25).to(dtype)


def _summarize(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "min_ms": min(samples),
        "p50_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "max_ms": max(samples),
        "repeat": len(samples),
    }


def _bench(
    fn: Callable[[], object], *, warmup: int, repeat: int
) -> dict[str, float | int]:
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
    return _summarize([start.elapsed_time(end) for start, end in zip(starts, ends)])


def _accuracy(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float().cpu()
    expected_f = expected.float().cpu()
    diff = actual_f - expected_f
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rms": float(diff.square().mean().sqrt().item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_f.reshape(-1), expected_f.reshape(-1), dim=0
            ).item()
        ),
    }


def run_case(
    *,
    mode: str,
    seq_len: int,
    warmup: int,
    repeat: int,
    seed: int,
    run_dense: bool,
    dtype: torch.dtype,
    q_heads: int,
    kv_heads: int,
) -> dict[str, object]:
    if kv_heads <= 0 or q_heads % kv_heads != 0 or q_heads // kv_heads not in (8, 16):
        raise ValueError(f"q_heads/kv_heads must be 8 or 16, got {q_heads}/{kv_heads}")
    device = _device(dtype)
    q_len = seq_len if mode == "prefill" else 1
    kv_len = seq_len
    num_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    qo_offset_value = kv_len - q_len
    softmax_scale = HEAD_DIM**-0.5

    torch.manual_seed(seed)
    q = _rand_input((q_len, q_heads, HEAD_DIM), device, dtype)
    k = _rand_input((num_pages, PAGE_SIZE, kv_heads, HEAD_DIM), device, dtype)
    v = _rand_input((num_pages, PAGE_SIZE, kv_heads, HEAD_DIM), device, dtype)
    proxy_q = q[:, :: q_heads // kv_heads, :].contiguous()

    qo_lens = torch.tensor([q_len], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    qo_offset = torch.tensor([qo_offset_value], dtype=torch.int32, device=device)
    query_positions = (
        torch.arange(q_len, dtype=torch.int64, device=device) + qo_offset_value
    )
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        1, num_pages
    )
    flat_page_table = page_table.view(-1)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)

    maxscore_plan = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        kv_heads,
        num_kv_heads=kv_heads,
        qo_offset=qo_offset,
        page_size=PAGE_SIZE,
        causal=True,
        output_maxscore=True,
        split_prefill_decode=False,
    )
    padded_pages = ((num_pages + 127) // 128) * 128
    max_score = torch.empty(
        (q_len, kv_heads, padded_pages),
        dtype=torch.float32,
        device=device,
    )
    block_indexes = torch.empty(
        (q_len, kv_heads, TOPK), dtype=torch.int32, device=device
    )
    selector_bypassed = num_pages <= TOPK
    all_page_indexes = torch.arange(TOPK, dtype=torch.int32, device=device).masked_fill(
        torch.arange(TOPK, dtype=torch.int32, device=device) >= num_pages,
        -1,
    )
    all_page_indexes = all_page_indexes.view(1, 1, TOPK).expand(q_len, kv_heads, TOPK)
    sparse_out = torch.empty_like(q)
    sparse_lse = torch.empty((q_len, q_heads), dtype=torch.float32, device=device)

    def maxscore_stage():
        return msa.msa(
            proxy_q,
            k,
            v,
            maxscore_plan,
            kv_indices=flat_page_table,
            output_o=False,
            output_maxscore=True,
            max_score=max_score,
        )

    def topk_stage():
        return msa.sparse_topk_select(
            max_score,
            topk=TOPK,
            num_valid_pages=num_pages,
            output=block_indexes,
            query_positions=query_positions,
        )

    def sparse_attention_stage():
        return _msa_fwd(
            q,
            k,
            v,
            block_indexes,
            cu_q,
            kv_lens,
            qo_offset,
            flat_page_table,
            kv_page_indptr=page_indptr,
            max_seqlen_q=q_len,
            max_seqlen_k=kv_len,
            causal=True,
            softmax_scale=softmax_scale,
            out=sparse_out,
            lse=sparse_lse,
        )

    def sparse_e2e_stage():
        if not selector_bypassed:
            maxscore_stage()
            topk_stage()
        return sparse_attention_stage()

    maxscore_stage()
    topk_stage()
    sparse_attention_stage()
    torch.musa.synchronize()

    stages = {
        "maxscore": _bench(maxscore_stage, warmup=warmup, repeat=repeat),
        "topk_select": _bench(topk_stage, warmup=warmup, repeat=repeat),
        "sparse_attention": _bench(
            sparse_attention_stage, warmup=warmup, repeat=repeat
        ),
    }
    if selector_bypassed:
        block_indexes.copy_(all_page_indexes)
    stages["sparse_e2e"] = _bench(sparse_e2e_stage, warmup=warmup, repeat=repeat)

    result: dict[str, object] = {
        "mode": mode,
        "seq_len": seq_len,
        "q_len": q_len,
        "kv_len": kv_len,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "topk": TOPK,
        "selected_tokens": TOPK * PAGE_SIZE,
        "dtype": next(name for name, value in DTYPES.items() if value == dtype),
        "causal": True,
        "num_pages": num_pages,
        "selector_bypassed": selector_bypassed,
        "stage_ms": stages,
    }

    if run_dense:
        scheduler_metadata = get_scheduler_metadata(
            batch_size=1,
            max_seqlen_q=q_len,
            max_seqlen_k=kv_len,
            num_heads_q=q_heads,
            num_heads_kv=kv_heads,
            headdim=HEAD_DIM,
            headdim_v=HEAD_DIM,
            seqused_q=qo_lens,
            seqused_k=kv_lens,
            cu_seqlens_q=cu_q,
            qkv_dtype=dtype,
            page_size=PAGE_SIZE,
            num_splits=1,
            causal=True,
        )

        def dense_stage():
            dense_result, *_ = flash_attn_with_kvcache(
                q=q,
                k_cache=k,
                v_cache=v,
                cache_seqlens=kv_lens,
                scheduler_metadata=scheduler_metadata,
                num_splits=1,
                max_seqlen_q=q_len,
                causal=True,
                page_table=page_table,
                cu_seqlens_q=cu_q,
                softmax_scale=softmax_scale,
                return_softmax_lse=True,
            )
            return dense_result

        dense_stage()
        torch.musa.synchronize()
        dense_stats = _bench(dense_stage, warmup=warmup, repeat=repeat)
        result["stage_ms"]["dense"] = dense_stats
        sparse_ms = float(stages["sparse_attention"]["p50_ms"])
        e2e_ms = float(stages["sparse_e2e"]["p50_ms"])
        dense_ms = float(dense_stats["p50_ms"])
        result["sparse_attention_over_dense"] = sparse_ms / dense_ms
        result["sparse_e2e_over_dense"] = e2e_ms / dense_ms
        if num_pages <= TOPK:
            sparse_e2e_stage()
            dense_out = dense_stage().to(dtype)
            torch.musa.synchronize()
            result["all_pages_accuracy"] = _accuracy(sparse_out, dense_out)

    return result


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(item) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("--lengths must contain positive comma-separated ints")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument("--lengths", default="1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default=DEFAULT_DTYPE)
    parser.add_argument("--q-heads", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--kv-heads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    modes = ("prefill", "decode") if args.mode == "both" else (args.mode,)
    for mode in modes:
        for seq_len in _parse_lengths(args.lengths):
            row = run_case(
                mode=mode,
                seq_len=seq_len,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed,
                run_dense=not args.skip_dense,
                dtype=DTYPES[args.dtype],
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
            )
            print(json.dumps(row, sort_keys=True), flush=True)
            del row
            gc.collect()
            torch.musa.empty_cache()


if __name__ == "__main__":
    main()
