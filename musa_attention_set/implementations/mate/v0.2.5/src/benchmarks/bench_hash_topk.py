from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable

import torch

import mate


@dataclass(frozen=True)
class BenchResult:
    name: str
    ms: float
    torch_ms: float
    bytes_rw: int

    @property
    def speedup(self) -> float:
        return self.torch_ms / self.ms if self.ms > 0 else float("inf")

    @property
    def bandwidth_gbps(self) -> float:
        return self.bytes_rw / (self.ms * 1.0e-3) / 1.0e9 if self.ms > 0 else 0.0


def _sync() -> None:
    torch.musa.synchronize()


def _bench_ms(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start_events = [torch.musa.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.musa.Event(enable_timing=True) for _ in range(iters)]
    for idx in range(iters):
        start_events[idx].record()
        fn()
        end_events[idx].record()
    _sync()
    return statistics.median(
        start_events[idx].elapsed_time(end_events[idx]) for idx in range(iters)
    )


def _hash_topk_ref(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    routed_ids = tid2eid[input_ids.long()].to(torch.int64)
    routed_scores = (
        torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
    )
    routed_scores = routed_scores / routed_scores.sum(dim=-1, keepdim=True).clamp_min(
        1e-20
    )
    if num_fused_shared_experts == 0:
        return routed_scores.to(torch.float32), routed_ids
    shared_ids = torch.arange(
        router_logits.shape[1],
        router_logits.shape[1] + num_fused_shared_experts,
        dtype=torch.int64,
        device=router_logits.device,
    ).expand(router_logits.shape[0], -1)
    shared_weights = torch.full(
        (router_logits.shape[0], num_fused_shared_experts),
        1.0 / routed_scaling_factor,
        dtype=torch.float32,
        device=router_logits.device,
    )
    return torch.cat(
        [routed_scores.to(torch.float32), shared_weights], dim=1
    ), torch.cat([routed_ids, shared_ids], dim=1)


def _bytes_rw(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> int:
    num_tokens = input_ids.shape[0]
    topk = tid2eid.shape[1]
    return (
        num_tokens * input_ids.element_size()
        + num_tokens * topk * tid2eid.element_size()
        + num_tokens * topk * router_logits.element_size()
        + weights.numel() * weights.element_size()
        + ids.numel() * ids.element_size()
    )


def bench_hash_topk(
    *,
    num_tokens: int,
    num_experts: int,
    topk: int,
    num_tid: int,
    shared: int,
    tid_dtype: torch.dtype,
    warmup: int,
    iters: int,
    check: bool,
) -> BenchResult:
    device = torch.device("musa")
    scaling = 2.0
    router_logits = torch.linspace(
        -4.0,
        4.0,
        steps=num_tokens * num_experts,
        dtype=torch.float32,
        device=device,
    ).reshape(num_tokens, num_experts)
    input_ids = torch.arange(num_tokens, dtype=torch.int64, device=device) % num_tid
    tid2eid = (
        (
            torch.arange(num_tid * topk, dtype=torch.int64, device=device).reshape(
                num_tid, topk
            )
            * 37
        )
        % num_experts
    ).to(tid_dtype)

    weights, ids = mate.hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=shared,
        routed_scaling_factor=scaling,
    )
    if check:
        ref_weights, ref_ids = _hash_topk_ref(
            router_logits, input_ids, tid2eid, shared, scaling
        )
        torch.testing.assert_close(
            weights.cpu(), ref_weights.cpu(), rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(ids.cpu(), ref_ids.cpu(), rtol=0, atol=0)

    def run_kernel() -> object:
        return mate.hash_topk(
            router_logits,
            input_ids,
            tid2eid,
            num_fused_shared_experts=shared,
            routed_scaling_factor=scaling,
            topk_weights=weights,
            topk_ids=ids,
        )

    def run_torch() -> object:
        return _hash_topk_ref(router_logits, input_ids, tid2eid, shared, scaling)

    return BenchResult(
        name=(
            f"hash_topk_tokens{num_tokens}_experts{num_experts}_topk{topk}_"
            f"shared{shared}_{str(tid_dtype).split('.')[-1]}"
        ),
        ms=_bench_ms(run_kernel, warmup=warmup, iters=iters),
        torch_ms=_bench_ms(run_torch, warmup=warmup, iters=iters),
        bytes_rw=_bytes_rw(router_logits, input_ids, tid2eid, weights, ids),
    )


def _print_result(result: BenchResult) -> None:
    print(
        f"{result.name:<56s} "
        f"{result.ms:>9.4f} ms "
        f"{result.torch_ms:>9.4f} ms "
        f"{result.speedup:>8.2f}x "
        f"{result.bandwidth_gbps:>9.2f} GB/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MATE hash-id topk router.")
    parser.add_argument("--tokens", default="1,16,128,256,1024,4096,8192")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--num-tid", type=int, default=32000)
    parser.add_argument("--shared", type=int, default=1)
    parser.add_argument("--tid-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    tid_dtype = torch.int32 if args.tid_dtype == "int32" else torch.int64
    for token_text in args.tokens.split(","):
        token_text = token_text.strip()
        if not token_text:
            continue
        result = bench_hash_topk(
            num_tokens=int(token_text),
            num_experts=args.num_experts,
            topk=args.topk,
            num_tid=args.num_tid,
            shared=args.shared,
            tid_dtype=tid_dtype,
            warmup=args.warmup,
            iters=args.iters,
            check=not args.no_check,
        )
        _print_result(result)


if __name__ == "__main__":
    main()
