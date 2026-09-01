#!/usr/bin/env python3
"""Benchmark TileLang FlashAttention backward through the public bwd interface.

This script intentionally calls only ``flashattn_varlen_bwd_interface`` for the
timed path. Per-kernel timings are collected from eager profiler events.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import warnings
from pathlib import Path
from typing import Callable

import torch
import torch_musa  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mate.flash_attention.tilelang.flash_attention_varlen_bwd import (  # noqa: E402
    flashattn_varlen_bwd_interface,
)


@dataclasses.dataclass(frozen=True)
class InterfaceCase:
    batch: int
    heads: int
    heads_kv: int
    total_seq_q: int
    total_seq_kv: int
    dim: int
    dtype: torch.dtype
    causal: bool
    deterministic: bool

    @property
    def max_seq_q(self) -> int:
        return self.total_seq_q // self.batch

    @property
    def max_seq_kv(self) -> int:
        return self.total_seq_kv // self.batch

    @property
    def plan(self) -> str:
        heads_q_eq_heads_kv = self.heads == self.heads_kv
        if self.dim == 256:
            return "split_separate" if self.deterministic else "split"
        if self.dim == 128:
            return (
                "unsplit_separate"
                if self.deterministic or not heads_q_eq_heads_kv
                else "unsplit"
            )
        raise ValueError(f"unsupported dim: {self.dim}")

    @property
    def uses_separate_kernels(self) -> bool:
        return self.plan.endswith("_separate")

    @property
    def term_scale(self) -> int:
        return 1 if self.causal else 2

    def flops_for_terms(self, terms: int) -> float:
        return (
            self.total_seq_q
            * self.total_seq_kv
            / self.batch
            * self.heads
            * terms
            * self.term_scale
            * self.dim
        )

    @property
    def dkdv_flops(self) -> float:
        return self.flops_for_terms(4)

    @property
    def dq_flops(self) -> float:
        return self.flops_for_terms(3)

    @property
    def actual_flops(self) -> float:
        return self.flops_for_terms(7 if self.uses_separate_kernels else 5)

    @property
    def logical_flops(self) -> float:
        return self.flops_for_terms(5)


def _sync() -> None:
    torch.musa.synchronize()


def _new_event() -> torch.musa.Event:
    return torch.musa.Event(enable_timing=True)


def _profiler_activity():
    if hasattr(torch.profiler.ProfilerActivity, "MUSA"):
        return torch.profiler.ProfilerActivity.MUSA
    return torch.profiler.ProfilerActivity.CUDA


def _kernel_time_us(evt) -> float:
    for attr in (
        "device_time_total",
        "self_device_time_total",
        "musa_time_total",
        "self_musa_time_total",
    ):
        value = getattr(evt, attr, 0.0)
        if value:
            return float(value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return float(
            getattr(evt, "cuda_time_total", 0.0)
            or getattr(evt, "self_cuda_time_total", 0.0)
            or 0.0
        )


def _event_name(evt) -> str:
    return str(getattr(evt, "name", None) or getattr(evt, "key", "<unknown>"))


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


def _make_perf_inputs(case: InterfaceCase, device: str, *, synthetic: bool):
    torch.manual_seed(42)
    torch.musa.manual_seed(42)
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
            (case.total_seq_kv, case.heads_kv, case.dim),
            device=device,
            dtype=case.dtype,
        )
        / 10
    ).contiguous()
    v = (
        torch.randn(
            (case.total_seq_kv, case.heads_kv, case.dim),
            device=device,
            dtype=case.dtype,
        )
        / 10
    ).contiguous()
    dout = (torch.randn_like(q) / 10).contiguous()
    cu_q = _make_cu_seqlens(case.total_seq_q, case.batch, device)
    cu_k = _make_cu_seqlens(case.total_seq_kv, case.batch, device)
    if synthetic:
        out = (torch.randn_like(q) / 10).contiguous()
        softmax_lse = torch.zeros(
            (case.heads, case.total_seq_q),
            device=device,
            dtype=torch.float32,
        )
    else:
        out, lse_tq_h = _ref_flashattn_fwd(q, k, v, cu_q, cu_k, causal=case.causal)
        softmax_lse = lse_tq_h.transpose(0, 1).contiguous()
    return q, k, v, out, dout, softmax_lse, cu_q, cu_k


def _ref_flashattn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, heads_q, dim = q.shape
    heads_kv = k.shape[1]
    head_groups = heads_q // heads_kv
    scale = dim**-0.5
    out = torch.zeros_like(q, dtype=torch.float32)
    lse_out = torch.zeros((q.shape[0], heads_q), dtype=torch.float32, device=q.device)

    for batch_idx in range(cu_q.numel() - 1):
        q0, q1 = int(cu_q[batch_idx].item()), int(cu_q[batch_idx + 1].item())
        k0, k1 = int(cu_k[batch_idx].item()), int(cu_k[batch_idx + 1].item())
        q_batch = q[q0:q1].float()
        k_batch = k[k0:k1].float()
        v_batch = v[k0:k1].float()
        q_len = q1 - q0
        kv_len = k1 - k0
        for head_q in range(heads_q):
            head_kv = head_q // head_groups
            scores = q_batch[:, head_q, :] @ k_batch[:, head_kv, :].T
            scores = scores * scale
            if causal:
                q_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
                k_idx = torch.arange(kv_len, device=q.device).unsqueeze(0)
                valid = k_idx <= q_idx + (kv_len - q_len)
                scores = scores.masked_fill(~valid, float("-inf"))
            lse = torch.nan_to_num(torch.logsumexp(scores, dim=-1))
            prob = torch.exp(scores - lse.unsqueeze(-1))
            out[q0:q1, head_q, :] = prob @ v_batch[:, head_kv, :]
            lse_out[q0:q1, head_q] = lse
    return out.to(q.dtype), lse_out


def _ref_flashattn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    lse_tq_h: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, heads_q, dim = q.shape
    heads_kv = k.shape[1]
    head_groups = heads_q // heads_kv
    scale = dim**-0.5
    out, _ = _ref_flashattn_fwd(q, k, v, cu_q, cu_k, causal=causal)
    delta = (out.float() * dout.float()).sum(dim=-1)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    for batch_idx in range(cu_q.numel() - 1):
        q0, q1 = int(cu_q[batch_idx].item()), int(cu_q[batch_idx + 1].item())
        k0, k1 = int(cu_k[batch_idx].item()), int(cu_k[batch_idx + 1].item())
        q_batch = q[q0:q1].float()
        k_batch = k[k0:k1].float()
        v_batch = v[k0:k1].float()
        do_batch = dout[q0:q1].float()
        q_len = q1 - q0
        kv_len = k1 - k0
        for head_q in range(heads_q):
            head_kv = head_q // head_groups
            scores = q_batch[:, head_q, :] @ k_batch[:, head_kv, :].T
            scores = scores * scale
            if causal:
                q_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
                k_idx = torch.arange(kv_len, device=q.device).unsqueeze(0)
                valid = k_idx <= q_idx + (kv_len - q_len)
                scores = scores.masked_fill(~valid, float("-inf"))
            prob = torch.exp(scores - lse_tq_h[q0:q1, head_q].unsqueeze(-1))
            dp = do_batch[:, head_q, :] @ v_batch[:, head_kv, :].T
            ds = prob * (dp - delta[q0:q1, head_q].unsqueeze(-1)) * scale
            dq[q0:q1, head_q, :] += ds @ k_batch[:, head_kv, :]
            dk[k0:k1, head_kv, :] += ds.T @ q_batch[:, head_q, :]
            dv[k0:k1, head_kv, :] += prob.T @ do_batch[:, head_q, :]
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def _call_interface(case: InterfaceCase, inputs):
    q, k, v, out, dout, softmax_lse, cu_q, cu_k = inputs
    return flashattn_varlen_bwd_interface(
        q,
        k,
        v,
        out,
        dout,
        softmax_lse,
        case.max_seq_q,
        case.max_seq_kv,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        is_causal=case.causal,
        smscale=None,
        dtype=None,
        is_bhsd=False,
        deterministic=case.deterministic,
    )


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


def _kernel_role(name: str, case: InterfaceCase) -> str:
    low = name.lower()
    if "dkdv" in low:
        return "dkdv"
    if "_dq_kernel" in low or "unsplit_dq" in low or "split_dq" in low:
        return "dq"
    if "pack_dq" in low:
        return "pack_dq"
    if "reduce" in low and ("kv" in low or "grad" in low):
        return "reduce_kv"
    if "flashattn_bwd" in low or "bwd_ws_kernel" in low:
        return "bwd"
    if name == "main_kernel":
        return "delta" if case.deterministic else "main_kernel"
    if "kernelfill" in low or "kernel_fill" in low:
        if "__mt_bfloat16" in low or "__half" in low or "float" in low:
            return "zero_workspace"
        return "fill_aux"
    return "other_device"


def _collect_device_events(prof) -> list[tuple[str, float]]:
    events = []
    event_getter = getattr(prof, "events", None)
    raw_events = event_getter() if event_getter is not None else []
    for evt in raw_events:
        us = _kernel_time_us(evt)
        if us <= 0:
            continue
        name = _event_name(evt)
        events.append((name, us))
    return events


def _profile_eager_kernels(
    fn: Callable[[], object],
    *,
    profile_repeat: int,
    case: InterfaceCase,
) -> tuple[dict[str, float], list[tuple[str, str, float]]]:
    fn()
    _sync()
    with torch.profiler.profile(activities=[_profiler_activity()]) as prof:
        for _ in range(profile_repeat):
            fn()
            prof.step()
    _sync()
    events = _collect_device_events(prof)
    role_times: dict[str, list[float]] = {}
    ordered: list[tuple[str, str, float]] = []
    first_call_roles = ["fill_aux", "delta"]
    if case.uses_separate_kernels:
        first_call_roles.extend(["zero_workspace", "dkdv", "dq"])
    else:
        first_call_roles.extend(["zero_workspace", "bwd", "pack_dq"])
    first_replay_seen = set()
    first_call_done = False
    for name, us in events:
        role = _kernel_role(name, case)
        role_times.setdefault(role, []).append(us)
        if not first_call_done:
            ordered.append((role, name, us))
            first_replay_seen.add(role)
            if all(role in first_replay_seen for role in first_call_roles[1:]):
                first_call_done = True
    medians = {
        role: statistics.median(times) for role, times in role_times.items() if times
    }
    return medians, ordered


def _tflops(flops: float, us: float) -> float:
    return flops / (us * 1e-6) / 1e12


def _validate_correctness(case: InterfaceCase) -> None:
    check_case = dataclasses.replace(
        case,
        batch=1,
        heads=2,
        heads_kv=2,
        total_seq_q=256,
        total_seq_kv=256,
        dtype=torch.bfloat16,
    )
    inputs = _make_perf_inputs(check_case, "musa", synthetic=False)
    dq, dk, dv = _call_interface(check_case, inputs)
    q, k, v, _out, dout, softmax_lse, cu_q, cu_k = inputs
    ref_dq, ref_dk, ref_dv = _ref_flashattn_bwd(
        q,
        k,
        v,
        dout,
        softmax_lse.transpose(0, 1).contiguous(),
        cu_q,
        cu_k,
        causal=check_case.causal,
    )
    _sync()
    limit = 4.01 / 128
    for name, actual, ref in (
        ("dQ", dq, ref_dq),
        ("dK", dk, ref_dk),
        ("dV", dv, ref_dv),
    ):
        err = (actual.float() - ref.float()).abs().max().item()
        mean = (actual.float() - ref.float()).abs().mean().item()
        print(f"[correctness] {check_case.plan} {name} max={err:.6f} mean={mean:.6f}")
        if not torch.isfinite(torch.tensor(err)) or err > limit:
            raise AssertionError(f"{name} max error {err} exceeds {limit}")
    print(f"[correctness] {check_case.plan} pass")


def _print_case_header(case: InterfaceCase, *, synthetic: bool) -> None:
    print(
        "[bench] "
        f"batch={case.batch} heads={case.heads} heads_kv={case.heads_kv} "
        f"seq_q={case.total_seq_q} seq_kv={case.total_seq_kv} dim={case.dim} "
        f"dtype={case.dtype} causal={case.causal} deterministic={case.deterministic} "
        f"plan={case.plan} synthetic_inputs={synthetic}"
    )


def _run_case(args: argparse.Namespace, case: InterfaceCase) -> None:
    if args.check_correctness:
        _validate_correctness(case)

    inputs = _make_perf_inputs(case, "musa", synthetic=args.synthetic_inputs)
    fn = lambda: _call_interface(case, inputs)
    fn()
    _sync()
    _print_case_header(case, synthetic=args.synthetic_inputs)

    event_us = _bench_event_us(fn, args.warmup, args.repeat)
    print(f"[bench] event_e2e={event_us:.3f} us")
    print(
        f"[bench] event_actual_TFLOPS="
        f"{_tflops(case.actual_flops, event_us):.3f} "
        f"event_logical_TFLOPS={_tflops(case.logical_flops, event_us):.3f}",
        flush=True,
    )

    role_medians, ordered = _profile_eager_kernels(
        fn,
        profile_repeat=args.profile_repeat,
        case=case,
    )
    print("[profile] role median per eager interface call:")
    roles = [
        "delta",
        "bwd",
        "dkdv",
        "dq",
        "pack_dq",
        "reduce_kv",
        "zero_workspace",
        "fill_aux",
        "main_kernel",
        "other_device",
    ]
    for role in roles:
        if role not in role_medians:
            continue
        us = role_medians[role]
        suffix = ""
        if role == "dkdv":
            suffix = f" TFLOPS_4gemm={_tflops(case.dkdv_flops, us):.3f}"
        elif role == "dq":
            suffix = f" TFLOPS_3gemm={_tflops(case.dq_flops, us):.3f}"
        print(f"  {role:<14} {us:9.3f} us{suffix}")
    extra_roles = [role for role in role_medians if role not in roles]
    for role in extra_roles[:20]:
        print(f"  {role:<14} {role_medians[role]:9.3f} us")
    print("[profile] first interface call kernel order:")
    for idx, (role, name, us) in enumerate(ordered):
        print(f"  {idx:02d} {role:<14} {us:9.3f} us {name}")


def _variant_to_case(args: argparse.Namespace, variant: str) -> InterfaceCase:
    dim_arg = args.dim
    if variant == "deterministic-split":
        dim, deterministic = dim_arg or 256, True
    elif variant == "deterministic-unsplit":
        dim, deterministic = dim_arg or 128, True
    elif variant == "split":
        dim, deterministic = dim_arg or 256, False
    elif variant == "unsplit":
        dim, deterministic = dim_arg or 128, False
    else:
        dim, deterministic = dim_arg or 128, args.deterministic
    return InterfaceCase(
        batch=args.batch,
        heads=args.heads,
        heads_kv=args.heads_kv,
        total_seq_q=args.total_seq_q,
        total_seq_kv=args.total_seq_kv,
        dim=dim,
        dtype=_dtype_from_name(args.dtype),
        causal=args.causal,
        deterministic=deterministic,
    )


def _compile_branch_cases(args: argparse.Namespace) -> list[InterfaceCase]:
    cases: list[InterfaceCase] = []
    gqa_heads_kv = args.branch_gqa_heads_kv
    if args.heads % gqa_heads_kv != 0:
        raise ValueError(
            f"--heads ({args.heads}) must be divisible by --branch-gqa-heads-kv ({gqa_heads_kv})"
        )
    for dtype_name in ("bf16", "fp16"):
        for causal in (True, False):
            for heads_kv in (args.heads, gqa_heads_kv):
                for dim in (128, 256):
                    for deterministic in (False, True):
                        cases.append(
                            InterfaceCase(
                                batch=args.batch,
                                heads=args.heads,
                                heads_kv=heads_kv,
                                total_seq_q=args.total_seq_q,
                                total_seq_kv=args.total_seq_kv,
                                dim=dim,
                                dtype=_dtype_from_name(dtype_name),
                                causal=causal,
                                deterministic=deterministic,
                            )
                        )
    return cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--heads-kv", type=int, default=28)
    parser.add_argument("--total-seq-q", type=int, default=8192)
    parser.add_argument("--total-seq-kv", type=int, default=8192)
    parser.add_argument(
        "--dim",
        type=int,
        choices=[128, 256],
        default=None,
        help="Override the head dim selected by --variant. Defaults to 128 for auto, 256 for split variants, and 128 for unsplit variants.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--causal", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--variant",
        choices=[
            "auto",
            "split",
            "unsplit",
            "deterministic-split",
            "deterministic-unsplit",
        ],
        default="auto",
    )
    parser.add_argument(
        "--suite",
        choices=["single", "deterministic", "all", "compile-branches"],
        default="single",
    )
    parser.add_argument(
        "--branch-gqa-heads-kv",
        type=int,
        default=1,
        help="heads_kv value used by --suite compile-branches for the heads_q != heads_kv branch.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument(
        "--skip-correctness", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--synthetic-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use synthetic out/lse for perf runs instead of Python reference fwd.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is required for this benchmark")
    torch.set_default_device("musa")

    if args.suite == "compile-branches":
        cases = _compile_branch_cases(args)
    elif args.suite == "deterministic":
        variants = ("deterministic-split", "deterministic-unsplit")
        cases = [_variant_to_case(args, variant) for variant in variants]
    elif args.suite == "all":
        variants = ("split", "unsplit", "deterministic-split", "deterministic-unsplit")
        cases = [_variant_to_case(args, variant) for variant in variants]
    else:
        variants = (args.variant,)
        cases = [_variant_to_case(args, variant) for variant in variants]

    for idx, case in enumerate(cases):
        if idx:
            print("")
        _run_case(args, case)


if __name__ == "__main__":
    main()
