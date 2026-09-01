from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable

import torch

import mate
import mate.deep_gemm as deep_gemm


DEFAULT_SHAPES = "16x4096,512x1280,1024x2560,2048x4096,8192x4096"


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


def _parse_shapes(spec: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            num_tokens, hidden_size = item.split("x", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid shape {item!r}; expected '<num_tokens>x<hidden_size>'."
            ) from exc
        shapes.append((int(num_tokens), int(hidden_size)))
    return shapes


def _parse_ints(spec: str) -> list[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def _sync() -> None:
    torch.musa.synchronize()


def _bench_ms(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    start_events = [torch.musa.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.musa.Event(enable_timing=True) for _ in range(iters)]
    for idx in range(iters):
        start_events[idx].record()
        fn()
        end_events[idx].record()
    _sync()
    for idx in range(iters):
        times.append(start_events[idx].elapsed_time(end_events[idx]))
    return statistics.median(times)


def _num_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _make_inputs(
    num_tokens: int,
    hidden_size: int,
    *,
    mhc_mult: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    device = torch.device("musa")
    residual = (
        torch.randn(
            num_tokens,
            mhc_mult,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    hc_fn = (
        torch.randn(
            mhc_mult * (2 + mhc_mult),
            mhc_mult * hidden_size,
            device=device,
            dtype=torch.float32,
        )
        * 0.01
    )
    hc_scale = torch.tensor([0.8, 0.7, 0.5], device=device, dtype=torch.float32)
    hc_base = torch.linspace(
        -0.2,
        0.2,
        mhc_mult * (2 + mhc_mult),
        device=device,
        dtype=torch.float32,
    )
    return residual, hc_fn, hc_scale, hc_base


def _sinkhorn_ref(logits: torch.Tensor, eps: float, repeat: int) -> torch.Tensor:
    row_max = logits.amax(dim=-1, keepdim=True)
    comb = torch.exp(logits - row_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def _mhc_pre_ref(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.reshape(-1, mhc_mult, hidden_size)
    x_flat = residual_flat.float().reshape(-1, mhc_mult * hidden_size)
    rms = torch.rsqrt(
        x_flat.pow(2).sum(dim=-1, keepdim=True) / (mhc_mult * hidden_size) + rms_eps
    )
    mixes = torch.matmul(x_flat, hc_fn.t()) * rms
    pre = (
        torch.sigmoid(mixes[:, :mhc_mult] * hc_scale[0] + hc_base[:mhc_mult])
        + mhc_pre_eps
    )
    post = torch.sigmoid(
        mixes[:, mhc_mult : 2 * mhc_mult] * hc_scale[1]
        + hc_base[mhc_mult : 2 * mhc_mult]
    )
    post = post * mhc_post_mult_value
    comb_logits = mixes[:, 2 * mhc_mult :].reshape(-1, mhc_mult, mhc_mult)
    comb_base = hc_base[2 * mhc_mult :].reshape(mhc_mult, mhc_mult)
    comb = _sinkhorn_ref(
        comb_logits * hc_scale[2] + comb_base,
        mhc_sinkhorn_eps,
        sinkhorn_repeat,
    )
    layer_input = (pre.unsqueeze(-1) * residual_flat.float()).sum(dim=-2).bfloat16()
    return post, comb.reshape(-1, mhc_mult * mhc_mult), layer_input


def _mhc_pre_big_fuse_ref(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual: torch.Tensor,
    *,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gemm_out_mul.dim() == 3:
        gemm_out_mul = gemm_out_mul.sum(dim=0)
    if gemm_out_sqrsum.dim() == 2:
        gemm_out_sqrsum = gemm_out_sqrsum.sum(dim=0)
    mhc_mult = residual.shape[1]
    hidden_size = residual.shape[2]
    mixes = gemm_out_mul * torch.rsqrt(
        gemm_out_sqrsum[:, None] / (mhc_mult * hidden_size) + rms_eps
    )
    pre = (
        torch.sigmoid(mixes[:, :mhc_mult] * mhc_scale[0] + mhc_base[:mhc_mult])
        + mhc_pre_eps
    )
    post = torch.sigmoid(
        mixes[:, mhc_mult : 2 * mhc_mult] * mhc_scale[1]
        + mhc_base[mhc_mult : 2 * mhc_mult]
    )
    post = post * mhc_post_mult_value
    comb_logits = mixes[:, 2 * mhc_mult :].reshape(-1, mhc_mult, mhc_mult)
    comb_base = mhc_base[2 * mhc_mult :].reshape(mhc_mult, mhc_mult)
    comb = _sinkhorn_ref(
        comb_logits * mhc_scale[2] + comb_base,
        mhc_sinkhorn_eps,
        sinkhorn_repeat,
    )
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=-2).bfloat16()
    return post, comb.reshape(-1, mhc_mult * mhc_mult), layer_input


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float
) -> None:
    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        atol=atol,
        rtol=rtol,
    )


def _print_result(result: BenchResult) -> None:
    print(
        f"{result.name:<68s} "
        f"{result.ms:>10.4f} ms "
        f"{result.torch_ms:>10.4f} ms "
        f"{result.speedup:>8.2f}x "
        f"{result.bandwidth_gbps:>10.2f} GB/s"
    )


def bench_mhc_pre(
    *,
    num_tokens: int,
    hidden_size: int,
    split_k: int,
    warmup: int,
    iters: int,
    sinkhorn_repeat: int,
    check: bool,
) -> BenchResult:
    residual, hc_fn, hc_scale, hc_base = _make_inputs(num_tokens, hidden_size)
    post, comb, layer_input = mate.hyperconnection.mhc_pre(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        split_k=split_k,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    if check:
        ref_post, ref_comb, ref_layer = _mhc_pre_ref(
            residual,
            hc_fn,
            hc_scale,
            hc_base,
            rms_eps=1e-6,
            mhc_pre_eps=1e-6,
            mhc_sinkhorn_eps=1e-6,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        _assert_close(post.reshape(num_tokens, 4), ref_post, atol=2e-4, rtol=2e-3)
        _assert_close(comb.reshape(num_tokens, 16), ref_comb, atol=2e-4, rtol=2e-3)
        _assert_close(layer_input, ref_layer, atol=2e-2, rtol=2e-2)

    def run_kernel():
        return mate.hyperconnection.mhc_pre(
            residual,
            hc_fn,
            hc_scale,
            hc_base,
            split_k=split_k,
            sinkhorn_repeat=sinkhorn_repeat,
        )

    def run_torch():
        return _mhc_pre_ref(
            residual,
            hc_fn,
            hc_scale,
            hc_base,
            rms_eps=1e-6,
            mhc_pre_eps=1e-6,
            mhc_sinkhorn_eps=1e-6,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=sinkhorn_repeat,
        )

    bytes_rw = _num_bytes(residual, hc_fn, hc_scale, hc_base, post, comb, layer_input)
    return BenchResult(
        name=f"mhc_pre_deepgemm_tilelang_tokens{num_tokens}_h{hidden_size}_split{split_k}",
        ms=_bench_ms(run_kernel, warmup=warmup, iters=iters),
        torch_ms=_bench_ms(run_torch, warmup=warmup, iters=iters),
        bytes_rw=bytes_rw,
    )


def bench_big_fuse(
    *,
    num_tokens: int,
    hidden_size: int,
    split_k: int,
    warmup: int,
    iters: int,
    sinkhorn_repeat: int,
    check: bool,
) -> BenchResult:
    residual, _, hc_scale, hc_base = _make_inputs(num_tokens, hidden_size)
    mhc_mult = residual.shape[1]
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    torch.manual_seed(split_k)
    d_part = (
        torch.randn(
            split_k,
            num_tokens,
            mhc_mult3,
            device=residual.device,
            dtype=torch.float32,
        )
        * 0.01
    )
    s_part = torch.rand(
        split_k,
        num_tokens,
        device=residual.device,
        dtype=torch.float32,
    ) * (mhc_mult * hidden_size * 0.01 / split_k)
    post_buf = torch.empty(
        num_tokens, mhc_mult, device=residual.device, dtype=torch.float32
    )
    comb_buf = torch.empty(
        num_tokens,
        mhc_mult * mhc_mult,
        device=residual.device,
        dtype=torch.float32,
    )
    layer_buf = torch.empty(
        num_tokens, hidden_size, device=residual.device, dtype=torch.bfloat16
    )

    def run_kernel():
        return mate.hyperconnection.mhc_pre_big_fuse(
            d_part,
            s_part,
            hc_scale,
            hc_base,
            residual,
            backend="tilelang",
            threads=128,
            hidden_block=512 if num_tokens <= 64 else 1024,
            pass_config="safe",
            sinkhorn_repeat=sinkhorn_repeat,
            post_mix=post_buf,
            comb_mix=comb_buf,
            layer_input=layer_buf,
        )

    def run_torch():
        return _mhc_pre_big_fuse_ref(
            d_part,
            s_part,
            hc_scale,
            hc_base,
            residual,
            rms_eps=1e-6,
            mhc_pre_eps=1e-6,
            mhc_sinkhorn_eps=1e-6,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=sinkhorn_repeat,
        )

    out_post, out_comb, out_layer = run_kernel()
    if check:
        ref_post, ref_comb, ref_layer = run_torch()
        _assert_close(out_post, ref_post, atol=2e-4, rtol=2e-3)
        _assert_close(out_comb, ref_comb, atol=2e-4, rtol=2e-3)
        _assert_close(out_layer, ref_layer, atol=2e-2, rtol=2e-2)

    bytes_rw = _num_bytes(
        d_part, s_part, hc_scale, hc_base, residual, post_buf, comb_buf, layer_buf
    )
    return BenchResult(
        name=f"mhc_pre_big_fuse_tokens{num_tokens}_h{hidden_size}_split{split_k}",
        ms=_bench_ms(run_kernel, warmup=warmup, iters=iters),
        torch_ms=_bench_ms(run_torch, warmup=warmup, iters=iters),
        bytes_rw=bytes_rw,
    )


def bench_prenorm(
    *,
    num_tokens: int,
    hidden_size: int,
    split_k: int,
    warmup: int,
    iters: int,
    check: bool,
) -> BenchResult:
    residual, hc_fn, _, _ = _make_inputs(num_tokens, hidden_size)
    x_flat = residual.reshape(num_tokens, -1)
    d_part = torch.empty(
        split_k,
        num_tokens,
        hc_fn.shape[0],
        device=residual.device,
        dtype=torch.float32,
    )
    s_part = torch.empty(
        split_k, num_tokens, device=residual.device, dtype=torch.float32
    )

    def run_kernel():
        deep_gemm.tf32_hc_prenorm_gemm(
            x_flat,
            hc_fn.contiguous(),
            d_part,
            s_part,
            num_splits=split_k,
        )
        return d_part, s_part

    def run_torch():
        x = x_flat.float()
        return torch.mm(x, hc_fn.t()), x.square().sum(dim=-1)

    run_kernel()
    if check:
        ref_d, ref_s = run_torch()
        _assert_close(d_part.sum(dim=0), ref_d, atol=5e-2, rtol=5e-2)
        _assert_close(s_part.sum(dim=0), ref_s, atol=1e-1, rtol=1e-2)

    bytes_rw = _num_bytes(x_flat, hc_fn, d_part, s_part)
    return BenchResult(
        name=f"mhc_prenorm_deepgemm_tokens{num_tokens}_h{hidden_size}_split{split_k}",
        ms=_bench_ms(run_kernel, warmup=warmup, iters=iters),
        torch_ms=_bench_ms(run_torch, warmup=warmup, iters=iters),
        bytes_rw=bytes_rw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MATE MHC pre operators.")
    parser.add_argument(
        "--cases", choices=["all", "prenorm", "big-fuse", "pre"], default="all"
    )
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--splits", default="32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--sinkhorn-repeat", type=int, default=4)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise RuntimeError("MUSA device is not available.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.mudnn.allow_tf32 = True

    print(f"MUSA: {torch.musa.get_device_name(0)}")
    print(
        f"Shapes: {args.shapes}; splits: {args.splits}; "
        f"warmup={args.warmup}; iters={args.iters}; check={not args.no_check}"
    )
    print(
        f"{'Kernel':<68s} {'Current':>14s} {'Torch':>14s} "
        f"{'Speedup':>9s} {'Bandwidth':>15s}"
    )
    print("-" * 124)

    check = not args.no_check
    for num_tokens, hidden_size in _parse_shapes(args.shapes):
        for split_k in _parse_ints(args.splits):
            if args.cases in ("all", "prenorm"):
                _print_result(
                    bench_prenorm(
                        num_tokens=num_tokens,
                        hidden_size=hidden_size,
                        split_k=split_k,
                        warmup=args.warmup,
                        iters=args.iters,
                        check=check,
                    )
                )
            if args.cases in ("all", "big-fuse"):
                _print_result(
                    bench_big_fuse(
                        num_tokens=num_tokens,
                        hidden_size=hidden_size,
                        split_k=split_k,
                        warmup=args.warmup,
                        iters=args.iters,
                        sinkhorn_repeat=args.sinkhorn_repeat,
                        check=check,
                    )
                )
            if args.cases in ("all", "pre"):
                _print_result(
                    bench_mhc_pre(
                        num_tokens=num_tokens,
                        hidden_size=hidden_size,
                        split_k=split_k,
                        warmup=args.warmup,
                        iters=args.iters,
                        sinkhorn_repeat=args.sinkhorn_repeat,
                        check=check,
                    )
                )


if __name__ == "__main__":
    main()
