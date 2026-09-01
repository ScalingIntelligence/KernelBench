"""Utility functions for flash attention benchmarks."""

from typing import Optional, List, Tuple


def print_flash_attn_summary(
    batch_size: int,
    seqlen_q: List[int],
    seqlen_kv: List[int],
    head_q: int,
    head_kv: int,
    headdim_qk: int,
    headdim_vo: int,
    is_causal: bool,
    is_packgqa: bool,
    is_varlen_q: bool,
    is_varlen_kv: bool,
    is_paged_kv: bool,
    window_size: Tuple[Optional[int], Optional[int]],
    q_shape: tuple,
    k_shape,
    v_shape,
    cu_seqlens_q_shape: Optional[tuple] = None,
    cu_seqlens_kv_shape: Optional[tuple] = None,
    page_table_shape: Optional[tuple] = None,
    seqused_q: Optional[List[int]] = None,
    seqused_kv: Optional[List[int]] = None,
    flops: Optional[int] = None,
    io_bytes: Optional[int] = None,
    time_ms: Optional[float] = None,
    tflops: Optional[float] = None,
    bandwidth: Optional[float] = None,
    backend: Optional[str] = None,
    title: str = "Flash Attention Benchmark",
):
    """Print complete benchmark summary."""

    def fmt_num(n: int) -> str:
        return f"{n:,}"

    def fmt_shape(s) -> str:
        if isinstance(s, str):
            return s
        return str(tuple(s))

    print("\n" + "=" * 60)
    print(f"            {title}            ")
    print("=" * 60 + "\n")

    head_ratio = head_q // head_kv
    print("[Configuration]")
    print(f"  Batch Size       : {batch_size}")
    if backend is not None:
        print(f"  Backend          : {backend}")
    total_seqlen_q = sum(seqlen_q)
    total_seqlen_kv = sum(seqlen_kv)
    print(
        f"  Total Seqlen     : Q={fmt_num(total_seqlen_q)}, KV={fmt_num(total_seqlen_kv)}"
    )
    print(f"  Max Seqlen       : Q={max(seqlen_q)}, KV={max(seqlen_kv)}")
    if seqused_q is not None:
        print(f"  Total Seqused Q  : {fmt_num(sum(seqused_q))}")
        print(f"  Max Seqused Q    : {max(seqused_q)}")
    if seqused_kv is not None:
        print(f"  Total Seqused KV : {fmt_num(sum(seqused_kv))}")
        print(f"  Max Seqused KV   : {max(seqused_kv)}")
    print(f"  Heads            : Q={head_q}, KV={head_kv} (ratio={head_ratio})")
    print(f"  Head Dimensions  : QK={headdim_qk}, VO={headdim_vo}")
    modes = []
    if is_causal:
        modes.append("Causal")
    if is_packgqa:
        modes.append("PackGQA")
    print(f"  Attention Mode   : {' | '.join(modes) if modes else 'Standard'}")
    print(f"  Varlen           : Q={is_varlen_q}, KV={is_varlen_kv}")
    print(f"  Paged KV         : {is_paged_kv}")
    print(f"  Local Attention  : {window_size != (None, None)}")

    print("\n[Tensor Shapes]")
    print(f"  Q                : {fmt_shape(q_shape)}")
    print(f"  K                : {fmt_shape(k_shape)}")
    print(f"  V                : {fmt_shape(v_shape)}")
    if cu_seqlens_q_shape is not None:
        print(f"  cu_seqlens_q     : {fmt_shape(cu_seqlens_q_shape)}")
    if cu_seqlens_kv_shape is not None:
        print(f"  cu_seqlens_kv    : {fmt_shape(cu_seqlens_kv_shape)}")
    if page_table_shape is not None:
        print(f"  page_table       : {fmt_shape(page_table_shape)}")

    if flops is not None and io_bytes is not None:
        print("\n[Compute Metrics]")
        print(f"  FLOP             : {fmt_num(flops)}")
        print(f"  I/O Bytes        : {fmt_num(io_bytes)}")

    if time_ms is not None and tflops is not None and bandwidth is not None:
        print("\n[Benchmark Results]")
        print(f"  Time             : {time_ms:.4f} ms")
        print(f"  TFLOPs           : {tflops:.4f}")
        print(f"  GB/s             : {bandwidth:.2f}")
    print("=" * 60 + "\n")


def print_flash_attn_benchmark(
    cfg,
    t: float,
    backend: Optional[str] = None,
    title: str = "Flash Attention Benchmark",
):
    """
    Print complete benchmark summary from config and timing result.

    Args:
        cfg: FlashAttnBenchConfig instance
        t: Time in seconds from bench_kineto
        title: Optional title for the benchmark
    """
    metrics = cfg.get_metrics()

    q_shape: tuple[int, ...]
    if cfg.is_varlen_q:
        q_shape = (sum(cfg.seqlen_q), cfg.head_q, cfg.headdim_qk)
    else:
        q_shape = (cfg.batch_size, cfg.max_seqlen_q, cfg.head_q, cfg.headdim_qk)

    k_shape: str | tuple[int, ...]
    v_shape: str | tuple[int, ...]
    if cfg.is_paged_kv:
        k_shape = "paged"
        v_shape = "paged"
    elif cfg.is_varlen_kv:
        k_shape = (sum(cfg.seqlen_kv), cfg.head_kv, cfg.headdim_qk)
        v_shape = (sum(cfg.seqlen_kv), cfg.head_kv, cfg.headdim_vo)
    else:
        k_shape = (cfg.batch_size, cfg.max_seqlen_kv, cfg.head_kv, cfg.headdim_qk)
        v_shape = (cfg.batch_size, cfg.max_seqlen_kv, cfg.head_kv, cfg.headdim_vo)

    cu_seqlens_q_shape = (cfg.batch_size + 1,) if cfg.is_varlen_q else None
    cu_seqlens_kv_shape = (cfg.batch_size + 1,) if cfg.is_varlen_kv else None

    print_flash_attn_summary(
        batch_size=cfg.batch_size,
        seqlen_q=cfg.bench_seqlens_q,
        seqlen_kv=cfg.bench_seqlens_kv,
        head_q=cfg.head_q,
        head_kv=cfg.head_kv,
        headdim_qk=cfg.headdim_qk,
        headdim_vo=cfg.headdim_vo,
        is_causal=cfg.is_causal,
        is_packgqa=cfg.is_packgqa,
        is_varlen_q=cfg.is_varlen_q,
        is_varlen_kv=cfg.is_varlen_kv,
        is_paged_kv=cfg.is_paged_kv,
        window_size=cfg.window_size,
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        cu_seqlens_q_shape=cu_seqlens_q_shape,
        cu_seqlens_kv_shape=cu_seqlens_kv_shape,
        seqused_q=cfg.seqused_q,
        seqused_kv=cfg.seqused_kv,
        flops=metrics.flops,
        io_bytes=metrics.io_bytes,
        time_ms=t * 1000,
        tflops=metrics.flops * 1e-12 / t,
        bandwidth=metrics.io_bytes * 1e-9 / t,
        backend=backend,
        title=title,
    )
