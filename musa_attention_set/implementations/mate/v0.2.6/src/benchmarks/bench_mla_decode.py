# Copyright (c) 2025, Ted Zadouri, Tri Dao.

# We recommend locking GPU clocks before running the benchmark to ensure consistent results.
# This can be done using the following commands (1830 MHz is the clock for H100):
# sudo nvidia-smi -i 0 -pm 1
# sudo nvidia-smi -i 0 --lock-gpu-clocks 1830,1830
# See more here: https://github.com/triton-lang/triton/blob/d9f10ebdc5da53f73eb852fde73d8d7d80b679d1/python/triton/testing.py#L487
import time
import json
import os
import torch
import mate

from einops import rearrange
from torch.profiler import profile, ProfilerActivity  # noqa: F401
from itertools import product
from collections import namedtuple
from mate.testing.utils import bench_kineto


# Device Setup
device = "musa"
torch.manual_seed(0)

# Metadata
dtype = torch.bfloat16
seqlen_kv = 4096
seqlen_q = 1
sm_scale = 1 / ((512 + 64) ** 0.5)

PerfCfg = namedtuple(
    "PerfCfg",
    [
        "batch_size",
        "seqlen_q",
        "seqlen_kv",
        "nheads_q",
        "causal",
    ],
)

nheads_kv = 1
headdim_rope = 64
headdim_latent = 512
has_qv = headdim_rope == 64 and headdim_latent > 64
page_size = 64

# Perf Configs
batch_sizes = [30, 56]
seqlen_qs = [[1, 1]]  # [min, max] for varlen
seqlen_kvs = [int(s * 1024) for s in [4, 8, 16]]
nheads_qs = [8, 16, 32]  # [8, 16, 32, 64, 128]
causals = [False]
benchmark_backends = [
    backend.strip()
    for backend in os.environ.get(
        "MATE_MLA_DECODE_BACKENDS", "fmha_qv,flash_mla"
    ).split(",")
    if backend.strip()
]


def summarize_values(values):
    return {
        "first8": values[:8],
        "last8": values[-8:],
        "counts": {str(value): values.count(value) for value in sorted(set(values))},
    }


def summarize_split_prefix(prefix):
    return summarize_values(
        [prefix[idx + 1] - prefix[idx] for idx in range(len(prefix) - 1)]
    )


perf_configs = product(batch_sizes, seqlen_qs, seqlen_kvs, nheads_qs, causals)
perf_idx = 0
for cfg in (PerfCfg(*perf_config) for perf_config in perf_configs):
    batch_size = cfg.batch_size
    seqlen_q = cfg.seqlen_q
    seqlen_kv = cfg.seqlen_kv
    nheads_q = cfg.nheads_q
    causal = cfg.causal
    varlen_q = type(seqlen_q) is not int
    cu_seqlens_q = None
    max_seqlen_q = max(seqlen_q) if varlen_q else seqlen_q
    if varlen_q:
        seqlen_q = torch.randint(
            seqlen_q[0], seqlen_q[1] + 1, (batch_size,), device=device
        )
        cu_seqlens_q = (
            torch.nn.functional.pad(torch.cumsum(seqlen_q, dim=0), (1, 0))
            .to(torch.int32)
            .to(device)
        )
        # seqlen_q = seqlen_q.tolist()

    cache_seqlens = torch.tensor(
        [seqlen_kv] * batch_size, device=device, dtype=torch.int
    )
    if varlen_q:
        q_rope = torch.randn(
            sum(seqlen_q).item(),
            nheads_q,
            headdim_rope,
            dtype=dtype,
            device=device,
        )
        q_latent = torch.randn(
            sum(seqlen_q).item(),
            nheads_q,
            headdim_latent,
            dtype=dtype,
            device=device,
        )
    else:
        q_rope = torch.randn(
            batch_size,
            seqlen_q,
            nheads_q,
            headdim_rope,
            dtype=dtype,
            device=device,
        )
        q_latent = torch.randn(
            batch_size,
            seqlen_q,
            nheads_q,
            headdim_latent,
            dtype=dtype,
            device=device,
        )
    try:
        v_cache = torch.randn(
            batch_size,
            seqlen_kv,
            nheads_kv,
            headdim_latent,
            dtype=dtype,
            device=device,
        )
        k_cache = torch.randn(
            batch_size,
            seqlen_kv,
            nheads_kv,
            headdim_rope,
            dtype=dtype,
            device=device,
        )
        if page_size is not None:
            assert seqlen_kv % page_size == 0
            k_cache, v_cache = [
                rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size)
                for x in [k_cache, v_cache]
            ]
            page_table = rearrange(
                torch.arange(
                    batch_size * seqlen_kv // page_size,
                    device=device,
                    dtype=torch.int32,
                ),
                "(b s) -> b s",
                s=seqlen_kv // page_size,
            )
        else:
            page_table = None
    except torch.OutOfMemoryError:
        continue

    kv = torch.cat([v_cache, k_cache], dim=-1)

    def run_attention(scheduler_metadata):
        return mate.flash_attn_with_kvcache(
            q=q_rope,
            k_cache=kv[..., headdim_latent:],
            v_cache=kv[..., :headdim_latent],
            qv=q_latent,
            cache_seqlens=cache_seqlens,  # cache_seqlens
            page_table=page_table,  # page_table
            cu_seqlens_q=cu_seqlens_q,  # cu_query_lens
            max_seqlen_q=max_seqlen_q,  # max_seqlen_q
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            attention_chunk=0,
            softcap=0.0,
            rotary_interleaved=False,
            scheduler_metadata=scheduler_metadata,
            num_splits=0,
            pack_gqa=None,
            sm_margin=0,
            return_softmax_lse=True,
        )

    size_dtype = torch.finfo(dtype).bits // 8  # 2 for fp16/bf16, 4 for fp32
    size_lse = torch.finfo(torch.float32).bits // 8
    total_seqlen_kv = (
        seqlen_kv * batch_size if cache_seqlens is None else cache_seqlens.sum().item()
    )
    total_seqlen_q = sum(seqlen_q).item() if varlen_q else seqlen_q * batch_size

    for backend in benchmark_backends:
        if backend == "fmha_qv":
            scheduler_metadata = mate.get_scheduler_metadata(
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=seqlen_kv,
                num_heads_q=nheads_q,
                num_heads_kv=nheads_kv,
                headdim=headdim_rope,
                headdim_v=headdim_latent,
                seqused_k=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                page_size=page_size,
                causal=causal,
                num_splits=0,
                pack_gqa=None,
                has_qv=True,
            )
            metadata_view = scheduler_metadata.detach().cpu().view(4, -1)
            num_splits_prefix = [0]
            for splits in metadata_view[0].tolist():
                num_splits_prefix.append(num_splits_prefix[-1] + int(splits))
            kernel_names = ("FmhaFwdKernelWarpSpecialized", "FmhaFwdCombine")

            def fn0():
                return run_attention(scheduler_metadata)

            metadata_summary = {
                "fmha_shape": list(metadata_view.shape),
                "num_splits": summarize_values(metadata_view[0].tolist()),
                "batch_table": summarize_values(metadata_view[1].tolist()),
            }
        elif backend == "flash_mla":
            workspace = torch.empty(
                (16 * 1024 * 1024,), dtype=torch.uint8, device=device
            )
            tile_metadata, mla_num_splits = mate.get_mla_metadata(
                cache_seqlens,
                num_q_tokens_per_head_k=max_seqlen_q * nheads_q // nheads_kv,
                num_heads_k=nheads_kv,
                num_heads_q=nheads_q,
            )

            # Initialize FlashMLA scheduler metadata once; timed runs reuse it.
            run_attention((workspace, False))
            torch.musa.synchronize()
            num_splits_prefix = [int(x) for x in mla_num_splits.detach().cpu().tolist()]
            kernel_names = ("MlaKernelTmeWarpSpecialized", "mpxx_mla_combine_kernel")

            def fn0():
                return run_attention((workspace, True))

            metadata_summary = {
                "mla_metadata_shape": list(tile_metadata.shape),
                "num_splits": summarize_split_prefix(num_splits_prefix),
            }
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        time.sleep(1)  # to avoid power throttling

        t_splitkv, t_combine = bench_kineto(
            fn0,
            kernel_names,
            suppress_kineto_output=True,
            num_tests=10,
            trace_path=None,
            with_multiple_kernels=True,
        )

        # fn0()
        # t0 = 100
        # t0 = triton.musa_testing.do_bench(fn0, warmup=3, rep=10)
        # with torch.musa.stream(torch.musa.Stream()):
        #     t0 = do_bench_cudagraph(fn0, rep=10)
        # graph = torch.musa.MUSAGraph()
        # with torch.musa.graph(graph):
        #     fn0()

        # def g():
        #     return graph.replay()

        # t0 = do_bench(graph.replay, warmup=1, rep=10)
        # Split-related
        total_combine_splits, total_combine_batchs = 0, 0
        intact_batchs = []
        for batch_idx in range(len(num_splits_prefix) - 1):
            splits = num_splits_prefix[batch_idx + 1] - num_splits_prefix[batch_idx]
            total_combine_splits += splits if splits > 1 else 0
            total_combine_batchs += 1 if splits > 1 else 0
            if splits == 1:
                intact_batchs.append(batch_idx)
        total_intact_batchs = batch_size - total_combine_batchs
        if varlen_q:
            total_intact_seqlen_q = seqlen_q[intact_batchs].sum().item()
        else:
            total_intact_seqlen_q = seqlen_q * total_intact_batchs
        # mem io for splitkv kernel. Need to handle differently depending on split case.
        mem_io = (
            total_seqlen_kv
            * nheads_kv
            * (headdim_rope + headdim_latent)
            * size_dtype  # Load K
            + q_rope.numel() * size_dtype  # Load q_rope
            + (q_latent.numel() if q_latent is not None else 0)
            * size_dtype  # Load q_latent
            + size_dtype
            * (
                max_seqlen_q
                * nheads_q
                * headdim_latent
                * total_combine_splits  # out_accum
                + total_intact_seqlen_q * nheads_q * headdim_latent  # out
            )  # out / out_accum
            + size_lse
            * (
                max_seqlen_q * nheads_q * total_combine_splits  # lse_accum
                + total_intact_seqlen_q * nheads_q  # lse
            )  # lse / lse_accum
        )  # last term is for the output
        if varlen_q:
            flops = (
                total_seqlen_q
                * nheads_q  # q
                * seqlen_kv
                * (headdim_rope + headdim_latent * (2 if has_qv else 1))  # kv
                * 2  # dtype
            )
        else:
            flops = (
                seqlen_q
                * nheads_q  # q
                * total_seqlen_kv
                * (headdim_rope + headdim_latent * (2 if has_qv else 1))  # kv
                * 2  # dtype
            )
        # Get combine mem io
        mem_io_combine = (
            total_combine_splits
            * (
                max_seqlen_q * nheads_q * headdim_latent * size_dtype  # out_accum
                + max_seqlen_q * nheads_q * size_lse  # lse_accum
            )  # Read from accum buffers
            + total_combine_batchs
            * (
                max_seqlen_q * nheads_q * headdim_latent * size_dtype  # out
                + max_seqlen_q * nheads_q * size_lse  # lse
            )  # Write to output & lse
        )
        result = {
            "backend": backend,
            "config": ", ".join(f"{k}={v}" for k, v in cfg._asdict().items()),
            "metadata": metadata_summary,
            "perf units  ": "Time (us), Bandwidth (GB/s), Compute (TFLOPS/s)",
            "main-perf": f"{t_splitkv * 1e6:.2f}     {mem_io * 1e-9 / (t_splitkv):.2f}            {flops * 1e-12 / (t_splitkv):.2f}",
            "combine-perf": f"{t_combine * 1e6:.2f}      {mem_io_combine * 1e-9 / (t_combine) if t_combine > 0 else 0:.2f}            N/A",
        }
        print(
            f"[PERF {perf_idx:02d} {'VarlenQ' if varlen_q else 'FixedQ'}]"
            + json.dumps(result, indent=2)
        )
        perf_idx += 1
