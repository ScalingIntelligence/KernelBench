import torch
from typing import cast, List, Tuple, Union, Optional

from bench_flash_attn_config import FlashAttnBenchConfig
from bench_utils import print_flash_attn_benchmark
from mate.testing.utils import bench_kineto
from mate import flash_attn_varlen_func, flash_attn_with_kvcache
from mate.mha_interface import get_scheduler_metadata

MATE_BENCH_FMHA_DTYPE = torch.bfloat16
# MATE_BENCH_FMHA_DTYPE = torch.float8_e4m3fn
MATE_BENCH_FMHA_DEVICE = "musa"

MATE_BENCH_FMHA_ENABLE_TRACE = False
MATE_BENCH_FMHA_FLUSH_L2 = True


def gen_bench_config() -> List[dict]:
    cases = [
        {
            "name": "Prefill",
            "batch_size": 56,
            "seqlen_q": [1024 * 4] * 56,
            "seqlen_kv": [1024 * 4] * 56,
            "head_q": 64,
            "head_kv": 4,
            "headdim_qk": 128,
            "headdim_vo": 128,
            "is_packgqa": False,
            "is_causal": False,
            "window_size": (None, None),
            "page_size": None,
            "num_splits": 0,
            "backend": "mutlass",
            "dtype": MATE_BENCH_FMHA_DTYPE,
            "device": MATE_BENCH_FMHA_DEVICE,
            "force_flash_attn_with_kvcache": False,
            "seqused_q": None,
            "seqused_kv": None,
        },
    ]

    return cases


def run_bench(
    batch_size: int,
    seqlen_q: Union[int, List[int]],
    seqlen_kv: Union[int, List[int]],
    head_q: int,
    head_kv: int,
    headdim_qk: int,
    headdim_vo: int,
    is_packgqa: bool,
    is_causal: bool,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    page_size: Optional[int] = None,
    num_splits: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    backend: str = "mutlass",
    device: str = "musa",
    num_tests: int = 10,
    force_flash_attn_with_kvcache: bool = False,
    name: str = "",
    seqused_q: Optional[List[int]] = None,
    seqused_kv: Optional[List[int]] = None,
) -> float:
    cfg = FlashAttnBenchConfig(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        head_q=head_q,
        head_kv=head_kv,
        headdim_qk=headdim_qk,
        headdim_vo=headdim_vo,
        is_packgqa=is_packgqa,
        is_causal=is_causal,
        window_size=window_size,
        page_size=page_size,
        device=device,
        dtype=dtype,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
    )

    if backend == "mubin":
        # only check part of unsupport cases
        if cfg.is_varlen_q != cfg.is_varlen_kv:
            raise ValueError(
                f"For 'mubin' backend, Q and KV must be both varlen or both non-varlen. "
                f"Got is_varlen_q={cfg.is_varlen_q}, is_varlen_kv={cfg.is_varlen_kv}"
            )
        if force_flash_attn_with_kvcache:
            raise ValueError(
                "For 'mubin' backend, force_flash_attn_with_kvcache must be False"
            )

    q = cfg.q_tensor
    kv = cfg.kv_tensor
    k, v = kv.k, kv.v

    if num_splits < 0:
        metadata = None
    else:
        metadata = get_scheduler_metadata(
            batch_size=cfg.batch_size,
            max_seqlen_q=cfg.max_seqlen_q,
            max_seqlen_k=cfg.max_seqlen_kv,
            num_heads_q=cfg.head_q,
            num_heads_kv=cfg.head_kv,
            headdim=cfg.headdim_qk,
            seqused_q=cfg.seqused_q_tensor,
            seqused_k=cfg.seqused_kv_tensor,
            headdim_v=cfg.headdim_vo,
            qkv_dtype=cfg.dtype,
            cu_seqlens_q=cfg.cu_seqlens_q,
            cu_seqlens_k=cfg.cu_seqlens_kv,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            max_seqlen_k_new=0,
            causal=cfg.is_causal,
            window_size=cfg.window_size,
            page_size=cfg.page_size,
            attention_chunk=0,
            has_softcap=False,
            num_splits=num_splits,
            pack_gqa=cfg.is_packgqa,
            mp_margin=0,
        )
        print(metadata.view(4, -1))

    def fn():
        if cfg.is_paged_kv or force_flash_attn_with_kvcache:
            return flash_attn_with_kvcache(
                q=q,
                k_cache=kv.k,
                v_cache=kv.v,
                k=None,
                v=None,
                qv=None,
                rotary_cos=None,
                rotary_sin=None,
                cache_seqlens=cfg.seqused_kv_tensor,
                cache_batch_idx=None,
                cache_leftpad=None,
                page_table=kv.page_table,
                cu_seqlens_q=cfg.cu_seqlens_q,
                cu_seqlens_k_new=None,
                max_seqlen_q=cfg.max_seqlen_q,
                rotary_seqlens=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                softmax_scale=None,
                causal=cfg.is_causal,
                window_size=cfg.window_size,
                learnable_sink=None,
                attention_chunk=0,
                softcap=0.0,
                rotary_interleaved=True,
                scheduler_metadata=metadata,
                num_splits=num_splits,
                pack_gqa=cfg.is_packgqa,
                sm_margin=0,
                return_softmax_lse=False,
            )
        else:
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cfg.cu_seqlens_q,
                cu_seqlens_k=cfg.cu_seqlens_kv,
                max_seqlen_q=cfg.max_seqlen_q,
                max_seqlen_k=cfg.max_seqlen_kv,
                seqused_q=cfg.seqused_q_tensor,
                seqused_k=cfg.seqused_kv_tensor,
                softmax_scale=None,
                causal=cfg.is_causal,
                qv=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                window_size=cfg.window_size,
                learnable_sink=None,
                attention_chunk=0,
                softcap=0.0,
                scheduler_metadata=metadata,
                num_splits=num_splits,
                pack_gqa=cfg.is_packgqa,
                deterministic=False,
                sm_margin=0,
                return_softmax_lse=False,
                backend=backend,
            )

    trace_path = (
        f"trace_mate_fmha_{name}.json"
        if MATE_BENCH_FMHA_ENABLE_TRACE and name
        else None
    )

    kernel_name_mubin = "bf16tce" if dtype == torch.bfloat16 else "htce"
    kernel_names = (
        kernel_name_mubin if backend == "mubin" else "FmhaFwdKernelWarpSpecialized"
    )
    t = bench_kineto(
        fn,
        kernel_names=kernel_names,
        num_tests=num_tests,
        suppress_kineto_output=True,
        trace_path=trace_path,
        flush_l2=MATE_BENCH_FMHA_FLUSH_L2,
        with_multiple_kernels=True,
    )

    effective_backend = (
        None if (cfg.is_paged_kv or force_flash_attn_with_kvcache) else backend
    )
    print_flash_attn_benchmark(
        cfg,
        cast(float, t),
        backend=effective_backend,
        title=f"Flash Attention Benchmark - {name}"
        if name
        else "Flash Attention Benchmark",
    )

    return cast(float, t)


def main():
    for cfg_dict in gen_bench_config():
        run_bench(**cfg_dict)


if __name__ == "__main__":
    main()
