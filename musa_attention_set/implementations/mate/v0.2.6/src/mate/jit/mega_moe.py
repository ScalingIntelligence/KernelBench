from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .gemm.deep_gemm.deep_gemm_utils import DEEP_GEMM_CUDA_FLAGS
from .utils import EXPORT_FUNC, TVM_HEADER


MEGA_MOE_EXTRA_CFLAGS = [
    "-O3",
    "-Wno-switch-bool",
]

MEGA_MOE_EXTRA_MUSA_CFLAGS = list(DEEP_GEMM_CUDA_FLAGS) + [
    "-fno-signed-zeros",
    "-DMARCH_TYPE=310",
]

MEGA_MOE_BLOCK_M = 32
MEGA_MOE_BLOCK_N = 256
MEGA_MOE_BLOCK_K = 128
MEGA_MOE_STAGE1_NUM_STAGES = 3
MEGA_MOE_STAGE1_NUM_DISPATCH_THREADS = 128
MEGA_MOE_STAGE1_NUM_COMPUTE_THREADS = 384
MEGA_MOE_STAGE2_NUM_THREADS = 512
MEGA_MOE_STAGE2_NUM_SMEM_STAGES = 4
MEGA_MOE_STAGE2_NUM_CD_STAGES = 1
MEGA_MOE_STAGE2_CD_DESC_WORDS = 4

MEGA_MOE_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader((jit_env.MATE_TEMPLATE_DIR / "mega_moe").as_posix()),
    keep_trailing_newline=True,
)


@dataclass(frozen=True)
class MegaMoEKernelConfig:
    num_ranks: int
    num_experts: int
    num_max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate_hidden: int
    num_mps: int
    fast_math: bool = True


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def get_block_m_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
) -> int:
    _ = (num_ranks, num_experts, num_max_tokens_per_rank, num_topk)
    return MEGA_MOE_BLOCK_M


def _num_max_pool_tokens(config: MegaMoEKernelConfig, block_m: int) -> int:
    num_experts_per_rank = config.num_experts // config.num_ranks
    num_max_recv_tokens = config.num_ranks * config.num_max_tokens_per_rank
    num_max_experts_per_token = min(config.num_topk, num_experts_per_rank)
    return _align(
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (block_m - 1),
        block_m,
    )


def _num_padded_sf_pool_tokens(num_max_pool_tokens: int, block_m: int) -> int:
    return (num_max_pool_tokens // block_m) * _align(block_m, 128)


def _validate_config(config: MegaMoEKernelConfig) -> None:
    if config.num_ranks <= 0:
        raise ValueError("num_ranks must be positive")
    if config.num_experts <= 0 or config.num_experts % config.num_ranks != 0:
        raise ValueError("num_experts must be positive and divisible by num_ranks")
    if config.num_max_tokens_per_rank <= 0:
        raise ValueError("num_max_tokens_per_rank must be positive")
    if config.num_topk <= 0:
        raise ValueError("num_topk must be positive")
    if config.hidden <= 0 or config.hidden % MEGA_MOE_BLOCK_N != 0:
        raise ValueError(f"hidden must be positive and divisible by {MEGA_MOE_BLOCK_N}")
    if (
        config.intermediate_hidden <= 0
        or config.intermediate_hidden % MEGA_MOE_BLOCK_N != 0
    ):
        raise ValueError(
            f"intermediate_hidden must be positive and divisible by {MEGA_MOE_BLOCK_N}"
        )
    if config.num_mps <= 0:
        raise ValueError("num_mps must be positive")


def _normalize_config(
    config: Mapping[str, Any] | MegaMoEKernelConfig,
) -> MegaMoEKernelConfig:
    if isinstance(config, MegaMoEKernelConfig):
        result = config
    else:
        result = MegaMoEKernelConfig(
            num_ranks=int(config["num_ranks"]),
            num_experts=int(config["num_experts"]),
            num_max_tokens_per_rank=int(config["num_max_tokens_per_rank"]),
            num_topk=int(config["num_topk"]),
            hidden=int(config["hidden"]),
            intermediate_hidden=int(config["intermediate_hidden"]),
            num_mps=int(config["num_mps"]),
            fast_math=bool(config.get("fast_math", True)),
        )
    _validate_config(result)
    return result


def _stage1_render_config(config: MegaMoEKernelConfig) -> dict[str, object]:
    block_m = MEGA_MOE_BLOCK_M
    block_n = MEGA_MOE_BLOCK_N
    block_k = MEGA_MOE_BLOCK_K
    num_stages = MEGA_MOE_STAGE1_NUM_STAGES
    num_dispatch_threads = MEGA_MOE_STAGE1_NUM_DISPATCH_THREADS
    num_compute_threads = MEGA_MOE_STAGE1_NUM_COMPUTE_THREADS
    num_max_pool_tokens = _num_max_pool_tokens(config, block_m)
    num_padded_sf_pool_tokens = _num_padded_sf_pool_tokens(num_max_pool_tokens, block_m)

    smem_expert_count_size = _align(config.num_experts * 4, 4096)
    smem_token_pull_size = _align(config.hidden, 4096) * (num_dispatch_threads // 32)
    smem_a_size = num_stages * block_m * block_k
    smem_b_size = num_stages * block_n * block_k
    smem_output_amax_size = _align(block_m * 4, 256)
    smem_size = (
        smem_expert_count_size
        + smem_token_pull_size
        + _align(
            smem_a_size + smem_b_size + smem_output_amax_size,
            256,
        )
    )

    return {
        **config.__dict__,
        "num_experts_per_rank": config.num_experts // config.num_ranks,
        "num_experts_per_wave": 1,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_max_pool_tokens": num_max_pool_tokens,
        "num_padded_sf_pool_tokens": num_padded_sf_pool_tokens,
        "num_stages": num_stages,
        "num_dispatch_threads": num_dispatch_threads,
        "num_compute_threads": num_compute_threads,
        "num_threads": num_dispatch_threads + num_compute_threads,
        "smem_size": smem_size,
    }


def _stage2_render_config(config: MegaMoEKernelConfig) -> dict[str, object]:
    block_m = MEGA_MOE_BLOCK_M
    block_n = MEGA_MOE_BLOCK_N
    block_k = MEGA_MOE_BLOCK_K
    num_threads = MEGA_MOE_STAGE2_NUM_THREADS
    num_smem_stages = MEGA_MOE_STAGE2_NUM_SMEM_STAGES
    store_block_m = block_m
    num_cd_stages = MEGA_MOE_STAGE2_NUM_CD_STAGES
    cd_desc_words = MEGA_MOE_STAGE2_CD_DESC_WORDS
    num_max_pool_tokens = _num_max_pool_tokens(config, block_m)
    num_padded_sf_pool_tokens = _num_padded_sf_pool_tokens(num_max_pool_tokens, block_m)

    a_tile_bytes = block_m * block_k
    b_tile_bytes = block_n * block_k
    cd_slot_bytes = store_block_m * block_n * 2
    cd_desc_bytes = _align(num_cd_stages * cd_desc_words * 4, 256)
    smem_size = _align(
        num_smem_stages * (a_tile_bytes + b_tile_bytes)
        + num_cd_stages * cd_slot_bytes
        + cd_desc_bytes,
        1024,
    )

    return {
        **config.__dict__,
        "num_experts_per_rank": config.num_experts // config.num_ranks,
        "num_experts_per_wave": 1,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_max_pool_tokens": num_max_pool_tokens,
        "num_padded_sf_pool_tokens": num_padded_sf_pool_tokens,
        "num_threads": num_threads,
        "smem_size": smem_size,
        "fp8_element_size": 1,
    }


def _dispatch_name(prefix: str, config: MegaMoEKernelConfig) -> str:
    suffix = (
        f"r{config.num_ranks}_e{config.num_experts}"
        f"_mt{config.num_max_tokens_per_rank}_topk{config.num_topk}"
        f"_h{config.hidden}_ih{config.intermediate_hidden}"
        f"_mps{config.num_mps}_fm{int(config.fast_math)}"
    )
    return f"mate_jit_mega_moe_{prefix}_{suffix}"


def _generated_source_path(dispatch_name: str) -> Path:
    return jit_env.MATE_GEN_SRC_DIR / "mega_moe" / f"{dispatch_name}.mu"


def _extra_include_paths() -> list[Path]:
    return [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
        jit_env.MUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include",
        jit_env.MUTLASS_INCLUDE_DIR.parent / "experimental" / "fmha",
    ]


def _render_source(
    template_name: str,
    dispatch_name: str,
    render_config: dict[str, object],
) -> str:
    render_config = dict(render_config)
    render_config["func_name"] = dispatch_name
    return (
        TVM_HEADER
        + MEGA_MOE_TEMPLATE_ENV.get_template(template_name).render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


def _freeze_config(
    config: MegaMoEKernelConfig,
) -> tuple[int, int, int, int, int, int, int, bool]:
    return (
        config.num_ranks,
        config.num_experts,
        config.num_max_tokens_per_rank,
        config.num_topk,
        config.hidden,
        config.intermediate_hidden,
        config.num_mps,
        config.fast_math,
    )


def _spec(
    dispatch_name: str,
    source_file: Path,
    source: str,
) -> JitSpec:
    return gen_jit_spec(
        name=dispatch_name,
        sources=[source_file],
        generated_sources={source_file: source},
        extra_cflags=list(MEGA_MOE_EXTRA_CFLAGS),
        extra_cuda_cflags=list(MEGA_MOE_EXTRA_MUSA_CFLAGS),
        extra_include_paths=_extra_include_paths(),
    )


def gen_fp8_fp8_mega_moe_stage1_spec(
    config: Mapping[str, Any] | MegaMoEKernelConfig,
) -> JitSpec:
    kernel_config = _normalize_config(config)
    dispatch_name = _dispatch_name("fp8_fp8_stage1", kernel_config)
    source_file = _generated_source_path(dispatch_name)
    source = _render_source(
        "fp8_fp8_stage1_kern.j2",
        dispatch_name,
        _stage1_render_config(kernel_config),
    )
    return _spec(dispatch_name, source_file, source)


def gen_fp8_fp8_mega_moe_stage2_spec(
    config: Mapping[str, Any] | MegaMoEKernelConfig,
) -> JitSpec:
    kernel_config = _normalize_config(config)
    dispatch_name = _dispatch_name("fp8_fp8_stage2", kernel_config)
    source_file = _generated_source_path(dispatch_name)
    source = _render_source(
        "fp8_fp8_stage2_kern.j2",
        dispatch_name,
        _stage2_render_config(kernel_config),
    )
    return _spec(dispatch_name, source_file, source)


@functools.cache
def _load_stage1_module(frozen_config: tuple[int, int, int, int, int, int, int, bool]):
    config = MegaMoEKernelConfig(*frozen_config)
    return gen_fp8_fp8_mega_moe_stage1_spec(config).build_and_load()


@functools.cache
def _load_stage2_module(frozen_config: tuple[int, int, int, int, int, int, int, bool]):
    config = MegaMoEKernelConfig(*frozen_config)
    return gen_fp8_fp8_mega_moe_stage2_spec(config).build_and_load()


def get_fp8_fp8_mega_moe_stage1_module(
    config: Mapping[str, Any] | MegaMoEKernelConfig,
):
    kernel_config = _normalize_config(config)
    return (
        _dispatch_name("fp8_fp8_stage1", kernel_config),
        _load_stage1_module(_freeze_config(kernel_config)),
    )


def get_fp8_fp8_mega_moe_stage2_module(
    config: Mapping[str, Any] | MegaMoEKernelConfig,
):
    kernel_config = _normalize_config(config)
    return (
        _dispatch_name("fp8_fp8_stage2", kernel_config),
        _load_stage2_module(_freeze_config(kernel_config)),
    )


def gen_mega_moe_runtime_utils_spec() -> JitSpec:
    name = "mate_mega_moe_runtime_utils"
    return gen_jit_spec(
        name=name,
        sources=[jit_env.MATE_CSRC_DIR / "mega_moe_runtime_utils.mu"],
        extra_cflags=list(MEGA_MOE_EXTRA_CFLAGS),
        extra_cuda_cflags=list(MEGA_MOE_EXTRA_MUSA_CFLAGS),
        extra_include_paths=_extra_include_paths(),
    )


@functools.cache
def get_mega_moe_runtime_utils_module():
    return gen_mega_moe_runtime_utils_spec().build_and_load()


def gen_mega_moe_specs(
    configs: Sequence[Mapping[str, Any] | MegaMoEKernelConfig],
) -> list[JitSpec]:
    specs: list[JitSpec] = [gen_mega_moe_runtime_utils_spec()]
    for config in configs:
        specs.append(gen_fp8_fp8_mega_moe_stage1_spec(config))
        specs.append(gen_fp8_fp8_mega_moe_stage2_spec(config))
    return specs


__all__ = [
    "MegaMoEKernelConfig",
    "gen_fp8_fp8_mega_moe_stage1_spec",
    "gen_fp8_fp8_mega_moe_stage2_spec",
    "gen_mega_moe_runtime_utils_spec",
    "gen_mega_moe_specs",
    "get_block_m_for_mega_moe",
    "get_fp8_fp8_mega_moe_stage1_module",
    "get_fp8_fp8_mega_moe_stage2_module",
    "get_mega_moe_runtime_utils_module",
]
