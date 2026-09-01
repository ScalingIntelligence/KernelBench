from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

from jinja2 import Environment, FileSystemLoader

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...utils import EXPORT_FUNC, TVM_HEADER
from ....utils import round_up
from .. import gemm_utils
from .deep_gemm_utils import DEEP_GEMM_CUDA_FLAGS


_ALIGN = 32
_AOT_METADATA_BATCH_SIZES = [32, 64, 128]

CXX_FLAGS = [
    "-O3",
    "-Wno-switch-bool",
]


@dataclass(frozen=True)
class PagedMqaLogitsJitConfig:
    next_n: int
    num_heads: int
    head_dim: int
    block_kv: int
    is_context_lens_2d: bool


PAGED_MQA_LOGITS_NEXT_N_OPTIONS = (1, 2, 4)
PAGED_MQA_LOGITS_NUM_HEADS_OPTIONS = (32, 64)
PAGED_MQA_LOGITS_HEAD_DIM_OPTIONS = (32, 64, 128)
PAGED_MQA_LOGITS_BLOCK_KV_OPTIONS = (64,)

PAGED_MQA_LOGITS_CONFIGS: List[PagedMqaLogitsJitConfig] = [
    PagedMqaLogitsJitConfig(
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_kv=block_kv,
        is_context_lens_2d=is_context_lens_2d,
    )
    for is_context_lens_2d in (False, True)
    for next_n in PAGED_MQA_LOGITS_NEXT_N_OPTIONS
    for num_heads in PAGED_MQA_LOGITS_NUM_HEADS_OPTIONS
    for head_dim in PAGED_MQA_LOGITS_HEAD_DIM_OPTIONS
    for block_kv in PAGED_MQA_LOGITS_BLOCK_KV_OPTIONS
]


_paged_mqa_logits_template_env = Environment(
    loader=FileSystemLoader((gemm_utils.gemm_template_dir / "deep_gemm").as_posix()),
    keep_trailing_newline=True,
)


def get_paged_mqa_logits_template(name: str):
    return _paged_mqa_logits_template_env.get_template(name)


def _generated_source_path(dispatch_name: str) -> Path:
    return jit_env.MATE_GEN_SRC_DIR / "deep_gemm" / f"{dispatch_name}.mu"


def _paged_mqa_logits_include_paths() -> list[Path]:
    return [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
    ]


def _validate_config(cfg: PagedMqaLogitsJitConfig) -> None:
    if cfg.next_n not in PAGED_MQA_LOGITS_NEXT_N_OPTIONS:
        raise ValueError(f"Unsupported paged MQA logits next_n={cfg.next_n}")
    if cfg.num_heads not in PAGED_MQA_LOGITS_NUM_HEADS_OPTIONS:
        raise ValueError(f"Unsupported paged MQA logits num_heads={cfg.num_heads}")
    if cfg.head_dim not in PAGED_MQA_LOGITS_HEAD_DIM_OPTIONS:
        raise ValueError(f"Unsupported paged MQA logits head_dim={cfg.head_dim}")
    if cfg.block_kv not in PAGED_MQA_LOGITS_BLOCK_KV_OPTIONS:
        raise ValueError(f"Unsupported paged MQA logits block_kv={cfg.block_kv}")


@functools.cache
def select_paged_mqa_logits_config(
    next_n: int,
    num_heads: int,
    head_dim: int,
    block_kv: int,
    is_context_lens_2d: bool,
) -> PagedMqaLogitsJitConfig:
    cfg = PagedMqaLogitsJitConfig(
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_kv=block_kv,
        is_context_lens_2d=is_context_lens_2d,
    )
    _validate_config(cfg)
    return cfg


def paged_mqa_logits_config_dict(cfg: PagedMqaLogitsJitConfig) -> Dict[str, object]:
    return {
        "kind": "fp8_paged_mqa_logits",
        "next_n": cfg.next_n,
        "num_heads": cfg.num_heads,
        "head_dim": cfg.head_dim,
        "block_kv": cfg.block_kv,
        "is_context_lens_2d": cfg.is_context_lens_2d,
        "is_context_lens_2d_literal": "true" if cfg.is_context_lens_2d else "false",
    }


def paged_mqa_logits_dispatch_name(config: Mapping[str, object]) -> str:
    lens = "ctx2d" if config["is_context_lens_2d"] else "ctx1d"
    return (
        "mate_jit_fp8_paged_mqa_logits_"
        f"n{config['next_n']}_"
        f"h{config['num_heads']}_"
        f"d{config['head_dim']}_"
        f"kv{config['block_kv']}_"
        f"{lens}"
    )


def render_paged_mqa_logits_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = "fp8_paged_mqa_logits"
    return (
        TVM_HEADER
        + get_paged_mqa_logits_template("paged_mqa_logits_kern.j2").render(
            render_config
        )
        + EXPORT_FUNC.render(render_config)
    )


def _render_metadata_source(aligned_batch_size: int) -> str:
    template = get_paged_mqa_logits_template("mqa_logits_metadata_kern.j2")
    render_config = {
        "aligned_batch_size": aligned_batch_size,
        "func_name": "get_paged_mqa_logits_metadata",
    }
    return (
        TVM_HEADER + template.render(render_config) + EXPORT_FUNC.render(render_config)
    )


def gen_paged_mqa_logits_metadata_spec(aligned_batch_size: int) -> JitSpec:
    source_path = (
        jit_env.MATE_GEN_SRC_DIR
        / "deep_gemm"
        / f"mqa_logits_metadata_b{aligned_batch_size}.mu"
    )
    return gen_jit_spec(
        f"mqa_logits_metadata_b{aligned_batch_size}",
        [source_path],
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(DEEP_GEMM_CUDA_FLAGS) + ["-fno-signed-zeros"],
        extra_include_paths=_paged_mqa_logits_include_paths(),
        generated_sources={source_path: _render_metadata_source(aligned_batch_size)},
    )


def gen_paged_mqa_logits_spec(config: PagedMqaLogitsJitConfig) -> JitSpec:
    _validate_config(config)
    render_config = paged_mqa_logits_config_dict(config)
    dispatch_name = paged_mqa_logits_dispatch_name(render_config)
    source_path = _generated_source_path(dispatch_name)
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cuda_cflags=list(DEEP_GEMM_CUDA_FLAGS) + ["-fno-signed-zeros"],
        extra_include_paths=_paged_mqa_logits_include_paths(),
        generated_sources={source_path: render_paged_mqa_logits_source(render_config)},
    )


def gen_paged_mqa_logits_aot() -> list[JitSpec]:
    specs = [gen_paged_mqa_logits_spec(config) for config in PAGED_MQA_LOGITS_CONFIGS]
    seen: set[int] = set()
    for batch_size in _AOT_METADATA_BATCH_SIZES:
        aligned = round_up(batch_size, _ALIGN)
        if aligned not in seen:
            seen.add(aligned)
            specs.append(gen_paged_mqa_logits_metadata_spec(aligned))
    return specs


@functools.cache
def _load_paged_mqa_logits_metadata_module(aligned_batch_size: int):
    return gen_paged_mqa_logits_metadata_spec(aligned_batch_size).build_and_load()


def get_paged_mqa_logits_metadata_module(batch_size: int):
    aligned = round_up(batch_size, _ALIGN)
    return _load_paged_mqa_logits_metadata_module(aligned)


@functools.cache
def _load_paged_mqa_logits_module(frozen_config: tuple[int, int, int, int, bool]):
    config = PagedMqaLogitsJitConfig(*frozen_config)
    return gen_paged_mqa_logits_spec(config).build_and_load()


@functools.cache
def get_paged_mqa_logits_module(
    next_n: int,
    num_heads: int,
    head_dim: int,
    block_kv: int,
    is_context_lens_2d: bool,
):
    config = select_paged_mqa_logits_config(
        next_n,
        num_heads,
        head_dim,
        block_kv,
        is_context_lens_2d,
    )
    frozen_config = (
        config.next_n,
        config.num_heads,
        config.head_dim,
        config.block_kv,
        config.is_context_lens_2d,
    )
    return _load_paged_mqa_logits_module(frozen_config)


__all__ = [
    "PagedMqaLogitsJitConfig",
    "PAGED_MQA_LOGITS_CONFIGS",
    "gen_paged_mqa_logits_aot",
    "gen_paged_mqa_logits_metadata_spec",
    "gen_paged_mqa_logits_spec",
    "get_paged_mqa_logits_metadata_module",
    "get_paged_mqa_logits_module",
]
