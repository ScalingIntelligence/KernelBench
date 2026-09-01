from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

from jinja2 import Environment, FileSystemLoader

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...utils import EXPORT_FUNC, TVM_HEADER
from .. import gemm_utils
from .deep_gemm_utils import DEEP_GEMM_CUDA_FLAGS


@dataclass(frozen=True)
class MqaLogitsJitConfig:
    block_q: int
    block_kv: int
    num_heads: int
    head_dim: int
    compressed_logits: bool


MQA_LOGITS_BLOCK_Q_OPTIONS = (2, 4)
MQA_LOGITS_BLOCK_KV_OPTIONS = (64, 128)
MQA_LOGITS_NUM_HEADS_OPTIONS = (32, 64)
MQA_LOGITS_HEAD_DIM_OPTIONS = (32, 64, 128)

MQA_LOGITS_CONFIGS: List[MqaLogitsJitConfig] = [
    MqaLogitsJitConfig(
        block_q=block_q,
        block_kv=block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        compressed_logits=compressed_logits,
    )
    for compressed_logits in (False, True)
    for num_heads in MQA_LOGITS_NUM_HEADS_OPTIONS
    for head_dim in MQA_LOGITS_HEAD_DIM_OPTIONS
    for block_q in MQA_LOGITS_BLOCK_Q_OPTIONS
    for block_kv in MQA_LOGITS_BLOCK_KV_OPTIONS
]


_mqa_logits_template_env = Environment(
    loader=FileSystemLoader((gemm_utils.gemm_template_dir / "deep_gemm").as_posix()),
    keep_trailing_newline=True,
)


def get_mqa_logits_template(name: str):
    return _mqa_logits_template_env.get_template(name)


def _generated_source_path(dispatch_name: str) -> Path:
    return jit_env.MATE_GEN_SRC_DIR / "deep_gemm" / f"{dispatch_name}.mu"


def _mqa_logits_include_paths() -> list[Path]:
    return [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
    ]


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _validate_config(cfg: MqaLogitsJitConfig) -> None:
    if cfg.block_q not in MQA_LOGITS_BLOCK_Q_OPTIONS:
        raise ValueError(f"Unsupported MQA logits block_q={cfg.block_q}")
    if cfg.block_kv not in MQA_LOGITS_BLOCK_KV_OPTIONS:
        raise ValueError(f"Unsupported MQA logits block_kv={cfg.block_kv}")
    if cfg.num_heads not in MQA_LOGITS_NUM_HEADS_OPTIONS:
        raise ValueError(f"Unsupported MQA logits num_heads={cfg.num_heads}")
    if cfg.head_dim not in MQA_LOGITS_HEAD_DIM_OPTIONS:
        raise ValueError(f"Unsupported MQA logits head_dim={cfg.head_dim}")


@functools.cache
def select_mqa_logits_config(
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    compressed_logits: bool,
    num_mps: int = 1,
) -> MqaLogitsJitConfig:
    del seq_len, num_mps

    block_q = _env_int("MATE_MQA_LOGITS_BLOCK_Q")
    block_kv = _env_int("MATE_MQA_LOGITS_BLOCK_KV")

    if block_q is None:
        block_q = 2

    if block_kv is None:
        block_kv = 128
        tail = seq_len_kv % (4 * block_kv)
        if (
            not compressed_logits
            and num_heads == 32
            and head_dim == 128
            and 0 < tail <= block_kv
        ):
            block_kv = 64

    cfg = MqaLogitsJitConfig(
        block_q=block_q,
        block_kv=block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        compressed_logits=compressed_logits,
    )
    _validate_config(cfg)
    return cfg


def mqa_logits_config_dict(cfg: MqaLogitsJitConfig) -> Dict[str, object]:
    return {
        "kind": "fp8_mqa_logits",
        "block_q": cfg.block_q,
        "block_kv": cfg.block_kv,
        "num_heads": cfg.num_heads,
        "head_dim": cfg.head_dim,
        "compressed_logits": cfg.compressed_logits,
        "compressed_logits_literal": "true" if cfg.compressed_logits else "false",
    }


def mqa_logits_dispatch_name(config: Mapping[str, object]) -> str:
    mode = "comp" if config["compressed_logits"] else "full"
    return (
        "mate_jit_fp8_mqa_logits_"
        f"h{config['num_heads']}_"
        f"d{config['head_dim']}_"
        f"q{config['block_q']}_"
        f"kv{config['block_kv']}_"
        f"{mode}"
    )


def render_mqa_logits_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = "fp8_mqa_logits"
    return (
        TVM_HEADER
        + get_mqa_logits_template("mqa_logits_kern.j2").render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


def gen_mqa_logits_spec(config: MqaLogitsJitConfig) -> JitSpec:
    _validate_config(config)
    render_config = mqa_logits_config_dict(config)
    dispatch_name = mqa_logits_dispatch_name(render_config)
    source_path = _generated_source_path(dispatch_name)
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cuda_cflags=list(DEEP_GEMM_CUDA_FLAGS) + ["-fno-signed-zeros"],
        extra_include_paths=_mqa_logits_include_paths(),
        generated_sources={source_path: render_mqa_logits_source(render_config)},
    )


def gen_mqa_logits_aot() -> list[JitSpec]:
    return [gen_mqa_logits_spec(config) for config in MQA_LOGITS_CONFIGS]


@functools.cache
def _load_mqa_logits_module(frozen_config: tuple[int, int, int, int, bool]):
    config = MqaLogitsJitConfig(*frozen_config)
    return gen_mqa_logits_spec(config).build_and_load()


@functools.cache
def get_mqa_logits_module(
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    compressed_logits: bool,
    num_mps: int = 1,
):
    config = select_mqa_logits_config(
        seq_len,
        seq_len_kv,
        num_heads,
        head_dim,
        compressed_logits,
        num_mps,
    )
    frozen_config = (
        config.block_q,
        config.block_kv,
        config.num_heads,
        config.head_dim,
        config.compressed_logits,
    )
    return _load_mqa_logits_module(frozen_config)


__all__ = [
    "MqaLogitsJitConfig",
    "MQA_LOGITS_CONFIGS",
    "gen_mqa_logits_aot",
    "gen_mqa_logits_spec",
    "get_mqa_logits_module",
]
