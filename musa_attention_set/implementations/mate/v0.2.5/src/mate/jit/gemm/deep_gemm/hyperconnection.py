from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

from jinja2 import Environment, FileSystemLoader

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...cpp_ext import get_mudnn_ldflags
from ...utils import EXPORT_FUNC, TVM_HEADER
from ....utils import ceil_div
from .. import gemm_utils
from .deep_gemm_utils import DEEP_GEMM_CUDA_FLAGS


@dataclass(frozen=True)
class HcPrenormGemmJitConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    stages: int
    num_mma_warp_squads: int


HC_PRENORM_TILE_M_OPTIONS = (32, 64)
HC_PRENORM_STAGE_OPTIONS = (3, 2)
HC_PRENORM_SMEM_BYTES_PER_MP = 192 * 1024
HC_PRENORM_BF16_BYTES = 2
HC_PRENORM_FP32_BYTES = 4

HC_PRENORM_CONFIGS: List[HcPrenormGemmJitConfig] = [
    HcPrenormGemmJitConfig(
        tile_m=tile_m,
        tile_n=32,
        tile_k=32,
        stages=stages,
        num_mma_warp_squads=2,
    )
    for stages in HC_PRENORM_STAGE_OPTIONS
    for tile_m in HC_PRENORM_TILE_M_OPTIONS
]

_HC_PRENORM_CONFIGS_BY_TILE_M_AND_STAGES = {
    (cfg.tile_m, cfg.stages): cfg for cfg in HC_PRENORM_CONFIGS
}


_hc_prenorm_template_env = Environment(
    loader=FileSystemLoader((gemm_utils.gemm_template_dir / "deep_gemm").as_posix()),
    keep_trailing_newline=True,
)


def get_hc_prenorm_template(name: str):
    return _hc_prenorm_template_env.get_template(name)


def _generated_source_path(dispatch_name: str) -> Path:
    return jit_env.MATE_GEN_SRC_DIR / "deep_gemm" / f"{dispatch_name}.mu"


def _hyperconnection_include_paths() -> list[Path]:
    return [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
    ]


def _hc_prenorm_waves(
    m: int,
    n: int,
    num_splits: int,
    num_mps: int,
    cfg: HcPrenormGemmJitConfig,
) -> int:
    smem_per_stage = (
        cfg.tile_m * cfg.tile_k * HC_PRENORM_BF16_BYTES
        + cfg.tile_m * cfg.tile_k * HC_PRENORM_FP32_BYTES
        + cfg.tile_n * cfg.tile_k * HC_PRENORM_FP32_BYTES
    )
    ctas_per_mp = max(1, HC_PRENORM_SMEM_BYTES_PER_MP // (cfg.stages * smem_per_stage))
    total_ctas = max(num_splits, 1) * ceil_div(m, cfg.tile_m) * ceil_div(n, cfg.tile_n)
    return ceil_div(total_ctas, max(num_mps, 1) * ctas_per_mp)


@functools.cache
def select_hc_prenorm_config(
    m: int,
    n: int,
    num_splits: int = 1,
    num_mps: int = 1,
) -> HcPrenormGemmJitConfig:
    target_tile_m = 32 if m <= 32 else 64
    cfg_2stage = _HC_PRENORM_CONFIGS_BY_TILE_M_AND_STAGES[(target_tile_m, 2)]
    cfg_3stage = _HC_PRENORM_CONFIGS_BY_TILE_M_AND_STAGES[(target_tile_m, 3)]
    target_stages = (
        2
        if _hc_prenorm_waves(m, n, num_splits, num_mps, cfg_2stage)
        < _hc_prenorm_waves(m, n, num_splits, num_mps, cfg_3stage)
        else 3
    )

    return _HC_PRENORM_CONFIGS_BY_TILE_M_AND_STAGES[(target_tile_m, target_stages)]


def hc_prenorm_config_dict(cfg: HcPrenormGemmJitConfig) -> Dict[str, object]:
    return {
        "kind": "tf32_hc_prenorm",
        "tile_m": cfg.tile_m,
        "tile_n": cfg.tile_n,
        "tile_k": cfg.tile_k,
        "stages": cfg.stages,
        "num_mma_warp_squads": cfg.num_mma_warp_squads,
    }


def hc_prenorm_dispatch_name(config: Mapping[str, object]) -> str:
    return (
        "mate_jit_tf32_hc_prenorm_"
        f"{config['tile_m']}x"
        f"{config['tile_n']}x"
        f"{config['tile_k']}_"
        f"{config['stages']}stages"
    )


def render_hc_prenorm_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = "tf32_hc_prenorm_gemm"
    return (
        TVM_HEADER
        + get_hc_prenorm_template("hc_prenorm_gemm_kern.j2").render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


def gen_hyperconnection_spec(config: HcPrenormGemmJitConfig) -> JitSpec:
    render_config = hc_prenorm_config_dict(config)
    dispatch_name = hc_prenorm_dispatch_name(render_config)
    source_path = _generated_source_path(dispatch_name)
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cuda_cflags=list(DEEP_GEMM_CUDA_FLAGS),
        extra_ldflags=list(get_mudnn_ldflags()),
        extra_include_paths=_hyperconnection_include_paths(),
        generated_sources={source_path: render_hc_prenorm_source(render_config)},
    )


def gen_hyperconnection_aot() -> list[JitSpec]:
    return [gen_hyperconnection_spec(config) for config in HC_PRENORM_CONFIGS]


@functools.cache
def _load_hyperconnection_module(frozen_config: tuple[int, int, int, int, int]):
    config = HcPrenormGemmJitConfig(*frozen_config)
    return gen_hyperconnection_spec(config).build_and_load()


@functools.cache
def get_hyperconnection_module(m: int, n: int, num_splits: int = 1, num_mps: int = 1):
    config = select_hc_prenorm_config(m, n, num_splits, num_mps)
    frozen_config = (
        config.tile_m,
        config.tile_n,
        config.tile_k,
        config.stages,
        config.num_mma_warp_squads,
    )
    return _load_hyperconnection_module(frozen_config)


__all__ = [
    "HcPrenormGemmJitConfig",
    "HC_PRENORM_CONFIGS",
    "gen_hyperconnection_aot",
    "gen_hyperconnection_spec",
]
