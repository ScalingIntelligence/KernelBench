import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from jinja2 import Environment, FileSystemLoader

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...cpp_ext import get_mudnn_ldflags
from ...utils import EXPORT_FUNC, TVM_HEADER, dtype_torch2mutlass_map
from .deep_gemm_utils import DEEP_GEMM_CUDA_FLAGS


dtype_torch2mutlass_map.setdefault(torch.float8_e4m3fn, "mutlass::float_e4m3_t")

TVM_DTYPE_BF16 = "dl_bfloat16"
TVM_DTYPE_FP8_E4M3FN = "dl_float8_e4m3fn"


DEEP_GEMM_GEMM_EXTRA_CFLAGS = [
    "-O3",
    "-Wno-switch-bool",
]

DEEP_GEMM_GEMM_EXTRA_CUDA_CFLAGS = list(DEEP_GEMM_CUDA_FLAGS) + [
    "-fno-signed-zeros",
]

DEEP_GEMM_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(
        (jit_env.MATE_TEMPLATE_DIR / "gemm" / "deep_gemm").as_posix()
    ),
    keep_trailing_newline=True,
)

GEMM_TYPE_NORMAL = "mate::deep_gemm::GemmType::Normal"
GEMM_TYPE_BATCHED = "mate::deep_gemm::GemmType::Batched"
GEMM_TYPE_M_GROUPED_CONTIGUOUS = "mate::deep_gemm::GemmType::MGroupedContiguous"
GEMM_TYPE_M_GROUPED_CONTIGUOUS_PSUM = (
    "mate::deep_gemm::GemmType::MGroupedContiguousWithPsumLayout"
)
GEMM_TYPE_M_GROUPED_MASKED = "mate::deep_gemm::GemmType::MGroupedMasked"

_SUPPORTED_KINDS = ("bf16", "fp8")
_SUPPORTED_GEMM_TYPES = (
    GEMM_TYPE_NORMAL,
    GEMM_TYPE_BATCHED,
    GEMM_TYPE_M_GROUPED_CONTIGUOUS,
    GEMM_TYPE_M_GROUPED_CONTIGUOUS_PSUM,
    GEMM_TYPE_M_GROUPED_MASKED,
)

FP8_SCALE_ACCUMULATION_MODE_AUTO = "auto"
FP8_SCALE_ACCUMULATION_MODE_ITERATIVE = "iterative"
FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER = "dualbuffer"
_FP8_SCALE_ACCUMULATION_MODES = (
    FP8_SCALE_ACCUMULATION_MODE_ITERATIVE,
    FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER,
)
_FP8_SCALE_ACCUMULATION_MODE_ALIASES = {
    FP8_SCALE_ACCUMULATION_MODE_AUTO: FP8_SCALE_ACCUMULATION_MODE_AUTO,
    FP8_SCALE_ACCUMULATION_MODE_ITERATIVE: FP8_SCALE_ACCUMULATION_MODE_ITERATIVE,
    "iter": FP8_SCALE_ACCUMULATION_MODE_ITERATIVE,
    FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER: FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER,
    "dual_buffer": FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER,
    "dual-buffer": FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER,
    "2buffer": FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER,
    "two_buffer": FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER,
    "twobuffer": FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER,
}
_FP8_SCALE_ACCUMULATION_MODE_CPP = {
    FP8_SCALE_ACCUMULATION_MODE_ITERATIVE: ("mate::deep_gemm::ScaleMode::Iterative"),
    FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER: ("mate::deep_gemm::ScaleMode::DualBuffer"),
}
_FP8_SCALE_ACCUMULATION_MODE_ENV = "MATE_DEEP_GEMM_FP8_SCALE_ACCUM_MODE"


@dataclass(frozen=True)
class GemmJitConfig:
    tile_m: int
    tile_n: int
    stages: int
    num_mma_warp_squads: int
    scale_accumulation_mode: Optional[str] = None


@dataclass(frozen=True)
class DeepGemmGemmKernelConfig:
    kind: str
    gemm_type: str
    tile_m: int
    tile_n: int
    tile_k: int
    stages: int
    num_mma_warp_squads: int
    quant_tile: int = 0
    scale_accumulation_mode: str = FP8_SCALE_ACCUMULATION_MODE_ITERATIVE


DEEP_GEMM_CONFIGS = [
    GemmJitConfig(tile_m=32, tile_n=256, stages=4, num_mma_warp_squads=2),
    GemmJitConfig(tile_m=64, tile_n=256, stages=3, num_mma_warp_squads=2),
    GemmJitConfig(tile_m=128, tile_n=256, stages=3, num_mma_warp_squads=2),
    GemmJitConfig(tile_m=256, tile_n=256, stages=3, num_mma_warp_squads=4),
]


def deep_gemm_gemm_extra_include_paths() -> list[Path]:
    return [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
        jit_env.MUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include",
        jit_env.MUTLASS_INCLUDE_DIR.parent / "experimental" / "fmha",
    ]


def get_deep_gemm_gemm_template(name: str):
    return DEEP_GEMM_TEMPLATE_ENV.get_template(name)


def _generated_source_path(dispatch_name: str) -> Path:
    return jit_env.MATE_GEN_SRC_DIR / "deep_gemm" / f"{dispatch_name}.mu"


def _select_config(m: int) -> GemmJitConfig:
    best = DEEP_GEMM_CONFIGS[0]
    for cfg in DEEP_GEMM_CONFIGS:
        if cfg.tile_m <= m:
            best = cfg
    return best


def _select_fp8_grouped_config(m: int, alignment_m: int) -> GemmJitConfig:
    if alignment_m not in (128, 256):
        raise ValueError(f"alignment_m must be 128 or 256, got {alignment_m}")
    return _select_config(min(m, alignment_m))


def _deep_gemm_gemm_op_name(gemm_type: str, kind: str) -> str:
    if gemm_type == GEMM_TYPE_NORMAL:
        return f"deep_gemm_gemm_{kind}_gemm_nt"
    if gemm_type == GEMM_TYPE_BATCHED:
        return f"deep_gemm_gemm_{kind}_bmm"
    if gemm_type == GEMM_TYPE_M_GROUPED_CONTIGUOUS_PSUM:
        return f"deep_gemm_gemm_{kind}_m_grouped_contiguous_psum"
    if gemm_type == GEMM_TYPE_M_GROUPED_CONTIGUOUS:
        return f"deep_gemm_gemm_{kind}_m_grouped_contiguous"
    if gemm_type == GEMM_TYPE_M_GROUPED_MASKED:
        return f"deep_gemm_gemm_{kind}_m_grouped_masked"
    raise ValueError(f"Unsupported DeepGEMM GEMM type: {gemm_type}")


def _deep_gemm_gemm_func_name(config: DeepGemmGemmKernelConfig) -> str:
    return _deep_gemm_gemm_op_name(config.gemm_type, config.kind)


def _default_num_mma_warp_squads(config: DeepGemmGemmKernelConfig) -> Optional[int]:
    for jit_config in DEEP_GEMM_CONFIGS:
        if (
            jit_config.tile_m == config.tile_m
            and jit_config.tile_n == config.tile_n
            and jit_config.stages == config.stages
        ):
            return jit_config.num_mma_warp_squads
    return None


def deep_gemm_gemm_dispatch_name(config: DeepGemmGemmKernelConfig) -> str:
    suffix = f"{config.tile_m}x{config.tile_n}x{config.tile_k}_{config.stages}stages"
    default_wgs = _default_num_mma_warp_squads(config)
    if default_wgs != config.num_mma_warp_squads:
        suffix += f"_{config.num_mma_warp_squads}wgs"
    if config.kind == "fp8":
        suffix += f"_quant1x{config.quant_tile}x{config.quant_tile}"
        if config.scale_accumulation_mode == FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER:
            suffix += "_dualbuffer"
    return f"mate_jit_{_deep_gemm_gemm_func_name(config)}_{suffix}"


def _deep_gemm_gemm_config_dict(
    config: DeepGemmGemmKernelConfig,
) -> dict[str, object]:
    _validate_deep_gemm_gemm_kernel_config(config)
    render_config: dict[str, object] = {
        "kind": config.kind,
        "gemm_type": config.gemm_type,
        "tile_m": config.tile_m,
        "tile_n": config.tile_n,
        "tile_k": config.tile_k,
        "stages": config.stages,
        "num_mma_warp_squads": config.num_mma_warp_squads,
    }
    if config.kind == "bf16":
        render_config.update(
            element_a=dtype_torch2mutlass_map[torch.bfloat16],
            element_b=dtype_torch2mutlass_map[torch.bfloat16],
            element_d=dtype_torch2mutlass_map[torch.bfloat16],
            tvm_dtype_a=TVM_DTYPE_BF16,
            tvm_dtype_b=TVM_DTYPE_BF16,
            tvm_dtype_d=TVM_DTYPE_BF16,
        )
    else:
        render_config.update(
            quant_tile=config.quant_tile,
            scale_accumulation_mode=config.scale_accumulation_mode,
            scale_accumulation_mode_cpp=_FP8_SCALE_ACCUMULATION_MODE_CPP[
                config.scale_accumulation_mode
            ],
            element_a=dtype_torch2mutlass_map[torch.float8_e4m3fn],
            element_b=dtype_torch2mutlass_map[torch.float8_e4m3fn],
            element_d=dtype_torch2mutlass_map[torch.bfloat16],
            tvm_dtype_a=TVM_DTYPE_FP8_E4M3FN,
            tvm_dtype_b=TVM_DTYPE_FP8_E4M3FN,
            tvm_dtype_d=TVM_DTYPE_BF16,
        )
    return render_config


def _validate_deep_gemm_gemm_kernel_config(config: DeepGemmGemmKernelConfig) -> None:
    if config.kind not in _SUPPORTED_KINDS:
        raise ValueError(
            f"Unsupported DeepGEMM GEMM kind: {config.kind}. "
            f"Expected one of {_SUPPORTED_KINDS}"
        )
    if config.gemm_type not in _SUPPORTED_GEMM_TYPES:
        raise ValueError(
            f"Unsupported DeepGEMM GEMM type: {config.gemm_type}. "
            f"Expected one of {_SUPPORTED_GEMM_TYPES}"
        )
    if config.kind == "fp8" and config.scale_accumulation_mode not in (
        _FP8_SCALE_ACCUMULATION_MODES
    ):
        raise ValueError(
            "Unsupported DeepGEMM FP8 scale accumulation mode: "
            f"{config.scale_accumulation_mode}. "
            f"Expected one of {_FP8_SCALE_ACCUMULATION_MODES}"
        )


def _normalize_fp8_scale_accumulation_mode(mode: object, *, allow_auto: bool) -> str:
    key = str(mode).strip().lower()
    normalized = _FP8_SCALE_ACCUMULATION_MODE_ALIASES.get(key)
    allowed = (
        (FP8_SCALE_ACCUMULATION_MODE_AUTO, *_FP8_SCALE_ACCUMULATION_MODES)
        if allow_auto
        else _FP8_SCALE_ACCUMULATION_MODES
    )
    if normalized not in allowed:
        raise ValueError(
            f"Unsupported DeepGEMM FP8 scale accumulation mode: {mode}. "
            f"Expected one of {allowed}"
        )
    return normalized


def _resolve_fp8_scale_accumulation_mode(
    tile_m: int, override: Optional[str] = None
) -> str:
    mode = override
    if mode is None:
        mode = os.environ.get(
            _FP8_SCALE_ACCUMULATION_MODE_ENV, FP8_SCALE_ACCUMULATION_MODE_AUTO
        )
    normalized = _normalize_fp8_scale_accumulation_mode(mode, allow_auto=True)
    if normalized == FP8_SCALE_ACCUMULATION_MODE_AUTO:
        return (
            FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER
            if tile_m <= 64
            else FP8_SCALE_ACCUMULATION_MODE_ITERATIVE
        )
    return normalized


def _normalize_deep_gemm_gemm_config(
    config: Mapping[str, Any] | DeepGemmGemmKernelConfig,
) -> DeepGemmGemmKernelConfig:
    if isinstance(config, DeepGemmGemmKernelConfig):
        return config

    kind = str(config["kind"])
    scale_accumulation_mode = FP8_SCALE_ACCUMULATION_MODE_ITERATIVE
    if kind == "fp8":
        scale_accumulation_mode = _normalize_fp8_scale_accumulation_mode(
            config.get(
                "scale_accumulation_mode", FP8_SCALE_ACCUMULATION_MODE_ITERATIVE
            ),
            allow_auto=False,
        )
    return DeepGemmGemmKernelConfig(
        kind=kind,
        gemm_type=str(config["gemm_type"]),
        tile_m=int(config["tile_m"]),
        tile_n=int(config["tile_n"]),
        tile_k=int(config["tile_k"]),
        stages=int(config["stages"]),
        num_mma_warp_squads=int(config["num_mma_warp_squads"]),
        quant_tile=int(config.get("quant_tile", 128 if kind == "fp8" else 0)),
        scale_accumulation_mode=scale_accumulation_mode,
    )


def _render_deep_gemm_gemm_source(
    config: Mapping[str, Any] | DeepGemmGemmKernelConfig,
) -> str:
    kernel_config = _normalize_deep_gemm_gemm_config(config)
    _validate_deep_gemm_gemm_kernel_config(kernel_config)
    render_config = _deep_gemm_gemm_config_dict(kernel_config)
    render_config["func_name"] = deep_gemm_gemm_dispatch_name(kernel_config)
    template_name = (
        "bf16_gemm_kern.j2" if kernel_config.kind == "bf16" else "fp8_gemm_1d2d_kern.j2"
    )
    return (
        TVM_HEADER
        + get_deep_gemm_gemm_template(template_name).render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


def _freeze_deep_gemm_gemm_config(
    config: DeepGemmGemmKernelConfig,
) -> tuple[str, str, int, int, int, int, int, int, str]:
    return (
        config.kind,
        config.gemm_type,
        config.tile_m,
        config.tile_n,
        config.tile_k,
        config.stages,
        config.num_mma_warp_squads,
        config.quant_tile,
        config.scale_accumulation_mode,
    )


@functools.cache
def _load_deep_gemm_gemm_module(
    frozen_config: tuple[str, str, int, int, int, int, int, int, str],
):
    config = DeepGemmGemmKernelConfig(*frozen_config)
    return gen_deep_gemm_gemm_spec(config).build_and_load()


def get_deep_gemm_gemm_module(
    kind: str,
    gemm_type: str,
    config_m: int | None = None,
    alignment_m: int = 128,
    config: Optional[GemmJitConfig] = None,
    scale_accumulation_mode: Optional[str] = None,
):
    kernel_config = _resolve_deep_gemm_gemm_kernel_config(
        kind=kind,
        gemm_type=gemm_type,
        config_m=config_m,
        alignment_m=alignment_m,
        config=config,
        scale_accumulation_mode=scale_accumulation_mode,
    )
    return (
        deep_gemm_gemm_dispatch_name(kernel_config),
        _load_deep_gemm_gemm_module(_freeze_deep_gemm_gemm_config(kernel_config)),
    )


def select_deep_gemm_gemm_config(
    *,
    kind: str,
    gemm_type: str,
    config_m: int | None,
    alignment_m: int = 128,
    config: Optional[GemmJitConfig] = None,
) -> GemmJitConfig:
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(f"Unsupported DeepGEMM GEMM kind: {kind}")
    if gemm_type not in _SUPPORTED_GEMM_TYPES:
        raise ValueError(f"Unsupported DeepGEMM GEMM type: {gemm_type}")
    if config is not None:
        return config
    if config_m is None:
        raise ValueError("config_m must be set when config is not provided")
    if kind == "fp8" and gemm_type in (
        GEMM_TYPE_M_GROUPED_CONTIGUOUS,
        GEMM_TYPE_M_GROUPED_CONTIGUOUS_PSUM,
        GEMM_TYPE_M_GROUPED_MASKED,
    ):
        return _select_fp8_grouped_config(config_m, alignment_m)
    return _select_config(config_m)


def _make_kernel_config(
    *,
    kind: str,
    gemm_type: str,
    config: GemmJitConfig,
    scale_accumulation_mode: Optional[str] = None,
) -> DeepGemmGemmKernelConfig:
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(f"Unsupported DeepGEMM GEMM kind: {kind}")
    if gemm_type not in _SUPPORTED_GEMM_TYPES:
        raise ValueError(f"Unsupported DeepGEMM GEMM type: {gemm_type}")
    if kind == "bf16":
        return DeepGemmGemmKernelConfig(
            kind=kind,
            gemm_type=gemm_type,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            tile_k=64,
            stages=config.stages,
            num_mma_warp_squads=config.num_mma_warp_squads,
        )
    return DeepGemmGemmKernelConfig(
        kind=kind,
        gemm_type=gemm_type,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=128,
        stages=config.stages,
        num_mma_warp_squads=config.num_mma_warp_squads,
        quant_tile=128,
        scale_accumulation_mode=_resolve_fp8_scale_accumulation_mode(
            config.tile_m,
            scale_accumulation_mode
            if scale_accumulation_mode is not None
            else config.scale_accumulation_mode,
        ),
    )


def _resolve_deep_gemm_gemm_kernel_config(
    *,
    kind: str,
    gemm_type: str,
    config_m: int | None,
    alignment_m: int = 128,
    config: Optional[GemmJitConfig] = None,
    scale_accumulation_mode: Optional[str] = None,
) -> DeepGemmGemmKernelConfig:
    selected = select_deep_gemm_gemm_config(
        kind=kind,
        gemm_type=gemm_type,
        config_m=config_m,
        alignment_m=alignment_m,
        config=config,
    )
    return _make_kernel_config(
        kind=kind,
        gemm_type=gemm_type,
        config=selected,
        scale_accumulation_mode=scale_accumulation_mode,
    )


def get_deep_gemm_gemm_aot_configs() -> list[dict[str, object]]:
    configs: list[DeepGemmGemmKernelConfig] = []
    gemm_types = (
        GEMM_TYPE_NORMAL,
        GEMM_TYPE_M_GROUPED_CONTIGUOUS,
        GEMM_TYPE_M_GROUPED_CONTIGUOUS_PSUM,
        GEMM_TYPE_M_GROUPED_MASKED,
        GEMM_TYPE_BATCHED,
    )
    for gemm_type in gemm_types:
        for jit_config in DEEP_GEMM_CONFIGS:
            configs.append(
                _make_kernel_config(kind="bf16", gemm_type=gemm_type, config=jit_config)
            )
            configs.append(
                _make_kernel_config(kind="fp8", gemm_type=gemm_type, config=jit_config)
            )
    deduped = {}
    for kernel_config in configs:
        deduped[deep_gemm_gemm_dispatch_name(kernel_config)] = kernel_config
    return [
        _deep_gemm_gemm_config_dict(kernel_config) for kernel_config in deduped.values()
    ]


def gen_deep_gemm_gemm_spec(
    config: Mapping[str, Any] | DeepGemmGemmKernelConfig,
) -> JitSpec:
    kernel_config = _normalize_deep_gemm_gemm_config(config)
    dispatch_name = deep_gemm_gemm_dispatch_name(kernel_config)
    source_file = _generated_source_path(dispatch_name)
    return gen_jit_spec(
        name=dispatch_name,
        sources=[source_file],
        generated_sources={source_file: _render_deep_gemm_gemm_source(kernel_config)},
        extra_cflags=list(DEEP_GEMM_GEMM_EXTRA_CFLAGS),
        extra_cuda_cflags=list(DEEP_GEMM_GEMM_EXTRA_CUDA_CFLAGS),
        extra_ldflags=list(get_mudnn_ldflags()),
        extra_include_paths=deep_gemm_gemm_extra_include_paths(),
    )


def gen_deep_gemm_gemm_specs(
    configs: Sequence[Mapping[str, object] | DeepGemmGemmKernelConfig],
) -> list[JitSpec]:
    return [gen_deep_gemm_gemm_spec(config) for config in configs]


def gen_deep_gemm_gemm_aot() -> list[JitSpec]:
    return gen_deep_gemm_gemm_specs(get_deep_gemm_gemm_aot_configs())


__all__ = [
    "GemmJitConfig",
    "DeepGemmGemmKernelConfig",
    "DEEP_GEMM_CONFIGS",
    "GEMM_TYPE_NORMAL",
    "GEMM_TYPE_BATCHED",
    "GEMM_TYPE_M_GROUPED_CONTIGUOUS",
    "GEMM_TYPE_M_GROUPED_CONTIGUOUS_PSUM",
    "GEMM_TYPE_M_GROUPED_MASKED",
    "FP8_SCALE_ACCUMULATION_MODE_AUTO",
    "FP8_SCALE_ACCUMULATION_MODE_ITERATIVE",
    "FP8_SCALE_ACCUMULATION_MODE_DUAL_BUFFER",
    "get_deep_gemm_gemm_aot_configs",
    "get_deep_gemm_gemm_module",
    "select_deep_gemm_gemm_config",
    "gen_deep_gemm_gemm_spec",
    "gen_deep_gemm_gemm_specs",
    "gen_deep_gemm_gemm_aot",
]
