from __future__ import annotations

import functools
from pathlib import Path
from typing import Mapping

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .utils import dtype_torch2mutlass_map


CXX_FLAGS = [
    "-O3",
    "-Wno-switch-bool",
]

CUDA_FLAGS = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

INCLUDE_PATHS = [
    jit_env.MATE_INCLUDE_DIR,
    jit_env.MATE_CSRC_DIR,
    jit_env.MUTLASS_INCLUDE_DIR,
]

KDA_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}

KDA_DTYPE_FFI = {
    torch.float16: "dl_float16",
    torch.bfloat16: "dl_bfloat16",
}

KDA_CU_SEQLENS_ELEMENT_NAMES = {
    torch.int32: "int32_t",
    torch.int64: "int64_t",
}

KDA_CU_SEQLENS_DTYPE_NAMES = {
    torch.int32: "i32",
    torch.int64: "i64",
}

KDA_CU_SEQLENS_DTYPE_FFI = {
    torch.int32: "dl_int32",
    torch.int64: "dl_int64",
}

KDA_FUSED_BOOL_KEYS = (
    "has_state_in",
    "has_state_out",
    "state_fp32",
    "has_gate_params",
    "is_varlen",
    "normalize_qk",
)
KDA_FUSED_CONFIG_KEYS = (
    "element",
    "element_dtype",
    "dtype_name",
    "state_element",
    "state_dtype_name",
    "cu_seqlens_element",
    "cu_seqlens_dtype",
    "cu_seqlens_dtype_name",
    *KDA_FUSED_BOOL_KEYS,
)

KDA_FUSED_DEFAULT_CONFIG = {
    "element": dtype_torch2mutlass_map[torch.bfloat16],
    "element_dtype": KDA_DTYPE_FFI[torch.bfloat16],
    "dtype_name": KDA_DTYPE_NAMES[torch.bfloat16],
    "state_element": dtype_torch2mutlass_map[torch.bfloat16],
    "state_dtype_name": KDA_DTYPE_NAMES[torch.bfloat16],
    "cu_seqlens_element": KDA_CU_SEQLENS_ELEMENT_NAMES[torch.int64],
    "cu_seqlens_dtype": KDA_CU_SEQLENS_DTYPE_FFI[torch.int64],
    "cu_seqlens_dtype_name": KDA_CU_SEQLENS_DTYPE_NAMES[torch.int64],
    "has_state_in": True,
    "has_state_out": True,
    "state_fp32": False,
    "has_gate_params": True,
    "is_varlen": False,
    "normalize_qk": True,
}


@functools.lru_cache
def _get_kda_template_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _get_kda_template(template_name: str):
    template_dir = (jit_env.MATE_TEMPLATE_DIR / "flat" / "kda").as_posix()
    return _get_kda_template_env(template_dir).get_template(template_name)


KDA_FUSED_TEMPLATE = _get_kda_template("chunk_kda_kernel.j2")


def make_kda_fused_ops_config(
    dtype: torch.dtype,
    *,
    state_dtype: torch.dtype,
    cu_seqlens_dtype: torch.dtype | None = None,
    has_state_in: bool,
    has_state_out: bool,
    state_fp32: bool,
    has_gate_params: bool,
    is_varlen: bool,
    normalize_qk: bool,
) -> dict[str, object]:
    if dtype not in dtype_torch2mutlass_map or dtype not in KDA_DTYPE_FFI:
        raise TypeError(
            "chunk_kda fused supports torch.float16 and torch.bfloat16 inputs."
        )
    if state_dtype not in dtype_torch2mutlass_map or state_dtype not in KDA_DTYPE_NAMES:
        raise TypeError(
            "chunk_kda fused supports torch.float16, torch.bfloat16, and torch.float32 state dtypes."
        )
    if cu_seqlens_dtype is None:
        cu_seqlens_dtype = torch.int64
    if (
        cu_seqlens_dtype not in KDA_CU_SEQLENS_ELEMENT_NAMES
        or cu_seqlens_dtype not in KDA_CU_SEQLENS_DTYPE_FFI
    ):
        raise TypeError(
            "chunk_kda fused supports torch.int32 and torch.int64 cu_seqlens."
        )
    return {
        "element": dtype_torch2mutlass_map[dtype],
        "element_dtype": KDA_DTYPE_FFI[dtype],
        "dtype_name": KDA_DTYPE_NAMES[dtype],
        "state_element": dtype_torch2mutlass_map[state_dtype],
        "state_dtype_name": KDA_DTYPE_NAMES[state_dtype],
        "cu_seqlens_element": KDA_CU_SEQLENS_ELEMENT_NAMES[cu_seqlens_dtype],
        "cu_seqlens_dtype": KDA_CU_SEQLENS_DTYPE_FFI[cu_seqlens_dtype],
        "cu_seqlens_dtype_name": KDA_CU_SEQLENS_DTYPE_NAMES[cu_seqlens_dtype],
        "has_state_in": bool(has_state_in),
        "has_state_out": bool(has_state_out),
        "state_fp32": bool(state_fp32),
        "has_gate_params": bool(has_gate_params),
        "is_varlen": bool(is_varlen),
        "normalize_qk": bool(normalize_qk),
    }


def _normalize_kda_fused_ops_config(
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized = dict(KDA_FUSED_DEFAULT_CONFIG)
    if config is not None:
        normalized.update(config)

    for key in KDA_FUSED_CONFIG_KEYS:
        if key not in normalized:
            raise ValueError(f"Missing KDA fused JIT config key: {key}")
    for key in KDA_FUSED_BOOL_KEYS:
        normalized[key] = bool(normalized[key])
    if not normalized["is_varlen"]:
        normalized["cu_seqlens_element"] = KDA_FUSED_DEFAULT_CONFIG[
            "cu_seqlens_element"
        ]
        normalized["cu_seqlens_dtype"] = KDA_FUSED_DEFAULT_CONFIG["cu_seqlens_dtype"]
        normalized["cu_seqlens_dtype_name"] = KDA_FUSED_DEFAULT_CONFIG[
            "cu_seqlens_dtype_name"
        ]
    return {key: normalized[key] for key in KDA_FUSED_CONFIG_KEYS}


def _kda_fused_ops_encode(config: Mapping[str, object]) -> str:
    bool_suffix = "_".join(
        f"{key}_{int(bool(config[key]))}" for key in KDA_FUSED_BOOL_KEYS
    )
    cu_seqlens_suffix = (
        f"_seqlens_{config['cu_seqlens_dtype_name']}"
        if bool(config["is_varlen"])
        else ""
    )
    return (
        f"chunk_kda_fused_mutlass_dtype_{config['dtype_name']}"
        f"_state_{config['state_dtype_name']}_{bool_suffix}"
        f"{cu_seqlens_suffix}"
    )


def _render_kda_fused_ops_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = _kda_fused_ops_encode(config)
    for key in KDA_FUSED_BOOL_KEYS:
        render_config[key] = "true" if bool(config[key]) else "false"
    return KDA_FUSED_TEMPLATE.render(render_config)


def get_kda_fused_ops_function_name(config: Mapping[str, object] | None = None) -> str:
    return _kda_fused_ops_encode(_normalize_kda_fused_ops_config(config))


def gen_kda_fused_ops_spec(config: Mapping[str, object] | None = None) -> JitSpec:
    config = _normalize_kda_fused_ops_config(config)
    dispatch_name = _kda_fused_ops_encode(config)
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / "kda" / f"{dispatch_name}.mu")
    return gen_jit_spec(
        dispatch_name,
        [source_file],
        generated_sources={source_file: _render_kda_fused_ops_source(config)},
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    )


def gen_kda_fused_ops_aot() -> list[JitSpec]:
    return [gen_kda_fused_ops_spec()]


@functools.cache
def _load_kda_fused_ops_module(frozen_config: tuple[tuple[str, object], ...]):
    return gen_kda_fused_ops_spec(dict(frozen_config)).build_and_load()


def get_kda_fused_ops_module(config: Mapping[str, object] | None = None):
    normalized = _normalize_kda_fused_ops_config(config)
    return _load_kda_fused_ops_module(tuple(sorted(normalized.items())))
