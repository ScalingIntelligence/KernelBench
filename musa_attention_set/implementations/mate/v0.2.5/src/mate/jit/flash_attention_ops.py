from __future__ import annotations

import functools

from . import env as jit_env
from .cpp_ext import get_mudnn_ldflags
from .core import JitSpec, gen_jit_spec


CXX_FLAGS = [
    "-O3",
    "-Wno-switch-bool",
]

CUDA_FLAGS = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
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


def gen_flash_attention_ops_spec() -> JitSpec:
    sources = [
        jit_env.MATE_CSRC_DIR / "flash_atten_bwd.mu",
    ]
    return gen_jit_spec(
        "flash_attention_ops",
        sources,
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_ldflags=list(get_mudnn_ldflags()),
        extra_include_paths=list(INCLUDE_PATHS),
    )


def gen_flash_attention_ops_aot() -> list[JitSpec]:
    return [gen_flash_attention_ops_spec()]


@functools.cache
def get_flash_attention_ops_module():
    return gen_flash_attention_ops_spec().build_and_load()
