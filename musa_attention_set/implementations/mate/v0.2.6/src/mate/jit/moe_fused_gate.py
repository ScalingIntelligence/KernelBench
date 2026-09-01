from __future__ import annotations

import functools

from .core import JitSpec, gen_jit_spec
from . import env as jit_env


MCC_FLAGS = [
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

CXX_FLAGS = [
    "-Wno-switch-bool",
]


def gen_moe_fused_gate_spec() -> JitSpec:
    return gen_jit_spec(
        "moe_fused_gate",
        [jit_env.MATE_CSRC_DIR / "moe_fused_gate.mu"],
        extra_cflags=CXX_FLAGS,
        extra_cuda_cflags=MCC_FLAGS + CXX_FLAGS,
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR,
            jit_env.MATE_CSRC_DIR,
            jit_env.MUTLASS_INCLUDE_DIR,
        ],
    )


def gen_moe_fused_gate_aot() -> list[JitSpec]:
    return [gen_moe_fused_gate_spec()]


@functools.cache
def get_moe_fused_gate_module():
    return gen_moe_fused_gate_spec().build_and_load()
