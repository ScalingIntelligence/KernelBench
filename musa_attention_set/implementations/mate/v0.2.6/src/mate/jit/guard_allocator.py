from __future__ import annotations

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


CXX_FLAGS = [
    "-O2",
]


def gen_guard_allocator_spec() -> JitSpec:
    return gen_jit_spec(
        "guard_allocator",
        [jit_env.MATE_CSRC_DIR / "guard_allocator.cpp"],
        extra_cflags=list(CXX_FLAGS),
    )


def gen_guard_allocator_aot() -> list[JitSpec]:
    return [gen_guard_allocator_spec()]
