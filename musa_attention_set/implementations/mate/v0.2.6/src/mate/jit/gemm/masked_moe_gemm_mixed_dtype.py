from __future__ import annotations

import functools

import torch
from jinja2 import Environment, FileSystemLoader

from ...mate_runtime import resolve_num_mps
from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec
from ..utils import EXPORT_FUNC, TVM_HEADER, dtype_torch2mutlass_map
from . import gemm_utils

MIXED_DTYPE_GEMM_OUTPUT_DTYPE_TAGS = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}
MIXED_DTYPE_GEMM_A_QUANT_RECIPE_TAGS = {
    (1, -1): "Apertoken",
    (1, 128): "Apergroup",
}

MIXED_DTYPE_GEMM_MCC_FLAGS = ["-Od3", "-O3", "-std=c++17"]
MIXED_DTYPE_GEMM_CXX_FLAGS = ["-O3"]


def get_mixed_dtype_moe_gemm_config() -> tuple[int, int, int, int]:
    tile_m = 32
    tile_n = 256
    tile_k = 256
    stages = 3
    return tile_m, tile_n, tile_k, stages


def masked_moe_gemm_mixed_dtype_dispatch_name(
    out_dtype: torch.dtype,
    a_quant_recipe: tuple[int, int],
    scale_a_major: str = "K",
) -> str:
    if scale_a_major not in ("K", "M"):
        raise ValueError(f"scale_a_major must be 'K' or 'M', got {scale_a_major!r}")
    if scale_a_major == "M" and a_quant_recipe != (1, 128):
        raise ValueError("M-major Scale-A requires grouped A quantization")
    out_dtype_tag = MIXED_DTYPE_GEMM_OUTPUT_DTYPE_TAGS[out_dtype]
    a_quant_tag = MIXED_DTYPE_GEMM_A_QUANT_RECIPE_TAGS[a_quant_recipe]
    scale_a_major_tag = "_Mmajor" if scale_a_major == "M" else ""
    tile_m, tile_n, tile_k, stages = get_mixed_dtype_moe_gemm_config()
    return (
        "mp31_masked_moe_gemm_mixed_dtype_"
        f"{out_dtype_tag}_{tile_m}x{tile_n}x{tile_k}_s{stages}_{a_quant_tag}{scale_a_major_tag}"
    )


def render_masked_moe_gemm_mixed_dtype_source(
    out_dtype: torch.dtype,
    a_quant_recipe: tuple[int, int],
    scale_a_major: str = "K",
) -> str:
    tile_m, tile_n, tile_k, stages = get_mixed_dtype_moe_gemm_config()
    render_config = {
        "func_name": masked_moe_gemm_mixed_dtype_dispatch_name(
            out_dtype, a_quant_recipe, scale_a_major
        ),
        "element_a": dtype_torch2mutlass_map[torch.float8_e4m3fn],
        "element_scale_a": dtype_torch2mutlass_map[torch.float32],
        "element_scale_b": dtype_torch2mutlass_map[torch.bfloat16],
        "element_d": dtype_torch2mutlass_map[out_dtype],
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "stages": stages,
        "scale_a_block_k": a_quant_recipe[1],
        "scale_a_major": scale_a_major,
    }
    return (
        TVM_HEADER
        + Environment(
            loader=FileSystemLoader(
                (
                    gemm_utils.gemm_template_dir / "masked_moe_gemm_mixed_dtype"
                ).as_posix()
            ),
            keep_trailing_newline=True,
        )
        .get_template("masked_moe_gemm_mixed_dtype_kern.j2")
        .render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


def gen_masked_moe_gemm_mixed_dtype_spec(
    out_dtype: torch.dtype,
    a_quant_recipe: tuple[int, int],
    scale_a_major: str = "K",
) -> JitSpec:
    dispatch_name = masked_moe_gemm_mixed_dtype_dispatch_name(
        out_dtype, a_quant_recipe, scale_a_major
    )
    source_path = (
        jit_env.MATE_GEN_SRC_DIR / "masked_moe_gemm_mixed_dtype" / f"{dispatch_name}.mu"
    )
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cflags=MIXED_DTYPE_GEMM_CXX_FLAGS,
        extra_cuda_cflags=MIXED_DTYPE_GEMM_MCC_FLAGS,
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR,
            jit_env.MATE_CSRC_DIR,
            jit_env.MUTLASS_INCLUDE_DIR,
        ],
        generated_sources={
            source_path: render_masked_moe_gemm_mixed_dtype_source(
                out_dtype, a_quant_recipe, scale_a_major
            )
        },
    )


def gen_masked_moe_gemm_mixed_dtype_aot() -> list[JitSpec]:
    return [
        gen_masked_moe_gemm_mixed_dtype_spec(out_dtype, a_quant_recipe, scale_a_major)
        for out_dtype in MIXED_DTYPE_GEMM_OUTPUT_DTYPE_TAGS
        for a_quant_recipe in MIXED_DTYPE_GEMM_A_QUANT_RECIPE_TAGS
        for scale_a_major in (("K", "M") if a_quant_recipe == (1, 128) else ("K",))
    ]


@functools.cache
def _load_masked_moe_gemm_mixed_dtype_module(
    out_dtype: torch.dtype,
    a_quant_recipe: tuple[int, int],
    scale_a_major: str = "K",
):
    return gen_masked_moe_gemm_mixed_dtype_spec(
        out_dtype, a_quant_recipe, scale_a_major
    ).build_and_load()


def get_masked_moe_gemm_mixed_dtype_module(
    out_dtype: torch.dtype,
    a_quant_recipe: tuple[int, int],
    scale_a_major: str = "K",
):
    return (
        masked_moe_gemm_mixed_dtype_dispatch_name(
            out_dtype, a_quant_recipe, scale_a_major
        ),
        _load_masked_moe_gemm_mixed_dtype_module(
            out_dtype, a_quant_recipe, scale_a_major
        ),
    )


def masked_moe_gemm_mixed_dtype_mutlass(
    input_a: tuple[torch.Tensor, torch.Tensor],
    input_b: tuple[torch.Tensor, torch.Tensor],
    masked_tokens_info: torch.Tensor,
    out: torch.Tensor,
    expect_tokens: int,
    a_quant_recipe: tuple[int, int],
    num_mps: int | None = None,
) -> torch.Tensor:
    a, scale_a = input_a
    b, scale_b = input_b
    if scale_b.shape[-1] % 2:
        padded_scale_b = torch.zeros(
            (*scale_b.shape[:-1], scale_b.shape[-1] + 1),
            device=scale_b.device,
            dtype=scale_b.dtype,
        )
        padded_scale_b[..., :-1].copy_(scale_b)
        scale_b = padded_scale_b
    scale_a_major = (
        "M" if a_quant_recipe == (1, 128) and scale_a.stride(1) == 1 else "K"
    )
    dispatch_name, module = get_masked_moe_gemm_mixed_dtype_module(
        out.dtype, a_quant_recipe, scale_a_major
    )
    module.get_function(dispatch_name)(
        a,
        scale_a,
        b,
        scale_b,
        out,
        masked_tokens_info,
        int(expect_tokens),
        resolve_num_mps(a.device, num_mps),
    )
    return out
