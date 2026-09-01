import functools

import mate
from jinja2 import StrictUndefined, Template
from mate.jit.core import gen_jit_spec


def render_mubin_launcher(family: str, **context) -> str:
    template_path = (
        mate.jit.env.MATE_TEMPLATE_DIR
        / "mubin"
        / family
        / "launch_kernel_template.jinja"
    )
    template = Template(
        template_path.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )
    return template.render(**context)


@functools.cache
def render_gemm_mubin_launcher(asm_id, kernel_name: str) -> str:
    from .gemm.types import MoeGemmMode, TensorMajor, TensorQuantMode

    group_mode = {
        MoeGemmMode.NO_GROUP: "MoeGemmMode::NoGroup",
        MoeGemmMode.RAGGED_EXPERT_LAYOUT: "MoeGemmMode::RaggedPSumLayout",
        MoeGemmMode.RESERVE1: "MoeGemmMode::Reserve1",
        MoeGemmMode.RAGGED: "MoeGemmMode::Ragged",
        MoeGemmMode.MASKED: "MoeGemmMode::Masked",
        MoeGemmMode.BYTE_ML_RAGGED: "MoeGemmMode::ByteML_Ragged",
        MoeGemmMode.K_CONTIG: "MoeGemmMode::KContig",
    }[asm_id.group_mode]
    major = {
        TensorMajor.MN: "TensorMajor::MN",
        TensorMajor.K: "TensorMajor::K",
    }
    quant_mode = {mode: f"TensorQuantMode::{mode.name}" for mode in TensorQuantMode}
    block = asm_id.kernel_block
    return render_mubin_launcher(
        "gemm",
        func_name=kernel_name,
        b_pack_bits=asm_id.b_pack_bits,
        num_thread=block.num_thread,
        tile_m=block.tile_m,
        tile_n=block.tile_n,
        tile_k=block.tile_k,
        blk_per_mp=block.blk_per_mp,
        num_squad_m=block.num_squad_m,
        num_squad_n=block.num_squad_n,
        macro_tile_x=block.macro_tile_x,
        switch_swizzle=bool(block.switch_swizzle),
        major_a=major[asm_id.major_a],
        major_b=major[asm_id.major_b],
        quant_mode_a=quant_mode[asm_id.quant_mode_a],
        quant_mode_b=quant_mode[asm_id.quant_mode_b],
        group_mode=group_mode,
    )


def gen_mubin_launcher_spec(family: str, kernel_name: str, source: str):
    jit_env = mate.jit.env
    dispatch_name = f"run_{kernel_name}"
    generated_path = jit_env.MATE_GEN_SRC_DIR / "mubin" / family / f"{dispatch_name}.mu"
    sources = [generated_path]
    if family == "flash_mla":
        sources.append(jit_env.MATE_CSRC_DIR / "attention_combine.mu")

    include_paths = [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
        jit_env.MUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include",
        jit_env.MUTLASS_INCLUDE_DIR.parent / "experimental" / "fmha",
    ]
    return gen_jit_spec(
        dispatch_name,
        sources,
        extra_include_paths=include_paths,
        generated_sources={generated_path: source},
    )


@functools.cache
def get_mubin_launch_function(family: str, kernel_name: str, source: str):
    module = gen_mubin_launcher_spec(family, kernel_name, source).build_and_load()
    return module.get_function(f"run_{kernel_name}")
