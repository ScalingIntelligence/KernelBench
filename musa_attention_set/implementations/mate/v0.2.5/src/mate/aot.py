import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import SupportsInt, cast

import tvm_ffi.cpp  # noqa: F401

from mate.jit import build_jit_specs, copy_built_kernels, env as jit_env
from mate.jit.attention.fmha import gen_fmha_aot
from mate.jit.gemm.deep_gemm.gemm import gen_deep_gemm_gemm_aot
from mate.jit.gemm.deep_gemm.hyperconnection import gen_hyperconnection_aot
from mate.jit.gemm.deep_gemm.mqa_logits import gen_mqa_logits_aot
from mate.jit.gemm.deep_gemm.paged_mqa_logits import gen_paged_mqa_logits_aot
from mate.jit.gemm_ops import gen_gemm_ops_aot
from mate.jit.guard_allocator import gen_guard_allocator_aot
from mate.jit.mla_ops import gen_mla_ops_aot
from mate.jit.mega_moe import gen_mega_moe_runtime_utils_spec
from mate.jit.moe_fused_gate import gen_moe_fused_gate_aot


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true"}:
        return True
    if lowered in {"0", "false"}:
        return False
    raise argparse.ArgumentTypeError("Boolean flag must be one of: true/false/1/0")


def validate_aot_level(value: str) -> int:
    ivalue = int(value)
    if ivalue not in [0, 1, 2]:
        raise argparse.ArgumentTypeError("AOT level must be one of: 0, 1, 2")
    return ivalue


@contextmanager
def override_jit_env(
    *,
    include_dir: Path,
    csrc_dir: Path,
    mutlass_include_dir: Path,
    gen_src_dir: Path,
    jit_dir: Path,
    aot_dir: Path,
):
    previous = {
        "MATE_INCLUDE_DIR": jit_env.MATE_INCLUDE_DIR,
        "MATE_CSRC_DIR": jit_env.MATE_CSRC_DIR,
        "MATE_TEMPLATE_DIR": jit_env.MATE_TEMPLATE_DIR,
        "MUTLASS_INCLUDE_DIR": jit_env.MUTLASS_INCLUDE_DIR,
        "MATE_GEN_SRC_DIR": jit_env.MATE_GEN_SRC_DIR,
        "MATE_JIT_DIR": jit_env.MATE_JIT_DIR,
        "MATE_AOT_DIR": jit_env.MATE_AOT_DIR,
    }
    jit_env.MATE_INCLUDE_DIR = include_dir
    jit_env.MATE_CSRC_DIR = csrc_dir
    jit_env.MATE_TEMPLATE_DIR = csrc_dir / "templates"
    jit_env.MUTLASS_INCLUDE_DIR = mutlass_include_dir
    jit_env.MATE_GEN_SRC_DIR = gen_src_dir
    jit_env.MATE_JIT_DIR = jit_dir
    jit_env.MATE_AOT_DIR = aot_dir
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(jit_env, name, value)


def get_default_config() -> dict[str, object]:
    return {
        "attention_aot_level": 1,
        "add_gemm": True,
        "add_moe": True,
    }


def _dedupe_specs(specs):
    deduped = {}
    for spec in specs:
        existing = deduped.get(spec.name)
        if existing is None:
            deduped[spec.name] = spec
        elif existing != spec:
            raise ValueError(f"JIT spec collision for name={spec.name}")
    return list(deduped.values())


def gen_all_modules(config: dict[str, object] | None = None):
    final_config = get_default_config()
    if config is not None:
        final_config.update(config)

    attention_aot_level = int(cast(SupportsInt, final_config["attention_aot_level"]))
    add_gemm = bool(final_config["add_gemm"])
    add_moe = bool(final_config["add_moe"])

    specs = []
    specs.extend(gen_guard_allocator_aot())
    if attention_aot_level > 0:
        specs.extend(gen_fmha_aot(attention_aot_level))
        specs.extend(gen_mla_ops_aot())
    if add_gemm:
        specs.extend(gen_gemm_ops_aot())
        specs.extend(gen_deep_gemm_gemm_aot())
        specs.extend(gen_mqa_logits_aot())
        specs.extend(gen_paged_mqa_logits_aot())
        specs.extend(gen_hyperconnection_aot())
    if add_moe:
        specs.extend(gen_moe_fused_gate_aot())
        specs.append(gen_mega_moe_runtime_utils_spec())

    return _dedupe_specs(specs)


def register_default_modules(config: dict[str, object] | None = None) -> int:
    return len(gen_all_modules(config))


def compile_and_package_aot(
    output_dir: Path,
    build_dir: Path,
    project_root: Path,
    *,
    attention_aot_level: int = 1,
    add_gemm: bool = True,
    add_moe: bool = True,
    jobs: int | None = None,
    dry_run: bool = False,
):
    with override_jit_env(
        include_dir=project_root / "include",
        csrc_dir=project_root / "csrc",
        mutlass_include_dir=project_root / "3rdparty" / "mutlass" / "include",
        gen_src_dir=build_dir / "generated",
        jit_dir=build_dir / "cached_ops",
        aot_dir=output_dir,
    ):
        jit_env.MATE_GEN_SRC_DIR.mkdir(parents=True, exist_ok=True)
        jit_env.MATE_JIT_DIR.mkdir(parents=True, exist_ok=True)

        specs = gen_all_modules(
            {
                "attention_aot_level": attention_aot_level,
                "add_gemm": add_gemm,
                "add_moe": add_moe,
            }
        )

        if dry_run:
            return specs

        build_jit_specs(specs, jobs=jobs, verbose=True, skip_prebuilt=False)
        copy_built_kernels(specs, output_dir)
        return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ahead-of-Time (AOT) build JIT-managed MATE modules"
    )
    parser.add_argument("--out-dir", type=Path, help="Output directory")
    parser.add_argument("--build-dir", type=Path, help="Build directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve the spec list without compiling",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Maximum parallel ninja jobs",
    )
    parser.add_argument(
        "--attention-aot-level",
        type=validate_aot_level,
        default=1,
        help="Attention AOT level for the attention family "
        "(FMHA, flash attention ops, and MLA). "
        "Use 0 to disable attention AOT entirely.",
    )
    parser.add_argument(
        "--add-gemm",
        type=parse_bool,
        default=True,
        help="Whether to include the GEMM family in the AOT build "
        "(gemm ops, deep_gemm_gemm, and deep_gemm attention).",
    )
    parser.add_argument(
        "--add-moe",
        type=parse_bool,
        default=True,
        help="Whether to include the MOE family in the AOT build.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.out_dir or (project_root / "aot-ops")
    build_dir = args.build_dir or (project_root / "build" / "aot")

    specs = compile_and_package_aot(
        output_dir=output_dir,
        build_dir=build_dir,
        project_root=project_root,
        attention_aot_level=args.attention_aot_level,
        add_gemm=args.add_gemm,
        add_moe=args.add_moe,
        jobs=args.jobs,
        dry_run=args.dry_run,
    )
    print(f"Resolved {len(specs)} JIT specs")
    if not args.dry_run:
        print(f"AOT kernels saved to: {output_dir}")


if __name__ == "__main__":
    main()
