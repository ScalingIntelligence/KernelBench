import importlib

from . import env as env
from .core import JitSpec as JitSpec
from .core import JitSpecRegistry as JitSpecRegistry
from .core import JitSpecStatus as JitSpecStatus
from .core import MissingJITCacheError as MissingJITCacheError
from .core import build_jit_specs as build_jit_specs
from .core import clear_cache_dir as clear_cache_dir
from .core import copy_built_kernels as copy_built_kernels
from .core import default_aot_path as default_aot_path
from .core import gen_jit_spec as gen_jit_spec
from .core import jit_spec_registry as jit_spec_registry
from .core import temporary_max_jobs as temporary_max_jobs
from .flash_attention_ops import (
    gen_flash_attention_ops_aot as gen_flash_attention_ops_aot,
)
from .flash_attention_ops import (
    gen_flash_attention_ops_spec as gen_flash_attention_ops_spec,
)
from .flash_attention_ops import (
    get_flash_attention_ops_module as get_flash_attention_ops_module,
)
from .gemm.deep_gemm import gen_deep_gemm_gemm_aot as gen_deep_gemm_gemm_aot
from .gemm.deep_gemm import gen_deep_gemm_gemm_spec as gen_deep_gemm_gemm_spec
from .gemm.deep_gemm import gen_deep_gemm_gemm_specs as gen_deep_gemm_gemm_specs
from .gemm.deep_gemm import (
    get_deep_gemm_gemm_aot_configs as get_deep_gemm_gemm_aot_configs,
)
from .gemm.deep_gemm import (
    get_deep_gemm_gemm_module as get_deep_gemm_gemm_module,
)
from .gemm.deep_gemm.hyperconnection import (
    gen_hyperconnection_aot as gen_hyperconnection_aot,
)
from .gemm.deep_gemm.hyperconnection import (
    gen_hyperconnection_spec as gen_hyperconnection_spec,
)
from .gemm.deep_gemm.hyperconnection import (
    get_hyperconnection_module as get_hyperconnection_module,
)
from .gemm.deep_gemm.mqa_logits import gen_mqa_logits_aot as gen_mqa_logits_aot
from .gemm.deep_gemm.mqa_logits import gen_mqa_logits_spec as gen_mqa_logits_spec
from .gemm.deep_gemm.mqa_logits import (
    get_mqa_logits_module as get_mqa_logits_module,
)
from .gemm.deep_gemm.paged_mqa_logits import (
    gen_paged_mqa_logits_aot as gen_paged_mqa_logits_aot,
)
from .gemm.deep_gemm.paged_mqa_logits import (
    gen_paged_mqa_logits_spec as gen_paged_mqa_logits_spec,
)
from .gemm.deep_gemm.paged_mqa_logits import (
    get_paged_mqa_logits_module as get_paged_mqa_logits_module,
)
from .gemm.masked_moe_gemm_mixed_dtype import (
    gen_masked_moe_gemm_mixed_dtype_aot as gen_masked_moe_gemm_mixed_dtype_aot,
)
from .gemm_ops import gen_gemm_ops_aot as gen_gemm_ops_aot
from .gemm_ops import gen_gemm_ops_spec as gen_gemm_ops_spec
from .gemm_ops import get_gemm_ops_module as get_gemm_ops_module
from .guard_allocator import gen_guard_allocator_aot as gen_guard_allocator_aot
from .guard_allocator import gen_guard_allocator_spec as gen_guard_allocator_spec
from .kda_ops import (
    gen_kda_fused_ops_spec as gen_kda_fused_ops_spec,
)
from .kda_ops import (
    get_kda_fused_ops_module as get_kda_fused_ops_module,
)
from .mla_ops import gen_mla_ops_aot as gen_mla_ops_aot
from .mla_ops import gen_mla_ops_spec as gen_mla_ops_spec
from .mla_ops import get_mla_ops_module as get_mla_ops_module
from .mega_moe import (
    MegaMoEKernelConfig as MegaMoEKernelConfig,
    gen_fp8_fp8_mega_moe_stage1_spec as gen_fp8_fp8_mega_moe_stage1_spec,
    gen_fp8_fp8_mega_moe_stage2_spec as gen_fp8_fp8_mega_moe_stage2_spec,
    gen_mega_moe_runtime_utils_spec as gen_mega_moe_runtime_utils_spec,
    gen_mega_moe_specs as gen_mega_moe_specs,
    get_fp8_fp8_mega_moe_stage1_module as get_fp8_fp8_mega_moe_stage1_module,
    get_fp8_fp8_mega_moe_stage2_module as get_fp8_fp8_mega_moe_stage2_module,
    get_mega_moe_runtime_utils_module as get_mega_moe_runtime_utils_module,
)
from .msa_ops import gen_msa_ops_aot as gen_msa_ops_aot
from .msa_ops import gen_msa_ops_spec as gen_msa_ops_spec
from .msa_ops import get_msa_ops_module as get_msa_ops_module
from .moe_fused_gate import gen_moe_fused_gate_aot as gen_moe_fused_gate_aot
from .moe_fused_gate import gen_moe_fused_gate_spec as gen_moe_fused_gate_spec
from .moe_fused_gate import get_moe_fused_gate_module as get_moe_fused_gate_module

__all__ = [
    "env",
    "attention",
    "JitSpec",
    "JitSpecRegistry",
    "JitSpecStatus",
    "MissingJITCacheError",
    "build_jit_specs",
    "clear_cache_dir",
    "copy_built_kernels",
    "default_aot_path",
    "gen_deep_gemm_gemm_aot",
    "gen_deep_gemm_gemm_spec",
    "gen_deep_gemm_gemm_specs",
    "get_deep_gemm_gemm_aot_configs",
    "get_deep_gemm_gemm_module",
    "gen_flash_attention_ops_aot",
    "gen_flash_attention_ops_spec",
    "get_flash_attention_ops_module",
    "gen_gemm_ops_aot",
    "gen_gemm_ops_spec",
    "gen_guard_allocator_aot",
    "gen_guard_allocator_spec",
    "get_gemm_ops_module",
    "gen_kda_fused_ops_spec",
    "get_kda_fused_ops_module",
    "gen_hyperconnection_aot",
    "gen_hyperconnection_spec",
    "get_hyperconnection_module",
    "gen_mqa_logits_aot",
    "gen_mqa_logits_spec",
    "get_mqa_logits_module",
    "gen_masked_moe_gemm_mixed_dtype_aot",
    "gen_paged_mqa_logits_aot",
    "gen_paged_mqa_logits_spec",
    "get_paged_mqa_logits_module",
    "gen_mla_ops_aot",
    "gen_mla_ops_spec",
    "get_mla_ops_module",
    "MegaMoEKernelConfig",
    "gen_fp8_fp8_mega_moe_stage1_spec",
    "gen_fp8_fp8_mega_moe_stage2_spec",
    "gen_mega_moe_runtime_utils_spec",
    "gen_mega_moe_specs",
    "get_fp8_fp8_mega_moe_stage1_module",
    "get_fp8_fp8_mega_moe_stage2_module",
    "get_mega_moe_runtime_utils_module",
    "gen_msa_ops_aot",
    "gen_msa_ops_spec",
    "get_msa_ops_module",
    "gen_jit_spec",
    "jit_spec_registry",
    "gen_moe_fused_gate_aot",
    "gen_moe_fused_gate_spec",
    "get_moe_fused_gate_module",
    "prewarm_modules",
    "temporary_max_jobs",
]


def __getattr__(name):
    if name == "attention":
        return importlib.import_module(".attention", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def prewarm_modules(
    *,
    fmha_metadata_configs=None,
    fmha_combine_configs=None,
    fmha_fwd_configs=None,
    include_flash_attention: bool = False,
    include_gemm: bool = False,
    include_deep_gemm: bool = False,
    include_hyperconnection: bool = False,
    include_mqa_logits: bool = False,
    include_paged_mqa_logits: bool = False,
    include_kda: bool = False,
    include_mla: bool = False,
    include_msa: bool = False,
    include_moe_fused_gate: bool = False,
    include_sage_attention: bool = False,
    jobs=None,
    verbose: bool = False,
    skip_prebuilt: bool = True,
):
    specs = []
    from .attention.fmha import (
        gen_fmha_fwd_combine_specs,
        gen_fmha_fwd_specs,
        gen_fmha_metadata_specs,
    )

    if fmha_metadata_configs:
        specs.extend(gen_fmha_metadata_specs(fmha_metadata_configs))
    if fmha_combine_configs:
        specs.extend(gen_fmha_fwd_combine_specs(fmha_combine_configs))
    if fmha_fwd_configs:
        specs.extend(gen_fmha_fwd_specs(fmha_fwd_configs))
    if include_flash_attention:
        from .flash_attention_ops import gen_flash_attention_ops_spec

        specs.append(gen_flash_attention_ops_spec())
    if include_gemm:
        from .gemm_ops import gen_gemm_ops_spec

        specs.append(gen_gemm_ops_spec())
    if include_deep_gemm:
        from .gemm.deep_gemm import gen_deep_gemm_gemm_aot
        from .gemm.masked_moe_gemm_mixed_dtype import (
            gen_masked_moe_gemm_mixed_dtype_aot,
        )

        specs.extend(gen_deep_gemm_gemm_aot())
        specs.extend(gen_masked_moe_gemm_mixed_dtype_aot())
    if include_hyperconnection:
        from .gemm.deep_gemm.hyperconnection import gen_hyperconnection_aot

        specs.extend(gen_hyperconnection_aot())
    if include_mqa_logits:
        from .gemm.deep_gemm.mqa_logits import gen_mqa_logits_aot

        specs.extend(gen_mqa_logits_aot())
    if include_paged_mqa_logits:
        from .gemm.deep_gemm.paged_mqa_logits import gen_paged_mqa_logits_aot

        specs.extend(gen_paged_mqa_logits_aot())
    if include_kda:
        from .kda_ops import gen_kda_fused_ops_spec

        specs.append(gen_kda_fused_ops_spec())
    if include_mla:
        from .mla_ops import gen_mla_ops_spec

        specs.append(gen_mla_ops_spec())
    if include_msa:
        from .msa_ops import gen_msa_ops_aot

        specs.extend(gen_msa_ops_aot())
    if include_moe_fused_gate:
        from .moe_fused_gate import gen_moe_fused_gate_spec

        specs.append(gen_moe_fused_gate_spec())
    build_jit_specs(
        specs,
        jobs=jobs,
        verbose=verbose,
        skip_prebuilt=skip_prebuilt,
    )
    return specs
