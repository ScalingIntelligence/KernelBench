from importlib import import_module


def _load_version() -> str:
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("mate")
    except Exception:
        try:
            from ._build_meta import __version__ as build_version

            return build_version
        except Exception:
            return "unknown"


__version__ = _load_version()

_LAZY_SUBMODULES = {
    "aot",
    "artifacts",
    "api_logging",
    "deep_gemm",
    "flashmla",
    "gdn_decode",
    "gdn_prefill",
    "gemm",
    "hyperconnection",
    "jit",
    "memory_debug",
    "kda",
    "mega_moe",
    "mha_interface",
    "msa_interface",
    "sage_attention_interface",
    "sparse_mla_interface",
    "testing",
    "utils",
    "version",
}

_LAZY_ATTR_MODULES = {
    "flash_attn_combine": "mha_interface",
    "flash_attn_varlen_func": "mha_interface",
    "flash_attn_with_kvcache": "mha_interface",
    "get_scheduler_metadata": "mha_interface",
    "get_mla_metadata": "flashmla",
    "flash_mla_with_kvcache": "flashmla",
    "gated_delta_rule_decode": "gdn_decode",
    "gdn_prefill": "gdn_prefill",
    "chunk_kda": "kda",
    "hash_topk": "hash_topk",
    "mate_api": "api_logging",
    "mhc_pre": "hyperconnection",
    "mhc_pre_big_fuse": "hyperconnection",
    "mhc_prenorm_gemm_sqrsum": "hyperconnection",
    "moe_fused_gate": "moe_fused_gate",
    "msa": "msa_interface",
    "msa_plan": "msa_interface",
    "MsaRuntimeMetadata": "msa_interface",
    "sparse_msa": "msa_interface",
    "sparse_msa_plan": "msa_interface",
    "sparse_topk_select": "msa_interface",
    "sage_attn_quantized": "sage_attention_interface",
    "sage_attn_quantized_with_kvcache": "sage_attention_interface",
}

__all__ = [
    "aot",
    "artifacts",
    "api_logging",
    "deep_gemm",
    "flash_attn_combine",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "flash_mla_with_kvcache",
    "flashmla",
    "gated_delta_rule_decode",
    "gdn_decode",
    "gdn_prefill",
    "gemm",
    "hash_topk",
    "get_mla_metadata",
    "get_scheduler_metadata",
    "hyperconnection",
    "jit",
    "memory_debug",
    "kda",
    "mega_moe",
    "chunk_kda",
    "mate_api",
    "mha_interface",
    "msa",
    "msa_interface",
    "mhc_pre",
    "mhc_pre_big_fuse",
    "mhc_prenorm_gemm_sqrsum",
    "moe_fused_gate",
    "msa_plan",
    "MsaRuntimeMetadata",
    "sage_attn_quantized",
    "sage_attn_quantized_with_kvcache",
    "sage_attention_interface",
    "sparse_msa",
    "sparse_msa_plan",
    "sparse_topk_select",
    "sparse_mla_interface",
    "testing",
    "utils",
    "version",
    "__version__",
]


def _load_relative_module(name):
    module = import_module(f".{name}", __name__)
    return module


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = _load_relative_module(name)
        globals()[name] = module
        return module
    if name in _LAZY_ATTR_MODULES:
        module = _load_relative_module(_LAZY_ATTR_MODULES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
