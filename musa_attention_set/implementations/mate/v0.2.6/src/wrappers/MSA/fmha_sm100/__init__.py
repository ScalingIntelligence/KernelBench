from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fmha_sm100.api import fmha_sm100, fmha_sm100_plan, sparse_topk_select

try:
    from ._build_meta import __git_version__ as __git_version__
except Exception:
    __git_version__ = "unknown"

_SPARSE_LAZY_EXPORTS = frozenset(
    {
        "SparseDecodePagedAttentionWrapper",
        "SparseK2qCsrBuilderSm100",
        "fp4_indexer_block_scores",
        "sparse_atten_nvfp4_kv_func",
        "sparse_decode_atten_func",
        "sparse_fmha",
        "sparse_fmha_plan",
    }
)


def _load_version() -> str:
    try:
        return version("fmha_sm100")
    except PackageNotFoundError:
        pass

    try:
        from ._build_meta import __version__ as build_version

        return build_version
    except Exception:
        pass

    try:
        return (
            (Path(__file__).resolve().parents[3] / "version.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def __getattr__(name: str):
    if name in _SPARSE_LAZY_EXPORTS:
        from . import sparse as _sparse

        return getattr(_sparse, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_SPARSE_LAZY_EXPORTS})


__version__ = _load_version()

__all__ = [
    "__git_version__",
    "__version__",
    "fmha_sm100",
    "fmha_sm100_plan",
    "sparse_topk_select",
    *sorted(_SPARSE_LAZY_EXPORTS),
]
