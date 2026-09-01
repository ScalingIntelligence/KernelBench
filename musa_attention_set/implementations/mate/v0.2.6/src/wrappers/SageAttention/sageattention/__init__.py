"""
SageAttention compatibility package for MATE.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sageattention.interface import sageattn, sageattn_qk_int8_pv_fp8_cuda_sm90


try:
    from ._build_meta import __git_version__ as __git_version__
except Exception:
    __git_version__ = "unknown"


def _load_version() -> str:
    try:
        return version("sageattention")
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


__version__ = _load_version()


__all__ = [
    "__version__",
    "__git_version__",
    "sageattn",
    "sageattn_qk_int8_pv_fp8_cuda_sm90",
]
