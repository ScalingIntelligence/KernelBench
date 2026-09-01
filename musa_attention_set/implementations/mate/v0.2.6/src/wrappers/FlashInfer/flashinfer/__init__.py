from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import decode as decode
from . import gemm as gemm
from . import rope as rope
from .gemm import bmm_bf16 as bmm_bf16
from .gemm import bmm_fp8 as bmm_fp8

try:
    from ._build_meta import __git_version__ as __git_version__
except Exception:
    __git_version__ = "unknown"


def _load_version() -> str:
    try:
        return version("flashinfer-python")
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
    "__git_version__",
    "__version__",
    "bmm_bf16",
    "bmm_fp8",
    "decode",
    "gemm",
    "rope",
]
