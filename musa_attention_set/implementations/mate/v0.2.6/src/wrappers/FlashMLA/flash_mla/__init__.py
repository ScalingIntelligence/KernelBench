from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from flash_mla.flash_mla_interface import (
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)


try:
    from ._build_meta import __git_version__ as __git_version__
except Exception:
    __git_version__ = "unknown"


def _load_version() -> str:
    try:
        return version("flash_mla")
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
    "FlashMLASchedMeta",
    "__git_version__",
    "__version__",
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_mla_sparse_fwd",
]
