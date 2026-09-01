import os
from pathlib import Path


MUBIN_DIR = Path(__file__).parent / "mubin"


def get_mubin_dir() -> str:
    """Return the directory containing packaged MUBIN artifacts."""
    return str(MUBIN_DIR)


def list_mubin_files() -> list[str]:
    """List packaged MUBIN object files relative to MUBIN_DIR."""
    if not MUBIN_DIR.exists():
        return []

    mubin_files: list[str] = []
    for root, _, files in os.walk(MUBIN_DIR):
        for file_name in files:
            if not file_name.endswith(".o"):
                continue
            rel_path = os.path.relpath(os.path.join(root, file_name), MUBIN_DIR)
            mubin_files.append(rel_path)
    return sorted(mubin_files)


def get_mubin_path(relative_path: str) -> str:
    """Return the absolute path to a packaged MUBIN artifact."""
    return str(MUBIN_DIR / relative_path)


def _get_version() -> str:
    try:
        from . import _build_meta

        return _build_meta.__version__
    except (ImportError, AttributeError):
        pass

    version_file = Path(__file__).parent.parent.parent / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            return f.read().strip()
    return "0.0.0"


def _get_git_version() -> str:
    try:
        from . import _build_meta

        return _build_meta.__git_version__
    except (ImportError, AttributeError):
        pass

    return "unknown"


__version__ = _get_version()
__git_version__ = _get_git_version()
__all__ = [
    "MUBIN_DIR",
    "get_mubin_dir",
    "get_mubin_path",
    "list_mubin_files",
]
