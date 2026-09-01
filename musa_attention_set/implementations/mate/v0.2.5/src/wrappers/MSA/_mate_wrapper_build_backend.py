"""PEP 517 backend for MATE compatibility wrapper packages."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from setuptools import build_meta as _orig

_WRAPPER_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _WRAPPER_ROOT.parents[1]
_VERSION_METADATA = _WRAPPER_ROOT / "_mate_wrapper_version.txt"
_REQUIREMENTS_METADATA = _WRAPPER_ROOT / "_mate_wrapper_requirements.txt"


def _read_project_name() -> str:
    text = (_WRAPPER_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_match = re.search(r"(?ms)^\[project\]\s*(.*?)(?:^\[|\Z)", text)
    if not project_match:
        raise RuntimeError("pyproject.toml is missing a [project] section")
    name_match = re.search(
        r"(?m)^name\s*=\s*['\"]([^'\"]+)['\"]", project_match.group(1)
    )
    if not name_match:
        raise RuntimeError("pyproject.toml is missing project.name")
    return name_match.group(1)


def _package_name() -> str:
    return _read_project_name().replace("-", "_")


def _with_dev_suffix(version: str) -> str:
    dev_suffix = os.environ.get("MATE_DEV_RELEASE_SUFFIX", "").strip()
    public_part, separator, local_part = version.partition("+")
    if dev_suffix and ".dev" not in public_part:
        public_part = f"{public_part}.dev{dev_suffix}"
    return f"{public_part}{separator}{local_part}" if separator else public_part


def _without_musa_local_version(version: str) -> str:
    public_part, separator, local_part = version.partition("+")
    if not separator:
        return version
    local_parts = local_part.split(".")
    if local_parts[-1] != "musa":
        return version
    remaining_local = ".".join(local_parts[:-1])
    return f"{public_part}+{remaining_local}" if remaining_local else public_part


def _with_musa_local_version(version: str) -> str:
    public_part, separator, local_part = version.partition("+")
    if not separator:
        return f"{public_part}+musa"
    if local_part.split(".")[-1] == "musa":
        return version
    return f"{public_part}+{local_part}.musa"


def _read_mate_version() -> str:
    repo_version_file = _REPO_ROOT / "version.txt"
    if repo_version_file.exists():
        return _with_dev_suffix(repo_version_file.read_text(encoding="utf-8").strip())
    if _REQUIREMENTS_METADATA.exists():
        requirement = _REQUIREMENTS_METADATA.read_text(encoding="utf-8").strip()
        match = re.fullmatch(r"mate\s*==\s*(\S+)", requirement)
        if match:
            return match.group(1)
    if _VERSION_METADATA.exists():
        return _without_musa_local_version(
            _VERSION_METADATA.read_text(encoding="utf-8").strip()
        )
    raise RuntimeError("Unable to resolve MATE wrapper version")


def _parse_existing_git_version(build_meta_file: Path) -> str | None:
    if not build_meta_file.exists():
        return None
    match = re.search(
        r"(?m)^__git_version__\s*=\s*['\"]([^'\"]+)['\"]",
        build_meta_file.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def _read_git_version(build_meta_file: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return _parse_existing_git_version(build_meta_file) or "unknown"


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _prepare_wrapper_metadata() -> None:
    mate_version = _read_mate_version()
    wrapper_version = _with_musa_local_version(mate_version)
    package_dir = _WRAPPER_ROOT / _package_name()
    if not package_dir.exists():
        raise RuntimeError(f"Wrapper package directory not found: {package_dir}")

    build_meta_file = package_dir / "_build_meta.py"
    git_version = _read_git_version(build_meta_file)

    _write_if_changed(_VERSION_METADATA, f"{wrapper_version}\n")
    _write_if_changed(_REQUIREMENTS_METADATA, f"mate=={mate_version}\n")
    _write_if_changed(
        build_meta_file,
        '"""Build metadata for MATE wrapper package."""\n'
        f'__version__ = "{wrapper_version}"\n'
        f'__git_version__ = "{git_version}"\n',
    )


def get_requires_for_build_wheel(config_settings=None):
    _prepare_wrapper_metadata()
    return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    _prepare_wrapper_metadata()
    return _orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    _prepare_wrapper_metadata()
    return _orig.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _prepare_wrapper_metadata()
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _prepare_wrapper_metadata()
    return _orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_wrapper_metadata()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory):
    _prepare_wrapper_metadata()
    return _orig.build_sdist(sdist_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_wrapper_metadata()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)
