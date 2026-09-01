import os
import re
import shutil
import sys
from pathlib import Path

from setuptools import build_meta as orig

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_utils import build_version_string
from mate.artifacts import (
    ArtifactPath,
    KernelMapHash,
    download_artifacts,
    parse_kernel_map,
    resolve_mubin_artifact_path,
    resolve_mubin_module_dir,
    verify_mubin_module_artifacts,
)

_root = Path(__file__).parent.resolve()
_repo_root = _root.parent
_package_dir = _root / "mate_mubin"
_mubin_dir = _package_dir / "mubin"


def _remove_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _read_existing_build_version(build_meta_file: Path) -> str | None:
    if not build_meta_file.exists():
        return None
    match = re.search(
        r"(?m)^__version__\s*=\s*['\"]([^'\"]+)['\"]",
        build_meta_file.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def _create_build_metadata() -> str:
    version_file = _repo_root / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            base_version = f.read().strip()
    else:
        base_version = "0.0.0"

    version, git_version = build_version_string(
        base_version=base_version,
        cwd=_repo_root,
        dev_suffix=os.environ.get("MATE_DEV_RELEASE_SUFFIX", ""),
        local_version=os.environ.get("MATE_LOCAL_VERSION"),
    )

    build_meta_file = _package_dir / "_build_meta.py"
    in_git_repo = (_repo_root / ".git").exists()
    if build_meta_file.exists() and not in_git_repo:
        existing_version = _read_existing_build_version(build_meta_file)
        if existing_version is None:
            raise RuntimeError(f"Invalid MATE MUBIN build metadata: {build_meta_file}")
        print("Build metadata file already exists (not in git repo), keeping it")
        return existing_version

    _write_if_changed(
        build_meta_file,
        '"""Build metadata for mate-mubin package."""\n'
        f'__version__ = "{version}"\n'
        f'__git_version__ = "{git_version}"\n',
    )

    print(f"Created build metadata file with version {version}")
    return version


def _source_mubin_dir() -> Path | None:
    source_dir = os.environ.get("MATE_MUBIN_SOURCE_DIR")
    if source_dir:
        return Path(source_dir).expanduser().resolve()
    return None


def _copy_payload(source_dir: Path, destination_dir: Path) -> None:
    kernel_map_path = source_dir / KernelMapHash.KERNEL_MAP_FILE
    if not kernel_map_path.is_file():
        raise RuntimeError(f"Missing MATE MUBIN kernel map: {kernel_map_path}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(kernel_map_path, destination_dir / KernelMapHash.KERNEL_MAP_FILE)
    for entry in parse_kernel_map(kernel_map_path):
        relative_path = Path(resolve_mubin_artifact_path(entry.file_name))
        destination_path = destination_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / relative_path, destination_path)


def _copy_source_mubin_dir(source_dir: Path) -> None:
    for _, artifact_path in ArtifactPath.iter_module_paths():
        artifact_rel = str(artifact_path).strip("/")
        _copy_payload(source_dir / artifact_rel, _mubin_dir / artifact_rel)


def _verify_staged_mubin_sources() -> None:
    for module, _ in ArtifactPath.iter_module_paths():
        module_dir = resolve_mubin_module_dir(_mubin_dir, module)
        if not verify_mubin_module_artifacts(module, module_dir):
            raise RuntimeError(
                f"Staged MUBIN artifacts for {module} are missing or failed "
                f"verification: {module_dir}"
            )


def _remove_staging_lock_files() -> None:
    for lock_path in _mubin_dir.rglob("*.lock"):
        lock_path.unlink()


def _stage_mubin_sources() -> None:
    source_dir = _source_mubin_dir()
    if source_dir:
        if not source_dir.is_dir():
            raise RuntimeError(f"MUBIN source directory does not exist: {source_dir}")
        _remove_path(_mubin_dir)
        _copy_source_mubin_dir(source_dir)
        source_text = str(source_dir)
    else:
        _remove_path(_mubin_dir)
        original_mubin_dir = os.environ.get("MATE_MUBIN_DIR")
        os.environ["MATE_MUBIN_DIR"] = str(_mubin_dir)
        try:
            download_artifacts()
        finally:
            if original_mubin_dir is None:
                os.environ.pop("MATE_MUBIN_DIR", None)
            else:
                os.environ["MATE_MUBIN_DIR"] = original_mubin_dir
        source_text = "remote artifacts"

    _remove_staging_lock_files()
    payload_files = [path for path in _mubin_dir.rglob("*") if path.is_file()]
    if not payload_files:
        raise RuntimeError(f"No MUBIN files staged from {source_text}")
    _verify_staged_mubin_sources()
    print(f"Staged {len(payload_files)} MUBIN files from {source_text}")


_create_build_metadata()


def _prepare_build() -> None:
    _stage_mubin_sources()


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_build()
    return orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_build()
    return orig.build_editable(wheel_directory, config_settings, metadata_directory)
