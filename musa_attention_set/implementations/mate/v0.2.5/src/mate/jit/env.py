import importlib.util
import os
import pathlib

from packaging.version import InvalidVersion, Version

_package_root: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
_repo_root: pathlib.Path = _package_root.parent


def _load_build_meta_version() -> str | None:
    build_meta_path = _package_root / "_build_meta.py"
    if not build_meta_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "mate._build_meta_runtime", build_meta_path
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version = getattr(module, "__version__", None)
    return version if isinstance(version, str) and version else None


def _load_installed_package_version() -> str | None:
    try:
        from importlib.metadata import version as package_version

        resolved = package_version("mate")
        return resolved if resolved else None
    except Exception:
        return None


def _load_repo_version() -> str | None:
    version_file = _repo_root / "version.txt"
    if not version_file.exists():
        return None
    version = version_file.read_text().strip()
    return version if version else None


def _prefer_repo_tree() -> bool:
    return (_repo_root / ".git").exists()


def resolve_runtime_version_string() -> str:
    candidates = (
        (
            _load_repo_version(),
            _load_build_meta_version(),
            _load_installed_package_version(),
        )
        if _prefer_repo_tree()
        else (
            _load_build_meta_version(),
            _load_installed_package_version(),
            _load_repo_version(),
        )
    )

    for candidate in candidates:
        if not candidate:
            continue
        try:
            Version(candidate)
        except InvalidVersion:
            continue
        return candidate
    return "0.0.0"


def _resolve_mate_base_version(version_string: str | None = None) -> str:
    return Version(version_string or resolve_runtime_version_string()).base_version


def _parse_arch_token(arch: str) -> tuple[int, int]:
    parts = arch.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Invalid MUSA architecture: {arch}")
    major, minor = (int(part) for part in parts)
    return major, minor


def _normalize_arch_pairs(arch_pairs: list[tuple[int, int]]) -> tuple[str, ...]:
    unique_arch_pairs = sorted(set(arch_pairs))
    if not unique_arch_pairs:
        raise ValueError("No MUSA architectures resolved for JIT cache namespace")
    return tuple(f"mp{major}{minor}" for major, minor in unique_arch_pairs)


def _normalize_musa_arch_list(arch_list: str) -> tuple[str, ...]:
    arch_pairs = [_parse_arch_token(arch) for arch in arch_list.split()]
    return _normalize_arch_pairs(arch_pairs)


def _detect_visible_musa_arch_list() -> tuple[str, ...]:
    try:
        import torch_musa
    except Exception as exc:
        raise RuntimeError(
            "Unable to auto-detect visible MUSA architectures because torch_musa "
            "is not available. Set MATE_MUSA_ARCH_LIST explicitly, for example "
            "'MATE_MUSA_ARCH_LIST=3.1'."
        ) from exc

    if not torch_musa.is_available():
        raise RuntimeError(
            "Unable to auto-detect visible MUSA architectures because no MUSA "
            "device is available. Set MATE_MUSA_ARCH_LIST explicitly, for example "
            "'MATE_MUSA_ARCH_LIST=3.1'."
        )

    arch_pairs: list[tuple[int, int]] = []
    for device_idx in range(torch_musa.device_count()):
        props = torch_musa.get_device_properties(device_idx)
        arch_pairs.append((int(props.major), int(props.minor)))
    return _normalize_arch_pairs(arch_pairs)


def resolve_musa_arch_list() -> tuple[str, ...]:
    arch_list = os.environ.get("MATE_MUSA_ARCH_LIST")
    if arch_list is not None:
        stripped = arch_list.strip()
        if not stripped:
            raise ValueError("MATE_MUSA_ARCH_LIST is set but empty.")
        return _normalize_musa_arch_list(stripped)
    return _detect_visible_musa_arch_list()


def resolve_musa_target_flags() -> tuple[str, ...]:
    return tuple(f"--offload-arch=mp_{arch[2:]}" for arch in resolve_musa_arch_list())


def resolve_musa_arch_key() -> str:
    return "_".join(resolve_musa_arch_list())


MATE_VERSION = Version(resolve_runtime_version_string())
MATE_BASE_VERSION = MATE_VERSION.base_version
MATE_MUSA_ARCH_LIST = resolve_musa_arch_list()
MATE_MUSA_TARGET_FLAGS = tuple(
    f"--offload-arch=mp_{arch[2:]}" for arch in MATE_MUSA_ARCH_LIST
)
MATE_MUSA_ARCH_KEY = "_".join(MATE_MUSA_ARCH_LIST)

MATE_BASE_DIR: pathlib.Path = pathlib.Path(
    os.getenv("MATE_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)

MATE_CACHE_DIR: pathlib.Path = MATE_BASE_DIR / ".cache" / "mate"


def _resolve_repo_or_packaged(
    repo_relative: str, packaged_relative: str
) -> pathlib.Path:
    repo_path = _repo_root / repo_relative
    packaged_path = _package_root / "data" / packaged_relative
    if _prefer_repo_tree() and repo_path.exists():
        return repo_path
    if packaged_path.exists():
        return packaged_path
    return repo_path


def resolve_csrc_dir() -> pathlib.Path:
    return _resolve_repo_or_packaged("csrc", "csrc")


MATE_WORKSPACE_DIR: pathlib.Path = (
    MATE_CACHE_DIR / MATE_BASE_VERSION / MATE_MUSA_ARCH_KEY
)
MATE_GEN_SRC_DIR: pathlib.Path = MATE_WORKSPACE_DIR / "generated"
MATE_JIT_DIR: pathlib.Path = MATE_WORKSPACE_DIR / "cached_ops"
MATE_DATA: pathlib.Path = _package_root / "data"
MATE_CSRC_DIR: pathlib.Path = resolve_csrc_dir()
MATE_TEMPLATE_DIR: pathlib.Path = MATE_CSRC_DIR / "templates"
MATE_INCLUDE_DIR: pathlib.Path = _resolve_repo_or_packaged("include", "include")
MUTLASS_INCLUDE_DIR: pathlib.Path = _resolve_repo_or_packaged(
    "3rdparty/mutlass/include", "mutlass/include"
)
MUALG_INCLUDE_DIR: pathlib.Path = _resolve_repo_or_packaged("3rdparty/muAlg", "muAlg")
MUTHRUST_INCLUDE_DIR: pathlib.Path = _resolve_repo_or_packaged(
    "3rdparty/muThrust", "muThrust"
)
MATE_AOT_DIR: pathlib.Path = MATE_DATA / "aot"


def disable_jit_enabled() -> bool:
    return os.environ.get("MATE_DISABLE_JIT", "0") == "1"
