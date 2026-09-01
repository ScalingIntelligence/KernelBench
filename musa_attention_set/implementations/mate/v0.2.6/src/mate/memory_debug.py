from __future__ import annotations

import ctypes
import dataclasses
import importlib.machinery
import importlib.metadata
import importlib.util
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

_GUARD_ALLOCATOR_NAME = "guard_allocator"
_AUTO_INSTALL_ENV = "MATE_GUARD_ALLOCATOR_AUTO_INSTALL"
_ACTIVE_ENV = "MATE_GUARD_ALLOCATOR_ACTIVE"
_MODE_ENV = "MATE_GUARD_ALLOCATOR_MODE"
_SYNC_ON_FREE_ENV = "MATE_GUARD_ALLOCATOR_SYNC_ON_FREE"
_LOG_ALLOCATIONS_ENV = "MATE_GUARD_ALLOCATOR_LOG_ALLOCATIONS"

_SUPPORTED_MODES = {"tail", "head"}
_BOOTSTRAP_ERROR_PREFIX = "mate guard-run bootstrap failed"
_GRAPH_ERROR = (
    "MATE guard allocator does not support torch.musa.MUSAGraph capture. "
    "Disable guard allocator or graph capture for this run."
)

_install_lock = threading.Lock()
_install_state: "_InstallState | None" = None


@dataclasses.dataclass(frozen=True)
class GuardAllocatorConfig:
    mode: str = "tail"
    sync_on_free: bool = True
    log_allocations: bool = False

    def normalized(self) -> "GuardAllocatorConfig":
        mode = self.mode.strip().lower()
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f"Unsupported guard allocator mode: {self.mode!r}")
        return GuardAllocatorConfig(
            mode=mode,
            sync_on_free=bool(self.sync_on_free),
            log_allocations=bool(self.log_allocations),
        )


@dataclasses.dataclass
class _InstallState:
    config: GuardAllocatorConfig
    library_path: Path
    library: ctypes.CDLL
    allocator: Any


class _AllocatorShim:
    def __init__(self, allocator: Any):
        self._allocator = allocator

    def allocator(self) -> Any:
        return self._allocator


class _DisabledMUSAGraph:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(_GRAPH_ERROR)


def _parse_bool_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _get_bootstrap_dir() -> Path:
    return Path(__file__).resolve().parent / "_bootstrap"


def _get_package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _get_package_dir() -> Path:
    return Path(__file__).resolve().parent


def _get_sitecustomize_search_path() -> list[str]:
    bootstrap_dir = _get_bootstrap_dir().resolve()
    search_path: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or os.curdir).resolve()
        except OSError:
            resolved = None
        if resolved == bootstrap_dir:
            continue
        search_path.append(entry)
    return search_path


def _run_next_sitecustomize() -> None:
    spec = importlib.machinery.PathFinder.find_spec(
        "sitecustomize",
        _get_sitecustomize_search_path(),
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    # Replace the bootstrap shim so later imports see the caller's module.
    sys.modules["sitecustomize"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("sitecustomize", None)
        raise


def _load_build_meta_version() -> str | None:
    build_meta_path = _get_package_dir() / "_build_meta.py"
    if not build_meta_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "mate._guard_allocator_build_meta", build_meta_path
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version = getattr(module, "__version__", None)
    return version if isinstance(version, str) and version else None


def _load_repo_version() -> str | None:
    version_file = _get_package_root() / "version.txt"
    if not version_file.exists():
        return None
    version = version_file.read_text().strip()
    return version if version else None


def _load_installed_package_version() -> str | None:
    try:
        version = importlib.metadata.version("mate")
        return version if version else None
    except Exception:
        return None


def _resolve_runtime_base_version() -> str:
    from packaging.version import InvalidVersion, Version

    candidates = (
        (
            _load_repo_version(),
            _load_build_meta_version(),
            _load_installed_package_version(),
        )
        if (_get_package_root() / ".git").exists()
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
            return Version(candidate).base_version
        except InvalidVersion:
            continue
    return "0.0.0"


def _resolve_guard_allocator_source() -> Path:
    repo_source = _get_package_root() / "csrc" / f"{_GUARD_ALLOCATOR_NAME}.cpp"
    packaged_source = (
        _get_package_dir() / "data" / "csrc" / f"{_GUARD_ALLOCATOR_NAME}.cpp"
    )
    if (_get_package_root() / ".git").exists() and repo_source.exists():
        return repo_source
    if packaged_source.exists():
        return packaged_source
    return repo_source


def _resolve_guard_allocator_aot_path() -> Path:
    return (
        _get_package_dir()
        / "data"
        / "aot"
        / _GUARD_ALLOCATOR_NAME
        / f"{_GUARD_ALLOCATOR_NAME}.so"
    )


def _resolve_guard_allocator_build_dir() -> Path:
    base_dir = Path(os.getenv("MATE_WORKSPACE_BASE", Path.home().as_posix()))
    return (
        base_dir
        / ".cache"
        / "mate"
        / _resolve_runtime_base_version()
        / "host"
        / "cached_ops"
        / _GUARD_ALLOCATOR_NAME
    )


def _parse_env_flags(name: str) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return []
    try:
        return shlex.split(value)
    except ValueError:
        return value.split()


def _get_musa_home() -> Path:
    from tvm_ffi.cpp import extension as tvm_ffi_ext

    return Path(tvm_ffi_ext._find_musa_home())


def _run_build_command(command: list[str], *, cwd: Path, verbose: bool) -> None:
    if verbose:
        subprocess.run(command, cwd=str(cwd), check=True, text=True)
        return

    try:
        subprocess.run(
            command,
            cwd=str(cwd),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or ""
        raise RuntimeError(
            "Failed to build guard allocator library:\n"
            + " ".join(shlex.quote(part) for part in command)
            + ("\n" + output if output else "")
        ) from exc


def _build_guard_allocator_library(
    source: Path, output: Path, *, verbose: bool
) -> None:
    musa_home = _get_musa_home()
    musa_include = musa_home / "include"
    musa_lib = musa_home / "lib"
    musa_lib64 = musa_home / "lib64"
    cxx = os.environ.get("CXX", "c++")

    output.parent.mkdir(parents=True, exist_ok=True)
    object_path = output.parent / f"{_GUARD_ALLOCATOR_NAME}.o"
    dep_path = output.parent / f"{_GUARD_ALLOCATOR_NAME}.d"

    cflags = [
        "-std=c++17",
        "-fPIC",
        "-O2",
        f"-I{musa_include}",
        *_parse_env_flags("MATE_EXTRA_CFLAGS"),
    ]
    ldflags = [
        "-shared",
        f"-L{musa_lib}",
        *([f"-L{musa_lib64}"] if musa_lib64.exists() else []),
        "-lmusa",
        "-lmusart",
        *_parse_env_flags("MATE_EXTRA_LDFLAGS"),
    ]

    _run_build_command(
        [
            cxx,
            "-MMD",
            "-MF",
            str(dep_path),
            *cflags,
            "-c",
            str(source),
            "-o",
            str(object_path),
        ],
        cwd=output.parent,
        verbose=verbose,
    )
    _run_build_command(
        [cxx, str(object_path), *ldflags, "-o", str(output)],
        cwd=output.parent,
        verbose=verbose,
    )


def _ensure_guard_allocator_library() -> Path:
    aot_path = _resolve_guard_allocator_aot_path()
    if aot_path.exists():
        return aot_path

    if os.environ.get("MATE_DISABLE_JIT", "0") == "1":
        raise RuntimeError(
            "Guard allocator AOT library was not found and runtime builds are "
            "disabled via MATE_DISABLE_JIT=1."
        )

    source = _resolve_guard_allocator_source()
    if not source.exists():
        raise RuntimeError(f"Guard allocator source not found: {source}")

    build_dir = _resolve_guard_allocator_build_dir()
    library_path = build_dir / f"{_GUARD_ALLOCATOR_NAME}.so"
    if library_path.exists():
        return library_path

    from tvm_ffi.utils import FileLock

    build_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(build_dir / f"{_GUARD_ALLOCATOR_NAME}.lock")):
        if library_path.exists():
            return library_path
        verbose = os.environ.get("MATE_JIT_VERBOSE", "0") == "1"
        _build_guard_allocator_library(source, library_path, verbose=verbose)
    return library_path


def _configure_library(lib: ctypes.CDLL, config: GuardAllocatorConfig) -> None:
    lib.mate_guard_configure.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.mate_guard_configure.restype = None
    lib.mate_guard_configure(
        1 if config.mode == "tail" else 0,
        1 if config.sync_on_free else 0,
        1 if config.log_allocations else 0,
    )


def _patch_graph_capture() -> None:
    import torch
    import torch_musa
    import torch_musa.musa_graph.graphs as musa_graphs

    def _raise_capture_error(self, *args, **kwargs):
        raise RuntimeError(_GRAPH_ERROR)

    def _raise_graph_pool_error():
        raise RuntimeError(_GRAPH_ERROR)

    _raise_capture_error._mate_guard_allocator = True  # type: ignore[attr-defined]

    musa_graphs.MUSAGraph.capture_begin = _raise_capture_error
    torch.musa.MUSAGraph.capture_begin = _raise_capture_error
    torch_musa.MUSAGraph.capture_begin = _raise_capture_error
    musa_graphs.graph_pool_handle = _raise_graph_pool_error
    torch.musa.graph_pool_handle = musa_graphs.graph_pool_handle
    torch_musa.graph_pool_handle = musa_graphs.graph_pool_handle
    musa_graphs.MUSAGraph = _DisabledMUSAGraph
    torch.musa.MUSAGraph = _DisabledMUSAGraph
    torch_musa.MUSAGraph = _DisabledMUSAGraph


def install_guard_allocator(config: GuardAllocatorConfig | None = None) -> None:
    global _install_state

    normalized = (config or GuardAllocatorConfig()).normalized()
    with _install_lock:
        if _install_state is not None:
            if _install_state.config != normalized:
                raise RuntimeError(
                    "Guard allocator is already active with a different configuration."
                )
            return

        import torch_musa

        if not torch_musa.is_available():
            raise RuntimeError("torch_musa is available but reports no MUSA devices")

        library_path = _ensure_guard_allocator_library()
        library = ctypes.CDLL(str(library_path))
        _configure_library(library, normalized)

        alloc_ptr = ctypes.cast(library.mate_guard_alloc, ctypes.c_void_p).value
        free_ptr = ctypes.cast(library.mate_guard_free, ctypes.c_void_p).value
        base_ptr = ctypes.cast(library.mate_guard_base_alloc, ctypes.c_void_p).value
        if alloc_ptr is None or free_ptr is None or base_ptr is None:
            raise RuntimeError("Failed to resolve guard allocator entry points")

        internal_allocator = torch_musa._MUSAC._musa_customAllocator(
            alloc_ptr, free_ptr
        )
        if not hasattr(internal_allocator, "set_base_alloc_fn"):
            raise RuntimeError(
                "Current torch_musa build does not expose set_base_alloc_fn for custom allocators"
            )
        internal_allocator.set_base_alloc_fn(base_ptr)

        try:
            torch_musa.change_current_allocator(_AllocatorShim(internal_allocator))
        except RuntimeError as exc:
            raise RuntimeError(
                "Guard allocator must be installed before the first MUSA allocation in the process"
            ) from exc

        _patch_graph_capture()
        os.environ[_ACTIVE_ENV] = "1"
        os.environ[_MODE_ENV] = normalized.mode
        os.environ[_SYNC_ON_FREE_ENV] = "1" if normalized.sync_on_free else "0"
        os.environ[_LOG_ALLOCATIONS_ENV] = "1" if normalized.log_allocations else "0"
        _install_state = _InstallState(
            config=normalized,
            library_path=library_path,
            library=library,
            allocator=internal_allocator,
        )


def is_guard_allocator_active() -> bool:
    return _install_state is not None


def auto_install_from_env() -> None:
    if not _parse_bool_env(os.environ.get(_AUTO_INSTALL_ENV), default=False):
        return
    config = GuardAllocatorConfig(
        mode=os.environ.get(_MODE_ENV, "tail"),
        sync_on_free=_parse_bool_env(os.environ.get(_SYNC_ON_FREE_ENV), default=True),
        log_allocations=_parse_bool_env(
            os.environ.get(_LOG_ALLOCATIONS_ENV), default=False
        ),
    )
    install_guard_allocator(config)


def bootstrap_guard_run_sitecustomize() -> None:
    try:
        auto_install_from_env()
    except Exception as exc:
        print(
            f"{_BOOTSTRAP_ERROR_PREFIX}: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        os._exit(1)

    _run_next_sitecustomize()


def build_guard_run_env(
    config: GuardAllocatorConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    normalized = (config or GuardAllocatorConfig()).normalized()
    base_env = dict(os.environ if env is None else env)
    extra_paths = [str(_get_bootstrap_dir()), str(_get_package_root())]
    pythonpath = base_env.get("PYTHONPATH")
    if pythonpath:
        paths = [path for path in pythonpath.split(os.pathsep) if path]
    else:
        paths = []
    for extra_path in reversed(extra_paths):
        if extra_path not in paths:
            paths.insert(0, extra_path)
    base_env["PYTHONPATH"] = os.pathsep.join(paths)

    base_env[_AUTO_INSTALL_ENV] = "1"
    base_env[_MODE_ENV] = normalized.mode
    base_env[_SYNC_ON_FREE_ENV] = "1" if normalized.sync_on_free else "0"
    base_env[_LOG_ALLOCATIONS_ENV] = "1" if normalized.log_allocations else "0"
    return base_env


def run_guarded_command(
    command: list[str],
    config: GuardAllocatorConfig | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    guarded_env = build_guard_run_env(config, env=env)
    completed = subprocess.run(command, env=guarded_env, check=False)
    return completed.returncode
