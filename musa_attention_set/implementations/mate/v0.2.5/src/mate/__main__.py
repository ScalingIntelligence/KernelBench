#!/usr/bin/env python3
"""
MATE CLI - Command Line Interface for MATE (MUSA AI Tensor Engine)

Usage:
    mate --help
    mate show-config
    mate replay --dir mate_dumps/
    mate guard-run -- python your_script.py

Requirements:
    pip install click>=8.0.0
"""

import os
import sys
import json
import re
import importlib
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any

import click

TVM_FFI_PACKAGE_NAME = "apache-tvm-ffi"
MATE_WHEEL_INDEX_URL = "https://dl.mthreads.com/repo/api/pypi/pypi/simple"
TVM_FFI_INSTALL_HINT = (
    "Install/build apache-tvm-ffi from the MooreThreads MUSA fork: "
    "https://github.com/MooreThreads/tvm-ffi. Use a release tag "
    "containing 'musa' (for example v0.1.9.post2+musa.1); the upstream "
    "public build is not compatible with MATE."
)
WRAPPER_PACKAGES = (
    {
        "distribution": "flash_attn_3",
        "module": "flash_attn_3",
        "display": "flash_attn_3",
    },
    {
        "distribution": "sageattention",
        "module": "sageattention",
        "display": "sageattention",
    },
    {"distribution": "flash_mla", "module": "flash_mla", "display": "flash_mla"},
    {"distribution": "deep-gemm", "module": "deep_gemm", "display": "deep-gemm"},
)

_DEV_SUFFIX_RE = re.compile(r"\.dev\d+$")


# Version - try multiple sources
def _load_build_meta():
    """Load version info from _build_meta.py directly to avoid importing mate package."""
    import importlib.util

    build_meta_path = Path(__file__).parent / "_build_meta.py"
    if build_meta_path.exists():
        spec = importlib.util.spec_from_file_location("_build_meta", build_meta_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, "__version__", "unknown"), getattr(
                module, "__git_version__", "unknown"
            )
    return "unknown", "unknown"


__version__, __git_version__ = _load_build_meta()

# Fallback to installed package if direct load fails
if __version__ == "unknown":
    try:
        from importlib.metadata import version

        __version__ = version("mate")
    except Exception:
        pass


def _get_replay_functions():
    """Load replay functions from api_logging module.

    Returns:
        (replay_from_dump, replay_sequence) on success, or (None, error_message) on failure.
    """
    try:
        from mate.api_logging import replay_from_dump, replay_sequence

        return replay_from_dump, replay_sequence
    except Exception as e:
        return None, str(e)


# Environment variables documentation
ENV_VARS = {
    "MATE_LOGLEVEL": "Logging level (0, 1, 3, 5, 10)",
    "MATE_LOGDEST": "Log destination (stdout, stderr, or file path)",
    "MATE_DUMP_DIR": "Directory for Level 10 tensor dumps",
    "MATE_DUMP_MAX_SIZE_GB": "Max total dump size in GB",
    "MATE_DUMP_MAX_COUNT": "Max number of dumps",
    "MATE_DUMP_SAFETENSORS": "Use safetensors format (0/1)",
    "MATE_DUMP_INCLUDE": "Include patterns for dumping",
    "MATE_DUMP_EXCLUDE": "Exclude patterns for dumping",
    "MATE_MUSA_ARCH_LIST": "MUSA architecture list for JIT/AOT compilation",
    "MATE_WORKSPACE_BASE": "Base directory for the MATE cache workspace",
    "MATE_DISABLE_JIT": "Disable runtime JIT and require matching AOT modules",
    "MATE_JIT_VERBOSE": "Show verbose ninja output for runtime JIT builds",
    "MATE_DEEPGEMM_MK_ALIGNMENT": "DeepGEMM contiguous grouped GEMM M alignment (128 or 256)",
    "MATE_PYTEST_GUARD_ALLOC": "Enable guard allocator by default for pytest runs (0/1)",
    "MATE_PYTEST_GUARD_MODE": "Default pytest guard allocator mode (tail or head)",
    "MATE_PYTEST_GUARD_LOG_ALLOCATIONS": "Log guarded alloc/free events during pytest runs (0/1)",
    "MATE_GUARD_ALLOCATOR_AUTO_INSTALL": "Internal bootstrap flag for guard-run",
    "MATE_GUARD_ALLOCATOR_MODE": "Guard allocator mode (tail or head)",
    "MATE_GUARD_ALLOCATOR_SYNC_ON_FREE": "Synchronize device before guarded frees (0/1)",
    "MATE_GUARD_ALLOCATOR_LOG_ALLOCATIONS": "Log guarded alloc/free events to stderr (0/1)",
    "MATE_EXTRA_CFLAGS": "Extra host compiler flags for JIT builds",
    "MATE_EXTRA_MUSAFLAGS": "Extra mcc flags for JIT builds",
    "MATE_EXTRA_LDFLAGS": "Extra linker flags for JIT builds",
    "MATE_MCC": "Override the mcc compiler path used by JIT builds",
    "MATE_MUBIN_DIR": "Directory for explicit MUBIN artifact downloads",
    "MATE_MUBIN_NO_DOWNLOAD": "Set to disable automatic MUBIN artifact downloads",
    "MATE_MUBIN_VERIFY_DISABLED": "Set to disable MUBIN artifact hash verification",
    "MATE_MUBIN_DOWNLOAD_VERBOSE": "Show verbose MUBIN download output (0/1)",
}


def _read_installed_package_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version as package_version

        resolved = package_version(package_name)
        return resolved if resolved else None
    except Exception:
        return None


def _public_version(version: str) -> str:
    try:
        from packaging.version import Version

        return Version(version).public
    except Exception:
        return version.split("+", 1)[0]


def _wrapper_compat_version(version: str) -> str:
    return _DEV_SUFFIX_RE.sub("", _public_version(version))


def _short_commit(commit: str | None) -> str:
    if not commit or commit == "unknown":
        return "unknown"
    return commit[:16]


def _parse_build_meta(text: str) -> dict[str, str]:
    values = {}
    for name in ("__version__", "__git_version__"):
        match = re.search(rf"(?m)^{name}\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            values[name] = match.group(1)
    return values


def _read_wrapper_build_meta_from_distribution(distribution, module_name: str):
    try:
        files = distribution.files or []
    except Exception:
        files = []

    expected_path = f"{module_name}/_build_meta.py"
    for file in files:
        if file.as_posix() != expected_path:
            continue
        try:
            return _parse_build_meta(
                distribution.locate_file(file).read_text(encoding="utf-8")
            )
        except Exception:
            return {}

    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        spec = None
    if not spec or not spec.submodule_search_locations:
        return {}

    build_meta_path = (
        Path(next(iter(spec.submodule_search_locations))) / "_build_meta.py"
    )
    if not build_meta_path.exists():
        return {}
    try:
        return _parse_build_meta(build_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_exact_requirement_pin(requirement_specifier) -> str | None:
    specifiers = list(requirement_specifier)
    if len(specifiers) != 1:
        return None

    specifier = specifiers[0]
    if specifier.operator != "==" or specifier.version.endswith(".*"):
        return None
    return specifier.version


def _read_mate_requirement(distribution) -> tuple[str | None, str | None]:
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except Exception:
        return None, None

    for raw_requirement in distribution.requires or []:
        try:
            requirement = Requirement(raw_requirement)
        except Exception:
            continue
        if canonicalize_name(requirement.name) == "mate":
            return raw_requirement, _extract_exact_requirement_pin(
                requirement.specifier
            )
    return None, None


def _get_wrapper_package_statuses(expected_version: str, expected_git_version: str):
    from importlib import metadata

    expected_wrapper_version = _wrapper_compat_version(expected_version)
    statuses = []
    for wrapper in WRAPPER_PACKAGES:
        status: dict[str, Any] = {
            **wrapper,
            "installed": False,
            "version": None,
            "git_version": None,
            "mate_requirement": None,
            "errors": [],
        }

        try:
            distribution = metadata.distribution(wrapper["distribution"])
        except metadata.PackageNotFoundError:
            statuses.append(status)
            continue

        status["installed"] = True
        status["version"] = distribution.version
        build_meta = _read_wrapper_build_meta_from_distribution(
            distribution, wrapper["module"]
        )
        status["git_version"] = build_meta.get("__git_version__")
        status["build_meta_version"] = build_meta.get("__version__")
        mate_requirement, mate_exact_version = _read_mate_requirement(distribution)
        status["mate_requirement"] = mate_requirement

        wrapper_compat_version = (
            _wrapper_compat_version(status["version"]) if status["version"] else None
        )
        if wrapper_compat_version != expected_wrapper_version:
            status["errors"].append(
                "version "
                f"{status['version']} does not match MATE-compatible version "
                f"{expected_wrapper_version} (ignoring dev suffix)"
            )

        if (
            status["build_meta_version"]
            and status["build_meta_version"] != status["version"]
        ):
            status["errors"].append(
                "build metadata version "
                f"{status['build_meta_version']} does not match package metadata "
                f"{status['version']}"
            )

        expected_specifier = f"=={expected_wrapper_version}"
        if not mate_exact_version:
            status["errors"].append(
                f"dependency on mate is {mate_requirement or 'missing'}, "
                f"expected an exact mate{expected_specifier} pin"
            )
        elif _wrapper_compat_version(mate_exact_version) != expected_wrapper_version:
            status["errors"].append(
                f"dependency on mate is {mate_requirement}, expected mate-compatible "
                f"version {expected_wrapper_version} (ignoring dev suffix)"
            )

        if not expected_git_version or expected_git_version == "unknown":
            status["errors"].append("MATE commit metadata is unavailable")
        elif not status["git_version"] or status["git_version"] == "unknown":
            status["errors"].append("wrapper commit metadata is unavailable")
        elif status["git_version"] != expected_git_version:
            status["errors"].append(
                "wrapper commit "
                f"{_short_commit(status['git_version'])} does not match MATE "
                f"{_short_commit(expected_git_version)}"
            )

        statuses.append(status)

    return statuses


def _read_tvm_ffi_module_version() -> str | None:
    try:
        import tvm_ffi
    except Exception:
        return None

    version = getattr(tvm_ffi, "__version__", None)
    return version if isinstance(version, str) and version else None


def _is_musa_enabled_tvm_ffi(version: str | None) -> bool:
    return bool(version) and "musa" in version.lower()


def _get_tvm_ffi_status():
    version = _read_installed_package_version(TVM_FFI_PACKAGE_NAME)
    version_source = "package metadata"
    if not version:
        version = _read_tvm_ffi_module_version()
        version_source = "tvm_ffi.__version__" if version else None

    if not version:
        return {
            "package_name": TVM_FFI_PACKAGE_NAME,
            "version": "Not installed",
            "installed": False,
            "musa_enabled": False,
            "version_source": version_source,
            "status_message": (
                f"{TVM_FFI_PACKAGE_NAME} is not installed in the current environment."
            ),
            "install_hint": TVM_FFI_INSTALL_HINT,
        }

    if _is_musa_enabled_tvm_ffi(version):
        return {
            "package_name": TVM_FFI_PACKAGE_NAME,
            "version": version,
            "installed": True,
            "musa_enabled": True,
            "version_source": version_source,
            "status_message": (
                f"Detected MUSA-enabled {TVM_FFI_PACKAGE_NAME} build: {version}"
            ),
            "install_hint": None,
        }

    return {
        "package_name": TVM_FFI_PACKAGE_NAME,
        "version": version,
        "installed": True,
        "musa_enabled": False,
        "version_source": version_source,
        "status_message": (
            f"Detected {TVM_FFI_PACKAGE_NAME} {version}, but it is not a "
            "MUSA-enabled build."
        ),
        "install_hint": TVM_FFI_INSTALL_HINT,
    }


def _resolve_musa_arch_source() -> str:
    return "env" if "MATE_MUSA_ARCH_LIST" in os.environ else "auto-detected"


def _describe_musa_arch_source(source: str) -> str:
    if source == "env":
        return "MATE_MUSA_ARCH_LIST"
    return "auto-detected from visible MUSA devices"


def _get_musa_arch_status():
    source = _resolve_musa_arch_source()

    try:
        jit_env = importlib.import_module("mate.jit.env")
    except Exception as exc:
        return {
            "musa_arch_list": None,
            "musa_arch_key": None,
            "musa_arch_source": source,
            "musa_arch_available": False,
            "musa_arch_status": "Unavailable",
            "musa_arch_error": str(exc),
        }

    return {
        "musa_arch_list": list(jit_env.MATE_MUSA_ARCH_LIST),
        "musa_arch_key": jit_env.MATE_MUSA_ARCH_KEY,
        "musa_arch_source": source,
        "musa_arch_available": True,
        "musa_arch_status": (
            "Resolved from MATE_MUSA_ARCH_LIST"
            if source == "env"
            else "Auto-detected from visible MUSA devices"
        ),
        "musa_arch_error": None,
    }


def _get_jit_status_info(register_modules: bool = False):
    try:
        from mate.jit import env as jit_env
        from mate.jit import jit_spec_registry

        registration_error = None
        if register_modules and not jit_spec_registry.get_all_statuses():
            try:
                from mate.aot import register_default_modules

                register_default_modules()
            except Exception as exc:
                registration_error = str(exc)

        return {
            "available": True,
            "error": registration_error,
            "cache_dir": str(jit_env.MATE_CACHE_DIR),
            "workspace_dir": str(jit_env.MATE_WORKSPACE_DIR),
            "jit_dir": str(jit_env.MATE_JIT_DIR),
            "gen_src_dir": str(jit_env.MATE_GEN_SRC_DIR),
            "aot_dir": str(jit_env.MATE_AOT_DIR),
            "disable_jit": jit_env.disable_jit_enabled(),
            "jit_verbose": os.environ.get("MATE_JIT_VERBOSE", "0") == "1",
            "stats": jit_spec_registry.get_stats(),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "cache_dir": None,
            "workspace_dir": None,
            "jit_dir": None,
            "gen_src_dir": None,
            "aot_dir": None,
            "disable_jit": os.environ.get("MATE_DISABLE_JIT", "0") == "1",
            "jit_verbose": os.environ.get("MATE_JIT_VERBOSE", "0") == "1",
            "stats": {
                "total": 0,
                "compiled": 0,
                "aot": 0,
                "jit_compiled": 0,
                "not_compiled": 0,
            },
        }


def _install_guard_allocator_for_replay(mode: str) -> None:
    from mate.memory_debug import GuardAllocatorConfig, install_guard_allocator

    install_guard_allocator(GuardAllocatorConfig(mode=mode))


def get_system_info():
    """Gather system information."""
    tvm_ffi_status = _get_tvm_ffi_status()
    musa_arch_status = _get_musa_arch_status()
    jit_status = _get_jit_status_info(register_modules=True)
    info = {
        "mate_version": __version__,
        "git_version": __git_version__,
        "python_version": sys.version.split()[0],
        "tvm_ffi_version": tvm_ffi_status["version"],
        "tvm_ffi_installed": tvm_ffi_status["installed"],
        "tvm_ffi_musa_enabled": tvm_ffi_status["musa_enabled"],
        "tvm_ffi_status": tvm_ffi_status["status_message"],
        "tvm_ffi_install_hint": tvm_ffi_status["install_hint"],
        "musa_arch_list": musa_arch_status["musa_arch_list"],
        "musa_arch_key": musa_arch_status["musa_arch_key"],
        "musa_arch_source": musa_arch_status["musa_arch_source"],
        "musa_arch_available": musa_arch_status["musa_arch_available"],
        "musa_arch_status": musa_arch_status["musa_arch_status"],
        "musa_arch_error": musa_arch_status["musa_arch_error"],
        "jit_available": jit_status["available"],
        "jit_error": jit_status["error"],
        "jit_cache_dir": jit_status["cache_dir"],
        "jit_workspace_dir": jit_status["workspace_dir"],
        "jit_dir": jit_status["jit_dir"],
        "jit_gen_src_dir": jit_status["gen_src_dir"],
        "aot_dir": jit_status["aot_dir"],
        "disable_jit": jit_status["disable_jit"],
        "jit_verbose": jit_status["jit_verbose"],
        "jit_stats": jit_status["stats"],
    }

    # PyTorch info
    try:
        import torch

        info["torch_version"] = torch.__version__
    except ImportError:
        info["torch_version"] = "Not installed"

    # MUSA info
    try:
        import torch_musa

        info["torch_musa_version"] = torch_musa.__version__
        info["musa_available"] = torch_musa.is_available()
        if torch_musa.is_available():
            info["musa_device_count"] = torch_musa.device_count()
            info["musa_devices"] = [
                torch_musa.get_device_name(i) for i in range(torch_musa.device_count())
            ]
    except ImportError:
        info["torch_musa_version"] = "Not installed"
        info["musa_available"] = False

    return info


def print_header(text: str, fg: str = "yellow"):
    """Print a section header."""
    click.secho(f"\n=== {text} ===", fg=fg, bold=True)


def print_kv(
    key: str, value: str, key_color: str = "magenta", value_color: str = "cyan"
):
    """Print a key-value pair."""
    click.secho(f"{key}:", fg=key_color, nl=False)
    click.secho(f" {value}", fg=value_color)


def _ensure_modules_registered(announce: bool = True):
    try:
        from mate.aot import register_default_modules
        from mate.jit import jit_spec_registry

        statuses = jit_spec_registry.get_all_statuses()
        if not statuses:
            if announce:
                click.secho(
                    "No modules found. Registering default modules...", fg="yellow"
                )
            count = register_default_modules()
            if announce:
                click.secho(f"Registered {count} modules", fg="green")
            statuses = jit_spec_registry.get_all_statuses()
        return statuses
    except Exception as exc:
        click.secho(f"Unable to register JIT modules: {exc}", fg="red")
        if "MATE_MUSA_ARCH_LIST" not in os.environ:
            click.secho(
                "Set MATE_MUSA_ARCH_LIST explicitly for offline diagnostics, "
                "for example: MATE_MUSA_ARCH_LIST=3.1",
                fg="yellow",
            )
        return []


def _status_color(status: str) -> str:
    if status == "AOT":
        return "green"
    if status == "JIT Compiled":
        return "cyan"
    return "red"


def _print_status_summary(statuses):
    from mate.jit import jit_spec_registry

    stats = jit_spec_registry.get_stats()
    print_kv("Total modules", str(stats["total"]))
    print_kv("Compiled", str(stats["compiled"]), value_color="green")
    print_kv("AOT", str(stats["aot"]), value_color="green")
    print_kv("JIT compiled", str(stats["jit_compiled"]), value_color="cyan")
    print_kv("Not compiled", str(stats["not_compiled"]), value_color="red")


def _status_as_dict(status):
    return {
        "name": status.name,
        "status": status.status,
        "library_path": str(status.library_path) if status.library_path else None,
        "aot_path": str(status.aot_path),
        "jit_library_path": str(status.jit_library_path),
        "build_dir": str(status.build_dir),
        "ninja_path": str(status.ninja_path),
        "sources": [str(source) for source in status.sources],
        "num_sources": status.num_sources,
        "num_generated_sources": status.num_generated_sources,
    }


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="mate")
@click.pass_context
def cli(ctx):
    """MATE CLI - MUSA AI Tensor Engine Command Line Interface

    Examples:
        mate show-config              # Display configuration
        mate replay --dir dumps/      # Replay API calls
        mate guard-run -- python app.py
        mate env                      # Show environment variables
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("show-config")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def show_config(output_json: bool):
    """Display MATE configuration and system information."""
    info = get_system_info()

    if output_json:
        click.echo(json.dumps(info, indent=2))
        return

    # Version Info
    print_header("Version Information")
    print_kv("MATE version", info["mate_version"])
    print_kv("Git version", info.get("git_version", "unknown")[:16])
    print_kv("Python version", info["python_version"])
    print_kv("PyTorch version", info.get("torch_version", "N/A"))

    # MUSA Info
    print_header("MUSA Information")
    print_kv("torch_musa version", info.get("torch_musa_version", "N/A"))

    if info.get("musa_available"):
        click.secho("MUSA available: ", fg="magenta", nl=False)
        click.secho("Yes", fg="green")
        print_kv("Device count", str(info["musa_device_count"]))

        for i, name in enumerate(info.get("musa_devices", [])):
            click.secho(f"  Device {i}:", fg="white", nl=False)
            click.secho(f" {name}", fg="cyan")
    else:
        click.secho("MUSA available: ", fg="magenta", nl=False)
        click.secho("No", fg="red")

    # JIT Architecture Info
    print_header("JIT Architecture")
    arch_source = info["musa_arch_source"]
    print_kv("Source", _describe_musa_arch_source(arch_source))
    print_kv(
        "Architecture list",
        ", ".join(info["musa_arch_list"]) if info["musa_arch_list"] else "N/A",
        value_color="cyan" if info["musa_arch_available"] else "yellow",
    )
    print_kv(
        "Arch key",
        info["musa_arch_key"] or "N/A",
        value_color="cyan" if info["musa_arch_available"] else "yellow",
    )
    click.secho("Status: ", fg="magenta", nl=False)
    click.secho(
        info["musa_arch_status"],
        fg="green" if info["musa_arch_available"] else "yellow",
        bold=not info["musa_arch_available"],
    )
    if info["musa_arch_error"]:
        click.secho("Reason: ", fg="magenta", nl=False)
        click.secho(info["musa_arch_error"], fg="yellow", bold=True)

    # JIT/AOT Info
    print_header("JIT/AOT Status")
    if info["jit_available"]:
        print_kv("Cache directory", info["jit_cache_dir"] or "N/A")
        print_kv("Workspace directory", info["jit_workspace_dir"] or "N/A")
        print_kv("Cached ops directory", info["jit_dir"] or "N/A")
        print_kv("Generated source directory", info["jit_gen_src_dir"] or "N/A")
        print_kv("AOT directory", info["aot_dir"] or "N/A")
        print_kv("MATE_DISABLE_JIT", str(info["disable_jit"]))
        print_kv("MATE_JIT_VERBOSE", str(info["jit_verbose"]))
        stats = info["jit_stats"]
        print_kv("Registered modules", str(stats["total"]))
        print_kv("Compiled modules", str(stats["compiled"]), value_color="green")
        print_kv("AOT modules", str(stats["aot"]), value_color="green")
        print_kv("JIT compiled modules", str(stats["jit_compiled"]), value_color="cyan")
        print_kv("Not compiled modules", str(stats["not_compiled"]), value_color="red")
        if info["jit_error"]:
            click.secho("JIT registration warning: ", fg="magenta", nl=False)
            click.secho(info["jit_error"], fg="yellow", bold=True)
    else:
        click.secho("JIT unavailable: ", fg="magenta", nl=False)
        click.secho(info["jit_error"], fg="yellow", bold=True)

    # TVM-FFI Info
    print_header("TVM-FFI")
    version_color = "cyan" if info["tvm_ffi_musa_enabled"] else "red"
    print_kv(TVM_FFI_PACKAGE_NAME, info["tvm_ffi_version"], value_color=version_color)
    click.secho("Status: ", fg="magenta", nl=False)
    click.secho(
        info["tvm_ffi_status"],
        fg="green" if info["tvm_ffi_musa_enabled"] else "red",
        bold=not info["tvm_ffi_musa_enabled"],
    )
    if info["tvm_ffi_install_hint"]:
        click.secho("Install hint: ", fg="magenta", nl=False)
        click.secho(info["tvm_ffi_install_hint"], fg="yellow", bold=True)

    # AOT Info
    print_header("AOT Libraries")
    try:
        import mate

        mate_path = Path(mate.__file__).parent
        aot_dir = mate_path / "data" / "aot"

        if aot_dir.exists():
            aot_files = list(aot_dir.rglob("*.so"))
            print_kv("AOT directory", str(aot_dir))
            print_kv("AOT libraries", str(len(aot_files)))
            for f in aot_files[:5]:
                click.secho(f"  - {f.name}", fg="white")
            if len(aot_files) > 5:
                click.secho(f"  ... and {len(aot_files) - 5} more", fg="white")
        else:
            click.secho("AOT directory not found", fg="yellow")
    except Exception as e:
        click.secho(f"Error checking AOT: {e}", fg="red")

    # Installed packages
    print_header("Installed Packages")
    try:
        import importlib.metadata as metadata

        packages = ["mate", "torch", "torch_musa", TVM_FFI_PACKAGE_NAME]
        for pkg in packages:
            if pkg == TVM_FFI_PACKAGE_NAME:
                value_color = "cyan" if info["tvm_ffi_musa_enabled"] else "red"
                print_kv(pkg, info["tvm_ffi_version"], value_color=value_color)
                continue
            try:
                version = metadata.version(pkg)
                print_kv(pkg, version)
            except metadata.PackageNotFoundError:
                print_kv(pkg, "Not installed", value_color="red")
    except Exception as e:
        click.secho(f"Error checking packages: {e}", fg="yellow")


@cli.command("env")
def env_cmd():
    """Display MATE environment variables and their current values."""
    print_header("Environment Variables")

    max_key_len = max(len(k) for k in ENV_VARS.keys())

    for var, description in ENV_VARS.items():
        value = os.environ.get(var, "<not set>")
        display_value = value[:50] + "..." if len(value) > 50 else value
        value_color = "yellow" if value != "<not set>" else "red"

        click.secho(f"{var:<{max_key_len}} ", fg="cyan", nl=False)
        click.secho(f"{display_value:<53} ", fg=value_color, nl=False)
        click.secho(description, fg="white")


@cli.command("module-status")
@click.option("--detailed", is_flag=True, help="Show source and path details")
def module_status_cmd(detailed: bool):
    """Show registered JIT/AOT module status without compiling."""
    statuses = _ensure_modules_registered()
    if not statuses:
        return

    statuses = sorted(statuses, key=lambda status: status.name)
    print_header("Module Status")
    _print_status_summary(statuses)
    click.echo()

    for status in statuses:
        click.secho(f"{status.name:<72} ", fg="white", nl=False)
        click.secho(status.status, fg=_status_color(status.status))
        if detailed:
            if status.library_path:
                print_kv("  Library", str(status.library_path), key_color="white")
            print_kv("  AOT path", str(status.aot_path), key_color="white")
            print_kv("  JIT path", str(status.jit_library_path), key_color="white")
            print_kv("  Ninja", str(status.ninja_path), key_color="white")
            print_kv("  Sources", str(status.num_sources), key_color="white")
            print_kv(
                "  Generated sources",
                str(status.num_generated_sources),
                key_color="white",
            )


@cli.command("list-modules")
@click.argument("module_name", required=False)
def list_modules_cmd(module_name: str | None):
    """List registered JIT/AOT modules or inspect one module."""
    statuses = _ensure_modules_registered(announce=module_name is None)
    if not statuses:
        return

    if module_name:
        match = next(
            (status for status in statuses if status.name == module_name), None
        )
        if match is None:
            click.secho(f"Module '{module_name}' not found.", fg="red")
            return
        click.echo(json.dumps(_status_as_dict(match), indent=2))
        return

    for status in sorted(statuses, key=lambda item: item.name):
        click.secho(f"{status.name:<72} ", fg="white", nl=False)
        click.secho(status.status, fg=_status_color(status.status))


@cli.command("export-compile-commands")
@click.argument("path", required=False, default="compile_commands.json")
def export_compile_commands_cmd(path: str):
    """Export JIT module compile commands for IDEs and tooling."""
    statuses = _ensure_modules_registered()
    if not statuses:
        return

    from mate.jit import jit_spec_registry

    compile_commands = []
    for spec in jit_spec_registry.get_all_specs().values():
        compile_commands.extend(spec.get_compile_commands())

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(compile_commands, f, indent=2)
    click.secho(
        f"Exported {len(compile_commands)} compile commands to {output_path}",
        fg="green",
    )


@cli.command("clear-cache")
def clear_cache_cmd():
    """Clear the runtime JIT cache directory, leaving AOT libraries untouched."""
    try:
        from mate.jit import clear_cache_dir

        clear_cache_dir()
        click.secho("JIT cache cleared successfully.", fg="green")
    except Exception as exc:
        click.secho(f"Failed to clear JIT cache: {exc}", fg="red")


@cli.command("install-mubin-wheel")
@click.option(
    "--mate-version",
    default=None,
    help="MATE version to match (defaults to the running MATE version).",
)
@click.option(
    "--index-url",
    default=MATE_WHEEL_INDEX_URL,
    show_default=True,
    help="Python package index containing mate-mubin wheels.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the pip command without running it.",
)
def install_mubin_wheel_cmd(
    mate_version: str | None,
    index_url: str,
    dry_run: bool,
):
    """Install the mate-mubin wheel matching MATE."""
    version = _public_version(mate_version or __version__)
    if version == "unknown":
        raise click.ClickException(
            "Unable to resolve the MATE version; pass --mate-version explicitly."
        )

    requirement = f"mate-mubin=={version}"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-deps",
        "--index-url",
        index_url,
        requirement,
    ]
    click.echo(shlex.join(command))
    if dry_run:
        return

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise click.ClickException(
            f"Failed to install {requirement} (pip exited with {result.returncode})"
        )


@cli.command("download-mubin")
def download_mubin_cmd():
    """Download MATE MUBIN metadata and kernel artifacts."""
    try:
        from mate.artifacts import download_artifacts, resolve_mubin_output_dir

        output_dir = resolve_mubin_output_dir()
        downloaded = download_artifacts(output_dir)
        click.secho(
            f"Downloaded {len(downloaded)} MATE MUBIN metadata/kernel files to "
            f"{output_dir}",
            fg="green",
        )
    except Exception as exc:
        click.secho(f"Failed to download MATE MUBIN artifacts: {exc}", fg="red")


@cli.command("list-mubins")
def list_mubins_cmd():
    """List active MATE MUBIN artifact status by module."""
    try:
        from mate.artifacts import get_active_mubin_artifacts_status

        active_dir, source, statuses = get_active_mubin_artifacts_status()
    except Exception as exc:
        click.secho(f"Failed to list MATE MUBIN artifacts: {exc}", fg="red")
        return

    print_header("MATE MUBIN Artifacts")
    print_kv("Source", source)
    print_kv("Directory", str(active_dir))
    click.echo()
    for module, status, module_dir in statuses:
        status_color = {
            "Downloaded": "green",
            "Metadata only": "cyan",
            "Incomplete": "yellow",
        }.get(status, "red")
        click.secho(f"{module:<20} ", fg="white", nl=False)
        click.secho(f"{status:<12} ", fg=status_color, nl=False)
        click.secho(str(module_dir), fg="cyan")


@cli.command("clear-mubin")
def clear_mubin_cmd():
    """Clear the configured MATE MUBIN artifact directory."""
    try:
        from mate.artifacts import clear_mubin_artifacts

        output_dir, cleared = clear_mubin_artifacts()
        if cleared:
            click.secho(f"MUBIN artifacts cleared: {output_dir}", fg="green")
        else:
            click.secho(
                f"MUBIN artifact directory does not exist: {output_dir}", fg="yellow"
            )
    except Exception as exc:
        click.secho(f"Failed to clear MATE MUBIN artifacts: {exc}", fg="red")


@cli.command("replay")
@click.option(
    "--dir",
    "dump_dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing dump files",
)
@click.option(
    "--device", default="musa", help="Target device for replay (musa, cpu, musa:N)"
)
@click.option(
    "--run/--no-run", default=True, help="Actually execute functions (default: run)"
)
@click.option(
    "--compare/--no-compare", default=True, help="Compare outputs with expected values"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option(
    "--guard-alloc/--no-guard-alloc",
    default=False,
    help="Install the MATE guard allocator before replaying MUSA tensors",
)
@click.option(
    "--guard-mode",
    type=click.Choice(["tail", "head"]),
    default="tail",
    show_default=True,
    help="Which side of the allocation should use the unmapped guard page",
)
def replay_cmd(
    dump_dir: str,
    device: str,
    run: bool,
    compare: bool,
    verbose: bool,
    guard_alloc: bool,
    guard_mode: str,
):
    """Replay API calls from Level 10 dump directory.

    Examples:
        mate replay --dir mate_dumps/20260310_xxx_call0001
        mate replay --dir mate_dumps/ --verbose
        mate replay --dir dumps/ --device cpu --no-run
    """
    if guard_alloc:
        if not device.startswith("musa"):
            raise click.ClickException(
                "--guard-alloc only supports replay targets on MUSA devices"
            )
        _install_guard_allocator_for_replay(guard_mode)

    replay_from_dump, replay_sequence = _get_replay_functions()

    if replay_from_dump is None:
        error_msg = replay_sequence  # second return value holds the error string
        click.secho("❌ Failed to load MATE API logging module:", fg="red")
        click.secho(f"   {error_msg}", fg="yellow")
        click.secho(
            "   Ensure MATE is correctly installed: pip install mate", fg="yellow"
        )
        sys.exit(1)

    dump_path = Path(dump_dir)
    is_single_dump = (dump_path / "metadata.jsonl").exists()

    if is_single_dump:
        click.secho(f"Replaying single dump: {dump_dir}", fg="cyan")

        try:
            result = replay_from_dump(
                str(dump_path), device=device, run=run, compare_outputs=compare
            )

            if verbose:
                click.secho("\nDump Information:", fg="yellow")
                metadata = result.get("metadata", {})
                print_kv("Function", metadata.get("function_name", "N/A"))
                print_kv("Module", metadata.get("module", "N/A"))
                print_kv("Status", metadata.get("execution_status", "N/A"))

                click.secho("\nArguments:", fg="yellow")
                for i, arg in enumerate(result.get("args", [])):
                    if hasattr(arg, "shape"):
                        click.secho(
                            f"  arg[{i}]: {list(arg.shape)} {arg.dtype}", fg="white"
                        )
                    else:
                        click.secho(f"  arg[{i}]: {arg}", fg="white")

            if "error" in result:
                click.secho(f"❌ Error: {result['error']}", fg="red")
                sys.exit(1)

            if run:
                if "execution_error" in result:
                    click.secho(
                        f"❌ Execution failed: {result['execution_error']}", fg="red"
                    )
                    sys.exit(1)

                if compare:
                    if result.get("comparison_match"):
                        click.secho("✅ Replay passed - outputs match!", fg="green")
                    else:
                        click.secho(
                            "⚠️  Replay finished but outputs don't match", fg="yellow"
                        )
                        sys.exit(1)
                else:
                    click.secho("✅ Replay completed", fg="green")
            else:
                click.secho(
                    "✅ Dump loaded successfully (execution skipped)", fg="green"
                )

        except Exception as e:
            click.secho(f"❌ Replay failed: {e}", fg="red")
            if verbose:
                import traceback

                click.echo(traceback.format_exc())
            sys.exit(1)

    else:
        click.secho(f"Replaying sequence from: {dump_dir}", fg="cyan")

        try:
            results = replay_sequence(str(dump_path), device=device)

            if not results:
                click.secho("No dumps found in directory", fg="yellow")
                return

            click.secho(f"\nFound {len(results)} dumps:", fg="yellow")

            passed = 0
            failed = 0

            for i, res in enumerate(results):
                func_name = res.get("metadata", {}).get("function_name", "unknown")

                if "error" in res:
                    click.secho(f"  [{i + 1}] {func_name}: ❌ Error", fg="red")
                    failed += 1
                elif res.get("comparison_match"):
                    click.secho(f"  [{i + 1}] {func_name}: ✅ Passed", fg="green")
                    passed += 1
                else:
                    click.secho(f"  [{i + 1}] {func_name}: ⚠️  Mismatch", fg="yellow")
                    failed += 1

            click.secho(f"\nSummary: {passed} passed, {failed} failed", fg="white")

            if failed > 0:
                sys.exit(1)

        except Exception as e:
            click.secho(f"❌ Replay failed: {e}", fg="red")
            sys.exit(1)


@cli.command("guard-run")
@click.option(
    "--mode",
    type=click.Choice(["tail", "head"]),
    default="tail",
    show_default=True,
    help="Which side of the allocation should use the unmapped guard page",
)
@click.option(
    "--log-allocations",
    is_flag=True,
    help="Print guarded alloc/free events to stderr in the child process",
)
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def guard_run_cmd(mode: str, log_allocations: bool, command: tuple[str, ...]):
    """Run a child command with the MATE guard allocator installed at startup."""
    if not command:
        raise click.UsageError("guard-run requires a command after '--'")

    from mate.memory_debug import GuardAllocatorConfig, run_guarded_command

    exit_code = run_guarded_command(
        list(command),
        GuardAllocatorConfig(mode=mode, log_allocations=log_allocations),
    )
    _exit_with_subprocess_status(exit_code)


def _exit_with_subprocess_status(returncode: int) -> None:
    if returncode >= 0:
        raise SystemExit(returncode)

    signum = -returncode
    try:
        signal.signal(signum, signal.SIG_DFL)
    except (OSError, RuntimeError, ValueError):
        pass

    # Mirror signal termination instead of converting it to a shell exit code.
    os.kill(os.getpid(), signum)
    raise SystemExit(128 + signum)


@cli.command("list-dumps")
@click.argument("dump_root", required=False, default="mate_dumps")
@click.option("--details", is_flag=True, help="Show detailed information")
def list_dumps(dump_root: str, details: bool):
    """List all Level 10 dumps in the specified directory."""
    root_path = Path(dump_root)

    if not root_path.exists():
        click.secho(f"Directory not found: {dump_root}", fg="red")
        return

    dumps = []
    for item in root_path.iterdir():
        if item.is_dir() and (item / "metadata.jsonl").exists():
            dumps.append(item)

    if not dumps:
        click.secho(f"No dumps found in {dump_root}", fg="yellow")
        return

    dumps.sort()
    click.secho(f"Found {len(dumps)} dumps in {dump_root}:", fg="cyan")

    for dump_path in dumps:
        try:
            with open(dump_path / "metadata.jsonl") as f:
                lines = f.readlines()
                if lines:
                    metadata = json.loads(lines[-1])
                    func_name = metadata.get("function_name", "unknown")
                    status = metadata.get("execution_status", "unknown")
                    timestamp = metadata.get("timestamp", "unknown")
                    status_color = "green" if status == "completed" else "red"

                    if details:
                        click.echo()
                        click.secho(f"  {dump_path.name}", fg="cyan", bold=True)
                        print_kv("    Function", func_name, key_color="white")
                        print_kv(
                            "    Status",
                            status,
                            key_color="white",
                            value_color=status_color,
                        )
                        print_kv("    Timestamp", timestamp, key_color="white")

                        tensor_info = metadata.get("tensor_info", {})
                        input_keys = tensor_info.get("input_tensor_keys", [])
                        output_keys = tensor_info.get("output_tensor_keys", [])
                        print_kv("    Inputs", str(len(input_keys)), key_color="white")
                        print_kv(
                            "    Outputs", str(len(output_keys)), key_color="white"
                        )
                    else:
                        status_str = click.style(status, fg=status_color)
                        click.echo(f"  {dump_path.name}: {func_name} [{status_str}]")

        except Exception as e:
            click.secho(f"  {dump_path.name}: Error reading metadata - {e}", fg="red")


@cli.command("check")
def check_cmd():
    """Check MATE installation and environment."""
    print_header("MATE Installation Check")

    errors = []
    warnings = []

    if sys.version_info < (3, 9):
        errors.append("Python 3.9+ required")
    else:
        click.secho("✓ Python version", fg="green")

    tvm_ffi_status = _get_tvm_ffi_status()
    if tvm_ffi_status["installed"] and tvm_ffi_status["musa_enabled"]:
        click.secho("✓ MUSA-enabled apache-tvm-ffi installed", fg="green")
        click.secho(f"  Version: {tvm_ffi_status['version']}", fg="cyan")
    else:
        click.secho("✗ TVM-FFI MUSA build check failed", fg="red", bold=True)
        errors.append(
            f"{tvm_ffi_status['status_message']} {tvm_ffi_status['install_hint']}"
        )

    try:
        import torch  # noqa: F401

        click.secho("✓ PyTorch installed", fg="green")

        try:
            import torch_musa  # noqa: F401

            if torch_musa.is_available():
                click.secho("✓ MUSA available", fg="green")
                click.secho(f"  Devices: {torch_musa.device_count()}", fg="cyan")
            else:
                warnings.append("MUSA not available (CPU mode only)")
        except ImportError:
            errors.append("torch_musa not installed")
    except ImportError:
        errors.append("PyTorch not installed")

    try:
        import mate

        click.secho("✓ MATE installed", fg="green")
        click.secho(f"  Version: {mate.__version__}", fg="cyan")

        mate_path = Path(mate.__file__).parent
        aot_dir = mate_path / "data" / "aot"
        if aot_dir.exists() and list(aot_dir.rglob("*.so")):
            click.secho("✓ AOT libraries found", fg="green")
        else:
            warnings.append("AOT libraries not found (JIT mode only)")

        print_header("Wrapper Packages")
        expected_wrapper_version = _wrapper_compat_version(mate.__version__)
        wrapper_statuses = _get_wrapper_package_statuses(
            mate.__version__, __git_version__
        )
        for status in wrapper_statuses:
            display = status["display"]
            if not status["installed"]:
                click.secho(f"⚠ {display} not installed", fg="yellow")
                warnings.append(f"{display} wrapper not installed")
                continue

            if status["errors"]:
                click.secho(f"✗ {display} mismatch", fg="red", bold=True)
                click.secho(f"  Version: {status['version']}", fg="cyan")
                click.secho(
                    "  Expected MATE-compatible version: "
                    f"{expected_wrapper_version} (dev suffix ignored)",
                    fg="cyan",
                )
                click.secho(
                    f"  Wrapper commit: {_short_commit(status['git_version'])}",
                    fg="cyan",
                )
                click.secho(
                    f"  MATE commit: {_short_commit(__git_version__)}",
                    fg="cyan",
                )
                for error in status["errors"]:
                    errors.append(f"{display}: {error}")
                    click.secho(f"  - {error}", fg="red")
                continue

            click.secho(f"✓ {display} matches MATE", fg="green")
            click.secho(f"  Version: {status['version']}", fg="cyan")
            click.secho(f"  Commit: {_short_commit(status['git_version'])}", fg="cyan")

    except ImportError as exc:
        errors.append(f"MATE import failed: {exc}")

    print_header("Summary")
    if errors:
        click.secho(f"❌ {len(errors)} errors:", fg="red")
        for error in errors:
            click.secho(f"  - {error}", fg="red")

    if warnings:
        click.secho(f"⚠️  {len(warnings)} warnings:", fg="yellow")
        for warning in warnings:
            click.secho(f"  - {warning}", fg="yellow")

    if not errors and not warnings:
        click.secho("✅ All checks passed!", fg="green")
    elif not errors:
        click.secho("\n⚠️  Installation OK with warnings", fg="yellow")
    else:
        click.secho("\n❌ Installation has errors", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    cli()
