from __future__ import annotations

import dataclasses
from functools import cache
import warnings

import os
import hashlib
import pytest
import torch
from pathlib import Path

from mate.execution_context import MateDryRunComplete
from mate.testing.arch import MUSA_ARCH_CHECKER_ATTR, MUSA_ARCH_REQUIREMENT_ATTR


_ENV_ENABLE = "MATE_PYTEST_GUARD_ALLOC"
_ENV_MODE = "MATE_PYTEST_GUARD_MODE"
_ENV_LOG_ALLOCATIONS = "MATE_PYTEST_GUARD_LOG_ALLOCATIONS"
_SKIP_REASON = "guard allocator debug mode does not support MUSA graph capture"


@dataclasses.dataclass(frozen=True)
class _PytestGuardConfig:
    enabled: bool
    mode: str
    log_allocations: bool


@pytest.fixture
def disable_mudnn_tf32():
    mudnn = getattr(torch.backends, "mudnn", None)
    if mudnn is None:
        yield
        return

    original_allow_tf32 = mudnn.allow_tf32
    try:
        mudnn.allow_tf32 = False
        yield
    finally:
        mudnn.allow_tf32 = original_allow_tf32


@pytest.fixture
def enable_musa_tf32():
    musa_matmul = torch.backends.musa.matmul
    mudnn = torch.backends.mudnn
    original_musa_allow_tf32 = musa_matmul.allow_tf32
    original_mudnn_allow_tf32 = mudnn.allow_tf32
    try:
        musa_matmul.allow_tf32 = True
        mudnn.allow_tf32 = True
        yield
    finally:
        mudnn.allow_tf32 = original_mudnn_allow_tf32
        musa_matmul.allow_tf32 = original_musa_allow_tf32


def _parse_bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise pytest.UsageError(f"Invalid boolean value for {name}: {value!r}")


def _resolve_guard_config(config: pytest.Config) -> _PytestGuardConfig:
    enabled_opt = config.getoption("guard_alloc")
    if enabled_opt is None:
        enabled = _parse_bool_env(_ENV_ENABLE, default=False)
    else:
        enabled = enabled_opt

    mode_opt = config.getoption("guard_mode")
    mode = (mode_opt or os.environ.get(_ENV_MODE, "tail")).strip().lower()
    if mode not in {"tail", "head"}:
        raise pytest.UsageError(
            f"Invalid {_ENV_MODE} value: {mode!r}. Expected 'tail' or 'head'."
        )

    log_allocations_opt = config.getoption("guard_log_allocations")
    if log_allocations_opt is None:
        log_allocations = _parse_bool_env(_ENV_LOG_ALLOCATIONS, default=False)
    else:
        log_allocations = log_allocations_opt

    return _PytestGuardConfig(
        enabled=enabled,
        mode=mode,
        log_allocations=log_allocations,
    )


def _format_cc(cc: int) -> str:
    return f"MP{cc}"


def _format_arch_requirement(requirement: dict) -> str:
    if requirement["mode"] == "allowlist":
        ccs = ", ".join(_format_cc(cc) for cc in sorted(requirement["ccs"]))
        return f"one of [{ccs}]"
    if requirement["mode"] == "ge":
        return f">= {_format_cc(requirement['cc'])}"
    return "an unsupported MUSA architecture requirement"


@cache
def _current_musa_compute_capability() -> tuple[int | None, str]:
    try:
        import torch
        import torch_musa  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, f"unable to import torch/torch_musa: {exc}"

    if not hasattr(torch, "musa"):
        return None, "torch.musa is not available"

    try:
        if not torch.musa.is_available():
            return None, "MUSA is not available"
        if torch.musa.device_count() <= 0:
            return None, "no MUSA devices are available"
        device = torch.musa.current_device()
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, f"unable to initialize MUSA: {exc}"

    try:
        major, minor = torch.musa.get_device_capability(device)
    except Exception as capability_exc:
        try:
            properties = torch.musa.get_device_properties(device)
            major, minor = properties.major, properties.minor
        except Exception as properties_exc:  # pragma: no cover - environment dependent
            return (
                None,
                "unable to determine MUSA compute capability: "
                f"{capability_exc}; fallback failed: {properties_exc}",
            )

    try:
        cc = int(major) * 10 + int(minor)
    except (TypeError, ValueError) as exc:
        return None, f"invalid MUSA compute capability ({major!r}, {minor!r}): {exc}"

    return cc, f"current MUSA device {device} has {_format_cc(cc)}"


def _musa_arch_skip_reason(
    requirement: dict, current_cc: int | None, detail: str
) -> str:
    required = _format_arch_requirement(requirement)
    if current_cc is None:
        return f"requires MUSA compute capability {required}; {detail}"
    return (
        f"requires MUSA compute capability {required}; "
        f"detected {_format_cc(current_cc)} ({detail})"
    )


def _is_dry_run_complete(excinfo) -> bool:
    return excinfo is not None and excinfo.errisinstance(MateDryRunComplete)


def _is_musa_oom(excinfo) -> bool:
    if excinfo is None:
        return False

    musa_oom_types = tuple(
        exc
        for exc in (
            getattr(torch, "MusaOutOfMemory", None),
            getattr(getattr(torch, "musa", None), "OutOfMemoryError", None),
        )
        if exc is not None
    )
    if bool(musa_oom_types) and excinfo.errisinstance(musa_oom_types):
        return True

    # TVM-FFI/DLPack allocation failures can surface as a plain MemoryError.
    # Some torch_musa paths surface OOM as RuntimeError: "MUSA error: out of memory".
    if excinfo.errisinstance((MemoryError, RuntimeError)):
        message = str(excinfo.value).lower()
        return "musa" in message and "out of memory" in message

    return False


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("mate-guard")
    group.addoption(
        "--guard-alloc",
        action="store_true",
        dest="guard_alloc",
        default=None,
        help=(
            "Install the MATE guard allocator before collecting tests. "
            f"Can also be enabled with {_ENV_ENABLE}=1."
        ),
    )
    group.addoption(
        "--no-guard-alloc",
        action="store_false",
        dest="guard_alloc",
        default=None,
        help="Disable the MATE guard allocator even if the environment enables it.",
    )
    group.addoption(
        "--guard-mode",
        action="store",
        choices=("tail", "head"),
        default=None,
        help=(
            "Select which side of each allocation should use the unmapped guard page. "
            f"Can also be set with {_ENV_MODE}=tail|head."
        ),
    )
    group.addoption(
        "--guard-log-allocations",
        action="store_true",
        dest="guard_log_allocations",
        default=None,
        help=(
            "Log guard allocator alloc/free events to stderr. "
            f"Can also be enabled with {_ENV_LOG_ALLOCATIONS}=1."
        ),
    )
    group.addoption(
        "--no-guard-log-allocations",
        action="store_false",
        dest="guard_log_allocations",
        default=None,
        help="Disable guard allocator allocation logging.",
    )
    fmha_group = parser.getgroup("mate-fmha-bwd")
    fmha_group.addoption(
        "--fmha-bwd-stress-iters",
        "--dnn-fmha-stress-iters",
        dest="fmha_bwd_stress_iters",
        action="store",
        type=int,
        default=0,
        help=(
            "Run this many extra stress iterations after each "
            "tests/test_fmha_bwd.py correctness case."
        ),
    )
    fmha_group.addoption(
        "--fmha-bwd-stress-mode",
        "--dnn-fmha-stress-mode",
        dest="fmha_bwd_stress_mode",
        action="store",
        choices=("kernel-only", "check"),
        default="kernel-only",
        help=(
            "FMHA BWD stress mode: kernel-only only launches/synchronizes; "
            "check also validates every iteration."
        ),
    )
    fmha_group.addoption(
        "--fmha-bwd-stress-progress-interval",
        "--dnn-fmha-stress-progress-interval",
        dest="fmha_bwd_stress_progress_interval",
        action="store",
        type=int,
        default=0,
        help=(
            "Print FMHA BWD stress progress every N iterations. "
            "0 disables progress output."
        ),
    )
    sparse_mla_group = parser.getgroup("mate-sparse-mla")
    sparse_mla_group.addoption(
        "--sparse-mla-full",
        action="store_true",
        default=False,
        help="Run the large sparse MLA correctness cases.",
    )
    sparse_mla_group.addoption(
        "--sparse-mla-stress-iters",
        action="store",
        type=int,
        default=0,
        help=(
            "Run the persistent-row sparse MLA stress case for N iterations. "
            "0 disables it; enabled runs require at least 10000 iterations."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "guard_incompatible: skip this test when pytest guard allocator mode is active",
    )

    guard_config = _resolve_guard_config(config)
    setattr(config, "_mate_guard_config", guard_config)
    if not guard_config.enabled:
        return

    from mate.memory_debug import GuardAllocatorConfig, install_guard_allocator

    try:
        install_guard_allocator(
            GuardAllocatorConfig(
                mode=guard_config.mode,
                log_allocations=guard_config.log_allocations,
            )
        )
    except RuntimeError as exc:
        raise pytest.UsageError(
            f"Failed to enable the MATE guard allocator before test collection: {exc}"
        ) from exc


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    current_cc, detail = _current_musa_compute_capability()
    guard_config = getattr(config, "_mate_guard_config", None)
    guard_enabled = guard_config is not None and guard_config.enabled
    skip_graph = pytest.mark.skip(reason=_SKIP_REASON)

    for item in items:
        test_func = getattr(item, "obj", None)
        requirement = getattr(test_func, MUSA_ARCH_REQUIREMENT_ATTR, None)
        if requirement is not None:
            checker = getattr(test_func, MUSA_ARCH_CHECKER_ATTR)
            if current_cc is None or not checker(current_cc):
                item.add_marker(
                    pytest.mark.skip(
                        reason=_musa_arch_skip_reason(requirement, current_cc, detail)
                    )
                )

        if not guard_enabled:
            continue

        should_skip = item.get_closest_marker("guard_incompatible") is not None
        callspec = getattr(item, "callspec", None)
        if callspec is not None and callspec.params.get("use_graph", False):
            should_skip = True
        if should_skip:
            item.add_marker(skip_graph)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    """
    Treat selected runtime exceptions as non-failures for the test call phase.
    """
    report = yield

    if call.when == "call" and _is_dry_run_complete(call.excinfo):
        report.outcome = "passed"
        report.longrepr = None
    elif call.when == "call" and _is_musa_oom(call.excinfo):
        warnings.warn(
            # f"MUSA out of memory; skipping test: {call.excinfo.value}",
            "MUSA out of memory; skipping test",
            stacklevel=1,
        )
        report.outcome = "skipped"
        report.longrepr = ("", 0, "Skipped: MUSA out of memory")

    return report


def _get_shard_config():
    shard_total = int(os.environ.get("MATE_PYTEST_SHARD_TOTAL", "1"))
    shard_index = int(os.environ.get("MATE_PYTEST_SHARD_INDEX", "0"))
    mode = os.environ.get("MATE_PYTEST_SHARD_MODE", "file")

    if shard_total < 1:
        raise pytest.UsageError(
            f"MATE_PYTEST_SHARD_TOTAL must be >= 1, got {shard_total}"
        )
    if shard_index < 0 or shard_index >= shard_total:
        raise pytest.UsageError(
            f"PYTEST_SHARD_INDEX must be in the range [0, {shard_total}), got {shard_index}"
        )
    if mode not in ("file", "item"):
        raise pytest.UsageError(
            f"MATE_PYTEST_SHARD_MODE must be either 'file' or 'item', got {mode}"
        )

    return shard_total, shard_index, mode


def _stable_shard_id(key: Path, total: int) -> int:
    if key.name in {"test_fmha.py", "test_fmha_bwd.py"}:
        return total - 1  # FMHA tests always run on the final shard in file mode

    if total == 1:
        return 0

    digest = hashlib.blake2b(key.as_posix().encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (total - 1)


def pytest_ignore_collect(collection_path, config):
    total, index, mode = _get_shard_config()

    # NOTE: Only support file now.
    if total == 1 or mode != "file":
        return None

    path = Path(collection_path)
    if not path.name.startswith("test_") or path.suffix != ".py":
        return None

    try:
        key = path.relative_to(config.rootpath)
    except ValueError:
        key = path

    return _stable_shard_id(key, total) != index
