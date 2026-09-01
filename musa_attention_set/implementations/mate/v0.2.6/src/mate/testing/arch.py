from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

MUSA_ARCH_REQUIREMENT_ATTR = "_musa_arch_requirement"
MUSA_ARCH_CHECKER_ATTR = "is_musa_compute_capability_supported"


def _validate_cc(cc: int, name: str) -> int:
    if isinstance(cc, bool) or not isinstance(cc, int):
        raise TypeError(
            f"{name} must be a positive integer compute capability, got {type(cc).__name__}: {cc!r}"
        )
    if cc <= 0:
        raise ValueError(
            f"{name} must be a positive integer compute capability, got {cc!r}"
        )
    return cc


def _validate_ccs(supported_ccs: Iterable[int]) -> frozenset[int]:
    try:
        ccs = list(supported_ccs)
    except TypeError:
        raise TypeError(
            f"supported_ccs must be an iterable of positive integer compute capabilities, got {type(supported_ccs).__name__}"
        ) from None

    if not ccs:
        raise ValueError("supported_ccs must not be empty")

    return frozenset(
        _validate_cc(cc, f"supported_ccs[{idx}]") for idx, cc in enumerate(ccs)
    )


def supported_musa_compute_capability(supported_ccs: Iterable[int]) -> Callable[[F], F]:
    """Mark a test as supported only on the listed MUSA compute capabilities.

    Compute capabilities use ``major * 10 + minor`` encoding, e.g. MP31 is 31.
    """

    ccs = _validate_ccs(supported_ccs)

    def decorator(func: F) -> F:
        def is_supported(cc: int) -> bool:
            return _validate_cc(cc, "cc") in ccs

        setattr(func, MUSA_ARCH_REQUIREMENT_ATTR, {"mode": "allowlist", "ccs": ccs})
        setattr(func, MUSA_ARCH_CHECKER_ATTR, is_supported)
        return func

    return decorator


def requires_musa_compute_capability_ge(min_cc: int) -> Callable[[F], F]:
    """Mark a test as requiring at least the given MUSA compute capability.

    Compute capabilities use ``major * 10 + minor`` encoding, e.g. MP31 is 31.
    """

    min_cc = _validate_cc(min_cc, "min_cc")

    def decorator(func: F) -> F:
        def is_supported(cc: int) -> bool:
            return _validate_cc(cc, "cc") >= min_cc

        setattr(func, MUSA_ARCH_REQUIREMENT_ATTR, {"mode": "ge", "cc": min_cc})
        setattr(func, MUSA_ARCH_CHECKER_ATTR, is_supported)
        return func

    return decorator
