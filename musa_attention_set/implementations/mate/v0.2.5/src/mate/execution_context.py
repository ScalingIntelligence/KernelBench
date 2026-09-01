from __future__ import annotations

import os
from contextlib import nullcontext
from functools import wraps, lru_cache
from typing import Any, Callable

import torch
from torch._guards import active_fake_mode

from torch._subclasses.fake_tensor import FakeTensorMode

MATE_DRY_RUN_ENV = "MATE_DRY_RUN"


class MateDryRunComplete(RuntimeError):
    """Raised to stop the current dry-run path once compilation is complete."""


@lru_cache()
def is_dry_run_enabled() -> bool:
    return os.environ.get(MATE_DRY_RUN_ENV, "0") == "1"


def maybe_fake_tensor_mode(fake: bool = True):
    """
    One way to populate/pre-compile cache is to use torch fake tensor mode,
    which does not allocate actual GPU tensors but retains tensor shape/dtype
    metadata for cute.compile.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with FakeTensorMode() if fake else nullcontext():
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def is_fake_mode() -> bool:
    return active_fake_mode() is not None


def raise_complete_if_dry_run():
    if is_dry_run_enabled():
        raise MateDryRunComplete


def empty_if_dry_run(
    fn: Callable[..., torch.Tensor],
    *,
    last: bool = False,
    empty_values: Any = None,
) -> Callable[..., torch.Tensor]:
    def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
        if is_fake_mode():
            return empty_values
        else:
            try:
                return fn(*args, **kwargs)
            except MateDryRunComplete:
                assert is_fake_mode()
                if not last:
                    return empty_values
                raise

    return wrapper
