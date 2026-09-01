from typing import Optional

import torch

from mate.mate_runtime import (
    get_num_mps,
    resolve_num_mps,
    set_num_mps,
)

__all__ = [
    "get_num_sms",
    "resolve_num_sms",
    "set_num_sms",
    "get_tc_util",
    "set_tc_util",
]


_tc_util: float = 1.0


def get_num_sms() -> int:
    """Return the effective SM count using DeepGEMM compatibility terminology."""
    return get_num_mps()


def resolve_num_sms(device: Optional[torch.device] = None) -> int:
    """Return the effective SM count using DeepGEMM compatibility terminology."""
    return resolve_num_mps(device)


def set_num_sms(num: Optional[int]) -> None:
    """Set the effective SM count using DeepGEMM compatibility terminology."""
    set_num_mps(num)


def get_tc_util() -> float:
    """Return the tensor-core utilization ratio used by heuristics.

    mate kernels do not consume it internally.
    """
    return _tc_util


def set_tc_util(ratio: float) -> None:
    """Set the tensor-core utilization ratio used by heuristics.

    get_tc_util(). Has no effect on kernel dispatch on MUSA.
    """
    global _tc_util
    assert 0.0 < ratio <= 1.0, f"tc_util ratio must be in (0, 1], got {ratio}"
    _tc_util = ratio
