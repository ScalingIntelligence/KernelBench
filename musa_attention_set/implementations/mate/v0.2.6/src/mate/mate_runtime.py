"""Runtime helpers shared by MATE Python APIs and compatibility wrappers."""

import functools
from typing import Optional

import torch

_num_mps: Optional[int] = None


def _device_index(device: Optional[torch.device] = None) -> int:
    if device is None or device.index is None:
        return torch.musa.current_device()
    return device.index


@functools.cache
def _get_physical_num_mps(device_index: int) -> int:
    return torch.musa.get_device_properties(device_index).multi_processor_count


def get_physical_num_mps(device: Optional[torch.device] = None) -> int:
    """Return the physical MUSA MP count for ``device``."""
    return _get_physical_num_mps(_device_index(device))


def get_num_mps() -> int:
    """Return the effective MP count for MATE kernel dispatch."""
    if _num_mps is not None:
        return _num_mps
    return get_physical_num_mps()


def resolve_num_mps(
    device: Optional[torch.device] = None, num_mps: Optional[int] = None
) -> int:
    """Resolve an explicit MP count or fall back to the effective device count."""
    if num_mps is not None and num_mps > 0:
        return num_mps
    if _num_mps is not None:
        return _num_mps
    return get_physical_num_mps(device)


def set_num_mps(num: Optional[int]) -> None:
    """Limit the maximum MP count available to MATE JIT kernels.

    Pass ``None`` to reset to the physical MP count of the target device.
    """
    global _num_mps
    assert num is None or (isinstance(num, int) and num > 0), (
        f"num_mps must be a positive int or None, got {num}"
    )
    _num_mps = num


__all__ = [
    "get_num_mps",
    "get_physical_num_mps",
    "resolve_num_mps",
    "set_num_mps",
]
