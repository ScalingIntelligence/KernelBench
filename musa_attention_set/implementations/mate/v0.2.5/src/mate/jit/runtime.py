from __future__ import annotations

from collections.abc import Sequence

import torch


def ffi_to_torch(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple):
        return tuple(ffi_to_torch(item) for item in value)
    if isinstance(value, list):
        return [ffi_to_torch(item) for item in value]
    if isinstance(value, Sequence):
        return [ffi_to_torch(item) for item in value]
    return torch.from_dlpack(value)
