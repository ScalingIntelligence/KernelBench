"""MUSA debug language helpers."""

from __future__ import annotations

from tilelang.cuda.language.print import print as print
from tilelang.musa.debug import device_assert as device_assert

__all__ = ["device_assert", "print"]
