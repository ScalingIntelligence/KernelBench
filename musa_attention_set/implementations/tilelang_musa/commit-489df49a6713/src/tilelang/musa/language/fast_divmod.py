"""Explicit fast integer division operations for the MUSA dialect."""

from tvm import DataType, tirx
from tvm.tirx import PrimExpr


def _normalize_operands(dividend: PrimExpr, divisor: PrimExpr) -> tuple[PrimExpr, PrimExpr]:
    dividend = tirx.convert(dividend)
    divisor = tirx.convert(divisor)
    if dividend.dtype != divisor.dtype:
        raise TypeError(
            "fast integer division requires dividend and divisor to have the "
            f"same dtype, got {dividend.dtype} and {divisor.dtype}"
        )
    if DataType(dividend.dtype) != DataType("int32"):
        raise TypeError(
            "fast integer division currently supports scalar int32 operands "
            f"only, got {dividend.dtype}"
        )
    return dividend, divisor


def fast_div(dividend: PrimExpr, divisor: PrimExpr) -> PrimExpr:
    """Return ``dividend // divisor`` using explicit MUSA fast division.

    Both operands must be scalar ``int32``. The dividend must be non-negative
    and the divisor must be strictly positive.
    """

    dividend, divisor = _normalize_operands(dividend, divisor)
    return tirx.call_intrin(
        dividend.dtype, tirx.op.Op.get("tl.musa.fast_div"), dividend, divisor
    )


def fast_mod(dividend: PrimExpr, divisor: PrimExpr) -> PrimExpr:
    """Return ``dividend % divisor`` using explicit MUSA fast division."""

    dividend, divisor = _normalize_operands(dividend, divisor)
    return tirx.call_intrin(
        dividend.dtype, tirx.op.Op.get("tl.musa.fast_mod"), dividend, divisor
    )


def fast_divmod(dividend: PrimExpr, divisor: PrimExpr) -> tuple[PrimExpr, PrimExpr]:
    """Return a quotient/remainder pair using one fast quotient calculation."""

    dividend, divisor = _normalize_operands(dividend, divisor)
    return (
        tirx.call_intrin(
            dividend.dtype, tirx.op.Op.get("tl.musa.fast_div"), dividend, divisor
        ),
        tirx.call_intrin(
            dividend.dtype, tirx.op.Op.get("tl.musa.fast_mod"), dividend, divisor
        ),
    )


__all__ = ["fast_div", "fast_mod", "fast_divmod"]
