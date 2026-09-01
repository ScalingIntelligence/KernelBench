from __future__ import annotations

from typing import TYPE_CHECKING

from tilelang import _ffi_api
from tilelang._typing import BufferLikeType, BufferLikeTypeTuple
from tvm import tirx

if TYPE_CHECKING:
    from tilelang.layout import Fragment, Layout


def _get_buffer_info(buffer_or_load_or_region: BufferLikeType) -> tuple[tirx.Buffer, list[int], str]:
    if isinstance(buffer_or_load_or_region, tirx.Buffer):
        return buffer_or_load_or_region, list(buffer_or_load_or_region.shape), buffer_or_load_or_region.dtype
    if isinstance(buffer_or_load_or_region, tirx.BufferRegion):
        buf = buffer_or_load_or_region.buffer
        return buf, [r.extent for r in buffer_or_load_or_region.region], buf.dtype
    if isinstance(buffer_or_load_or_region, BufferLikeTypeTuple):
        buf = buffer_or_load_or_region.buffer
        return buf, list(buf.shape), buf.dtype
    raise TypeError(f"Expected BufferLikeType, got {type(buffer_or_load_or_region)}")


def make_mp31_sqmma_shared_ab(
    buffer: BufferLikeType,
    continuity: int | None = None,
    k_major: bool = True,
) -> Layout:
    """Create the MP31 SQMMA shared-memory operand layout."""

    buf, _, _ = _get_buffer_info(buffer)
    if continuity is None:
        continuity = -1
    return _ffi_api.make_mp31_sqmma_shared_ab(buf, int(continuity), bool(k_major))


def make_mp31_sqmma_fragment_c(
    buffer: BufferLikeType,
    warp_m: int,
    warp_n: int,
    inst_shape: tuple[int, int, int] | list[int] | None = None,
) -> Fragment:
    """Create the MP31 SQMMA accumulator fragment layout."""

    _, shape, _ = _get_buffer_info(buffer)
    if not shape:
        raise ValueError("make_mp31_sqmma_fragment_c expects a non-scalar buffer")
    if inst_shape is None:
        inst_shape = []
    return _ffi_api.make_mp31_sqmma_fragment_c(shape, int(warp_m), int(warp_n), list(inst_shape))
