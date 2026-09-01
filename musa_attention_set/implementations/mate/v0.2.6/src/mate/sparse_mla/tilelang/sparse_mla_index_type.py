"""Select sparse MLA JIT index width from runtime tensor address spans."""

from dataclasses import replace
from typing import Any

import torch
import tilelang


INT32_ADDRESS_SPACE_BYTES = torch.iinfo(torch.int32).max


def tensor_byte_span(tensor: torch.Tensor) -> int:
    """Return the byte span addressed by a positive-strided tensor view."""

    shape = tensor.shape
    if any(extent == 0 for extent in shape):
        return 0
    element_span = 1
    for extent, stride in zip(shape, tensor.stride()):
        element_span += (extent - 1) * stride
    return element_span * tensor.element_size()


def needs_index_type_promotion(*tensors: torch.Tensor) -> bool:
    """Whether any tensor argument exceeds the signed int32 address space."""

    for tensor in tensors:
        if tensor.untyped_storage().nbytes() <= INT32_ADDRESS_SPACE_BYTES:
            continue
        if tensor_byte_span(tensor) > INT32_ADDRESS_SPACE_BYTES:
            return True
    return False


_JIT_INDEX_PROMOTION_VARIANTS: dict[int, Any] = {}


def _index_promotion_variant(jit_impl):
    key = id(jit_impl)
    variant = _JIT_INDEX_PROMOTION_VARIANTS.get(key)
    if variant is not None:
        return variant

    promote = getattr(jit_impl, "with_index_type_promotion", None)
    if promote is not None:
        variant = promote()
    else:
        pass_configs = dict(jit_impl.pass_configs)
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION] = False
        variant = replace(jit_impl, pass_configs=pass_configs)
    _JIT_INDEX_PROMOTION_VARIANTS[key] = variant
    return variant


def jit_for_tensor_addressing(jit_impl, *tensors: torch.Tensor):
    """Keep int32 indexing unless a runtime tensor requires promotion."""

    if not needs_index_type_promotion(*tensors):
        return jit_impl
    return _index_promotion_variant(jit_impl)
