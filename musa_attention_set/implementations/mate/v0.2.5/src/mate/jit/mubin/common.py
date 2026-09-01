from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class Arch:
    major: int
    minor: int


MP31_ARCH = Arch(major=3, minor=1)


def get_asm_dtype_from_torch_dtype(
    tensor_dtype: torch.dtype,
    pack_bits: int = 8,
) -> str:
    if pack_bits == 4:
        if tensor_dtype != torch.int8:
            raise ValueError("4-bit packed ASM dtype requires torch.int8 storage")
        return "int4"
    if pack_bits != 8:
        raise ValueError(f"pack_bits must be 4 or 8, got {pack_bits}")

    return {
        torch.float16: "half",
        torch.bfloat16: "bfloat16",
        torch.float32: "float",
        torch.float8_e4m3fn: "fp8_e4m3",
        torch.float8_e5m2: "fp8_e5m2",
        torch.int8: "int8",
    }[tensor_dtype]


def check_tensor(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.device != b.device:
        raise ValueError("all tensors must be on the same device")


def check_tensor_same_device(tensors: Sequence[torch.Tensor]) -> None:
    if not tensors:
        return
    reference = tensors[0]
    for tensor in tensors[1:]:
        check_tensor(reference, tensor)


def check_musa(tensor: torch.Tensor) -> None:
    if tensor.device.type != "musa":
        raise ValueError("tensor must be on MUSA device")


def check_shape(tensor: torch.Tensor, shape: Sequence[int]) -> None:
    if tensor.shape is None:
        raise ValueError("tensor shape must be available")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"tensor shape must be {tuple(shape)}, got {tuple(tensor.shape)}"
        )


def check_type(tensor: torch.Tensor, expected_type: torch.dtype) -> None:
    if tensor.dtype != expected_type:
        raise ValueError(f"tensor dtype must be {expected_type}, got {tensor.dtype}")


def check_contiguous(tensor: torch.Tensor, dim: Optional[int] = None) -> None:
    if tensor.shape is None:
        raise ValueError("tensor shape must be available")

    if dim is not None:
        ndim = len(tuple(tensor.shape))
        dim = dim + ndim if dim < 0 else dim
        if dim < 0 or dim >= ndim:
            raise ValueError(f"dim must be in range [{-ndim}, {ndim}), got {dim}")
        if tensor.shape[dim] <= 1:
            return
        if tensor.stride(dim) != 1:
            raise ValueError(f"tensor must be contiguous at dim {dim}")
        return

    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
