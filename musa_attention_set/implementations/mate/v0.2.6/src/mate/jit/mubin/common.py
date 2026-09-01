from dataclasses import dataclass

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
