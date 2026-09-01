from typing import Optional, Tuple, Union

import torch

# DeepGEMM spells packed E2M1 as `kPackedFP4`, which is still `torch::kInt8`;
# upstream plans to move it to `torch::kFloat4_e2m1fn_x2` (csrc/utils/math.hpp:10).
_PACKED_FP4_DTYPE = torch.int8


def _is_packed_fp4(t: torch.Tensor) -> bool:
    return t.dtype == _PACKED_FP4_DTYPE


def _ceil_to_ue8m0_code(sf: torch.Tensor) -> torch.Tensor:
    """Biased exponent of ``sf`` rounded up to the next power of two.

    Must stay bit-identical to DeepGEMM's `ceil_to_ue8m0`, so this works on the
    float32 exponent field and bumps it whenever any mantissa bit is set. Computing
    it as ``ceil(log2(sf))`` instead rounds down for values just above a power of
    two, which would let the FP4 payload overflow.
    """
    bits = sf.abs().contiguous().view(torch.int32)
    return (((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()).clamp(1, 254)


def _prepare_m_grouped_w4a8_operands(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    recipe: Optional[Tuple[int, int, int]],
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
]:
    """Validate and adapt packed-FP4 B operands for Mate grouped W4A8."""
    a_is_fp8_e4m3 = a[0].dtype == torch.float8_e4m3fn
    if not a_is_fp8_e4m3:
        raise ValueError("Mate W4A8 requires FP8 E4M3 A and packed FP4 B")

    if isinstance(b[1], tuple):
        if recipe is not None or (recipe_a, recipe_b) not in (
            (None, None),
            ((1, -1), (1, 32)),
        ):
            raise ValueError(
                "Mate W4A8 scales require recipe_a=(1, -1) and recipe_b=(1, 32)"
            )
        return a, (b[0], b[1])

    if recipe is not None or recipe_a != (1, 128) or recipe_b != (1, 32):
        raise ValueError(
            "DeepGEMM FP4 scales require recipe_a=(1, 128) and recipe_b=(1, 32)"
        )
    if b[0].shape[-1] * 2 != 128 or a[1].shape[-1] != 1:
        raise NotImplementedError(
            "Mate W4A8 has one A scale per row; DeepGEMM per-128-K "
            "A scales are losslessly supported only when K=128"
        )
    a_scale_is_fp32 = a[1].dtype == torch.float32
    b_scale_is_fp32 = b[1].dtype == torch.float32
    if not a_scale_is_fp32 or not b_scale_is_fp32:
        raise NotImplementedError("Mate W4A8 only adapts DeepGEMM FP32 logical scales")

    scale_a = (_ceil_to_ue8m0_code(a[1]) << 23).view(torch.float32)
    residual_e8m0 = _ceil_to_ue8m0_code(b[1]).to(torch.uint8)
    epilogue_fp32 = torch.ones(b[1].shape[:-1], dtype=torch.float32, device=b[1].device)
    return (a[0], scale_a), (b[0], (residual_e8m0, epilogue_fp32))
