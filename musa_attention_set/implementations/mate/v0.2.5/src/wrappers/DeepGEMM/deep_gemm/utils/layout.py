import functools
import os

import torch

__all__ = [
    "get_mk_alignment_for_contiguous_layout",
    "get_col_major_tma_aligned_tensor",
    "get_mn_major_tma_aligned_tensor",
]

_MK_ALIGNMENT_ENV = "MATE_DEEPGEMM_MK_ALIGNMENT"
_VALID_MK_ALIGNMENTS = (128, 256)


@functools.cache
def get_mk_alignment_for_contiguous_layout() -> int:
    """Return the M-axis alignment requirement for contiguous grouped GEMM.

    Each expert segment in a contiguous-layout grouped GEMM must have its
    token count padded to a multiple of this value before being passed to
    m_grouped_{fp8,bf16}_gemm_nt_contiguous.

    Defaults to 128 and can be overridden with MATE_DEEPGEMM_MK_ALIGNMENT.
    The value is resolved once per process and cached.
    """
    value = os.environ.get(_MK_ALIGNMENT_ENV)
    if value is None:
        return 128

    value = value.strip()
    if not value:
        raise ValueError(
            f"{_MK_ALIGNMENT_ENV} must be one of {_VALID_MK_ALIGNMENTS}, "
            "got an empty value"
        )

    try:
        alignment = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{_MK_ALIGNMENT_ENV} must be one of {_VALID_MK_ALIGNMENTS}, got {value!r}"
        ) from exc

    if alignment not in _VALID_MK_ALIGNMENTS:
        raise ValueError(
            f"{_MK_ALIGNMENT_ENV} must be one of {_VALID_MK_ALIGNMENTS}, got {value!r}"
        )

    return alignment


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Return a column-major, TMA-aligned view of a scale tensor.

    This function returns a contiguous copy of x without reordering data.

    Callers porting DeepGEMM code should call this on the LHS FP32 scale
    tensor (shape [m, ceil(k/128)]) before packaging it as a (fp8, scale)
    tuple — the call is a no-op here but keeps the call-site portable.
    """
    return x.contiguous()


get_mn_major_tma_aligned_tensor = get_col_major_tma_aligned_tensor
