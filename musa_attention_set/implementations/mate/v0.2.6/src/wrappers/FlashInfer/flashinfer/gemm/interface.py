from typing import Literal, Optional, Tuple

import torch

from mate._backend import resolve_backend
from mate.deep_gemm import (
    m_grouped_fp8_gemm_nt_contiguous as _mate_group_deepgemm_fp8_nt_groupwise,
)
from mate.deep_gemm import (
    m_grouped_fp8_gemm_nt_masked as _mate_batch_deepgemm_fp8_nt_groupwise,
)
from mate.gemm import bmm as _mate_bmm


def bmm_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal["auto", "mudnn"] = "auto",
) -> torch.Tensor:
    r"""Perform FlashInfer-compatible 16-bit BMM on MUSA.

    This function computes ``A @ B`` for each batch and stores the result in
    the requested output dtype. FP16 and BF16 inputs are supported.

    Parameters
    ----------
    A : torch.Tensor
        Input A with shape ``(batch, m, k)`` and FP16 or BF16 dtype.
    B : torch.Tensor
        Input B with shape ``(batch, k, n)`` and the same dtype as A.
    out : Optional[torch.Tensor]
        Preallocated output with shape ``(batch, m, n)``. When provided, the
        same tensor is returned.
    out_dtype : torch.dtype
        Output dtype. BF16, FP16, and FP32 are supported. Default is BF16.
    backend : Literal["auto", "mudnn"]
        Backend selector. ``"auto"`` selects muDNN.

    Returns
    -------
    torch.Tensor
        Output with shape ``(batch, m, n)``.
    """
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("A and B must be three-dimensional tensors")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("out_dtype must be bfloat16, float16, or float32")

    expected_shape = (A.shape[0], A.shape[1], B.shape[2])
    if out is not None:
        if out.shape != expected_shape:
            raise ValueError(
                f"Output shape mismatch. Expected {expected_shape}, got {out.shape}."
            )
        if out.device != A.device:
            raise ValueError(
                f"Output device mismatch. Expected {A.device}, got {out.device}."
            )
        if out.dtype != out_dtype:
            raise ValueError(
                f"Output dtype mismatch. Expected {out_dtype}, got {out.dtype}."
            )

    return _mate_bmm(
        A,
        B.transpose(-2, -1),
        out,
        trans_a=False,
        trans_b=True,
        out_dtype=out_dtype,
        backend=backend,
    )


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["auto", "mudnn"] = "auto",
) -> torch.Tensor:
    r"""Perform FlashInfer-compatible FP8 BMM with scaling on MUSA.

    This function computes ``A @ B`` for each batch, applies the FP32 scaling
    factors for A and B, and stores the result in BF16 or FP16.

    Parameters
    ----------
    A : torch.Tensor
        FP8 E4M3 or E5M2 input A with shape ``(batch, m, k)``.
    B : torch.Tensor
        FP8 E4M3 or E5M2 input B with shape ``(batch, k, n)``.
    A_scale : torch.Tensor
        FP32 tensorwise scalar or rank-3 channelwise scaling factors for A.
    B_scale : torch.Tensor
        FP32 tensorwise scalar or rank-3 channelwise scaling factors for B.
    dtype : torch.dtype
        Output dtype. BF16 and FP16 are supported.
    out : Optional[torch.Tensor]
        Preallocated output with shape ``(batch, m, n)``. When provided, the
        same tensor is returned.
    backend : Literal["auto", "mudnn"]
        Backend selector. ``"auto"`` selects muDNN.

    Returns
    -------
    torch.Tensor
        Output with shape ``(batch, m, n)``.
    """
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"Unsupported output dtype: {dtype}. "
            "Only torch.bfloat16 and torch.float16 are supported for FP8 GEMM operations."
        )

    recipe_a = (1, -1) if A_scale.ndim == 3 else (-1, -1)
    recipe_b = (1, -1) if B_scale.ndim == 3 else (-1, -1)
    return _mate_bmm(
        A,
        B.transpose(-2, -1),
        out,
        trans_a=False,
        trans_b=True,
        scale_a=A_scale,
        scale_b=B_scale,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        out_dtype=dtype,
        backend=backend,
    )


def gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[Literal["MN", "K"]] = None,
    mma_sm: int = 1,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    backend: Literal["auto", "mudnn", "mubin"] = "auto",
    output_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Perform FlashInfer-compatible groupwise FP8 NT GEMM on MUSA.

    This function computes the matrix product of two FP8 tensors, applies
    groupwise scaling, and stores the result in the requested output dtype.
    It forms batch-size-one views and dispatches through MATE BMM.

    Parameters
    ----------
    a : torch.Tensor
        Row-major FP8 input A with shape ``(m, k)``.
    b : torch.Tensor
        Row-major FP8 input B with shape ``(n, k)``. The operation computes
        ``a @ b.T``.
    a_scale : torch.Tensor
        FP32 scaling factors for A. Shape is ``(m, k_blocks)`` for K-major
        scales or ``(k_blocks, m)`` for MN-major scales. It must be contiguous.
    b_scale : torch.Tensor
        FP32 scaling factors for B. Shape is ``(n_blocks, k_blocks)`` for
        K-major scales or ``(k_blocks, n_blocks)`` for MN-major scales. A
        non-scalar tensor must be contiguous.
    scale_major_mode : Optional[Literal["MN", "K"]]
        Common scale layout. None defaults to ``"K"``.
    mma_sm : int
        MMA configuration. MUSA currently supports 1. Default is 1.
    scale_granularity_mnk : Tuple[int, int, int]
        Scale granularity ``(m, n, k)``. Default is ``(1, 128, 128)``.
    out : Optional[torch.Tensor]
        Preallocated output with shape ``(m, n)``. Its dtype takes precedence
        over ``out_dtype`` and the same tensor is returned.
    out_dtype : Optional[torch.dtype]
        Output dtype when ``out`` is omitted. Without ``output_scale``, BF16
        and FP16 are supported and the default is BF16. With ``output_scale``,
        FP8 E4M3 is required and selected by default.
    backend : Literal["auto", "mudnn", "mubin"]
        Backend selector. Without ``output_scale``, ``"auto"`` selects muDNN.
        With ``output_scale``, ``"auto"`` selects MUBIN.
    output_scale : Optional[torch.Tensor]
        FP32 scales for FP8 E4M3 output. Providing this MUSA extension selects
        the MUBIN output path.

    Returns
    -------
    torch.Tensor
        Output with shape ``(m, n)``.
    """
    if output_scale is None:
        backend = resolve_backend(
            backend, supported=("mudnn",), allow_auto=True, default="auto"
        )
        if backend == "auto":
            backend = "mudnn"
        supported_out_dtypes = (torch.bfloat16, torch.float16)
        default_out_dtype = torch.bfloat16
        out_dtype_error = "Only bf16 and fp16 are supported for out_type!"
    else:
        backend = resolve_backend(
            backend, supported=("mubin",), allow_auto=True, default="auto"
        )
        if backend == "auto":
            backend = "mubin"
        supported_out_dtypes = (torch.float8_e4m3fn,)
        default_out_dtype = torch.float8_e4m3fn
        out_dtype_error = "fp8_output only supports e4m3 now"

    effective_out_dtype = out.dtype if out is not None else out_dtype
    if effective_out_dtype is None:
        effective_out_dtype = default_out_dtype
    if effective_out_dtype not in supported_out_dtypes:
        raise ValueError(out_dtype_error)

    if scale_major_mode is None:
        scale_major_mode = "K"
    elif scale_major_mode not in ("K", "MN"):
        raise ValueError("scale_major_mode must be either 'K' or 'MN'")

    if mma_sm != 1:
        print("Warning: only mma_sm=1 is supported now, set mma_sm=1")
    if not a_scale.is_contiguous():
        raise ValueError("a_scale must be contiguous")
    if b_scale.ndim != 0 and not b_scale.is_contiguous():
        raise ValueError("b_scale must be contiguous")

    out_batched = out.unsqueeze(0) if out is not None else None
    scale_a_batched = a_scale if a_scale.ndim == 0 else a_scale.unsqueeze(0)
    scale_b_batched = b_scale if b_scale.ndim == 0 else b_scale.unsqueeze(0)
    scale_out_batched = output_scale.unsqueeze(0) if output_scale is not None else None
    result = _mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        out_batched,
        trans_a=False,
        trans_b=True,
        scale_a=scale_a_batched,
        scale_b=scale_b_batched,
        scale_out=scale_out_batched,
        recipe_a=(scale_granularity_mnk[0], scale_granularity_mnk[2]),
        recipe_b=(scale_granularity_mnk[1], scale_granularity_mnk[2]),
        out_dtype=effective_out_dtype,
        fixed_scale_layout=scale_major_mode == "MN",
        backend=backend,
    )
    return out if out is not None else result.squeeze(0)


def group_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Perform contiguous grouped FP8 NT GEMM.

    For each valid row ``i``, this function computes
    ``out[i] = a[i] @ b[m_indices[i]].T``. Rows for each expert must occupy a
    contiguous, 128-row-aligned region. Entries with ``m_indices == -1`` are
    padding and their output values are unspecified.

    Parameters
    ----------
    a : torch.Tensor
        K-major E4M3 input with shape ``(m, k)``.
    b : torch.Tensor
        K-major E4M3 expert weights with shape ``(num_groups, n, k)``.
    a_scale : torch.Tensor
        FP32 K-major scales for ``a`` with shape ``(m, k // 128)``.
    b_scale : torch.Tensor
        FP32 K-major scales for ``b`` with shape
        ``(num_groups, n // 128, k // 128)``.
    m_indices : torch.Tensor
        Contiguous int32 expert indices with shape ``(m,)``. Use ``-1`` for
        padded rows.
    scale_granularity_mnk : Tuple[int, int, int]
        Scale granularity ``(m, n, k)``. Defaults to ``(1, 128, 128)``.
    out : Optional[torch.Tensor]
        Preallocated BF16 output with shape ``(m, n)``. When provided, this
        exact tensor is returned.
    out_dtype : Optional[torch.dtype]
        Output dtype used only when ``out`` is omitted. The supported and
        default dtype is BF16. Ignored when ``out`` is provided.

    Returns
    -------
    torch.Tensor
        BF16 output with shape ``(m, n)``.
    """
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("a must be rank 2 and b must be rank 3")
    if a_scale.ndim != 2 or b_scale.ndim != 3:
        raise ValueError("a_scale must be rank 2 and b_scale must be rank 3")
    if m_indices.ndim != 1 or m_indices.dtype != torch.int32:
        raise ValueError("m_indices must be a rank-1 int32 tensor")
    if not m_indices.is_contiguous():
        raise ValueError("m_indices must be contiguous")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("a and b must have float8_e4m3fn dtype")
    if a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32:
        raise ValueError("a_scale and b_scale must have float32 dtype")

    expected_shape = (a.shape[0], b.shape[1])
    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        if out_dtype != torch.bfloat16:
            raise ValueError("out_dtype must be bfloat16")
        out = torch.empty(expected_shape, dtype=out_dtype, device=a.device)
    else:
        if out.shape != expected_shape:
            raise ValueError(
                f"Output shape mismatch. Expected {expected_shape}, got {tuple(out.shape)}."
            )
        if out.device != a.device:
            raise ValueError(
                f"Output device mismatch. Expected {a.device}, got {out.device}."
            )
        if out.dtype != torch.bfloat16:
            raise ValueError("out must have bfloat16 dtype")

    _mate_group_deepgemm_fp8_nt_groupwise(
        (a, a_scale),
        (b, b_scale),
        out,
        m_indices,
        recipe=scale_granularity_mnk,
    )
    return out


def batch_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Perform masked batched FP8 NT GEMM.

    For each expert ``g``, this function computes
    ``out[g, :masked_m[g]] = a[g, :masked_m[g]] @ b[g].T``. Output rows at or
    beyond ``masked_m[g]`` are unspecified.

    Parameters
    ----------
    a : torch.Tensor
        K-major E4M3 input with shape ``(num_groups, max_m, k)``.
    b : torch.Tensor
        K-major E4M3 expert weights with shape ``(num_groups, n, k)``.
    a_scale : torch.Tensor
        FP32 K-major scales for ``a`` with shape
        ``(num_groups, max_m, k // 128)``.
    b_scale : torch.Tensor
        FP32 K-major scales for ``b`` with shape
        ``(num_groups, n // 128, k // 128)``.
    masked_m : torch.Tensor
        Contiguous int32 valid-row counts with shape ``(num_groups,)``.
    expected_m : int
        Host-side expected row-count hint used for kernel selection.
    scale_granularity_mnk : Tuple[int, int, int]
        Scale granularity ``(m, n, k)``. Defaults to ``(1, 128, 128)``.
    out : Optional[torch.Tensor]
        Preallocated BF16 output with shape ``(num_groups, max_m, n)``. When
        provided, this exact tensor is returned.
    out_dtype : Optional[torch.dtype]
        Output dtype used only when ``out`` is omitted. The supported and
        default dtype is BF16. Ignored when ``out`` is provided.

    Returns
    -------
    torch.Tensor
        BF16 output with shape ``(num_groups, max_m, n)``.
    """
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a and b must be rank 3")
    if a_scale.ndim != 3 or b_scale.ndim != 3:
        raise ValueError("a_scale and b_scale must be rank 3")
    if masked_m.ndim != 1 or masked_m.dtype != torch.int32:
        raise ValueError("masked_m must be a rank-1 int32 tensor")
    if not masked_m.is_contiguous():
        raise ValueError("masked_m must be contiguous")
    if expected_m <= 0:
        raise ValueError("expected_m must be greater than 0")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("a and b must have float8_e4m3fn dtype")
    if a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32:
        raise ValueError("a_scale and b_scale must have float32 dtype")

    expected_shape = (a.shape[0], a.shape[1], b.shape[1])
    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        if out_dtype != torch.bfloat16:
            raise ValueError("out_dtype must be bfloat16")
        out = torch.empty(expected_shape, dtype=out_dtype, device=a.device)
    else:
        if out.shape != expected_shape:
            raise ValueError(
                f"Output shape mismatch. Expected {expected_shape}, got {tuple(out.shape)}."
            )
        if out.device != a.device:
            raise ValueError(
                f"Output device mismatch. Expected {a.device}, got {out.device}."
            )
        if out.dtype != torch.bfloat16:
            raise ValueError("out must have bfloat16 dtype")

    _mate_batch_deepgemm_fp8_nt_groupwise(
        (a, a_scale),
        (b, b_scale),
        out,
        masked_m,
        expected_m,
        recipe=scale_granularity_mnk,
    )
    return out


__all__ = [
    "batch_deepgemm_fp8_nt_groupwise",
    "bmm_bf16",
    "bmm_fp8",
    "gemm_fp8_nt_groupwise",
    "group_deepgemm_fp8_nt_groupwise",
]
