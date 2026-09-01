import torch
from typing import Optional, Tuple

from mate.deep_gemm import (
    fp8_einsum as mate_fp8_einsum,
    fp8_mqa_logits as mate_fp8_mqa_logits,
    fp8_paged_mqa_logits as mate_fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata as mate_get_paged_mqa_logits_metadata,
    tf32_hc_prenorm_gemm as mate_tf32_hc_prenorm_gemm,
)
from mate.gemm import (
    bmm_fp16,
    gemm_fp8_nt_groupwise,
    masked_moe_gemm_16bit,
    masked_moe_gemm_8bit,
    ragged_k_moe_gemm_8bit,
    ragged_m_moe_gemm_16bit,
    ragged_m_moe_gemm_8bit,
)
from .utils import get_mk_alignment_for_contiguous_layout


def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
):
    _ = compiled_dims

    if a.dim() != 2 or b.dim() != 2 or d.dim() != 2:
        raise ValueError("bf16_gemm_nt expects 2D a, b and d tensors")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("bf16_gemm_nt expects bf16 a and b tensors")
    if d.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("bf16_gemm_nt expects bf16 or fp32 d tensor")
    if a.device != b.device or a.device != d.device:
        raise ValueError("a, b and d must be on the same device")

    m, k = a.shape
    n, kb = b.shape
    if k != kb or tuple(d.shape) != (m, n):
        raise ValueError("bf16_gemm_nt expects a[m,k], b[n,k] and d[m,n]")

    if c is not None:
        if c.dim() != 2 or tuple(c.shape) != (m, n):
            raise ValueError("c must have the same shape as d")
        if c.device != d.device:
            raise ValueError("c must be on the same device as d")
        if d.dtype != torch.float32 or c.dtype != torch.float32:
            raise ValueError("bf16_gemm_nt with c expects fp32 c and d tensors")

    bmm_fp16(
        a.unsqueeze(0),
        b.unsqueeze(0).transpose(-2, -1),
        d.dtype,
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
    )


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
    alignment_m: Optional[int] = None,
    backend: str = "auto",
):
    if alignment_m is None:
        alignment_m = get_mk_alignment_for_contiguous_layout()

    ragged_m_moe_gemm_16bit(
        a,
        b,
        m_indices,
        d,
        alignment_m=alignment_m,
        backend=backend,
    )


def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk",
    enable_overlap: bool = False,
    signal: torch.Tensor = None,
    backend: str = "auto",
):
    res = masked_moe_gemm_16bit(
        a,
        b,
        masked_m,
        d,
        expect_tokens=expected_m,
        enable_overlap=enable_overlap,
        signal=signal,
        backend=backend,
    )

    return res[2:] if enable_overlap else None


def m_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    m_indices: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    alignment_m: Optional[int] = None,
    backend: str = "auto",
):
    if not disable_ue8m0_cast:
        raise Exception("m_grouped_fp8_gemm_nt_contiguous UE8M0 cast is not supported!")
    if alignment_m is None:
        alignment_m = get_mk_alignment_for_contiguous_layout()

    ragged_m_moe_gemm_8bit(
        a,
        b,
        m_indices,
        d,
        scale_granularity_mnk=recipe,
        alignment_m=alignment_m,
        backend=backend,
    )


def m_grouped_fp8_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    enable_overlap: bool = False,
    signal: torch.Tensor = None,
    backend: str = "auto",
):
    if not disable_ue8m0_cast:
        raise Exception("m_grouped_fp8_gemm_nt_masked UE8M0 cast is not supported!")

    res = masked_moe_gemm_8bit(
        a,
        b,
        masked_m,
        d,
        recipe,
        expected_m,
        enable_overlap=enable_overlap,
        signal=signal,
        backend=backend,
    )

    return res[2:] if enable_overlap else None


def k_grouped_fp8_gemm_tn_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: list[int],
    ks_tensor: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
):
    ragged_k_moe_gemm_8bit(
        a,
        b,
        ks_tensor,
        d,
        major_a_mode="M",
        major_b_mode="N",
        scale_granularity_mnk=recipe,
    )


# legacy deepgemm api
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked


def fp8_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
):
    assert c is None, "Not support GEMM with C"

    return gemm_fp8_nt_groupwise(
        a[0], b[0], a[1], b[1], scale_granularity_mnk=recipe, out=d
    )


def fp8_einsum(
    expr: str,
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 128, 128),
):
    return mate_fp8_einsum(expr, a, b, d, c=c, recipe=recipe)


def tf32_hc_prenorm_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    sqr_sum: torch.Tensor,
    num_splits: Optional[int] = None,
):
    return mate_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=num_splits)


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_kv: int, num_mps: int = 0
) -> torch.Tensor:
    r"""Get metadata for paged MQA logits

    Parameters
    ----------
    context_lens: Tensor
        Context lengths of each query, shape ``(batch_size)``
    block_kv: Tensor
        Block size of kv cache, **must be 64 now**.
    num_mps: int
        Number of MP to execute. 0 means use all MPs of the current device

    Returns
    -------
    Tensor
        Schedule metadata, shape ``(num_mps + 1, 2)``
    """
    return mate_get_paged_mqa_logits_metadata(context_lens, block_kv, num_mps)


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    r"""FP8 Paged MQA logits

    Parameters
    ----------
    q: Tensor
        The FP8 query tensor with shape ``(batch_size, next_n, heads, index_dim)``
    fused_kv_cache: Tensor
        The FP8 kv cache with fp32 scale, shape ``(num_blocks, block_size, 1, index_dim + 4)``
    weights: Tensor
        The FP32 weight tensor for each query, shape ``(batch_size * next_n, heads)``
    context_lens: Tensor
        Context lengths tensor, supports two layouts:

        - **1D** ``(batch_size,)`` — all ``next_n`` draft tokens of request ``i`` share the
          same context length ``context_lens[i]``.  The visible KV range for draft token
          ``j`` is implicitly ``[0, context_lens[i] - next_n + j]``.
        - **2D** ``(batch_size, next_n)`` — each draft token has an independent context
          length ``context_lens[i, j]``, with visible KV range ``[0, context_lens[i, j] - 1]``.
          Useful for tree-based speculative decoding (e.g. Medusa / EAGLE) where tokens
          on different branches see different KV prefixes.

        The shape is auto-detected; ``get_paged_mqa_logits_metadata`` must be called with
        the same ``context_lens`` tensor.
    block_table: Tensor
        Block table tensor with shape ``(batch_size, max_blocks)``
    schedule_meta: Tensor
        Schedule metadata tensor with shape ``(num_mps + 1, 2)``, produced by
        :func:`get_paged_mqa_logits_metadata`
    max_context_len: int
        Maximum context length
    clean_logits: bool
        Whether to zero-fill logit positions that are out of the valid KV range

    Returns
    -------
    Tensor
        FP32 logits, shape ``(batch_size * next_n, max_context_len)``
    """
    return mate_fp8_paged_mqa_logits(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
    )


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
) -> torch.Tensor:
    r"""FP8 MQA logits.

    This operator computes MQA (multi-query attention) logits for a query sequence against
    a *non-paged* KV tensor. It supports both full logits and "compressed logits" mode
    (when ``max_seqlen_k > 0``), where the output width is limited to a window size.

    Parameters
    ----------
    q : torch.Tensor
        FP8 query tensor with shape ``(seq_len, heads, head_dim)`` and dtype
        ``torch.float8_e4m3fn``.
    kv : tuple[torch.Tensor, torch.Tensor]
        A tuple ``(kv_fp8, kv_scale)``:
        - ``kv_fp8``: FP8 KV tensor with shape ``(seq_len_kv, head_dim)`` and dtype
          ``torch.float8_e4m3fn``. (MQA uses a single KV head.)
        - ``kv_scale``: FP32 scale tensor with shape ``(seq_len_kv,)`` and dtype
          ``torch.float32``.
    weights : torch.Tensor
        FP32 weight tensor with shape ``(seq_len, heads)`` and dtype ``torch.float32``.
    cu_seq_len_k_start : torch.Tensor
        Per-row valid KV start offsets (inclusive) for each query row, with shape
        ``(seq_len,)`` and dtype ``torch.int32``.
    cu_seq_len_k_end : torch.Tensor
        Per-row valid KV end offsets (exclusive) for each query row, with shape
        ``(seq_len,)`` and dtype ``torch.int32``.
    clean_logits : bool, default=False
        Whether to clean logits outside valid KV range. Must be ``False`` when
        ``max_seqlen_k > 0``.
    max_seqlen_k : int, default=0
        If > 0, enables compressed logits mode. The output width becomes ``max_seqlen_k``
        (a windowed logits range per row). In this mode, ``clean_logits`` must be ``False``.

    Returns
    -------
    torch.Tensor
        FP32 logits tensor with shape:
        - ``(seq_len, seq_len_kv)`` if ``max_seqlen_k == 0``
        - ``(seq_len, max_seqlen_k)`` if ``max_seqlen_k > 0``
        and dtype ``torch.float32``.
    """
    kv_fp8, kv_scale = kv
    if max_seqlen_k > 0 and clean_logits:
        raise ValueError("max_seq_len_k is not supported with clean_logits")
    return mate_fp8_mqa_logits(
        q,
        (kv_fp8, kv_scale),
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        clean_logits,
        int(max_seqlen_k),
    )
