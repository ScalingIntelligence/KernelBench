import torch
from typing import Optional, Tuple, Union

from mate.deep_gemm import (
    bf16_gemm_nt as mate_bf16_gemm_nt,
    fp8_einsum as mate_fp8_einsum,
    fp8_gemm_nt as mate_fp8_gemm_nt,
    fp8_gemm_nt_skip_head_mid as mate_fp8_gemm_nt_skip_head_mid,
    fp8_mqa_logits as mate_fp8_mqa_logits,
    fp8_paged_mqa_logits as mate_fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata as mate_get_paged_mqa_logits_metadata,
    tf32_hc_prenorm_gemm as mate_tf32_hc_prenorm_gemm,
)
from mate.gemm import (
    bmm as mate_bmm,
    masked_moe_gemm_16bit,
    masked_moe_gemm_8bit,
    masked_moe_gemm_mixed_dtype,
    ragged_k_moe_gemm_8bit,
    ragged_k_moe_gemm_16bit,
    ragged_m_moe_gemm_16bit,
    ragged_m_moe_gemm_8bit,
    ragged_moe_gemm_mixed_dtype,
)
from .utils import get_mk_alignment_for_contiguous_layout
from .utils.w4a8 import _is_packed_fp4, _prepare_m_grouped_w4a8_operands


def _reject_unsupported_parameters(parameter: object) -> None:
    if parameter is not None:
        raise ValueError("parameter is not supported by Mate; leave it unset")


def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
    backend: str = "auto",
):
    mate_bf16_gemm_nt(
        a,
        b,
        d,
        c=c,
        compiled_dims=compiled_dims,
        backend=backend,
    )


def bf16_gemm_nn(a, b, d, c=None, compiled_dims: str = "nk", backend: str = "auto"):
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=False,
        trans_b=False,
        backend=backend,
    )


def bf16_gemm_tn(a, b, d, c=None, compiled_dims: str = "mn", backend: str = "auto"):
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=True,
        trans_b=False,
        backend=backend,
    )


def bf16_gemm_tt(a, b, d, c=None, compiled_dims: str = "mn", backend: str = "auto"):
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=True,
        trans_b=True,
        backend=backend,
    )


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    compiled_dims: str = "nk",
    use_psum_layout: Optional[bool] = None,
    ensure_zero_padding: Optional[bool] = None,
    expected_m_for_psum_layout: Optional[int] = None,
    alignment_m: Optional[int] = None,
    backend: str = "auto",
):
    _ = compiled_dims
    _reject_unsupported_parameters(use_psum_layout)
    _reject_unsupported_parameters(ensure_zero_padding)
    _reject_unsupported_parameters(expected_m_for_psum_layout)
    if alignment_m is None:
        alignment_m = get_mk_alignment_for_contiguous_layout()

    ragged_m_moe_gemm_16bit(
        a,
        b,
        grouped_layout,
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
    signal: Optional[torch.Tensor] = None,
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


def m_grouped_fp8_fp4_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
    use_psum_layout: Optional[bool] = None,
    ensure_zero_padding: Optional[bool] = None,
    expected_m_for_psum_layout: Optional[int] = None,
    alignment_m: Optional[int] = None,
    backend: str = "auto",
):
    _ = compiled_dims
    _reject_unsupported_parameters(disable_ue8m0_cast)
    _reject_unsupported_parameters(use_psum_layout)
    _reject_unsupported_parameters(ensure_zero_padding)
    _reject_unsupported_parameters(expected_m_for_psum_layout)

    a_is_fp4 = _is_packed_fp4(a[0])
    b_is_fp4 = _is_packed_fp4(b[0])
    a_is_fp8 = a[0].dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    b_is_fp8 = b[0].dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    if a_is_fp4:
        raise NotImplementedError("Mate W4A8 does not support packed FP4 A")
    if a_is_fp8 and b_is_fp4:
        w4a8_a, w4a8_b = _prepare_m_grouped_w4a8_operands(
            a, b, recipe, recipe_a, recipe_b
        )

        ragged_moe_gemm_mixed_dtype(
            w4a8_a,
            w4a8_b,
            grouped_layout,
            d,
            alignment_m=alignment_m,
            mixed_dtype="fp4fp8",
            backend=backend,
            a_quant_recipe=(1, -1),
            b_quant_recipe=(1, 32),
        )
        return None
    if b_is_fp4:
        raise ValueError("Mate W4A8 requires FP8 E4M3 A and packed FP4 B")

    _reject_unsupported_parameters(recipe_a)
    _reject_unsupported_parameters(recipe_b)
    if not a_is_fp8 or not b_is_fp8:
        raise ValueError(
            "m_grouped_fp8_fp4_gemm_nt_contiguous expects FP8 A and B tensors"
        )
    if alignment_m is None:
        alignment_m = get_mk_alignment_for_contiguous_layout()

    ragged_m_moe_gemm_8bit(
        a,
        b,
        grouped_layout,
        d,
        scale_granularity_mnk=recipe,
        alignment_m=alignment_m,
        backend=backend,
    )


def m_grouped_fp8_fp4_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    backend: str = "auto",
):
    _ = compiled_dims
    _reject_unsupported_parameters(disable_ue8m0_cast)

    a_is_fp4 = _is_packed_fp4(a[0])
    b_is_fp4 = _is_packed_fp4(b[0])
    a_is_fp8 = a[0].dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    b_is_fp8 = b[0].dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    if a_is_fp4:
        raise NotImplementedError("Mate W4A8 does not support packed FP4 A")
    if a_is_fp8 and b_is_fp4:
        w4a8_a, w4a8_b = _prepare_m_grouped_w4a8_operands(
            a, b, recipe, recipe_a, recipe_b
        )

        res = masked_moe_gemm_mixed_dtype(
            w4a8_a,
            w4a8_b,
            masked_m,
            d,
            expect_tokens=expected_m,
            enable_overlap=enable_overlap,
            signal=signal,
            mixed_dtype="fp4fp8",
            backend=backend,
            a_quant_recipe=(1, -1),
            b_quant_recipe=(1, 32),
        )
        return res[2:] if enable_overlap else None
    if b_is_fp4:
        raise ValueError("Mate W4A8 requires FP8 E4M3 A and packed FP4 B")

    _reject_unsupported_parameters(recipe_a)
    _reject_unsupported_parameters(recipe_b)
    if not a_is_fp8 or not b_is_fp8:
        raise ValueError("m_grouped_fp8_fp4_gemm_nt_masked expects FP8 A and B tensors")

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
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 128),
    compiled_dims: str = "mn",
    use_psum_layout: Optional[bool] = None,
):
    _ = compiled_dims
    _reject_unsupported_parameters(ks_cpu)
    _reject_unsupported_parameters(use_psum_layout)
    if a[0].shape[0] != b[0].shape[0]:
        raise ValueError("a.shape[0] and b.shape[0] must be the same")
    if c is not None:
        if c.shape != d.shape:
            raise ValueError("c and d must have the same shape")
        if c.dtype != d.dtype:
            raise ValueError("c and d must have the same dtype")
        if c.device != d.device:
            raise ValueError("c and d must be on the same device")
        if not c.is_contiguous():
            raise ValueError("c must be contiguous")
        if c is not d:
            d.copy_(c)
    else:
        d.zero_()

    ragged_k_moe_gemm_8bit(
        a,
        b,
        grouped_layout,
        d,
        scale_granularity_mnk=recipe,
    )

    return d


def k_grouped_bf16_gemm_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn",
    use_psum_layout: Optional[bool] = None,
):
    _ = compiled_dims
    _reject_unsupported_parameters(ks_cpu)
    _reject_unsupported_parameters(use_psum_layout)
    if a.shape[0] != b.shape[0]:
        raise ValueError("a.shape[0] and b.shape[0] must be the same")
    if c is not None:
        if c.shape != d.shape:
            raise ValueError("c and d must have the same shape")
        if c.dtype != d.dtype:
            raise ValueError("c and d must have the same dtype")
        if c.device != d.device:
            raise ValueError("c and d must be on the same device")
        if not c.is_contiguous():
            raise ValueError("c must be contiguous")
        if c is not d:
            d.copy_(c)
    else:
        d.zero_()
    ragged_k_moe_gemm_16bit(a, b, grouped_layout, d)

    return d


# Legacy DeepGEMM API.
m_grouped_fp8_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
m_grouped_fp8_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked


def _fp8_fp4_gemm_dispatch(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor],
    recipe: Optional[Tuple[int, int, int]],
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
    major_a_mode: str,
    major_b_mode: str,
    backend: str,
):
    if recipe_a is None and recipe is None:
        recipe = (1, 1, 128) if b[1].dtype == torch.int32 else (1, 128, 128)
    if (recipe_a is None) != (recipe_b is None):
        raise ValueError("recipe_a and recipe_b must be provided together")
    if (recipe_a is None) == (recipe is None):
        raise ValueError("provide either recipe or recipe_a and recipe_b")
    if recipe is not None:
        recipe_a = (recipe[0], recipe[2])
        recipe_b = (recipe[1], recipe[2])
    mate_bmm(
        a[0].unsqueeze(0),
        b[0].unsqueeze(0),
        d.unsqueeze(0),
        scale_a=a[1].unsqueeze(0),
        scale_b=b[1].unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=major_a_mode == "M",
        trans_b=major_b_mode == "K",
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        backend=backend,
    )


def fp8_fp4_gemm_nt(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, recipe_a, recipe_b, "K", "K", backend)


def fp8_fp4_gemm_nn(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, recipe_a, recipe_b, "K", "N", backend)


def fp8_fp4_gemm_tn(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="mn",
    disable_ue8m0_cast=False,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, recipe_a, recipe_b, "M", "N", backend)


def fp8_fp4_gemm_tt(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="mn",
    disable_ue8m0_cast=False,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, recipe_a, recipe_b, "M", "K", backend)


def fp8_gemm_nt(
    a,
    b,
    d,
    c=None,
    recipe=None,
    compiled_dims="nk",
    disable_ue8m0_cast=True,
    backend="auto",
):
    mate_fp8_gemm_nt(
        a,
        b,
        d,
        c=c,
        recipe=recipe,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=disable_ue8m0_cast,
        backend=backend,
    )


def fp8_gemm_nn(
    a,
    b,
    d,
    c=None,
    recipe=None,
    compiled_dims="nk",
    disable_ue8m0_cast=True,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, None, None, "K", "N", backend)


def fp8_gemm_tn(
    a,
    b,
    d,
    c=None,
    recipe=None,
    compiled_dims="mn",
    disable_ue8m0_cast=True,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, None, None, "M", "N", backend)


def fp8_gemm_tt(
    a,
    b,
    d,
    c=None,
    recipe=None,
    compiled_dims="mn",
    disable_ue8m0_cast=True,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, None, None, "M", "K", backend)


def einsum(
    expr: str,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
):
    supported_expressions = (
        "bmk,bnk->mn",
        "bhr,hdr->bhd",
        "bhd,hdr->bhr",
    )
    if expr not in supported_expressions:
        raise ValueError(f"Unsupported einsum expression: {expr}")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("einsum expects BF16 a and b tensors")
    if d.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("einsum expects BF16 or FP32 d tensor")
    if expr != "bmk,bnk->mn" and d.dtype != torch.bfloat16:
        raise ValueError(f"einsum('{expr}') expects a BF16 d tensor")
    if c is not None:
        if expr != "bmk,bnk->mn":
            raise ValueError(f"einsum('{expr}') does not support C accumulation")
        if c.dtype != torch.float32 or d.dtype != torch.float32:
            raise ValueError("einsum with c expects FP32 c and d tensors")
        if c.data_ptr() != d.data_ptr():
            raise ValueError("einsum expects c to alias d")
    elif d.dtype == torch.float32:
        raise ValueError("einsum with FP32 d requires c")

    result = torch.einsum(expr, a, b)
    if c is None:
        d.copy_(result)
    else:
        d.add_(result)


def fp8_gemm_nt_skip_head_mid(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    head_splits: Tuple[int, int, int],
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
):
    return mate_fp8_gemm_nt_skip_head_mid(
        a,
        b,
        d,
        head_splits,
        recipe=recipe,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=disable_ue8m0_cast,
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
    context_lens: torch.Tensor,
    block_kv: int,
    num_mps: int = 0,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Get metadata for paged MQA logits

    Parameters
    ----------
    context_lens: Tensor
        Context lengths of each query, shape ``(batch_size)``
    block_kv: Tensor
        Block size of kv cache, **must be 64 now**.
    num_mps: int
        Number of MPs to execute. 0 means use all MPs of the current device.
    indices: Tensor, optional
        Not supported by Mate; must be left unset.

    Returns
    -------
    Tensor
        Schedule metadata, shape ``(num_mps + 1, 2)``
    """
    _reject_unsupported_parameters(indices)
    return mate_get_paged_mqa_logits_metadata(context_lens, block_kv, num_mps)


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""FP8 Paged MQA logits

    Parameters
    ----------
    q: Tensor
        The FP8 query tensor with shape ``(batch_size, next_n, heads, index_dim)``
    kv_cache: Tensor
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
    indices: Tensor, optional
        Not supported by Mate; must be left unset.

    Returns
    -------
    Tensor
        FP32 logits, shape ``(batch_size * next_n, max_context_len)``
    """
    _reject_unsupported_parameters(indices)
    return mate_fp8_paged_mqa_logits(
        q,
        kv_cache,
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
