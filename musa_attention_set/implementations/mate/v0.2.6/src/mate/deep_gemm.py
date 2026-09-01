import torch
from typing import Optional, Tuple
from mate.api_logging import mate_api
from mate.mate_runtime import (
    get_num_mps as get_num_mps,
    resolve_num_mps,
    set_num_mps as set_num_mps,
)
from mate.gemm import (
    bmm as mate_bmm,
    ragged_m_moe_gemm_16bit,
    masked_moe_gemm_16bit,
    ragged_k_moe_gemm_16bit,
    ragged_m_moe_gemm_8bit,
    masked_moe_gemm_8bit,
    ragged_k_moe_gemm_8bit,
)
from mate.jit.gemm.deep_gemm.gemm import (
    DEEP_GEMM_EPILOGUE_HEAD_SPLITS,
    GEMM_TYPE_NORMAL,
    get_deep_gemm_gemm_module,
)
from mate.jit.gemm.deep_gemm.hyperconnection import get_hyperconnection_module
from mate.jit.gemm.deep_gemm.mqa_logits import get_mqa_logits_module
from mate.jit.gemm.deep_gemm.paged_mqa_logits import (
    get_paged_mqa_logits_metadata_module,
    get_paged_mqa_logits_module,
)
from mate.jit.runtime import ffi_to_torch
from mate.mega_moe import (
    SymmBuffer as SymmBuffer,
    fp8_fp8_mega_moe as fp8_fp8_mega_moe,
    get_symm_buffer_for_mega_moe as get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe as transform_weights_for_mega_moe,
)


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
    alignment_m: int = 128,
    backend: str = "auto",
):
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
    alignment_m: int = 128,
    backend: str = "auto",
):
    if not disable_ue8m0_cast:
        raise Exception("m_grouped_fp8_gemm_nt_contiguous UE8M0 cast is not supported!")

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
    ks: Optional[list[int]] = None,
    ks_tensor: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
):
    if ks_tensor is None:
        if ks is None:
            raise Exception("Must give the ks whether on host or device!")
        ks_tensor = torch.tensor(ks, device="musa", dtype=torch.int32)

    ragged_k_moe_gemm_8bit(
        a,
        b,
        ks_tensor,
        d,
        scale_granularity_mnk=recipe,
    )

    return d


def k_grouped_bf16_gemm_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    ks: Optional[list[int]] = None,
    ks_tensor: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
):
    if ks_tensor is None:
        if ks is None:
            raise Exception("Must give the ks whether on host or device!")
        ks_tensor = torch.tensor(ks, device="musa", dtype=torch.int32)

    ragged_k_moe_gemm_16bit(
        a,
        b,
        ks_tensor,
        d,
    )

    return d


# legacy deepgemm api
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked


def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
    backend: str = "auto",
) -> torch.Tensor:
    r"""Perform BF16 NT GEMM and write the result to ``d``.

    This compatibility API computes ``d = a @ b.T`` or ``d = c + a @ b.T``
    through :func:`mate.gemm.bmm`. ``compiled_dims`` is accepted for
    DeepGEMM compatibility and is ignored.
    """
    _ = compiled_dims
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=False,
        trans_b=True,
        backend=backend,
    )
    return d


def fp8_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    backend: str = "auto",
) -> torch.Tensor:
    r"""Perform scaled FP8 NT GEMM and write the result to ``d``.

    This compatibility API computes ``d = a @ b.T`` or ``d = c + a @ b.T``
    through :func:`mate.gemm.bmm`. ``compiled_dims`` is accepted for
    DeepGEMM compatibility and is ignored. UE8M0 scale casting is not
    supported.
    """
    _ = compiled_dims
    if not disable_ue8m0_cast:
        raise ValueError("fp8_gemm_nt does not support UE8M0 casting")
    if recipe is None:
        recipe = (1, 128, 128)

    mate_bmm(
        a[0].unsqueeze(0),
        b[0].unsqueeze(0),
        d.unsqueeze(0),
        scale_a=a[1].unsqueeze(0),
        scale_b=b[1].unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=False,
        trans_b=True,
        recipe_a=(recipe[0], recipe[2]),
        recipe_b=(recipe[1], recipe[2]),
        backend=backend,
    )
    return d


@mate_api
def fp8_gemm_nt_skip_head_mid(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    head_splits: Tuple[int, int, int],
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
):
    if not disable_ue8m0_cast:
        raise Exception("fp8_gemm_nt_skip_head_mid UE8M0 cast is not supported!")
    if len(head_splits) != 3:
        raise ValueError(f"head_splits must have 3 elements, got {head_splits}")

    left, mid, right = head_splits
    if left <= 0 or mid < 0 or right <= 0:
        raise ValueError(f"Invalid head_splits: {head_splits}")

    compact_head_dim = left + right
    if b[0].shape[0] % compact_head_dim != 0:
        raise ValueError(
            f"b.shape[0] must be divisible by left + right, got {b[0].shape[0]}"
        )

    full_n = b[0].shape[0] // compact_head_dim * (left + mid + right)
    if d.shape[0] != a[0].shape[0] or d.shape[1] != full_n:
        raise ValueError(
            f"d shape must be ({a[0].shape[0]}, {full_n}), got {tuple(d.shape)}"
        )

    dispatch_name, mod = get_deep_gemm_gemm_module(
        kind="fp8",
        gemm_type=GEMM_TYPE_NORMAL,
        config_m=a[0].shape[0],
        epilogue=DEEP_GEMM_EPILOGUE_HEAD_SPLITS,
        head_splits=head_splits,
    )
    mod.get_function(dispatch_name)(
        a[0],
        a[1],
        b[0],
        b[1],
        d,
        None,
        0,
        resolve_num_mps(a[0].device),
    )


def _validate_fp8_einsum_pair(
    name: str, value: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a tuple of (fp8_tensor, scale_tensor)")
    tensor, scale = value
    if tensor.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"{name}[0] must be an FP8 tensor")
    if scale.dtype != torch.float32:
        raise ValueError(f"{name}[1] must be a float32 scale tensor")
    if tensor.device != scale.device:
        raise ValueError(f"{name}[0] and {name}[1] must be on the same device")
    return tensor, scale


@mate_api
def fp8_einsum(
    expr: str,
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 128, 128),
) -> None:
    r"""DeepGEMM-compatible FP8 einsum.

    Supported expressions are ``"bhr,hdr->bhd"``, ``"bhd,hdr->bhr"``, and
    ``"bhd,bhr->hdr"``. The quantization recipe must be ``(1, 128, 128)`` or
    ``(1, 1, 128)``.
    """
    recipe_values = tuple(int(x) for x in recipe)
    if len(recipe_values) != 3:
        raise ValueError("fp8_einsum recipe must be a 3-tuple")
    recipe = (recipe_values[0], recipe_values[1], recipe_values[2])
    if recipe not in ((1, 128, 128), (1, 1, 128)):
        raise ValueError("fp8_einsum only supports recipe (1, 128, 128) or (1, 1, 128)")
    recipe_a = (recipe[0], recipe[2])
    recipe_b = (recipe[1], recipe[2])

    a_fp8, a_scale = _validate_fp8_einsum_pair("a", a)
    b_fp8, b_scale = _validate_fp8_einsum_pair("b", b)
    if a_fp8.device != b_fp8.device or a_fp8.device != d.device:
        raise ValueError("a, b and d must be on the same device")
    if c is not None:
        if c.device != d.device:
            raise ValueError("c must be on the same device as d")
        if c.dtype != torch.float32 or d.dtype != torch.float32:
            raise ValueError("fp8_einsum with c expects fp32 c and d tensors")

    if expr == "bhr,hdr->bhd":
        if a_fp8.dim() != 3 or b_fp8.dim() != 3 or d.dim() != 3:
            raise ValueError("fp8_einsum('bhr,hdr->bhd') expects 3D tensors")
        batch, heads, r_dim = a_fp8.shape
        h_b, d_dim, r_b = b_fp8.shape
        if heads != h_b or r_dim != r_b or tuple(d.shape) != (batch, heads, d_dim):
            raise ValueError("expected a[b,h,r], b[h,d,r] and d[b,h,d]")
        c_view = c.permute(1, 0, 2) if c is not None else None
        mate_bmm(
            a_fp8.permute(1, 0, 2),
            b_fp8,
            d.permute(1, 0, 2),
            scale_a=a_scale.permute(1, 0, 2),
            scale_b=b_scale,
            c=c_view,
            trans_a=False,
            trans_b=True,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        return None

    if expr == "bhd,hdr->bhr":
        if a_fp8.dim() != 3 or b_fp8.dim() != 3 or d.dim() != 3:
            raise ValueError("fp8_einsum('bhd,hdr->bhr') expects 3D tensors")
        batch, heads, d_dim = a_fp8.shape
        h_b, d_b, r_dim = b_fp8.shape
        if heads != h_b or d_dim != d_b or tuple(d.shape) != (batch, heads, r_dim):
            raise ValueError("expected a[b,h,d], b[h,d,r] and d[b,h,r]")
        c_view = c.permute(1, 0, 2) if c is not None else None
        mate_bmm(
            a_fp8.permute(1, 0, 2),
            b_fp8,
            d.permute(1, 0, 2),
            scale_a=a_scale.permute(1, 0, 2),
            scale_b=b_scale,
            c=c_view,
            trans_a=False,
            trans_b=False,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        return None

    if expr == "bhd,bhr->hdr":
        if a_fp8.dim() != 3 or b_fp8.dim() != 3 or d.dim() != 3:
            raise ValueError("fp8_einsum('bhd,bhr->hdr') expects 3D tensors")
        batch, heads, d_dim = a_fp8.shape
        b_batch, h_b, r_dim = b_fp8.shape
        if batch != b_batch or heads != h_b or tuple(d.shape) != (heads, d_dim, r_dim):
            raise ValueError("expected a[b,h,d], b[b,h,r] and d[h,d,r]")
        mate_bmm(
            a_fp8.permute(1, 0, 2),
            b_fp8.permute(1, 0, 2),
            d,
            scale_a=a_scale.permute(1, 0, 2),
            scale_b=b_scale.permute(1, 0, 2),
            c=c,
            trans_a=True,
            trans_b=False,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        return None

    raise ValueError(f"Unsupported fp8_einsum expression: {expr}")


@mate_api
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
    num_mps = resolve_num_mps(context_lens.device, num_mps)
    schedule_meta = torch.empty(
        (num_mps + 1, 2), device=context_lens.device, dtype=torch.int32
    )
    batch_size = context_lens.shape[0]
    get_paged_mqa_logits_metadata_module(batch_size).get_function(
        "get_paged_mqa_logits_metadata"
    )(context_lens, block_kv, schedule_meta)
    return schedule_meta


@mate_api
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
    next_n = q.shape[1]
    num_heads = q.shape[2]
    head_dim = q.shape[3]
    block_kv = fused_kv_cache.shape[1]
    is_context_lens_2d = context_lens.dim() == 2

    return ffi_to_torch(
        get_paged_mqa_logits_module(
            next_n,
            num_heads,
            head_dim,
            block_kv,
            is_context_lens_2d,
        ).get_function("fp8_paged_mqa_logits")(
            q,
            fused_kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            clean_logits,
        )
    )


@mate_api
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
        A tuple ``(kv_fp8, kv_scale)`` containing:

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
        FP32 logits tensor with dtype ``torch.float32`` and shape:

        - ``(seq_len, seq_len_kv)`` if ``max_seqlen_k == 0``
        - ``(seq_len, max_seqlen_k)`` if ``max_seqlen_k > 0``
    """
    kv_fp8, kv_scale = kv
    if max_seqlen_k > 0 and clean_logits:
        raise ValueError("max_seqlen_k is not supported with clean_logits")

    seq_len = q.shape[0]
    seq_len_kv = kv_fp8.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    compressed_logits = max_seqlen_k > 0
    num_mps = resolve_num_mps(q.device)

    return ffi_to_torch(
        get_mqa_logits_module(
            seq_len,
            seq_len_kv,
            num_heads,
            head_dim,
            compressed_logits,
            num_mps,
        ).get_function("fp8_mqa_logits")(
            q,
            kv_fp8,
            weights,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            kv_scale,
            clean_logits,
            int(max_seqlen_k),
        )
    )


@mate_api
def tf32_hc_prenorm_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    sqr_sum: torch.Tensor,
    num_splits: Optional[int] = None,
) -> None:
    r"""TF32 HyperConnection prenorm GEMM.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor with shape ``(M, K)`` and dtype ``torch.bfloat16``.
    b : torch.Tensor
        Weight tensor with shape ``(N, K)`` and dtype ``torch.float32``.
    d : torch.Tensor
        Output GEMM tensor with dtype ``torch.float32``. Shape is ``(M, N)`` when
        ``num_splits`` is ``None`` or ``<= 1``; otherwise ``(num_splits, M, N)``.
    sqr_sum : torch.Tensor
        Output row-wise squared-sum tensor with dtype ``torch.float32``. Shape is
        ``(M,)`` when ``num_splits`` is ``None`` or ``<= 1``; otherwise
        ``(num_splits, M)``.
    num_splits : int, default=None
        Optional split-K factor. When greater than 1, the kernel writes per-split
        partial outputs and callers should reduce them along dim 0.

    Returns
    -------
    None
    """

    m = a.shape[0]
    n = b.shape[0]
    num_splits = 1 if num_splits is None or num_splits <= 1 else int(num_splits)
    num_mps = resolve_num_mps(a.device)

    get_hyperconnection_module(m, n, num_splits, num_mps).get_function(
        "tf32_hc_prenorm_gemm"
    )(
        a,
        b,
        d,
        sqr_sum,
        num_splits,
    )
