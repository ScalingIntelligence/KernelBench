from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from mate.api_logging import mate_api


_BF16 = torch.bfloat16
_FP8_E4M3 = torch.float8_e4m3fn


@dataclass(frozen=True)
class _DecodeConfig:
    batch: int
    q_len: int
    heads: int
    topk: int


@dataclass
class _DecodeMetadata:
    config: _DecodeConfig
    max_num_splits: int
    num_scheduler_ctas: int
    scheduler_metadata: torch.Tensor
    num_splits: torch.Tensor


def _check_musa_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "musa":
        raise ValueError(f"{name} must be on a MUSA device, got {tensor.device}")


def _check_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


@mate_api
def mla_rope_quantize_fp8(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    q_rope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MLA RoPE and per-tensor Q/K quantization to FP8 E4M3.

    ``quant_scale_q`` and ``quant_scale_kv`` are independent scalar
    multipliers: ``Q_fp8 = cast(Q * quant_scale_q)`` and
    ``K_fp8 = cast(K * quant_scale_kv)``. ``cos_sin_cache`` may be float32 or
    bfloat16; the kernel performs the rotary arithmetic in float32. All tensor
    arguments must be on the same MUSA device.

    Parameters
    ----------
    q_rope : torch.Tensor
        BF16 query RoPE values with shape ``[nnz, num_heads, 64]``. The last
        dimension must be contiguous.
    k_rope : torch.Tensor
        BF16 key RoPE values with shape ``[nnz, 64]``. The last dimension must
        be contiguous.
    q_nope : torch.Tensor
        BF16 query latent values with shape ``[nnz, num_heads, 512]``. The
        ``nnz`` and ``num_heads`` dimensions must match ``q_rope`` and the last
        dimension must be contiguous.
    k_nope : torch.Tensor
        BF16 key latent values with shape ``[nnz, 512]``. Its ``nnz`` dimension
        must match ``k_rope`` and the last dimension must be contiguous.
    cos_sin_cache : torch.Tensor
        Contiguous FP32 or BF16 rotary cache with shape ``[max_seq_len, 64]``.
        Columns ``[:32]`` contain cosine values and columns ``[32:]`` contain
        sine values.
    pos_ids : torch.Tensor
        Contiguous int32 or int64 positions with shape ``[nnz]``. Every value
        must be in ``[0, max_seq_len)``.
    is_neox : bool
        Select the rotary layout. ``True`` rotates the two contiguous 32-value
        halves; ``False`` rotates adjacent even/odd value pairs.
    quantize_dtype : Optional[torch.dtype]
        Output quantization dtype. Only ``torch.float8_e4m3fn`` is supported.
        When omitted, it is inferred from the first supplied output buffer and
        otherwise defaults to ``torch.float8_e4m3fn``.
    quant_scale_q : float
        Host scalar multiplied into both query components before the FP8 cast.
    quant_scale_kv : float
        Host scalar multiplied into both key components before the FP8 cast.
    q_rope_out : Optional[torch.Tensor]
        Optional FP8 output buffer with shape ``[nnz, num_heads, 64]``.
        Strided leading dimensions are supported; the last dimension must be
        contiguous. A new buffer is allocated when omitted.
    k_rope_out : Optional[torch.Tensor]
        Optional FP8 output buffer with shape ``[nnz, 64]``. The last dimension
        must be contiguous. A new buffer is allocated when omitted.
    q_nope_out : Optional[torch.Tensor]
        Optional FP8 output buffer with shape ``[nnz, num_heads, 512]``.
        Strided leading dimensions are supported; the last dimension must be
        contiguous. A new buffer is allocated when omitted.
    k_nope_out : Optional[torch.Tensor]
        Optional FP8 output buffer with shape ``[nnz, 512]``. The last dimension
        must be contiguous. A new buffer is allocated when omitted.
    enable_pdl : bool
        Unused CUDA PDL compatibility argument. It has no effect on MUSA.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(q_rope_out, k_rope_out, q_nope_out, k_nope_out)``. Supplied output
        buffers are returned directly; omitted buffers are newly allocated on
        the corresponding input device.
    """
    del enable_pdl  # PDL is a CUDA launch hint and has no MUSA equivalent.

    if cos_sin_cache.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("cos_sin_cache must be float32 or bfloat16")

    if quantize_dtype is None:
        quantize_dtype = next(
            (
                out.dtype
                for out in (q_rope_out, k_rope_out, q_nope_out, k_nope_out)
                if out is not None
            ),
            _FP8_E4M3,
        )
    if quantize_dtype != _FP8_E4M3:
        raise ValueError("quantize_dtype must be torch.float8_e4m3fn")

    q_rope_out = (
        q_rope_out
        if q_rope_out is not None
        else torch.empty_like(q_rope, dtype=quantize_dtype)
    )
    k_rope_out = (
        k_rope_out
        if k_rope_out is not None
        else torch.empty_like(k_rope, dtype=quantize_dtype)
    )
    q_nope_out = (
        q_nope_out
        if q_nope_out is not None
        else torch.empty_like(q_nope, dtype=quantize_dtype)
    )
    k_nope_out = (
        k_nope_out
        if k_nope_out is not None
        else torch.empty_like(k_nope, dtype=quantize_dtype)
    )

    from .sparse_mla.tilelang.mla_rope_quantize_fp8 import (
        run_mla_rope_quantize_fp8,
    )

    run_mla_rope_quantize_fp8(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        cos_sin_cache,
        pos_ids,
        q_rope_out,
        k_rope_out,
        q_nope_out,
        k_nope_out,
        quant_scale_q,
        quant_scale_kv,
        is_neox,
    )
    return q_rope_out, k_rope_out, q_nope_out, k_nope_out


def _validate_seq_lens(
    seq_lens: Optional[torch.Tensor], batch: int, device: torch.device
) -> torch.Tensor:
    if seq_lens is None:
        raise TypeError("seq_lens must be a torch.Tensor")
    if seq_lens.ndim == 2 and seq_lens.shape[1] == 1:
        seq_lens = seq_lens.view(-1)
    if seq_lens.shape != (batch,):
        raise ValueError(
            f"seq_lens must have shape ({batch},), got {tuple(seq_lens.shape)}"
        )
    if seq_lens.dtype != torch.int32:
        raise TypeError("seq_lens must have dtype torch.int32")
    if seq_lens.device != device:
        raise ValueError("seq_lens must be on the query device")
    if not seq_lens.is_contiguous():
        raise ValueError("seq_lens must be contiguous")
    return seq_lens


def _select_decode_schedule(
    config: _DecodeConfig, device: torch.device
) -> tuple[int, int]:
    from .sparse_mla.tilelang.sparse_mla_fp8_decode_fwd_scheduled import (
        select_sparse_mla_fp8_max_num_splits,
        select_sparse_mla_fp8_num_scheduler_ctas,
    )

    max_num_splits = select_sparse_mla_fp8_max_num_splits(
        batch=config.batch,
        seq_len=config.q_len,
        num_heads=config.heads,
        kv_group=1,
        topk=config.topk,
        device=device,
    )
    num_scheduler_ctas = select_sparse_mla_fp8_num_scheduler_ctas(
        batch=config.batch,
        seq_len=config.q_len,
        num_heads=config.heads,
        kv_group=1,
        device=device,
        topk=config.topk,
    )
    return max_num_splits, num_scheduler_ctas


def _dispatch_decode_metadata(
    query: torch.Tensor,
    seq_lens: torch.Tensor,
    sparse_mla_top_k: int,
    scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
) -> None:
    from .flashmla import _dispatch_mla_metadata

    _dispatch_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=query.shape[1] * query.shape[2],
        num_heads_k=1,
        num_heads_q=query.shape[2],
        is_fp8_kvcache=True,
        topk=sparse_mla_top_k,
        q=query,
        bs=query.shape[0],
        topk_length=seq_lens,
        tile_scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
    )


def get_batch_decode_metadata_mla(
    query: torch.Tensor,
    seq_lens: torch.Tensor,
    sparse_mla_top_k: int,
) -> _DecodeMetadata:
    """Prepare reusable sparse MLA scheduler metadata.

    Pass the returned object to decode through ``metadata``. Prepare a new object
    after changing ``seq_lens`` or the query configuration.
    """
    _check_musa_tensor("query", query)
    if query.ndim != 4 or query.shape[-1] != 576:
        raise ValueError("query must have shape [batch, q_len, heads, 576]")
    batch, q_len, heads, _ = query.shape
    if q_len <= 0 or heads <= 0:
        raise ValueError("query q_len and heads must be positive")
    if sparse_mla_top_k <= 0 or sparse_mla_top_k % 64 != 0:
        raise ValueError("sparse_mla_top_k must be a positive multiple of 64")
    selected_seq_lens = _validate_seq_lens(seq_lens, batch, query.device)
    config = _DecodeConfig(batch, q_len, heads, sparse_mla_top_k)
    metadata = _create_decode_metadata(config, query)
    _dispatch_decode_metadata(
        query,
        selected_seq_lens,
        sparse_mla_top_k,
        metadata.scheduler_metadata,
        metadata.num_splits,
    )
    return metadata


def _create_decode_metadata(
    config: _DecodeConfig,
    query: torch.Tensor,
) -> _DecodeMetadata:
    max_num_splits, num_scheduler_ctas = _select_decode_schedule(config, query.device)
    scheduler_metadata = torch.empty(
        (num_scheduler_ctas, 8), dtype=torch.int32, device=query.device
    )
    num_splits = torch.empty(
        (config.batch + 1,), dtype=torch.int32, device=query.device
    )
    return _DecodeMetadata(
        config=config,
        max_num_splits=max_num_splits,
        num_scheduler_ctas=num_scheduler_ctas,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
    )


def _validate_decode_metadata(
    metadata: _DecodeMetadata,
    config: _DecodeConfig,
    query: torch.Tensor,
) -> _DecodeMetadata:
    if not isinstance(metadata, _DecodeMetadata):
        raise TypeError("metadata must be returned by get_batch_decode_metadata_mla")
    if metadata.config != config:
        raise ValueError(
            "metadata does not match the decode configuration; prepare it again"
        )
    if metadata.scheduler_metadata.device != query.device:
        raise ValueError("metadata must be prepared on the same device as the query")
    return metadata


@mate_api
def sparse_mla_fp8_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: object,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    sparse_mla_top_k: int = 0,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: float | torch.Tensor = 1.0,
    bmm2_scale: float | torch.Tensor = 1.0,
    sinks: Optional[list[torch.Tensor]] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    enable_pdl: Optional[bool] = None,
    backend: str = "auto",
    is_var_seq: bool = True,
    uses_shared_paged_kv_idx: bool = True,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cute_dsl_impl: str = "auto",
    kv_scale_format: str = "auto",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
    metadata: Optional[_DecodeMetadata] = None,
):
    """Run FP8 DeepSeek V3.2 sparse MLA decode through the MATE backend.

    For per-tensor FP8 inputs, the caller folds Q/K descales and the attention
    scale into ``bmm1_scale``. The V descale and any output scale are folded
    into ``bmm2_scale``, matching the FlashInfer contract.

    Parameters
    ----------
    query : torch.Tensor
        Contiguous FP8 E4M3 query with shape ``[batch, q_len, heads, 576]``.
    kv_cache : torch.Tensor
        Contiguous FP8 E4M3 cache with shape ``[pages, page_size, 576]`` or
        ``[pages, 1, page_size, 576]``.
    workspace_buffer : object
        Unused compatibility placeholder for the FlashInfer API.
    qk_nope_head_dim : int
        Unused compatibility argument. The kernel uses a fixed 512-dimensional
        latent vector plus a 64-dimensional RoPE tail.
    kv_lora_rank : int
        Latent-vector width. Only ``512`` is supported.
    qk_rope_head_dim : int
        RoPE width. Only ``64`` is supported.
    block_tables : torch.Tensor
        Contiguous int32 physical-token indices with shape
        ``[batch, q_len, sparse_mla_top_k]``.
    seq_lens : Optional[torch.Tensor]
        Contiguous int32 valid sparse lengths with shape ``[batch]``.
    max_seq_len : int
        Unused compatibility argument; runtime work is bounded by
        ``sparse_mla_top_k`` and ``seq_lens``.
    sparse_mla_top_k : int
        Sparse index capacity. Must be a positive multiple of 64.
    out : Optional[torch.Tensor]
        Optional contiguous BF16 output buffer with shape
        ``[batch, q_len, heads, 512]``.
    bmm1_scale : float | torch.Tensor
        QK/softmax scale. Only a host scalar is supported.
    bmm2_scale : float | torch.Tensor
        Output scale. Only a host scalar is supported.
    sinks : Optional[list[torch.Tensor]]
        Unused FlashInfer compatibility argument.
    skip_softmax_threshold_scale_factor : Optional[float]
        Unused FlashInfer compatibility argument.
    enable_pdl : Optional[bool]
        Unused FlashInfer CUDA-launch compatibility argument.
    backend : str
        Unused FlashInfer compatibility argument. MATE always uses its FP8
        sparse-MLA kernel regardless of this value.
    is_var_seq : bool
        Unused compatibility argument. Variable sparse lengths are always read
        from ``seq_lens``.
    uses_shared_paged_kv_idx : bool
        Unused FlashInfer compatibility argument. The MATE kernel always treats
        the supplied sparse indices as shared by all query heads.
    lse : Optional[torch.Tensor]
        Optional contiguous FP32 LSE buffer with shape
        ``[batch * q_len, heads]`` or ``[batch, q_len, heads]``. The kernel
        writes directly into this buffer.
    return_lse : bool
        Return ``(out, lse)`` instead of only ``out``.
    cute_dsl_impl : str
        Unused FlashInfer compatibility argument. MATE has no CuTeDSL decode
        branch, so this value does not refine or override ``backend``.
    kv_scale_format : str
        Unused compatibility argument. FP8 scales are supplied through
        ``bmm1_scale`` and ``bmm2_scale``.
    cum_seq_lens_q : Optional[torch.Tensor]
        Unused FlashInfer ragged-query compatibility argument.
    max_q_len : Optional[int]
        Unused FlashInfer ragged-query compatibility argument.
    multi_ctas_kv_counter_buffer : Optional[torch.Tensor]
        Unused FlashInfer multi-CTA compatibility argument.
    metadata : Optional[_DecodeMetadata]
        Reusable metadata returned by :func:`get_batch_decode_metadata_mla`.
        When omitted, metadata is prepared for this call. Prepare a new object
        whenever ``seq_lens`` values, query shape, or top-k change.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        The BF16 output tensor, or ``(out, lse)`` when ``return_lse`` is true.
    """
    if sparse_mla_top_k <= 0:
        raise ValueError("sparse_mla_top_k must be positive")
    if (kv_lora_rank, qk_rope_head_dim) != (512, 64):
        raise ValueError("only kv_lora_rank=512 and qk_rope_head_dim=64 are supported")
    if isinstance(bmm1_scale, torch.Tensor):
        raise NotImplementedError("tensor bmm1_scale is not supported")
    if isinstance(bmm2_scale, torch.Tensor):
        raise NotImplementedError("tensor bmm2_scale is not supported")
    for name, tensor in (
        ("query", query),
        ("kv_cache", kv_cache),
        ("block_tables", block_tables),
    ):
        _check_musa_tensor(name, tensor)
    if query.ndim != 4 or query.shape[-1] != 576:
        raise ValueError("query must have shape [batch, q_len, heads, 576]")
    if query.dtype != _FP8_E4M3 or not query.is_contiguous():
        raise TypeError("query must be contiguous torch.float8_e4m3fn")
    batch, q_len, heads, _ = query.shape
    if q_len <= 0 or heads <= 0:
        raise ValueError("query q_len and heads must be positive")

    if kv_cache.ndim == 4:
        if kv_cache.shape[1] != 1 or kv_cache.shape[-1] != 576:
            raise ValueError("4D kv_cache must have shape [pages, 1, page_size, 576]")
        num_pages, _, page_size, _ = kv_cache.shape
        normalized_kv = kv_cache.view(num_pages, page_size, 1, 576)
    elif kv_cache.ndim == 3:
        if kv_cache.shape[-1] != 576:
            raise ValueError("3D kv_cache must have shape [pages, page_size, 576]")
        num_pages, page_size, _ = kv_cache.shape
        normalized_kv = kv_cache.view(num_pages, page_size, 1, 576)
    else:
        raise ValueError("kv_cache must be 3D or 4D")
    if page_size <= 0:
        raise ValueError("kv_cache page_size must be positive")
    if kv_cache.dtype != _FP8_E4M3 or not kv_cache.is_contiguous():
        raise TypeError("kv_cache must be contiguous torch.float8_e4m3fn")
    if kv_cache.device != query.device:
        raise ValueError("kv_cache must be on the query device")

    expected_block_shape = (batch, q_len, sparse_mla_top_k)
    _check_shape("block_tables", block_tables, expected_block_shape)
    if block_tables.dtype != torch.int32 or not block_tables.is_contiguous():
        raise TypeError("block_tables must be contiguous torch.int32")
    if block_tables.device != query.device:
        raise ValueError("block_tables must be on the query device")
    if sparse_mla_top_k % 64 != 0:
        raise ValueError("sparse_mla_top_k must be a multiple of 64")

    expected_out_shape = (batch, q_len, heads, 512)
    if out is None:
        selected_out = torch.empty(expected_out_shape, dtype=_BF16, device=query.device)
    else:
        selected_out = out
        _check_musa_tensor("out", out)
        _check_shape("out", out, expected_out_shape)
        if out.dtype != _BF16 or not out.is_contiguous():
            raise TypeError("out must be contiguous torch.bfloat16")
        if out.device != query.device:
            raise ValueError("out must be on the query device")

    selected_lse = None
    if lse is not None or return_lse:
        flat_lse_shape = (batch * q_len, heads)
        nested_lse_shape = (batch, q_len, heads)
        if lse is None:
            selected_lse = torch.empty(
                flat_lse_shape, dtype=torch.float32, device=query.device
            )
        else:
            selected_lse = lse
            _check_musa_tensor("lse", lse)
            if (
                lse.dtype != torch.float32
                or lse.device != query.device
                or not lse.is_contiguous()
            ):
                raise TypeError(
                    "lse must be contiguous torch.float32 on the query device"
                )
            if lse.shape not in (flat_lse_shape, nested_lse_shape):
                raise ValueError(
                    "lse must have shape [batch * q_len, heads] or "
                    "[batch, q_len, heads]"
                )

    selected_seq_lens = _validate_seq_lens(seq_lens, batch, query.device)
    config = _DecodeConfig(batch=batch, q_len=q_len, heads=heads, topk=sparse_mla_top_k)
    if metadata is None:
        decode_metadata = _create_decode_metadata(config, query)
        _dispatch_decode_metadata(
            query,
            selected_seq_lens,
            sparse_mla_top_k,
            decode_metadata.scheduler_metadata,
            decode_metadata.num_splits,
        )
    else:
        decode_metadata = _validate_decode_metadata(metadata, config, query)

    from .sparse_mla.tilelang.sparse_mla_fp8_decode_fwd_scheduled import (
        sparse_mla_fp8_decode_interface,
    )

    result, _ = sparse_mla_fp8_decode_interface(
        query,
        normalized_kv.view(torch.uint8).view(-1, 1, 576),
        block_tables.unsqueeze(2),
        selected_seq_lens,
        decode_metadata.scheduler_metadata,
        decode_metadata.num_splits,
        bmm1_scale=float(bmm1_scale),
        bmm2_scale=float(bmm2_scale),
        d_v=512,
        max_num_splits=decode_metadata.max_num_splits,
        num_scheduler_ctas=decode_metadata.num_scheduler_ctas,
        out=selected_out,
        lse=(
            selected_lse.view(batch, q_len, heads) if selected_lse is not None else None
        ),
    )

    if return_lse:
        return result, selected_lse
    return result


__all__ = [
    "get_batch_decode_metadata_mla",
    "mla_rope_quantize_fp8",
    "sparse_mla_fp8_decode",
]
