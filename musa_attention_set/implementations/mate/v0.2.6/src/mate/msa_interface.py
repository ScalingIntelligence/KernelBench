from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch

from mate.api_logging import mate_api
from mate.jit.msa_ops import _msa_fwd, _msa_maxscore, _msa_sparse_topk_select

__all__ = [
    "MsaDecodePlan",
    "MsaPlan",
    "MsaPlanInfo",
    "MsaPrefillPlan",
    "MsaRuntimeMetadata",
    "build_page_table_from_flat_kv_indices",
    "msa",
    "msa_plan",
    "sparse_decode_atten_func",
    "sparse_msa",
    "sparse_msa_plan",
    "sparse_topk_select",
]

MsaPlanMode = Literal["dense", "paged", "sparse_prefill", "sparse_decode"]
MsaPlanInfo = tuple[bool, int, int, "MsaPlan", Optional["MsaPlan"]]

_SUPPORTED_SPARSE_TOPK = (4, 8, 16)
_SPARSE_BLOCK_SIZE = 128
_SPARSE_TOPK_MAX_OUTPUT_BLOCKS = 64
_FP8_E4M3_DTYPE = getattr(torch, "float8_e4m3fn", None)
_SUPPORTED_FWD_DTYPES = {torch.float16, torch.bfloat16}
if _FP8_E4M3_DTYPE is not None:
    _SUPPORTED_FWD_DTYPES.add(_FP8_E4M3_DTYPE)


@dataclass
class MsaPrefillPlan:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    total_seqlen_k: int
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass
class MsaDecodePlan:
    cache_seqlens: torch.Tensor
    page_size: int
    sparse_block_size: int
    scheduler_metadata: Optional[torch.Tensor] = None


@dataclass
class MsaRuntimeMetadata:
    """Device-resident, per-call metadata for a static MSA plan.

    :func:`msa_plan` captures only static capacity and launch information; this
    object supplies the live sequence metadata on every invocation. All
    populated tensors must stay at fixed addresses/shapes for graph replay;
    MATE does not copy them to another device when this object is used.

    ``cu_seqlens_q`` and ``cu_seqlens_k`` are optional for eager use and are
    derived from ``qo_lens``/``kv_lens`` when omitted.  Graph callers should
    provide them (and ``seqused_k``/``page_table``) explicitly so the call
    performs no metadata allocation or host synchronization.  ``page_table``
    is the preferred representation for paged MSA: it has a fixed
    ``[batch, max_pages]`` shape and can therefore be captured safely.  The
    flat ``kv_page_indptr`` representation remains available for legacy
    callers.
    """

    qo_lens: torch.Tensor
    kv_lens: torch.Tensor
    qo_offset: torch.Tensor
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    kv_page_indptr: Optional[torch.Tensor] = None
    page_table: Optional[torch.Tensor] = None
    seqused_k: Optional[torch.Tensor] = None


@dataclass
class MsaPlan:
    batch_size: int
    qo_lens: torch.Tensor
    kv_lens: torch.Tensor
    qo_offset: torch.Tensor
    kv_page_indptr: Optional[torch.Tensor]
    num_qo_heads: int
    num_kv_heads: int
    page_size: int
    sparse_block_size: int
    num_kv_splits: int
    kv_block_num: int
    force_begin_blocks: int
    force_end_blocks: int
    force_blocks_count_in_topk: bool
    causal: bool
    output_maxscore: bool
    sparse_kernel_mode: str
    use_fp8_kvcache: bool
    mode: MsaPlanMode
    prefill_plan: MsaPrefillPlan
    decode_plan: Optional[MsaDecodePlan] = None
    # ``True`` means that the placeholder length tensors in this plan are not
    # executable metadata.  A runtime ``MsaRuntimeMetadata`` object is required
    # when invoking a static plan.
    is_static: bool = False

    @property
    def selected_block_count(self) -> int:
        if self.kv_block_num <= 0:
            return self.kv_block_num
        return _sparse_selected_block_count(
            self.kv_block_num,
            self.force_begin_blocks,
            self.force_end_blocks,
            self.force_blocks_count_in_topk,
        )


def _prefill_qlen_threshold(sparse: bool) -> int:
    return 32 if sparse else 128


def _ensure_int32_vector(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be a 1D tensor, got shape {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.int32)


def _cpu_int_values(name: str, tensor: torch.Tensor) -> list[int]:
    """Return eager planning metadata without synchronizing an accelerator."""

    if tensor.device.type != "cpu":
        raise ValueError(
            f"{name} must be a CPU tensor for eager MSA planning; use msa_plan "
            "with device-resident MsaRuntimeMetadata for capture or hot-path execution"
        )
    return [int(value) for value in tensor.tolist()]


def _make_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    lengths = _ensure_int32_vector("lengths", lengths)
    cu_seqlens = torch.zeros(
        lengths.numel() + 1,
        dtype=torch.int32,
        device=lengths.device,
    )
    if lengths.numel() > 0:
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    return cu_seqlens


def _normalize_qo_offset(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    qo_offset: Optional[Union[int, torch.Tensor]],
) -> torch.Tensor:
    if qo_offset is None:
        return kv_segment_lens - qo_segment_lens
    if isinstance(qo_offset, int):
        return torch.full_like(qo_segment_lens, qo_offset, dtype=torch.int32)
    offset = _ensure_int32_vector("qo_offset", qo_offset)
    if offset.shape != qo_segment_lens.shape:
        raise ValueError(
            "qo_offset must have the same shape as qo_segment_lens, got "
            f"{tuple(offset.shape)} vs {tuple(qo_segment_lens.shape)}"
        )
    return offset


def _page_counts(kv_lens: torch.Tensor, page_size: int) -> torch.Tensor:
    return torch.div(
        kv_lens + page_size - 1,
        page_size,
        rounding_mode="floor",
    ).to(torch.int32)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _round_up(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def _sparse_selected_block_count(
    topk: int,
    force_begin_blocks: int,
    force_end_blocks: int,
    force_blocks_count_in_topk: bool,
) -> int:
    topk = int(topk)
    force_begin_blocks = int(force_begin_blocks)
    force_end_blocks = int(force_end_blocks)
    if topk not in _SUPPORTED_SPARSE_TOPK:
        raise ValueError(
            "sparse selection only supports topk in "
            f"{_SUPPORTED_SPARSE_TOPK}, got {topk}"
        )
    if force_begin_blocks < 0 or force_end_blocks < 0:
        raise ValueError(
            "force_begin_blocks and force_end_blocks must be non-negative, got "
            f"{force_begin_blocks}, {force_end_blocks}"
        )
    forced_slots = force_begin_blocks + force_end_blocks
    if force_blocks_count_in_topk and forced_slots > topk:
        raise ValueError(
            "force_begin_blocks + force_end_blocks must not exceed topk when "
            "force_blocks_count_in_topk=True, got "
            f"{force_begin_blocks} + {force_end_blocks} > {topk}"
        )
    selected_block_count = topk if force_blocks_count_in_topk else topk + forced_slots
    if selected_block_count > _SPARSE_TOPK_MAX_OUTPUT_BLOCKS:
        raise ValueError(
            "sparse selection output width must not exceed "
            f"{_SPARSE_TOPK_MAX_OUTPUT_BLOCKS}, got {selected_block_count}"
        )
    return selected_block_count


def _build_page_table_from_flat_kv_indices(
    kv_indices: torch.Tensor,
    kv_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    kv_indices = _ensure_int32_vector("kv_indices", kv_indices)
    kv_lens = _ensure_int32_vector("kv_lens", kv_lens)
    kv_lengths = _cpu_int_values("kv_lens", kv_lens)
    page_counts = [_ceil_div(length, page_size) for length in kv_lengths]
    total_pages = sum(page_counts)
    if kv_indices.numel() != total_pages:
        raise ValueError(
            "kv_indices does not match the page count implied by kv_lens: "
            f"{kv_indices.numel()} vs {total_pages}"
        )
    max_pages = max(page_counts, default=0)
    page_table = torch.zeros(
        (kv_lens.numel(), max_pages),
        dtype=torch.int32,
        device=kv_indices.device,
    )
    start = 0
    for batch_idx, count in enumerate(page_counts):
        end = start + count
        if count > 0:
            page_table[batch_idx, :count] = kv_indices[start:end]
        start = end
    return page_table


def build_page_table_from_flat_kv_indices(
    kv_indices: torch.Tensor,
    kv_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    return _build_page_table_from_flat_kv_indices(kv_indices, kv_lens, page_size)


def _to_plan_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device=device, dtype=torch.int32, non_blocking=True)


def _require_runtime_vector(
    name: str,
    tensor: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    length: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"runtime {name} must be a torch.Tensor")
    if tensor.dtype != torch.int32 or tensor.ndim != 1:
        raise TypeError(
            f"runtime {name} must be a contiguous rank-1 int32 tensor, got "
            f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}"
        )
    expected = batch_size if length is None else int(length)
    if int(tensor.shape[0]) != expected:
        raise ValueError(
            f"runtime {name} must have length {expected}, got {int(tensor.shape[0])}"
        )
    if tensor.device != device:
        raise ValueError(
            f"runtime {name} must be on {device}, got {tensor.device}; "
            "MATE does not copy static-plan metadata during execution"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"runtime {name} must be contiguous")
    return tensor


def _require_runtime_page_table(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    max_pages: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("runtime page_table must be a torch.Tensor")
    expected_shape = (int(batch_size), int(max_pages))
    if tensor.dtype != torch.int32 or tensor.ndim != 2:
        raise TypeError(
            "runtime page_table must be a contiguous rank-2 int32 tensor, got "
            f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}"
        )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            "runtime page_table must have shape "
            f"{expected_shape}, got {tuple(tensor.shape)}"
        )
    if tensor.device != device:
        raise ValueError(
            f"runtime page_table must be on {device}, got {tensor.device}; "
            "MATE does not copy static-plan metadata during execution"
        )
    if not tensor.is_contiguous():
        raise ValueError("runtime page_table must be contiguous")
    return tensor


def _resolve_runtime_metadata(
    plan: MsaPlan,
    runtime_metadata: Optional[MsaRuntimeMetadata],
    *,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Resolve per-call metadata without host reads for static plans."""

    if runtime_metadata is None:
        if plan.is_static:
            raise ValueError(
                "static MSA plans require runtime_metadata with device-resident "
                "qo_lens, kv_lens, qo_offset, and cumulative lengths"
            )
        qo_lens = _to_plan_device(plan.qo_lens, device)
        kv_lens = _to_plan_device(plan.kv_lens, device)
        qo_offset = _to_plan_device(plan.qo_offset, device)
        cu_seqlens_q = _to_plan_device(plan.prefill_plan.cu_seqlens_q, device)
        cu_seqlens_k = _to_plan_device(plan.prefill_plan.cu_seqlens_k, device)
        kv_page_indptr = (
            _to_plan_device(plan.kv_page_indptr, device)
            if plan.kv_page_indptr is not None
            else None
        )
        seqused_k = _resolve_seqused_k(
            qo_lens=qo_lens,
            kv_lens=kv_lens,
            qo_offset=qo_offset,
            causal=plan.causal,
        )
        return (
            qo_lens,
            kv_lens,
            seqused_k,
            qo_offset,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            kv_page_indptr,
        )

    if not isinstance(runtime_metadata, MsaRuntimeMetadata):
        raise TypeError(
            "runtime_metadata must be an MsaRuntimeMetadata instance, got "
            f"{type(runtime_metadata).__name__}"
        )
    batch_size = plan.batch_size
    qo_lens = _require_runtime_vector(
        "qo_lens", runtime_metadata.qo_lens, batch_size=batch_size, device=device
    )
    kv_lens = _require_runtime_vector(
        "kv_lens", runtime_metadata.kv_lens, batch_size=batch_size, device=device
    )
    qo_offset = _require_runtime_vector(
        "qo_offset", runtime_metadata.qo_offset, batch_size=batch_size, device=device
    )
    if plan.is_static and runtime_metadata.cu_seqlens_q is None:
        raise ValueError(
            "static MSA runtime_metadata requires cu_seqlens_q to avoid "
            "allocating cumulative lengths during execution"
        )
    if plan.is_static and runtime_metadata.cu_seqlens_k is None:
        raise ValueError(
            "static MSA runtime_metadata requires cu_seqlens_k to avoid "
            "allocating cumulative lengths during execution"
        )
    if runtime_metadata.cu_seqlens_q is None:
        # This fallback keeps eager use concise.  Capture callers should pass a
        # preallocated cumulative-length buffer to avoid an allocation in the
        # captured region.
        cu_seqlens_q = _make_cu_seqlens(qo_lens)
    else:
        cu_seqlens_q = _require_runtime_vector(
            "cu_seqlens_q",
            runtime_metadata.cu_seqlens_q,
            batch_size=batch_size,
            device=device,
            length=batch_size + 1,
        )
    if runtime_metadata.cu_seqlens_k is None:
        cu_seqlens_k = _make_cu_seqlens(kv_lens)
    else:
        cu_seqlens_k = _require_runtime_vector(
            "cu_seqlens_k",
            runtime_metadata.cu_seqlens_k,
            batch_size=batch_size,
            device=device,
            length=batch_size + 1,
        )
    if runtime_metadata.kv_page_indptr is not None:
        kv_page_indptr = _require_runtime_vector(
            "kv_page_indptr",
            runtime_metadata.kv_page_indptr,
            batch_size=batch_size,
            device=device,
            length=batch_size + 1,
        )
    elif plan.kv_page_indptr is not None and not plan.is_static:
        # A legacy plan may still carry host page-indptr metadata.  Static
        # plans intentionally do not, so this branch is only for compatibility.
        kv_page_indptr = _to_plan_device(plan.kv_page_indptr, device)
    else:
        kv_page_indptr = None
    if runtime_metadata.seqused_k is None:
        if plan.is_static:
            raise ValueError(
                "static MSA runtime_metadata requires seqused_k to avoid "
                "deriving per-request lengths during execution"
            )
        seqused_k = _resolve_seqused_k(
            qo_lens=qo_lens,
            kv_lens=kv_lens,
            qo_offset=qo_offset,
            causal=plan.causal,
        )
    else:
        seqused_k = _require_runtime_vector(
            "seqused_k",
            runtime_metadata.seqused_k,
            batch_size=batch_size,
            device=device,
        )
    page_table = None
    if runtime_metadata.page_table is not None:
        page_table = _require_runtime_page_table(
            runtime_metadata.page_table,
            batch_size=batch_size,
            max_pages=_ceil_div(plan.prefill_plan.max_seqlen_k, _SPARSE_BLOCK_SIZE),
            device=device,
        )
        if kv_page_indptr is not None:
            raise ValueError(
                "runtime_metadata.page_table and kv_page_indptr are mutually exclusive"
            )
    return (
        qo_lens,
        kv_lens,
        seqused_k,
        qo_offset,
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        kv_page_indptr,
    )


def _resolve_seqused_k(
    *,
    qo_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    qo_offset: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    return (
        torch.minimum(kv_lens, torch.clamp(qo_lens + qo_offset, min=0))
        if causal
        else kv_lens
    )


def _resolve_plan_mode(
    max_qo_len: int,
    page_size: int,
    kv_block_num: int,
    sparse_kernel_mode: str,
) -> MsaPlanMode:
    if kv_block_num > 0:
        if page_size <= 0:
            raise ValueError("sparse MSA planning requires page_size > 0")
        if sparse_kernel_mode not in {"auto", "prefill", "decode"}:
            raise ValueError(
                "sparse_kernel_mode must be one of 'auto', 'prefill', or 'decode', "
                f"got {sparse_kernel_mode!r}"
            )
        if sparse_kernel_mode == "prefill":
            return "sparse_prefill"
        if sparse_kernel_mode == "decode":
            return "sparse_decode"
        return (
            "sparse_prefill"
            if max_qo_len > _prefill_qlen_threshold(True)
            else "sparse_decode"
        )
    return "paged" if page_size > 0 else "dense"


def _build_single_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    qo_offset: torch.Tensor,
    *,
    num_kv_splits: int,
    page_size: int,
    sparse_block_size: int,
    output_maxscore: bool,
    kv_block_num: int,
    force_begin_blocks: int,
    force_end_blocks: int,
    force_blocks_count_in_topk: bool,
    causal: bool,
    sparse_kernel_mode: str,
    use_fp8_kvcache: bool,
) -> MsaPlan:
    qo_lengths = _cpu_int_values("qo_segment_lens", qo_segment_lens)
    kv_lengths = _cpu_int_values("kv_segment_lens", kv_segment_lens)
    prefill_plan = MsaPrefillPlan(
        cu_seqlens_q=_make_cu_seqlens(qo_segment_lens),
        cu_seqlens_k=_make_cu_seqlens(kv_segment_lens),
        total_seqlen_k=sum(kv_lengths),
        max_seqlen_q=max(qo_lengths, default=0),
        max_seqlen_k=max(kv_lengths, default=0),
    )
    mode = _resolve_plan_mode(
        max_qo_len=prefill_plan.max_seqlen_q,
        page_size=page_size,
        kv_block_num=kv_block_num,
        sparse_kernel_mode=sparse_kernel_mode,
    )
    if mode in {"sparse_prefill", "sparse_decode"}:
        if page_size != sparse_block_size:
            raise ValueError(
                "paged sparse MSA requires page_size == sparse_block_size, got "
                f"page_size={page_size} and sparse_block_size={sparse_block_size}"
            )
        if sparse_block_size != _SPARSE_BLOCK_SIZE:
            raise ValueError(
                "sparse MSA planning currently requires sparse_block_size == "
                f"{_SPARSE_BLOCK_SIZE}, got {sparse_block_size}"
            )
    if output_maxscore and page_size > 0 and page_size != _SPARSE_BLOCK_SIZE:
        raise ValueError(
            "paged MSA maxscore requires page_size == maxscore/sparse block size == "
            f"{_SPARSE_BLOCK_SIZE}, got {page_size}"
        )
    decode_plan = None
    if mode in {"paged", "sparse_decode"}:
        decode_plan = MsaDecodePlan(
            cache_seqlens=kv_segment_lens.clone(),
            page_size=page_size,
            sparse_block_size=sparse_block_size,
        )
    return MsaPlan(
        batch_size=int(qo_segment_lens.numel()),
        qo_lens=qo_segment_lens.clone(),
        kv_lens=kv_segment_lens.clone(),
        qo_offset=qo_offset.clone(),
        kv_page_indptr=(
            _make_cu_seqlens(_page_counts(kv_segment_lens, page_size))
            if page_size > 0
            else None
        ),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        sparse_block_size=sparse_block_size,
        num_kv_splits=num_kv_splits,
        kv_block_num=kv_block_num,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
        force_blocks_count_in_topk=force_blocks_count_in_topk,
        causal=causal,
        output_maxscore=output_maxscore,
        sparse_kernel_mode=sparse_kernel_mode,
        use_fp8_kvcache=use_fp8_kvcache,
        mode=mode,
        prefill_plan=prefill_plan,
        decode_plan=decode_plan,
    )


def _build_static_plan(
    *,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    total_seqlen_k: int,
    num_qo_heads: int,
    num_kv_heads: int,
    num_kv_splits: int,
    page_size: int,
    sparse_block_size: int,
    output_maxscore: bool,
    kv_block_num: int,
    force_begin_blocks: int,
    force_end_blocks: int,
    force_blocks_count_in_topk: bool,
    causal: bool,
    sparse_kernel_mode: str,
    use_fp8_kvcache: bool,
) -> MsaPlan:
    """Build a plan whose launch geometry is independent of live lengths.

    The normal planner stores concrete ``qo_lens``/``kv_lens`` tensors and
    computes maxima and mode from them.  Static plans retain only a capacity
    (the two ``max_seqlen`` values); the zero-filled vectors are deliberately
    placeholders and are rejected at execution time unless a
    :class:`MsaRuntimeMetadata` object is supplied.
    """

    batch_size = int(batch_size)
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)
    total_seqlen_k = int(total_seqlen_k)
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError(
            "static MSA planning requires positive max_seqlen_q/max_seqlen_k, "
            f"got {max_seqlen_q} and {max_seqlen_k}"
        )
    if total_seqlen_k <= 0:
        raise ValueError(
            f"static MSA planning requires positive total_seqlen_k, got {total_seqlen_k}"
        )
    if total_seqlen_k < max_seqlen_k:
        raise ValueError(
            "static MSA total_seqlen_k must cover max_seqlen_k, got "
            f"{total_seqlen_k} < {max_seqlen_k}"
        )

    mode = _resolve_plan_mode(
        max_qo_len=max_seqlen_q,
        page_size=page_size,
        kv_block_num=kv_block_num,
        sparse_kernel_mode=sparse_kernel_mode,
    )
    if mode in {"sparse_prefill", "sparse_decode"}:
        if page_size != sparse_block_size:
            raise ValueError(
                "paged sparse MSA requires page_size == sparse_block_size, got "
                f"page_size={page_size} and sparse_block_size={sparse_block_size}"
            )
        if sparse_block_size != _SPARSE_BLOCK_SIZE:
            raise ValueError(
                "sparse MSA planning currently requires sparse_block_size == "
                f"{_SPARSE_BLOCK_SIZE}, got {sparse_block_size}"
            )
    if output_maxscore and page_size > 0 and page_size != _SPARSE_BLOCK_SIZE:
        raise ValueError(
            "paged MSA maxscore requires page_size == maxscore/sparse block size == "
            f"{_SPARSE_BLOCK_SIZE}, got {page_size}"
        )
    if kv_block_num > 0:
        _sparse_selected_block_count(
            kv_block_num,
            force_begin_blocks,
            force_end_blocks,
            force_blocks_count_in_topk,
        )
    elif force_begin_blocks or force_end_blocks:
        raise ValueError(
            "forced sparse blocks require kv_block_num to select sparse execution"
        )

    # Keep the legacy fields present for introspection and for type/API
    # compatibility, but mark them as placeholders.  Runtime execution uses
    # MsaRuntimeMetadata instead of these tensors.
    qo_lens = torch.zeros((batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros((batch_size,), dtype=torch.int32)
    qo_offset = torch.zeros((batch_size,), dtype=torch.int32)
    prefill_plan = MsaPrefillPlan(
        cu_seqlens_q=torch.zeros((batch_size + 1,), dtype=torch.int32),
        cu_seqlens_k=torch.zeros((batch_size + 1,), dtype=torch.int32),
        total_seqlen_k=total_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    decode_plan = None
    if mode in {"paged", "sparse_decode"}:
        decode_plan = MsaDecodePlan(
            cache_seqlens=kv_lens.clone(),
            page_size=page_size,
            sparse_block_size=sparse_block_size,
        )
    return MsaPlan(
        batch_size=batch_size,
        qo_lens=qo_lens,
        kv_lens=kv_lens,
        qo_offset=qo_offset,
        kv_page_indptr=None,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        sparse_block_size=sparse_block_size,
        num_kv_splits=num_kv_splits,
        kv_block_num=kv_block_num,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
        force_blocks_count_in_topk=force_blocks_count_in_topk,
        causal=causal,
        output_maxscore=output_maxscore,
        sparse_kernel_mode=sparse_kernel_mode,
        use_fp8_kvcache=use_fp8_kvcache,
        mode=mode,
        prefill_plan=prefill_plan,
        decode_plan=decode_plan,
        is_static=True,
    )


def _unpack_plan_info(plan_info: Union[MsaPlan, MsaPlanInfo]) -> MsaPlanInfo:
    if isinstance(plan_info, MsaPlan):
        return (False, 0, plan_info.batch_size, plan_info, None)
    if (
        isinstance(plan_info, tuple)
        and len(plan_info) == 5
        and isinstance(plan_info[3], MsaPlan)
    ):
        return plan_info
    raise TypeError(
        "plan_info must be a MsaPlan or an MSA-compatible 5-tuple returned by msa_plan"
    )


def _resolve_qo_offset(
    plan: MsaPlan,
    q_offset_override: Optional[Union[int, torch.Tensor]],
    *,
    device: torch.device,
) -> torch.Tensor:
    qo_lens = _to_plan_device(plan.qo_lens, device)
    kv_lens = _to_plan_device(plan.kv_lens, device)
    if q_offset_override is None:
        return _to_plan_device(plan.qo_offset, device)
    return _normalize_qo_offset(qo_lens, kv_lens, q_offset_override).to(
        device=device, dtype=torch.int32, non_blocking=True
    )


def _normalize_kv_block_indexes(
    kv_block_indexes: torch.Tensor,
    *,
    total_q: int,
    num_kv_heads: int,
    num_qo_heads: int,
    qhead_per_kv: int,
) -> torch.Tensor:
    if kv_block_indexes.dtype != torch.int32:
        raise TypeError(
            f"kv_block_indexes must be torch.int32, got {kv_block_indexes.dtype}"
        )
    if kv_block_indexes.ndim != 3:
        raise ValueError(
            "kv_block_indexes must have shape [total_q, Hkv or Hq, topK], got "
            f"{tuple(kv_block_indexes.shape)}"
        )
    if int(kv_block_indexes.shape[0]) != total_q:
        raise ValueError(
            "kv_block_indexes total_q dimension mismatch: "
            f"{int(kv_block_indexes.shape[0])} vs {total_q}"
        )
    head_dim = int(kv_block_indexes.shape[1])
    if head_dim == num_kv_heads:
        return kv_block_indexes.contiguous()
    if head_dim == num_qo_heads and num_qo_heads != num_kv_heads:
        return kv_block_indexes[:, ::qhead_per_kv, :].contiguous()
    raise ValueError(
        "kv_block_indexes head dimension must match num_kv_heads or num_qo_heads, "
        f"got {head_dim}, num_kv_heads={num_kv_heads}, num_qo_heads={num_qo_heads}"
    )


def _split_qkv_for_mixed_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_indices: Optional[torch.Tensor],
    split_plan: MsaPlan,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    q_split = sum(_cpu_int_values("plan.qo_lens", split_plan.qo_lens))
    q0, q1 = q[:q_split], q[q_split:]
    if kv_indices is None:
        kv_split = sum(_cpu_int_values("plan.kv_lens", split_plan.kv_lens))
        return (
            q0,
            k[:kv_split],
            v[:kv_split],
            None,
            q1,
            k[kv_split:],
            v[kv_split:],
            None,
        )

    page_split = sum(
        _ceil_div(length, split_plan.page_size)
        for length in _cpu_int_values("plan.kv_lens", split_plan.kv_lens)
    )
    return (
        q0,
        k[:page_split],
        v[:page_split],
        kv_indices[:page_split],
        q1,
        k[page_split:],
        v[page_split:],
        kv_indices[page_split:],
    )


def _to_mate_paged_kv_layout(
    tensor: torch.Tensor,
    *,
    num_kv_heads: int,
    page_size: int,
    name: str,
) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(
            f"{name} must be a 4D paged tensor, got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[1] == num_kv_heads and tensor.shape[2] == page_size:
        return tensor.permute(0, 2, 1, 3).contiguous()
    if tensor.shape[1] == page_size and tensor.shape[2] == num_kv_heads:
        return tensor.contiguous()
    raise ValueError(
        f"{name} must use [pages, H, page, D] or [pages, page, H, D] layout, "
        f"got shape {tuple(tensor.shape)}"
    )


def _decode_cu_seqlens_q(
    q: torch.Tensor,
    qo_lens: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    *,
    max_seqlen_q: int,
) -> Optional[torch.Tensor]:
    if q.ndim == 3:
        return cu_seqlens_q if cu_seqlens_q is not None else _make_cu_seqlens(qo_lens)
    if q.ndim == 4:
        batch, seqlen_q, _, _ = q.shape
        if batch != int(qo_lens.numel()):
            raise ValueError(f"q batch mismatch: {batch} vs {int(qo_lens.numel())}")
        if seqlen_q != int(max_seqlen_q):
            raise ValueError(f"q seqlen mismatch: {seqlen_q} vs {int(max_seqlen_q)}")
        return None
    raise ValueError(f"q must be rank-3 or rank-4, got shape {tuple(q.shape)}")


def _run_sparse_fwd_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: MsaPlan,
    *,
    qo_lens: torch.Tensor,
    seqused_k: torch.Tensor,
    qo_offset: torch.Tensor,
    kv_indices: Optional[torch.Tensor],
    kv_block_indexes: torch.Tensor,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    sm_scale: Optional[float],
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    return_softmax_lse: bool = False,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    kv_page_indptr: Optional[torch.Tensor] = None,
):
    num_splits = plan.num_kv_splits if plan.num_kv_splits > 0 else 1
    if num_splits != 1:
        raise ValueError("MSA forward currently requires num_kv_splits <= 1")
    if q.dtype not in _SUPPORTED_FWD_DTYPES:
        raise TypeError("MSA forward supports float8_e4m3fn, float16, and bfloat16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must share the same dtype")
    if q.ndim != 3 or int(q.shape[-1]) != 128:
        raise ValueError(f"q must have shape [total_q, Hq, 128], got {tuple(q.shape)}")
    if plan.num_kv_heads <= 0 or plan.num_qo_heads % plan.num_kv_heads != 0:
        raise ValueError("MSA SQMMA forward requires Hq to be divisible by Hkv")
    if plan.num_qo_heads // plan.num_kv_heads not in (8, 16):
        raise ValueError("MSA SQMMA forward requires local Hq/Hkv ratio 8 or 16")
    if int(q.shape[1]) != plan.num_qo_heads:
        raise ValueError(
            f"q head count {int(q.shape[1])} does not match plan.num_qo_heads "
            f"{plan.num_qo_heads}"
        )
    if plan.page_size != 128 or plan.sparse_block_size != 128:
        raise ValueError("MSA forward currently requires page/block size 128")
    if (
        plan.kv_block_num > 0
        and int(kv_block_indexes.shape[-1]) != plan.selected_block_count
    ):
        raise ValueError(
            "kv_block_indexes selected-block width mismatch: "
            f"{int(kv_block_indexes.shape[-1])} vs {plan.selected_block_count}"
        )

    if page_table is not None:
        if page_table.ndim != 2:
            raise ValueError("page_table must be rank 2")
        page_indices = page_table.to(
            device=q.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        kv_page_indptr = None
    else:
        if kv_indices is None:
            raise ValueError("MSA forward requires page indices")
        page_indices = _ensure_int32_vector("kv_indices", kv_indices).to(
            device=q.device, non_blocking=True
        )
        if kv_page_indptr is None:
            if plan.kv_page_indptr is None:
                raise ValueError("paged MSA forward plan is missing kv_page_indptr")
            kv_page_indptr = _to_plan_device(plan.kv_page_indptr, q.device)
    k_runtime = _to_mate_paged_kv_layout(
        k,
        num_kv_heads=plan.num_kv_heads,
        page_size=plan.page_size,
        name="k",
    )
    v_runtime = _to_mate_paged_kv_layout(
        v,
        num_kv_heads=plan.num_kv_heads,
        page_size=plan.page_size,
        name="v",
    )
    qhead_per_kv = plan.num_qo_heads // plan.num_kv_heads
    block_indexes = _normalize_kv_block_indexes(
        kv_block_indexes.to(device=q.device, dtype=torch.int32, non_blocking=True),
        total_q=int(q.shape[0]),
        num_kv_heads=plan.num_kv_heads,
        num_qo_heads=plan.num_qo_heads,
        qhead_per_kv=qhead_per_kv,
    )
    cu_seqlens_q = _decode_cu_seqlens_q(
        q,
        qo_lens,
        cu_seqlens_q,
        max_seqlen_q=plan.prefill_plan.max_seqlen_q,
    )
    max_seqlen_k = plan.prefill_plan.max_seqlen_k
    out_tensor, lse = _msa_fwd(
        q,
        k_runtime,
        v_runtime,
        block_indexes,
        cu_seqlens_q,
        seqused_k,
        qo_offset,
        page_indices,
        max_seqlen_q=plan.prefill_plan.max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=plan.causal,
        kv_page_indptr=kv_page_indptr,
        softmax_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        out=out,
        lse=lse,
    )
    if return_softmax_lse:
        return out_tensor, lse
    return out_tensor


def _materialize_decode_metadata(
    plan: MsaPlan,
    q: torch.Tensor,
    v: torch.Tensor,
    *,
    qo_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> MsaDecodePlan:
    if plan.decode_plan is None:
        raise ValueError(f"plan mode {plan.mode!r} does not define decode metadata")
    plan.decode_plan.cache_seqlens = kv_lens
    if plan.decode_plan.scheduler_metadata is None:
        from mate.mha_interface import get_scheduler_metadata

        plan.decode_plan.scheduler_metadata = get_scheduler_metadata(
            batch_size=plan.batch_size,
            max_seqlen_q=plan.prefill_plan.max_seqlen_q,
            max_seqlen_k=plan.prefill_plan.max_seqlen_k,
            num_heads_q=plan.num_qo_heads,
            num_heads_kv=plan.num_kv_heads,
            headdim=q.shape[-1],
            headdim_v=v.shape[-1],
            seqused_q=qo_lens,
            seqused_k=kv_lens,
            cu_seqlens_q=cu_seqlens_q,
            page_size=plan.page_size,
            num_splits=plan.num_kv_splits if plan.num_kv_splits > 0 else 0,
            causal=plan.causal,
        )
    return plan.decode_plan


def _run_single_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: MsaPlan,
    *,
    kv_indices: Optional[torch.Tensor],
    kv_block_indexes: Optional[torch.Tensor],
    q_offset_override: Optional[Union[int, torch.Tensor]],
    out: Optional[torch.Tensor],
    sm_scale: Optional[float],
    lse: Optional[torch.Tensor] = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    return_softmax_lse: bool = False,
    page_table: Optional[torch.Tensor] = None,
    runtime_metadata: Optional[MsaRuntimeMetadata] = None,
) -> Optional[torch.Tensor]:
    (
        qo_lens,
        kv_lens,
        runtime_seqused_k,
        runtime_qo_offset,
        cu_seqlens_q,
        cu_seqlens_k,
        runtime_page_table,
        runtime_kv_page_indptr,
    ) = _resolve_runtime_metadata(plan, runtime_metadata, device=q.device)
    if runtime_metadata is not None and q_offset_override is not None:
        raise ValueError(
            "q_offset_override cannot be combined with runtime_metadata.qo_offset"
        )
    qo_offset = (
        runtime_qo_offset
        if q_offset_override is None
        else _normalize_qo_offset(qo_lens, kv_lens, q_offset_override).to(
            device=q.device, dtype=torch.int32, non_blocking=True
        )
    )
    seqused_k = runtime_seqused_k
    if runtime_metadata is not None and runtime_metadata.page_table is not None:
        if page_table is not None:
            raise ValueError(
                "page_table argument and runtime_metadata.page_table are mutually exclusive"
            )
        page_table = runtime_page_table

    if plan.mode in {"dense", "paged"} and (k_scale != 1.0 or v_scale != 1.0):
        raise ValueError("k_scale/v_scale are currently supported only by sparse MSA")

    if plan.mode == "dense":
        from mate.mha_interface import flash_attn_varlen_func

        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=plan.prefill_plan.max_seqlen_q,
            max_seqlen_k=plan.prefill_plan.max_seqlen_k,
            seqused_q=qo_lens,
            seqused_k=seqused_k,
            softmax_scale=sm_scale,
            causal=plan.causal,
            return_softmax_lse=False,
            out=out,
        )

    if plan.mode == "paged":
        from mate.mha_interface import flash_attn_with_kvcache

        if page_table is None:
            if kv_indices is None:
                raise ValueError(
                    "paged MSA execution requires kv_indices or page_table"
                )
            if plan.is_static or runtime_metadata is not None:
                raise ValueError(
                    "static/runtime-metadata paged MSA requires a preallocated "
                    "page_table; building one from live device lengths is not "
                    "capture-safe"
                )
            page_table = _build_page_table_from_flat_kv_indices(
                kv_indices.to(device=q.device, dtype=torch.int32, non_blocking=True),
                plan.kv_lens,
                plan.page_size,
            )
        decode_plan = _materialize_decode_metadata(
            plan,
            q,
            v,
            qo_lens=qo_lens,
            kv_lens=kv_lens,
            cu_seqlens_q=cu_seqlens_q,
        )
        k_cache = _to_mate_paged_kv_layout(
            k,
            num_kv_heads=plan.num_kv_heads,
            page_size=plan.page_size,
            name="k",
        )
        v_cache = _to_mate_paged_kv_layout(
            v,
            num_kv_heads=plan.num_kv_heads,
            page_size=plan.page_size,
            name="v",
        )
        result = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=decode_plan.cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=plan.prefill_plan.max_seqlen_q,
            softmax_scale=sm_scale,
            causal=plan.causal,
            scheduler_metadata=decode_plan.scheduler_metadata,
            num_splits=plan.num_kv_splits if plan.num_kv_splits > 0 else 0,
            return_softmax_lse=False,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    if plan.mode in {"sparse_prefill", "sparse_decode"}:
        if kv_indices is None and page_table is None:
            raise ValueError("sparse MSA forward requires kv_indices or page_table")
        if kv_block_indexes is None:
            raise ValueError("sparse MSA forward requires kv_block_indexes")
        return _run_sparse_fwd_kernel(
            q,
            k,
            v,
            plan,
            qo_lens=qo_lens,
            seqused_k=seqused_k,
            qo_offset=qo_offset,
            kv_indices=kv_indices,
            page_table=page_table,
            kv_block_indexes=kv_block_indexes,
            out=out,
            sm_scale=sm_scale,
            lse=lse,
            k_scale=k_scale,
            v_scale=v_scale,
            return_softmax_lse=return_softmax_lse,
            cu_seqlens_q=cu_seqlens_q,
            kv_page_indptr=runtime_kv_page_indptr,
        )
    raise ValueError(f"unknown MSA plan mode {plan.mode!r}")


def _run_maxscore_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    plan: MsaPlan,
    *,
    kv_indices: Optional[torch.Tensor],
    q_offset_override: Optional[Union[int, torch.Tensor]],
    max_score: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor] = None,
    runtime_metadata: Optional[MsaRuntimeMetadata] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    runtime_qo_offset: Optional[torch.Tensor] = None,
    runtime_kv_page_indptr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if plan.mode not in {"dense", "paged"}:
        raise NotImplementedError(
            "MSA maxscore is implemented for dense/paged KV; run the dense/proxy "
            "pass before sparse_topk_select instead of using a sparse execution plan"
        )
    if runtime_metadata is not None:
        (
            _runtime_qo_lens,
            _runtime_kv_lens,
            _runtime_seqused_k,
            runtime_qo_offset_from_meta,
            cu_seqlens_q_from_meta,
            cu_seqlens_k_from_meta,
            runtime_page_table,
            runtime_kv_page_indptr_from_meta,
        ) = _resolve_runtime_metadata(plan, runtime_metadata, device=q.device)
        del _runtime_qo_lens, _runtime_kv_lens, _runtime_seqused_k
        if (
            runtime_qo_offset is not None
            or cu_seqlens_q is not None
            or cu_seqlens_k is not None
        ):
            raise ValueError(
                "explicit maxscore runtime metadata cannot be combined with runtime_metadata"
            )
        runtime_qo_offset = runtime_qo_offset_from_meta
        cu_seqlens_q = cu_seqlens_q_from_meta
        cu_seqlens_k = cu_seqlens_k_from_meta
        if page_table is not None and runtime_page_table is not None:
            raise ValueError(
                "page_table argument and runtime_metadata.page_table are mutually exclusive"
            )
        page_table = runtime_page_table if page_table is None else page_table
        if (
            runtime_kv_page_indptr is not None
            and runtime_kv_page_indptr_from_meta is not None
        ):
            raise ValueError(
                "explicit kv_page_indptr and runtime_metadata.kv_page_indptr are mutually exclusive"
            )
        runtime_kv_page_indptr = (
            runtime_kv_page_indptr
            if runtime_kv_page_indptr is not None
            else runtime_kv_page_indptr_from_meta
        )
    if runtime_qo_offset is None:
        qo_offset = _resolve_qo_offset(plan, q_offset_override, device=q.device)
    elif q_offset_override is not None:
        raise ValueError(
            "q_offset_override cannot be combined with runtime_metadata.qo_offset"
        )
    else:
        qo_offset = runtime_qo_offset
    if cu_seqlens_q is None:
        cu_seqlens_q = _to_plan_device(plan.prefill_plan.cu_seqlens_q, q.device)
    if cu_seqlens_k is None:
        cu_seqlens_k = _to_plan_device(plan.prefill_plan.cu_seqlens_k, q.device)
    page_indices = None
    kv_page_indptr = None
    k_runtime = k
    if plan.mode == "paged":
        if page_table is not None:
            page_indices = _require_runtime_page_table(
                page_table,
                batch_size=plan.batch_size,
                max_pages=_ceil_div(plan.prefill_plan.max_seqlen_k, _SPARSE_BLOCK_SIZE),
                device=q.device,
            )
            if runtime_kv_page_indptr is not None:
                raise ValueError(
                    "2-D page_table cannot be combined with kv_page_indptr"
                )
        else:
            if kv_indices is None:
                raise ValueError("paged MSA maxscore requires page_table or kv_indices")
            page_indices = _ensure_int32_vector("kv_indices", kv_indices).to(
                device=q.device, non_blocking=True
            )
            if runtime_kv_page_indptr is not None:
                kv_page_indptr = runtime_kv_page_indptr
            else:
                if plan.kv_page_indptr is None:
                    raise ValueError(
                        "paged MSA maxscore plan is missing kv_page_indptr"
                    )
                kv_page_indptr = _to_plan_device(plan.kv_page_indptr, q.device)
        k_runtime = _to_mate_paged_kv_layout(
            k,
            num_kv_heads=plan.num_kv_heads,
            page_size=plan.page_size,
            name="k",
        )

    return _msa_maxscore(
        q,
        k_runtime,
        cu_seqlens_q,
        cu_seqlens_k,
        qo_offset,
        max_seqlen_q=plan.prefill_plan.max_seqlen_q,
        max_seqlen_k=plan.prefill_plan.max_seqlen_k,
        causal=plan.causal,
        page_table=page_indices,
        kv_page_indptr=kv_page_indptr,
        max_score=max_score,
    )


@mate_api
def _msa_plan_from_lengths(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    *args,
    qo_offset: Optional[Union[int, torch.Tensor]] = None,
    split_prefill_decode: bool = True,
    **kwargs,
) -> MsaPlanInfo:
    """Build an eager plan from CPU sequence lengths.

    Accelerator-resident lengths are intentionally rejected: deriving Python
    launch metadata from them would introduce a device-to-host synchronization.
    Hot paths must use :func:`msa_plan` and :class:`MsaRuntimeMetadata`.
    """

    qo_segment_lens = _ensure_int32_vector("qo_segment_lens", qo_segment_lens)
    kv_segment_lens = _ensure_int32_vector("kv_segment_lens", kv_segment_lens)
    qo_lengths = _cpu_int_values("qo_segment_lens", qo_segment_lens)
    _cpu_int_values("kv_segment_lens", kv_segment_lens)
    if qo_segment_lens.shape != kv_segment_lens.shape:
        raise ValueError(
            "qo_segment_lens and kv_segment_lens must have the same shape, got "
            f"{tuple(qo_segment_lens.shape)} vs {tuple(kv_segment_lens.shape)}"
        )
    if len(args) > 2:
        raise TypeError(
            "the eager MSA planner accepts num_qo_heads and optional "
            "num_kv_heads as "
            f"positional planner arguments, got {len(args)}"
        )

    if args:
        num_qo_heads = int(args[0])
    else:
        if "num_qo_heads" not in kwargs:
            raise TypeError(
                "the eager MSA planner requires num_qo_heads as the first positional "
                "argument or as a num_qo_heads keyword argument"
            )
        num_qo_heads = int(kwargs.pop("num_qo_heads"))
    if len(args) == 2:
        num_kv_heads = int(args[1])
    else:
        num_kv_heads = int(kwargs.pop("num_kv_heads", num_qo_heads))
    if num_kv_heads <= 0:
        num_kv_heads = num_qo_heads
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads "
            f"({num_kv_heads})"
        )

    num_kv_splits = int(kwargs.pop("num_kv_splits", -1))
    page_size = int(kwargs.pop("page_size", -1))
    sparse_block_size = int(kwargs.pop("sparse_block_size", _SPARSE_BLOCK_SIZE))
    output_maxscore = bool(kwargs.pop("output_maxscore", False))
    kv_block_num = int(kwargs.pop("kv_block_num", -1))
    force_begin_blocks = int(kwargs.pop("force_begin_blocks", 0))
    force_end_blocks = int(kwargs.pop("force_end_blocks", 0))
    force_blocks_count_in_topk = bool(kwargs.pop("force_blocks_count_in_topk", True))
    kwargs.pop("usable_SM_count", None)
    causal = bool(kwargs.pop("causal", True))
    sparse_kernel_mode = str(kwargs.pop("sparse_kernel_mode", "auto"))
    use_fp8_kvcache = bool(kwargs.pop("use_fp8_kvcache", False))
    kwargs.pop("device", None)
    kwargs.pop("stream", None)
    if kwargs:
        raise TypeError(
            "the eager MSA planner got unexpected keyword arguments: "
            f"{', '.join(sorted(kwargs))}"
        )

    qo_offset_tensor = _normalize_qo_offset(
        qo_segment_lens=qo_segment_lens,
        kv_segment_lens=kv_segment_lens,
        qo_offset=qo_offset,
    )
    _cpu_int_values("qo_offset", qo_offset_tensor)

    batch_size = int(qo_segment_lens.numel())
    split = 0
    has_mixed_prefill = False
    sparse = kv_block_num > 0
    if sparse:
        _sparse_selected_block_count(
            kv_block_num,
            force_begin_blocks,
            force_end_blocks,
            force_blocks_count_in_topk,
        )
    elif force_begin_blocks or force_end_blocks:
        raise ValueError(
            "forced sparse blocks require kv_block_num to select sparse execution"
        )
    split_threshold = _prefill_qlen_threshold(sparse)
    if split_prefill_decode and qo_lengths and max(qo_lengths) > split_threshold:
        split = next(
            (idx for idx, length in enumerate(qo_lengths) if length > split_threshold),
            0,
        )
        has_mixed_prefill = split > 0

    if has_mixed_prefill:
        decode_plan = _build_single_plan(
            qo_segment_lens[:split],
            kv_segment_lens[:split],
            num_qo_heads,
            num_kv_heads,
            qo_offset_tensor[:split],
            num_kv_splits=num_kv_splits,
            page_size=page_size,
            sparse_block_size=sparse_block_size,
            output_maxscore=output_maxscore,
            kv_block_num=kv_block_num,
            force_begin_blocks=force_begin_blocks,
            force_end_blocks=force_end_blocks,
            force_blocks_count_in_topk=force_blocks_count_in_topk,
            causal=causal,
            sparse_kernel_mode=sparse_kernel_mode,
            use_fp8_kvcache=use_fp8_kvcache,
        )
        prefill_plan = _build_single_plan(
            qo_segment_lens[split:],
            kv_segment_lens[split:],
            num_qo_heads,
            num_kv_heads,
            qo_offset_tensor[split:],
            num_kv_splits=num_kv_splits,
            page_size=page_size,
            sparse_block_size=sparse_block_size,
            output_maxscore=output_maxscore,
            kv_block_num=kv_block_num,
            force_begin_blocks=force_begin_blocks,
            force_end_blocks=force_end_blocks,
            force_blocks_count_in_topk=force_blocks_count_in_topk,
            causal=causal,
            sparse_kernel_mode=sparse_kernel_mode,
            use_fp8_kvcache=use_fp8_kvcache,
        )
        return (True, split, batch_size, decode_plan, prefill_plan)

    plan = _build_single_plan(
        qo_segment_lens,
        kv_segment_lens,
        num_qo_heads,
        num_kv_heads,
        qo_offset_tensor,
        num_kv_splits=num_kv_splits,
        page_size=page_size,
        sparse_block_size=sparse_block_size,
        output_maxscore=output_maxscore,
        kv_block_num=kv_block_num,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
        force_blocks_count_in_topk=force_blocks_count_in_topk,
        causal=causal,
        sparse_kernel_mode=sparse_kernel_mode,
        use_fp8_kvcache=use_fp8_kvcache,
    )
    return (False, 0, batch_size, plan, None)


@mate_api
def msa_plan(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_qo_heads: int,
    *,
    total_seqlen_k: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    num_kv_splits: int = -1,
    page_size: int = -1,
    sparse_block_size: int = _SPARSE_BLOCK_SIZE,
    output_maxscore: bool = False,
    kv_block_num: int = -1,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
    force_blocks_count_in_topk: bool = True,
    causal: bool = True,
    sparse_kernel_mode: str = "auto",
    use_fp8_kvcache: bool = False,
) -> MsaPlanInfo:
    """Build a capture-safe MSA plan from static capacity information.

    This function never inspects live sequence-length values.
    ``max_seqlen_q`` and ``max_seqlen_k`` determine launch geometry.
    ``total_seqlen_k`` is the physical K-cache token capacity; paged callers
    should pass the full cache capacity rather than the sum of current request
    lengths.  Per-request lengths and page indirection must be supplied to
    :func:`msa` through :class:`MsaRuntimeMetadata`.
    """

    batch_size = int(batch_size)
    max_seqlen_k = int(max_seqlen_k)
    total_seqlen_k = max_seqlen_k if total_seqlen_k is None else int(total_seqlen_k)
    num_qo_heads = int(num_qo_heads)
    num_kv_heads = num_qo_heads if num_kv_heads is None else int(num_kv_heads)
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            "num_qo_heads and num_kv_heads must be positive, got "
            f"{num_qo_heads} and {num_kv_heads}"
        )
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads "
            f"({num_kv_heads})"
        )
    plan = _build_static_plan(
        batch_size=batch_size,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=max_seqlen_k,
        total_seqlen_k=total_seqlen_k,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        num_kv_splits=int(num_kv_splits),
        page_size=int(page_size),
        sparse_block_size=int(sparse_block_size),
        output_maxscore=bool(output_maxscore),
        kv_block_num=int(kv_block_num),
        force_begin_blocks=int(force_begin_blocks),
        force_end_blocks=int(force_end_blocks),
        force_blocks_count_in_topk=bool(force_blocks_count_in_topk),
        causal=bool(causal),
        sparse_kernel_mode=str(sparse_kernel_mode),
        use_fp8_kvcache=bool(use_fp8_kvcache),
    )
    return (False, 0, batch_size, plan, None)


@mate_api
def msa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info: Union[MsaPlan, MsaPlanInfo],
    kv_indices: Optional[torch.Tensor] = None,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override: Optional[Union[int, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the initial MATE-side implementation of MSA."""

    has_mixed_prefill, split, _, decode_plan, prefill_plan = _unpack_plan_info(
        plan_info
    )
    if kv_block_indexes is not None and has_mixed_prefill:
        raise NotImplementedError(
            "mate.msa_interface initial interface does not implement mixed decode+prefill "
            "splitting together with kv_block_indexes sparse execution yet"
        )
    sm_scale = kwargs.pop("sm_scale", None)
    lse = kwargs.pop("lse", None)
    k_scale = float(kwargs.pop("k_scale", 1.0))
    v_scale = float(kwargs.pop("v_scale", 1.0))
    runtime_metadata = kwargs.pop("runtime_metadata", None)
    page_table = kwargs.pop("page_table", None)
    output_o = bool(kwargs.pop("output_o", True))
    output_maxscore = (
        bool(kwargs.pop("output_maxscore", decode_plan.output_maxscore))
        or max_score is not None
    )
    kwargs.pop("check_input_valid", False)
    if kwargs:
        raise TypeError(
            f"msa got unexpected keyword arguments: {', '.join(sorted(kwargs))}"
        )
    if not output_o and not output_maxscore:
        raise ValueError("msa requires output_o=True or output_maxscore=True")
    if output_maxscore and kv_block_indexes is not None:
        raise NotImplementedError(
            "MSA maxscore is the dense/proxy pass input to sparse_topk_select and "
            "does not run with kv_block_indexes"
        )

    if has_mixed_prefill:
        if runtime_metadata is not None:
            raise NotImplementedError(
                "runtime_metadata is not supported with mixed prefill/decode plans"
            )
        if output_maxscore:
            raise NotImplementedError(
                "MSA maxscore for mixed decode+prefill split is not implemented yet"
            )
        assert prefill_plan is not None
        (
            decode_q,
            decode_k,
            decode_v,
            decode_kv_indices,
            prefill_q,
            prefill_k,
            prefill_v,
            prefill_kv_indices,
        ) = _split_qkv_for_mixed_plan(q, k, v, kv_indices, decode_plan)

        out_decode = out[: decode_q.shape[0]] if out is not None else None
        out_prefill = out[decode_q.shape[0] :] if out is not None else None

        decode_override = None
        prefill_override = None
        if isinstance(q_offset_override, torch.Tensor):
            decode_override = q_offset_override[:split]
            prefill_override = q_offset_override[split:]
        elif q_offset_override is not None:
            decode_override = q_offset_override
            prefill_override = q_offset_override

        decode_out = _run_single_plan(
            decode_q,
            decode_k,
            decode_v,
            decode_plan,
            kv_indices=decode_kv_indices,
            kv_block_indexes=None,
            q_offset_override=decode_override,
            out=out_decode,
            sm_scale=sm_scale,
            lse=lse[: decode_q.shape[0]] if lse is not None else None,
            k_scale=k_scale,
            v_scale=v_scale,
            runtime_metadata=runtime_metadata,
        )
        prefill_out = _run_single_plan(
            prefill_q,
            prefill_k,
            prefill_v,
            prefill_plan,
            kv_indices=prefill_kv_indices,
            kv_block_indexes=None,
            q_offset_override=prefill_override,
            out=out_prefill,
            sm_scale=sm_scale,
            lse=lse[decode_q.shape[0] :] if lse is not None else None,
            k_scale=k_scale,
            v_scale=v_scale,
            runtime_metadata=runtime_metadata,
        )
        if out is not None:
            return out, None
        return torch.cat([decode_out, prefill_out], dim=0), None

    out_tensor = (
        _run_single_plan(
            q,
            k,
            v,
            decode_plan,
            kv_indices=kv_indices,
            kv_block_indexes=kv_block_indexes,
            q_offset_override=q_offset_override,
            out=out,
            sm_scale=sm_scale,
            lse=lse,
            k_scale=k_scale,
            v_scale=v_scale,
            runtime_metadata=runtime_metadata,
        )
        if output_o
        else None
    )
    max_score_tensor = (
        _run_maxscore_plan(
            q,
            k,
            decode_plan,
            kv_indices=kv_indices,
            q_offset_override=q_offset_override,
            max_score=max_score,
            page_table=page_table,
            runtime_metadata=runtime_metadata,
        )
        if output_maxscore
        else None
    )
    return out_tensor, max_score_tensor


def _fill_identity_topk_output(output: torch.Tensor, valid_pages: int) -> torch.Tensor:
    topk = int(output.shape[-1])
    ids = torch.arange(topk, dtype=torch.int32, device=output.device)
    if valid_pages < topk:
        ids = torch.where(ids < int(valid_pages), ids, torch.full_like(ids, -1))
    output.copy_(ids.view(1, 1, topk).expand_as(output))
    return output


@mate_api
def sparse_topk_select(
    max_score: torch.Tensor,
    topk: int,
    num_valid_pages: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
    force_blocks_count_in_topk: bool = True,
    query_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if max_score.dtype != torch.float32:
        raise TypeError(f"max_score must be torch.float32, got {max_score.dtype}")
    if max_score.ndim != 3:
        raise ValueError(
            "max_score must have shape [total_q, num_qo_heads, max_k_tiles], "
            f"got {tuple(max_score.shape)}"
        )
    if not max_score.is_contiguous():
        raise ValueError("max_score must be contiguous")
    topk = int(topk)
    total_q, num_qo_heads, max_k_tiles = map(int, max_score.shape)
    if num_qo_heads <= 0 or max_k_tiles <= 0:
        raise ValueError(
            "max_score must have positive num_qo_heads and max_k_tiles, got "
            f"{tuple(max_score.shape)}"
        )
    if max_k_tiles >= 12288:
        raise ValueError(
            f"sparse_topk_select supports max_k_tiles < 12288, got {max_k_tiles}"
        )
    if query_positions is not None:
        if query_positions.dtype != torch.int64:
            raise TypeError(
                f"query_positions must be torch.int64, got {query_positions.dtype}"
            )
        if query_positions.ndim != 1 or int(query_positions.shape[0]) != total_q:
            raise ValueError(
                "query_positions must have shape [total_q], got "
                f"{tuple(query_positions.shape)} for total_q={total_q}"
            )
        if not query_positions.is_contiguous():
            raise ValueError("query_positions must be contiguous")
        if query_positions.device != max_score.device:
            raise ValueError("query_positions must be on the same device as max_score")
    if num_valid_pages is None:
        valid_pages = max_k_tiles
    else:
        valid_pages = int(num_valid_pages)
        if valid_pages <= 0 or valid_pages > max_k_tiles:
            raise ValueError(
                "num_valid_pages must be in (0, max_k_tiles], got "
                f"{valid_pages} for max_k_tiles={max_k_tiles}"
            )
    force_begin_blocks = int(force_begin_blocks)
    force_end_blocks = int(force_end_blocks)
    force_blocks_count_in_topk = bool(force_blocks_count_in_topk)
    selected_block_count = _sparse_selected_block_count(
        topk,
        force_begin_blocks,
        force_end_blocks,
        force_blocks_count_in_topk,
    )

    expected_shape = (total_q, num_qo_heads, selected_block_count)
    if output is None:
        output = torch.empty(
            expected_shape,
            dtype=torch.int32,
            device=max_score.device,
        )
    else:
        if output.dtype != torch.int32:
            raise TypeError(f"output must be torch.int32, got {output.dtype}")
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"output shape must be {expected_shape}, got {tuple(output.shape)}"
            )
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")
        if output.device != max_score.device:
            raise ValueError("output must be on the same device as max_score")

    if total_q == 0:
        return output

    if query_positions is None and valid_pages <= selected_block_count:
        return _fill_identity_topk_output(output, valid_pages)

    return _msa_sparse_topk_select(
        max_score,
        output,
        topk=topk,
        num_valid_pages=valid_pages,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
        force_blocks_count_in_topk=force_blocks_count_in_topk,
        query_positions=query_positions,
    )


@mate_api
def sparse_msa_plan(*args, **kwargs):
    kwargs.setdefault("split_prefill_decode", False)
    kwargs["sparse_kernel_mode"] = "prefill"
    plan_info = _msa_plan_from_lengths(*args, **kwargs)
    has_mixed_prefill, _, _, plan, extra = _unpack_plan_info(plan_info)
    if has_mixed_prefill or extra is not None:
        raise ValueError("sparse_msa_plan expects a pure sparse-prefill batch")
    if plan.mode != "sparse_prefill":
        raise ValueError(
            f"sparse_msa_plan expected sparse_prefill mode, got {plan.mode!r}"
        )
    return plan_info


@mate_api
def sparse_msa(*args, **kwargs):
    if len(args) < 4:
        raise TypeError("sparse_msa requires q, k, v, and plan_info")
    q, k, v, plan_info, *rest = args
    if rest:
        raise TypeError(
            "sparse_msa accepts at most q, k, v, plan_info as positional arguments"
        )

    out = kwargs.pop("out", None)
    lse = kwargs.pop("lse", None)
    max_score = kwargs.pop("max_score", None)
    sm_scale = kwargs.pop("sm_scale", None)
    k_scale = float(kwargs.pop("k_scale", 1.0))
    v_scale = float(kwargs.pop("v_scale", 1.0))
    runtime_metadata = kwargs.pop("runtime_metadata", None)
    kv_indices = kwargs.pop("kv_indices", None)
    output_maxscore = bool(kwargs.pop("output_maxscore", False))
    output_o = bool(kwargs.pop("output_o", True))
    kv_block_indexes = kwargs.pop("kv_block_indexes", None)
    q_offset_override = kwargs.pop("q_offset_override", None)
    kwargs.pop("check_input_valid", False)
    if kwargs:
        raise TypeError(
            f"sparse_msa got unexpected keyword arguments: {', '.join(sorted(kwargs))}"
        )
    if max_score is not None or output_maxscore:
        raise NotImplementedError(
            "mate.msa_interface sparse_msa does not implement max-score output yet"
        )
    if not output_o:
        raise NotImplementedError(
            "mate.msa_interface sparse_msa does not implement output_o=False yet"
        )
    if kv_block_indexes is None:
        raise ValueError("sparse_msa requires kv_block_indexes")

    has_mixed_prefill, _, _, plan, extra = _unpack_plan_info(plan_info)
    if has_mixed_prefill or extra is not None:
        raise ValueError("sparse_msa expects a pure sparse-prefill plan")
    if plan.mode != "sparse_prefill":
        raise ValueError(
            f"sparse_msa expects plan.mode == 'sparse_prefill', got {plan.mode!r}"
        )

    result = _run_single_plan(
        q,
        k,
        v,
        plan,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        q_offset_override=q_offset_override,
        out=out,
        sm_scale=sm_scale,
        lse=lse,
        k_scale=k_scale,
        v_scale=v_scale,
        runtime_metadata=runtime_metadata,
    )
    return result, None


@mate_api
def sparse_decode_atten_func(*args, **kwargs):
    if len(args) < 4:
        raise TypeError("sparse_decode_atten_func requires q, k, v, and plan_info")
    q, k, v, plan_info, *rest = args
    if rest:
        raise TypeError(
            "sparse_decode_atten_func accepts at most q, k, v, plan_info as "
            "positional arguments"
        )

    out = kwargs.pop("out", None)
    lse = kwargs.pop("lse", None)
    max_score = kwargs.pop("max_score", None)
    sm_scale = kwargs.pop("sm_scale", None)
    k_scale = float(kwargs.pop("k_scale", 1.0))
    v_scale = float(kwargs.pop("v_scale", 1.0))
    runtime_metadata = kwargs.pop("runtime_metadata", None)
    kv_indices = kwargs.pop("kv_indices", None)
    page_table = kwargs.pop("page_table", None)
    kv_block_indexes = kwargs.pop("kv_block_indexes", None)
    q_offset_override = kwargs.pop("q_offset_override", None)
    output_maxscore = bool(kwargs.pop("output_maxscore", False))
    output_o = bool(kwargs.pop("output_o", True))
    return_softmax_lse = bool(kwargs.pop("return_softmax_lse", False))
    kwargs.pop("check_input_valid", False)
    if kwargs:
        raise TypeError(
            "sparse_decode_atten_func got unexpected keyword arguments: "
            f"{', '.join(sorted(kwargs))}"
        )
    if max_score is not None or output_maxscore:
        raise NotImplementedError(
            "mate.msa_interface sparse_decode_atten_func does not implement max-score output yet"
        )
    if not output_o:
        raise NotImplementedError(
            "mate.msa_interface sparse_decode_atten_func does not implement output_o=False yet"
        )

    has_mixed_prefill, _, _, plan, extra = _unpack_plan_info(plan_info)
    if has_mixed_prefill or extra is not None:
        raise ValueError("sparse_decode_atten_func expects a pure sparse-decode plan")
    if plan.mode != "sparse_decode":
        raise ValueError(
            f"sparse_decode_atten_func expects plan.mode == 'sparse_decode', got {plan.mode!r}"
        )
    return _run_single_plan(
        q,
        k,
        v,
        plan,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        q_offset_override=q_offset_override,
        out=out,
        sm_scale=sm_scale,
        lse=lse,
        k_scale=k_scale,
        v_scale=v_scale,
        return_softmax_lse=return_softmax_lse,
        page_table=page_table,
        runtime_metadata=runtime_metadata,
    )
