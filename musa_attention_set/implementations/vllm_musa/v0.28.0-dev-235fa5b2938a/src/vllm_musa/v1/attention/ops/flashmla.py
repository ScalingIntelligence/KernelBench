# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.metadata import version
from typing import Any, Callable, TypeVar

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_FLASHMLA_IMPORT_ERROR: Exception | None = None
try:
    import flash_mla as _flash_mla
except Exception as exc:
    _FLASHMLA_IMPORT_ERROR = exc
    _flash_mla = None
else:
    _REQUIRED_FLASHMLA_ATTRS = (
        "FlashMLASchedMeta",
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
    )
    for _attr in _REQUIRED_FLASHMLA_ATTRS:
        if not hasattr(_flash_mla, _attr):
            _FLASHMLA_IMPORT_ERROR = AttributeError(
                f"flash_mla is missing required attribute {_attr!r}"
            )
            _flash_mla = None
            break

_F = TypeVar("_F", bound=Callable[..., Any])


def _identity_decorator(func: _F) -> _F:
    return func


_MATE_SPARSE_DIRECT_OUT_IMPORT_ERROR: Exception | None = None
try:
    from mate.api_logging import mate_api as _mate_api
    from mate.execution_context import (
        raise_complete_if_dry_run as _mate_raise_complete_if_dry_run,
    )
    from mate.mate_runtime import resolve_num_mps as _mate_resolve_num_mps
    from mate.sparse_mla.tilelang.sparse_mla_model1_fwd_pipelined import (
        sparse_attention_fwd_kernel_model1 as _mate_sparse_model1_kernel,
    )
    from mate.sparse_mla.tilelang.sparse_mla_prefill_common import (
        optional_prefill_attn_sink as _mate_optional_prefill_attn_sink,
    )
    from mate.sparse_mla.tilelang.sparse_mla_prefill_common import (
        require_token_lengths as _mate_require_token_lengths,
    )
except Exception as exc:
    # This is an optional private fast path. Missing symbols, shared libraries,
    # or incompatible TileLang/MATE imports must leave the public path usable.
    _MATE_SPARSE_DIRECT_OUT_IMPORT_ERROR = exc
    _mate_api = _identity_decorator
    _mate_raise_complete_if_dry_run = None
    _mate_resolve_num_mps = None
    _mate_sparse_model1_kernel = None
    _mate_optional_prefill_attn_sink = None
    _mate_require_token_lengths = None

try:
    _MATE_SPARSE_DIRECT_OUT_ABI_SUPPORTED = version("mate").split("+", 1)[0] == "0.2.4"
except Exception:
    _MATE_SPARSE_DIRECT_OUT_ABI_SUPPORTED = False

_DSV4_SPARSE_PREFILL_OVERLAP_TOKEN_THRESHOLD = 1024
_DSV4_SPARSE_PREFILL_OVERLAP_BRANCHES = 3

_MATE_SPARSE_DIRECT_OUT_PARAM_NAMES = (
    "q_handle",
    "kv_handle",
    "indices_handle",
    "topk_length_handle",
    "extra_kv_handle",
    "extra_indices_handle",
    "extra_topk_length_handle",
    "attn_sink_handle",
    "output_handle",
    "max_logits_out_handle",
    "lse_handle",
)
_MATE_SPARSE_DIRECT_OUT_RESULT_INDICES = (8, 9, 10)


class _UnavailableFlashMLASchedMeta:
    def __init__(
        self,
        tile_scheduler_metadata: torch.Tensor | None = None,
        num_splits: torch.Tensor | None = None,
    ) -> None:
        self.tile_scheduler_metadata = tile_scheduler_metadata
        self.num_splits = num_splits


FlashMLASchedMeta = (
    _flash_mla.FlashMLASchedMeta
    if _flash_mla is not None
    else _UnavailableFlashMLASchedMeta
)


def _flashmla_unavailable_reason() -> str:
    if _FLASHMLA_IMPORT_ERROR is None:
        return "flash_mla is not available."
    return (
        "flash_mla is not available: "
        f"{type(_FLASHMLA_IMPORT_ERROR).__name__}: {_FLASHMLA_IMPORT_ERROR}"
    )


def _raise_flashmla_unavailable(*_args, **_kwargs):
    raise RuntimeError(_flashmla_unavailable_reason())


def _require_flashmla():
    if _flash_mla is None:
        _raise_flashmla_unavailable()
    return _flash_mla


def _is_flashmla_available() -> tuple[bool, str | None]:
    if _flash_mla is None:
        return False, _flashmla_unavailable_reason()
    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    is_available, reason = _is_flashmla_available()
    if not is_available:
        return False, reason
    if not current_platform.is_musa():
        return False, "FlashMLA Dense is only supported on MUSA devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    is_available, reason = _is_flashmla_available()
    if not is_available:
        return False, reason
    device_capability = current_platform.get_device_capability()
    if device_capability is None or device_capability[0] != 3:
        return False, "FlashMLA Sparse is only supported on MUSA devices."
    return True, None


def _coerce_flashmla_sched_meta(
    tile_scheduler_metadata: FlashMLASchedMeta | torch.Tensor,
    num_splits: torch.Tensor | None,
) -> FlashMLASchedMeta:
    if isinstance(tile_scheduler_metadata, FlashMLASchedMeta):
        if num_splits is not None:
            tile_scheduler_metadata.num_splits = num_splits
        return tile_scheduler_metadata

    return FlashMLASchedMeta(
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
    )


def get_mla_metadata(*args, **kwargs):
    flash_mla = _require_flashmla()
    return flash_mla.get_mla_metadata(*args, **kwargs)


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return current_platform.is_musa() and tensor.device.type == "musa"


def _is_current_stream_capturing() -> bool:
    try:
        is_capturing = getattr(
            torch.get_device_module(),
            "is_current_stream_capturing",
            None,
        )
        return True if is_capturing is None else bool(is_capturing())
    except Exception:
        # The private direct-output shim is not capture-safe. Fail closed when
        # the accelerator cannot report capture state.
        return True


def _tensors_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False
    left_start = left.data_ptr()
    left_end = left_start + left.numel() * left.element_size()
    right_start = right.data_ptr()
    right_end = right_start + right.numel() * right.element_size()
    return left_start < right_end and right_start < left_end


def _out_overlaps_inputs(
    out: torch.Tensor,
    *inputs: torch.Tensor | None,
) -> bool:
    return any(
        input_tensor is not None and _tensors_overlap(out, input_tensor)
        for input_tensor in inputs
    )


def _can_use_dsv4_tp8_sparse_direct_out(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor | None,
    out: torch.Tensor | None,
    d_v: int,
    allow_dsv4_tp8_mtp_direct_out: bool,
) -> bool:
    return (
        allow_dsv4_tp8_mtp_direct_out
        and _MATE_SPARSE_DIRECT_OUT_IMPORT_ERROR is None
        and _MATE_SPARSE_DIRECT_OUT_ABI_SUPPORTED
        and _mate_raise_complete_if_dry_run is not None
        and _mate_resolve_num_mps is not None
        and _mate_sparse_model1_kernel is not None
        and _mate_optional_prefill_attn_sink is not None
        and _mate_require_token_lengths is not None
        and out is not None
        and topk_length is not None
        and _is_musa_tensor(q)
        and not _is_current_stream_capturing()
        and d_v == 512
        and q.ndim == 3
        and q.shape[0] > 0
        and q.shape[1:] == (64, 512)
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
        and out.shape == q.shape
        and out.dtype == q.dtype
        and out.device == q.device
        and out.is_contiguous()
        and kv.ndim == 3
        and kv.shape[1:] == (1, 512)
        and kv.dtype == torch.bfloat16
        and kv.device == q.device
        and kv.is_contiguous()
        and indices.ndim == 3
        and indices.shape[0] == q.shape[0]
        and indices.shape[1] == 1
        and indices.shape[2] % 64 == 0
        and indices.dtype == torch.int32
        and indices.device == q.device
        and indices.is_contiguous()
        and topk_length.shape == (q.shape[0],)
        and topk_length.dtype == torch.int32
        and topk_length.device == q.device
        and (
            attn_sink is None
            or (
                attn_sink.shape == (64,)
                and attn_sink.dtype == torch.float32
                and attn_sink.device == q.device
                and attn_sink.is_contiguous()
            )
        )
        and not _out_overlaps_inputs(
            out,
            q,
            kv,
            indices,
            topk_length,
            attn_sink,
        )
    )


def _has_expected_mate_sparse_direct_out_abi(kernel: Any) -> bool:
    adapter = getattr(kernel, "adapter", None)
    prim_func = getattr(kernel, "prim_func", None)
    params = getattr(prim_func, "params", ())
    param_names = tuple(getattr(param, "name", None) for param in params)
    result_indices = tuple(getattr(adapter, "result_idx", ()))
    return (
        getattr(kernel, "execution_backend", None) == "tvm_ffi"
        and callable(getattr(adapter, "executable", None))
        and param_names == _MATE_SPARSE_DIRECT_OUT_PARAM_NAMES
        and result_indices == _MATE_SPARSE_DIRECT_OUT_RESULT_INDICES
    )


def _dsv4_sparse_prefill_persistent_blocks(q: torch.Tensor) -> int:
    assert _mate_resolve_num_mps is not None
    num_mps = _mate_resolve_num_mps(q.device, None)
    if q.shape[0] <= _DSV4_SPARSE_PREFILL_OVERLAP_TOKEN_THRESHOLD:
        return num_mps
    # Long DSV4 prefill overlaps q projection, KV compression, and the DSA
    # indexer. Give the persistent DSA branch only its share of the MPs so the
    # other full-device branches can always make forward progress.
    return max(num_mps // _DSV4_SPARSE_PREFILL_OVERLAP_BRANCHES, 1)


@_mate_api
def _flash_mla_sparse_fwd_direct_out(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    assert _mate_raise_complete_if_dry_run is not None
    assert _mate_resolve_num_mps is not None
    assert _mate_sparse_model1_kernel is not None
    assert _mate_optional_prefill_attn_sink is not None
    assert _mate_require_token_lengths is not None

    seq_len, heads, dim = q.shape
    normalized_topk_length = _mate_require_token_lengths(
        topk_length,
        seq_len,
        indices.shape[-1],
        q.device,
        "topk_length",
    )
    kernel = _mate_sparse_model1_kernel(
        heads,
        dim,
        has_extra=False,
        kv_group=kv.shape[1],
        sm_scale=sm_scale,
        has_attn_sink=attn_sink is not None,
        is_persistence=True,
        persistent_blocks=_dsv4_sparse_prefill_persistent_blocks(q),
    )
    if not _has_expected_mate_sparse_direct_out_abi(kernel):
        logger.warning_once(
            "MATE sparse-prefill private ABI does not match the validated "
            "0.2.4 layout; using the public FlashMLA path."
        )
        return None

    _mate_raise_complete_if_dry_run()
    attn_sink_arg, _ = _mate_optional_prefill_attn_sink(
        attn_sink,
        heads,
        q.device,
    )
    max_logits = torch.empty((seq_len, heads), dtype=torch.float32, device=q.device)
    lse = torch.empty((seq_len, heads), dtype=torch.float32, device=q.device)
    kernel.adapter.executable(
        q,
        kv,
        indices,
        normalized_topk_length,
        kv,
        indices[:, :, :0],
        normalized_topk_length,
        attn_sink_arg,
        out,
        max_logits,
        lse,
    )
    logger.info_once(
        "Using caller-owned output for DeepSeek-V4 TP8 MATE sparse prefill."
    )
    return out, max_logits, lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    allow_dsv4_tp8_mtp_direct_out: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flash_mla = _require_flashmla()

    if allow_dsv4_tp8_mtp_direct_out:
        if _MATE_SPARSE_DIRECT_OUT_IMPORT_ERROR is not None:
            logger.warning_once(
                "MATE sparse direct-out is unavailable (%s); using the "
                "public FlashMLA path.",
                _MATE_SPARSE_DIRECT_OUT_IMPORT_ERROR,
            )
        elif not _MATE_SPARSE_DIRECT_OUT_ABI_SUPPORTED:
            logger.warning_once(
                "MATE sparse direct-out requires the validated 0.2.4 ABI; "
                "using the public FlashMLA path."
            )

    can_use_direct_out = _can_use_dsv4_tp8_sparse_direct_out(
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        out,
        d_v,
        allow_dsv4_tp8_mtp_direct_out,
    )
    if can_use_direct_out:
        assert topk_length is not None
        assert out is not None
        direct_result = _flash_mla_sparse_fwd_direct_out(
            q,
            kv,
            indices,
            sm_scale,
            attn_sink,
            topk_length,
            out,
        )
        if direct_result is not None:
            return direct_result

    result, max_logits, lse = flash_mla.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    if out is not None:
        out.copy_(result)
        result = out
    return result, max_logits, lse


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta | torch.Tensor,
    num_splits: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flash_mla = _require_flashmla()
    scheduler_metadata = _coerce_flashmla_sched_meta(
        tile_scheduler_metadata, num_splits
    )
    result, softmax_lse = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=scheduler_metadata,
        num_splits=None,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )
    if out is not None:
        out.copy_(result)
        result = out
    return result, softmax_lse


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _flash_mla is None:
        _raise_flashmla_unavailable()
    scheduler_metadata, _ = get_mla_metadata(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        is_fp8_kvcache=True,
    )
    if (
        scheduler_metadata.tile_scheduler_metadata is None
        or scheduler_metadata.num_splits is None
    ):
        raise RuntimeError("FlashMLA FP8 decode metadata was not initialized.")
    return scheduler_metadata.tile_scheduler_metadata, scheduler_metadata.num_splits


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if descale_q is not None or descale_k is not None:
        raise NotImplementedError(
            "MUSA FlashMLA FP8 dense path does not support explicit descale tensors."
        )
    return flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=FlashMLASchedMeta(
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
        ),
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=True,
    )
