# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import copy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
import vllm.envs as envs
from vllm.config import (
    VllmConfig,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_musa.optimization_contract import (
    OptimizationFeature,
    resolve_optimization_contract,
)
from vllm_musa.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from vllm_musa.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )

logger = init_logger(__name__)


def _is_musa_qwen_text_generation_architecture(model_config: Any) -> bool:
    return resolve_optimization_contract(model_config=model_config).prefers(
        OptimizationFeature.QWEN_FA3_SCHEDULER
    )


def _use_musa_qwen_direct_decode_schedule(
    *,
    aot_schedule: bool,
    causal: bool,
    is_bfloat16: bool,
    is_qwen_family: bool,
    use_full_cuda_graph: bool,
    common_prefix_len: int,
    dcp_world_size: int,
    graph_num_reqs: int,
    graph_num_decodes: int,
    graph_num_tokens: int,
    graph_num_decode_tokens: int,
    max_query_len: int,
    max_num_splits: int,
) -> bool:
    return (
        aot_schedule
        and causal
        and is_bfloat16
        and is_qwen_family
        and use_full_cuda_graph
        and common_prefix_len == 0
        and dcp_world_size == 1
        and graph_num_reqs == 64
        and graph_num_decodes == graph_num_reqs
        and graph_num_tokens == graph_num_reqs
        and graph_num_decode_tokens == graph_num_tokens
        and max_query_len == 1
        and max_num_splits == 1
    )


def _is_qwen_family_scheduler_lookup_base_config(
    vllm_config: VllmConfig,
) -> bool:
    """Check the Qwen FA3 scheduler configuration envelope."""
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if scheduler_config is None or parallel_config is None:
        return False

    max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
    contract = resolve_optimization_contract(vllm_config)

    return (
        contract.prefers(OptimizationFeature.QWEN_FA3_SINGLE_REQUEST_METADATA)
        and isinstance(max_num_seqs, int)
        and max_num_seqs >= 1
        and getattr(vllm_config, "speculative_config", None) is None
        and all(
            getattr(parallel_config, name, None) == 1
            for name in (
                "tensor_parallel_size",
                "decode_context_parallel_size",
                "pipeline_parallel_size",
            )
        )
    )


def _is_qwen_family_scheduler_lookup_config(vllm_config: VllmConfig) -> bool:
    """Gate the direct FA3 scheduler path before runtime shape checks."""
    return _is_qwen_family_scheduler_lookup_base_config(vllm_config)


def _has_supported_fa3_scheduler_layout() -> bool:
    """The direct builder mirrors the pinned MATE 0.2.4 metadata layout."""
    try:
        mate_version = version("mate")
        flash_attn_version = version("flash_attn_3")
    except PackageNotFoundError:
        return False
    return mate_version == "0.2.4" and flash_attn_version in {
        "0.2.4",
        "0.2.4+musa",
    }


def _torch_reduce_scatter_dim(
    input_: torch.Tensor,
    group,
    dim: int,
) -> torch.Tensor:
    world_size = group.world_size
    if world_size == 1:
        return input_
    if dim < 0:
        dim += input_.dim()
    if input_.shape[dim] % world_size != 0:
        raise RuntimeError(
            "DCP reduce-scatter dimension must be divisible by world size, "
            f"got shape={tuple(input_.shape)}, dim={dim}, world_size={world_size}."
        )

    chunks = [chunk.contiguous() for chunk in input_.chunk(world_size, dim=dim)]
    output = torch.empty_like(chunks[group.rank_in_group])
    torch.distributed.reduce_scatter(
        output,
        chunks,
        group=getattr(group, "device_group", group),
    )
    return output


class _DCPGroupWithTorchReduceScatter:
    def __init__(self, group) -> None:
        self._group = group
        self.world_size = group.world_size
        self.rank_in_group = group.rank_in_group

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return self._group.all_gather(tensor, dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return _torch_reduce_scatter_dim(tensor, self._group, dim)


def _musa_cp_lse_ag_out_rs(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group,
    ctx=None,
    return_lse: bool = False,
    is_lse_base_on_e=True,
):
    safe_group = _DCPGroupWithTorchReduceScatter(cp_group)
    return cp_lse_ag_out_rs(
        cp_attn_out,
        cp_attn_lse,
        safe_group,
        ctx=ctx,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class MUSAFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        # mate flash_attn_varlen_func(deterministic=True) is
        # batch-invariant (probe: seq0 solo vs batched max_abs_err=0.0), matching
        # upstream FlashAttentionBackend. Probe: generated/musa0400/probe_fa_caps.py.
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        # mate's FA3 wrapper (flash_attn_varlen_func) plumbs causal= through;
        # required for dflash spec-decode verify (non-causal block attention).
        # Base AttentionBackend defaults this to False, which made
        # validate_configuration reject FLASH_ATTN for dflash.
        return True

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # required for multimodal-model backend SELECTION (e.g.
        # gemma-4, which registers as mm_prefix_lm even when served text-only — the
        # dflash workload). mate flash_attn_varlen_func supports causal + window +
        # attention_chunk masks but NOT an arbitrary partial 2D mask. This was
        # VALIDATED on Qwen2.5-VL-7B (image input -> correct description,
        # regression), so the mm-prefix patterns these models actually use are
        # handled correctly. A model whose mm-prefix needed an arbitrary partial 2D
        # mask (not causal/window/chunk) would be wrong on this path — none is
        # currently known or tested; revisit if such a model is served on FLASH_ATTN.
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        # mate applies q/k/v descale per (batch, kv-head), so distinct per-head
        # scales are honoured rather than broadcast from head 0.
        return True

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        raise NotImplementedError(
            f"kv_cache_dtype {kv_cache_dtype!r} is not supported for "
            "FlashAttention on MUSA; mate provides an e4m3 FMHA kernel only."
        )

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 512

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        # MUSA: e4m3 only — mate has no e5m2 FMHA kernel.
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # MUSA: attention sinks are supported regardless of compute capability.
        if has_sink and current_platform.is_musa():
            return None
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # ==================== MUSA ADAPTATION ====================
    num_decodes: int
    num_decode_tokens: int
    decode_query_start_loc: torch.Tensor | None
    decode_seq_lens: torch.Tensor | None
    decode_block_table: torch.Tensor | None

    num_prefills: int
    num_prefill_tokens: int
    prefill_query_start_loc: torch.Tensor | None
    prefill_max_seq_len: int

    cu_seqlens_k: torch.Tensor | None
    # ========================== END ==========================

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True

    # MUSA: pure-prefill batch with no cached prefix (every context_len == 0),
    # so prefill attention reads contiguous K/V (no block_table) and mate routes
    # to the faster mubin TCE flash-attention.
    prefill_no_prefix: bool = False


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of FlashAttention sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        if not isinstance(layer.impl, FlashAttentionImpl):
            continue
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

    # ==================== MUSA ADAPTATION ====================
    reorder_batch_threshold: int = 1
    # ========================== END ==========================

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3
        self._musa_optimization_contract = resolve_optimization_contract(vllm_config)

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        self._sm_count = 60
        self._sm_count_query_succeeded = False
        # ==================== MUSA ADAPTATION ====================
        self._cu_seqlens_k_buffer: torch.Tensor | None = None
        self._use_qwen_single_request_scheduler_lookup = (
            _is_qwen_family_scheduler_lookup_config(vllm_config)
            and _has_supported_fa3_scheduler_layout()
        )
        # ========================== END ==========================

        if self.use_full_cuda_graph and self.aot_schedule:
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )
            try:
                self._sm_count = torch.musa.get_device_properties(
                    self.device
                ).multi_processor_count
                self._sm_count_query_succeeded = True
            except Exception:
                self._sm_count = 60

        # ==================== MUSA ADAPTATION ====================
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None
            else 0
        )
        # With MTP, a decode/verify row carries 1 + draft tokens. Keep the
        # attention split aligned with Mamba/GDN metadata so pure MTP verify
        # batches are not misclassified as prefills.
        self.decode_threshold = (
            self.reorder_batch_threshold + self.num_speculative_tokens
        )

        if self.use_full_cuda_graph:
            self._cu_seqlens_k_buffer = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
        # ========================== END ==========================

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # ==================== MUSA ADAPTATION ====================
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
                require_uniform=False,
                treat_short_extends_as_decodes=(
                    self.num_speculative_tokens == 0
                    or common_attn_metadata.is_prefilling is None
                ),
            )
        )

        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens
        assert num_decodes + num_prefills == num_reqs
        # ========================== END ==========================

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            # MUSA: scale the captured KV-split count by the decode token count.
            # A large decode batch already saturates the SMs, so a fixed split
            # count over-partitions the KV (extra partial passes + a heavier
            # combine) and slows decode; small batches still need splits to fill
            # the SMs. Use decode tokens only (not decode+prefill) so a mixed
            # step does not collapse the decode split count to 1. splits =
            # ceil(sms / (decode_tokens * kv_heads)); equals the pure-decode
            # capture value since there decode tokens == total tokens.
            _decode_bs = max(1, num_decode_tokens)
            max_num_splits = max(
                1,
                min(
                    self.max_num_splits,
                    -(-self._sm_count // (_decode_bs * self.num_heads_kv)),
                ),
            )

        # ==================== MUSA ADAPTATION ====================
        if num_decodes > 0:
            decode_query_start_loc = common_attn_metadata.query_start_loc[
                : num_decodes + 1
            ]
            decode_seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            decode_block_table_tensor = common_attn_metadata.block_table_tensor[
                :num_decodes
            ]
        else:
            decode_query_start_loc = None
            decode_seq_lens = None
            decode_block_table_tensor = None

        if num_prefills > 0:
            prefill_query_start_loc = (
                common_attn_metadata.query_start_loc[num_decodes : num_reqs + 1]
                - common_attn_metadata.query_start_loc[num_decodes]
            )
            prefill_seq_lens = common_attn_metadata.seq_lens[num_decodes:num_reqs]
            prefill_max_seq_len = int(prefill_seq_lens.max().item())
            # MUSA: sum(seq_lens) == num_prefill_tokens iff every prefill request
            # has zero cached context (seq_len_i >= query_len_i), i.e. no prefix.
            prefill_no_prefix = int(prefill_seq_lens.sum().item()) == num_prefill_tokens
        else:
            prefill_query_start_loc = None
            prefill_seq_lens = None
            prefill_max_seq_len = 0
            prefill_no_prefix = False
        # ========================== END ==========================

        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1

        # With one KV split, the FA3 AOT path still selects the persistent
        # scheduler and launches a combine kernel per attention layer.  The
        # direct path selects the single-tile scheduler and needs neither AOT
        # metadata nor that combine step.
        direct_decode_schedule = _use_musa_qwen_direct_decode_schedule(
            aot_schedule=aot_schedule,
            causal=common_attn_metadata.causal is True,
            is_bfloat16=(
                self.model_config.dtype == torch.bfloat16
                and self.kv_cache_dtype in ("auto", "bfloat16", torch.bfloat16)
            ),
            is_qwen_family=self._musa_optimization_contract.prefers(
                OptimizationFeature.QWEN_FA3_SCHEDULER
            ),
            use_full_cuda_graph=self.use_full_cuda_graph,
            common_prefix_len=common_prefix_len,
            dcp_world_size=self.dcp_world_size,
            graph_num_reqs=num_reqs,
            graph_num_decodes=num_decodes,
            graph_num_tokens=num_actual_tokens,
            graph_num_decode_tokens=num_decode_tokens,
            max_query_len=max_query_len,
            max_num_splits=max_num_splits,
        )
        if direct_decode_schedule:
            aot_schedule = False
            logger.info_once(
                "Using MUSA Qwen direct FA3 decode schedule for graph size 64."
            )

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = MUSAFlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_seqlens_k = None
        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None
        direct_metadata_built = False

        if self.dcp_world_size > 1:
            query_kv_lens = query_start_loc[1:] - query_start_loc[:-1]
            dcp_context_kv_lens = seq_lens - query_kv_lens

            dcp_context_kv_lens = get_dcp_local_seq_lens(
                dcp_context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = None

            # The direct builder handles only supported Qwen batch-one decode
            # inputs. All other inputs continue through schedule() below.
            use_qwen_scheduler_lookup = (
                self._use_qwen_single_request_scheduler_lookup
                and self.use_full_cuda_graph
                and aot_schedule
                and self._cu_seqlens_k_buffer is not None
                and num_reqs == 1
                and num_decodes == 1
                and num_decode_tokens == 1
                and num_prefills == 0
                and max_query_len == 1
                and 1 <= max_seq_len <= 4096
                and causal
                and self.aot_sliding_window == (-1, -1)
                and self._sm_count_query_succeeded
                and self._sm_count == 60
                and 1 <= max_num_splits <= 60
                and 1 <= self.num_heads_kv <= 32
                and self.num_heads_q % self.num_heads_kv == 0
                and 1 <= self.num_heads_q // self.num_heads_kv <= 32
                and self.headdim in (64, 128)
                and self.block_size == 64
                and self.kv_cache_dtype == torch.bfloat16
            )
            if use_qwen_scheduler_lookup:
                from vllm_musa.jit_kernel.fa3_metadata import (
                    try_build_qwen_single_request_fa3_metadata,
                )

                buf = self._cu_seqlens_k_buffer
                direct_metadata_built = try_build_qwen_single_request_fa3_metadata(
                    seq_lens[:1],
                    buf[:2],
                    self.scheduler_metadata,
                    max_seq_len,
                    self.num_heads_q,
                    self.num_heads_kv,
                    self.headdim,
                    max_num_splits,
                )
                if direct_metadata_built:
                    cu_seqlens_k = buf[:2]
                    scheduler_metadata = self.scheduler_metadata[:16]
            if scheduler_metadata is None:
                scheduler_metadata = schedule(
                    batch_size=num_reqs,
                    cu_query_lens=query_start_loc,
                    max_query_len=max_query_len,
                    seqlens=seq_lens,
                    max_seq_len=max_seq_len,
                    causal=causal,
                )

            # ==================== MUSA ADAPTATION ====================
            if self.use_full_cuda_graph and not direct_metadata_built:
                if self._cu_seqlens_k_buffer is not None:
                    n = num_reqs + 1
                    buf = self._cu_seqlens_k_buffer
                    buf[0].zero_()
                    seq_lens_i32 = seq_lens[:num_reqs].to(dtype=torch.int32)
                    torch.cumsum(seq_lens_i32, dim=0, dtype=torch.int32, out=buf[1:n])
                    cu_seqlens_k = buf[:n]
                else:
                    cu_seqlens_k = F.pad(
                        seq_lens[:num_reqs],
                        (1, 0),
                        value=0,
                    ).cumsum(dim=0, dtype=torch.int32)
            # ========================== END ==========================

        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            if not direct_metadata_built:
                self.scheduler_metadata[:n] = scheduler_metadata
                # NOTE(woosuk): We should zero out the rest of the scheduler
                # metadata to guarantee the correctness. Otherwise, some thread
                # blocks may use the invalid scheduler metadata and overwrite the
                # output buffer.
                self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            decode_query_start_loc=decode_query_start_loc,
            decode_seq_lens=decode_seq_lens,
            decode_block_table=decode_block_table_tensor,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_max_seq_len=prefill_max_seq_len,
            prefill_no_prefix=prefill_no_prefix,
            cu_seqlens_k=cu_seqlens_k,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.decode_block_table = (
            blk_table[: metadata.num_decodes] if metadata.num_decodes > 0 else None
        )
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        logger.info_once(
            "Using FlashAttention version %s",
            self.vllm_flash_attn_version,
            scope="local",
        )
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT

        if is_quantized_kv_cache(self.kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert (
                flash_attn_supports_sinks()
            ), "Sinks are only supported in FlashAttention 3"
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        # MUSA: layer-static eligibility for the contiguous-KV mubin-TCE prefill
        # path (none of these features can be represented by the plain varlen call).
        self._mubin_prefill_ok = (
            self.attn_type == AttentionType.DECODER
            and self.sinks is None
            and (self.sliding_window is None or self.sliding_window[0] < 0)
            and not self.logits_soft_cap
            and not self.kv_cache_dtype.startswith("fp8")
        )

        # mate's FMHA requires q, k and v to share a dtype, so an fp8 KV cache needs an
        # fp8 query. Letting the layer quantize Q keeps the two in step; on a non-fp8
        # cache the layer leaves the query untouched.
        self.supports_quant_query_input = True

        vllm_config = get_current_vllm_config_or_none()
        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        # ==================== MUSA ADAPTATION ====================
        self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else _musa_cp_lse_ag_out_rs
        # ========================== END ==========================

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."
        assert (
            self.vllm_flash_attn_version is not None
        ), "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_cache_dtype.startswith("fp8"):
            # queries are quantized in the attention layer
            dtype = MUSAFlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            q_descale = layer._q_scale.expand(descale_shape)
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                return output
            else:
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                )

                # ==================== MUSA ADAPTATION ====================
                num_decodes = attn_metadata.num_decodes
                num_decode_tokens = attn_metadata.num_decode_tokens
                num_prefills = attn_metadata.num_prefills
                use_decode_fast_path = (
                    num_prefills == 0
                    and num_decodes > 0
                    and num_decode_tokens <= num_decodes
                    and max_seqlen_q == 1
                    and self.sinks is None
                    and attn_metadata.decode_block_table is not None
                    and attn_metadata.decode_seq_lens is not None
                    and attn_metadata.decode_query_start_loc is not None
                )

                if use_decode_fast_path:
                    # MUSA branch 1/4 - pure decode batch (no prefill, 1 query token per
                    # seq): whole batch attends through the paged KV cache decode kernel.
                    decode_descale_shape = (num_decodes, self.num_kv_heads)
                    decode_output = flash_attn_with_kvcache(
                        q=query[:num_decode_tokens].view(
                            -1, self.num_heads, self.head_size
                        ),
                        k_cache=key_cache,
                        v_cache=value_cache,
                        page_table=attn_metadata.decode_block_table,
                        cache_seqlens=attn_metadata.decode_seq_lens,
                        cu_seqlens_q=attn_metadata.decode_query_start_loc,
                        max_seqlen_q=1,
                        softmax_scale=self.scale,
                        causal=attn_metadata.causal,
                        window_size=sliding_window_size,
                        softcap=self.logits_soft_cap,
                        q_descale=layer._q_scale.expand(decode_descale_shape),
                        k_descale=layer._k_scale.expand(decode_descale_shape),
                        v_descale=layer._v_scale.expand(decode_descale_shape),
                        scheduler_metadata=attn_metadata.scheduler_metadata,
                        num_splits=attn_metadata.max_num_splits,
                    )
                    output[:num_decode_tokens] = decode_output
                elif (
                    num_decodes == 0
                    and num_prefills > 0
                    and attn_metadata.prefill_no_prefix
                    and attn_metadata.causal
                    and not attn_metadata.use_cascade
                    and self._mubin_prefill_ok
                ):
                    # MUSA branch 2/4 - pure prefill with no cached prefix: attend the
                    # contiguous new K/V (no block_table/seqused_k) so mate routes to the
                    # fast mubin varlen_causal TCE kernel, not the slow paged FmhaFwd.
                    flash_attn_varlen_func(
                        q=query[:num_actual_tokens].view(
                            -1, self.num_heads, self.head_size
                        ),
                        k=key[:num_actual_tokens].view(
                            -1, self.num_kv_heads, self.head_size
                        ),
                        v=value[:num_actual_tokens].view(
                            -1, self.num_kv_heads, self.head_size
                        ),
                        out=output[:num_actual_tokens].view(
                            -1, self.num_heads, self.head_size
                        ),
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_q,
                        softmax_scale=self.scale,
                        causal=True,
                        num_splits=attn_metadata.max_num_splits,
                    )
                elif (
                    num_decodes > 0
                    and num_prefills > 0
                    and num_decode_tokens == num_decodes
                    and attn_metadata.prefill_no_prefix
                    and attn_metadata.causal
                    and not attn_metadata.use_cascade
                    and self._mubin_prefill_ok
                    and self.sinks is None
                    and attn_metadata.decode_block_table is not None
                    and attn_metadata.decode_seq_lens is not None
                    and attn_metadata.decode_query_start_loc is not None
                    and attn_metadata.prefill_query_start_loc is not None
                ):
                    # MUSA branch 3/4 - mixed decode+prefill step: split the query into
                    # [decode | prefill] (decode tokens come first). Attend decode tokens through the
                    # paged KV cache and the no-prefix prefill tokens against their
                    # contiguous new K/V (no block_table) so mate routes prefill to the
                    # mubin TCE kernel instead of the slow paged FmhaFwd used for the
                    # whole batch when decode and prefill share a step.
                    decode_descale_shape = (num_decodes, self.num_kv_heads)
                    # MUSA: no scheduler_metadata here. It is built for the full
                    # num_reqs (decode+prefill) layout and does not match this
                    # decode-only slice; a mixed step runs eager (never captured
                    # under FULL_DECODE_ONLY), so the kernel builds its own
                    # decode schedule.
                    output[:num_decode_tokens] = flash_attn_with_kvcache(
                        q=query[:num_decode_tokens].view(
                            -1, self.num_heads, self.head_size
                        ),
                        k_cache=key_cache,
                        v_cache=value_cache,
                        page_table=attn_metadata.decode_block_table,
                        cache_seqlens=attn_metadata.decode_seq_lens,
                        cu_seqlens_q=attn_metadata.decode_query_start_loc,
                        max_seqlen_q=1,
                        softmax_scale=self.scale,
                        causal=attn_metadata.causal,
                        window_size=sliding_window_size,
                        softcap=self.logits_soft_cap,
                        k_descale=layer._k_scale.expand(decode_descale_shape),
                        v_descale=layer._v_scale.expand(decode_descale_shape),
                        num_splits=attn_metadata.max_num_splits,
                    )
                    flash_attn_varlen_func(
                        q=query[num_decode_tokens:num_actual_tokens].view(
                            -1, self.num_heads, self.head_size
                        ),
                        k=key[num_decode_tokens:num_actual_tokens].view(
                            -1, self.num_kv_heads, self.head_size
                        ),
                        v=value[num_decode_tokens:num_actual_tokens].view(
                            -1, self.num_kv_heads, self.head_size
                        ),
                        out=output[num_decode_tokens:num_actual_tokens].view(
                            -1, self.num_heads, self.head_size
                        ),
                        cu_seqlens_q=attn_metadata.prefill_query_start_loc,
                        cu_seqlens_k=attn_metadata.prefill_query_start_loc,
                        max_seqlen_q=attn_metadata.prefill_max_seq_len,
                        max_seqlen_k=attn_metadata.prefill_max_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        num_splits=attn_metadata.max_num_splits,
                    )
                else:
                    # MUSA branch 4/4 - paged fallback for every other case: cascade
                    # attention, prefill with a cached prefix (chunked/extend), spec/Eagle
                    # verify (multiple query tokens per seq), or layers ineligible for the
                    # mubin fast path (sliding-window/softcap/fp8/sinks). All tokens attend
                    # through the paged KV cache with the original query_start_loc layout.
                    use_fused_kv_verify = (
                        num_decodes > 0
                        and num_prefills == 0
                        and num_decode_tokens > num_decodes
                        and max_seqlen_q > 1
                        and attn_metadata.causal
                    )

                    if use_fused_kv_verify:
                        # MTP verify has multiple query tokens per request. KV
                        # cache has already been populated by do_kv_cache_update;
                        # use the paged decode kernel for the verify attention
                        # instead of the varlen paged fallback, which is not
                        # graph-safe for this MUSA shape.
                        output[:num_actual_tokens] = flash_attn_with_kvcache(
                            q=query[:num_actual_tokens].view(
                                -1, self.num_heads, self.head_size
                            ),
                            k_cache=key_cache,
                            v_cache=value_cache,
                            cache_seqlens=seqused_k,
                            page_table=block_table,
                            cu_seqlens_q=cu_seqlens_q,
                            max_seqlen_q=max_seqlen_q,
                            softmax_scale=self.scale,
                            causal=attn_metadata.causal,
                            window_size=sliding_window_size,
                            softcap=self.logits_soft_cap,
                            q_descale=q_descale,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            num_splits=max(attn_metadata.max_num_splits, 1),
                            s_aux=self.sinks,
                        )
                    else:
                        flash_attn_varlen_func(
                            q=query[:num_actual_tokens].view(
                                -1, self.num_heads, self.head_size
                            ),
                            k=key_cache,
                            v=value_cache,
                            out=output[:num_actual_tokens].view(
                                -1, self.num_heads, self.head_size
                            ),
                            cu_seqlens_q=cu_seqlens_q,
                            max_seqlen_q=max_seqlen_q,
                            seqused_k=seqused_k,
                            max_seqlen_k=max_seqlen_k,
                            softmax_scale=self.scale,
                            causal=attn_metadata.causal,
                            window_size=sliding_window_size,
                            block_table=block_table,
                            softcap=self.logits_soft_cap,
                            scheduler_metadata=scheduler_metadata,
                            q_descale=q_descale,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            num_splits=attn_metadata.max_num_splits,
                            s_aux=self.sinks,
                        )
                # ========================== END ==========================

                return output

        # XXX (MUSA): Requires adaptation for Cascade attention
        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    @staticmethod
    def _view_attn_output(
        attn_output: torch.Tensor,
        num_tokens: int,
        num_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        if attn_output.dim() == 3 and attn_output.shape == (
            num_tokens,
            num_heads,
            head_size,
        ):
            return attn_output
        if attn_output.dim() == 4 and attn_output.shape[1] == 1:
            return attn_output.squeeze(1)
        return attn_output.view(num_tokens, num_heads, head_size)

    @staticmethod
    def _lse_to_tokens_heads(
        lse: torch.Tensor,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        if lse.shape == (num_heads, num_tokens):
            return lse.transpose(0, 1).contiguous()
        if lse.shape == (num_tokens, num_heads):
            return lse.contiguous()
        if lse.dim() == 3 and lse.shape[-1] == 1:
            return FlashAttentionImpl._lse_to_tokens_heads(
                lse.squeeze(-1), num_tokens, num_heads
            )
        raise RuntimeError(
            "Unexpected FlashAttention LSE shape "
            f"{tuple(lse.shape)} for tokens={num_tokens}, heads={num_heads}."
        )

    @staticmethod
    def _lse_to_heads_tokens(
        lse: torch.Tensor,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        if lse.shape == (num_heads, num_tokens):
            return lse.contiguous()
        if lse.shape == (num_tokens, num_heads):
            return lse.transpose(0, 1).contiguous()
        if lse.dim() == 3 and lse.shape[-1] == 1:
            return FlashAttentionImpl._lse_to_heads_tokens(
                lse.squeeze(-1), num_tokens, num_heads
            )
        raise RuntimeError(
            "Unexpected FlashAttention LSE shape "
            f"{tuple(lse.shape)} for tokens={num_tokens}, heads={num_heads}."
        )

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ==================== MUSA ADAPTATION ====================
        assert (
            self.vllm_flash_attn_version is not None
        ), "FlashAttention version not detected."
        if attn_metadata.dcp_context_kv_lens is None:
            raise RuntimeError("DCP metadata is missing dcp_context_kv_lens.")
        if attn_metadata.max_dcp_context_kv_len is None:
            raise RuntimeError("DCP metadata is missing max_dcp_context_kv_len.")
        if self.sinks is not None:
            raise NotImplementedError(
                "MUSA FlashAttention DCP does not support attention sinks."
            )
        if self.alibi_slopes is not None:
            raise NotImplementedError("MUSA FlashAttention DCP does not support ALiBi.")
        if not is_quantized_kv_cache(self.kv_cache_dtype):
            q_descale = None
            k_descale = None
            v_descale = None
        # ========================== END ==========================

        num_tokens = query.shape[0]
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )

        query = query.view(num_tokens, self.num_heads, self.head_size).contiguous()
        key = key.view(num_tokens, self.num_kv_heads, self.head_size).to(query.dtype)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size).to(
            query.dtype
        )

        dcp_group = get_dcp_group()
        query_across_dcp = dcp_group.all_gather(query, dim=1)
        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            window_size=sliding_window_size,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        # ==================== MUSA ADAPTATION ====================
        context_attn_out = self._view_attn_output(
            context_attn_out,
            num_tokens,
            self.num_heads * self.dcp_world_size,
            self.head_size,
        )
        context_lse = self._lse_to_tokens_heads(
            context_lse,
            num_tokens,
            self.num_heads * self.dcp_world_size,
        )
        context_attn_out, context_lse = self.dcp_combine(
            context_attn_out,
            context_lse,
            dcp_group,
            return_lse=True,
        )
        context_lse = self._lse_to_heads_tokens(
            context_lse,
            num_tokens,
            self.num_heads,
        )
        # ========================== END ==========================

        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        # ==================== MUSA ADAPTATION ====================
        query_attn_out = self._view_attn_output(
            query_attn_out,
            num_tokens,
            self.num_heads,
            self.head_size,
        )
        query_lse = self._lse_to_heads_tokens(
            query_lse,
            num_tokens,
            self.num_heads,
        )

        output_view = output.view(num_tokens, self.num_heads, self.head_size)
        # ========================== END ==========================
        merge_attn_states(
            output_view,
            context_attn_out,
            context_lse,
            query_attn_out,
            query_lse,
        )
        return output

    def fused_rope_kvcache_supported(self) -> bool:
        """Whether this layer is inside the exact Qwen2 fusion envelope."""
        return (
            get_flash_attn_version() == 3
            and self.num_heads == 14
            and self.num_kv_heads == 2
            and self.head_size == 64
            and self.attn_type == AttentionType.DECODER
            and self.kv_cache_dtype in ("auto", "bfloat16", torch.bfloat16)
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and not self.logits_soft_cap
            and self.sinks is None
            and self.kv_sharing_target_layer_name is None
        )

    def qwen3_qk_rope_kvcache_supported(self) -> bool:
        """Whether this layer is inside a validated Qwen3 cache-out envelope."""
        supported_shape = (
            (self.num_heads, self.num_kv_heads, self.head_size)
            in ((16, 8, 128), (32, 8, 128), (4, 1, 256), (32, 2, 256))
        )
        return (
            get_flash_attn_version() == 3
            and get_kv_cache_layout() == "NHD"
            and supported_shape
            and self.attn_type == AttentionType.DECODER
            and self.kv_cache_dtype in ("auto", "bfloat16", torch.bfloat16)
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and not self.logits_soft_cap
            and self.sinks is None
            and self.kv_sharing_target_layer_name is None
        )

    def do_qwen3_qk_rope_and_kv_cache_update(
        self,
        layer: torch.nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor | None,
        is_neox: bool = True,
        mrope_section_t: int = 64,
        mrope_section_h: int = 0,
        mrope_section_w: int = 0,
        is_interleaved: bool = False,
        eps: float = 1e-6,
        gemma: bool = False,
    ) -> None:
        """Run the exact Qwen3 provider, falling back before cache fusion."""
        from vllm_musa.jit_kernel.csrc.norm import (
            fused_qk_rmsnorm_mrope_cache_out,
        )

        positions_3d = (
            positions if positions.dim() == 2 else positions.unsqueeze(0).expand(3, -1)
        )
        can_fuse_cache = (
            layer_slot_mapping is not None
            and layer_slot_mapping.shape[0] == q.shape[0]
            and get_kv_cache_layout() == "NHD"
        )
        if can_fuse_cache:
            key_cache, value_cache = kv_cache.unbind(0)
            expected_tail = (self.num_kv_heads, self.head_size)
            flat_cache_supported = (
                key_cache.is_contiguous()
                and value_cache.is_contiguous()
                and tuple(key_cache.shape[-2:]) == expected_tail
                and tuple(value_cache.shape[-2:]) == expected_tail
            )
            h256_paged_specialization = (
                gemma
                and is_interleaved
                and self.head_size == 256
                and cos_sin_cache.shape[-1] == 64
                and (mrope_section_t, mrope_section_h, mrope_section_w)
                == (11, 11, 10)
            )
            paged_cache_supported = (
                h256_paged_specialization
                and key_cache.dim() == 4
                and value_cache.dim() == 4
                and tuple(key_cache.shape[-2:]) == expected_tail
                and tuple(value_cache.shape[-2:]) == expected_tail
                and key_cache.stride(3) == 1
                and value_cache.stride(3) == 1
                and key_cache.stride(2) == self.head_size
                and value_cache.stride(2) == self.head_size
                and key_cache.stride(1) == self.num_kv_heads * self.head_size
                and value_cache.stride(1) == self.num_kv_heads * self.head_size
            )
            if flat_cache_supported or paged_cache_supported:
                key_cache_out = (
                    key_cache.view(-1, self.num_kv_heads * self.head_size)
                    if flat_cache_supported
                    else key_cache
                )
                value_cache_out = (
                    value_cache.view(-1, self.num_kv_heads * self.head_size)
                    if flat_cache_supported
                    else value_cache
                )
                fused_qk_rmsnorm_mrope_cache_out(
                    q,
                    k,
                    v,
                    q_weight,
                    k_weight,
                    positions_3d,
                    cos_sin_cache,
                    q_out,
                    k_out,
                    key_cache_out,
                    value_cache_out,
                    layer_slot_mapping,
                    is_neox,
                    mrope_section_t,
                    mrope_section_h,
                    mrope_section_w,
                    is_interleaved,
                    eps,
                    gemma,
                )
                return

        torch.ops.vllm.musa_csrc_fused_qk_rmsnorm_mrope(
            q,
            k,
            q_weight,
            k_weight,
            positions_3d,
            cos_sin_cache,
            q_out,
            k_out,
            is_neox,
            mrope_section_t,
            mrope_section_h,
            mrope_section_w,
            is_interleaved,
            eps,
            gemma,
        )
        if layer_slot_mapping is not None:
            self.do_kv_cache_update(
                layer,
                k_out,
                v,
                kv_cache,
                layer_slot_mapping,
            )

    def do_rope_and_kv_cache_update(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ) -> None:
        """Apply Qwen2 RoPE and populate its NHD cache in one MUSA launch."""
        if query.shape[0] <= 1:
            key_cache, value_cache = kv_cache.unbind(0)
            from vllm_musa.kernels.qwen2_rope_kv import try_qwen2_rope_kv_cache

            if try_qwen2_rope_kv_cache(
                query,
                key,
                value,
                positions,
                cos_sin_cache,
                is_neox,
                key_cache,
                value_cache,
                layer_slot_mapping,
            ):
                return

        # Keep the fused pass correctness-safe if a runtime tensor property
        # falls outside the static model/layer gate.
        from vllm_musa.jit_kernel.csrc.rope import rotary_embedding

        rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            cos_sin_cache,
            is_neox,
        )
        self.do_kv_cache_update(
            layer,
            key,
            value,
            kv_cache,
            layer_slot_mapping,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        assert (
            self.vllm_flash_attn_version is not None
        ), "FlashAttention version not detected."

        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        # ==================== MUSA ADAPTATION ====================
        encoder_output = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            window_size=sliding_window_size,
            return_softmax_lse=False,
        )
        output.copy_(encoder_output)
        # ========================== END ==========================

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (
        -1,
        -1,
    ), "Cascade attention does not support sliding window."

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        # s_aux is incorporated into prefix_lse inside the GPU kernel,
        # enabling its effect during the final attention merge.
        s_aux=s_aux,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
