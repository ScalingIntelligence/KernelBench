# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA TurboQuant attention backend."""

from typing import ClassVar

# isort: off
import torchada  # noqa: F401
import torch

# isort: on
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import turboquant_attn as _turboquant_attn
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.ops import triton_turboquant_decode as _turboquant_decode
from vllm.v1.attention.ops import triton_turboquant_store as _turboquant_store

from vllm_musa.v1.attention.backends import fa_utils as _musa_fa_utils


def _musa_use_fp8_e4b15(device: int = 0) -> int:
    # XXX (MUSA): The e4b15 dtype is supported in later Triton versions, please refer to JIRA#79331
    return 0


_turboquant_decode._use_fp8_e4b15 = _musa_use_fp8_e4b15
_turboquant_store._use_fp8_e4b15 = _musa_use_fp8_e4b15
_turboquant_attn._use_fp8_e4b15 = _musa_use_fp8_e4b15
_turboquant_attn.get_flash_attn_version = _musa_fa_utils.get_flash_attn_version
_turboquant_attn.is_flash_attn_varlen_func_available = (
    _musa_fa_utils.is_flash_attn_varlen_func_available
)
_turboquant_attn._HAS_FLASH_ATTN = _musa_fa_utils.is_flash_attn_varlen_func_available()


class MUSATurboQuantAttentionImpl(_turboquant_attn.TurboQuantAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fa_version = _musa_fa_utils.get_flash_attn_version(
            head_size=self.head_size
        )

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        return _musa_fa_utils.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
        )


@register_backend(AttentionBackendEnum.TURBOQUANT)
class MUSATurboQuantAttentionBackend(_turboquant_attn.TurboQuantAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_impl_cls() -> type["MUSATurboQuantAttentionImpl"]:
        return MUSATurboQuantAttentionImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 3

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
        if use_mla:
            return "TurboQuant is not supported for MLA attention on MUSA"
        if has_sink:
            return "TurboQuant is not supported with attention sinks on MUSA"
        if use_sparse:
            return "TurboQuant is not supported for sparse attention on MUSA"
        return super().supports_combination(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
            use_mm_prefix=use_mm_prefix,
            device_capability=device_capability,
        )
