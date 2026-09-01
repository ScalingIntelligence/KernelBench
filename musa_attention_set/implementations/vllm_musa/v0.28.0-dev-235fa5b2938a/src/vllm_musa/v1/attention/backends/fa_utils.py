# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_musa():
    from flash_attn_interface import (  # noqa: F401
        flash_attn_varlen_func as _musa_flash_attn_varlen_func,
        flash_attn_with_kvcache,
        get_scheduler_metadata,
    )

    from vllm import _custom_ops as ops
    from vllm_musa import _custom_ops as musa_ops

    _MUSA_OPS_NAMESPACE = getattr(torch.ops, "_C_musa_ops", None)
    _HAS_NATIVE_RESHAPE_CACHE_FLASH = hasattr(
        _MUSA_OPS_NAMESPACE, "musa_reshape_and_cache_flash_nhd"
    )

    def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:
        dropout_p = kwargs.pop("dropout_p", 0.0)
        if dropout_p not in (0, 0.0, None):
            raise ValueError("MUSA flash attention does not support non-zero dropout.")

        kwargs.pop("fa_version", None)
        alibi_slopes = kwargs.pop("alibi_slopes", None)
        if alibi_slopes is not None:
            raise ValueError("MUSA flash attention does not support ALiBi.")

        return _musa_flash_attn_varlen_func(*args, **kwargs)

    def _can_use_musa_reshape_and_cache_flash_nhd(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    ) -> bool:
        # Keep this guard cheap: the native op has full TORCH_CHECK coverage.
        # Python only filters known fallback cases that are valid upstream.
        head_size = key.shape[2] if key.dim() == 3 else None
        return (
            _HAS_NATIVE_RESHAPE_CACHE_FLASH
            and kv_cache_dtype in ("auto", "float16", "bfloat16")
            and key.dtype in (torch.float16, torch.bfloat16)
            and value.dtype == key.dtype
            and key_cache.dtype == key.dtype
            and value_cache.dtype == value.dtype
            and key.dim() == 3
            and value.dim() == 3
            and head_size is not None
            and head_size % 8 == 0
            and key.stride(2) == 1
            and value.stride(2) == 1
            and key.stride(1) == head_size
            and value.stride(1) == value.shape[2]
            and k_scale.numel() == 1
            and v_scale.numel() == 1
            and key_cache.dim() == 4
            and value_cache.dim() == 4
            and key_cache.stride(3) == 1
            and value_cache.stride(3) == 1
            and key_cache.stride(2) == key_cache.shape[3]
            and value_cache.stride(2) == value_cache.shape[3]
            and key_cache.stride(1) == key_cache.shape[2] * key_cache.shape[3]
            and (value_cache.stride(1) == value_cache.shape[2] * value_cache.shape[3])
            # The native kernel receives one page/block stride and applies it
            # to both K and V caches.  Do not dispatch it for independently
            # padded/as_strided cache tensors.
            and key_cache.stride(0) == value_cache.stride(0)
            and key_cache.stride(1) == value_cache.stride(1)
        )

    def reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    ) -> None:
        if _can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        ):
            musa_ops.musa_reshape_and_cache_flash_nhd(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
            )
            return
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> int | None:
    logger.info_once("MUSA platform use FLASH_ATTN with version 3.")
    return 3


def flash_attn_supports_fp8() -> bool:
    # MATE FMHA supports e4m3 Q/K/V with per-(batch, KV-head) descales.
    return True


def flash_attn_supports_sinks() -> bool:
    return True


def flash_attn_supports_mla() -> bool:
    return True


def is_flash_attn_varlen_func_available() -> bool:
    return "flash_attn_varlen_func" in globals()
