from __future__ import annotations

from typing import Optional

import torch

from mate.api_logging import mate_api
from mate.hash_topk_kernels.tilelang.hash_topk import run_hash_topk

__all__ = ["hash_topk"]


def _validate_hash_topk_inputs(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    scoring_func: str,
) -> tuple[int, int, int]:
    if scoring_func != "sqrtsoftplus":
        raise NotImplementedError(
            f"mate.hash_topk only supports scoring_func='sqrtsoftplus', got {scoring_func!r}."
        )
    if router_logits.dtype != torch.float32:
        raise ValueError(f"router_logits must be float32, got {router_logits.dtype}.")
    if router_logits.dim() != 2:
        raise ValueError(
            f"router_logits must have shape [num_tokens, num_experts], got {tuple(router_logits.shape)}."
        )
    if input_ids.dim() != 1:
        raise ValueError(f"input_ids must be 1D, got {tuple(input_ids.shape)}.")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"input_ids must be int32 or int64, got {input_ids.dtype}.")
    if tid2eid.dim() != 2:
        raise ValueError(
            f"tid2eid must have shape [vocab_size, topk], got {tuple(tid2eid.shape)}."
        )
    if tid2eid.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"tid2eid must be int32 or int64, got {tid2eid.dtype}.")
    if input_ids.shape[0] != router_logits.shape[0]:
        raise ValueError(
            f"input_ids length {input_ids.shape[0]} must match router_logits tokens {router_logits.shape[0]}."
        )
    if (
        input_ids.device != router_logits.device
        or tid2eid.device != router_logits.device
    ):
        raise ValueError(
            "router_logits, input_ids, and tid2eid must be on the same device."
        )
    if num_fused_shared_experts < 0:
        raise ValueError(
            f"num_fused_shared_experts must be non-negative, got {num_fused_shared_experts}."
        )
    if routed_scaling_factor == 0:
        raise ValueError("routed_scaling_factor must be non-zero.")

    return router_logits.shape[0], router_logits.shape[1], tid2eid.shape[1]


@mate_api
def hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
    *,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Run DeepSeek-style hash-id MoE topk routing.

    This API implements the hash router path used by early DeepSeek V4 MoE
    layers on MUSA.

    Parameters
    ----------
    router_logits:
        Float32 tensor with shape ``[num_tokens, num_experts]``. Strided input
        views are supported.
    input_ids:
        1D int32/int64 tensor with ``num_tokens`` entries. Each value indexes
        the first dimension of ``tid2eid``. Strided input views are supported.
    tid2eid:
        Int32/int64 lookup table with shape ``[vocab_size, topk]``. The selected
        routed expert ids are ``tid2eid[input_ids]``. Strided input views are
        supported.
    num_fused_shared_experts:
        Number of shared experts appended after the hash-selected routed
        experts. Shared ids are ``num_experts + i``.
    routed_scaling_factor:
        Divisor for appended shared expert weights. Shared weights are
        ``1 / routed_scaling_factor``.
    scoring_func:
        Routing score transform. Only ``"sqrtsoftplus"`` is supported.
    topk_weights:
        Optional contiguous float32 output buffer with shape
        ``[num_tokens, topk + num_fused_shared_experts]``.
    topk_ids:
        Optional contiguous int64 output buffer with shape
        ``[num_tokens, topk + num_fused_shared_experts]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(topk_weights, topk_ids)``. ``topk_weights`` is float32 and
        ``topk_ids`` is int64, both with shape
        ``[num_tokens, topk + num_fused_shared_experts]``.

    The routed expert ids are ``tid2eid[input_ids]``. Routing weights are
    computed from ``sqrt(softplus(router_logits))`` gathered at those expert
    ids, then normalized across the routed experts. If
    ``num_fused_shared_experts`` is non-zero, shared experts are appended with
    ids ``num_experts + i`` and weights ``1 / routed_scaling_factor``.
    """
    num_tokens, num_experts, topk = _validate_hash_topk_inputs(
        router_logits,
        input_ids,
        tid2eid,
        int(num_fused_shared_experts),
        float(routed_scaling_factor),
        scoring_func,
    )
    output_topk = topk + int(num_fused_shared_experts)

    if router_logits.device.type != "musa":
        raise ValueError("hash_topk requires MUSA tensors.")

    if topk_weights is None:
        topk_weights = torch.empty(
            (num_tokens, output_topk), dtype=torch.float32, device=router_logits.device
        )
    if topk_ids is None:
        topk_ids = torch.empty(
            (num_tokens, output_topk), dtype=torch.int64, device=router_logits.device
        )
    if (
        topk_weights.shape != (num_tokens, output_topk)
        or topk_weights.dtype != torch.float32
    ):
        raise ValueError(
            f"topk_weights must be float32 with shape {(num_tokens, output_topk)}, "
            f"got shape={tuple(topk_weights.shape)} dtype={topk_weights.dtype}."
        )
    if not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous.")
    if topk_ids.shape != (num_tokens, output_topk) or topk_ids.dtype != torch.int64:
        raise ValueError(
            f"topk_ids must be int64 with shape {(num_tokens, output_topk)}, "
            f"got shape={tuple(topk_ids.shape)} dtype={topk_ids.dtype}."
        )
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous.")

    run_hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        topk_weights,
        topk_ids,
        num_fused_shared_experts=int(num_fused_shared_experts),
        routed_scaling_factor=float(routed_scaling_factor),
    )
    # Keep num_experts in validation scope and make the expected shared-id base explicit.
    _ = num_experts
    return topk_weights, topk_ids
