from __future__ import annotations

import pytest
import torch

import mate
from mate.testing import supported_musa_compute_capability


def _hash_topk_ref(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    routed_ids = tid2eid[input_ids.long()].to(torch.int64)
    routed_scores = (
        torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
    )
    routed_scores = routed_scores / routed_scores.sum(dim=-1, keepdim=True).clamp_min(
        1e-20
    )
    if num_fused_shared_experts == 0:
        return routed_scores.to(torch.float32), routed_ids
    shared_ids = torch.arange(
        router_logits.shape[1],
        router_logits.shape[1] + num_fused_shared_experts,
        dtype=torch.int64,
        device=router_logits.device,
    ).expand(router_logits.shape[0], -1)
    shared_weights = torch.full(
        (router_logits.shape[0], num_fused_shared_experts),
        1.0 / routed_scaling_factor,
        dtype=torch.float32,
        device=router_logits.device,
    )
    return torch.cat(
        [routed_scores.to(torch.float32), shared_weights], dim=1
    ), torch.cat([routed_ids, shared_ids], dim=1)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("tid_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_fused_shared_experts", [0, 1])
def test_hash_topk_sqrtsoftplus_hash_id_router_matches_reference(
    tid_dtype: torch.dtype,
    input_dtype: torch.dtype,
    num_fused_shared_experts: int,
) -> None:
    device = torch.device("musa")
    num_tokens = 257
    num_experts = 256
    topk = 6
    num_tid = 32000
    scaling = 2.0
    router_logits = torch.linspace(
        -4.0,
        4.0,
        steps=num_tokens * num_experts,
        device=device,
        dtype=torch.float32,
    ).reshape(num_tokens, num_experts)
    input_ids = (
        torch.arange(num_tokens, device=device, dtype=torch.int64) % num_tid
    ).to(input_dtype)
    tid2eid = (
        (
            torch.arange(num_tid * topk, device=device, dtype=torch.int64).reshape(
                num_tid, topk
            )
            * 37
        )
        % num_experts
    ).to(tid_dtype)

    weights, ids = mate.hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=scaling,
    )
    ref_weights, ref_ids = _hash_topk_ref(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts,
        scaling,
    )

    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int64
    torch.testing.assert_close(weights.cpu(), ref_weights.cpu(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(ids.cpu(), ref_ids.cpu(), rtol=0, atol=0)


@supported_musa_compute_capability([31])
def test_hash_topk_preserves_strided_inputs_without_materialization() -> None:
    device = torch.device("musa")
    num_tokens = 17
    num_experts = 32
    topk = 4
    num_tid = 64
    router_storage = torch.linspace(
        -3.0,
        3.0,
        steps=num_tokens * num_experts * 2 + 11,
        device=device,
        dtype=torch.float32,
    )
    router_logits = torch.as_strided(
        router_storage,
        (num_tokens, num_experts),
        (num_experts * 2, 2),
        storage_offset=5,
    )
    input_storage = (
        torch.arange(num_tokens * 3 + 5, device=device, dtype=torch.int64) % num_tid
    )
    input_ids = torch.as_strided(input_storage, (num_tokens,), (3,), storage_offset=2)
    tid_storage = (
        torch.arange(num_tid * topk * 2 + 13, device=device, dtype=torch.int64) * 11
    ) % num_experts
    tid2eid = torch.as_strided(
        tid_storage,
        (num_tid, topk),
        (topk * 2, 2),
        storage_offset=7,
    )

    weights, ids = mate.hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=2,
        routed_scaling_factor=4.0,
    )
    ref_weights, ref_ids = _hash_topk_ref(router_logits, input_ids, tid2eid, 2, 4.0)
    torch.testing.assert_close(weights.cpu(), ref_weights.cpu(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(ids.cpu(), ref_ids.cpu(), rtol=0, atol=0)
