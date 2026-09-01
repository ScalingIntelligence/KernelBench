import pytest
import torch
from mate.moe_fused_gate import moe_fused_gate
from typing import Optional
from mate.testing import supported_musa_compute_capability


def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.sort(group_scores, dim=-1, descending=True, stable=True)[1][
        :, :topk_group
    ]

    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]

    topk_ids = torch.sort(tmp_scores, dim=-1, descending=True, stable=True)[1][:, :topk]
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
    if apply_routed_scaling_factor_on_output:
        topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    return topk_weights, topk_ids


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10))
    + [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,  # 32768
    ],
)
@pytest.mark.parametrize(
    "params",
    [
        (128, 4, 2, 4),
        (256, 8, 4, 8),  # deepseek v3
        (384, 8, 4, 8),
        (160, 4, 2, 4),
        (512, 8, 4, 8),
        (160, 1, 2, 4),
        (384, 1, 1, 4),
        (256, 1, 4, 8),
    ],
)
@pytest.mark.parametrize("num_fused_shared_experts", [0])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_moe_fused_gate_combined(
    seq_length,
    params,
    num_fused_shared_experts,
    apply_routed_scaling_factor_on_output,
    renormalize,
):
    num_experts, num_expert_group, topk_group, topk = params
    dtype = torch.float32

    torch.manual_seed(seq_length)
    device = "musa:0"
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device=device)
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device=device)
    topk = topk + num_fused_shared_experts

    output, indices = moe_fused_gate(
        tensor,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=2.5,
        renormalize=renormalize,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    ref_output, ref_indices = biased_grouped_topk_impl(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # When num_fused_shared_experts > 0, ignore the comparison of the last topk dimension
    if num_fused_shared_experts > 0:
        original_indices = indices.clone()
        original_ref_indices = ref_indices.clone()

        indices = indices[:, :-1]
        ref_indices = ref_indices[:, :-1]

        valid_min = num_experts
        valid_max = num_experts + num_fused_shared_experts
        shared_indices = original_indices[:, -1]
        shared_ref_indices = original_ref_indices[:, -1]
        if shared_indices is not None:
            assert torch.all(
                (shared_indices >= valid_min) & (shared_indices < valid_max)
            ), (
                f"Shared expert indices out of range: found values outside [{valid_min}, {valid_max})"
            )
        if shared_ref_indices is not None:
            assert torch.all(
                (shared_ref_indices >= valid_min) & (shared_ref_indices < valid_max)
            ), (
                f"Shared expert reference indices out of range: found values outside [{valid_min}, {valid_max})"
            )

    idx_check = torch.allclose(
        ref_indices.sort()[0].to(torch.int32),
        indices.sort()[0].to(torch.int32),
        rtol=1e-04,
        atol=1e-05,
    )
    output_check = torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-02,
        atol=1e-03,
    )

    assert idx_check, (
        f"Indices mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}, num_fused_shared_experts {num_fused_shared_experts}"
    )
    assert output_check, (
        f"Output mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}, num_fused_shared_experts {num_fused_shared_experts}"
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("seq_length", [1, 64, 2904])
@pytest.mark.parametrize(
    "params",
    [
        (256, 1, 1, 8),
        (256, 8, 4, 8),
        (384, 1, 1, 8),
        (384, 8, 4, 8),
        (160, 1, 1, 8),
    ],
)
def test_moe_fused_gate_small_scores_large_bias(seq_length, params):
    num_experts, num_expert_group, topk_group, topk = params
    dtype = torch.float32
    device = "musa:0"

    torch.manual_seed(seq_length + num_experts)
    tensor = torch.full(
        (seq_length, num_experts), -16.0, dtype=dtype, device=device
    )  # sigmoid(-16) ~= 1.125e-7
    tensor += torch.rand_like(tensor) * 0.5
    scores = tensor.clone()

    bias = torch.full((num_experts,), 11.0, dtype=dtype, device=device)
    bias += torch.rand_like(bias) * 0.5

    output, indices = moe_fused_gate(
        tensor,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=2.5,
        renormalize=True,
        apply_routed_scaling_factor_on_output=False,
    )
    ref_output, ref_indices = biased_grouped_topk_impl(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=0,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=False,
    )

    assert not torch.isnan(output).any()
    assert torch.allclose(
        ref_indices.sort()[0].to(torch.int32),
        indices.sort()[0].to(torch.int32),
        rtol=1e-04,
        atol=1e-05,
    )
    assert torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-02,
        atol=1e-03,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    "seq_length",
    [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
)
@pytest.mark.parametrize(
    "params",
    [
        (128, 4, 2, 4),
        (256, 8, 4, 8),  # deepseek v3
        (384, 8, 4, 8),
    ],
)
@pytest.mark.parametrize("map_policy", [2])
@pytest.mark.parametrize("num_token_ratio", [0.8, -1])
def test_moe_fused_gate_with_post_processing(
    seq_length, params, num_token_ratio, map_policy
):
    num_experts, num_expert_group, topk_group, topk = params
    dtype = torch.float32

    torch.manual_seed(seq_length)
    device = "musa:0"
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device=device)
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device=device)
    num_token_non_padded = int(seq_length * num_token_ratio)

    num_physical_experts = 4
    dynamic_index_map = (
        torch.randperm(num_experts * num_physical_experts, dtype=torch.int32)
        .to(device)
        .reshape(num_experts, num_physical_experts)
    )

    dynamic_index_map_valid = torch.randint(
        1, num_physical_experts, (num_experts,), device=device, dtype=torch.int32
    )
    random_index = torch.randint(
        0, 65536, (seq_length, topk), device=device, dtype=torch.int32
    )
    static_index_map = torch.randperm(num_experts, dtype=torch.int32).to(device)

    output, indices = moe_fused_gate(
        tensor,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=2.5,
        renormalize=True,
        apply_routed_scaling_factor_on_output=False,
        num_token_non_padded=num_token_non_padded,
        static_index_map=static_index_map,
        dynamic_index_map=dynamic_index_map,
        dynamic_index_map_valid=dynamic_index_map_valid,
        random_index=random_index,
        num_physical_experts=num_physical_experts,
        map_policy=map_policy,
    )

    ref_output, ref_indices = biased_grouped_topk_impl(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=0,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=False,
    )

    if map_policy == 1:
        ref_indices = static_index_map[ref_indices]
    elif map_policy == 2:
        chosen_idx = random_index % dynamic_index_map_valid[ref_indices]
        ref_indices = dynamic_index_map[ref_indices, chosen_idx]

    if num_token_non_padded > 0:
        ref_indices[num_token_non_padded:] = -1
        output = output[:num_token_non_padded]
        ref_output = ref_output[:num_token_non_padded]

    assert torch.equal(indices, ref_indices), (
        "Dynamic map indices mismatch for non-padded tokens"
    )
    assert torch.allclose(
        output,
        ref_output,
        rtol=1e-02,
        atol=1e-03,
    ), "Output mismatch for non-padded tokens under dynamic map"
