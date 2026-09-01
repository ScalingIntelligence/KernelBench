import torch
from mate.moe_fused_gate import moe_fused_gate
from triton.testing import do_bench


def biased_grouped_topk_org_fuse_kernel(
    scores, bias, num_expert_group, topk_group, topk
):
    return moe_fused_gate(
        scores,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=2.5,
        renormalize=True,
        apply_routed_scaling_factor_on_output=False,
        num_token_non_padded=-1,
        static_index_map=None,
        dynamic_index_map=None,
        dynamic_index_map_valid=None,
        random_index=None,
        num_physical_experts=0,
        map_policy=0,
    )


seq_length_range = list(range(1, 10)) + [
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
    16384,
    32768,
    65536,
    65536 * 2,
]


def benchmark(seq_length, num_experts, num_expert_group, topk_group, topk):
    dtype = torch.float32
    device = torch.device("musa")

    scores = torch.randn((seq_length, num_experts), device=device, dtype=dtype)
    bias = torch.rand(num_experts, device=device, dtype=dtype)

    quantiles = [0.5]

    ms = do_bench(
        lambda: biased_grouped_topk_org_fuse_kernel(
            scores, bias, num_expert_group, topk_group, topk
        ),
        quantiles=quantiles,
    )

    time_ms = float(ms)
    output_bytes = seq_length * topk * 4  # float32 output
    indices_bytes = seq_length * topk * 4  # int32 indices
    total_bytes = (
        scores.numel() * scores.element_size()
        + bias.numel() * bias.element_size()
        + output_bytes
        + indices_bytes
    )
    bandwidth_gb_s = total_bytes / (time_ms / 1000.0) / 1e9

    return time_ms, bandwidth_gb_s


if __name__ == "__main__":
    expert_configs = [
        (256, 8, 4, 8),
        (384, 8, 4, 8),
        (160, 1, 2, 8),
        (384, 1, 2, 8),
    ]
    print(
        "num_experts, num_expert_group, topk_group, topk, seq_length, p50_ms, bandwidth_GB_s"
    )
    for num_experts, num_expert_group, topk_group, topk in expert_configs:
        for seq_length in seq_length_range:
            p50_ms, bandwidth_gb_s = benchmark(
                seq_length, num_experts, num_expert_group, topk_group, topk
            )
            print(
                f"{num_experts}, {num_expert_group}, {topk_group}, {topk}, "
                f"{seq_length}, {p50_ms:.3f}, {bandwidth_gb_s:.3f}"
            )
