import functools

import torch
from mate.api_logging import mate_api
from mate.jit.moe_fused_gate import get_moe_fused_gate_module


def _contiguous_or_none(x: torch.Tensor | None) -> torch.Tensor | None:
    return None if x is None else x.contiguous()


@functools.cache
def _get_module():
    return get_moe_fused_gate_module()


@mate_api
def moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    renormalize: bool,
    apply_routed_scaling_factor_on_output: bool,
    num_token_non_padded: int = 0,
    static_index_map: torch.Tensor | None = None,
    dynamic_index_map: torch.Tensor | None = None,
    dynamic_index_map_valid: torch.Tensor | None = None,
    random_index: torch.Tensor | None = None,
    num_physical_experts: int = 0,
    map_policy: int = 0,
):
    """
    Perform fused MoE gating (sigmoid + grouped topk) and optional expert index mapping.

    Parameters
    ----------
    input : Tensor
        Gating input tensor with shape ``(num_tokens, num_experts)``.
    bias : Tensor
        Bias tensor with shape ``(num_experts,)``.
    num_expert_group : int
        Number of expert groups.
    topk_group : int
        Number of groups selected per token.
    topk : int
        Number of experts selected per token (including fused shared experts if any).
    num_fused_shared_experts : int
        Number of fused shared experts to append.
    routed_scaling_factor : float
        Scaling factor applied to routed weights.
    renormalize : bool
        Whether to renormalize topk weights to sum to 1.
    apply_routed_scaling_factor_on_output : bool
        Whether to apply routed scaling on output weights.
    num_token_non_padded : int, optional
        Number of valid (non-padded) tokens. Default is 0 means all tokens are valid.
    static_index_map : Optional[Tensor]
        Static expert index mapping table. Required when ``map_policy=1``.
    dynamic_index_map : Optional[Tensor]
        Dynamic expert index mapping table. Required when ``map_policy=2``.
    dynamic_index_map_valid : Optional[Tensor]
        Valid counts per expert for dynamic mapping. Required when ``map_policy=2``.
    random_index : Optional[Tensor]
        Random index tensor for dynamic mapping. Required when ``map_policy=2``.
    num_physical_experts : int, optional
        Number of physical experts used for mapping. Default is 0.
    map_policy : int, optional
        Mapping policy: 0=no map, 1=static map, 2=dynamic map. Default is 0.

    Returns
    -------
    Tuple[Tensor, Tensor]
        - topk_weights: float tensor of shape ``(num_tokens, topk)``
        - topk_indices: int tensor of shape ``(num_tokens, topk)``
    """
    input = input.contiguous()
    bias = bias.contiguous()
    topk_group = min(topk_group, num_expert_group)
    static_index_map = _contiguous_or_none(static_index_map)
    dynamic_index_map = _contiguous_or_none(dynamic_index_map)
    dynamic_index_map_valid = _contiguous_or_none(dynamic_index_map_valid)
    random_index = _contiguous_or_none(random_index)

    output = torch.empty(
        (input.shape[0], topk), dtype=torch.float32, device=input.device
    )
    indices = torch.empty(
        (input.shape[0], topk), dtype=torch.int32, device=input.device
    )

    _get_module().run(
        output,
        indices,
        input,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
        renormalize,
        apply_routed_scaling_factor_on_output,
        num_token_non_padded,
        static_index_map,
        dynamic_index_map,
        dynamic_index_map_valid,
        random_index,
        num_physical_experts,
        map_policy,
    )
    return output, indices
