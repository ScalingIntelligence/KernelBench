from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from mate.jit.runtime import ffi_to_torch
from mate.jit.mega_moe import (
    MegaMoEKernelConfig,
    get_block_m_for_mega_moe,
    get_fp8_fp8_mega_moe_stage1_module,
    get_fp8_fp8_mega_moe_stage2_module,
    get_mega_moe_runtime_utils_module,
)
from mate.mate_runtime import resolve_num_mps


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _get_buffer_layout(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
):
    utils = get_mega_moe_runtime_utils_module()
    num_bytes, slice_input_buffers = utils.get_function("mega_moe_get_buffer_layout")(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    return int(num_bytes), slice_input_buffers


def _view_fp8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).view(torch.float8_e4m3fn).view(tensor.shape)


class _SymmHandle:
    def __init__(self, buffer_ptrs_tensor: torch.Tensor) -> None:
        self.buffer_ptrs_tensor = buffer_ptrs_tensor

    @property
    def buffer_ptrs(self) -> List[int]:
        return [int(value) for value in self.buffer_ptrs_tensor.tolist()]


class MegaMoESymmBuffer:
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        use_fp8_dispatch: bool = True,
        activation: str = "swiglu",
    ) -> None:
        assert group.size() <= 8, (
            "MegaMoE symmetric buffer supports intranode groups only"
        )
        assert num_experts % group.size() == 0
        assert activation == "swiglu"
        assert use_fp8_dispatch
        assert hidden % 128 == 0, "hidden must be divisible by 128"
        assert intermediate_hidden % 256 == 0, (
            "intermediate_hidden must be divisible by 256"
        )

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        num_bytes, slice_input_buffers = _get_buffer_layout(
            group.size(),
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        self.num_bytes = num_bytes
        self.buffer = torch.empty(num_bytes, dtype=torch.int8, device="musa")
        views = list(ffi_to_torch(slice_input_buffers(self.buffer)))
        for index in (0, 4, 7):
            views[index] = _view_fp8(views[index])
        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.l1_acts,
            self.l1_acts_sf,
            self.l1_topk_weights,
            self.l2_acts,
            self.l2_acts_sf,
            self.combine_acts,
        ) = views

        if group.size() == 1:
            ptrs = torch.tensor(
                [self.buffer.data_ptr()], dtype=torch.int64, device="cpu"
            )
        else:
            utils = get_mega_moe_runtime_utils_module()
            handle_size = int(utils.get_function("mega_moe_get_ipc_handle_size")())
            local_handle = torch.empty(handle_size, dtype=torch.uint8, device="cpu")
            utils.get_function("mega_moe_get_ipc_handle")(self.buffer, local_handle)

            gathered: List[Optional[bytes]] = [None] * group.size()
            dist.all_gather_object(gathered, bytes(local_handle.tolist()), group)
            handles = torch.empty(
                (group.size(), handle_size), dtype=torch.uint8, device="cpu"
            )
            for idx, handle in enumerate(gathered):
                assert handle is not None and len(handle) == handle_size
                handles[idx].copy_(
                    torch.tensor(list(handle), dtype=torch.uint8, device="cpu")
                )

            ptrs = torch.empty(group.size(), dtype=torch.int64, device="cpu")
            utils.get_function("mega_moe_open_ipc_handles")(
                handles, self.buffer, group.rank(), ptrs
            )

        self.handle = _SymmHandle(ptrs)
        self.y = None
        self.buffer.zero_()
        dist.barrier(group=group)
        torch.musa.synchronize()

    def destroy(self) -> None:
        if getattr(self, "handle", None) is not None and self.group is not None:
            if self.group.size() > 1:
                utils = get_mega_moe_runtime_utils_module()
                utils.get_function("mega_moe_close_ipc_handles")(
                    self.handle.buffer_ptrs_tensor,
                    self.group.rank(),
                )
            self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.l1_acts = None
        self.l1_acts_sf = None
        self.l1_topk_weights = None
        self.l2_acts = None
        self.l2_acts_sf = None
        self.combine_acts = None
        self.y = None


SymmBuffer = MegaMoESymmBuffer


def get_symm_buffer_for_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
) -> MegaMoESymmBuffer:
    block_m = get_block_m_for_mega_moe(
        group.size(),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
    )
    num_max_tokens_per_rank = _align(num_max_tokens_per_rank, block_m)
    return MegaMoESymmBuffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        use_fp8_dispatch,
        activation,
    )


def _check_grouped_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    assert tensor.dim() == 3, f"{name} must be a 3D grouped tensor"
    assert tensor.dtype == dtype, f"{name} must use dtype {dtype}, got {tensor.dtype}"
    assert tensor.is_contiguous(), f"{name} must be contiguous"


def _check_scale_tensor(
    name: str, tensor: torch.Tensor, expected_shape: Tuple[int, int, int]
) -> None:
    assert tensor.dim() == 3, f"{name} must be a 3D grouped scale tensor"
    assert tensor.dtype == torch.float32, (
        f"{name} must use float32 scales, got {tensor.dtype}"
    )
    assert tensor.is_contiguous(), f"{name} must be contiguous"
    assert tuple(tensor.shape) == expected_shape, (
        f"{name} shape must be {expected_shape}, got {tuple(tensor.shape)}"
    )


def _check_logical_weight_pair_for_transform(
    name: str,
    weights: Tuple[torch.Tensor, torch.Tensor],
    *,
    require_even_n: bool,
) -> Tuple[int, int, int]:
    assert isinstance(weights, tuple) and len(weights) == 2, (
        f"{name} must be a (weight, scale) tuple"
    )
    weight, scale = weights
    _check_grouped_tensor(f"{name}[0]", weight, torch.float8_e4m3fn)
    num_groups, n, k = weight.shape
    if require_even_n:
        assert n % 2 == 0, f"{name}[0] N dimension must be even for gate/up weights"
        assert n % 256 == 0, f"{name}[0] N dimension must be a multiple of 256"
    assert n % 128 == 0, f"{name}[0] N dimension must be a multiple of 128"
    assert k % 128 == 0, f"{name}[0] K dimension must be a multiple of 128"
    _check_scale_tensor(f"{name}[1]", scale, (num_groups, n // 128, k // 128))
    return num_groups, n, k


def _interleave_l1_weights(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    def interleave_weight(tensor: torch.Tensor, gran: int = 64) -> torch.Tensor:
        g, n, *rest = tensor.shape
        half = n // 2
        gate = tensor[:, :half].reshape(g, half // gran, gran, *rest)
        up = tensor[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.stack((gate, up), dim=2).reshape(g, n, *rest).contiguous()

    def expand_scale(scale: torch.Tensor, gran: int = 64) -> torch.Tensor:
        g, n_blocks, *rest = scale.shape
        half_blocks = n_blocks // 2
        chunks = torch.arange(0, half_blocks * 128, gran, device=scale.device)
        gate_idx = chunks // 128
        up_idx = gate_idx + half_blocks
        idx = torch.stack((gate_idx, up_idx), dim=1).reshape(-1)
        return scale.index_select(1, idx).contiguous()

    weight, scale = l1_weights
    return interleave_weight(weight), expand_scale(scale)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
]:
    _check_logical_weight_pair_for_transform(
        "l1_weights", l1_weights, require_even_n=True
    )
    l1_w, l1_sf = _interleave_l1_weights(l1_weights)
    if l2_weights is None:
        return l1_w, l1_sf
    _check_logical_weight_pair_for_transform(
        "l2_weights", l2_weights, require_even_n=False
    )
    return (l1_w, l1_sf), l2_weights


def _check_transformed_mega_moe_weights(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: MegaMoESymmBuffer,
) -> None:
    assert isinstance(l1_weights, tuple) and len(l1_weights) == 2
    assert isinstance(l2_weights, tuple) and len(l2_weights) == 2

    l1_weight, l1_scale = l1_weights
    l2_weight, l2_scale = l2_weights
    _check_grouped_tensor("l1_weights[0]", l1_weight, torch.float8_e4m3fn)
    _check_grouped_tensor("l2_weights[0]", l2_weight, torch.float8_e4m3fn)

    num_experts_per_rank = sym_buffer.num_experts // sym_buffer.group.size()
    expected_l1_shape = (
        num_experts_per_rank,
        sym_buffer.intermediate_hidden * 2,
        sym_buffer.hidden,
    )
    expected_l2_shape = (
        num_experts_per_rank,
        sym_buffer.hidden,
        sym_buffer.intermediate_hidden,
    )
    assert tuple(l1_weight.shape) == expected_l1_shape, (
        f"l1_weights[0] shape must be {expected_l1_shape}, got {tuple(l1_weight.shape)}"
    )
    assert tuple(l2_weight.shape) == expected_l2_shape, (
        f"l2_weights[0] shape must be {expected_l2_shape}, got {tuple(l2_weight.shape)}"
    )

    _check_scale_tensor(
        "l1_weights[1]",
        l1_scale,
        (
            num_experts_per_rank,
            (sym_buffer.intermediate_hidden * 2) // 64,
            sym_buffer.hidden // 128,
        ),
    )
    _check_scale_tensor(
        "l2_weights[1]",
        l2_scale,
        (
            num_experts_per_rank,
            sym_buffer.hidden // 128,
            sym_buffer.intermediate_hidden // 128,
        ),
    )


def _kernel_config(
    sym_buffer: MegaMoESymmBuffer, y: torch.Tensor, fast_math: bool
) -> MegaMoEKernelConfig:
    return MegaMoEKernelConfig(
        num_ranks=sym_buffer.group.size(),
        num_experts=sym_buffer.num_experts,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=sym_buffer.num_topk,
        hidden=sym_buffer.hidden,
        intermediate_hidden=sym_buffer.intermediate_hidden,
        num_mps=resolve_num_mps(y.device),
        fast_math=fast_math,
    )


def fp8_fp8_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: MegaMoESymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
) -> None:
    assert activation == "swiglu"
    assert y.dim() == 2
    assert y.dtype == torch.bfloat16
    assert y.is_contiguous()
    assert y.size(1) == sym_buffer.hidden
    assert y.device == sym_buffer.buffer.device
    _check_transformed_mega_moe_weights(l1_weights, l2_weights, sym_buffer)

    config = _kernel_config(sym_buffer, y, fast_math)
    num_tokens = y.size(0)
    clamp_value = float("inf") if activation_clamp is None else float(activation_clamp)

    stage1_name, stage1_module = get_fp8_fp8_mega_moe_stage1_module(config)
    stage1_module.get_function(stage1_name)(
        sym_buffer.l1_acts,
        sym_buffer.l1_acts_sf,
        sym_buffer.l2_acts,
        sym_buffer.l2_acts_sf,
        l1_weights[0],
        l1_weights[1],
        sym_buffer.handle.buffer_ptrs_tensor,
        sym_buffer.group.rank(),
        num_tokens,
        clamp_value,
    )

    _ = cumulative_local_expert_recv_stats

    stage2_name, stage2_module = get_fp8_fp8_mega_moe_stage2_module(config)
    stage2_module.get_function(stage2_name)(
        sym_buffer.l2_acts,
        sym_buffer.l2_acts_sf,
        l2_weights[0],
        l2_weights[1],
        y,
        sym_buffer.handle.buffer_ptrs_tensor,
        sym_buffer.group.rank(),
        num_tokens,
    )
    sym_buffer.y = y


def get_token_alignment_for_mega_moe() -> int:
    return get_block_m_for_mega_moe(1, 1, 1, 1)


__all__ = [
    "SymmBuffer",
    "fp8_fp8_mega_moe",
    "get_symm_buffer_for_mega_moe",
    "transform_weights_for_mega_moe",
]
