import argparse
import functools
import os
import random

import pytest
import torch
import torch.distributed as dist
import tilelang
import tilelang.language as T
import triton
import triton.language as tl

from mate import deep_gemm
from mate.testing import supported_musa_compute_capability
from utils import dist_print, init_dist


DEFAULT_TOKEN_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_TOKEN_SWEEP_STR = ",".join(str(token) for token in DEFAULT_TOKEN_SWEEP)
DEFAULT_NUM_EXPERTS_PER_RANK = 32
PYTEST_PROCESS_COUNTS = (1, 8)


def _parse_token_values(text: str) -> list[int]:
    token_values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not token_values:
        raise ValueError("token sweep must contain at least one token count")
    if any(token <= 0 for token in token_values):
        raise ValueError("token sweep only accepts positive token counts")
    return token_values


@functools.lru_cache(None)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    }
)
def _tilelang_deep_ep_style_per_token_cast_to_fp8_kernel(hidden: int):
    assert hidden % 128 == 0
    num_tokens = T.dynamic("num_tokens")
    num_scales = hidden // 128
    threads = ((hidden // 8 + 31) // 32) * 32

    @T.prim_func
    def kernel(
        x: T.Tensor((num_tokens, hidden), T.bfloat16),
        x_fp8: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        x_scale: T.Tensor((num_tokens, num_scales), T.float32),
    ):
        with T.Kernel(num_tokens, threads=threads) as (token_idx,):
            tid = T.get_thread_binding()
            lane = tid % 16
            group_idx = tid // 16
            values = T.alloc_local((8,), dtype=T.float32)
            amax = T.alloc_local((1,), dtype=T.float32)
            amax[0] = 1.0e-4

            if tid < hidden // 8:
                for j in T.unroll(8):
                    value = T.cast(x[token_idx, tid * 8 + j], T.float32)
                    values[j] = value
                    amax[0] = T.max(amax[0], T.abs(value))

                amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 8))
                amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 4))
                amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 2))
                amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 1))

                if lane == 0:
                    x_scale[token_idx, group_idx] = amax[0] / 448.0

                for j in T.unroll(8):
                    x_fp8[token_idx, tid * 8 + j] = T.cast(
                        values[j] * (448.0 / amax[0]), T.float8_e4m3fn
                    )

    return kernel.with_attr(
        "global_symbol",
        f"tilelang_deep_ep_style_per_token_cast_to_fp8_h{hidden}",
    )


def deep_ep_style_per_token_cast_to_fp8_tilelang(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert x.dtype == torch.bfloat16
    assert x.is_contiguous()
    assert x.size(1) % 128 == 0
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(
        (x.size(0), x.size(1) // 128), dtype=torch.float32, device=x.device
    )
    if x.size(0) != 0:
        _tilelang_deep_ep_style_per_token_cast_to_fp8_kernel(x.size(1))(
            x, x_fp8, x_scale
        )
    return x_fp8, x_scale


def per_block_cast_to_fp8(
    x: torch.Tensor, gran_m: int = 128, gran_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, k = x.shape
    assert m % gran_m == 0 and k % gran_k == 0
    x_view = x.view(m // gran_m, gran_m, k // gran_k, gran_k)
    scale = x_view.abs().float().amax(dim=(1, 3)).clamp_min(1e-4) / 448.0
    x_fp8 = (
        (x_view * (1.0 / scale[:, None, :, None]))
        .to(torch.float8_e4m3fn)
        .view(m, k)
        .contiguous()
    )
    return x_fp8, scale.contiguous()


@triton.jit
def _silu_and_mul_post_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)
    block_num_per_expert = tl.num_programs(1)
    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + hidden_dim_block_index * stride_output_scale_2
    )

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        )
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        output_s = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10) / fp8_max
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(
            output_ptr.dtype.element_ty
        )
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(output_scale_offs + token_index * stride_output_scale_1, output_s)


def sglang_deep_ep_style_swiglu_quantize_fp8(
    gateup: torch.Tensor,
    masked_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    intermediate_hidden = gateup.size(2) // 2
    out_fp8 = torch.empty(
        (gateup.size(0), gateup.size(1), intermediate_hidden),
        dtype=torch.float8_e4m3fn,
        device=gateup.device,
    )
    out_scale = torch.empty(
        (gateup.size(0), gateup.size(1), intermediate_hidden // 128),
        dtype=torch.float32,
        device=gateup.device,
    )
    block_num_per_expert = 64 if gateup.size(0) < 4 else 32
    block_n = 128
    finfo = torch.finfo(torch.float8_e4m3fn)
    _silu_and_mul_post_quant_kernel[
        (
            triton.cdiv(intermediate_hidden, block_n),
            block_num_per_expert,
            gateup.size(0),
        )
    ](
        gateup,
        *gateup.stride(),
        out_fp8,
        *out_fp8.stride(),
        out_scale,
        *out_scale.stride(),
        masked_m,
        intermediate_hidden,
        finfo.max,
        -finfo.max,
        BLOCK_N=block_n,
        NUM_STAGE=6,
        num_warps=1,
    )
    return out_fp8, out_scale


@functools.lru_cache(None)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    }
)
def _python_low_latency_combine_kernel(
    num_experts: int,
    num_topk: int,
    hidden: int,
):
    assert hidden % 8 == 0
    num_tokens = T.dynamic("num_tokens")
    threads = 512
    values_per_thread = 8

    @T.prim_func
    def kernel(
        expert_output: T.Tensor((num_experts, num_tokens, hidden), T.bfloat16),
        topk_idx: T.Tensor((num_tokens, num_topk), T.int64),
        topk_weights: T.Tensor((num_tokens, num_topk), T.float32),
        out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(num_tokens, threads=threads) as (token_idx,):
            tid = T.get_thread_binding()
            acc = T.alloc_local((values_per_thread,), T.float32)
            T.clear(acc)

            for hidden_base in T.serial(
                tid * values_per_thread,
                hidden,
                threads * values_per_thread,
            ):
                for topk_pos in T.unroll(num_topk):
                    expert_idx = topk_idx[token_idx, topk_pos]
                    if expert_idx >= 0:
                        weight = topk_weights[token_idx, topk_pos]
                        for elem_idx in T.unroll(values_per_thread):
                            value = T.cast(
                                expert_output[
                                    expert_idx,
                                    token_idx,
                                    hidden_base + elem_idx,
                                ],
                                T.float32,
                            )
                            acc[elem_idx] = T.call_pure_extern(
                                T.float32,
                                "__mt_fmaf_rn_f32",
                                value,
                                weight,
                                acc[elem_idx],
                            )

                for elem_idx in T.unroll(values_per_thread):
                    out[token_idx, hidden_base + elem_idx] = T.cast(
                        acc[elem_idx], T.bfloat16
                    )

    return kernel.with_attr(
        "global_symbol",
        f"python_low_latency_combine_e{num_experts}_k{num_topk}_h{hidden}",
    )


class PythonLowLatencyDispatchHandle:
    def __init__(
        self,
        recv_count: torch.Tensor,
        src_rank_by_row: torch.Tensor,
        src_token_by_row: torch.Tensor,
    ):
        self.recv_count = recv_count
        self.src_rank_by_row = src_rank_by_row
        self.src_token_by_row = src_token_by_row


class PythonLowLatencyDispatchSimulator:
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
    ):
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.num_ranks = dist.get_world_size(group=group)
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.hidden = hidden
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.num_ranks

    def dispatch(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor], torch.Tensor, PythonLowLatencyDispatchHandle
    ]:
        num_tokens, hidden = x_fp8.shape
        assert num_max_dispatch_tokens_per_rank == num_tokens
        assert num_max_dispatch_tokens_per_rank <= self.num_max_dispatch_tokens_per_rank
        assert hidden == self.hidden
        assert num_experts == self.num_experts
        assert x_fp8.dtype == torch.float8_e4m3fn
        assert x_scale.dtype == torch.float32
        assert topk_idx.dtype == torch.int64

        all_x = torch.empty(
            (self.num_ranks, num_tokens, hidden),
            dtype=x_fp8.dtype,
            device=x_fp8.device,
        )
        all_x_scale = torch.empty(
            (self.num_ranks, num_tokens, x_scale.size(1)),
            dtype=x_scale.dtype,
            device=x_scale.device,
        )
        all_topk_idx = torch.empty(
            (self.num_ranks, num_tokens, topk_idx.size(1)),
            dtype=topk_idx.dtype,
            device=topk_idx.device,
        )
        dist.all_gather_into_tensor(all_x, x_fp8.contiguous(), group=self.group)
        dist.all_gather_into_tensor(all_x_scale, x_scale.contiguous(), group=self.group)
        dist.all_gather_into_tensor(
            all_topk_idx, topk_idx.contiguous(), group=self.group
        )

        max_rows = num_tokens * self.num_ranks
        recv_x = torch.empty(
            (self.num_local_experts, max_rows, hidden),
            dtype=x_fp8.dtype,
            device=x_fp8.device,
        )
        recv_x_scale = torch.empty(
            (self.num_local_experts, max_rows, x_scale.size(1)),
            dtype=x_scale.dtype,
            device=x_scale.device,
        )
        recv_x_scale.zero_()

        recv_count = torch.empty(
            (self.num_local_experts,), dtype=torch.int32, device=x_fp8.device
        )
        src_rank_by_row = torch.full(
            (self.num_local_experts, max_rows),
            -1,
            dtype=torch.int64,
            device=x_fp8.device,
        )
        src_token_by_row = torch.full_like(src_rank_by_row, -1)

        expert_offset = self.rank * self.num_local_experts
        for local_expert_idx in range(self.num_local_experts):
            expert_idx = expert_offset + local_expert_idx
            src_rank_idx, src_token_idx, _ = torch.nonzero(
                all_topk_idx == expert_idx, as_tuple=True
            )
            count = src_rank_idx.numel()
            recv_count[local_expert_idx] = count
            if count == 0:
                continue
            for row_idx in range(count):
                src_rank = int(src_rank_idx[row_idx].item())
                src_token = int(src_token_idx[row_idx].item())
                recv_x[local_expert_idx, row_idx].copy_(all_x[src_rank, src_token])
                recv_x_scale[local_expert_idx, row_idx].copy_(
                    all_x_scale[src_rank, src_token]
                )
            src_rank_by_row[local_expert_idx, :count].copy_(src_rank_idx)
            src_token_by_row[local_expert_idx, :count].copy_(src_token_idx)

        handle = PythonLowLatencyDispatchHandle(
            recv_count=recv_count,
            src_rank_by_row=src_rank_by_row,
            src_token_by_row=src_token_by_row,
        )
        return (recv_x, recv_x_scale), recv_count, handle

    def combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: PythonLowLatencyDispatchHandle,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens, num_topk = topk_idx.shape
        hidden = x.size(2)
        assert hidden == self.hidden
        assert x.size(0) == self.num_local_experts
        assert topk_weights.shape == topk_idx.shape

        local_dense = torch.zeros(
            (self.num_local_experts, self.num_ranks, num_tokens, hidden),
            dtype=x.dtype,
            device=x.device,
        )
        for local_expert_idx in range(self.num_local_experts):
            count = int(handle.recv_count[local_expert_idx].item())
            if count == 0:
                continue
            for row_idx in range(count):
                src_rank = int(handle.src_rank_by_row[local_expert_idx, row_idx].item())
                src_token = int(
                    handle.src_token_by_row[local_expert_idx, row_idx].item()
                )
                local_dense[local_expert_idx, src_rank, src_token].copy_(
                    x[local_expert_idx, row_idx]
                )

        send_dense = local_dense.permute(1, 0, 2, 3).contiguous()
        expert_output = torch.empty_like(send_dense)
        dist.all_to_all_single(expert_output, send_dense, group=self.group)
        expert_output = expert_output.contiguous().view(
            self.num_experts, num_tokens, hidden
        )

        if out is None:
            out = torch.empty((num_tokens, hidden), dtype=x.dtype, device=x.device)
        _python_low_latency_combine_kernel(self.num_experts, num_topk, hidden)(
            expert_output, topk_idx, topk_weights, out
        )
        return out


def run_python_dispatch_deepgemm_reference(
    group: dist.ProcessGroup,
    dispatch_simulator: PythonLowLatencyDispatchSimulator,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
) -> torch.Tensor:
    num_tokens, hidden = x_fp8.shape
    num_ranks = group.size()
    intermediate_hidden = l2_weights[0].size(2)
    expected_m = (
        num_tokens * num_ranks * topk_idx.size(1) + num_experts
    ) // num_experts

    packed_recv_x, packed_recv_count, handle = dispatch_simulator.dispatch(
        x_fp8,
        x_scale,
        topk_idx,
        num_max_dispatch_tokens_per_rank,
        num_experts,
    )
    torch.musa.synchronize()
    dist.barrier(group=group)

    recv_x, recv_x_scale = packed_recv_x
    recv_x = recv_x.contiguous() if not recv_x.is_contiguous() else recv_x
    recv_x_scale = recv_x_scale.contiguous()
    masked_m = packed_recv_count.to(torch.int32)
    max_m = recv_x.size(1)
    recipe = (1, 128, 128)

    gateup_output = torch.empty(
        (l1_weights[0].size(0), max_m, intermediate_hidden * 2),
        dtype=torch.bfloat16,
        device=x_fp8.device,
    )
    gateup_output.zero_()
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (recv_x, recv_x_scale),
        l1_weights,
        gateup_output,
        masked_m,
        expected_m,
        recipe=recipe,
        backend="mutlass",
    )

    down_input, down_input_scale = sglang_deep_ep_style_swiglu_quantize_fp8(
        gateup_output, masked_m
    )
    del gateup_output

    down_output = torch.empty(
        (l2_weights[0].size(0), max_m, hidden),
        dtype=torch.bfloat16,
        device=x_fp8.device,
    )
    down_output.zero_()
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (down_input, down_input_scale),
        l2_weights,
        down_output,
        masked_m,
        expected_m,
        recipe=recipe,
        backend="mutlass",
    )
    del down_input, down_input_scale

    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x_fp8.device)
    combined_y = dispatch_simulator.combine(
        down_output,
        topk_idx,
        topk_weights,
        handle,
        out=out,
    )
    torch.musa.synchronize()
    return combined_y


def assert_exact_y(
    rank_idx: int, actual_y: torch.Tensor, expected_y: torch.Tensor, num_tokens: int
) -> None:
    actual = actual_y.narrow(0, 0, num_tokens).contiguous()
    expected = expected_y.narrow(0, 0, num_tokens).contiguous()
    actual_bits = actual.view(torch.int16)
    expected_bits = expected.view(torch.int16)
    bad = actual_bits != expected_bits
    bad_count = int(bad.sum().item())
    if bad_count != 0:
        diff = (actual.float() - expected.float()).abs()
        flat_idx = int(diff.reshape(-1).argmax().item())
        row = flat_idx // actual.size(1)
        col = flat_idx % actual.size(1)
        raise AssertionError(
            f"rank {rank_idx}: mate fused y != Python dispatch + deep_gemm reference "
            f"bad={bad_count}/{bad.numel()} token={row} col={col} "
            f"actual={actual.float()[row, col].item():.8g} expected={expected.float()[row, col].item():.8g} "
            f"actual_bits=0x{int(actual_bits[row, col].item()) & 0xFFFF:04x} "
            f"expected_bits=0x{int(expected_bits[row, col].item()) & 0xFFFF:04x}"
        )
    print(f"rank {rank_idx}: exact y check passed bad=0/{bad.numel()}", flush=True)


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    os.environ["NVSHMEM_IBGDA_NIC_HANDLER"] = "cpu"
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    sym_buffer = None

    def cleanup() -> None:
        torch.musa.synchronize()
        dist.barrier(group=group)
        if sym_buffer is not None:
            sym_buffer.destroy()
        dist.destroy_process_group(group)
        dist.destroy_process_group()

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    token_values = args.token_values
    max_tokens = max(token_values)
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert hidden % 512 == 0
    assert intermediate_hidden % 256 == 0
    assert num_experts % num_ranks == 0
    assert max_tokens <= num_max_tokens_per_rank

    dispatch_simulator = PythonLowLatencyDispatchSimulator(
        group, max_tokens, hidden, num_experts
    )
    torch.musa.synchronize()
    dist.barrier(group=group)

    sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )

    dist_print("Config:", once_in_node=True)
    dist_print(
        f" > Tokens: {','.join(str(token) for token in token_values)}/{num_max_tokens_per_rank}",
        once_in_node=True,
    )
    dist_print(f" > Hidden: {hidden}", once_in_node=True)
    dist_print(f" > Intermediate: {intermediate_hidden}", once_in_node=True)
    dist_print(f" > Experts: {num_topk}/{num_experts}", once_in_node=True)
    dist_print(
        f" > Buffer: {sym_buffer.buffer.nbytes / 2**30:.3f} GiB", once_in_node=True
    )
    dist_print(once_in_node=True)

    x_bf16 = torch.rand((max_tokens, hidden), dtype=torch.bfloat16, device="musa")
    scores = torch.randn((max_tokens, num_experts), dtype=torch.float32, device="musa")
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )

    x_fp8, x_sf = deep_ep_style_per_token_cast_to_fp8_tilelang(x_bf16.contiguous())

    l1_weight_fp8 = []
    l1_weight_sf = []
    for _ in range(num_experts_per_rank):
        w_bf16 = (
            torch.rand(
                (intermediate_hidden * 2, hidden),
                dtype=torch.bfloat16,
                device="musa",
            )
            * 2
            - 1
        ).to(torch.bfloat16)
        w_fp8, w_sf = per_block_cast_to_fp8(w_bf16)
        l1_weight_fp8.append(w_fp8)
        l1_weight_sf.append(w_sf)
    l1_weights = (
        torch.stack(l1_weight_fp8).contiguous(),
        torch.stack(l1_weight_sf).contiguous(),
    )

    l2_weight_fp8 = []
    l2_weight_sf = []
    for _ in range(num_experts_per_rank):
        w_bf16 = (
            torch.rand(
                (hidden, intermediate_hidden),
                dtype=torch.bfloat16,
                device="musa",
            )
            * 2
            - 1
        ).to(torch.bfloat16)
        w_fp8, w_sf = per_block_cast_to_fp8(w_bf16)
        l2_weight_fp8.append(w_fp8)
        l2_weight_sf.append(w_sf)
    l2_weights = (
        torch.stack(l2_weight_fp8).contiguous(),
        torch.stack(l2_weight_sf).contiguous(),
    )

    transformed_l1_weights, transformed_l2_weights = (
        deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)
    )

    def prepare_fused_inputs(num_tokens: int) -> None:
        sym_buffer.x[:num_tokens].copy_(x_fp8[:num_tokens])
        sym_buffer.x_sf[:num_tokens].copy_(x_sf[:num_tokens])
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx[:num_tokens])
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights[:num_tokens])

    def run_fused(num_tokens: int) -> torch.Tensor:
        prepare_fused_inputs(num_tokens)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="musa")
        deep_gemm.fp8_fp8_mega_moe(
            y=y,
            l1_weights=transformed_l1_weights,
            l2_weights=transformed_l2_weights,
            sym_buffer=sym_buffer,
            cumulative_local_expert_recv_stats=None,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )
        torch.musa.synchronize()
        dist.barrier(group=group)
        return y

    for num_tokens in token_values:
        dist_print(
            f"Running MATE MegaMoE correctness, tokens={num_tokens}...",
            once_in_node=True,
        )
        y = run_fused(num_tokens)
        y_ref = run_python_dispatch_deepgemm_reference(
            group=group,
            dispatch_simulator=dispatch_simulator,
            x_fp8=x_fp8[:num_tokens].contiguous(),
            x_scale=x_sf[:num_tokens].contiguous(),
            topk_idx=topk_idx[:num_tokens].long(),
            topk_weights=topk_weights[:num_tokens],
            l1_weights=l1_weights,
            l2_weights=l2_weights,
            num_max_dispatch_tokens_per_rank=num_tokens,
            num_experts=num_experts,
        )
        torch.musa.synchronize()
        dist.barrier(group=group)
        assert_exact_y(rank_idx, y, y_ref, num_tokens)
        del y, y_ref
        dist_print(
            f"MATE MegaMoE correctness passed, tokens={num_tokens}.",
            once_in_node=True,
        )

    print(f"rank {rank_idx} OK", flush=True)
    cleanup()


def _default_token_values() -> list[int]:
    if "MATE_MEGA_MOE_TEST_TOKENS" in os.environ:
        return _parse_token_values(os.environ["MATE_MEGA_MOE_TEST_TOKENS"])
    if "MATE_MEGA_MOE_TEST_NUM_TOKENS" in os.environ:
        return [int(os.environ["MATE_MEGA_MOE_TEST_NUM_TOKENS"])]
    return list(DEFAULT_TOKEN_SWEEP)


def _default_args(num_processes: int | None = None) -> argparse.Namespace:
    if num_processes is None:
        num_processes = int(os.getenv("MATE_MEGA_MOE_TEST_NUM_PROCESSES", "8"))
    default_num_experts = DEFAULT_NUM_EXPERTS_PER_RANK * num_processes
    return argparse.Namespace(
        num_processes=num_processes,
        num_max_tokens_per_rank=int(os.getenv("MATE_MEGA_MOE_TEST_MAX_TOKENS", "1024")),
        token_values=_default_token_values(),
        hidden=int(os.getenv("MATE_MEGA_MOE_TEST_HIDDEN", "4096")),
        intermediate_hidden=int(os.getenv("MATE_MEGA_MOE_TEST_INTERMEDIATE", "2048")),
        activation_clamp=float(os.getenv("MATE_MEGA_MOE_TEST_ACTIVATION_CLAMP", "10")),
        num_experts=int(
            os.getenv("MATE_MEGA_MOE_TEST_NUM_EXPERTS", str(default_num_experts))
        ),
        num_topk=int(os.getenv("MATE_MEGA_MOE_TEST_NUM_TOPK", "6")),
        fast_math=int(os.getenv("MATE_MEGA_MOE_TEST_FAST_MATH", "1")),
    )


def _get_available_musa_device_count() -> int:
    if not hasattr(torch, "musa"):
        pytest.skip("torch.musa is not available")
    try:
        if not torch.musa.is_available():
            pytest.skip("MUSA is not available")
        return int(torch.musa.device_count())
    except Exception as exc:
        pytest.skip(f"Unable to query MUSA device count: {exc}")


def _resolve_pytest_num_processes(device_count: int) -> int:
    if device_count <= 0:
        pytest.skip("No visible MUSA devices are available")
    if "MATE_MEGA_MOE_TEST_NUM_PROCESSES" in os.environ:
        num_processes = int(os.environ["MATE_MEGA_MOE_TEST_NUM_PROCESSES"])
        if num_processes > device_count:
            pytest.skip(
                f"MATE_MEGA_MOE_TEST_NUM_PROCESSES={num_processes} requires "
                f"{num_processes} visible MUSA devices, found {device_count}"
            )
        return num_processes
    if device_count >= 8:
        return 8
    return 1


def _validate_pytest_args(args: argparse.Namespace, device_count: int) -> None:
    if args.num_processes not in PYTEST_PROCESS_COUNTS:
        pytest.skip(
            "MegaMoE pytest coverage runs only single-card or 8-card configs; "
            f"got num_processes={args.num_processes}"
        )
    if args.num_processes > device_count:
        pytest.skip(
            f"MegaMoE pytest config requires {args.num_processes} visible MUSA "
            f"devices, found {device_count}"
        )
    if args.num_experts % args.num_processes != 0:
        pytest.skip(
            f"MegaMoE pytest config requires num_experts ({args.num_experts}) to "
            f"be divisible by num_processes ({args.num_processes})"
        )
    if args.num_topk > args.num_experts:
        pytest.skip(
            f"MegaMoE pytest config requires num_topk ({args.num_topk}) <= "
            f"num_experts ({args.num_experts})"
        )


@supported_musa_compute_capability([31])
def test_mate_mega_moe_deepep_deepgemm_exact_and_profile():
    pytest.importorskip("torch_musa")
    pytest.importorskip("tilelang")
    device_count = _get_available_musa_device_count()
    num_processes = _resolve_pytest_num_processes(device_count)
    args = _default_args(num_processes)
    _validate_pytest_args(args, device_count)
    torch.multiprocessing.spawn(
        _worker, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test MATE MegaMoE against Python dispatch + mate.deep_gemm"
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=1024)
    parser.add_argument("--tokens", default=DEFAULT_TOKEN_SWEEP_STR)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--activation-clamp", type=float, default=10)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--fast-math", type=int, default=1)
    cli_args = parser.parse_args()
    if cli_args.num_tokens is not None:
        cli_args.token_values = [cli_args.num_tokens]
    else:
        cli_args.token_values = _parse_token_values(cli_args.tokens)
    if max(cli_args.token_values) > cli_args.num_max_tokens_per_rank:
        raise ValueError(
            f"max token count {max(cli_args.token_values)} exceeds "
            f"--num-max-tokens-per-rank={cli_args.num_max_tokens_per_rank}"
        )
    torch.multiprocessing.spawn(
        _worker, args=(cli_args.num_processes, cli_args), nprocs=cli_args.num_processes
    )
