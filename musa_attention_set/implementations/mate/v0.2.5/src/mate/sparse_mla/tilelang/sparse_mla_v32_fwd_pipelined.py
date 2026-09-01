# ruff: noqa
from typing import Any

import torch
import tilelang
from tilelang import language as T
from tvm import tir

from ...utils import cosize
from ...mate_runtime import resolve_num_mps
from .sparse_mla_prefill_common import (
    SPARSE_PREFILL_COMPILE_FLAGS,
    SPARSE_PREFILL_PASS_CONFIGS,
    prepare_sparse_mla_strided_tensor,
    validate_prefill_attn_sink,
    validate_token_lengths,
)
from .sparse_mla_index_type import jit_for_tensor_addressing
from ...execution_context import raise_complete_if_dry_run


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    raise RuntimeError("MUSA  is not available")


@tilelang.jit(
    out_idx=[3, 4, 5],
    pass_configs=SPARSE_PREFILL_PASS_CONFIGS,
    verbose=True,
    compile_flags=SPARSE_PREFILL_COMPILE_FLAGS,
)
def sparse_attention_fwd_kernel(
    num_heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=False,
    block_i=64,
    threads=640,
    has_attn_sink=False,
    has_topk_length=False,
    is_persistence=True,
    persistent_blocks=None,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        f"haven't check padding correctness yet, dim={tail_dim}"
    )
    assert is_causal == False, "casual is not supported for sparse_attention"
    assert topk % block_i == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )
    if sm_scale is None:
        logits_scale = (1.0 / (dim + tail_dim)) ** 0.5
    else:
        logits_scale = sm_scale
    sm_scale = logits_scale * 1.44269504  # log2(e)

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    kv_stride_s = T.dynamic("kv_stride_s")
    kv_stride_g = T.dynamic("kv_stride_g")
    indices_stride_s = T.dynamic("indices_stride_s")
    indices_stride_g = T.dynamic("indices_stride_g")

    head_kv = num_heads // kv_group
    q_shape = [seq_len, num_heads, dim + tail_dim]
    kv_shape = [seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [seq_len, num_heads, dim]
    lse_shape = [seq_len, num_heads]
    max_logits_shape = [seq_len, num_heads]
    indices_shape = [seq_len, kv_group, topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    dtype_bytes = 2

    q_strides = (q_stride_s, q_stride_h, 1)
    kv_strides = (kv_stride_s, kv_stride_g, 1)
    indices_strides = (indices_stride_s, indices_stride_g, 1)
    q_cosize = cosize(q_shape, q_strides)
    kv_cosize = cosize(kv_shape, kv_strides)
    q_type: Any = T.StridedTensor(q_shape, q_strides, dtype)
    kv_type: Any = T.StridedTensor(kv_shape, kv_strides, dtype)
    indices_type: Any = T.StridedTensor(indices_shape, indices_strides, indices_dtype)

    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), 64)
    if padded_head_kv != head_kv:
        assert kv_group == 1
    dim_qk = dim
    tail_dim_qk = tail_dim
    lanes_per_vec = block_i // 8

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        head_repeats = head_kv // 64
    else:
        head_repeats = 1

    heads_per_block = padded_head_kv if head_repeats == 1 else 64
    if is_persistence:
        persistent_blocks = resolve_num_mps(num_mps=persistent_blocks)
        assert persistent_blocks > 0, "persistent_blocks must be positive"
        assert kv_group == 1, "persistent V3.2 prefill currently supports MQA only"
    logical_blocks = seq_len * head_repeats
    launch_blocks = (
        T.min(persistent_blocks, logical_blocks) if is_persistence else logical_blocks
    )

    @T.macro
    def dsa_prefill_body(
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        output,
        max_logits_out,
        lse,
    ):
        with T.Kernel(launch_blocks, kv_group, threads=threads) as (
            bx,
            by,
        ):
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(kv_stride_s % 8 == 0)
            T.assume(kv_stride_g % 8 == 0)
            T.assume(indices_stride_s % 8 == 0)
            T.assume(indices_stride_g % 8 == 0)
            q_shared_l = T.alloc_shared([heads_per_block, dim_qk // 2], dtype)
            q_shared_r = T.alloc_shared([heads_per_block, dim_qk // 2], dtype)
            kv_shared_l = T.alloc_shared([block_i, dim_qk // 2], dtype)
            kv_shared_r = T.alloc_shared([block_i, dim_qk // 2], dtype)
            q_tail_shared = T.alloc_shared([heads_per_block, tail_dim_qk], dtype)
            k_tail_shared = T.alloc_shared([block_i, tail_dim_qk], dtype)

            v_shared_0 = T.alloc_shared([block_i, dim_qk // 4], dtype)
            v_shared_1 = T.alloc_shared([block_i, dim_qk // 4], dtype)

            scores_shared = T.alloc_shared([heads_per_block, block_i], dtype)
            sum_exp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            lse_shared = T.alloc_shared([heads_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([block_i], "bool", scope="shared")
            is_kv_perm_valid = T.alloc_shared([block_i], "bool", scope="shared")

            bar_q = T.alloc_barrier(arrive_count=512)
            bar_kv0_ready = T.alloc_barrier(arrive_count=128)
            bar_kv1_ready = T.alloc_barrier(arrive_count=128)
            bar_kv1_read_ready = T.alloc_barrier(arrive_count=256)
            bar_kv0_free = T.alloc_barrier(arrive_count=256)
            bar_kv1_free = T.alloc_barrier(arrive_count=256)

            bar_vl0_ready = T.alloc_barrier(arrive_count=256)
            bar_vl1_ready = T.alloc_barrier(arrive_count=256)
            bar_vr0_ready = T.alloc_barrier(arrive_count=256)
            bar_vr1_ready = T.alloc_barrier(arrive_count=256)
            bar_vl0_free = T.alloc_barrier(arrive_count=256)
            bar_vl1_free = T.alloc_barrier(arrive_count=256)

            bar_p_ready = T.alloc_barrier(arrive_count=256)
            bar_final = T.alloc_barrier(arrive_count=256)

            q_robust_desc = T.make_robust_desc(
                T.address_of(q[0, 0, 0]), q_cosize * dtype_bytes
            )
            kv_robust_desc = T.make_robust_desc(
                T.address_of(kv[0, 0, 0]),
                kv_cosize * dtype_bytes,
            )

            T.sync_threads()

            mask = T.alloc_fragment([block_i], "bool")

            tid = T.get_thread_binding()
            logical_bx = T.alloc_var(T.int32)
            phase_count = T.alloc_local([1], T.int32)
            logical_phase = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)
            T.fill(logical_phase, 0)
            logical_bx = bx
            while logical_bx < logical_blocks:
                if is_persistence:
                    tir.call_extern("void", "__musa_loop_transparent_outermost")
                g_i = by
                s_i = logical_bx if head_repeats == 1 else (logical_bx // head_repeats)
                q_i = s_i

                h0 = g_i * padded_head_kv + (
                    0 if head_repeats == 1 else (logical_bx % head_repeats) * 64
                )
                h1 = h0 + heads_per_block
                if tid < 512:
                    T.copy(q[s_i, h0:h1, 0 : dim_qk // 2], q_shared_l, barrier=bar_q)
                    T.copy(
                        q[s_i, h0:h1, dim_qk // 2 : dim_qk], q_shared_r, barrier=bar_q
                    )
                    T.copy(q[s_i, h0:h1, dim_qk:], q_tail_shared, barrier=bar_q)

                    T.barrier_arrive(bar_q)
                    T.barrier_wait(bar_q, logical_phase[0] & 1)

                if tid < 256:
                    # consumer 0
                    sumexp = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_inv = T.alloc_fragment([heads_per_block], accum_dtype)
                    alpha_local = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([heads_per_block], accum_dtype)
                    max_logits = T.alloc_fragment([heads_per_block], accum_dtype)
                    acc_s = T.alloc_fragment([heads_per_block, block_i], accum_dtype)
                    acc_s_cast = T.alloc_fragment([heads_per_block, block_i], dtype)
                    acc_o_l_0 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    acc_o_l_1 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    kv_reg_l = T.alloc_local([64], dtype)
                    ldg_tx = (tid) % 8
                    ldg_ty = (tid) // 8
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l_0, 0)
                    T.fill(acc_o_l_1, 0)

                    for i_i in range(T.ceildiv(topk, block_i)):
                        T.barrier_wait(bar_kv0_ready, phase_count[0] & 1)

                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                is_kv_valid[bi_i % 8 * 8 + bi_i // 8], 0, -(2**30)
                            )

                        T.annotate_layout(
                            {
                                kv_shared_l[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_l[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.gemm(
                            q_shared_l,
                            kv_shared_l[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )

                        # LMA.RD kv_reg_l
                        for r in T.unroll(2):
                            for u in T.unroll(4):
                                for v in T.vectorized(8):
                                    kv_reg_l[r * 32 + u * 8 + v] = kv_shared_l[
                                        ((ldg_ty + r * 32) % 8) * (block_i // 8)
                                        + (ldg_ty + r * 32) // 8,
                                        64 * u + ldg_tx * 8 + v,
                                    ]
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_free)

                        T.barrier_wait(bar_kv1_ready, phase_count[0] & 1)
                        T.annotate_layout(
                            {
                                kv_shared_r[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.gemm(
                            q_shared_r,
                            kv_shared_r[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.barrier_arrive(bar_kv1_read_ready)
                        T.warpgroup_wait(0)

                        T.annotate_layout(
                            {
                                k_tail_shared[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    k_tail_shared[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.gemm(
                            q_tail_shared,
                            k_tail_shared[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv1_free)

                        # online softmax
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        # for h_i in T.Parallel(heads_per_block):
                        #     m_i[h_i] = T.max(m_i_prev[h_i], m_i[h_i])
                        for h_i in T.Parallel(heads_per_block):
                            alpha_local[h_i] = T.exp2(
                                (m_i_prev[h_i] - m_i[h_i]) * sm_scale
                            )
                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )

                        T.copy(alpha_local, alpha_shared)
                        T.copy(acc_s, acc_s_cast)
                        for i, t in T.Parallel(heads_per_block, 8):
                            base = t * lanes_per_vec
                            for l in T.vectorized(lanes_per_vec):
                                scores_shared[i, base + l] = acc_s_cast[i, l * 8 + t]

                        T.lma_wait()
                        for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                            acc_o_l_0[h_i, d_i] *= alpha_local[h_i]
                            acc_o_l_1[h_i, d_i] *= alpha_local[h_i]
                        T.barrier_arrive(bar_p_ready)

                        T.annotate_layout(
                            {
                                v_shared_0[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    v_shared_0[:, :], k_major=False
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        # STS 2 V Buf 0
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_0[
                                        r * 32 + ldg_ty,
                                        64 * u + ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + u * 8 + v]
                        T.lma_wait()
                        T.barrier_arrive(bar_vl0_ready)
                        T.barrier_wait(bar_vl0_ready, phase_count[0] & 1)

                        T.gemm(
                            scores_shared,
                            v_shared_0,
                            acc_o_l_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.annotate_layout(
                            {
                                v_shared_1[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    v_shared_1[:, :], k_major=False
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        # STS 2 V Buf 1
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_1[
                                        r * 32 + ldg_ty,
                                        64 * u + ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + (u + 2) * 8 + v]

                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vl0_free)

                        T.lma_wait()
                        T.barrier_arrive(bar_vl1_ready)
                        T.barrier_wait(bar_vl1_ready, phase_count[0] & 1)

                        T.gemm(
                            scores_shared,
                            v_shared_1,
                            acc_o_l_1,
                            transpose_B=False,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(heads_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vl1_free)
                        phase_count[0] = phase_count[0] ^ 1

                    for h_i in T.Parallel(heads_per_block):
                        if m_i[h_i] > -(2**29):
                            sumexp_inv[h_i] = 1 / sumexp[h_i]
                            max_logits[h_i] = m_i[h_i] * logits_scale
                            sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                            if has_attn_sink:
                                if sumexp_inv[h_i] > 0:
                                    sumexp_inv[h_i] *= 1 / (
                                        1
                                        + T.exp2(
                                            attn_sink[h0 + h_i] * 1.4426950408889634
                                            - sumexp[h_i]
                                        )
                                    )
                        else:
                            sumexp_inv[h_i] = 0
                            max_logits[h_i] = -T.infinity(accum_dtype)
                            sumexp[h_i] = T.infinity(accum_dtype)
                        sum_exp_inv_shared[h_i] = sumexp_inv[h_i]
                        lse_shared[h_i] = sumexp[h_i]
                    T.barrier_arrive(bar_final)
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_l_0[h_i, d_i] *= sumexp_inv[h_i]
                        acc_o_l_1[h_i, d_i] *= sumexp_inv[h_i]

                    for h_i in T.Parallel(heads_per_block):
                        lse[s_i, h0 + h_i] = sumexp[h_i] * 0.6931471805599453
                    T.copy(max_logits, max_logits_out[s_i, h0:h1])
                    T.copy(acc_o_l_0, output[s_i, h0:h1, 0 : dim_qk // 4])
                    T.copy(acc_o_l_1, output[s_i, h0:h1, dim_qk // 4 : dim_qk // 2])
                elif tid >= 256 and tid < 512:
                    # consumer 1
                    acc_o_r_0 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    acc_o_r_1 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    kv_reg_r = T.alloc_local([64], dtype)
                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)

                    ldg_tx = (tid - 256) % 8
                    ldg_ty = (tid - 256) // 8

                    for i_i in range(T.ceildiv(topk, block_i)):
                        T.barrier_wait(bar_kv1_read_ready, phase_count[0] & 1)
                        # LMA.RD kv_reg_r
                        for r in T.unroll(2):
                            for u in T.unroll(4):
                                for v in T.vectorized(8):
                                    kv_reg_r[r * 32 + u * 8 + v] = kv_shared_r[
                                        ((ldg_ty + r * 32) % 8) * (block_i // 8)
                                        + (ldg_ty + r * 32) // 8,
                                        64 * u + ldg_tx * 8 + v,
                                    ]

                        T.lma_wait()
                        T.barrier_wait(bar_vl0_free, phase_count[0] & 1)
                        # STS 2 VR Buf 0
                        T.annotate_layout(
                            {
                                v_shared_0[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    v_shared_0[:, :], k_major=False
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_0[
                                        r * 32 + ldg_ty,
                                        64 * u + ldg_tx * 8 + v,
                                    ] = kv_reg_r[r * 32 + u * 8 + v]

                        T.lma_wait()
                        T.barrier_arrive(bar_vr0_ready)
                        T.barrier_wait(bar_vr0_ready, phase_count[0] & 1)

                        # compute v4-v7
                        T.barrier_wait(bar_p_ready, phase_count[0] & 1)
                        for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                            acc_o_r_0[h_i, d_i] *= alpha_shared[h_i]
                            acc_o_r_1[h_i, d_i] *= alpha_shared[h_i]

                        # bar arrive & wait
                        T.gemm(
                            scores_shared,
                            v_shared_0,
                            acc_o_r_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.wait_wgmma(0)
                        # T.warpgroup_commit_batch()

                        T.barrier_wait(bar_vl1_free, phase_count[0] & 1)
                        # STS 2 V Buf 1
                        T.annotate_layout(
                            {
                                v_shared_1[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    v_shared_1[:, :], k_major=False
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_1[
                                        r * 32 + ldg_ty,
                                        64 * u + ldg_tx * 8 + v,
                                    ] = kv_reg_r[r * 32 + (u + 2) * 8 + v]
                        T.lma_wait()
                        # T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vr1_ready)
                        T.barrier_wait(bar_vr1_ready, phase_count[0] & 1)

                        # compute v4-v7
                        T.gemm(
                            scores_shared,
                            v_shared_1,
                            acc_o_r_1,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.wait_wgmma(0)
                        # T.warpgroup_commit_batch()
                        # T.warpgroup_wait(0)
                        phase_count[0] = phase_count[0] ^ 1

                    T.barrier_wait(bar_final, logical_phase[0] & 1)
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_r_0[h_i, d_i] *= sum_exp_inv_shared[h_i]
                        acc_o_r_1[h_i, d_i] *= sum_exp_inv_shared[h_i]

                    T.copy(
                        acc_o_r_0,
                        output[s_i, h0:h1, dim_qk // 2 : dim_qk // 2 + dim_qk // 4],
                    )
                    T.copy(
                        acc_o_r_1,
                        output[s_i, h0:h1, dim_qk // 2 + dim_qk // 4 : dim_qk],
                    )
                elif tid >= 512:
                    mask_local = T.alloc_local([4], "bool")
                    indices_local = T.alloc_local([4], indices_dtype)

                    kperm_mask_local = T.alloc_local([4], "bool")
                    kperm_indices_local = T.alloc_local([4], "int32")
                    topk_len_local = T.alloc_local([1], indices_dtype)

                    # producer: 128 ldg_ty 16
                    ldg_tx = (tid - 512) % 8
                    ldg_ty = (tid - 512) // 8
                    if has_topk_length:
                        topk_len_local[0] = topk_length[s_i]
                    else:
                        topk_len_local[0] = topk
                    for i_i in range(T.ceildiv(topk, block_i)):
                        # Load sparse indices for the next kv block.
                        for r in T.unroll(4):
                            # indices_local[r] = indices[s_i, g_i, (i_i) * block_i + r * 16 + ldg_ty]
                            kperm_indices_local[r] = indices[
                                s_i,
                                g_i,
                                (i_i) * block_i
                                + ((r * 16 + ldg_ty) % 8) * (block_i // 8)
                                + (r * 16 + ldg_ty) // 8,
                            ]
                        for r in T.unroll(4):
                            token_pos = (
                                (i_i) * block_i
                                + ((r * 16 + ldg_ty) % 8) * (block_i // 8)
                                + (r * 16 + ldg_ty) // 8
                            )
                            # mask_local[r] = indices_local[r]>=0
                            # indices_local[r] = T.if_then_else(mask_local[r], indices_local[r], (seq_len_kv * kv_group * (dim + tail_dim))*2+1)
                            kperm_mask_local[r] = (
                                kperm_indices_local[r] >= 0
                                and kperm_indices_local[r] < seq_len_kv
                                and token_pos < topk_len_local[0]
                            )
                            kperm_indices_local[r] = T.if_then_else(
                                kperm_mask_local[r],
                                kperm_indices_local[r],
                                seq_len_kv,
                            )

                        T.barrier_wait(bar_kv0_free, (phase_count[0] & 1) ^ 1)
                        # load k0-k3
                        T.annotate_layout(
                            {
                                kv_shared_l[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_l[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(4):
                                for v in T.vectorized(8):
                                    T.copy(
                                        kv[
                                            kperm_indices_local[r],
                                            g_i,
                                            64 * u + ldg_tx * 8 + v,
                                        ],
                                        kv_shared_l[
                                            r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )

                        T.ptx_commit_group()
                        for r in T.unroll(4):
                            is_kv_valid[
                                ((r * 16 + ldg_ty) % 8) * (block_i // 8)
                                + (r * 16 + ldg_ty) // 8
                            ] = kperm_mask_local[r]

                        T.ptx_wait_group(0)
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_ready)

                        T.barrier_wait(bar_kv1_free, (phase_count[0] & 1) ^ 1)
                        # load k4-k7
                        T.annotate_layout(
                            {
                                kv_shared_r[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(4):
                                for v in T.vectorized(8):
                                    pass
                                    T.copy(
                                        kv[
                                            kperm_indices_local[r],
                                            g_i,
                                            dim_qk // 2 + 64 * u + ldg_tx * 8 + v,
                                        ],
                                        kv_shared_r[
                                            r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )

                        # load next k rope
                        T.annotate_layout(
                            {
                                k_tail_shared[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    k_tail_shared[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for v in T.vectorized(8):
                                pass
                                T.copy(
                                    kv[
                                        kperm_indices_local[r],
                                        g_i,
                                        dim_qk + ldg_tx * 8 + v,
                                    ],
                                    k_tail_shared[r * 16 + ldg_ty, ldg_tx * 8 + v],
                                    force_async_copy=True,
                                    src_robust_desc=kv_robust_desc,
                                )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.barrier_arrive(bar_kv1_ready)
                        T.sync_threads(1, 128)
                        phase_count[0] = phase_count[0] ^ 1

                if is_persistence:
                    logical_phase[0] = logical_phase[0] ^ 1
                    logical_bx += persistent_blocks
                else:
                    logical_bx = logical_blocks

    if has_topk_length and has_attn_sink:

        @T.prim_func
        def dsa_prefill(
            q: q_type,
            kv: kv_type,
            indices: indices_type,
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            topk_length: T.Tensor([seq_len], indices_dtype),
            attn_sink: T.Tensor([num_heads], accum_dtype),
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                topk_length,
                attn_sink,
                output,
                max_logits_out,
                lse,
            )

    elif has_topk_length:

        @T.prim_func
        def dsa_prefill(
            q: q_type,
            kv: kv_type,
            indices: indices_type,
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            topk_length: T.Tensor([seq_len], indices_dtype),
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                topk_length,
                None,
                output,
                max_logits_out,
                lse,
            )

    elif has_attn_sink:

        @T.prim_func
        def dsa_prefill(
            q: q_type,
            kv: kv_type,
            indices: indices_type,
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            attn_sink: T.Tensor([num_heads], accum_dtype),
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                None,
                attn_sink,
                output,
                max_logits_out,
                lse,
            )

    else:

        @T.prim_func
        def dsa_prefill(
            q: q_type,
            kv: kv_type,
            indices: indices_type,
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                None,
                None,
                output,
                max_logits_out,
                lse,
            )

    return dsa_prefill


def tilelang_sparse_mla_prefill_fwd_interface(
    q,
    kv,
    indices,
    sm_scale=None,
    topk_length=None,
    attn_sink=None,
    return_p_sum: bool = False,
    d_v=512,
    threads=640,
    verbose=False,
    return_max_logits: bool = False,
    is_persistence: bool = True,
    persistent_blocks: int | None = None,
):
    is_casual = False
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert kv.dtype == torch.bfloat16, "kv must be bfloat16"
    assert indices.dtype == torch.int32, "indices must be int32"
    q, q_shape = prepare_sparse_mla_strided_tensor("q", q, multiple=8)
    kv, kv_shape = prepare_sparse_mla_strided_tensor("kv", kv, multiple=8)
    indices, indices_shape = prepare_sparse_mla_strided_tensor(
        "indices", indices, multiple=8
    )
    seq_len, heads, dim_plus_tail_dim = q_shape
    seq_len_kv, kv_group, _ = kv_shape

    dim = d_v

    assert kv_shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    _, _, topk = indices_shape
    assert indices_shape == (seq_len, kv_group, topk)
    assert dim == 512, f"V3.2 kernel currently expects d_v=512, got {dim}"
    assert tail_dim == 64, f"V3.2 kernel currently expects tail_dim=64, got {tail_dim}"
    topk_length = validate_token_lengths(topk_length, seq_len, "topk_length")
    attn_sink = validate_prefill_attn_sink(attn_sink, heads)
    if is_persistence:
        persistent_blocks = resolve_num_mps(q.device, persistent_blocks)
    else:
        persistent_blocks = 0

    # These inputs dominate the contiguous output and auxiliary tensor spans.
    kernel_factory = jit_for_tensor_addressing(
        sparse_attention_fwd_kernel,
        q,
        kv,
        indices,
    )
    kernel = kernel_factory(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_causal=is_casual,
        threads=threads,
        has_attn_sink=attn_sink is not None,
        has_topk_length=topk_length is not None,
        is_persistence=is_persistence,
        persistent_blocks=persistent_blocks,
    )
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()

    args = [q, kv, indices]
    if topk_length is not None:
        args.append(topk_length)
    if attn_sink is not None:
        args.append(attn_sink)
    out = kernel(*args)
    out_tensor, max_logits, lse_tensor = out
    if return_max_logits:
        return out_tensor, max_logits, lse_tensor
    return out_tensor, lse_tensor
