# ruff: noqa
from typing import Any

import torch
import tilelang
from tilelang import language as T
import math

from ...utils import cosize
from .sparse_mla_decode_scheduled_common import (
    SCHEDULED_DECODE_COMPILE_FLAGS,
    SCHEDULED_DECODE_PASS_CONFIGS,
    make_scheduled_decode_combine,
    make_scheduled_decode_finalize_left,
    make_scheduled_decode_finalize_right,
    make_scheduled_decode_indices_loader,
    make_scheduled_decode_online_softmax,
    make_scheduled_decode_stage_value_shared,
    prepare_scheduled_decode_runtime,
    prepare_sparse_mla_decode_strided_tensor,
)
from .sparse_mla_index_type import jit_for_tensor_addressing
from ...execution_context import raise_complete_if_dry_run


@tilelang.jit(
    out_idx=[],
    pass_configs=SCHEDULED_DECODE_PASS_CONFIGS,
    verbose=True,
    compile_flags=SCHEDULED_DECODE_COMPILE_FLAGS,
)
def sparse_attention_fwd_kernel(
    num_heads,
    dim,
    tail_dim,
    *,
    kv_group=1,
    block_h=64,
    block_i=64,
    threads=640,
    max_nums_splits=32,
    has_attn_sink=False,
    has_topk_length=False,
    support_split=True,
    use_int64_cosize=False,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        f"haven't check padding correctness yet, dim={tail_dim}"
    )
    topk = T.dynamic("topk")
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_mp_parts = T.dynamic("num_mp_parts")
    q_stride_b = T.dynamic("q_stride_b")
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    kv_stride_s = T.dynamic("kv_stride_s")
    kv_stride_g = T.dynamic("kv_stride_g")
    k_pe_stride_s = T.dynamic("k_pe_stride_s")
    k_pe_stride_g = T.dynamic("k_pe_stride_g")
    quant_scales_stride_s = T.dynamic("quant_scales_stride_s")
    quant_scales_stride_g = T.dynamic("quant_scales_stride_g")
    indices_stride_b = T.dynamic("indices_stride_b")
    indices_stride_s = T.dynamic("indices_stride_s")
    indices_stride_g = T.dynamic("indices_stride_g")

    head_kv = num_heads // kv_group
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    dtype_bytes = 2
    dim_bytes = 656
    q_shape = [batch, seq_len, num_heads, dim + tail_dim]
    kv_shape = [seq_len_kv, kv_group, dim_bytes]
    k_pe_shape = [seq_len_kv, kv_group, dim_bytes // 2]
    quant_scales_shape = [seq_len_kv, kv_group, dim_bytes // 4]
    indices_shape = [batch, seq_len, kv_group, topk]
    q_strides = (q_stride_b, q_stride_s, q_stride_h, 1)
    kv_strides = (kv_stride_s, kv_stride_g, 1)
    k_pe_strides = (k_pe_stride_s, k_pe_stride_g, 1)
    quant_scales_strides = (
        quant_scales_stride_s,
        quant_scales_stride_g,
        1,
    )
    indices_strides = (indices_stride_b, indices_stride_s, indices_stride_g, 1)
    if use_int64_cosize:
        q_cosize = cosize(
            q_shape, tuple(T.cast(stride, "int64") for stride in q_strides)
        )
        kv_cosize = cosize(
            kv_shape, tuple(T.cast(stride, "int64") for stride in kv_strides)
        )
    else:
        q_cosize = cosize(q_shape, q_strides)
        kv_cosize = cosize(kv_shape, kv_strides)
    q_type: Any = T.StridedTensor(q_shape, q_strides, dtype)
    kv_type: Any = T.StridedTensor(kv_shape, kv_strides, "float8_e4m3")
    k_pe_type: Any = T.StridedTensor(k_pe_shape, k_pe_strides, dtype)
    quant_scales_type: Any = T.StridedTensor(
        quant_scales_shape, quant_scales_strides, T.float32
    )
    indices_type: Any = T.StridedTensor(indices_shape, indices_strides, indices_dtype)
    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), block_h)
    if padded_head_kv != head_kv:
        assert kv_group == 1
    kv_latent_dtype = "float8_e4m3"
    dim_qk = dim
    tail_dim_qk = tail_dim
    if head_kv > block_h:
        assert head_kv % block_h == 0, "head_kv should be a multiple of block_h"
        head_repeats = head_kv // block_h
    else:
        head_repeats = 1

    heads_per_block = padded_head_kv if head_repeats == 1 else block_h
    pv_mma_n = 64 if block_h == 32 else 128

    update_online_softmax = make_scheduled_decode_online_softmax(
        h_per_block=heads_per_block,
        block_i=block_i,
        out_width=dim_qk // 4,
        accum_dtype=accum_dtype,
    )
    finalize_left = make_scheduled_decode_finalize_left(
        h_per_block=heads_per_block,
        out_width=dim_qk // 4,
        num_heads=num_heads,
        accum_dtype=accum_dtype,
        has_attn_sink=has_attn_sink,
        use_strict_valid=False,
        add_denominator_epsilon=True,
        sink_only_unsplit=False,
        sink_invalid_zero=False,
        wait_before_final=True,
        l0_start=0,
        l1_start=dim_qk // 4,
        out_dtype=dtype,
        support_split=support_split,
    )
    finalize_right = make_scheduled_decode_finalize_right(
        h_per_block=heads_per_block,
        out_width=dim_qk // 4,
        num_heads=num_heads,
        has_attn_sink=has_attn_sink,
        wait_after_scale=True,
        r0_start=dim_qk // 2,
        r1_start=dim_qk // 2 + dim_qk // 4,
        out_dtype=dtype,
        support_split=support_split,
    )
    stage_value_shared = make_scheduled_decode_stage_value_shared(
        block_i=block_i,
        continuity=pv_mma_n,
    )
    load_indices = make_scheduled_decode_indices_loader(block_i=block_i)

    @T.macro
    def dsa_decode_split(
        q,
        kv,
        k_pe,
        quant_scales,
        indices,
        topk_length,
        attn_sink,
        tile_scheduler_metadata,
        num_splits,
        glse,
        output_partial,
        output,
        lse,
        sm_scale,
    ):
        with T.Kernel(
            seq_len * head_repeats, kv_group, num_mp_parts, threads=threads
        ) as (bx, by, bz):
            T.assume(q_stride_b % 8 == 0)
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(kv_stride_s % 8 == 0)
            T.assume(kv_stride_g % 8 == 0)
            T.assume(k_pe_stride_s % 8 == 0)
            T.assume(k_pe_stride_g % 8 == 0)
            T.assume(quant_scales_stride_s % 4 == 0)
            T.assume(quant_scales_stride_g % 4 == 0)
            T.assume(indices_stride_b % 8 == 0)
            T.assume(indices_stride_s % 8 == 0)
            T.assume(indices_stride_g % 8 == 0)
            kv_shared_l = T.alloc_shared([block_i, dim_qk // 2], dtype)
            kv_shared_r = T.alloc_shared([block_i, dim_qk // 2], dtype)
            q_shared_l = T.alloc_shared([heads_per_block, dim_qk // 2], dtype)
            q_shared_r = T.alloc_shared([heads_per_block, dim_qk // 2], dtype)
            q_tail_shared = T.alloc_shared([heads_per_block, tail_dim_qk], dtype)
            k_tail_shared = T.alloc_shared([block_i, tail_dim_qk], dtype)
            v_shared_0 = T.alloc_shared([block_i, dim_qk // 4], dtype)
            v_shared_1 = T.alloc_shared([block_i, dim_qk // 4], dtype)
            scores_shared = T.alloc_shared([heads_per_block, block_i], dtype)
            sum_exp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            sink_scale_shared = T.alloc_shared([heads_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([block_i], "bool", scope="shared")
            kv_indices = T.alloc_shared([block_i], "int32", scope="shared")
            quant_shared = T.alloc_shared([block_i, 4], "float32")
            bar_kv_mask_ready = T.alloc_barrier(arrive_count=128)
            bar_kv_mask_free = T.alloc_barrier(arrive_count=256)
            bar_q = T.alloc_barrier(arrive_count=512)
            # bar_q_free = T.alloc_barrier(arrive_count=256)
            bar_indices_ready = T.alloc_barrier(arrive_count=128)
            bar_kv0_ready = T.alloc_barrier(arrive_count=128)
            bar_kv1_ready = T.alloc_barrier(arrive_count=128)
            bar_kv0_lma_read_ready = T.alloc_barrier(arrive_count=256)
            bar_kv1_lma_read_ready = T.alloc_barrier(arrive_count=256)
            bar_kv0_quant_ready = T.alloc_barrier(arrive_count=256)
            bar_kv1_quant_ready = T.alloc_barrier(arrive_count=256)
            bar_kv0_free = T.alloc_barrier(arrive_count=256)
            bar_kv1_free = T.alloc_barrier(arrive_count=256)
            bar_vl0_ready = T.alloc_barrier(arrive_count=256)
            bar_vl1_ready = T.alloc_barrier(arrive_count=256)
            bar_vr0_ready = T.alloc_barrier(arrive_count=256)
            bar_vr1_ready = T.alloc_barrier(arrive_count=256)
            # bar_vr0_free = T.alloc_barrier(arrive_count=256)
            # bar_vr1_free = T.alloc_barrier(arrive_count=256)
            bar_vl0_free = T.alloc_barrier(arrive_count=256)
            bar_vl1_free = T.alloc_barrier(arrive_count=256)
            # bar_p_free = T.alloc_barrier(arrive_count=256)
            bar_p_ready = T.alloc_barrier(arrive_count=256)
            bar_final = T.alloc_barrier(arrive_count=256)
            # bar_final_free = T.alloc_barrier(arrive_count=256)
            T.sync_threads()

            begin_idx = T.alloc_var(T.int32)
            sched_begin_block_idx = T.alloc_var(T.int32)
            end_idx = T.alloc_var(T.int32)
            sched_end_block_idx = T.alloc_var(T.int32)
            begin_n_split_idx = T.alloc_var(T.int32)
            phase_count = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)
            begin_idx = tile_scheduler_metadata[bz, 0]
            sched_begin_block_idx = tile_scheduler_metadata[bz, 1]
            end_idx = tile_scheduler_metadata[bz, 2]
            sched_end_block_idx = tile_scheduler_metadata[bz, 3]
            begin_n_split_idx = tile_scheduler_metadata[bz, 4]

            q_robust_desc = T.make_robust_desc(
                T.address_of(q[0, 0, 0, 0]),
                q_cosize * dtype_bytes,
            )
            kv_robust_desc = T.make_robust_desc(T.address_of(kv[0, 0, 0]), kv_cosize)

            g_i = by
            s_i = bx if head_repeats == 1 else (bx // head_repeats)
            q_i = s_i

            h0 = g_i * padded_head_kv + (
                0 if head_repeats == 1 else (bx % head_repeats) * 64
            )
            h1 = h0 + heads_per_block
            tid = T.get_thread_binding()
            for b_i in range(begin_idx, end_idx + 1, 1):
                T.call_extern("void", "__musa_loop_transparent_outermost")
                start_block_idx = T.alloc_var(T.int32)
                end_block_idx = T.alloc_var(T.int32)
                n_split_idx = T.alloc_var(T.int32)
                dynamic_total_blocks = T.alloc_var(T.int32)
                if has_topk_length:
                    dynamic_total_blocks = T.max(
                        T.ceildiv(topk_length[b_i], block_i), 1
                    )
                else:
                    dynamic_total_blocks = T.max(T.ceildiv(topk, block_i), 1)
                start_block_idx = T.if_then_else(
                    b_i == begin_idx, sched_begin_block_idx, 0
                )
                end_block_idx = T.if_then_else(
                    b_i == end_idx, sched_end_block_idx, dynamic_total_blocks
                )
                n_split_idx = T.if_then_else(b_i == begin_idx, begin_n_split_idx, 0)
                is_unsplit = (num_splits[b_i + 1] - num_splits[b_i]) == 1
                if tid < 512:
                    # T.barrier_wait(bar_q_free, (b_i - begin_idx+1) & 1)
                    T.tma_copy(
                        q[b_i, s_i, h0:h1, 0 : dim_qk // 2],
                        q_shared_l,
                        barrier=bar_q,
                    )
                    T.tma_copy(
                        q[b_i, s_i, h0:h1, dim_qk // 2 : dim_qk],
                        q_shared_r,
                        barrier=bar_q,
                    )
                    T.tma_copy(
                        q[b_i, s_i, h0:h1, dim_qk:],
                        q_tail_shared,
                        barrier=bar_q,
                    )

                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    T.barrier_arrive(bar_q)
                    T.barrier_wait(bar_q, (b_i - begin_idx) & 1)
                if tid < 256:
                    sumexp = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_inv = T.alloc_fragment([heads_per_block], accum_dtype)
                    alpha_local = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([heads_per_block], accum_dtype)
                    acc_s = T.alloc_fragment([heads_per_block, block_i], accum_dtype)
                    acc_s_cast = T.alloc_fragment([heads_per_block, block_i], dtype)
                    acc_o_l_0 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    acc_o_l_1 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    kv_reg_l = T.alloc_local([64], dtype)
                    kv_reg_l_fp16 = T.view(kv_reg_l, [64], T.float16)
                    kv_reg_l_bf16_load = T.alloc_local([32], T.bfloat16)
                    kv_reg_l_fp8 = T.view(kv_reg_l_bf16_load, [64], kv_latent_dtype)
                    quant_local_l = T.alloc_local([2, 2], T.float32)
                    c0_ldg_tx = (tid) % 8
                    c0_ldg_ty = (tid) // 8
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l_0, 0)
                    T.fill(acc_o_l_1, 0)
                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv0_ready, (phase_count[0] & 1))
                        T.copy(quant_shared[c0_ldg_ty, 0:2], quant_local_l[0, :])
                        T.copy(quant_shared[c0_ldg_ty + 32, 0:2], quant_local_l[1, :])
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
                        # KV_l fp8 quant
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_l_bf16_load[r * 16 + u * 4 + v] = (
                                        kv_shared_l[
                                            (c0_ldg_ty + r * 32),
                                            64 * u + c0_ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_l[idx] = T.Cast(
                                        "bfloat16",
                                        kv_reg_l_fp16[idx] * quant_local_l[r, 0],
                                    )

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_shared_l[
                                        c0_ldg_ty + r * 32,
                                        64 * u + c0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + u * 8 + v]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_l_bf16_load[r * 16 + (u + 2) * 4 + v] = (
                                        kv_shared_l[
                                            (c0_ldg_ty + r * 32),
                                            64 * (u + 2) + c0_ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_l[idx] = T.Cast(
                                        "bfloat16",
                                        kv_reg_l_fp16[idx] * quant_local_l[r, 1],
                                    )

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_shared_l[
                                        c0_ldg_ty + r * 32,
                                        64 * (u + 2) + c0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + (u + 2) * 8 + v]

                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_quant_ready)
                        T.barrier_wait(bar_kv_mask_ready, (phase_count[0] & 1))
                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                is_kv_valid[bi_i % 8 * 8 + bi_i // 8], 0, -(2**30)
                            )
                        T.lma_wait()
                        T.barrier_arrive(bar_kv_mask_free)

                        T.barrier_wait(bar_kv0_quant_ready, (phase_count[0] & 1))
                        T.gemm(
                            q_shared_l[:, :],
                            kv_shared_l[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv0_free)

                        T.barrier_wait(bar_kv1_quant_ready, (phase_count[0] & 1))
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
                        # T.barrier_arrive(bar_q_free)
                        T.barrier_arrive(bar_kv1_free)

                        update_online_softmax(
                            acc_s,
                            acc_s_cast,
                            scores_shared,
                            m_i,
                            m_i_prev,
                            sumexp,
                            sumexp_i,
                            alpha_local,
                            alpha_shared,
                            acc_o_l_0,
                            acc_o_l_1,
                            sm_scale,
                        )

                        T.lma_wait()
                        T.barrier_arrive(bar_p_ready)
                        stage_value_shared(
                            v_shared_0, kv_reg_l, c0_ldg_ty, c0_ldg_tx, 0
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vl0_ready)
                        T.barrier_wait(bar_vl0_ready, (phase_count[0] & 1))

                        T.gemm(
                            scores_shared,
                            v_shared_0,
                            acc_o_l_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        stage_value_shared(
                            v_shared_1, kv_reg_l, c0_ldg_ty, c0_ldg_tx, 2
                        )

                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vl0_free)

                        T.lma_wait()
                        T.barrier_arrive(bar_vl1_ready)
                        T.barrier_wait(bar_vl1_ready, (phase_count[0] & 1))

                        T.gemm(
                            scores_shared,
                            v_shared_1,
                            acc_o_l_1,
                            transpose_B=False,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vl1_free)
                        phase_count[0] = phase_count[0] ^ 1
                    finalize_left(
                        b_i,
                        s_i,
                        h0,
                        h1,
                        is_unsplit,
                        m_i,
                        sumexp,
                        sumexp_inv,
                        sum_exp_inv_shared,
                        sink_scale_shared,
                        acc_o_l_0,
                        acc_o_l_1,
                        output,
                        output_partial,
                        lse,
                        glse,
                        n_split_idx,
                        num_splits,
                        attn_sink,
                        bar_final,
                        sm_scale,
                    )
                elif tid >= 256 and tid < 512:
                    acc_o_r_0 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    acc_o_r_1 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    kv_reg_r = T.alloc_local([64], dtype)
                    kv_reg_r_fp16 = T.view(kv_reg_r, [64], T.float16)
                    kv_reg_r_fp8 = T.alloc_local([64], kv_latent_dtype)
                    kv_reg_r_bf16 = T.view(kv_reg_r_fp8, [32], T.bfloat16)
                    quant_local_r = T.alloc_local([2, 2], T.float32)
                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)
                    c1_ldg_tx = (tid - 256) % 8
                    c1_ldg_ty = (tid - 256) // 8
                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv1_ready, (phase_count[0] & 1))
                        T.copy(quant_shared[c1_ldg_ty, 2:4], quant_local_r[0, :])
                        T.copy(quant_shared[c1_ldg_ty + 32, 2:4], quant_local_r[1, :])

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
                        # KV_r fp8 quant
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_r_bf16[r * 16 + u * 4 + v] = kv_shared_r[
                                        c1_ldg_ty + r * 32,
                                        64 * u + c1_ldg_tx * 8 + v,
                                    ]
                        T.lma_wait()

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_r[idx] = T.Cast(
                                        "bfloat16",
                                        kv_reg_r_fp16[idx] * quant_local_r[r, 0],
                                    )

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_shared_r[
                                        c1_ldg_ty + r * 32,
                                        64 * u + c1_ldg_tx * 8 + v,
                                    ] = kv_reg_r[r * 32 + u * 8 + v]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_r_bf16[r * 16 + (u + 2) * 4 + v] = (
                                        kv_shared_r[
                                            (c1_ldg_ty + r * 32),
                                            64 * (u + 2) + c1_ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_r[idx] = T.Cast(
                                        "bfloat16",
                                        kv_reg_r_fp16[idx] * quant_local_r[r, 1],
                                    )

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_shared_r[
                                        c1_ldg_ty + r * 32,
                                        64 * (u + 2) + c1_ldg_tx * 8 + v,
                                    ] = kv_reg_r[r * 32 + (u + 2) * 8 + v]

                        T.lma_wait()
                        T.barrier_arrive(bar_kv1_quant_ready)

                        T.barrier_wait(bar_vl0_free, (phase_count[0] & 1))

                        stage_value_shared(
                            v_shared_0, kv_reg_r, c1_ldg_ty, c1_ldg_tx, 0
                        )

                        T.lma_wait()
                        T.barrier_arrive(bar_vr0_ready)
                        T.barrier_wait(bar_vr0_ready, (phase_count[0] & 1))

                        # compute v4-v7
                        T.barrier_wait(bar_p_ready, (phase_count[0] & 1))
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
                        T.warpgroup_commit_batch()

                        T.barrier_wait(bar_vl1_free, (phase_count[0] & 1))

                        stage_value_shared(
                            v_shared_1, kv_reg_r, c1_ldg_ty, c1_ldg_tx, 2
                        )
                        T.lma_wait()
                        T.warpgroup_wait(0)
                        # T.barrier_arrive(bar_vr0_free)
                        T.barrier_arrive(bar_vr1_ready)
                        T.barrier_wait(bar_vr1_ready, (phase_count[0] & 1))

                        # compute v4-v7
                        T.gemm(
                            scores_shared,
                            v_shared_1,
                            acc_o_r_1,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        # T.barrier_arrive(bar_p_free)
                        # T.barrier_arrive(bar_vr1_free)
                        phase_count[0] = phase_count[0] ^ 1
                    finalize_right(
                        b_i,
                        s_i,
                        h0,
                        h1,
                        is_unsplit,
                        acc_o_r_0,
                        acc_o_r_1,
                        sum_exp_inv_shared,
                        sink_scale_shared,
                        output,
                        output_partial,
                        n_split_idx,
                        num_splits,
                        bar_final,
                        (b_i - begin_idx) & 1,
                    )
                elif tid >= 512:
                    kperm_mask_local = T.alloc_local([4], "bool")
                    kperm_indices_local = T.alloc_local([4], "int32")
                    # producer: 128 threads, 16 producer_ldg_ty rows
                    producer_ldg_tx = (tid - 512) % 8
                    producer_ldg_ty = (tid - 512) // 8
                    ldg_scale_tx = (tid - 512) % 2
                    ldg_scale_ty = (tid - 512) // 2
                    for i_i in range(start_block_idx, end_block_idx):
                        if has_topk_length:
                            effective_topk = topk_length[b_i]
                        else:
                            effective_topk = topk
                        load_indices(
                            indices,
                            b_i,
                            s_i,
                            g_i,
                            i_i,
                            effective_topk,
                            seq_len_kv,
                            producer_ldg_ty,
                            producer_ldg_tx,
                            phase_count[0],
                            kperm_indices_local,
                            kperm_mask_local,
                            is_kv_valid,
                            kv_indices,
                            bar_kv_mask_free,
                        )
                        T.barrier_wait(bar_kv0_free, (phase_count[0] & 1) ^ 1)
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
                                for v in T.vectorized(4):
                                    pass
                                    T.copy(
                                        k_pe[
                                            kperm_indices_local[r],
                                            g_i,
                                            32 * u + producer_ldg_tx * 4 + v,
                                        ],
                                        kv_shared_l[
                                            r * 16 + producer_ldg_ty,
                                            64 * u + producer_ldg_tx * 8 + v,
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )

                        T.lma_wait()
                        T.barrier_arrive(bar_kv_mask_ready)
                        T.barrier_arrive(bar_indices_ready)
                        T.barrier_wait(bar_indices_ready, (phase_count[0] & 1))

                        for c in T.vectorized(2):
                            T.copy(
                                quant_scales[
                                    kv_indices[ldg_scale_ty],
                                    g_i,
                                    128 + ldg_scale_tx * 2 + c,
                                ],
                                quant_shared[ldg_scale_ty, ldg_scale_tx * 2 + c],
                                src_robust_desc=kv_robust_desc,
                            )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.barrier_arrive(bar_kv0_ready)

                        T.barrier_wait(bar_kv1_free, (phase_count[0] & 1) ^ 1)

                        # load k rope
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
                                    k_pe[
                                        kperm_indices_local[r],
                                        g_i,
                                        dim_qk // 2 + 8 + producer_ldg_tx * 8 + v,
                                    ],
                                    k_tail_shared[
                                        r * 16 + producer_ldg_ty,
                                        producer_ldg_tx * 8 + v,
                                    ],
                                    force_async_copy=True,
                                    src_robust_desc=kv_robust_desc,
                                )

                        # load k4-k7
                        # T.annotate_layout(
                        #         { KV_shared_r[:, :]: tilelang.layout.make_sqmma_swizzled_layout(KV_shared_r[:, :], k_major=True) },
                        #         allow_reannotation=True,
                        #         allow_buffer_region=True)
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
                                for v in T.vectorized(4):
                                    pass
                                    T.copy(
                                        k_pe[
                                            kperm_indices_local[r],
                                            g_i,
                                            dim_qk // 4
                                            + 32 * u
                                            + producer_ldg_tx * 4
                                            + v,
                                        ],
                                        kv_shared_r[
                                            r * 16 + producer_ldg_ty,
                                            64 * u + producer_ldg_tx * 8 + v,
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.barrier_arrive(bar_kv1_ready)
                        T.sync_threads(1, 128)
                        phase_count[0] = phase_count[0] ^ 1

    dsa_combine = make_scheduled_decode_combine(
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        dim=dim,
        num_mp_parts=num_mp_parts,
        dtype=dtype,
        accum_dtype=accum_dtype,
        max_nums_splits=max_nums_splits,
        has_attn_sink=has_attn_sink,
        max_lse_init=-(2**30),
    )

    @T.macro
    def run_decode(
        q,
        kv,
        k_pe,
        quant_scales,
        indices,
        topk_length,
        attn_sink,
        tile_scheduler_metadata,
        num_splits,
        glse,
        output_partial,
        output,
        lse,
        sm_scale,
    ):
        dsa_decode_split(
            q,
            kv,
            k_pe,
            quant_scales,
            indices,
            topk_length,
            attn_sink,
            tile_scheduler_metadata,
            num_splits,
            glse,
            output_partial,
            output,
            lse,
            sm_scale,
        )
        if support_split:
            dsa_combine(
                tile_scheduler_metadata,
                num_splits,
                glse,
                output_partial,
                attn_sink,
                output,
                lse,
            )

    if support_split:
        glse_type = T.Tensor([batch + num_mp_parts, seq_len, num_heads], accum_dtype)
        output_partial_type = T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads, dim], accum_dtype
        )
        if has_topk_length and has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv: kv_type,
                k_pe: k_pe_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: T.Tensor([batch], T.int32),
                attn_sink: T.Tensor([num_heads], T.float32),
                tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),
                num_splits: T.Tensor([batch + 1], T.int32),
                glse: glse_type,
                output_partial: output_partial_type,
                output: T.Tensor([batch, seq_len, num_heads, dim], dtype),
                lse: T.Tensor([batch, num_heads, seq_len], accum_dtype),
                sm_scale: T.float32,
            ):
                run_decode(
                    q,
                    kv,
                    k_pe,
                    quant_scales,
                    indices,
                    topk_length,
                    attn_sink,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                    sm_scale,
                )

        elif has_topk_length:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv: kv_type,
                k_pe: k_pe_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: T.Tensor([batch], T.int32),
                tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),
                num_splits: T.Tensor([batch + 1], T.int32),
                glse: glse_type,
                output_partial: output_partial_type,
                output: T.Tensor([batch, seq_len, num_heads, dim], dtype),
                lse: T.Tensor([batch, num_heads, seq_len], accum_dtype),
                sm_scale: T.float32,
            ):
                run_decode(
                    q,
                    kv,
                    k_pe,
                    quant_scales,
                    indices,
                    topk_length,
                    None,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                    sm_scale,
                )

        elif has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv: kv_type,
                k_pe: k_pe_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                attn_sink: T.Tensor([num_heads], T.float32),
                tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),
                num_splits: T.Tensor([batch + 1], T.int32),
                glse: glse_type,
                output_partial: output_partial_type,
                output: T.Tensor([batch, seq_len, num_heads, dim], dtype),
                lse: T.Tensor([batch, num_heads, seq_len], accum_dtype),
                sm_scale: T.float32,
            ):
                run_decode(
                    q,
                    kv,
                    k_pe,
                    quant_scales,
                    indices,
                    None,
                    attn_sink,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                    sm_scale,
                )

        else:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv: kv_type,
                k_pe: k_pe_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),
                num_splits: T.Tensor([batch + 1], T.int32),
                glse: glse_type,
                output_partial: output_partial_type,
                output: T.Tensor([batch, seq_len, num_heads, dim], dtype),
                lse: T.Tensor([batch, num_heads, seq_len], accum_dtype),
                sm_scale: T.float32,
            ):
                run_decode(
                    q,
                    kv,
                    k_pe,
                    quant_scales,
                    indices,
                    None,
                    None,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                    sm_scale,
                )
    elif has_topk_length and has_attn_sink:

        @T.prim_func
        def dsa_decode(
            q: q_type,
            kv: kv_type,
            k_pe: k_pe_type,
            quant_scales: quant_scales_type,
            indices: indices_type,
            topk_length: T.Tensor([batch], T.int32),
            attn_sink: T.Tensor([num_heads], T.float32),
            tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),
            num_splits: T.Tensor([batch + 1], T.int32),
            output: T.Tensor([batch, seq_len, num_heads, dim], dtype),
            lse: T.Tensor([batch, num_heads, seq_len], accum_dtype),
            sm_scale: T.float32,
        ):
            run_decode(
                q,
                kv,
                k_pe,
                quant_scales,
                indices,
                topk_length,
                attn_sink,
                tile_scheduler_metadata,
                num_splits,
                None,
                None,
                output,
                lse,
                sm_scale,
            )

    elif has_topk_length:

        @T.prim_func
        def dsa_decode(
            q: q_type,
            kv: kv_type,
            k_pe: k_pe_type,
            quant_scales: quant_scales_type,
            indices: indices_type,
            topk_length: T.Tensor([batch], T.int32),
            tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),
            num_splits: T.Tensor([batch + 1], T.int32),
            output: T.Tensor([batch, seq_len, num_heads, dim], dtype),
            lse: T.Tensor([batch, num_heads, seq_len], accum_dtype),
            sm_scale: T.float32,
        ):
            run_decode(
                q,
                kv,
                k_pe,
                quant_scales,
                indices,
                topk_length,
                None,
                tile_scheduler_metadata,
                num_splits,
                None,
                None,
                output,
                lse,
                sm_scale,
            )

    elif has_attn_sink:

        @T.prim_func
        def dsa_decode(
            q: q_type,
            kv: kv_type,
            k_pe: k_pe_type,
            quant_scales: quant_scales_type,
            indices: indices_type,
            attn_sink: T.Tensor([num_heads], T.float32),
            tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),
            num_splits: T.Tensor([batch + 1], T.int32),
            output: T.Tensor([batch, seq_len, num_heads, dim], dtype),
            lse: T.Tensor([batch, num_heads, seq_len], accum_dtype),
            sm_scale: T.float32,
        ):
            run_decode(
                q,
                kv,
                k_pe,
                quant_scales,
                indices,
                None,
                attn_sink,
                tile_scheduler_metadata,
                num_splits,
                None,
                None,
                output,
                lse,
                sm_scale,
            )

    else:

        @T.prim_func
        def dsa_decode(
            q: q_type,
            kv: kv_type,
            k_pe: k_pe_type,
            quant_scales: quant_scales_type,
            indices: indices_type,
            tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),
            num_splits: T.Tensor([batch + 1], T.int32),
            output: T.Tensor([batch, seq_len, num_heads, dim], dtype),
            lse: T.Tensor([batch, num_heads, seq_len], accum_dtype),
            sm_scale: T.float32,
        ):
            run_decode(
                q,
                kv,
                k_pe,
                quant_scales,
                indices,
                None,
                None,
                tile_scheduler_metadata,
                num_splits,
                None,
                None,
                output,
                lse,
                sm_scale,
            )

    return dsa_decode


def tilelang_flashmla_interface(
    q,
    kv,
    indices,
    tile_scheduler_metadata,
    num_splits,
    sm_scale=None,
    topk_length=None,
    attn_sink=None,
    return_p_sum: bool = False,
    d_v=512,
    threads=640,
    verbose=False,
):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert kv.dtype == torch.uint8, "kv must be uint8"
    assert indices.dtype == torch.int32, "indices must be int32"
    q, q_shape = prepare_sparse_mla_decode_strided_tensor("q", q, multiple=8)
    kv, kv_shape = prepare_sparse_mla_decode_strided_tensor("kv", kv, multiple=16)
    indices, indices_shape = prepare_sparse_mla_decode_strided_tensor(
        "indices", indices, multiple=8
    )
    b, seq_len, heads, dim_plus_tail_dim = q_shape
    seq_len_kv, kv_group, _ = kv_shape
    #  In FP8+sparse mode, each token's kv cache is 656 bytes, structured as:
    #         - The shape of the tensor `k_cache` is (num_blocks*page_block_size*num_heads_k, head_dim), and num_heads_k must be 1.
    #         - First 512 bytes: The "quantized NoPE" part, containing 512 float8_e4m3 values.
    #         - Next 16 bytes: Scale factors, containing 4 float32 values. The first float32 is the scale for the first 128 float8_e4m3 values, the second for the next 128, and so on.
    #         - Last 128 bytes: The "RoPE" part, containing 64 bfloat16 values. This part is not quantized for accuracy.
    # assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v
    dim_bytes = 656
    assert kv_shape[-1] == dim_bytes
    tail_dim = dim_plus_tail_dim - dim
    _, _, _, topk = indices_shape
    assert indices_shape == (b, seq_len, kv_group, topk)
    assert topk % 64 == 0, "topk must be a multiple of 64"
    num_mp_parts = int(tile_scheduler_metadata.shape[0])
    support_split = num_mp_parts != 1
    runtime = prepare_scheduled_decode_runtime(
        batch=b,
        seq_len=seq_len,
        heads=heads,
        dim=d_v,
        topk_length=topk_length,
        attn_sink=attn_sink,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        out_dtype=q.dtype,
        device=q.device,
        variant_name="V3.2",
        support_split=support_split,
    )
    kv_latent_f8 = kv.view(torch.float8_e4m3fn)
    k_rope = kv.view(torch.bfloat16)
    scales = kv.view(torch.float32)
    address_tensors = [q, kv, indices]
    if runtime.out_partial is not None:
        address_tensors.append(runtime.out_partial)
    kernel_factory = jit_for_tensor_addressing(
        sparse_attention_fwd_kernel, *address_tensors
    )
    # kernel = sparse_attention_fwd_kernel_v1(
    threads = 640
    kernel = kernel_factory(
        heads,
        dim,
        tail_dim,
        kv_group=kv_group,
        threads=threads,
        # Match scheduled_max_num_splits(): 32 for <=32 MP parts and 64 for
        # larger supported metadata.  This keeps the finite AOT dispatch
        # exhaustive without making max_splits a runtime compile argument.
        max_nums_splits=runtime.max_nums_splits if support_split else 1,
        has_attn_sink=runtime.has_attn_sink,
        has_topk_length=runtime.topk_length is not None,
        support_split=support_split,
        use_int64_cosize=kernel_factory is not sparse_attention_fwd_kernel,
    )
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()
    args = [q, kv_latent_f8, k_rope, scales, indices]
    if runtime.topk_length is not None:
        args.append(runtime.topk_length)
    if runtime.attn_sink is not None:
        args.append(runtime.attn_sink)
    args.extend((tile_scheduler_metadata, num_splits))
    if support_split:
        assert runtime.glse is not None and runtime.out_partial is not None
        args.extend((runtime.glse, runtime.out_partial))
    args.extend((runtime.out, runtime.lse))
    runtime_sm_scale = (
        (1.0 / (dim + tail_dim)) ** 0.5 if sm_scale is None else float(sm_scale)
    )
    runtime_sm_scale *= 1.44269504
    args.append(runtime_sm_scale)
    kernel(*args)
    return runtime.out, runtime.lse
