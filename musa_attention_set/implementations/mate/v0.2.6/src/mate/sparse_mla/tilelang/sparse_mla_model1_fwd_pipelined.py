# ruff: noqa
"""
MODEL1 sparse MLA prefill with a pipelined producer/consumer schedule.

Compared with the DeepSeek V3.2 sparse prefill kernel:
- q/k/v head_dim is 512, so there is no 512 + 64 tail split.
- The sparse candidates can come from both main and extra kv tensors.
- Each source has its own dynamic effective topk length.
- This remains a bf16 prefill-only research kernel.
"""

from typing import Any

import torch
import tilelang
from tilelang import language as T

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


# tilelang.disable_cache()


MODEL1_SPARSE_PREFILL_PASS_CONFIGS = {
    **SPARSE_PREFILL_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
}


@tilelang.jit(
    out_idx=[3, 4, 5],
    pass_configs=MODEL1_SPARSE_PREFILL_PASS_CONFIGS,
    compile_flags=SPARSE_PREFILL_COMPILE_FLAGS,
)
def sparse_attention_fwd_kernel_model1(
    num_heads,
    dim,
    *,
    kv_group=1,
    is_causal=True,
    block_i=64,
    threads=640,
    has_attn_sink=False,
    has_topk_length=False,
    is_persistence=True,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    topk = T.dynamic("topk")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    kv_stride_s = T.dynamic("kv_stride_s")
    kv_stride_g = T.dynamic("kv_stride_g")
    indices_stride_s = T.dynamic("indices_stride_s")
    indices_stride_g = T.dynamic("indices_stride_g")

    head_kv = num_heads // kv_group
    q_shape = [seq_len, num_heads, dim]
    kv_shape = [seq_len_kv, kv_group, dim]
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
    qk_panel = block_i * 2
    assert dim_qk % qk_panel == 0
    qk_panels = dim_qk // qk_panel
    lanes_per_vec = block_i // 8
    topk_blocks = (topk + block_i - 1) // block_i

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        head_repeats = head_kv // 64
    else:
        head_repeats = 1

    heads_per_block = padded_head_kv if head_repeats == 1 else 64
    if is_persistence:
        assert kv_group == 1, "persistent MODEL1 prefill currently supports MQA only"
    logical_blocks = seq_len * head_repeats
    producer_threads = 128
    consumer0_threads = 256
    consumer1_threads = 256

    @T.macro
    def load_q_panels(q, q_shared, s_i, h0, h1, bar_q_ready):
        for panel in T.unroll(qk_panels):
            T.tma_copy(
                q[
                    s_i,
                    h0:h1,
                    panel * qk_panel : (panel + 1) * qk_panel,
                ],
                q_shared[panel, :, :],
                barrier=bar_q_ready,
            )

    @T.macro
    def load_kv_half(
        kv,
        kv_shared,
        kperm_indices_local,
        g_i,
        panel_offset,
        producer_ldg_tx,
        producer_ldg_ty,
        kv_robust_desc,
    ):
        for r in T.unroll(4):
            for panel in T.unroll(qk_panels // 2):
                for u in T.unroll(qk_panel // block_i):
                    for v in T.vectorized(8):
                        T.copy(
                            kv[
                                kperm_indices_local[r],
                                g_i,
                                (panel + panel_offset) * qk_panel
                                + block_i * u
                                + producer_ldg_tx * 8
                                + v,
                            ],
                            kv_shared[
                                panel + panel_offset,
                                r * 16 + producer_ldg_ty,
                                block_i * u + producer_ldg_tx * 8 + v,
                            ],
                            force_async_copy=True,
                            src_robust_desc=kv_robust_desc,
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
        sm_scale,
        persistent_blocks,
    ):
        launch_blocks = (
            T.min(persistent_blocks, logical_blocks)
            if is_persistence
            else logical_blocks
        )
        with T.Kernel(launch_blocks, kv_group, threads=threads) as (bx, by):
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(kv_stride_s % 8 == 0)
            T.assume(kv_stride_g % 8 == 0)
            T.assume(indices_stride_s % 8 == 0)
            T.assume(indices_stride_g % 8 == 0)
            q_shared = T.alloc_shared([qk_panels, heads_per_block, qk_panel], dtype)
            kv_shared = T.alloc_shared([qk_panels, block_i, qk_panel], dtype)
            v_shared_0 = T.alloc_shared([block_i, dim_qk // 4], dtype)
            v_shared_1 = T.alloc_shared([block_i, dim_qk // 4], dtype)

            scores_shared = T.alloc_shared([heads_per_block, block_i], dtype)
            sum_exp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([block_i], "bool", scope="shared")

            bar_q_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_q_free = T.alloc_barrier(arrive_count=consumer0_threads)

            bar_kv0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv1_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv1_read_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_kv0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            # kv_shared right panels are consumed by both groups: consumer0's
            # async SQMMA and consumer1's LMA read.  Producer reuse must wait
            # for both completions (consumer0 + consumer1 arrivals).
            bar_kv1_free = T.alloc_barrier(
                arrive_count=consumer0_threads + consumer1_threads
            )

            bar_vl_pair_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vr0_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vr1_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vl0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vl1_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vr0_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vr1_free = T.alloc_barrier(arrive_count=consumer1_threads)

            bar_p_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_final = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_final_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_producer_protect = T.alloc_barrier(arrive_count=producer_threads)

            q_robust_desc = T.make_robust_desc(
                T.address_of(q[0, 0, 0]),
                q_cosize * dtype_bytes,
            )
            kv_robust_desc = T.make_robust_desc(
                T.address_of(kv[0, 0, 0]),
                kv_cosize * dtype_bytes,
            )
            T.sync_threads()

            tid = T.get_thread_binding()
            logical_bx = T.alloc_var(T.int32)
            phase_count = T.alloc_local([1], T.int32)
            logical_phase = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)
            T.fill(logical_phase, 0)
            logical_bx = bx
            if tid >= 512 and logical_bx < seq_len * head_repeats:
                bootstrap_s_i = (
                    logical_bx if head_repeats == 1 else logical_bx // head_repeats
                )
                bootstrap_h0 = by * padded_head_kv + (
                    0 if head_repeats == 1 else (logical_bx % head_repeats) * 64
                )
                bootstrap_h1 = bootstrap_h0 + heads_per_block
                load_q_panels(
                    q,
                    q_shared,
                    bootstrap_s_i,
                    bootstrap_h0,
                    bootstrap_h1,
                    bar_q_ready,
                )
                T.barrier_arrive(bar_q_ready)

            if tid >= 256 and tid < 512:
                T.barrier_arrive(bar_vr0_free)
                T.barrier_arrive(bar_vr1_free)

            while logical_bx < seq_len * head_repeats:
                if is_persistence:
                    T.call_extern("void", "__musa_loop_transparent_outermost")
                g_i = by
                s_i = logical_bx if head_repeats == 1 else (logical_bx // head_repeats)
                h0 = g_i * padded_head_kv + (
                    0 if head_repeats == 1 else (logical_bx % head_repeats) * 64
                )
                h1 = h0 + heads_per_block
                dynamic_total_blocks = T.alloc_var(T.int32)
                if has_topk_length:
                    dynamic_total_blocks = T.ceildiv(topk_length[s_i], block_i)
                else:
                    dynamic_total_blocks = T.ceildiv(topk, block_i)

                if tid < 256:
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
                    consumer0_ldg_tx = tid % 8
                    consumer0_ldg_ty = tid // 8
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l_0, 0)
                    T.fill(acc_o_l_1, 0)
                    T.barrier_wait(bar_q_ready, logical_phase[0] & 1)
                    for i_i in range(dynamic_total_blocks):
                        T.barrier_wait(bar_kv0_ready, phase_count[0] & 1)

                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                is_kv_valid[bi_i % 8 * 8 + bi_i // 8],
                                0,
                                -(2**30),
                            )

                        T.annotate_layout(
                            {
                                kv_shared[
                                    0, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[0, :, :],
                                    k_major=True,
                                ),
                                kv_shared[
                                    1, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[1, :, :],
                                    k_major=True,
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.gemm(
                            q_shared[0, :, :],
                            kv_shared[0, :, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        for r in T.unroll(2):
                            for v in T.vectorized(8):
                                kv_reg_l[r * 32 + v] = kv_shared[
                                    0,
                                    ((consumer0_ldg_ty + r * 32) % 8) * (block_i // 8)
                                    + (consumer0_ldg_ty + r * 32) // 8,
                                    consumer0_ldg_tx * 8 + v,
                                ]
                        for r in T.unroll(2):
                            for v in T.vectorized(8):
                                kv_reg_l[r * 32 + 8 + v] = kv_shared[
                                    0,
                                    ((consumer0_ldg_ty + r * 32) % 8) * (block_i // 8)
                                    + (consumer0_ldg_ty + r * 32) // 8,
                                    64 + consumer0_ldg_tx * 8 + v,
                                ]

                        T.gemm(
                            q_shared[1, :, :],
                            kv_shared[1, :, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )

                        T.barrier_wait(bar_vr0_free, phase_count[0] & 1)
                        for r in T.unroll(2):
                            for v in T.vectorized(8):
                                v_shared_0[
                                    r * 32 + consumer0_ldg_ty,
                                    consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[r * 32 + v]
                        for r in T.unroll(2):
                            for v in T.vectorized(8):
                                v_shared_0[
                                    r * 32 + consumer0_ldg_ty,
                                    64 + consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[r * 32 + 8 + v]
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_reg_l[r * 32 + (u + 2) * 8 + v] = kv_shared[
                                        1,
                                        ((consumer0_ldg_ty + r * 32) % 8)
                                        * (block_i // 8)
                                        + (consumer0_ldg_ty + r * 32) // 8,
                                        64 * u + consumer0_ldg_tx * 8 + v,
                                    ]
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_free)

                        T.barrier_wait(bar_kv1_ready, phase_count[0] & 1)
                        T.barrier_arrive(bar_kv1_read_ready)
                        T.annotate_layout(
                            {
                                kv_shared[
                                    qk_panels // 2, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[qk_panels // 2, :, :],
                                    k_major=True,
                                ),
                                kv_shared[
                                    qk_panels // 2 + 1, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[qk_panels // 2 + 1, :, :],
                                    k_major=True,
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.gemm(
                            q_shared[qk_panels // 2, :, :],
                            kv_shared[qk_panels // 2, :, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.barrier_wait(bar_vr1_free, phase_count[0] & 1)
                        for v in T.vectorized(8):
                            v_shared_1[
                                consumer0_ldg_ty,
                                consumer0_ldg_tx * 8 + v,
                            ] = kv_reg_l[16 + v]
                        T.gemm(
                            q_shared[qk_panels // 2 + 1, :, :],
                            kv_shared[qk_panels // 2 + 1, :, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        for v in T.vectorized(8):
                            v_shared_1[
                                consumer0_ldg_ty,
                                64 + consumer0_ldg_tx * 8 + v,
                            ] = kv_reg_l[24 + v]
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                v_shared_1[
                                    32 + consumer0_ldg_ty,
                                    64 * u + consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[48 + u * 8 + v]
                        T.warpgroup_commit_batch()
                        T.lma_wait()
                        T.barrier_arrive(bar_vl_pair_ready)
                        T.copy(m_i, m_i_prev)
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv1_free)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(heads_per_block):
                            m_i[h_i] = T.max(m_i_prev[h_i], m_i[h_i])
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

                        T.annotate_layout(
                            {
                                v_shared_0[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    v_shared_0[:, :],
                                    k_major=False,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.barrier_wait(bar_vl_pair_ready, phase_count[0] & 1)

                        T.gemm(
                            scores_shared,
                            v_shared_0,
                            acc_o_l_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.barrier_arrive(bar_p_ready)
                        for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                            acc_o_l_1[h_i, d_i] *= alpha_local[h_i]

                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(heads_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]

                        T.annotate_layout(
                            {
                                v_shared_1[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    v_shared_1[:, :],
                                    k_major=False,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vl0_free)

                        T.gemm(
                            scores_shared,
                            v_shared_1,
                            acc_o_l_1,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vl1_free)
                        phase_count[0] = phase_count[0] ^ 1

                    T.barrier_arrive(bar_q_free)
                    T.barrier_wait(bar_final_free, (logical_phase[0] & 1) ^ 1)
                    for h_i in T.Parallel(heads_per_block):
                        if m_i[h_i] > -(2**29):
                            if has_attn_sink:
                                sumexp_inv[h_i] = 1 / (
                                    sumexp[h_i]
                                    + T.exp2(
                                        attn_sink[h0 + h_i] * 1.4426950408889634
                                        - m_i[h_i] * sm_scale
                                    )
                                )
                            else:
                                sumexp_inv[h_i] = 1 / sumexp[h_i]
                            max_logits[h_i] = m_i[h_i] * sm_scale * 0.6931471805599453
                            sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                        else:
                            sumexp_inv[h_i] = 0
                            max_logits[h_i] = -T.infinity(accum_dtype)
                            sumexp[h_i] = T.infinity(accum_dtype)
                        sum_exp_inv_shared[h_i] = sumexp_inv[h_i]
                    T.barrier_arrive(bar_final)
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_l_0[h_i, d_i] *= sumexp_inv[h_i]
                        acc_o_l_1[h_i, d_i] *= sumexp_inv[h_i]

                    T.copy(acc_o_l_0, output[s_i, h0:h1, 0 : dim_qk // 4])
                    T.copy(acc_o_l_1, output[s_i, h0:h1, dim_qk // 4 : dim_qk // 2])
                    T.copy(max_logits, max_logits_out[s_i, h0:h1])
                    for h_i in T.Parallel(heads_per_block):
                        lse[s_i, h0 + h_i] = sumexp[h_i] * 0.6931471805599453
                elif tid < 512:
                    acc_o_r_0 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    acc_o_r_1 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    kv_reg_r = T.alloc_local([64], dtype)
                    alpha_r = T.alloc_fragment([heads_per_block], accum_dtype)
                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)

                    consumer1_ldg_tx = (tid - 256) % 8
                    consumer1_ldg_ty = (tid - 256) // 8
                    T.barrier_wait(bar_q_ready, logical_phase[0] & 1)
                    for i_i in range(dynamic_total_blocks):
                        T.barrier_wait(bar_kv1_read_ready, phase_count[0] & 1)
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_reg_r[r * 32 + u * 8 + v] = kv_shared[
                                        qk_panels // 2,
                                        ((consumer1_ldg_ty + r * 32) % 8)
                                        * (block_i // 8)
                                        + (consumer1_ldg_ty + r * 32) // 8,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ]
                                    kv_reg_r[r * 32 + (u + 2) * 8 + v] = kv_shared[
                                        qk_panels // 2 + 1,
                                        ((consumer1_ldg_ty + r * 32) % 8)
                                        * (block_i // 8)
                                        + (consumer1_ldg_ty + r * 32) // 8,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ]

                        T.lma_wait()
                        T.barrier_arrive(bar_kv1_free)
                        T.barrier_wait(bar_p_ready, phase_count[0] & 1)
                        for h_i in T.Parallel(heads_per_block):
                            alpha_r[h_i] = alpha_shared[h_i]
                        for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                            acc_o_r_0[h_i, d_i] *= alpha_r[h_i]
                        T.sched_boundary()
                        T.barrier_wait(bar_vl0_free, phase_count[0] & 1)
                        T.annotate_layout(
                            {
                                v_shared_0[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    v_shared_0[:, :],
                                    k_major=False,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_0[
                                        r * 32 + consumer1_ldg_ty,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ] = kv_reg_r[r * 32 + u * 8 + v]

                        T.lma_wait()
                        T.sched_boundary()
                        T.barrier_arrive(bar_vr0_ready)
                        T.barrier_wait(bar_vr0_ready, phase_count[0] & 1)

                        T.gemm(
                            scores_shared,
                            v_shared_0,
                            acc_o_r_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_vr0_free)
                        T.sched_boundary()
                        for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                            acc_o_r_1[h_i, d_i] *= alpha_r[h_i]

                        T.barrier_wait(bar_vl1_free, phase_count[0] & 1)
                        T.annotate_layout(
                            {
                                v_shared_1[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    v_shared_1[:, :],
                                    k_major=False,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_1[
                                        r * 32 + consumer1_ldg_ty,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ] = kv_reg_r[r * 32 + (u + 2) * 8 + v]
                        T.lma_wait()
                        T.barrier_arrive(bar_vr1_ready)
                        T.barrier_wait(bar_vr1_ready, phase_count[0] & 1)

                        T.gemm(
                            scores_shared,
                            v_shared_1,
                            acc_o_r_1,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_vr1_free)
                        phase_count[0] = phase_count[0] ^ 1

                    T.barrier_wait(bar_final, logical_phase[0] & 1)
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_r_0[h_i, d_i] *= sum_exp_inv_shared[h_i]
                        acc_o_r_1[h_i, d_i] *= sum_exp_inv_shared[h_i]

                    T.barrier_arrive(bar_final_free)
                    T.copy(
                        acc_o_r_0,
                        output[s_i, h0:h1, dim_qk // 2 : dim_qk // 2 + dim_qk // 4],
                    )
                    T.copy(
                        acc_o_r_1,
                        output[s_i, h0:h1, dim_qk // 2 + dim_qk // 4 : dim_qk],
                    )
                else:
                    kperm_mask_local = T.alloc_local([4], "bool")
                    kperm_indices_local = T.alloc_local([4], indices_dtype)
                    topk_len_local = T.alloc_local([1], indices_dtype)
                    producer_ldg_tx = (tid - 512) % 8
                    producer_ldg_ty = (tid - 512) // 8
                    if has_topk_length:
                        topk_len_local[0] = topk_length[s_i]
                    else:
                        topk_len_local[0] = topk

                    if dynamic_total_blocks > 0:
                        T.barrier_arrive(bar_producer_protect)
                        for r in T.unroll(4, explicit=True):
                            token_pos = ((r * 16 + producer_ldg_ty) % 8) * (
                                block_i // 8
                            ) + (r * 16 + producer_ldg_ty) // 8
                            kperm_indices_local[r] = indices[s_i, g_i, token_pos]

                        for r in T.unroll(4, explicit=True):
                            token_pos = ((r * 16 + producer_ldg_ty) % 8) * (
                                block_i // 8
                            ) + (r * 16 + producer_ldg_ty) // 8
                            kperm_mask_local[r] = (
                                T.Cast("uint32", kperm_indices_local[r])
                                < T.Cast("uint32", seq_len_kv)
                            ) and (token_pos < topk_len_local[0])
                            kperm_indices_local[r] = T.if_then_else(
                                kperm_mask_local[r], kperm_indices_local[r], 0
                            )

                        T.barrier_wait(bar_kv0_free, (phase_count[0] & 1) ^ 1)
                        T.annotate_layout(
                            {
                                kv_shared[
                                    0, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[0, :, :],
                                    k_major=True,
                                ),
                                kv_shared[
                                    1, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[1, :, :],
                                    k_major=True,
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        load_kv_half(
                            kv,
                            kv_shared,
                            kperm_indices_local,
                            g_i,
                            0,
                            producer_ldg_tx,
                            producer_ldg_ty,
                            kv_robust_desc,
                        )
                        T.ldlms_crossbb_commit_0()
                        for r in T.unroll(4):
                            is_kv_valid[
                                ((r * 16 + producer_ldg_ty) % 8) * (block_i // 8)
                                + (r * 16 + producer_ldg_ty) // 8
                            ] = kperm_mask_local[r]
                        T.lma_wait()

                        T.barrier_wait(bar_kv1_free, (phase_count[0] & 1) ^ 1)
                        T.annotate_layout(
                            {
                                kv_shared[
                                    qk_panels // 2, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[qk_panels // 2, :, :],
                                    k_major=True,
                                ),
                                kv_shared[
                                    qk_panels // 2 + 1, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[qk_panels // 2 + 1, :, :],
                                    k_major=True,
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        load_kv_half(
                            kv,
                            kv_shared,
                            kperm_indices_local,
                            g_i,
                            qk_panels // 2,
                            producer_ldg_tx,
                            producer_ldg_ty,
                            kv_robust_desc,
                        )
                        T.ldlms_crossbb_commit_1()

                        for i_i in T.serial(
                            1,
                            dynamic_total_blocks if has_topk_length else topk_blocks,
                        ):
                            T.ldlms_crossbb_wait_0()
                            T.barrier_arrive(bar_kv0_ready)

                            T.ldlms_crossbb_wait_1()
                            T.barrier_arrive(bar_kv1_ready)
                            T.barrier_wait(bar_producer_protect, phase_count[0] & 1)
                            phase_count[0] = phase_count[0] ^ 1
                            T.barrier_arrive(bar_producer_protect)

                            for r in T.unroll(4, explicit=True):
                                token_pos = (
                                    i_i * block_i
                                    + ((r * 16 + producer_ldg_ty) % 8) * (block_i // 8)
                                    + (r * 16 + producer_ldg_ty) // 8
                                )
                                kperm_indices_local[r] = indices[s_i, g_i, token_pos]

                            for r in T.unroll(4, explicit=True):
                                token_pos = (
                                    i_i * block_i
                                    + ((r * 16 + producer_ldg_ty) % 8) * (block_i // 8)
                                    + (r * 16 + producer_ldg_ty) // 8
                                )
                                kperm_mask_local[r] = (
                                    T.Cast("uint32", kperm_indices_local[r])
                                    < T.Cast("uint32", seq_len_kv)
                                ) and (token_pos < topk_len_local[0])
                                kperm_indices_local[r] = T.if_then_else(
                                    kperm_mask_local[r], kperm_indices_local[r], 0
                                )

                            T.barrier_wait(bar_kv0_free, (phase_count[0] & 1) ^ 1)
                            load_kv_half(
                                kv,
                                kv_shared,
                                kperm_indices_local,
                                g_i,
                                0,
                                producer_ldg_tx,
                                producer_ldg_ty,
                                kv_robust_desc,
                            )
                            T.ldlms_crossbb_commit_0()
                            for r in T.unroll(4):
                                is_kv_valid[
                                    ((r * 16 + producer_ldg_ty) % 8) * (block_i // 8)
                                    + (r * 16 + producer_ldg_ty) // 8
                                ] = kperm_mask_local[r]
                            T.lma_wait()

                            T.barrier_wait(bar_kv1_free, (phase_count[0] & 1) ^ 1)
                            load_kv_half(
                                kv,
                                kv_shared,
                                kperm_indices_local,
                                g_i,
                                qk_panels // 2,
                                producer_ldg_tx,
                                producer_ldg_ty,
                                kv_robust_desc,
                            )
                            T.ldlms_crossbb_commit_1()

                        T.ldlms_crossbb_wait_0()
                        T.barrier_arrive(bar_kv0_ready)
                        T.ldlms_crossbb_wait_1()
                        T.barrier_arrive(bar_kv1_ready)
                        T.barrier_wait(bar_producer_protect, phase_count[0] & 1)
                        phase_count[0] = phase_count[0] ^ 1
                    if is_persistence:
                        next_logical_bx = logical_bx + persistent_blocks
                        if next_logical_bx < seq_len * head_repeats:
                            T.barrier_wait(bar_q_free, logical_phase[0] & 1)
                            next_s_i = (
                                next_logical_bx
                                if head_repeats == 1
                                else next_logical_bx // head_repeats
                            )
                            next_h0 = by * padded_head_kv + (
                                0
                                if head_repeats == 1
                                else (next_logical_bx % head_repeats) * 64
                            )
                            next_h1 = next_h0 + heads_per_block
                            load_q_panels(
                                q,
                                q_shared,
                                next_s_i,
                                next_h0,
                                next_h1,
                                bar_q_ready,
                            )
                            T.barrier_arrive(bar_q_ready)
                if is_persistence:
                    logical_phase[0] = logical_phase[0] ^ 1
                    logical_bx += persistent_blocks
                else:
                    logical_bx = seq_len * head_repeats

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
            sm_scale: T.float32,
            persistent_blocks: T.int32,
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
                sm_scale,
                persistent_blocks,
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
            sm_scale: T.float32,
            persistent_blocks: T.int32,
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
                sm_scale,
                persistent_blocks,
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
            sm_scale: T.float32,
            persistent_blocks: T.int32,
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
                sm_scale,
                persistent_blocks,
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
            sm_scale: T.float32,
            persistent_blocks: T.int32,
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
                sm_scale,
                persistent_blocks,
            )

    return dsa_prefill


def sparse_mla_fwd_interface_model1(
    q,
    kv,
    indices,
    topk_length=None,
    sm_scale=None,
    attn_sink=None,
    return_p_sum: bool = False,
    d_v=512,
    threads=640,
    verbose=False,
    return_max_logits: bool = False,
    is_persistence: bool = True,
    persistent_blocks: int | None = None,
):
    is_causal = True
    assert return_p_sum is False, "This kernel file is for fwd only"
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert kv.dtype == torch.bfloat16, "kv must be bfloat16"
    assert indices.dtype == torch.int32, "indices must be int32"
    q, q_shape = prepare_sparse_mla_strided_tensor("q", q, multiple=8)
    kv, kv_shape = prepare_sparse_mla_strided_tensor("kv", kv, multiple=8)
    indices, indices_shape = prepare_sparse_mla_strided_tensor(
        "indices", indices, multiple=8
    )
    seq_len, heads, dim_q = q_shape
    seq_len_kv, kv_group, _ = kv_shape

    dim = d_v
    assert kv_group == 1, (
        "Only MQA (kv_group == 1) is validated for model1 sparse prefill"
    )
    assert dim_q == dim, (
        f"MODEL1 expects q last_dim == kv last_dim == {dim}, got q={dim_q}, kv={kv_shape[-1]}"
    )
    assert kv_shape[-1] == dim, (
        f"MODEL1 expects kv last_dim == {dim}, got {kv_shape[-1]}"
    )
    assert dim == 512, f"MODEL1 kernel currently expects dim == d_v == 512, got {dim}"

    _, _, topk = indices_shape
    assert indices_shape == (seq_len, kv_group, topk)
    assert topk % 64 == 0, "MODEL1 sparse prefill requires topk to be a multiple of 64"
    if is_persistence:
        persistent_blocks = resolve_num_mps(q.device, persistent_blocks)
    else:
        persistent_blocks = 0

    topk_length = validate_token_lengths(topk_length, seq_len, "topk_length")
    attn_sink = validate_prefill_attn_sink(attn_sink, heads)

    runtime_sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else float(sm_scale)
    runtime_sm_scale *= 1.44269504
    kernel_kwargs = {
        "kv_group": kv_group,
        "is_causal": is_causal,
        "threads": threads,
        "has_attn_sink": attn_sink is not None,
        "has_topk_length": topk_length is not None,
        "is_persistence": is_persistence,
    }
    # These inputs dominate the contiguous output and auxiliary tensor spans.
    kernel_factory = jit_for_tensor_addressing(
        sparse_attention_fwd_kernel_model1,
        q,
        kv,
        indices,
    )
    kernel = kernel_factory(heads, dim, **kernel_kwargs)
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()

    args = [q, kv, indices]
    if topk_length is not None:
        args.append(topk_length)
    if attn_sink is not None:
        args.append(attn_sink)
    args.append(runtime_sm_scale)
    args.append(int(persistent_blocks))
    out = kernel(*args)

    out_tensor, max_logits, lse_tensor = out
    if return_max_logits:
        return out_tensor, max_logits, lse_tensor
    return out_tensor, lse_tensor
