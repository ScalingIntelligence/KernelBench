# ruff: noqa

import torch
import tilelang
from tilelang import language as T

from .sparse_mla_decode_scheduled_common import (
    SCHEDULED_DECODE_COMPILE_FLAGS,
    SCHEDULED_DECODE_PASS_CONFIGS,
    make_scheduled_decode_combine,
    make_scheduled_decode_finalize_left,
    make_scheduled_decode_finalize_right,
    make_scheduled_decode_indices_loader,
    make_scheduled_decode_stage_value_shared,
)
from ...execution_context import raise_complete_if_dry_run
from ...mate_runtime import get_physical_num_mps


@tilelang.jit(
    out_idx=[],
    pass_configs={
        **SCHEDULED_DECODE_PASS_CONFIGS,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    },
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
    max_num_splits=1,
    launch_combine_by_scheduler=False,
    preload_all_indices=False,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        f"haven't check padding correctness yet, dim={tail_dim}"
    )
    topk = T.dynamic("topk")
    softmax_log2e = 1.44269504
    # Match the fixed-product ASM C1 schedule: keep independent 48-register
    # semantic images for V-L and V-R, and hide the first two V-R conversion
    # commands under the outstanding V-L shared-memory writes.
    consumer1_vr_overlap_values = 16

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_mp_parts = T.dynamic("num_mp_parts")
    support_split = max_num_splits > 1

    head_kv = num_heads // kv_group
    indices_dtype = "int32"
    dtype = "bfloat16"
    q_dtype = "float8_e4m3"
    accum_dtype = "float"
    dim_bytes = 576
    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), block_h)
    if padded_head_kv != head_kv:
        assert kv_group == 1
    kv_latent_dtype = "float8_e4m3"
    dim_qk = dim
    tail_dim_qk = tail_dim
    qk_native_k = 128
    assert dim_qk == 4 * qk_native_k
    if head_kv > block_h:
        assert head_kv % block_h == 0, "head_kv should be a multiple of block_h"
        head_repeats = head_kv // block_h
    else:
        head_repeats = 1

    heads_per_block = padded_head_kv if head_repeats == 1 else block_h
    pv_mma_n = 64 if block_h == 32 else 128
    v_tile_width = dim_qk // 4
    score_swizzle = block_i // 8

    finalize_left = make_scheduled_decode_finalize_left(
        h_per_block=heads_per_block,
        out_width=v_tile_width,
        num_heads=num_heads,
        accum_dtype=accum_dtype,
        has_attn_sink=False,
        use_strict_valid=True,
        add_denominator_epsilon=True,
        sink_only_unsplit=False,
        sink_invalid_zero=False,
        wait_before_final=True,
        l0_start=0,
        l1_start=v_tile_width,
        out_dtype=dtype,
        guard_invalid_heads=padded_head_kv != head_kv,
        support_split=support_split,
        lse_layout="BNH",
    )
    finalize_right = make_scheduled_decode_finalize_right(
        h_per_block=heads_per_block,
        out_width=v_tile_width,
        num_heads=num_heads,
        has_attn_sink=False,
        wait_after_scale=True,
        r0_start=dim_qk // 2,
        r1_start=dim_qk // 2 + v_tile_width,
        out_dtype=dtype,
        guard_invalid_heads=padded_head_kv != head_kv,
        support_split=support_split,
    )
    stage_value_shared = make_scheduled_decode_stage_value_shared(
        block_i=block_i,
        continuity=pv_mma_n,
    )

    @T.macro
    def capture_kv_half(kv_shared, kv_reg_fp8, thread_row, thread_col, half_index):
        for r in T.unroll(2):
            for group in T.unroll(2):
                reg_start = r * 32 + half_index * 16 + group * 8
                shared_col = 64 * group + thread_col * 8
                T.copy(
                    kv_shared[thread_row + r * 32, shared_col : shared_col + 8],
                    kv_reg_fp8[reg_start : reg_start + 8],
                )

    @T.macro
    def load_query(
        q,
        b_i,
        s_i,
        h0,
        h1,
        q_shared_l_0,
        q_shared_l_1,
        q_shared_r_0,
        q_shared_r_1,
        q_tail_shared,
        bar_q,
    ):
        T.annotate_layout(
            {
                q_shared_l_0[:, :]: tilelang.layout.make_sqmma_swizzled_layout(
                    q_shared_l_0[:, :], k_major=True
                ),
                q_shared_l_1[:, :]: tilelang.layout.make_sqmma_swizzled_layout(
                    q_shared_l_1[:, :], k_major=True
                ),
                q_shared_r_0[:, :]: tilelang.layout.make_sqmma_swizzled_layout(
                    q_shared_r_0[:, :], k_major=True
                ),
                q_shared_r_1[:, :]: tilelang.layout.make_sqmma_swizzled_layout(
                    q_shared_r_1[:, :], k_major=True
                ),
                q_tail_shared[:, :]: tilelang.layout.make_sqmma_swizzled_layout(
                    q_tail_shared[:, :], k_major=True
                ),
            },
            allow_reannotation=True,
            allow_buffer_region=True,
        )
        T.tma_copy(
            q[b_i, s_i, h0:h1, 0:qk_native_k],
            q_shared_l_0,
            barrier=bar_q,
        )
        T.tma_copy(
            q[b_i, s_i, h0:h1, qk_native_k : 2 * qk_native_k],
            q_shared_l_1,
            barrier=bar_q,
        )
        T.tma_copy(
            q[b_i, s_i, h0:h1, 2 * qk_native_k : 3 * qk_native_k],
            q_shared_r_0,
            barrier=bar_q,
        )
        T.tma_copy(
            q[b_i, s_i, h0:h1, 3 * qk_native_k : dim_qk],
            q_shared_r_1,
            barrier=bar_q,
        )
        T.tma_copy(
            q[b_i, s_i, h0:h1, dim_qk:],
            q_tail_shared,
            barrier=bar_q,
        )
        T.barrier_arrive(bar_q)

    load_indices = make_scheduled_decode_indices_loader(
        block_i=block_i,
        store_kv_indices=False,
    )
    glse_shape = [batch + num_mp_parts, seq_len, num_heads] if support_split else [1]
    output_partial_shape = (
        [batch + num_mp_parts, seq_len, num_heads, dim] if support_split else [1]
    )

    @T.macro
    def dsa_decode_split(
        q: T.Tensor([batch, seq_len, num_heads, dim + tail_dim], q_dtype),  # type: ignore
        kv: T.Tensor([seq_len_kv, kv_group, dim_bytes], kv_latent_dtype),  # type: ignore
        kv_packed_bf16: T.Tensor([seq_len_kv, kv_group, dim_bytes // 2], T.bfloat16),  # type: ignore
        indices: T.Tensor([batch, seq_len, kv_group, topk], indices_dtype),  # type: ignore
        seq_lens: T.Tensor([batch], T.int32),  # type: ignore
        unused_attn_sink: T.Tensor([num_heads], T.float32),  # type: ignore
        scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),  # type: ignore
        num_splits: T.Tensor([batch + 1], T.int32),  # type: ignore
        glse: T.Tensor(glse_shape, T.float32),  # type: ignore
        output_partial: T.Tensor(output_partial_shape, accum_dtype),  # type: ignore
        output: T.Tensor([batch, seq_len, num_heads, dim], dtype),  # type: ignore
        lse: T.Tensor([batch, seq_len, num_heads], T.float32),  # type: ignore
        bmm1_scale: T.float32,
        bmm2_scale: T.float32,
    ):
        with T.Kernel(
            seq_len * head_repeats, kv_group, num_mp_parts, threads=threads
        ) as (bx, by, bz):
            kv_shared_l_0 = T.alloc_shared([block_i, qk_native_k], kv_latent_dtype)
            kv_shared_l_1 = T.alloc_shared([block_i, qk_native_k], kv_latent_dtype)
            kv_shared_r_0 = T.alloc_shared([block_i, qk_native_k], kv_latent_dtype)
            kv_shared_r_1 = T.alloc_shared([block_i, qk_native_k], kv_latent_dtype)
            q_shared_l_0 = T.alloc_shared([heads_per_block, qk_native_k], q_dtype)
            q_shared_l_1 = T.alloc_shared([heads_per_block, qk_native_k], q_dtype)
            q_shared_r_0 = T.alloc_shared([heads_per_block, qk_native_k], q_dtype)
            q_shared_r_1 = T.alloc_shared([heads_per_block, qk_native_k], q_dtype)
            q_tail_shared = T.alloc_shared([heads_per_block, tail_dim_qk], q_dtype)
            k_tail_shared = T.alloc_shared([block_i, tail_dim_qk], kv_latent_dtype)
            v_shared_l_0 = T.alloc_shared([block_i, v_tile_width], dtype)
            v_shared_l_1 = T.alloc_shared([block_i, v_tile_width], dtype)
            v_shared_r_0 = T.alloc_shared([block_i, v_tile_width], dtype)
            v_shared_r_1 = T.alloc_shared([block_i, v_tile_width], dtype)
            scores_shared = T.alloc_shared([heads_per_block, block_i], dtype)
            sum_exp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            sink_scale_shared = T.alloc_shared([heads_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([block_i], "bool")
            preloaded_indices = T.alloc_shared(
                [2048 if preload_all_indices else 1], indices_dtype
            )
            preloaded_index_mask = T.alloc_shared(
                [2048 if preload_all_indices else 1], "bool"
            )
            bar_kv_mask_ready = T.alloc_barrier(arrive_count=128)
            bar_kv_mask_free = T.alloc_barrier(arrive_count=256)
            bar_q = T.alloc_barrier(arrive_count=128)
            bar_q_free = T.alloc_barrier(arrive_count=256)
            bar_indices_ready = T.alloc_barrier(arrive_count=128)
            bar_kv0_ready = T.alloc_barrier(arrive_count=128)
            bar_kv1_ready = T.alloc_barrier(arrive_count=128)
            bar_kv0_free = T.alloc_barrier(arrive_count=512)
            bar_kv1_qk_free = T.alloc_barrier(arrive_count=512)
            bar_vl_ready = T.alloc_barrier(arrive_count=256)
            bar_vr_ready = T.alloc_barrier(arrive_count=256)
            bar_p_ready = T.alloc_barrier(arrive_count=256)
            bar_p_free = T.alloc_barrier(arrive_count=512)
            bar_final = T.alloc_barrier(arrive_count=256)
            bar_producer_protect = T.alloc_barrier(arrive_count=128)
            bar_indices_preloaded = T.alloc_barrier(arrive_count=128)
            if preload_all_indices:
                bar_preloaded_row_free = T.alloc_barrier(arrive_count=256)
            bar_consumer0_protect = T.alloc_barrier(arrive_count=256)
            bar_consumer1_protect = T.alloc_barrier(arrive_count=256)
            kv_robust_desc = T.make_robust_desc(
                T.address_of(kv[0, 0, 0]),
                seq_len_kv * kv_group * dim_bytes,
            )
            T.sync_threads()

            begin_idx = T.alloc_var(T.int32)
            sched_begin_block_idx = T.alloc_var(T.int32)
            end_idx = T.alloc_var(T.int32)
            sched_end_block_idx = T.alloc_var(T.int32)
            begin_n_split_idx = T.alloc_var(T.int32)
            phase_count = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)
            begin_idx = scheduler_metadata[bz, 0]
            sched_begin_block_idx = scheduler_metadata[bz, 1]
            end_idx = scheduler_metadata[bz, 2]
            sched_end_block_idx = scheduler_metadata[bz, 3]
            begin_n_split_idx = scheduler_metadata[bz, 4]

            g_i = by
            s_i = bx if head_repeats == 1 else (bx // head_repeats)

            h0 = g_i * padded_head_kv + (
                0 if head_repeats == 1 else (bx % head_repeats) * 64
            )
            h1 = h0 + heads_per_block
            tid = T.get_thread_binding()
            runtime_sm_scale = T.alloc_var(T.float32)
            runtime_sm_scale = bmm1_scale * softmax_log2e
            fp8_dequant_scale = T.alloc_var(T.float32)
            fp8_dequant_scale = 1.0
            for b_i in range(begin_idx, end_idx + 1, 1):
                T.call_extern("void", "__musa_loop_transparent_outermost")
                start_block_idx = T.alloc_var(T.int32)
                end_block_idx = T.alloc_var(T.int32)
                n_split_idx = T.alloc_var(T.int32)
                runtime_seq_len = T.alloc_var(T.int32)
                dynamic_total_blocks = T.alloc_var(T.int32)
                runtime_seq_len = T.max(T.min(seq_lens[b_i], topk), 0)
                dynamic_total_blocks = T.max(T.ceildiv(runtime_seq_len, block_i), 1)
                start_block_idx = T.if_then_else(
                    b_i == begin_idx, sched_begin_block_idx, 0
                )
                end_block_idx = T.if_then_else(
                    b_i == end_idx, sched_end_block_idx, dynamic_total_blocks
                )
                n_split_idx = T.if_then_else(b_i == begin_idx, begin_n_split_idx, 0)
                is_unsplit = (num_splits[b_i + 1] - num_splits[b_i]) == 1
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
                        [heads_per_block, v_tile_width], accum_dtype
                    )
                    acc_o_l_1 = T.alloc_fragment(
                        [heads_per_block, v_tile_width], accum_dtype
                    )
                    kv_reg_l = T.alloc_local([64], dtype)
                    kv_reg_l_fp16 = T.view(kv_reg_l, [64], T.float16)
                    kv_reg_l_fp8 = T.alloc_local([64], kv_latent_dtype)
                    kv_mask_local = T.alloc_local([8], "bool")
                    consumer0_value_thread_col = tid % 8
                    consumer0_value_thread_row = tid // 8
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l_0, 0)
                    T.fill(acc_o_l_1, 0)
                    if preload_all_indices:
                        T.barrier_wait(
                            bar_indices_preloaded,
                            (b_i - begin_idx) & 1,
                        )
                    T.barrier_wait(bar_q, (b_i - begin_idx) & 1)
                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_arrive(bar_consumer0_protect)
                        # Initialize acc_s while the producer finishes the K0 copy.
                        if preload_all_indices:
                            for v in T.unroll(8):
                                mask_pos = i_i * block_i + (tid % 8) * 8 + v
                                mask_shared_pos = mask_pos ^ (
                                    ((mask_pos % block_i) // 32) * 4
                                )
                                kv_mask_local[v] = preloaded_index_mask[mask_shared_pos]
                            T.lma_wait()
                            if i_i + 1 == end_block_idx:
                                # From here on this row's mask lives in registers;
                                # Producer may prepare the next persistent row.
                                T.barrier_arrive(bar_preloaded_row_free)
                        else:
                            T.barrier_wait(bar_kv_mask_ready, (phase_count[0] & 1))
                            for v in T.unroll(8):
                                kv_mask_local[v] = is_kv_valid[(tid % 8) * 8 + v]
                            T.lma_wait()
                            T.barrier_arrive(bar_kv_mask_free)
                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                kv_mask_local[bi_i // 8], 0, -(2**30)
                            )

                        T.barrier_wait(bar_kv0_ready, (phase_count[0] & 1))
                        # Expose each native K=128 QK as its own scheduling node.
                        T.gemm(
                            q_shared_l_0[:, :],
                            kv_shared_l_0[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        # Use the first native-K flight for all K-L captures.
                        T.annotate_layout(
                            {
                                kv_shared_l_0[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_l_0[:, :], k_major=True
                                ),
                                kv_shared_l_1[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_l_1[:, :], k_major=True
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        capture_kv_half(
                            kv_shared_l_0,
                            kv_reg_l_fp8,
                            consumer0_value_thread_row,
                            consumer0_value_thread_col,
                            0,
                        )
                        capture_kv_half(
                            kv_shared_l_1,
                            kv_reg_l_fp8,
                            consumer0_value_thread_row,
                            consumer0_value_thread_col,
                            1,
                        )
                        T.gemm(
                            q_shared_l_1[:, :],
                            kv_shared_l_1[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.barrier_wait(bar_kv1_ready, (phase_count[0] & 1))
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv0_free)
                        T.gemm(
                            q_shared_r_0[:, :],
                            kv_shared_r_0[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.lma_wait()
                        T.copy(kv_reg_l_fp8[0:32], kv_reg_l_fp16[0:32])
                        T.gemm(
                            q_shared_r_1[:, :],
                            kv_shared_r_1[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.copy(kv_reg_l_fp8[32:64], kv_reg_l_fp16[32:64])
                        for idx in T.vectorized(32):
                            kv_reg_l[idx] = T.Cast(
                                T.bfloat16,
                                kv_reg_l_fp16[idx] * fp8_dequant_scale,
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
                        for idx in T.vectorized(32):
                            kv_reg_l[idx + 32] = T.Cast(
                                T.bfloat16,
                                kv_reg_l_fp16[idx + 32] * fp8_dequant_scale,
                            )
                        stage_value_shared(
                            v_shared_l_0,
                            kv_reg_l,
                            consumer0_value_thread_row,
                            consumer0_value_thread_col,
                            0,
                        )
                        stage_value_shared(
                            v_shared_l_1,
                            kv_reg_l,
                            consumer0_value_thread_row,
                            consumer0_value_thread_col,
                            2,
                        )
                        T.copy(m_i, m_i_prev)
                        # P-free only protects the next scores_shared write. Wait
                        # while QK-R/tail is still in TCE flight.
                        T.barrier_wait(bar_p_free, (phase_count[0] & 1) ^ 1)
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv1_qk_free)

                        # Online softmax update and score staging for both PV consumers.
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(heads_per_block):
                            m_i[h_i] = T.max(m_i_prev[h_i], m_i[h_i])
                            alpha_local[h_i] = T.exp2(
                                (m_i_prev[h_i] - m_i[h_i]) * runtime_sm_scale
                            )
                        T.lma_wait()
                        T.barrier_arrive(bar_vl_ready)
                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.exp2(
                                (acc_s[h_i, bi_i] - m_i[h_i]) * runtime_sm_scale
                            )
                        for h_i, d_i in T.Parallel(heads_per_block, v_tile_width):
                            acc_o_l_0[h_i, d_i] *= alpha_local[h_i]

                        T.copy(alpha_local, alpha_shared)
                        T.copy(acc_s, acc_s_cast)
                        for h_i, t in T.Parallel(heads_per_block, 8):
                            base = t * score_swizzle
                            for lane in T.vectorized(score_swizzle):
                                scores_shared[h_i, base + lane] = acc_s_cast[
                                    h_i, lane * 8 + t
                                ]
                        T.lma_wait()
                        T.barrier_arrive(bar_p_ready)
                        T.barrier_wait(bar_vl_ready, (phase_count[0] & 1))

                        # Keep the two 128-wide V slices independently schedulable.
                        T.gemm(
                            scores_shared,
                            v_shared_l_0,
                            acc_o_l_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        # Preserve PV0 as the producer for the following scalar
                        # window; otherwise mtcc hoists the PV1 rescale above it.
                        T.sched_boundary()
                        # The second accumulator is independent of the first
                        # PV batch.  Scale it in that TCE flight instead of
                        # extending the zero-slack P-ready publication path.
                        for h_i, d_i in T.Parallel(heads_per_block, v_tile_width):
                            acc_o_l_1[h_i, d_i] *= alpha_local[h_i]
                        T.gemm(
                            scores_shared,
                            v_shared_l_1,
                            acc_o_l_1,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        # Reduce the score row while TCE consumes scores_shared.
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(heads_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        T.barrier_wait(
                            bar_consumer0_protect,
                            phase_count[0] & 1,
                        )
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_p_free)
                        phase_count[0] = phase_count[0] ^ 1
                    T.barrier_arrive(bar_q_free)
                    for h_i, d_i in T.Parallel(heads_per_block, v_tile_width):
                        acc_o_l_0[h_i, d_i] *= bmm2_scale
                        acc_o_l_1[h_i, d_i] *= bmm2_scale
                    for h_i in T.Parallel(heads_per_block):
                        # Keep the exact invalid sentinel through the runtime
                        # QK scale so finalize_left can emit +inf LSE.
                        m_i[h_i] = T.if_then_else(
                            m_i[h_i] == -(2**30),
                            -(2**30),
                            m_i[h_i] * bmm1_scale,
                        )
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
                        unused_attn_sink,
                        bar_final,
                        softmax_log2e,
                    )
                elif tid >= 256 and tid < 512:
                    acc_o_r_0 = T.alloc_fragment(
                        [heads_per_block, v_tile_width], accum_dtype
                    )
                    acc_o_r_1 = T.alloc_fragment(
                        [heads_per_block, v_tile_width], accum_dtype
                    )
                    kv_reg_r = T.alloc_local([64], dtype)
                    kv_reg_r_fp16 = T.view(kv_reg_r, [64], T.float16)
                    kv_reg_r_fp8 = T.alloc_local([64], kv_latent_dtype)
                    consumer1_value_thread_col = (tid - 256) % 8
                    consumer1_value_thread_row = (tid - 256) // 8
                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)
                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_arrive(bar_consumer1_protect)
                        # Keep Consumer1's generation-matched K0-free arrival.
                        T.barrier_wait(bar_kv0_ready, phase_count[0] & 1)
                        T.barrier_arrive(bar_kv0_free)

                        T.barrier_wait(bar_kv1_ready, phase_count[0] & 1)
                        # Capture the complete V-R input independently.
                        T.annotate_layout(
                            {
                                kv_shared_r_0[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r_0[:, :], k_major=True
                                ),
                                kv_shared_r_1[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r_1[:, :], k_major=True
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        capture_kv_half(
                            kv_shared_r_0,
                            kv_reg_r_fp8,
                            consumer1_value_thread_row,
                            consumer1_value_thread_col,
                            0,
                        )
                        capture_kv_half(
                            kv_shared_r_1,
                            kv_reg_r_fp8,
                            consumer1_value_thread_row,
                            consumer1_value_thread_col,
                            1,
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_kv1_qk_free)

                        T.copy(
                            kv_reg_r_fp8[0:consumer1_vr_overlap_values],
                            kv_reg_r_fp16[0:consumer1_vr_overlap_values],
                        )
                        T.copy(
                            kv_reg_r_fp8[consumer1_vr_overlap_values:64],
                            kv_reg_r_fp16[consumer1_vr_overlap_values:64],
                        )
                        for idx in T.vectorized(64):
                            kv_reg_r[idx] = T.Cast(
                                T.bfloat16,
                                kv_reg_r_fp16[idx] * fp8_dequant_scale,
                            )

                        stage_value_shared(
                            v_shared_r_0,
                            kv_reg_r,
                            consumer1_value_thread_row,
                            consumer1_value_thread_col,
                            0,
                        )
                        stage_value_shared(
                            v_shared_r_1,
                            kv_reg_r,
                            consumer1_value_thread_row,
                            consumer1_value_thread_col,
                            2,
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vr_ready)

                        T.barrier_wait(bar_p_ready, phase_count[0] & 1)
                        for h_i, d_i in T.Parallel(heads_per_block, v_tile_width):
                            acc_o_r_0[h_i, d_i] *= alpha_shared[h_i]
                            acc_o_r_1[h_i, d_i] *= alpha_shared[h_i]

                        T.barrier_wait(bar_vr_ready, phase_count[0] & 1)
                        T.gemm(
                            scores_shared,
                            v_shared_r_0,
                            acc_o_r_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.gemm(
                            scores_shared,
                            v_shared_r_1,
                            acc_o_r_1,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.barrier_wait(
                            bar_consumer1_protect,
                            phase_count[0] & 1,
                        )
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_p_free)
                        phase_count[0] = phase_count[0] ^ 1
                    for h_i, d_i in T.Parallel(heads_per_block, v_tile_width):
                        acc_o_r_0[h_i, d_i] *= bmm2_scale
                        acc_o_r_1[h_i, d_i] *= bmm2_scale
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
                    producer_thread_col = (tid - 512) % 8
                    producer_thread_row = (tid - 512) // 8
                    if preload_all_indices:
                        # Only Consumer0 reads the shared row. Wait for its prior-row
                        # release before Producer overwrites the single buffer.
                        if b_i != begin_idx:
                            T.barrier_wait(
                                bar_preloaded_row_free,
                                ((b_i - begin_idx) & 1) ^ 1,
                            )
                        producer_tid = tid - 512
                        for preload_i in T.serial((topk + 511) // 512):
                            token_base = preload_i * 512 + producer_tid * 4
                            shared_base = token_base ^ (
                                ((token_base % block_i) // 32) * 4
                            )
                            for v in T.vectorized(4):
                                token_pos = token_base + v
                                if token_pos < topk:
                                    # XOR bit 2 in the upper half of each 64-entry
                                    # tile. Producer and consumer shared reads then
                                    # hit eight distinct banks (plus broadcasts).
                                    preloaded_indices[shared_base + v] = indices[
                                        b_i,
                                        s_i,
                                        g_i,
                                        token_pos,
                                    ]
                        # Let the index loads make progress while waiting to reuse
                        # and refill the independent single-buffered Q row.
                        if b_i != begin_idx:
                            T.barrier_wait(
                                bar_q_free,
                                ((b_i - begin_idx) & 1) ^ 1,
                            )
                        load_query(
                            q,
                            b_i,
                            s_i,
                            h0,
                            h1,
                            q_shared_l_0,
                            q_shared_l_1,
                            q_shared_r_0,
                            q_shared_r_1,
                            q_tail_shared,
                            bar_q,
                        )
                        T.lma_wait()
                        # Normalize the partial tail once so hot-loop validity is
                        # represented solely by the -1 sentinel in shared memory.
                        for preload_i in T.serial((topk + 511) // 512):
                            token_base = preload_i * 512 + producer_tid * 4
                            shared_base = token_base ^ (
                                ((token_base % block_i) // 32) * 4
                            )
                            for v in T.vectorized(4):
                                token_pos = token_base + v
                                if token_pos < topk:
                                    if token_pos < runtime_seq_len:
                                        preloaded_index_mask[shared_base + v] = (
                                            preloaded_indices[shared_base + v] >= 0
                                            and preloaded_indices[shared_base + v]
                                            < seq_len_kv
                                        )
                                    else:
                                        preloaded_indices[shared_base + v] = -1
                                        preloaded_index_mask[shared_base + v] = False
                        T.lma_wait()
                        T.barrier_arrive(bar_indices_preloaded)
                        T.barrier_wait(bar_indices_preloaded, (b_i - begin_idx) & 1)
                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_arrive(bar_producer_protect)
                        if preload_all_indices:
                            for r in T.unroll(4):
                                token_pos = (
                                    i_i * block_i
                                    + ((r * 16 + producer_thread_row) % 8)
                                    * (block_i // 8)
                                    + (r * 16 + producer_thread_row) // 8
                                )
                                shared_pos = token_pos ^ (
                                    ((token_pos % block_i) // 32) * 4
                                )
                                kperm_indices_local[r] = preloaded_indices[shared_pos]
                                kperm_mask_local[r] = preloaded_index_mask[shared_pos]
                                kperm_indices_local[r] = T.if_then_else(
                                    kperm_mask_local[r],
                                    kperm_indices_local[r],
                                    seq_len_kv,
                                )
                        else:
                            load_indices(
                                indices,
                                b_i,
                                s_i,
                                g_i,
                                i_i,
                                runtime_seq_len,
                                seq_len_kv,
                                producer_thread_row,
                                producer_thread_col,
                                phase_count[0],
                                kperm_indices_local,
                                kperm_mask_local,
                                is_kv_valid,
                                None,
                                bar_kv_mask_free,
                            )
                        T.barrier_wait(
                            bar_kv0_free,
                            (phase_count[0] & 1) ^ 1,
                        )
                        T.annotate_layout(
                            {
                                kv_shared_l_0[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_l_0[:, :], k_major=True
                                ),
                                kv_shared_l_1[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_l_1[:, :], k_major=True
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(2):
                            row = r * 16 + producer_thread_row
                            for u in T.unroll(2):
                                col_base = 64 * u + producer_thread_col * 8
                                for lane in T.vectorized(8):
                                    col = col_base + lane
                                    T.copy(
                                        kv[kperm_indices_local[r], g_i, col],
                                        kv_shared_l_0[row, col],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )
                            for u in T.unroll(2):
                                col_base = 64 * u + producer_thread_col * 8
                                for lane in T.vectorized(8):
                                    col = col_base + lane
                                    T.copy(
                                        kv[
                                            kperm_indices_local[r],
                                            g_i,
                                            qk_native_k + col,
                                        ],
                                        kv_shared_l_1[row, col],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )
                        T.ptx_commit_group()
                        for r_local in T.unroll(2):
                            r = r_local + 2
                            row = r * 16 + producer_thread_row
                            for u in T.unroll(2):
                                col_base = 64 * u + producer_thread_col * 8
                                for lane in T.vectorized(8):
                                    col = col_base + lane
                                    T.copy(
                                        kv[kperm_indices_local[r], g_i, col],
                                        kv_shared_l_0[row, col],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )
                            for u in T.unroll(2):
                                col_base = 64 * u + producer_thread_col * 8
                                for lane in T.vectorized(8):
                                    col = col_base + lane
                                    T.copy(
                                        kv[
                                            kperm_indices_local[r],
                                            g_i,
                                            qk_native_k + col,
                                        ],
                                        kv_shared_l_1[row, col],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )

                        if not preload_all_indices:
                            T.lma_wait()
                            T.barrier_arrive(bar_kv_mask_ready)
                            T.barrier_arrive(bar_indices_ready)
                            T.barrier_wait(
                                bar_indices_ready,
                                phase_count[0] & 1,
                            )
                        T.ptx_commit_group()

                        T.barrier_wait(
                            bar_kv1_qk_free,
                            (phase_count[0] & 1) ^ 1,
                        )

                        # Preload two K-R quarters while K-L remains in flight.
                        T.annotate_layout(
                            {
                                kv_shared_r_0[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r_0[:, :], k_major=True
                                ),
                                kv_shared_r_1[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r_1[:, :], k_major=True
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(2):
                                row = r * 16 + producer_thread_row
                                col_base = 64 * u + producer_thread_col * 8
                                for lane in T.vectorized(8):
                                    col = col_base + lane
                                    T.copy(
                                        kv[
                                            kperm_indices_local[r],
                                            g_i,
                                            dim_qk // 2 + col,
                                        ],
                                        kv_shared_r_0[row, col],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )
                        T.ptx_commit_group()
                        if not preload_all_indices and i_i == start_block_idx:
                            if b_i != begin_idx:
                                T.barrier_wait(
                                    bar_q_free,
                                    ((b_i - begin_idx) & 1) ^ 1,
                                )
                            load_query(
                                q,
                                b_i,
                                s_i,
                                h0,
                                h1,
                                q_shared_l_0,
                                q_shared_l_1,
                                q_shared_r_0,
                                q_shared_r_1,
                                q_tail_shared,
                                bar_q,
                            )
                        T.ptx_wait_group(1)
                        T.barrier_arrive(bar_kv0_ready)

                        # Load the rope tail after K-L is visible.
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
                            row = r * 16 + producer_thread_row
                            col_base = producer_thread_col * 8
                            for lane in T.vectorized(8):
                                col = col_base + lane
                                T.copy(
                                    kv[
                                        kperm_indices_local[r],
                                        g_i,
                                        dim_qk + col,
                                    ],
                                    k_tail_shared[row, col],
                                    force_async_copy=True,
                                    src_robust_desc=kv_robust_desc,
                                )

                        # Complete the remaining two K-R quarters.
                        for r in T.unroll(4):
                            for u in T.unroll(2):
                                row = r * 16 + producer_thread_row
                                col_base = 64 * u + producer_thread_col * 8
                                for lane in T.vectorized(8):
                                    col = col_base + lane
                                    T.copy(
                                        kv[
                                            kperm_indices_local[r],
                                            g_i,
                                            dim_qk // 2 + qk_native_k + col,
                                        ],
                                        kv_shared_r_1[row, col],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.barrier_arrive(bar_kv1_ready)
                        T.barrier_wait(
                            bar_producer_protect,
                            phase_count[0] & 1,
                        )
                        phase_count[0] = phase_count[0] ^ 1

    dsa_combine = make_scheduled_decode_combine(
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        dim=dim,
        num_mp_parts=num_mp_parts,
        dtype=dtype,
        accum_dtype=accum_dtype,
        max_nums_splits=max_num_splits,
        has_attn_sink=False,
        max_lse_init=-(2**30) * softmax_log2e,
        launch_by_scheduler=launch_combine_by_scheduler,
        lse_layout="BNH",
    )

    @T.prim_func
    def dsa_decode(
        q: T.Tensor([batch, seq_len, num_heads, dim + tail_dim], q_dtype),  # type: ignore
        kv: T.Tensor([seq_len_kv, kv_group, dim_bytes], kv_latent_dtype),  # type: ignore
        kv_packed_bf16: T.Tensor([seq_len_kv, kv_group, dim_bytes // 2], T.bfloat16),  # type: ignore
        indices: T.Tensor([batch, seq_len, kv_group, topk], indices_dtype),  # type: ignore
        seq_lens: T.Tensor([batch], T.int32),  # type: ignore
        unused_attn_sink: T.Tensor([num_heads], T.float32),  # type: ignore
        scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),  # type: ignore
        num_splits: T.Tensor([batch + 1], T.int32),  # type: ignore
        glse: T.Tensor(glse_shape, accum_dtype),  # type: ignore
        output_partial: T.Tensor(output_partial_shape, accum_dtype),  # type: ignore
        output: T.Tensor([batch, seq_len, num_heads, dim], dtype),  # type: ignore
        lse: T.Tensor([batch, seq_len, num_heads], accum_dtype),  # type: ignore
        bmm1_scale: T.float32,
        bmm2_scale: T.float32,
    ):
        dsa_decode_split(
            q,
            kv,
            kv_packed_bf16,
            indices,
            seq_lens,
            unused_attn_sink,
            scheduler_metadata,
            num_splits,
            glse,
            output_partial,
            output,
            lse,
            bmm1_scale,
            bmm2_scale,
        )
        if support_split:
            dsa_combine(
                scheduler_metadata,
                num_splits,
                glse,
                output_partial,
                unused_attn_sink,
                output,
                lse,
            )

    return dsa_decode


def select_sparse_mla_fp8_max_num_splits(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    kv_group: int,
    topk: int,
    device,
    block_h: int = 64,
    block_i: int = 64,
) -> int:
    """Bound per-sequence partials created by the runtime block scheduler."""
    max_blocks_per_batch = max((topk + block_i - 1) // block_i, 1)
    num_scheduler_ctas = select_sparse_mla_fp8_num_scheduler_ctas(
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        kv_group=kv_group,
        device=device,
        topk=topk,
        block_h=block_h,
        block_i=block_i,
    )
    if num_scheduler_ctas == 1 or max_blocks_per_batch == 1:
        return 1
    return min(max_blocks_per_batch, num_scheduler_ctas, 64)


def select_sparse_mla_fp8_num_scheduler_ctas(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    kv_group: int,
    device,
    topk: int,
    block_h: int = 64,
    block_i: int = 64,
) -> int:
    """Match the CTA count selected by the FlashMLA metadata dispatch."""
    head_kv = num_heads // kv_group
    head_repeats = head_kv // block_h if head_kv > block_h else 1
    ctas_per_scheduler_part = max(seq_len * kv_group * head_repeats, 1)
    scheduler_capacity = max(get_physical_num_mps(device) // ctas_per_scheduler_part, 1)
    return scheduler_capacity


def sparse_mla_fp8_decode_interface(
    q,
    kv,
    indices,
    seq_lens,
    scheduler_metadata,
    num_splits,
    bmm1_scale,
    bmm2_scale,
    return_p_sum: bool = False,
    d_v=512,
    threads=640,
    verbose=False,
    max_num_splits: int | None = None,
    num_scheduler_ctas: int | None = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
):
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.dtype == torch.float8_e4m3fn, "q must be float8_e4m3fn"
    assert kv.dtype == torch.uint8, "kv must be uint8"
    assert indices.dtype == torch.int32, "indices must be int32"
    assert seq_lens.dtype == torch.int32, "seq_lens must be int32"
    assert scheduler_metadata.dtype == torch.int32
    assert num_splits.dtype == torch.int32
    assert all(
        tensor.is_contiguous()
        for tensor in (q, kv, indices, seq_lens, scheduler_metadata, num_splits)
    )
    b, seq_len, heads, dim_plus_tail_dim = q.shape
    seq_len_kv, kv_group, _ = kv.shape
    assert dim_plus_tail_dim == 576
    dim = d_v
    dim_bytes = 576
    assert kv.shape[-1] == dim_bytes
    tail_dim = dim_plus_tail_dim - dim
    _, _, _, topk = indices.shape
    assert indices.shape == (b, seq_len, kv_group, topk)
    assert topk % 64 == 0, "topk must be a multiple of 64"
    assert seq_lens.shape == (b,)
    if max_num_splits is None:
        max_num_splits = select_sparse_mla_fp8_max_num_splits(
            batch=b,
            seq_len=seq_len,
            num_heads=heads,
            kv_group=kv_group,
            topk=topk,
            device=q.device,
        )
    if num_scheduler_ctas is None:
        num_scheduler_ctas = int(scheduler_metadata.shape[0])
    assert scheduler_metadata.shape == (num_scheduler_ctas, 8)
    assert num_splits.shape == (b + 1,)
    assert 1 <= max_num_splits <= 64
    assert num_scheduler_ctas >= 1
    unused_attn_sink = torch.empty((heads,), dtype=torch.float32, device=q.device)
    if max_num_splits == 1:
        glse = torch.empty((1,), dtype=torch.float32, device=q.device)
        out_partial = torch.empty((1,), dtype=torch.float32, device=q.device)
    else:
        partial_rows = b + num_scheduler_ctas
        glse = torch.empty(
            (partial_rows, seq_len, heads), dtype=torch.float32, device=q.device
        )
        out_partial = torch.empty(
            (partial_rows, seq_len, heads, d_v),
            dtype=torch.float32,
            device=q.device,
        )
    if out is None:
        out = torch.empty(
            (b, seq_len, heads, d_v), dtype=torch.bfloat16, device=q.device
        )
    else:
        assert out.shape == (b, seq_len, heads, d_v)
        assert out.dtype == torch.bfloat16
        assert out.device == q.device
        assert out.is_contiguous()
    if lse is None:
        lse = torch.empty((b, seq_len, heads), dtype=torch.float32, device=q.device)
    else:
        assert lse.shape == (b, seq_len, heads)
        assert lse.dtype == torch.float32
        assert lse.device == q.device
        assert lse.is_contiguous()
    threads = 640
    kernel = sparse_attention_fwd_kernel(
        heads,
        dim,
        tail_dim,
        kv_group=kv_group,
        threads=threads,
        max_num_splits=1 if max_num_splits == 1 else 64,
        launch_combine_by_scheduler=b > 2 * num_scheduler_ctas,
        preload_all_indices=topk <= 2048,
    )
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()
    kv_latent_f8 = kv.view(torch.float8_e4m3fn)
    kv_packed_bf16 = kv.view(torch.bfloat16)
    kernel(
        q,
        kv_latent_f8,
        kv_packed_bf16,
        indices,
        seq_lens,
        unused_attn_sink,
        scheduler_metadata,
        num_splits,
        glse,
        out_partial,
        out,
        lse,
        float(bmm1_scale),
        float(bmm2_scale),
    )
    return out, lse
