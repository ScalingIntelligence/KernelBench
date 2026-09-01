# ruff: noqa
"""
MODEL1 sequence-pack causal prefill kernel for local H8/H16/H32.

This kernel handles the main-KV prefill fast path used by DeepSeek V4
SWA-only and C128 layers. It packs local heads across adjacent query tokens
so the existing BM64 attention layout can be reused:
- H8  uses token_pack=8.
- H16 uses token_pack=4.
- H32 uses token_pack=2.

C4/topk sparse prefill and decode extra-KV paths intentionally stay on their
separate kernels because their index streams are not contiguous sequence packs.
"""

import torch
import tilelang
from tilelang import language as T
from tvm import tir

from ...utils import cosize
from .sparse_mla_prefill_common import (
    SPARSE_PREFILL_COMPILE_FLAGS,
    SPARSE_PREFILL_PASS_CONFIGS,
    validate_prefill_attn_sink,
    validate_token_lengths,
)
from .sparse_mla_index_type import jit_for_tensor_addressing
from ...execution_context import raise_complete_if_dry_run


# tilelang.disable_cache()


@tilelang.jit(
    out_idx=[3, 4, 5],
    pass_configs=SPARSE_PREFILL_PASS_CONFIGS,
    compile_flags=SPARSE_PREFILL_COMPILE_FLAGS,
)
def sparse_attention_fwd_kernel_model1_pack(
    num_heads,
    dim,
    topk,
    token_pack,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_i=64,
    threads=640,
    has_attn_sink=False,
    has_topk_length=False,
    has_row_mask=False,
    causal_window=0,
    compressed_kv_len=0,
    compress_ratio=1,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    assert topk % block_i == 0, "topk must be a multiple of block_i"
    if sm_scale is None:
        logits_scale = (1.0 / dim) ** 0.5
    else:
        logits_scale = sm_scale
    sm_scale = logits_scale * 1.44269504
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    assert token_pack in (2, 4, 8), "token_pack must be 2, 4, or 8"
    head_kv = num_heads // kv_group
    local_heads = num_heads // token_pack
    q_shape = [seq_len, num_heads, dim]
    kv_shape = [seq_len_kv, kv_group, dim]
    o_shape = [seq_len, num_heads, dim]
    lse_shape = [seq_len, num_heads]
    max_logits_shape = [seq_len, num_heads]
    indices_shape = [seq_len, kv_group, topk]
    row_masks_shape = indices_shape
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    dtype_bytes = 2
    q_cosize = cosize(q_shape)
    kv_cosize = cosize(kv_shape)
    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), 64)
    if padded_head_kv != head_kv:
        assert kv_group == 1
    num_i_orig = tilelang.cdiv(topk, block_i)
    dim_qk = dim
    lanes_per_vec = block_i // 8

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        head_repeats = head_kv // 64
    else:
        head_repeats = 1

    heads_per_block = padded_head_kv if head_repeats == 1 else 64
    has_dynamic_lengths = has_topk_length

    @T.macro
    def dsa_prefill_body(
        q,
        kv,
        indices,
        topk_length,
        row_masks,
        attn_sink,
        output,
        max_logits_out,
        lse,
    ):
        with T.Kernel(seq_len * head_repeats, kv_group, threads=threads) as (bx, by):
            q_shared_l = T.alloc_shared([heads_per_block, dim_qk // 2], dtype)
            q_shared_r = T.alloc_shared([heads_per_block, dim_qk // 2], dtype)
            kv_shared_l = T.alloc_shared([block_i, dim_qk // 2], dtype)
            kv_shared_r = T.alloc_shared([block_i, dim_qk // 2], dtype)
            v_shared_0 = T.alloc_shared([block_i, dim_qk // 4], dtype)
            v_shared_1 = T.alloc_shared([block_i, dim_qk // 4], dtype)

            scores_shared = T.alloc_shared([heads_per_block, block_i], dtype)
            sum_exp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            lse_shared = T.alloc_shared([heads_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([block_i], "bool", scope="shared")
            kv_indices = T.alloc_shared([block_i], indices_dtype, scope="shared")
            row_valid_shared = T.alloc_shared(
                [block_i, token_pack], "bool", scope="shared"
            )

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
                T.address_of(q[0, 0, 0]),
                q_cosize * dtype_bytes,
            )
            kv_robust_desc = T.make_robust_desc(
                T.address_of(kv[0, 0, 0]),
                kv_cosize * dtype_bytes,
            )
            T.sync_threads()

            g_i = by
            s_i = bx if head_repeats == 1 else (bx // head_repeats)
            h0 = g_i * padded_head_kv + (
                0 if head_repeats == 1 else (bx % head_repeats) * 64
            )
            h1 = h0 + heads_per_block
            tid = T.get_thread_binding()
            if has_dynamic_lengths:
                topk_len = T.alloc_var(T.int32)
                main_num_i = T.alloc_var(T.int32)
                active_num_i = T.alloc_var(T.int32)
                if has_topk_length:
                    topk_len = T.min(T.max(topk_length[s_i], 0), topk)
                else:
                    topk_len = topk
                main_num_i = T.ceildiv(topk_len, block_i)
                active_num_i = main_num_i

            if tid < 512:
                T.copy(
                    q[s_i, h0:h1, 0 : dim_qk // 2],
                    q_shared_l,
                    force_async_copy=True,
                    src_robust_desc=q_robust_desc,
                )
                T.copy(
                    q[s_i, h0:h1, dim_qk // 2 : dim_qk],
                    q_shared_r,
                    force_async_copy=True,
                    src_robust_desc=q_robust_desc,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                T.barrier_arrive(bar_q)
                T.barrier_wait(bar_q, 0)

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
                row_block_valid = T.alloc_fragment([heads_per_block], "bool")
                row_any_valid = T.alloc_fragment([heads_per_block], "bool")
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
                T.fill(row_any_valid, False)
                has_any_valid = T.alloc_var("bool")
                has_any_valid = False

                for i_i in range(active_num_i if has_dynamic_lengths else num_i_orig):
                    T.barrier_wait(bar_kv0_ready, (i_i & 1))
                    block_has_valid = T.alloc_var("bool")
                    block_has_valid = False
                    for valid_i in range(block_i):
                        block_has_valid = block_has_valid or is_kv_valid[valid_i]
                    has_any_valid = has_any_valid or block_has_valid

                    for h_i in T.Parallel(heads_per_block):
                        row_block_valid[h_i] = False
                    for h_i in T.Parallel(heads_per_block):
                        for valid_i in range(block_i):
                            row_block_valid[h_i] = (
                                row_block_valid[h_i]
                                or row_valid_shared[valid_i, h_i // local_heads]
                            )
                    for h_i in T.Parallel(heads_per_block):
                        row_any_valid[h_i] = row_any_valid[h_i] or row_block_valid[h_i]

                    for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                        kv_perm_pos = bi_i % 8 * 8 + bi_i // 8
                        row_valid = T.alloc_var("bool")
                        row_valid = row_valid_shared[kv_perm_pos, h_i // local_heads]
                        acc_s[h_i, bi_i] = T.if_then_else(
                            row_valid,
                            0,
                            -(2**30),
                        )

                    T.annotate_layout(
                        {
                            kv_shared_l[
                                :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                kv_shared_l[:, :],
                                k_major=True,
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

                    for r in T.unroll(2):
                        for u in T.unroll(4):
                            for v in T.vectorized(8):
                                kv_reg_l[r * 32 + u * 8 + v] = kv_shared_l[
                                    ((consumer0_ldg_ty + r * 32) % 8) * (block_i // 8)
                                    + (consumer0_ldg_ty + r * 32) // 8,
                                    64 * u + consumer0_ldg_tx * 8 + v,
                                ]
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.lma_wait()
                    T.barrier_arrive(bar_kv0_free)

                    T.barrier_wait(bar_kv1_ready, (i_i & 1))
                    T.annotate_layout(
                        {
                            kv_shared_r[
                                :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                kv_shared_r[:, :],
                                k_major=True,
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
                    T.copy(m_i, m_i_prev)
                    T.warpgroup_wait(0)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(heads_per_block):
                        m_i[h_i] = T.if_then_else(
                            row_block_valid[h_i],
                            T.max(m_i_prev[h_i], m_i[h_i]),
                            m_i_prev[h_i],
                        )
                    for h_i in T.Parallel(heads_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            row_block_valid[h_i],
                            T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale),
                            0,
                        )

                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(heads_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_l_0[h_i, d_i] *= alpha_local[h_i]
                        acc_o_l_1[h_i, d_i] *= alpha_local[h_i]

                    T.copy(alpha_local, alpha_shared)
                    T.copy(acc_s, acc_s_cast)
                    for i, t in T.Parallel(heads_per_block, 8):
                        base = t * lanes_per_vec
                        for l in T.vectorized(lanes_per_vec):
                            scores_shared[i, base + l] = acc_s_cast[i, l * 8 + t]

                    T.lma_wait()
                    T.barrier_arrive(bar_p_ready)

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
                                    r * 32 + consumer0_ldg_ty,
                                    64 * u + consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[r * 32 + u * 8 + v]
                    T.lma_wait()
                    T.barrier_arrive(bar_vl0_ready)
                    T.barrier_wait(bar_vl0_ready, (i_i & 1))

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
                                    r * 32 + consumer0_ldg_ty,
                                    64 * u + consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[r * 32 + (u + 2) * 8 + v]

                    T.warpgroup_wait(0)
                    T.barrier_arrive(bar_vl0_free)

                    T.lma_wait()
                    T.barrier_arrive(bar_vl1_ready)
                    T.barrier_wait(bar_vl1_ready, (i_i & 1))

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

                for h_i in T.Parallel(heads_per_block):
                    if sumexp[h_i] > 0 and row_any_valid[h_i]:
                        sumexp_inv[h_i] = 1 / sumexp[h_i]
                        max_logits[h_i] = m_i[h_i] * logits_scale
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
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

                if has_attn_sink:
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        if sumexp_inv[h_i] > 0:
                            sink_scale = 1 / (
                                1
                                + T.exp2(
                                    attn_sink[h0 + h_i] * 1.4426950408889634
                                    - sumexp[h_i]
                                )
                            )
                            acc_o_l_0[h_i, d_i] *= sink_scale
                            acc_o_l_1[h_i, d_i] *= sink_scale

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
                T.fill(acc_o_r_0, 0)
                T.fill(acc_o_r_1, 0)

                consumer1_ldg_tx = (tid - 256) % 8
                consumer1_ldg_ty = (tid - 256) // 8

                for i_i in range(active_num_i if has_dynamic_lengths else num_i_orig):
                    T.barrier_wait(bar_kv1_read_ready, (i_i & 1))
                    for r in T.unroll(2):
                        for u in T.unroll(4):
                            for v in T.vectorized(8):
                                kv_reg_r[r * 32 + u * 8 + v] = kv_shared_r[
                                    ((consumer1_ldg_ty + r * 32) % 8) * (block_i // 8)
                                    + (consumer1_ldg_ty + r * 32) // 8,
                                    64 * u + consumer1_ldg_tx * 8 + v,
                                ]

                    T.lma_wait()
                    T.barrier_arrive(bar_kv1_free)
                    T.barrier_wait(bar_vl0_free, (i_i & 1))
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
                    T.barrier_arrive(bar_vr0_ready)
                    T.barrier_wait(bar_vr0_ready, (i_i & 1))

                    T.barrier_wait(bar_p_ready, (i_i & 1))
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_r_0[h_i, d_i] *= alpha_shared[h_i]
                        acc_o_r_1[h_i, d_i] *= alpha_shared[h_i]

                    T.gemm(
                        scores_shared,
                        v_shared_0,
                        acc_o_r_0,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1,
                    )
                    T.wait_wgmma(0)

                    T.barrier_wait(bar_vl1_free, (i_i & 1))
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
                    T.barrier_wait(bar_vr1_ready, (i_i & 1))

                    T.gemm(
                        scores_shared,
                        v_shared_1,
                        acc_o_r_1,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1,
                    )
                    T.wait_wgmma(0)

                T.barrier_wait(bar_final, 0)
                for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                    acc_o_r_0[h_i, d_i] *= sum_exp_inv_shared[h_i]
                    acc_o_r_1[h_i, d_i] *= sum_exp_inv_shared[h_i]

                if has_attn_sink:
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        if sum_exp_inv_shared[h_i] > 0:
                            sink_scale = 1 / (
                                1
                                + T.exp2(
                                    attn_sink[h0 + h_i] * 1.4426950408889634
                                    - lse_shared[h_i]
                                )
                            )
                            acc_o_r_0[h_i, d_i] *= sink_scale
                            acc_o_r_1[h_i, d_i] *= sink_scale

                T.copy(
                    acc_o_r_0,
                    output[s_i, h0:h1, dim_qk // 2 : dim_qk // 2 + dim_qk // 4],
                )
                T.copy(
                    acc_o_r_1, output[s_i, h0:h1, dim_qk // 2 + dim_qk // 4 : dim_qk]
                )
            else:
                kperm_mask_local = T.alloc_local([4], "bool")
                kperm_indices_local = T.alloc_local([4], indices_dtype)
                producer_ldg_tx = (tid - 512) % 8
                producer_ldg_ty = (tid - 512) // 8

                for i_i in range(active_num_i if has_dynamic_lengths else num_i_orig):
                    if i_i < (main_num_i if has_dynamic_lengths else num_i_orig):
                        orig_block_index = i_i
                        for r in T.unroll(4):
                            token_pos = (
                                orig_block_index * block_i
                                + ((r * 16 + producer_ldg_ty) % 8) * (block_i // 8)
                                + (r * 16 + producer_ldg_ty) // 8
                            )
                            kperm_indices_local[r] = indices[s_i, g_i, token_pos]
                            kperm_mask_local[r] = (
                                kperm_indices_local[r] >= 0
                                and kperm_indices_local[r] < seq_len_kv
                            )
                            if has_topk_length:
                                kperm_mask_local[r] = (
                                    kperm_mask_local[r] and token_pos < topk_len
                                )
                            kperm_indices_local[r] = T.if_then_else(
                                kperm_mask_local[r],
                                kperm_indices_local[r],
                                0,
                            )

                        T.barrier_wait(bar_kv0_free, (i_i & 1) ^ 1)
                        T.annotate_layout(
                            {
                                kv_shared_l[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_l[:, :],
                                    k_major=True,
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
                                            64 * u + producer_ldg_tx * 8 + v,
                                        ],
                                        kv_shared_l[
                                            r * 16 + producer_ldg_ty,
                                            64 * u + producer_ldg_tx * 8 + v,
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )
                        for r in T.unroll(4):
                            perm_row = ((r * 16 + producer_ldg_ty) % 8) * (
                                block_i // 8
                            ) + (r * 16 + producer_ldg_ty) // 8
                            is_kv_valid[perm_row] = kperm_mask_local[r]
                            kv_indices[perm_row] = kperm_indices_local[r]
                            for row_j in T.unroll(token_pack):
                                row_token = s_i * token_pack + row_j
                                kv_logical_pos = T.alloc_var(T.int32)
                                valid = T.alloc_var("bool")
                                kv_logical_pos = kperm_indices_local[r]
                                if compressed_kv_len > 0:
                                    if kperm_indices_local[r] < compressed_kv_len:
                                        kv_logical_pos = (
                                            kperm_indices_local[r] + 1
                                        ) * compress_ratio - 1
                                    else:
                                        kv_logical_pos = (
                                            kperm_indices_local[r] - compressed_kv_len
                                        )
                                if has_row_mask:
                                    valid = kperm_mask_local[r]
                                    row_mask = row_masks[
                                        s_i,
                                        g_i,
                                        orig_block_index * block_i + perm_row,
                                    ]
                                    valid = valid and (
                                        ((row_mask // (1 << row_j)) % 2) != 0
                                    )
                                else:
                                    valid = (
                                        kperm_mask_local[r]
                                        and kv_logical_pos <= row_token
                                    )
                                if causal_window > 0 and not has_row_mask:
                                    if compressed_kv_len > 0:
                                        valid = valid and (
                                            kperm_indices_local[r] < compressed_kv_len
                                            or kv_logical_pos
                                            > row_token - causal_window
                                        )
                                    else:
                                        valid = (
                                            valid
                                            and kv_logical_pos
                                            > row_token - causal_window
                                        )
                                row_valid_shared[perm_row, row_j] = valid
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_ready)

                        T.barrier_wait(bar_kv1_free, (i_i & 1) ^ 1)
                        T.annotate_layout(
                            {
                                kv_shared_r[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r[:, :],
                                    k_major=True,
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
                                            dim_qk // 2
                                            + 64 * u
                                            + producer_ldg_tx * 8
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

    if has_topk_length and has_row_mask and has_attn_sink:

        @T.prim_func
        def dsa_prefill(
            q: T.Tensor(q_shape, dtype),
            kv: T.Tensor(kv_shape, dtype),
            indices: T.Tensor(indices_shape, indices_dtype),
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            topk_length: T.Tensor([seq_len], indices_dtype),
            row_masks: T.Tensor(row_masks_shape, indices_dtype),
            attn_sink: T.Tensor([num_heads], accum_dtype),
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                topk_length,
                row_masks,
                attn_sink,
                output,
                max_logits_out,
                lse,
            )

    elif has_topk_length and has_row_mask:

        @T.prim_func
        def dsa_prefill(
            q: T.Tensor(q_shape, dtype),
            kv: T.Tensor(kv_shape, dtype),
            indices: T.Tensor(indices_shape, indices_dtype),
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            topk_length: T.Tensor([seq_len], indices_dtype),
            row_masks: T.Tensor(row_masks_shape, indices_dtype),
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                topk_length,
                row_masks,
                None,
                output,
                max_logits_out,
                lse,
            )

    elif has_topk_length and has_attn_sink:

        @T.prim_func
        def dsa_prefill(
            q: T.Tensor(q_shape, dtype),
            kv: T.Tensor(kv_shape, dtype),
            indices: T.Tensor(indices_shape, indices_dtype),
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
                None,
                attn_sink,
                output,
                max_logits_out,
                lse,
            )

    elif has_topk_length:

        @T.prim_func
        def dsa_prefill(
            q: T.Tensor(q_shape, dtype),
            kv: T.Tensor(kv_shape, dtype),
            indices: T.Tensor(indices_shape, indices_dtype),
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
                None,
                output,
                max_logits_out,
                lse,
            )

    elif has_row_mask and has_attn_sink:

        @T.prim_func
        def dsa_prefill(
            q: T.Tensor(q_shape, dtype),
            kv: T.Tensor(kv_shape, dtype),
            indices: T.Tensor(indices_shape, indices_dtype),
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            row_masks: T.Tensor(row_masks_shape, indices_dtype),
            attn_sink: T.Tensor([num_heads], accum_dtype),
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                None,
                row_masks,
                attn_sink,
                output,
                max_logits_out,
                lse,
            )

    elif has_row_mask:

        @T.prim_func
        def dsa_prefill(
            q: T.Tensor(q_shape, dtype),
            kv: T.Tensor(kv_shape, dtype),
            indices: T.Tensor(indices_shape, indices_dtype),
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            row_masks: T.Tensor(row_masks_shape, indices_dtype),
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                None,
                row_masks,
                None,
                output,
                max_logits_out,
                lse,
            )

    elif has_attn_sink:

        @T.prim_func
        def dsa_prefill(
            q: T.Tensor(q_shape, dtype),
            kv: T.Tensor(kv_shape, dtype),
            indices: T.Tensor(indices_shape, indices_dtype),
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
                None,
                attn_sink,
                output,
                max_logits_out,
                lse,
            )

    else:

        @T.prim_func
        def dsa_prefill(
            q: T.Tensor(q_shape, dtype),
            kv: T.Tensor(kv_shape, dtype),
            indices: T.Tensor(indices_shape, indices_dtype),
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
                None,
                output,
                max_logits_out,
                lse,
            )

    return dsa_prefill


def sparse_mla_fwd_interface_model1_pack(
    q,
    kv,
    indices=None,
    topk_length=None,
    row_masks=None,
    sm_scale=None,
    attn_sink=None,
    return_p_sum: bool = False,
    d_v=512,
    threads=640,
    verbose=False,
    return_max_logits: bool = False,
    causal_window: int = 0,
    compressed_kv_len: int = 0,
    compress_ratio: int = 1,
    token_pack: int = 8,
    pack_metadata=None,
):
    """Sequence-pack direct-causal prefill path.

    This packs q [S, H, D] into [ceil(S/pack_s), pack_s * H, D] so the existing BM64
    SQMMA/PV layout is reused. It assumes contiguous causal KV rows and uses
    row-wise masking inside each pack.
    """
    assert return_p_sum is False, "This kernel file is for fwd only"
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert q.is_contiguous() and kv.is_contiguous()
    seq_len, heads, dim_q = q.shape
    seq_len_kv, kv_group, _ = kv.shape
    dim = d_v
    assert token_pack in (2, 4, 8), f"unsupported token_pack={token_pack}"
    assert heads * token_pack == 64, (
        f"pack kernel expects heads * token_pack == 64, got heads={heads}, token_pack={token_pack}"
    )
    assert kv_group == 1
    assert dim_q == dim == 512
    # TODO(dsv4): support non-pack-aligned q lengths by padding/repacking
    # tail rows or by consuming explicit pack_metadata from the framework.
    # Keep this guard until irregular prefill is validated end-to-end.
    assert seq_len % token_pack == 0, (
        "pack kernel requires seq_len multiple of token_pack"
    )
    assert compressed_kv_len >= 0
    assert compress_ratio >= 1
    assert pack_metadata is None, (
        "pack_metadata is reserved for future row-level packing"
    )

    pack_s = token_pack
    packed_seq = seq_len // pack_s
    packed_heads = pack_s * heads
    device = q.device
    q_pack = q.view(packed_seq, packed_heads, dim).contiguous()
    if indices is None:
        assert row_masks is None
        if compressed_kv_len > 0:
            max_swa_k = causal_window + pack_s - 1 if causal_window > 0 else seq_len
            max_pack_k = min(compressed_kv_len, seq_len_kv) + min(
                max(seq_len_kv - compressed_kv_len, 0),
                max_swa_k,
            )
        elif causal_window > 0:
            max_pack_k = min(seq_len_kv, causal_window + pack_s - 1)
        else:
            max_pack_k = min(seq_len, seq_len_kv)
        topk = ((max_pack_k + 63) // 64) * 64
        offsets = torch.arange(topk, dtype=torch.int32, device=device)
        pack_starts = (
            torch.arange(packed_seq, dtype=torch.int32, device=device) * pack_s
        )
        pack_ends = torch.clamp(pack_starts + pack_s - 1, max=seq_len_kv - 1)
        if compressed_kv_len > 0:
            compressed_lens = torch.clamp(
                (pack_ends + 1) // compress_ratio,
                min=0,
                max=min(compressed_kv_len, seq_len_kv),
            )
            if causal_window > 0:
                swa_starts = torch.clamp(pack_starts - causal_window + 1, min=0)
            else:
                swa_starts = torch.zeros(
                    (packed_seq,), dtype=torch.int32, device=device
                )
            swa_ends = torch.clamp(
                pack_ends,
                min=0,
                max=max(seq_len_kv - compressed_kv_len - 1, 0),
            )
            swa_lens = torch.clamp(swa_ends - swa_starts + 1, min=0)
            topk_length = torch.clamp(compressed_lens + swa_lens, min=0, max=topk)
            indices_2d = torch.full(
                (packed_seq, topk), -1, dtype=torch.int32, device=device
            )
            compressed_offsets = offsets[None, :]
            compressed_mask = compressed_offsets < compressed_lens[:, None]
            indices_2d = torch.where(compressed_mask, compressed_offsets, indices_2d)
            swa_offsets = offsets[None, :] - compressed_lens[:, None]
            swa_mask = swa_offsets < swa_lens[:, None]
            indices_2d = torch.where(
                (~compressed_mask) & swa_mask,
                compressed_kv_len + swa_starts[:, None] + swa_offsets,
                indices_2d,
            )
        else:
            if causal_window > 0:
                starts = torch.clamp(pack_starts - causal_window + 1, min=0)
            else:
                starts = torch.zeros((packed_seq,), dtype=torch.int32, device=device)
            topk_length = torch.clamp(pack_ends - starts + 1, min=0, max=topk)
            indices_2d = starts[:, None] + offsets[None, :]
            indices_2d = torch.where(
                offsets[None, :] < topk_length[:, None],
                indices_2d,
                torch.full_like(indices_2d, -1),
            )
        indices = indices_2d[:, None, :].contiguous()
    else:
        assert indices.dtype == torch.int32 and indices.is_contiguous()
        assert indices.shape[0] == packed_seq and indices.shape[1] == 1
        topk = indices.shape[-1]
        assert topk % 64 == 0
        if row_masks is not None:
            assert row_masks.dtype == torch.int32 and row_masks.is_contiguous()
            assert row_masks.shape == indices.shape

    topk_length = validate_token_lengths(topk_length, packed_seq, "topk_length")
    attn_sink = validate_prefill_attn_sink(attn_sink, heads)
    attn_sink_pack = (
        attn_sink.repeat(pack_s).contiguous() if attn_sink is not None else None
    )

    kernel_kwargs = {
        "kv_group": kv_group,
        "sm_scale": sm_scale,
        "is_causal": True,
        "threads": threads,
        "has_attn_sink": attn_sink_pack is not None,
        "has_topk_length": topk_length is not None,
        "has_row_mask": row_masks is not None,
        "causal_window": causal_window,
        "compressed_kv_len": compressed_kv_len,
        "compress_ratio": compress_ratio,
    }
    # These inputs dominate the contiguous output and auxiliary tensor spans.
    kernel_factory = jit_for_tensor_addressing(
        sparse_attention_fwd_kernel_model1_pack,
        q_pack,
        kv,
        indices,
    )
    kernel = kernel_factory(packed_heads, dim, topk, pack_s, **kernel_kwargs)
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()

    args = [q_pack, kv, indices]
    if topk_length is not None:
        args.append(topk_length)
    if row_masks is not None:
        args.append(row_masks)
    if attn_sink_pack is not None:
        args.append(attn_sink_pack)
    out = kernel(*args)
    out_tensor, max_logits, lse_tensor = out
    out_tensor = out_tensor.view(seq_len, heads, dim)
    lse_tensor = lse_tensor.view(seq_len, heads)
    max_logits = max_logits.view(seq_len, heads)
    if return_max_logits:
        return out_tensor, max_logits, lse_tensor
    return out_tensor, lse_tensor
