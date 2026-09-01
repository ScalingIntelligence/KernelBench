# ruff: noqa

from typing import Any

import torch

if not hasattr(torch, "uint16"):
    torch.uint16 = torch.int16
if not hasattr(torch, "uint32"):
    torch.uint32 = torch.int32
if not hasattr(torch, "uint64"):
    torch.uint64 = torch.int64
import tilelang
from tilelang import language as T
from tvm import tir

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
    validate_batch_lengths,
)
from .sparse_mla_index_type import jit_for_tensor_addressing

from ...execution_context import raise_complete_if_dry_run


@tilelang.jit(
    out_idx=[],
    pass_configs=SCHEDULED_DECODE_PASS_CONFIGS,
    compile_flags=SCHEDULED_DECODE_COMPILE_FLAGS,
)
def sparse_attention_decode_fwd_scheduled_kernel_model1(
    num_heads,
    dim,
    *,
    has_extra=False,
    kv_group=1,
    sm_scale=None,
    block_m=64,
    block_i=64,
    threads=0,
    consumer0_threads=256,
    consumer1_threads=256,
    producer_threads=128,
    max_nums_splits=32,
    has_attn_sink=False,
    has_topk_length=False,
    has_extra_topk_length=False,
    page_block_size=64,
    extra_page_block_size=None,
    support_split=True,
    use_int64_cosize=False,
    use_8byte_kv_loads=False,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504
    assert has_extra or not has_extra_topk_length, (
        "has_extra_topk_length requires has_extra"
    )
    if has_extra:
        assert extra_page_block_size is not None
    topk = T.dynamic("topk")
    if has_extra:
        extra_topk = T.dynamic("extra_topk")
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    dim_bytes = 584
    rope_bytes_offset = 448
    scale_bytes_offset = 576
    page_block_bytes = page_block_size * dim_bytes
    if has_extra:
        extra_page_block_bytes = extra_page_block_size * dim_bytes
        page_block_bytes_extra = extra_page_block_bytes

    num_blocks = T.dynamic("num_blocks")
    seq_len_kv = num_blocks * page_block_size
    if has_extra:
        num_blocks_extra = T.dynamic("num_blocks_extra")
        page_block_size_extra = extra_page_block_size
        seq_len_kv_extra = num_blocks_extra * page_block_size_extra
    num_mp_parts = T.dynamic("num_mp_parts")
    q_stride_b = T.dynamic("q_stride_b")
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    kv_nope_stride_block = T.dynamic("kv_nope_stride_block")
    kv_rope_stride_block = T.dynamic("kv_rope_stride_block")
    quant_scales_stride_block = T.dynamic("quant_scales_stride_block")
    indices_stride_b = T.dynamic("indices_stride_b")
    indices_stride_s = T.dynamic("indices_stride_s")
    indices_stride_g = T.dynamic("indices_stride_g")
    if has_extra:
        extra_kv_nope_stride_block = T.dynamic("extra_kv_nope_stride_block")
        extra_kv_rope_stride_block = T.dynamic("extra_kv_rope_stride_block")
        extra_quant_scales_stride_block = T.dynamic("extra_quant_scales_stride_block")
        extra_indices_stride_b = T.dynamic("extra_indices_stride_b")
        extra_indices_stride_s = T.dynamic("extra_indices_stride_s")
        extra_indices_stride_g = T.dynamic("extra_indices_stride_g")

    head_kv = num_heads // kv_group
    q_shape = [batch, seq_len, num_heads, dim]
    o_shape = [batch, seq_len, num_heads, dim]
    lse_shape = [batch, num_heads, seq_len]
    indices_shape = [batch, seq_len, kv_group, topk]
    if has_extra:
        extra_indices_shape = [batch, seq_len, kv_group, extra_topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    kv_latent_dtype = "float8_e4m3"
    dtype_bytes = 2
    assert block_m in (16, 64), "MODEL1 decode supports BM16 or BM64"
    assert block_i == 64, "MODEL1 decode supports BN64"
    assert producer_threads == 128, (
        "MODEL1 decode producer currently requires 128 threads"
    )
    assert consumer0_threads in (128, 256), (
        "MODEL1 decode consumer0 must use one or two warpgroups"
    )
    assert consumer1_threads in (128, 256), (
        "MODEL1 decode consumer1 must use one or two warpgroups"
    )
    assert consumer0_threads % 128 == 0 and consumer1_threads % 128 == 0
    assert consumer0_threads % 8 == 0 and consumer1_threads % 8 == 0
    total_role_threads = consumer0_threads + consumer1_threads + producer_threads
    if threads is None or threads == 0:
        threads = total_role_threads
    else:
        assert threads == total_role_threads, (
            "threads must equal consumer0_threads + consumer1_threads + "
            "producer_threads"
        )
    consumer1_start = consumer0_threads
    producer_start = consumer0_threads + consumer1_threads
    consumer0_gemm_threads = 128 if block_m == 16 else consumer0_threads
    assert consumer0_threads >= consumer0_gemm_threads
    consumer0_ldg_ty_count = consumer0_threads // 8
    consumer1_ldg_ty_count = consumer1_threads // 8
    assert block_i % consumer0_ldg_ty_count == 0
    assert block_i % consumer1_ldg_ty_count == 0
    consumer0_rows_per_thread = block_i // consumer0_ldg_ty_count
    consumer1_rows_per_thread = block_i // consumer1_ldg_ty_count
    kv_nope_shape = [num_blocks, page_block_bytes]
    kv_rope_shape = [num_blocks, page_block_bytes // 2]
    quant_scales_shape = [num_blocks, page_block_bytes]
    if has_extra:
        extra_kv_nope_shape = [num_blocks_extra, page_block_bytes_extra]
        extra_kv_rope_shape = [num_blocks_extra, page_block_bytes_extra // 2]
        extra_quant_scales_shape = [num_blocks_extra, page_block_bytes_extra]
    q_strides = (q_stride_b, q_stride_s, q_stride_h, 1)
    kv_nope_strides = (kv_nope_stride_block, 1)
    kv_rope_strides = (kv_rope_stride_block, 1)
    quant_scales_strides = (quant_scales_stride_block, 1)
    indices_strides = (indices_stride_b, indices_stride_s, indices_stride_g, 1)
    if has_extra:
        extra_kv_nope_strides = (extra_kv_nope_stride_block, 1)
        extra_kv_rope_strides = (extra_kv_rope_stride_block, 1)
        extra_quant_scales_strides = (extra_quant_scales_stride_block, 1)
        extra_indices_strides = (
            extra_indices_stride_b,
            extra_indices_stride_s,
            extra_indices_stride_g,
            1,
        )
    if use_int64_cosize:
        q_cosize = cosize(
            q_shape, tuple(tir.Cast("int64", stride) for stride in q_strides)
        )
        kv_nope_cosize = cosize(
            kv_nope_shape,
            tuple(tir.Cast("int64", stride) for stride in kv_nope_strides),
        )
        kv_rope_cosize = cosize(
            kv_rope_shape,
            tuple(tir.Cast("int64", stride) for stride in kv_rope_strides),
        )
        quant_scales_cosize = cosize(
            quant_scales_shape,
            tuple(tir.Cast("int64", stride) for stride in quant_scales_strides),
        )
        if has_extra:
            extra_kv_nope_cosize = cosize(
                extra_kv_nope_shape,
                tuple(tir.Cast("int64", stride) for stride in extra_kv_nope_strides),
            )
            extra_kv_rope_cosize = cosize(
                extra_kv_rope_shape,
                tuple(tir.Cast("int64", stride) for stride in extra_kv_rope_strides),
            )
            extra_quant_scales_cosize = cosize(
                extra_quant_scales_shape,
                tuple(
                    tir.Cast("int64", stride) for stride in extra_quant_scales_strides
                ),
            )
    else:
        q_cosize = cosize(q_shape, q_strides)
        kv_nope_cosize = cosize(kv_nope_shape, kv_nope_strides)
        kv_rope_cosize = cosize(kv_rope_shape, kv_rope_strides)
        quant_scales_cosize = cosize(quant_scales_shape, quant_scales_strides)
        if has_extra:
            extra_kv_nope_cosize = cosize(extra_kv_nope_shape, extra_kv_nope_strides)
            extra_kv_rope_cosize = cosize(extra_kv_rope_shape, extra_kv_rope_strides)
            extra_quant_scales_cosize = cosize(
                extra_quant_scales_shape, extra_quant_scales_strides
            )

    q_type: Any = T.StridedTensor(q_shape, q_strides, dtype)
    kv_nope_type: Any = T.StridedTensor(kv_nope_shape, kv_nope_strides, kv_latent_dtype)
    kv_rope_type: Any = T.StridedTensor(kv_rope_shape, kv_rope_strides, dtype)
    quant_scales_type: Any = T.StridedTensor(
        quant_scales_shape, quant_scales_strides, "uint8"
    )
    indices_type: Any = T.StridedTensor(indices_shape, indices_strides, indices_dtype)
    if has_extra:
        extra_kv_nope_type: Any = T.StridedTensor(
            extra_kv_nope_shape, extra_kv_nope_strides, kv_latent_dtype
        )
        extra_kv_rope_type: Any = T.StridedTensor(
            extra_kv_rope_shape, extra_kv_rope_strides, dtype
        )
        extra_quant_scales_type: Any = T.StridedTensor(
            extra_quant_scales_shape, extra_quant_scales_strides, "uint8"
        )
        extra_indices_type: Any = T.StridedTensor(
            extra_indices_shape, extra_indices_strides, indices_dtype
        )

    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), block_m)
    if padded_head_kv != head_kv:
        assert kv_group == 1

    if head_kv > block_m:
        assert head_kv % block_m == 0, "head_kv should be a multiple of block_m"
        head_repeats = head_kv // block_m
    else:
        head_repeats = 1

    heads_per_block = padded_head_kv if head_repeats == 1 else block_m
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

    update_online_softmax = make_scheduled_decode_online_softmax(
        h_per_block=heads_per_block,
        block_i=block_i,
        out_width=128,
        accum_dtype=accum_dtype,
        sm_scale=sm_scale,
    )
    finalize_left = make_scheduled_decode_finalize_left(
        h_per_block=heads_per_block,
        out_width=128,
        num_heads=num_heads,
        accum_dtype=accum_dtype,
        sm_scale=sm_scale,
        has_attn_sink=has_attn_sink,
        use_strict_valid=True,
        add_denominator_epsilon=False,
        sink_only_unsplit=True,
        sink_invalid_zero=True,
        wait_before_final=False,
        l0_start=0,
        l1_start=128,
        out_dtype=dtype,
        guard_invalid_heads=head_kv < block_m,
        support_split=support_split,
    )
    finalize_right = make_scheduled_decode_finalize_right(
        h_per_block=heads_per_block,
        out_width=128,
        num_heads=num_heads,
        has_attn_sink=has_attn_sink,
        wait_after_scale=False,
        r0_start=256,
        r1_start=384,
        out_dtype=dtype,
        guard_invalid_heads=head_kv < block_m,
        support_split=support_split,
    )
    value_stage_continuity = 64 if block_m == 16 else 128
    stage_value_shared_c0 = make_scheduled_decode_stage_value_shared(
        block_i=block_i,
        continuity=value_stage_continuity,
        ldg_ty_count=consumer0_ldg_ty_count,
        rows_per_thread=consumer0_rows_per_thread,
    )
    stage_value_shared_c1 = make_scheduled_decode_stage_value_shared(
        block_i=block_i,
        continuity=value_stage_continuity,
        ldg_ty_count=consumer1_ldg_ty_count,
        rows_per_thread=consumer1_rows_per_thread,
    )
    load_indices = make_scheduled_decode_indices_loader(block_i=block_i)
    pv_gemm_policy = T.GemmWarpPolicy.FullRow

    @T.macro
    def load_model1_paged_kv_block(
        kv_rope,
        quant_scales,
        rope_robust_desc_arg,
        scale_robust_desc_arg,
        page_size,
        kperm_indices_local,
        kv_indices,
        kv_shared_l,
        kv_shared_r,
        quant_shared,
        ldg_ty,
        ldg_tx,
        ldg_scale_ty,
        ldg_scale_tx,
        phase,
        bar_kv0_free,
        bar_kv_mask_ready,
        bar_indices_ready,
        bar_kv0_ready,
        bar_kv1_free,
        bar_kv1_ready,
    ):
        T.barrier_wait(bar_kv0_free, (phase & 1) ^ 1)
        for r in T.unroll(4):
            for u in T.unroll(4):
                for v in T.vectorized(4):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + 32 * u
                            + ldg_tx * 4
                            + v,
                        ],
                        kv_shared_l[r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )
        T.lma_wait()
        T.barrier_arrive(bar_kv_mask_ready)
        T.barrier_arrive(bar_indices_ready)
        T.barrier_wait(bar_indices_ready, phase & 1)

        for c in T.vectorized(4):
            T.copy(
                quant_scales[
                    kv_indices[ldg_scale_ty] // page_size,
                    page_size * scale_bytes_offset
                    + (kv_indices[ldg_scale_ty] % page_size) * 8
                    + ldg_scale_tx * 4
                    + c,
                ],
                quant_shared[ldg_scale_ty, ldg_scale_tx * 4 + c],
                force_async_copy=True,
                src_robust_desc=scale_robust_desc_arg,
            )
        T.ptx_commit_group()
        T.ptx_wait_group(0)
        T.barrier_arrive(bar_kv0_ready)

        T.barrier_wait(bar_kv1_free, (phase & 1) ^ 1)
        for r in T.unroll(4):
            for u in T.unroll(3):
                for v in T.vectorized(4):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + dim // 4
                            + 32 * u
                            + ldg_tx * 4
                            + v,
                        ],
                        kv_shared_r[r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )

        if use_8byte_kv_loads:
            for r in T.unroll(4):
                for v in T.vectorized(4):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + rope_bytes_offset // 2
                            + ldg_tx * 8
                            + v,
                        ],
                        kv_shared_r[r * 16 + ldg_ty, 64 * 3 + ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )
                for v in T.vectorized(4):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + rope_bytes_offset // 2
                            + ldg_tx * 8
                            + 4
                            + v,
                        ],
                        kv_shared_r[r * 16 + ldg_ty, 64 * 3 + ldg_tx * 8 + 4 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )
        else:
            for r in T.unroll(4):
                for v in T.vectorized(8):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + rope_bytes_offset // 2
                            + ldg_tx * 8
                            + v,
                        ],
                        kv_shared_r[r * 16 + ldg_ty, 64 * 3 + ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )
        T.ptx_commit_group()
        T.ptx_wait_group(0)
        T.barrier_arrive(bar_kv1_ready)
        T.sync_threads(1, 128)

    @T.macro
    def dsa_decode_body(
        q,
        kv_nope,
        kv_rope,
        quant_scales,
        indices,
        topk_length,
        extra_kv_nope,
        extra_kv_rope,
        extra_quant_scales,
        extra_indices,
        extra_topk_length,
        attn_sink,
        tile_scheduler_metadata,
        num_splits,
        glse,
        output_partial,
        output,
        lse,
    ):
        # MODEL1 scheduled split main kernel. It follows the FlashMLA scheduler
        # contract: each program consumes one metadata part and either writes the
        # final output for unsplit batches or partial output/LSE for split ones.
        with T.Kernel(
            seq_len * head_repeats, kv_group, num_mp_parts, threads=threads
        ) as (bx, by, bz):
            T.assume(q_stride_b % 8 == 0)
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(kv_nope_stride_block % 8 == 0)
            if use_8byte_kv_loads:
                T.assume(kv_rope_stride_block % 4 == 0)
            else:
                T.assume(kv_rope_stride_block % 8 == 0)
            T.assume(quant_scales_stride_block % 8 == 0)
            T.assume(indices_stride_b % 8 == 0)
            T.assume(indices_stride_s % 8 == 0)
            T.assume(indices_stride_g % 8 == 0)
            if has_extra:
                T.assume(extra_kv_nope_stride_block % 8 == 0)
                if use_8byte_kv_loads:
                    T.assume(extra_kv_rope_stride_block % 4 == 0)
                else:
                    T.assume(extra_kv_rope_stride_block % 8 == 0)
                T.assume(extra_quant_scales_stride_block % 8 == 0)
                T.assume(extra_indices_stride_b % 8 == 0)
                T.assume(extra_indices_stride_s % 8 == 0)
                T.assume(extra_indices_stride_g % 8 == 0)
            q_shared_l = T.alloc_shared([heads_per_block, 256], dtype)
            q_shared_r = T.alloc_shared([heads_per_block, 256], dtype)
            kv_shared_l = T.alloc_shared([block_i, 256], dtype)
            kv_shared_r = T.alloc_shared([block_i, 256], dtype)
            v_shared_0 = T.alloc_shared([block_i, 128], dtype)
            v_shared_1 = T.alloc_shared([block_i, 128], dtype)

            scores_shared = T.alloc_shared([heads_per_block, block_i], dtype)
            sum_exp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            sink_scale_shared = T.alloc_shared([heads_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([block_i], "bool", scope="shared")
            kv_indices = T.alloc_shared([block_i], "int32", scope="shared")
            quant_shared = T.alloc_shared([block_i, 8], "uint8")

            bar_kv_mask_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv_mask_free = T.alloc_barrier(arrive_count=consumer0_gemm_threads)
            bar_q = T.alloc_barrier(arrive_count=producer_start)
            bar_indices_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv1_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv0_quant_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_kv1_quant_ready = T.alloc_barrier(arrive_count=consumer1_threads)

            bar_kv0_free = T.alloc_barrier(arrive_count=consumer0_gemm_threads)
            bar_kv1_free = T.alloc_barrier(arrive_count=consumer0_gemm_threads)

            bar_vl0_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vl1_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vr0_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vr1_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vl0_free = T.alloc_barrier(arrive_count=consumer0_gemm_threads)
            bar_vl1_free = T.alloc_barrier(arrive_count=consumer0_gemm_threads)

            bar_p_ready = T.alloc_barrier(arrive_count=consumer0_gemm_threads)
            bar_final = T.alloc_barrier(arrive_count=consumer0_gemm_threads)

            q_robust_desc = T.make_robust_desc(
                T.address_of(q[0, 0, 0, 0]),
                q_cosize * dtype_bytes,
            )
            nope_robust_desc = T.make_robust_desc(
                T.address_of(kv_nope[0, 0]),
                kv_nope_cosize,
            )
            rope_robust_desc = T.make_robust_desc(
                T.address_of(kv_rope[0, 0]),
                kv_rope_cosize * dtype_bytes,
            )
            scale_robust_desc = T.make_robust_desc(
                T.address_of(quant_scales[0, 0]),
                quant_scales_cosize,
            )
            if has_extra:
                extra_nope_robust_desc = T.make_robust_desc(
                    T.address_of(extra_kv_nope[0, 0]),
                    extra_kv_nope_cosize,
                )
                extra_rope_robust_desc = T.make_robust_desc(
                    T.address_of(extra_kv_rope[0, 0]),
                    extra_kv_rope_cosize * dtype_bytes,
                )
                extra_scale_robust_desc = T.make_robust_desc(
                    T.address_of(extra_quant_scales[0, 0]),
                    extra_quant_scales_cosize,
                )

            T.sync_threads()

            g_i = by
            s_i = bx if head_repeats == 1 else (bx // head_repeats)
            h0 = g_i * padded_head_kv + (
                0 if head_repeats == 1 else (bx % head_repeats) * block_m
            )
            h1 = h0 + heads_per_block
            tid = T.get_thread_binding()

            begin_idx = tile_scheduler_metadata[bz, 0]
            sched_begin_block_idx = tile_scheduler_metadata[bz, 1]
            end_idx = tile_scheduler_metadata[bz, 2]
            sched_end_block_idx = tile_scheduler_metadata[bz, 3]
            begin_n_split_idx = tile_scheduler_metadata[bz, 4]
            phase_count = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)

            for b_i in range(begin_idx, end_idx + 1, 1):
                tir.call_extern("void", "__musa_loop_transparent_outermost")
                start_block_idx = T.alloc_var(T.int32)
                end_block_idx = T.alloc_var(T.int32)
                n_split_idx = T.alloc_var(T.int32)
                dynamic_main_blocks = T.alloc_var(T.int32)
                dynamic_total_blocks = T.alloc_var(T.int32)
                if has_topk_length:
                    dynamic_main_blocks = T.max(T.ceildiv(topk_length[b_i], block_i), 1)
                else:
                    dynamic_main_blocks = T.max(T.ceildiv(topk, block_i), 1)
                dynamic_total_blocks = dynamic_main_blocks
                if has_extra:
                    if has_extra_topk_length:
                        dynamic_total_blocks += T.ceildiv(
                            extra_topk_length[b_i], block_i
                        )
                    else:
                        dynamic_total_blocks += T.ceildiv(extra_topk, block_i)
                start_block_idx = T.if_then_else(
                    b_i == begin_idx, sched_begin_block_idx, 0
                )
                end_block_idx = T.if_then_else(
                    b_i == end_idx, sched_end_block_idx, dynamic_total_blocks
                )
                n_split_idx = T.if_then_else(b_i == begin_idx, begin_n_split_idx, 0)
                is_unsplit = (num_splits[b_i + 1] - num_splits[b_i]) == 1

                if tid < producer_start:
                    T.copy(
                        q[b_i, s_i, h0:h1, 0:256],
                        q_shared_l,
                        barrier=bar_q,
                        src_robust_desc=q_robust_desc,
                    )
                    T.copy(
                        q[b_i, s_i, h0:h1, 256:512],
                        q_shared_r,
                        barrier=bar_q,
                        src_robust_desc=q_robust_desc,
                    )
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    T.barrier_arrive(bar_q)
                    T.barrier_wait(bar_q, (b_i - begin_idx) & 1)

                if tid < consumer0_gemm_threads:
                    sumexp = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_inv = T.alloc_fragment([heads_per_block], accum_dtype)
                    alpha_local = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([heads_per_block], accum_dtype)
                    acc_s = T.alloc_fragment([heads_per_block, block_i], accum_dtype)
                    acc_s_cast = T.alloc_fragment([heads_per_block, block_i], dtype)
                    acc_o_l_0 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    acc_o_l_1 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    kv_reg_l = T.alloc_local([consumer0_rows_per_thread * 32], dtype)
                    kv_reg_l_fp16 = T.view(
                        kv_reg_l, [consumer0_rows_per_thread * 32], T.float16
                    )
                    kv_reg_l_bf16_load = T.alloc_local(
                        [consumer0_rows_per_thread * 16], T.bfloat16
                    )
                    kv_reg_l_fp8 = T.view(
                        kv_reg_l_bf16_load,
                        [consumer0_rows_per_thread * 32],
                        kv_latent_dtype,
                    )
                    quant_u8_l = T.alloc_local([consumer0_rows_per_thread, 4], "uint8")
                    quant_local_l = T.alloc_local(
                        [consumer0_rows_per_thread, 4], T.float32
                    )
                    c0_helper_ldg_tx = tid % 8
                    c0_helper_ldg_ty = tid // 8
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l_0, 0)
                    T.fill(acc_o_l_1, 0)

                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv0_ready, (phase_count[0] & 1))

                        for r in T.unroll(consumer0_rows_per_thread):
                            row = c0_helper_ldg_ty + r * consumer0_ldg_ty_count
                            T.copy(quant_shared[row, 0:4], quant_u8_l[r, :])

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

                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    row = c0_helper_ldg_ty + r * consumer0_ldg_ty_count
                                    kv_reg_l_bf16_load[r * 16 + u * 4 + v] = (
                                        kv_shared_l[
                                            row,
                                            64 * u + c0_helper_ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(4):
                                quant_local_l[r, u] = T.reinterpret(
                                    "float32",
                                    T.Cast("int32", quant_u8_l[r, u]) << 23,
                                )
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_l[idx] = T.Cast(
                                        dtype,
                                        kv_reg_l_fp16[idx] * quant_local_l[r, u],
                                    )
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    row = c0_helper_ldg_ty + r * consumer0_ldg_ty_count
                                    kv_shared_l[
                                        row,
                                        64 * u + c0_helper_ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + u * 8 + v]
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    row = c0_helper_ldg_ty + r * consumer0_ldg_ty_count
                                    kv_reg_l_bf16_load[r * 16 + (u + 2) * 4 + v] = (
                                        kv_shared_l[
                                            row,
                                            64 * (u + 2) + c0_helper_ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_l[idx] = T.Cast(
                                        dtype,
                                        kv_reg_l_fp16[idx] * quant_local_l[r, u + 2],
                                    )
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    row = c0_helper_ldg_ty + r * consumer0_ldg_ty_count
                                    kv_shared_l[
                                        row,
                                        64 * (u + 2) + c0_helper_ldg_tx * 8 + v,
                                    ] = kv_reg_l[idx]
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_quant_ready)

                        T.barrier_wait(bar_kv_mask_ready, (phase_count[0] & 1))
                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                is_kv_valid[bi_i % 8 * 8 + bi_i // 8],
                                0,
                                -(2**30),
                            )
                        T.lma_wait()
                        T.barrier_arrive(bar_kv_mask_free)

                        T.barrier_wait(bar_kv0_quant_ready, (phase_count[0] & 1))
                        T.gemm(
                            q_shared_l,
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
                        T.warpgroup_wait(0)
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
                        )

                        T.lma_wait()
                        T.barrier_arrive(bar_p_ready)
                        stage_value_shared_c0(
                            v_shared_0,
                            kv_reg_l,
                            c0_helper_ldg_ty,
                            c0_helper_ldg_tx,
                            0,
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vl0_ready)
                        T.barrier_wait(bar_vl0_ready, (phase_count[0] & 1))

                        T.gemm(
                            scores_shared,
                            v_shared_0,
                            acc_o_l_0,
                            policy=pv_gemm_policy,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()

                        stage_value_shared_c0(
                            v_shared_1,
                            kv_reg_l,
                            c0_helper_ldg_ty,
                            c0_helper_ldg_tx,
                            2,
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
                            policy=pv_gemm_policy,
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
                    )
                elif tid < consumer0_threads:
                    kv_reg_l = T.alloc_local([consumer0_rows_per_thread * 32], dtype)
                    kv_reg_l_fp16 = T.view(
                        kv_reg_l, [consumer0_rows_per_thread * 32], T.float16
                    )
                    kv_reg_l_bf16_load = T.alloc_local(
                        [consumer0_rows_per_thread * 16], T.bfloat16
                    )
                    kv_reg_l_fp8 = T.view(
                        kv_reg_l_bf16_load,
                        [consumer0_rows_per_thread * 32],
                        kv_latent_dtype,
                    )
                    quant_u8_l = T.alloc_local([consumer0_rows_per_thread, 4], "uint8")
                    quant_local_l = T.alloc_local(
                        [consumer0_rows_per_thread, 4], T.float32
                    )
                    c0_ldg_tx = tid % 8
                    c0_ldg_ty = tid // 8

                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv0_ready, (phase_count[0] & 1))

                        for r in T.unroll(consumer0_rows_per_thread):
                            row = c0_ldg_ty + r * consumer0_ldg_ty_count
                            T.copy(quant_shared[row, 0:4], quant_u8_l[r, :])

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

                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    row = c0_ldg_ty + r * consumer0_ldg_ty_count
                                    kv_reg_l_bf16_load[r * 16 + u * 4 + v] = (
                                        kv_shared_l[
                                            row,
                                            64 * u + c0_ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(4):
                                quant_local_l[r, u] = T.reinterpret(
                                    "float32",
                                    T.Cast("int32", quant_u8_l[r, u]) << 23,
                                )
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_l[idx] = T.Cast(
                                        dtype,
                                        kv_reg_l_fp16[idx] * quant_local_l[r, u],
                                    )
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    row = c0_ldg_ty + r * consumer0_ldg_ty_count
                                    kv_shared_l[
                                        row,
                                        64 * u + c0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + u * 8 + v]
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    row = c0_ldg_ty + r * consumer0_ldg_ty_count
                                    kv_reg_l_bf16_load[r * 16 + (u + 2) * 4 + v] = (
                                        kv_shared_l[
                                            row,
                                            64 * (u + 2) + c0_ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_l[idx] = T.Cast(
                                        dtype,
                                        kv_reg_l_fp16[idx] * quant_local_l[r, u + 2],
                                    )
                        for r in T.unroll(consumer0_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    row = c0_ldg_ty + r * consumer0_ldg_ty_count
                                    kv_shared_l[
                                        row,
                                        64 * (u + 2) + c0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[idx]
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_quant_ready)

                        T.barrier_wait(bar_p_ready, (phase_count[0] & 1))
                        stage_value_shared_c0(
                            v_shared_0, kv_reg_l, c0_ldg_ty, c0_ldg_tx, 0
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vl0_ready)
                        T.barrier_wait(bar_vl0_ready, (phase_count[0] & 1))

                        stage_value_shared_c0(
                            v_shared_1, kv_reg_l, c0_ldg_ty, c0_ldg_tx, 2
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vl1_ready)
                        T.barrier_wait(bar_vl1_ready, (phase_count[0] & 1))
                        phase_count[0] = phase_count[0] ^ 1
                elif tid < producer_start:
                    acc_o_r_0 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    acc_o_r_1 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    kv_reg_r = T.alloc_local([consumer1_rows_per_thread * 32], dtype)
                    kv_reg_r_fp16 = T.view(
                        kv_reg_r, [consumer1_rows_per_thread * 32], T.float16
                    )
                    kv_reg_r_bf16_load = T.alloc_local(
                        [consumer1_rows_per_thread * 16], T.bfloat16
                    )
                    kv_reg_r_fp8 = T.view(
                        kv_reg_r_bf16_load,
                        [consumer1_rows_per_thread * 32],
                        kv_latent_dtype,
                    )
                    quant_u8_r = T.alloc_local([consumer1_rows_per_thread, 3], "uint8")
                    quant_local_r = T.alloc_local(
                        [consumer1_rows_per_thread, 3], T.float32
                    )
                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)
                    c1_ldg_tx = (tid - consumer1_start) % 8
                    c1_ldg_ty = (tid - consumer1_start) // 8

                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv1_ready, (phase_count[0] & 1))

                        for r in T.unroll(consumer1_rows_per_thread):
                            row = c1_ldg_ty + r * consumer1_ldg_ty_count
                            T.copy(quant_shared[row, 4:7], quant_u8_r[r, :])

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
                        for r in T.unroll(consumer1_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                    kv_reg_r_bf16_load[r * 16 + u * 4 + v] = (
                                        kv_shared_r[
                                            row,
                                            64 * u + c1_ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()
                        for r in T.unroll(consumer1_rows_per_thread):
                            for u in T.unroll(3):
                                quant_local_r[r, u] = T.reinterpret(
                                    "float32",
                                    T.Cast("int32", quant_u8_r[r, u]) << 23,
                                )
                        for r in T.unroll(consumer1_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]
                        for r in T.unroll(consumer1_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_r[idx] = T.Cast(
                                        dtype,
                                        kv_reg_r_fp16[idx] * quant_local_r[r, u],
                                    )
                        for r in T.unroll(consumer1_rows_per_thread):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                    kv_shared_r[
                                        row,
                                        64 * u + c1_ldg_tx * 8 + v,
                                    ] = kv_reg_r[r * 32 + u * 8 + v]

                        for r in T.unroll(consumer1_rows_per_thread):
                            for v in T.vectorized(4):
                                row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                kv_reg_r_bf16_load[r * 16 + 8 + v] = kv_shared_r[
                                    row,
                                    128 + c1_ldg_tx * 8 + v,
                                ]
                        T.lma_wait()
                        for r in T.unroll(consumer1_rows_per_thread):
                            for v in T.vectorized(8):
                                idx = r * 32 + 16 + v
                                kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]
                        for r in T.unroll(consumer1_rows_per_thread):
                            for v in T.vectorized(8):
                                idx = r * 32 + 16 + v
                                kv_reg_r[idx] = T.Cast(
                                    dtype,
                                    kv_reg_r_fp16[idx] * quant_local_r[r, 2],
                                )
                        for r in T.unroll(consumer1_rows_per_thread):
                            for v in T.vectorized(8):
                                idx = r * 32 + 16 + v
                                row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                kv_shared_r[
                                    row,
                                    128 + c1_ldg_tx * 8 + v,
                                ] = kv_reg_r[idx]

                        for r in T.unroll(consumer1_rows_per_thread):
                            for v in T.vectorized(8):
                                idx = r * 32 + 24 + v
                                row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                kv_reg_r[idx] = kv_shared_r[
                                    row,
                                    192 + c1_ldg_tx * 8 + v,
                                ]
                        T.lma_wait()
                        T.barrier_arrive(bar_kv1_quant_ready)
                        T.barrier_wait(bar_vl0_free, (phase_count[0] & 1))
                        stage_value_shared_c1(
                            v_shared_0, kv_reg_r, c1_ldg_ty, c1_ldg_tx, 0
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vr0_ready)
                        T.barrier_wait(bar_vr0_ready, (phase_count[0] & 1))

                        T.barrier_wait(bar_p_ready, (phase_count[0] & 1))
                        for h_i, d_i in T.Parallel(heads_per_block, 128):
                            acc_o_r_0[h_i, d_i] *= alpha_shared[h_i]
                            acc_o_r_1[h_i, d_i] *= alpha_shared[h_i]

                        T.gemm(
                            scores_shared,
                            v_shared_0,
                            acc_o_r_0,
                            policy=pv_gemm_policy,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()

                        T.barrier_wait(bar_vl1_free, (phase_count[0] & 1))
                        stage_value_shared_c1(
                            v_shared_1, kv_reg_r, c1_ldg_ty, c1_ldg_tx, 2
                        )
                        T.lma_wait()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vr1_ready)
                        T.barrier_wait(bar_vr1_ready, (phase_count[0] & 1))

                        T.gemm(
                            scores_shared,
                            v_shared_1,
                            acc_o_r_1,
                            policy=pv_gemm_policy,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
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
                else:
                    kperm_mask_local = T.alloc_local([4], "bool")
                    kperm_indices_local = T.alloc_local([4], indices_dtype)
                    topk_len_local = T.alloc_local([1], indices_dtype)
                    if has_extra:
                        extra_topk_len_local = T.alloc_local([1], indices_dtype)
                    producer_tid = tid - producer_start
                    ldg_tx = producer_tid % 8
                    ldg_ty = producer_tid // 8
                    ldg_scale_tx = producer_tid % 2
                    ldg_scale_ty = producer_tid // 2
                    if has_topk_length:
                        topk_len_local[0] = topk_length[b_i]
                    else:
                        topk_len_local[0] = topk
                    if has_extra:
                        if has_extra_topk_length:
                            extra_topk_len_local[0] = extra_topk_length[b_i]
                        else:
                            extra_topk_len_local[0] = extra_topk

                    for i_i in range(start_block_idx, end_block_idx):
                        main_block_count = T.max(
                            T.ceildiv(topk_len_local[0], block_i), 1
                        )
                        use_extra = i_i >= main_block_count
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
                        if has_extra:
                            if use_extra:
                                block_index = i_i - main_block_count
                                load_indices(
                                    extra_indices,
                                    b_i,
                                    s_i,
                                    g_i,
                                    block_index,
                                    extra_topk_len_local[0],
                                    seq_len_kv_extra,
                                    ldg_ty,
                                    ldg_tx,
                                    phase_count[0],
                                    kperm_indices_local,
                                    kperm_mask_local,
                                    is_kv_valid,
                                    kv_indices,
                                    bar_kv_mask_free,
                                )

                                load_model1_paged_kv_block(
                                    extra_kv_rope,
                                    extra_quant_scales,
                                    extra_rope_robust_desc,
                                    extra_scale_robust_desc,
                                    page_block_size_extra,
                                    kperm_indices_local,
                                    kv_indices,
                                    kv_shared_l,
                                    kv_shared_r,
                                    quant_shared,
                                    ldg_ty,
                                    ldg_tx,
                                    ldg_scale_ty,
                                    ldg_scale_tx,
                                    phase_count[0],
                                    bar_kv0_free,
                                    bar_kv_mask_ready,
                                    bar_indices_ready,
                                    bar_kv0_ready,
                                    bar_kv1_free,
                                    bar_kv1_ready,
                                )
                                phase_count[0] = phase_count[0] ^ 1
                            else:
                                block_index = i_i
                                load_indices(
                                    indices,
                                    b_i,
                                    s_i,
                                    g_i,
                                    block_index,
                                    topk_len_local[0],
                                    seq_len_kv,
                                    ldg_ty,
                                    ldg_tx,
                                    phase_count[0],
                                    kperm_indices_local,
                                    kperm_mask_local,
                                    is_kv_valid,
                                    kv_indices,
                                    bar_kv_mask_free,
                                )
                                load_model1_paged_kv_block(
                                    kv_rope,
                                    quant_scales,
                                    rope_robust_desc,
                                    scale_robust_desc,
                                    page_block_size,
                                    kperm_indices_local,
                                    kv_indices,
                                    kv_shared_l,
                                    kv_shared_r,
                                    quant_shared,
                                    ldg_ty,
                                    ldg_tx,
                                    ldg_scale_ty,
                                    ldg_scale_tx,
                                    phase_count[0],
                                    bar_kv0_free,
                                    bar_kv_mask_ready,
                                    bar_indices_ready,
                                    bar_kv0_ready,
                                    bar_kv1_free,
                                    bar_kv1_ready,
                                )
                                phase_count[0] = phase_count[0] ^ 1
                        else:
                            block_index = i_i
                            load_indices(
                                indices,
                                b_i,
                                s_i,
                                g_i,
                                block_index,
                                topk_len_local[0],
                                seq_len_kv,
                                ldg_ty,
                                ldg_tx,
                                phase_count[0],
                                kperm_indices_local,
                                kperm_mask_local,
                                is_kv_valid,
                                kv_indices,
                                bar_kv_mask_free,
                            )
                            load_model1_paged_kv_block(
                                kv_rope,
                                quant_scales,
                                rope_robust_desc,
                                scale_robust_desc,
                                page_block_size,
                                kperm_indices_local,
                                kv_indices,
                                kv_shared_l,
                                kv_shared_r,
                                quant_shared,
                                ldg_ty,
                                ldg_tx,
                                ldg_scale_ty,
                                ldg_scale_tx,
                                phase_count[0],
                                bar_kv0_free,
                                bar_kv_mask_ready,
                                bar_indices_ready,
                                bar_kv0_ready,
                                bar_kv1_free,
                                bar_kv1_ready,
                            )
                            phase_count[0] = phase_count[0] ^ 1

        if support_split:
            # MODEL1 scheduled combine kernel. Only batches with more than one split
            # enter this stage; unsplit batches are already written by the split kernel.
            dsa_combine(num_splits, glse, output_partial, attn_sink, output, lse)

    @T.macro
    def run_no_extra(
        q,
        kv_nope,
        kv_rope,
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
    ):
        dsa_decode_body(
            q,
            kv_nope,
            kv_rope,
            quant_scales,
            indices,
            topk_length,
            None,
            None,
            None,
            None,
            None,
            attn_sink,
            tile_scheduler_metadata,
            num_splits,
            glse,
            output_partial,
            output,
            lse,
        )

    @T.macro
    def run_extra(
        q,
        kv_nope,
        kv_rope,
        quant_scales,
        indices,
        topk_length,
        extra_kv_nope,
        extra_kv_rope,
        extra_quant_scales,
        extra_indices,
        extra_topk_length,
        attn_sink,
        tile_scheduler_metadata,
        num_splits,
        glse,
        output_partial,
        output,
        lse,
    ):
        dsa_decode_body(
            q,
            kv_nope,
            kv_rope,
            quant_scales,
            indices,
            topk_length,
            extra_kv_nope,
            extra_kv_rope,
            extra_quant_scales,
            extra_indices,
            extra_topk_length,
            attn_sink,
            tile_scheduler_metadata,
            num_splits,
            glse,
            output_partial,
            output,
            lse,
        )

    topk_length_type: Any = T.Tensor([batch], indices_dtype)
    extra_topk_length_type: Any = T.Tensor([batch], indices_dtype)
    attn_sink_type: Any = T.Tensor([num_heads], accum_dtype)
    scheduler_metadata_type: Any = T.Tensor([num_mp_parts, 8], T.int32)
    num_splits_type: Any = T.Tensor([batch + 1], T.int32)
    output_type: Any = T.Tensor(o_shape, dtype)
    lse_type: Any = T.Tensor(lse_shape, accum_dtype)

    if not has_extra and support_split:
        glse_type: Any = T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads], accum_dtype
        )
        output_partial_type: Any = T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads, dim], accum_dtype
        )
        if has_topk_length and has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_no_extra(
                    q,
                    kv_nope,
                    kv_rope,
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
                )

        elif has_topk_length:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_no_extra(
                    q,
                    kv_nope,
                    kv_rope,
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
                )

        elif has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_no_extra(
                    q,
                    kv_nope,
                    kv_rope,
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
                )

        else:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_no_extra(
                    q,
                    kv_nope,
                    kv_rope,
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
                )

    elif not has_extra:
        if has_topk_length and has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_no_extra(
                    q,
                    kv_nope,
                    kv_rope,
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
                )

        elif has_topk_length:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_no_extra(
                    q,
                    kv_nope,
                    kv_rope,
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
                )

        elif has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_no_extra(
                    q,
                    kv_nope,
                    kv_rope,
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
                )

        else:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_no_extra(
                    q,
                    kv_nope,
                    kv_rope,
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
                )

    elif support_split:
        glse_type: Any = T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads], accum_dtype
        )
        output_partial_type: Any = T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads, dim], accum_dtype
        )
        if has_topk_length and has_extra_topk_length and has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                extra_topk_length: extra_topk_length_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    topk_length,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    extra_topk_length,
                    attn_sink,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                )

        elif has_topk_length and has_extra_topk_length:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                extra_topk_length: extra_topk_length_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    topk_length,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    extra_topk_length,
                    None,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                )

        elif has_topk_length and has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    topk_length,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    None,
                    attn_sink,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                )

        elif has_topk_length:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    topk_length,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    None,
                    None,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                )

        elif has_extra_topk_length and has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                extra_topk_length: extra_topk_length_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    None,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    extra_topk_length,
                    attn_sink,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                )

        elif has_extra_topk_length:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                extra_topk_length: extra_topk_length_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    None,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    extra_topk_length,
                    None,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                )

        elif has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    None,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    None,
                    attn_sink,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                )

        else:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                glse: glse_type,
                output_partial: output_partial_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    None,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    None,
                    None,
                    tile_scheduler_metadata,
                    num_splits,
                    glse,
                    output_partial,
                    output,
                    lse,
                )

    else:
        if has_topk_length and has_extra_topk_length and has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                extra_topk_length: extra_topk_length_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    topk_length,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    extra_topk_length,
                    attn_sink,
                    tile_scheduler_metadata,
                    num_splits,
                    None,
                    None,
                    output,
                    lse,
                )

        elif has_topk_length and has_extra_topk_length:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                extra_topk_length: extra_topk_length_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    topk_length,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    extra_topk_length,
                    None,
                    tile_scheduler_metadata,
                    num_splits,
                    None,
                    None,
                    output,
                    lse,
                )

        elif has_topk_length and has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    topk_length,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    None,
                    attn_sink,
                    tile_scheduler_metadata,
                    num_splits,
                    None,
                    None,
                    output,
                    lse,
                )

        elif has_topk_length:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                topk_length: topk_length_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    topk_length,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    None,
                    None,
                    tile_scheduler_metadata,
                    num_splits,
                    None,
                    None,
                    output,
                    lse,
                )

        elif has_extra_topk_length and has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                extra_topk_length: extra_topk_length_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    None,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    extra_topk_length,
                    attn_sink,
                    tile_scheduler_metadata,
                    num_splits,
                    None,
                    None,
                    output,
                    lse,
                )

        elif has_extra_topk_length:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                extra_topk_length: extra_topk_length_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    None,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    extra_topk_length,
                    None,
                    tile_scheduler_metadata,
                    num_splits,
                    None,
                    None,
                    output,
                    lse,
                )

        elif has_attn_sink:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                attn_sink: attn_sink_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    None,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    None,
                    attn_sink,
                    tile_scheduler_metadata,
                    num_splits,
                    None,
                    None,
                    output,
                    lse,
                )

        else:

            @T.prim_func
            def dsa_decode(
                q: q_type,
                kv_nope: kv_nope_type,
                kv_rope: kv_rope_type,
                quant_scales: quant_scales_type,
                indices: indices_type,
                extra_kv_nope: extra_kv_nope_type,
                extra_kv_rope: extra_kv_rope_type,
                extra_quant_scales: extra_quant_scales_type,
                extra_indices: extra_indices_type,
                tile_scheduler_metadata: scheduler_metadata_type,
                num_splits: num_splits_type,
                output: output_type,
                lse: lse_type,
            ):
                run_extra(
                    q,
                    kv_nope,
                    kv_rope,
                    quant_scales,
                    indices,
                    None,
                    extra_kv_nope,
                    extra_kv_rope,
                    extra_quant_scales,
                    extra_indices,
                    None,
                    None,
                    tile_scheduler_metadata,
                    num_splits,
                    None,
                    None,
                    output,
                    lse,
                )

    return dsa_decode


def sparse_mla_decode_fwd_scheduled_interface_model1(
    q,
    kv_nope,
    kv_rope,
    kv_scales,
    indices,
    tile_scheduler_metadata,
    num_splits,
    *,
    extra_kv_nope=None,
    extra_kv_rope=None,
    extra_kv_scales=None,
    extra_indices=None,
    topk_length=None,
    extra_topk_length=None,
    sm_scale=None,
    attn_sink=None,
    return_p_sum: bool = False,
    d_v=512,
    block_m=64,
    block_i=64,
    threads=0,
    consumer0_threads=256,
    consumer1_threads=256,
    producer_threads=128,
    verbose=False,
    page_block_size=64,
    extra_page_block_size=None,
    _cache_strides_prepared=False,
):
    assert return_p_sum is False, "This kernel file is for decode only"
    q_dtype = q.dtype
    device = q.device
    assert q_dtype == torch.bfloat16, "q must be bfloat16"
    assert kv_nope.dtype == torch.float8_e4m3fn, "kv_nope must be float8_e4m3fn"
    assert kv_rope.dtype == torch.bfloat16, "kv_rope must be bfloat16"
    assert kv_scales.dtype == torch.uint8, "kv_scales must be uint8"
    assert indices.dtype == torch.int32, "indices must be int32"
    q, q_shape = prepare_sparse_mla_decode_strided_tensor("q", q, multiple=8)
    if _cache_strides_prepared:
        kv_nope_shape = kv_nope.shape
        kv_rope_shape = kv_rope.shape
        kv_scales_shape = kv_scales.shape
    else:
        kv_nope, kv_nope_shape = prepare_sparse_mla_decode_strided_tensor(
            "kv_nope", kv_nope, multiple=8
        )
        kv_rope, kv_rope_shape = prepare_sparse_mla_decode_strided_tensor(
            "kv_rope", kv_rope, multiple=4
        )
        kv_scales, kv_scales_shape = prepare_sparse_mla_decode_strided_tensor(
            "kv_scales", kv_scales, multiple=8
        )
    indices, indices_shape = prepare_sparse_mla_decode_strided_tensor(
        "indices", indices, multiple=8
    )
    batch, seq_len, heads, dim_q = q_shape
    assert int(num_splits.numel()) == batch + 1
    num_blocks, page_block_bytes = kv_nope_shape
    kv_group = 1
    assert dim_q == d_v == 512
    assert kv_rope_shape == (num_blocks, page_block_bytes // 2)
    assert kv_scales_shape == (num_blocks, page_block_bytes)
    _, _, _, topk = indices_shape
    assert indices_shape == (batch, seq_len, kv_group, topk)

    extra_inputs = (extra_kv_nope, extra_kv_rope, extra_kv_scales, extra_indices)
    has_extra = all(tensor is not None for tensor in extra_inputs)
    assert has_extra or not any(tensor is not None for tensor in extra_inputs), (
        "extra_kv_nope, extra_kv_rope, extra_kv_scales, and extra_indices "
        "must be provided together"
    )
    if has_extra:
        assert extra_page_block_size is not None, (
            "extra_page_block_size must be provided with extra cache inputs"
        )
        assert extra_indices.dtype == torch.int32, "extra_indices must be int32"
        extra_indices, extra_indices_shape = prepare_sparse_mla_decode_strided_tensor(
            "extra_indices", extra_indices, multiple=8
        )
        extra_topk = extra_indices_shape[-1]
        assert extra_topk > 0, "extra_indices must contain at least one index"
        assert extra_indices_shape == (batch, seq_len, kv_group, extra_topk)
        assert extra_kv_nope.dtype == torch.float8_e4m3fn, (
            "extra_kv_nope must be float8_e4m3fn"
        )
        assert extra_kv_rope.dtype == torch.bfloat16, "extra_kv_rope must be bfloat16"
        assert extra_kv_scales.dtype == torch.uint8, "extra_kv_scales must be uint8"
        if _cache_strides_prepared:
            extra_kv_nope_shape = extra_kv_nope.shape
            extra_kv_rope_shape = extra_kv_rope.shape
            extra_kv_scales_shape = extra_kv_scales.shape
        else:
            extra_kv_nope, extra_kv_nope_shape = (
                prepare_sparse_mla_decode_strided_tensor(
                    "extra_kv_nope", extra_kv_nope, multiple=8
                )
            )
            extra_kv_rope, extra_kv_rope_shape = (
                prepare_sparse_mla_decode_strided_tensor(
                    "extra_kv_rope", extra_kv_rope, multiple=4
                )
            )
            extra_kv_scales, extra_kv_scales_shape = (
                prepare_sparse_mla_decode_strided_tensor(
                    "extra_kv_scales", extra_kv_scales, multiple=8
                )
            )
        extra_num_blocks, extra_page_block_bytes = extra_kv_nope_shape
        assert extra_kv_rope_shape == (extra_num_blocks, extra_page_block_bytes // 2)
        assert extra_kv_scales_shape == (extra_num_blocks, extra_page_block_bytes)
        extra_topk_length = validate_batch_lengths(
            extra_topk_length, batch, "extra_topk_length"
        )
    else:
        assert extra_topk_length is None, (
            "extra_topk_length requires extra cache inputs"
        )

    num_mp_parts = int(tile_scheduler_metadata.shape[0])
    support_split = num_mp_parts != 1
    runtime = prepare_scheduled_decode_runtime(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=d_v,
        topk_length=topk_length,
        attn_sink=attn_sink,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        out_dtype=q_dtype,
        device=device,
        variant_name="MODEL1",
        support_split=support_split,
    )

    address_tensors = [q, kv_nope, indices]
    if runtime.out_partial is not None:
        address_tensors.append(runtime.out_partial)
    if not _cache_strides_prepared:
        address_tensors.extend((kv_rope, kv_scales))
    if has_extra:
        address_tensors.append(extra_kv_nope)
        if not _cache_strides_prepared:
            address_tensors.extend((extra_kv_rope, extra_kv_scales))
        address_tensors.append(extra_indices)

    use_8byte_kv_loads = kv_rope.stride(0) % 8 != 0
    if extra_kv_rope is not None:
        use_8byte_kv_loads = use_8byte_kv_loads or extra_kv_rope.stride(0) % 8 != 0

    # Prepared cache views share one byte span; out_partial dominates aux outputs.
    kernel_factory = jit_for_tensor_addressing(
        sparse_attention_decode_fwd_scheduled_kernel_model1,
        *address_tensors,
    )
    kernel = kernel_factory(
        heads,
        d_v,
        has_extra=has_extra,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_m=block_m,
        block_i=block_i,
        threads=threads,
        consumer0_threads=consumer0_threads,
        consumer1_threads=consumer1_threads,
        producer_threads=producer_threads,
        max_nums_splits=runtime.max_nums_splits,
        has_attn_sink=runtime.has_attn_sink,
        has_topk_length=runtime.topk_length is not None,
        has_extra_topk_length=extra_topk_length is not None,
        page_block_size=page_block_size,
        extra_page_block_size=extra_page_block_size,
        support_split=support_split,
        use_8byte_kv_loads=use_8byte_kv_loads,
        use_int64_cosize=(
            kernel_factory is not sparse_attention_decode_fwd_scheduled_kernel_model1
        ),
    )
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()
    args = [q, kv_nope, kv_rope, kv_scales, indices]
    if runtime.topk_length is not None:
        args.append(runtime.topk_length)
    if has_extra:
        args.extend((extra_kv_nope, extra_kv_rope, extra_kv_scales, extra_indices))
        if extra_topk_length is not None:
            args.append(extra_topk_length)
    if runtime.attn_sink is not None:
        args.append(runtime.attn_sink)
    args.extend((tile_scheduler_metadata, num_splits))
    if support_split:
        assert runtime.glse is not None and runtime.out_partial is not None
        args.extend((runtime.glse, runtime.out_partial))
    args.extend((runtime.out, runtime.lse))
    kernel(*args)

    return runtime.out, runtime.lse
