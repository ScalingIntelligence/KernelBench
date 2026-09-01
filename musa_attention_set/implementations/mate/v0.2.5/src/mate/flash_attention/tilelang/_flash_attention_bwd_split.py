# ruff: noqa
# type: ignore
from ._flash_attention_bwd_common import (
    JIT_COMPILE_FLAGS,
    JIT_PASS_CONFIGS,
    T,
    cosize,
    perm_n,
    tilelang,
    _annotate_sqmma,
    _tilelang_dtype_nbytes,
)


@tilelang.jit(
    out_idx=[],
    pass_configs=JIT_PASS_CONFIGS,
    verbose=True,
    compile_flags=JIT_COMPILE_FLAGS,
)
# fmt: off
def flashattn_bwd_ws_split(
    dim,
    is_causal,
    has_window_left,
    has_window_right,
    has_softcap,
    is_varlen,
    is_bhsd=False,
    heads_q_eq_heads_kv=False,
    has_seqused_q=False,
    has_seqused_k=False,
    block_M=64,
    block_N=64,
    dtype="bfloat16",
    use_static_dims=False,
    recompute_delta=False,
):
    max_seq_kv = T.dynamic("max_seq_kv")
    log2e = 1.44269504
    batch = T.dynamic("batch")
    heads_q = T.dynamic("heads_q")
    heads_kv = heads_q if heads_q_eq_heads_kv else T.dynamic("heads_kv")
    total_seq_q = T.dynamic("total_seq_q")
    total_seq_kv = T.dynamic("total_seq_kv")
    max_seq_q = T.dynamic("max_seq_q")
    seqlen_q = T.dynamic("seqlen_q")
    seqlen_k = T.dynamic("seqlen_k")
    head_groups = 1 if heads_q_eq_heads_kv else heads_q // heads_kv
    safe_seq_check = True
    safe_dim_check = False
    accum_dtype = "float"
    kv_grad_dtype = dtype if heads_q_eq_heads_kv else accum_dtype
    dk_grad_heads = heads_kv if heads_q_eq_heads_kv else heads_q
    dv_grad_heads = heads_kv if heads_q_eq_heads_kv else heads_q
    L = block_N // 8  # 8 means 8 threads, L means Local Elem
    consumer0_threads = 256
    consumer1_threads = 256
    producer_threads = 128
    threads = consumer0_threads + consumer1_threads + producer_threads
    consumer1_start = consumer0_threads
    producer_start = consumer0_threads + consumer1_threads
    q_stride_b = T.dynamic("q_stride_b")
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    k_stride_b = T.dynamic("k_stride_b")
    k_stride_s = T.dynamic("k_stride_s")
    k_stride_h = T.dynamic("k_stride_h")
    v_stride_b = T.dynamic("v_stride_b")
    v_stride_s = T.dynamic("v_stride_s")
    v_stride_h = T.dynamic("v_stride_h")
    o_stride_b = T.dynamic("o_stride_b")
    o_stride_s = T.dynamic("o_stride_s")
    o_stride_h = T.dynamic("o_stride_h")
    do_stride_b = T.dynamic("do_stride_b")
    do_stride_s = T.dynamic("do_stride_s")
    do_stride_h = T.dynamic("do_stride_h")
    dk_stride_b = T.dynamic("dk_stride_b")
    dk_stride_s = T.dynamic("dk_stride_s")
    dk_stride_h = T.dynamic("dk_stride_h")
    dv_stride_b = T.dynamic("dv_stride_b")
    dv_stride_s = T.dynamic("dv_stride_s")
    dv_stride_h = T.dynamic("dv_stride_h")
    qk_dim = dim if use_static_dims else T.dynamic("qk_dim")
    v_dim = dim if use_static_dims else T.dynamic("v_dim")
    dk_dim = dim if use_static_dims else T.dynamic("dk_dim")
    dv_dim = dim if use_static_dims else T.dynamic("dv_dim")
    q_shape = (
        (total_seq_q, heads_q, qk_dim)
        if is_varlen
        else (batch, seqlen_q, heads_q, qk_dim)
    )
    k_shape = (
        (total_seq_kv, heads_kv, qk_dim)
        if is_varlen
        else (batch, seqlen_k, heads_kv, qk_dim)
    )
    v_shape = (
        (total_seq_kv, heads_kv, v_dim)
        if is_varlen
        else (batch, seqlen_k, heads_kv, v_dim)
    )
    o_shape = (
        (total_seq_q, heads_q, v_dim)
        if is_varlen
        else (batch, seqlen_q, heads_q, v_dim)
    )
    dk_shape = (
        (total_seq_kv, dk_grad_heads, dk_dim)
        if is_varlen
        else (batch, seqlen_k, dk_grad_heads, dk_dim)
    )
    dv_shape = (
        (total_seq_kv, dv_grad_heads, dv_dim)
        if is_varlen
        else (batch, seqlen_k, dv_grad_heads, dv_dim)
    )
    q_strides = (
        (q_stride_s, q_stride_h, 1)
        if is_varlen
        else (q_stride_b, q_stride_s, q_stride_h, 1)
    )
    k_strides = (
        (k_stride_s, k_stride_h, 1)
        if is_varlen
        else (k_stride_b, k_stride_s, k_stride_h, 1)
    )
    v_strides = (
        (v_stride_s, v_stride_h, 1)
        if is_varlen
        else (v_stride_b, v_stride_s, v_stride_h, 1)
    )
    o_strides = (
        (o_stride_s, o_stride_h, 1)
        if is_varlen
        else (o_stride_b, o_stride_s, o_stride_h, 1)
    )
    do_strides = (
        (do_stride_s, do_stride_h, 1)
        if is_varlen
        else (do_stride_b, do_stride_s, do_stride_h, 1)
    )
    q_type = T.StridedTensor(q_shape, q_strides, dtype)
    k_type = T.StridedTensor(k_shape, k_strides, dtype)
    v_type = T.StridedTensor(v_shape, v_strides, dtype)
    o_type = T.StridedTensor(o_shape, o_strides, dtype)
    do_type = T.StridedTensor(o_shape, do_strides, dtype)
    dk_strides = (
        (dk_stride_s, dk_stride_h, 1)
        if is_varlen
        else (dk_stride_b, dk_stride_s, dk_stride_h, 1)
    )
    dv_strides = (
        (dv_stride_s, dv_stride_h, 1)
        if is_varlen
        else (dv_stride_b, dv_stride_s, dv_stride_h, 1)
    )
    dk_type = T.StridedTensor(dk_shape, dk_strides, kv_grad_dtype)
    dv_type = T.StridedTensor(dv_shape, dv_strides, kv_grad_dtype)
    dtype_bytes = _tilelang_dtype_nbytes(dtype)
    k_robust_bytes = cosize(k_shape, k_strides) * dtype_bytes
    v_robust_bytes = cosize(v_shape, v_strides) * dtype_bytes

    def q_block(tensor, batch_idx, q_start, begin_seq, head, dim_start):
        if is_varlen:
            return tensor[
                q_start : q_start + block_M,
                head,
                dim_start : dim_start + dim // 2,
            ]
        local_q_start = q_start - begin_seq
        return tensor[
            batch_idx,
            local_q_start : local_q_start + block_M,
            head,
            dim_start : dim_start + dim // 2,
        ]

    def kv_block(tensor, batch_idx, kv_start, begin_seq, head, dim_start):
        if is_varlen:
            return tensor[
                kv_start : kv_start + block_N,
                head,
                dim_start : dim_start + dim // 2,
            ]
        local_kv_start = kv_start - begin_seq
        return tensor[
            batch_idx,
            local_kv_start : local_kv_start + block_N,
            head,
            dim_start : dim_start + dim // 2,
        ]

    def kv_elem(tensor, batch_idx, kv_idx, begin_seq, head, dim_idx):
        if is_varlen:
            return tensor[kv_idx, head, dim_idx]
        return tensor[batch_idx, kv_idx - begin_seq, head, dim_idx]

    def kv_tile_e(tensor, batch_idx, kv_offset, begin_seq, head, dim_idx):
        return kv_elem(
            tensor, batch_idx, begin_seq + kv_offset, begin_seq, head, dim_idx
        )

    def kv_tile_b(tensor, batch_idx, kv_offset, begin_seq, head, dim_start):
        return kv_block(
            tensor, batch_idx, begin_seq + kv_offset, begin_seq, head, dim_start
        )

    def permuted_kv_idx(begin_seq, block_start, idx):
        return begin_seq + block_start + perm_n(idx, block_N)

    def permuted_kv_valid(begin_seq, block_start, idx, end_seq):
        return permuted_kv_idx(begin_seq, block_start, idx) < end_seq

    @T.macro
    def store_dv_half(
        tensor,
        cast,
        batch_idx,
        head,
        block_start,
        block_end,
        begin_seq,
        end_seq,
        dim_offset,
        grad_dim,
    ):
        if (block_end + begin_seq) > end_seq:
            for i, j in T.Parallel(block_N, dim // 2):
                kv_offset = block_start + i
                if begin_seq + kv_offset < end_seq:
                    if j < grad_dim - dim_offset:
                        T.copy(
                            cast[i, j],
                            kv_tile_e(
                                tensor,
                                batch_idx,
                                kv_offset,
                                begin_seq,
                                head,
                                dim_offset + j,
                            ),
                        )
        else:
            for i, j in T.Parallel(block_N, dim // 2):
                kv_offset = block_start + i
                if j < grad_dim - dim_offset:
                    T.copy(
                        cast[i, j],
                        kv_tile_e(
                            tensor,
                            batch_idx,
                            kv_offset,
                            begin_seq,
                            head,
                            dim_offset + j,
                        ),
                    )

    @T.macro
    def store_grouped_dv_chunk(
        tensor,
        stage,
        batch_idx,
        head,
        block_start,
        block_end,
        begin_seq,
        end_seq,
        row_offset,
        dim_offset,
        grad_dim,
    ):
        if (block_end + begin_seq) > end_seq:
            for i, j in T.Parallel(block_N // 4, dim // 2):
                kv_offset = block_start + row_offset + i
                if begin_seq + kv_offset < end_seq:
                    if j < grad_dim - dim_offset:
                        T.copy(
                            stage[i, j],
                            kv_tile_e(
                                tensor,
                                batch_idx,
                                kv_offset,
                                begin_seq,
                                head,
                                dim_offset + j,
                            ),
                        )
        else:
            for i, j in T.Parallel(block_N // 4, dim // 2):
                kv_offset = block_start + row_offset + i
                if j < grad_dim - dim_offset:
                    T.copy(
                        stage[i, j],
                        kv_tile_e(
                            tensor,
                            batch_idx,
                            kv_offset,
                            begin_seq,
                            head,
                            dim_offset + j,
                        ),
                    )

    @T.macro
    def stage_and_store_grouped_dv_half(
        tensor,
        accum,
        stage,
        barrier,
        batch_idx,
        head,
        block_start,
        block_end,
        begin_seq,
        end_seq,
        dim_offset,
        grad_dim,
    ):
        for row_chunk in T.unroll(4, explicit=True):
            for i, j in T.Parallel(block_N // 4, dim // 2):
                T.copy(
                    accum[row_chunk * (block_N // 4) + i, j],
                    stage[i, j],
                )
            T.sync_warp()
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 0)
            store_grouped_dv_chunk(
                tensor,
                stage,
                batch_idx,
                head,
                block_start,
                block_end,
                begin_seq,
                end_seq,
                row_chunk * (block_N // 4),
                dim_offset,
                grad_dim,
            )
            T.sync_warp()
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 1)

    def lse_elem(tensor, batch_idx, q_idx, begin_seq, head):
        if is_varlen:
            return tensor[head, q_idx]
        return tensor[batch_idx, head, q_idx - begin_seq]

    @T.prim_func
    def flashattn_bwd_ws_kernel(
        Q: q_type,  # type: ignore
        K: k_type,  # type: ignore
        V: v_type,  # type: ignore
        Output: o_type,  # type: ignore
        dQ_accum: T.Tensor([batch, heads_q, max_seq_q, dim], accum_dtype),  # type: ignore
        dK: dk_type,  # type: ignore
        dV: dv_type,  # type: ignore
        dO: do_type,  # type: ignore
        cu_seq_q: T.Tensor([batch + 1], T.int32),  # type: ignore
        cu_seq_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
        seqused_q: T.Tensor([batch], T.int32),  # type: ignore
        seqused_k: T.Tensor([batch], T.int32),  # type: ignore
        Lse: T.Tensor(
            [heads_q, total_seq_q] if is_varlen else [batch, heads_q, seqlen_q],
            accum_dtype,
        ),  # type: ignore
        Delta: T.Tensor([total_seq_q, heads_q], accum_dtype),  # type: ignore
        max_seq_kv: T.int32,  # type: ignore
        window_size_left: T.int32,  # type: ignore
        window_size_right: T.int32,  # type: ignore
        softcap: T.float32,  # type: ignore
        smscale: T.float32,  # type: ignore
        rln2_scale: T.float32,  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(max_seq_kv, block_N),
            heads_q,
            batch,
            threads=threads,
        ) as (bx, by, bz):
            T.assume(qk_dim <= dim)
            T.assume(v_dim <= dim)
            T.assume(dk_dim <= dim)
            T.assume(dv_dim <= dim)
            T.assume(qk_dim % 8 == 0)
            T.assume(v_dim % 8 == 0)
            T.assume(dk_dim % 8 == 0)
            T.assume(dv_dim % 8 == 0)
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(k_stride_s % 8 == 0)
            T.assume(k_stride_h % 8 == 0)
            T.assume(v_stride_s % 8 == 0)
            T.assume(v_stride_h % 8 == 0)
            T.assume(o_stride_s % 8 == 0)
            T.assume(o_stride_h % 8 == 0)
            T.assume(do_stride_s % 8 == 0)
            T.assume(do_stride_h % 8 == 0)
            T.assume(dk_stride_s % 8 == 0)
            T.assume(dk_stride_h % 8 == 0)
            T.assume(dv_stride_s % 8 == 0)
            T.assume(dv_stride_h % 8 == 0)
            if not is_varlen:
                T.assume(q_stride_b % 8 == 0)
                T.assume(k_stride_b % 8 == 0)
                T.assume(v_stride_b % 8 == 0)
                T.assume(o_stride_b % 8 == 0)
                T.assume(do_stride_b % 8 == 0)
                T.assume(dk_stride_b % 8 == 0)
                T.assume(dv_stride_b % 8 == 0)
            begin_seq_q = T.alloc_var(T.int32)
            end_seq_q = T.alloc_var(T.int32)
            begin_seq_kv = T.alloc_var(T.int32)
            end_seq_kv = T.alloc_var(T.int32)
            kv_group = T.alloc_var(T.int32)
            kv_group = by if heads_q_eq_heads_kv else by // head_groups
            if is_varlen:
                begin_seq_q = cu_seq_q[bz]
                end_seq_q = cu_seq_q[bz + 1]
                begin_seq_kv = cu_seq_kv[bz]
                end_seq_kv = cu_seq_kv[bz + 1]
            else:
                begin_seq_q = bz * seqlen_q
                end_seq_q = begin_seq_q + seqlen_q
                begin_seq_kv = bz * seqlen_k
                end_seq_kv = begin_seq_kv + seqlen_k
            if has_seqused_q:
                end_seq_q = begin_seq_q + seqused_q[bz]
            if has_seqused_k:
                end_seq_kv = begin_seq_kv + seqused_k[bz]

            Qa_shared_0 = T.alloc_shared([block_M, dim // 2], dtype)
            dOb_shared_0 = Qa_shared_0
            # dOb_shared_0 = T.view(Qa_shared_0, [block_M, dim//2], dtype)
            Qa_shared_1 = T.alloc_shared([block_M, dim // 2], dtype)
            dOb_shared_1 = Qa_shared_1
            # dOb_shared_1 = T.view(Qa_shared_1, [block_M, dim//2], dtype)
            Qb_shared_0 = T.alloc_shared([block_M, dim // 2], dtype)
            dOa_shared_0 = Qb_shared_0
            # dOa_shared_0 = T.view(Qb_shared_0, [block_M, dim//2], dtype)
            Qb_shared_1 = T.alloc_shared([block_M, dim // 2], dtype)
            dOa_shared_1 = Qb_shared_1
            # dOa_shared_1 = T.view(Qb_shared_1, [block_M, dim//2], dtype)
            K_shared_0 = T.alloc_shared([block_N, dim // 2], dtype)
            K_shared_1 = T.alloc_shared([block_N, dim // 2], dtype)
            Kt_shared_0 = T.alloc_shared([block_N, dim // 2], dtype)
            Kt_shared_1 = T.alloc_shared([block_N, dim // 2], dtype)
            Vt_shared_0 = T.alloc_shared([block_N, dim // 2], dtype)
            Vt_shared_1 = T.alloc_shared([block_N, dim // 2], dtype)
            Pt_shared = T.alloc_shared([block_M, block_N], dtype)
            dSt_shared = T.alloc_shared([block_M, block_N], dtype)
            dS_shared = T.alloc_shared([block_M, block_N], dtype)
            if not heads_q_eq_heads_kv:
                dv_stage = T.alloc_shared([block_N // 4, dim // 2], accum_dtype)
                bar_dv_stage = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_k = T.alloc_barrier(arrive_count=producer_threads)
            bar_kt = T.alloc_barrier(arrive_count=producer_threads)
            bar_vt = T.alloc_barrier(arrive_count=producer_threads)
            bar_qa0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_qa1_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_qb0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_qb1_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_doa0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_doa1_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_dob0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_dob1_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_qa0_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_qa1_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_qb0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_qb1_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_doa0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_doa1_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_dob0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_dob1_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_p_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_p_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_ds_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_ds_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_dst_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_dst_free = T.alloc_barrier(arrive_count=consumer0_threads)

            T.sync_threads()
            if is_varlen:
                k_robust_desc = T.make_robust_desc(
                    T.address_of(K[0, 0, 0]), k_robust_bytes
                )
                v_robust_desc = T.make_robust_desc(
                    T.address_of(V[0, 0, 0]), v_robust_bytes
                )
            else:
                k_robust_desc = T.make_robust_desc(
                    T.address_of(K[0, 0, 0, 0]), k_robust_bytes
                )
                v_robust_desc = T.make_robust_desc(
                    T.address_of(V[0, 0, 0, 0]), v_robust_bytes
                )
            if is_varlen:
                lse_robust_desc = T.make_robust_desc(
                    T.address_of(Lse[0, 0]), (total_seq_q * heads_q * 4)
                )
            else:
                lse_robust_desc = T.make_robust_desc(
                    T.address_of(Lse[0, 0, 0]), (batch * seqlen_q * heads_q * 4)
                )
            delta_robust_desc = T.make_robust_desc(
                T.address_of(Delta[0, 0]), (total_seq_q * heads_q * 4)
            )

            tid = T.get_thread_binding()
            block_kv_start = bx * block_N
            block_kv_end = (bx + 1) * block_N
            if is_causal or has_window_left or has_window_right:
                local_seq_q = T.alloc_var(T.int32)
                local_seq_kv = T.alloc_var(T.int32)
                causal_offset = T.alloc_var(T.int32)
                local_seq_q = end_seq_q - begin_seq_q
                local_seq_kv = end_seq_kv - begin_seq_kv
                causal_offset = local_seq_kv - local_seq_q
            if is_causal:
                causal_q_local_start = T.alloc_var(T.int32)
                causal_q_local_mask_end = T.alloc_var(T.int32)
                causal_q_start = T.alloc_var(T.int32)
                causal_q_mask_end = T.alloc_var(T.int32)
                causal_q_local_start = block_kv_start - causal_offset
                causal_q_local_start = T.if_then_else(
                    causal_q_local_start > 0, causal_q_local_start, 0
                )
                causal_q_start = (
                    begin_seq_q + T.floordiv(causal_q_local_start, block_M) * block_M
                )
                causal_q_local_mask_end = block_kv_end - causal_offset
                causal_q_local_mask_end = T.if_then_else(
                    causal_q_local_mask_end > 0, causal_q_local_mask_end, 0
                )
                causal_q_mask_end = (
                    begin_seq_q + T.ceildiv(causal_q_local_mask_end, block_M) * block_M
                )
                causal_q_mask_end = T.if_then_else(
                    causal_q_mask_end < end_seq_q, causal_q_mask_end, end_seq_q
                )

            # if not block_kv_all_invalid:
            if (block_kv_start + begin_seq_kv) < end_seq_kv:
                if tid >= producer_start:
                    # tme load k
                    phase_producer = T.alloc_var(T.int32)
                    phase_producer = 0
                    T.copy(
                        kv_block(
                            K,
                            bz,
                            begin_seq_kv + block_kv_start,
                            begin_seq_kv,
                            kv_group,
                            0,
                        ),
                        K_shared_0,
                        barrier=bar_k,
                    )
                    T.copy(
                        kv_block(
                            K,
                            bz,
                            begin_seq_kv + block_kv_start,
                            begin_seq_kv,
                            kv_group,
                            dim // 2,
                        ),
                        K_shared_1,
                        barrier=bar_k,
                    )
                    T.barrier_arrive(bar_k)
                    # K_buffer debug hook for direct global-memory addressing.
                    for i, j in T.Parallel(block_N, dim // 2):
                        T.copy(
                            kv_elem(
                                K,
                                bz,
                                permuted_kv_idx(begin_seq_kv, block_kv_start, i),
                                begin_seq_kv,
                                kv_group,
                                j,
                            ),
                            Kt_shared_0[i, j],
                            force_async_copy=True,
                            src_robust_desc=k_robust_desc,
                        )
                    for i, j in T.Parallel(block_N, dim // 2):
                        T.copy(
                            kv_elem(
                                K,
                                bz,
                                permuted_kv_idx(begin_seq_kv, block_kv_start, i),
                                begin_seq_kv,
                                kv_group,
                                j + dim // 2,
                            ),
                            Kt_shared_1[i, j],
                            force_async_copy=True,
                            src_robust_desc=k_robust_desc,
                        )
                    for i, j in T.Parallel(block_N, dim // 2):
                        T.copy(
                            kv_elem(
                                V,
                                bz,
                                permuted_kv_idx(begin_seq_kv, block_kv_start, i),
                                begin_seq_kv,
                                kv_group,
                                j,
                            ),
                            Vt_shared_0[i, j],
                            force_async_copy=True,
                            src_robust_desc=v_robust_desc,
                        )
                    for i, j in T.Parallel(block_N, dim // 2):
                        T.copy(
                            kv_elem(
                                V,
                                bz,
                                permuted_kv_idx(begin_seq_kv, block_kv_start, i),
                                begin_seq_kv,
                                kv_group,
                                j + dim // 2,
                            ),
                            Vt_shared_1[i, j],
                            force_async_copy=True,
                            src_robust_desc=v_robust_desc,
                        )
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    T.barrier_arrive(bar_kt)
                    T.barrier_arrive(bar_vt)
                    for q_start in range(
                        causal_q_start if is_causal else begin_seq_q, end_seq_q, block_M
                    ):
                        T.barrier_wait(bar_qa0_free, phase_producer ^ 1)
                        _annotate_sqmma(Qa_shared_0, k_major=True)
                        T.copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, 0),
                            Qa_shared_0,
                            barrier=bar_qa0_ready,
                        )
                        T.barrier_arrive(bar_qa0_ready)

                        T.barrier_wait(bar_qa1_free, phase_producer ^ 1)
                        _annotate_sqmma(Qa_shared_1, k_major=True)
                        T.copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, dim // 2),
                            Qa_shared_1,
                            barrier=bar_qa1_ready,
                        )
                        T.barrier_arrive(bar_qa1_ready)

                        T.barrier_wait(bar_doa0_free, phase_producer ^ 1)
                        _annotate_sqmma(dOa_shared_0, k_major=True)
                        T.copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, 0),
                            dOa_shared_0,
                            barrier=bar_doa0_ready,
                        )
                        T.barrier_arrive(bar_doa0_ready)

                        T.barrier_wait(bar_doa1_free, phase_producer ^ 1)
                        _annotate_sqmma(dOa_shared_1, k_major=True)
                        T.copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, dim // 2),
                            dOa_shared_1,
                            barrier=bar_doa1_ready,
                        )
                        T.barrier_arrive(bar_doa1_ready)

                        T.barrier_wait(bar_dob0_free, phase_producer)
                        _annotate_sqmma(
                            dOb_shared_0, k_major=False, continuity=dim // 4
                        )
                        T.copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, 0),
                            dOb_shared_0,
                            barrier=bar_dob0_ready,
                        )
                        T.barrier_arrive(bar_dob0_ready)

                        T.barrier_wait(bar_dob1_free, phase_producer)
                        _annotate_sqmma(
                            dOb_shared_1, k_major=False, continuity=dim // 4
                        )
                        T.copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, dim // 2),
                            dOb_shared_1,
                            barrier=bar_dob1_ready,
                        )
                        T.barrier_arrive(bar_dob1_ready)

                        T.barrier_wait(bar_qb0_free, phase_producer)
                        _annotate_sqmma(Qb_shared_0, k_major=False, continuity=dim // 4)
                        T.copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, 0),
                            Qb_shared_0,
                            barrier=bar_qb0_ready,
                        )
                        T.barrier_arrive(bar_qb0_ready)

                        T.barrier_wait(bar_qb1_free, phase_producer)
                        _annotate_sqmma(Qb_shared_1, k_major=False, continuity=dim // 4)
                        T.copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, dim // 2),
                            Qb_shared_1,
                            barrier=bar_qb1_ready,
                        )
                        T.barrier_arrive(bar_qb1_ready)
                        T.sync_threads(1, producer_threads)
                        phase_producer = phase_producer ^ 1
                elif tid >= consumer1_start:
                    # P*dOa -> dV
                    # ds*K -> dQ
                    phase_consumer1 = T.alloc_var(T.int32)
                    phase_consumer1 = 0
                    T.barrier_wait(bar_k, 0)
                    T.barrier_wait(bar_kt, 0)
                    T.barrier_wait(bar_vt, 0)
                    dv_accum_0 = T.alloc_fragment([block_N, dim // 2], accum_dtype)
                    dq_accum = T.alloc_fragment([block_M, dim // 2], accum_dtype)
                    dv_accum_1 = T.alloc_fragment([block_N, dim // 2], accum_dtype)
                    if heads_q_eq_heads_kv:
                        dv_cast = T.alloc_fragment([block_N, dim // 2], dtype)
                    T.fill(dv_accum_0, 0.0)
                    T.fill(dv_accum_1, 0.0)
                    dq_loop_start = causal_q_start if is_causal else begin_seq_q
                    for q_start in range(dq_loop_start, end_seq_q, block_M):
                        # is_last_q_tile = (q_start+block_M) >= end_seq_q
                        T.barrier_wait(bar_p_ready, phase_consumer1)
                        T.barrier_wait(bar_dob0_ready, phase_consumer1)
                        _annotate_sqmma(Pt_shared, k_major=False, continuity=block_N)
                        _annotate_sqmma(
                            dOb_shared_0, k_major=False, continuity=dim // 4
                        )
                        T.gemm(
                            Pt_shared,
                            dOb_shared_0,
                            dv_accum_0,
                            clear_accum=False,
                            wg_wait=0,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.barrier_wait(bar_dob1_ready, phase_consumer1)
                        _annotate_sqmma(
                            dOb_shared_1, k_major=False, continuity=dim // 4
                        )
                        T.gemm(
                            Pt_shared,
                            dOb_shared_1,
                            dv_accum_1,
                            clear_accum=False,
                            wg_wait=0,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_qa0_free)
                        T.barrier_arrive(bar_qa1_free)
                        T.barrier_arrive(bar_p_free)

                        T.barrier_wait(bar_ds_ready, phase_consumer1)
                        _annotate_sqmma(dS_shared, k_major=True)
                        T.gemm(
                            dS_shared,
                            K_shared_0,
                            dq_accum,
                            wg_wait=0,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.wait_wgmma(0)
                        local_q_start = q_start - begin_seq_q
                        for i, j in T.Parallel(block_M, dim // 2):
                            T.atomic_add(
                                dQ_accum[bz, by, local_q_start + i, j],
                                dq_accum[i, j],
                            )
                        T.gemm(
                            dS_shared,
                            K_shared_1,
                            dq_accum,
                            wg_wait=0,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_ds_free)
                        for i, j in T.Parallel(block_M, dim // 2):
                            T.atomic_add(
                                dQ_accum[
                                    bz,
                                    by,
                                    local_q_start + i,
                                    dim // 2 + j,
                                ],
                                dq_accum[i, j],
                            )
                        T.sync_threads(1, consumer1_threads)
                        phase_consumer1 = phase_consumer1 ^ 1
                    if heads_q_eq_heads_kv:
                        T.copy(dv_accum_0, dv_cast)
                        store_dv_half(
                            dV,
                            dv_cast,
                            bz,
                            by,
                            block_kv_start,
                            block_kv_end,
                            begin_seq_kv,
                            end_seq_kv,
                            0,
                            dv_dim,
                        )
                        T.copy(dv_accum_1, dv_cast)
                        store_dv_half(
                            dV,
                            dv_cast,
                            bz,
                            by,
                            block_kv_start,
                            block_kv_end,
                            begin_seq_kv,
                            end_seq_kv,
                            dim // 2,
                            dv_dim,
                        )
                    else:
                        stage_and_store_grouped_dv_half(
                            dV,
                            dv_accum_0,
                            dv_stage,
                            bar_dv_stage,
                            bz,
                            by,
                            block_kv_start,
                            block_kv_end,
                            begin_seq_kv,
                            end_seq_kv,
                            0,
                            dv_dim,
                        )
                        stage_and_store_grouped_dv_half(
                            dV,
                            dv_accum_1,
                            dv_stage,
                            bar_dv_stage,
                            bz,
                            by,
                            block_kv_start,
                            block_kv_end,
                            begin_seq_kv,
                            end_seq_kv,
                            dim // 2,
                            dv_dim,
                        )
                else:
                    # Qa * Kt -> s
                    # dOb * Vt -> dp
                    # dst * Qb -> dK
                    phase_consumer0 = T.alloc_var(T.int32)
                    phase_consumer0 = 0
                    dk_accum_0 = T.alloc_fragment([block_N, dim // 2], accum_dtype)
                    dk_accum_1 = T.alloc_fragment([block_N, dim // 2], accum_dtype)
                    accs_accum = T.alloc_fragment([block_M, block_N], accum_dtype)
                    dp_accum = T.alloc_fragment([block_M, block_N], accum_dtype)
                    accs_cast = T.alloc_fragment([block_M, block_N], dtype)
                    lse_buffer = T.alloc_fragment([block_M], accum_dtype)
                    delta_buffer = T.alloc_fragment([block_M], accum_dtype)
                    T.barrier_wait(bar_k, 0)
                    T.barrier_wait(bar_kt, 0)
                    T.barrier_wait(bar_vt, 0)
                    T.fill(dk_accum_0, 0.0)
                    T.fill(dk_accum_1, 0.0)
                    dk_cast_0 = T.alloc_fragment([block_N, dim // 2], dtype)
                    dk_cast_1 = T.alloc_fragment([block_N, dim // 2], dtype)
                    for q_start in range(
                        causal_q_start if is_causal else begin_seq_q, end_seq_q, block_M
                    ):
                        # is_last_q_tile = (q_start+block_M) >= end_seq_q
                        T.barrier_wait(bar_qa0_ready, phase_consumer0)
                        _annotate_sqmma(Qa_shared_0, k_major=True)
                        T.gemm(
                            Qa_shared_0,
                            Kt_shared_0,
                            accs_accum,
                            clear_accum=True,
                            wg_wait=0,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.barrier_wait(bar_qa1_ready, phase_consumer0)
                        _annotate_sqmma(Qa_shared_1, k_major=True)
                        T.gemm(
                            Qa_shared_1,
                            Kt_shared_1,
                            accs_accum,
                            wg_wait=0,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_dob0_free)
                        T.barrier_arrive(bar_dob1_free)
                        T.barrier_wait(bar_doa0_ready, phase_consumer0)
                        _annotate_sqmma(dOa_shared_0, k_major=True)
                        T.gemm(
                            dOa_shared_0,
                            Vt_shared_0,
                            dp_accum,
                            clear_accum=True,
                            wg_wait=-1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.barrier_wait(bar_doa1_ready, phase_consumer0)
                        _annotate_sqmma(dOa_shared_1, k_major=True)
                        T.gemm(
                            dOa_shared_1,
                            Vt_shared_1,
                            dp_accum,
                            wg_wait=-1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.wait_wgmma(0)
                        T.barrier_wait(bar_dob0_ready, phase_consumer0)
                        for i in T.Parallel(block_M):
                            T.copy(
                                lse_elem(Lse, bz, q_start + i, begin_seq_q, by),
                                lse_buffer[i],
                                src_robust_desc=lse_robust_desc,
                            )
                        for i in T.Parallel(block_M):
                            lse_buffer[i] = T.if_then_else(
                                T.isfinite(lse_buffer[i]), lse_buffer[i], 2**30
                            )
                        # if is_last_q -> check oob_m
                        if (q_start + block_M) >= end_seq_q:
                            for i in T.Parallel(block_M):
                                lse_buffer[i] = T.if_then_else(
                                    q_start + i < end_seq_q, lse_buffer[i], 2**30
                                )
                            # for i, j in T.Parallel(block_M, block_N):
                            #     accs_accum[i, j] = T.if_then_else(
                            #                                     -2**30)
                        if (block_kv_end + begin_seq_kv) > end_seq_kv:
                            for i, j in T.Parallel(block_M, block_N):
                                accs_accum[i, j] = T.if_then_else(
                                    permuted_kv_valid(
                                        begin_seq_kv, block_kv_start, j, end_seq_kv
                                    ),
                                    accs_accum[i, j],
                                    -(2**30),
                                )

                        if is_causal:
                            if q_start < causal_q_mask_end:
                                for i, j in T.Parallel(block_M, block_N):
                                    q_idx = q_start - begin_seq_q + i
                                    k_idx = bx * block_N + perm_n(j, block_N)
                                    valid = q_idx + causal_offset >= k_idx
                                    accs_accum[i, j] = T.if_then_else(
                                        valid, accs_accum[i, j], -(2**30)
                                    )
                        if has_window_left or has_window_right:
                            for i, j in T.Parallel(block_M, block_N):
                                q_idx = q_start - begin_seq_q + i
                                k_idx = bx * block_N + perm_n(j, block_N)
                                valid = (
                                    not has_window_left
                                    or k_idx >= q_idx + causal_offset - window_size_left
                                ) and (
                                    not has_window_right
                                    or k_idx
                                    <= q_idx + causal_offset + window_size_right
                                )
                                accs_accum[i, j] = T.if_then_else(
                                    valid, accs_accum[i, j], -(2**30)
                                )
                        if has_softcap:
                            if recompute_delta:
                                for i, j in T.Parallel(block_M, block_N):
                                    softcap_tanh = T.tanh(
                                        accs_accum[i, j] * smscale / softcap
                                    )
                                    accs_cast[i, j] = 1.0 - softcap_tanh * softcap_tanh
                                    accs_accum[i, j] = T.exp2(
                                        softcap_tanh * softcap * log2e
                                        - lse_buffer[i] * log2e
                                    )
                                if (block_kv_end + begin_seq_kv) > end_seq_kv:
                                    for i, j in T.Parallel(block_M, block_N):
                                        accs_accum[i, j] = T.if_then_else(
                                            permuted_kv_valid(
                                                begin_seq_kv,
                                                block_kv_start,
                                                j,
                                                end_seq_kv,
                                            ),
                                            accs_accum[i, j],
                                            0.0,
                                        )
                                if is_causal:
                                    if q_start < causal_q_mask_end:
                                        for i, j in T.Parallel(block_M, block_N):
                                            q_idx = q_start - begin_seq_q + i
                                            k_idx = bx * block_N + perm_n(j, block_N)
                                            accs_accum[i, j] = T.if_then_else(
                                                q_idx + causal_offset >= k_idx,
                                                accs_accum[i, j],
                                                0.0,
                                            )
                                if has_window_left or has_window_right:
                                    for i, j in T.Parallel(block_M, block_N):
                                        q_idx = q_start - begin_seq_q + i
                                        k_idx = bx * block_N + perm_n(j, block_N)
                                        valid = (
                                            not has_window_left
                                            or k_idx
                                            >= q_idx + causal_offset - window_size_left
                                        ) and (
                                            not has_window_right
                                            or k_idx
                                            <= q_idx + causal_offset + window_size_right
                                        )
                                        accs_accum[i, j] = T.if_then_else(
                                            valid, accs_accum[i, j], 0.0
                                        )
                                if (q_start + block_M) >= end_seq_q:
                                    for i, j in T.Parallel(block_M, block_N):
                                        accs_accum[i, j] = T.if_then_else(
                                            q_start + i < end_seq_q,
                                            accs_accum[i, j],
                                            0.0,
                                        )
                            else:
                                for i in T.Parallel(block_M):
                                    T.copy(
                                        Delta[q_start + i, by],
                                        delta_buffer[i],
                                        src_robust_desc=delta_robust_desc,
                                    )
                                if (q_start + block_M) >= end_seq_q:
                                    for i in T.Parallel(block_M):
                                        delta_buffer[i] = T.if_then_else(
                                            q_start + i < end_seq_q,
                                            delta_buffer[i],
                                            0.0,
                                        )
                                for i, j in T.Parallel(block_M, block_N):
                                    q_idx = q_start - begin_seq_q + i
                                    k_idx = bx * block_N + perm_n(j, block_N)
                                    valid_0 = True
                                    if is_causal:
                                        valid_1 = q_start >= causal_q_mask_end or (
                                            q_idx + causal_offset >= k_idx
                                        )
                                    else:
                                        valid_1 = valid_0
                                    if has_window_left:
                                        valid_2 = valid_1 and (
                                            k_idx
                                            >= q_idx + causal_offset - window_size_left
                                        )
                                    else:
                                        valid_2 = valid_1
                                    if has_window_right:
                                        valid_3 = valid_2 and (
                                            k_idx
                                            <= q_idx + causal_offset + window_size_right
                                        )
                                    else:
                                        valid_3 = valid_2
                                    valid_4 = valid_3 and T.if_then_else(
                                        (block_kv_end + begin_seq_kv) > end_seq_kv,
                                        permuted_kv_valid(
                                            begin_seq_kv,
                                            block_kv_start,
                                            j,
                                            end_seq_kv,
                                        ),
                                        True,
                                    )
                                    valid = valid_4 and T.if_then_else(
                                        (q_start + block_M) >= end_seq_q,
                                        q_start + i < end_seq_q,
                                        True,
                                    )
                                    softcap_tanh = T.tanh(
                                        accs_accum[i, j] * smscale / softcap
                                    )
                                    dp_accum[i, j] = (
                                        (dp_accum[i, j] - delta_buffer[i])
                                        * smscale
                                        * (1.0 - softcap_tanh * softcap_tanh)
                                    )
                                    accs_accum[i, j] = T.if_then_else(
                                        valid,
                                        T.exp2(
                                            softcap_tanh * softcap * log2e
                                            - lse_buffer[i] * log2e
                                        ),
                                        0.0,
                                    )
                        else:
                            for i, j in T.Parallel(block_M, block_N):
                                accs_accum[i, j] = accs_accum[i, j] * rln2_scale
                            for i, j in T.Parallel(block_M, block_N):
                                accs_accum[i, j] = T.exp2(
                                    accs_accum[i, j] - lse_buffer[i] * log2e
                                )
                        # Avoid forward-output quantization in Delta when this
                        # tile contains the sequence's complete probability row.
                        if recompute_delta:
                            for i, j in T.Parallel(block_M, block_N):
                                dp_accum[i, j] = dp_accum[i, j] * accs_accum[i, j]
                            T.reduce_sum(dp_accum, delta_buffer, dim=1)
                        if not (has_softcap and recompute_delta):
                            T.copy(accs_accum, accs_cast)
                        T.barrier_wait(bar_p_free, phase_consumer0 ^ 1)
                        for i, t in T.Parallel(block_M, 8):
                            base = t * L
                            for l in T.vectorized(L):
                                if has_softcap and recompute_delta:
                                    Pt_shared[i, base + l] = accs_accum[i, l * 8 + t]
                                else:
                                    Pt_shared[i, base + l] = accs_cast[i, l * 8 + t]
                        T.sync_warp()
                        T.barrier_arrive(bar_p_ready)
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_qb0_free)
                        T.barrier_arrive(bar_qb1_free)
                        if not has_softcap and not recompute_delta:
                            for i in T.Parallel(block_M):
                                T.copy(
                                    Delta[q_start + i, by],
                                    delta_buffer[i],
                                    src_robust_desc=delta_robust_desc,
                                )
                        if not has_softcap and (
                            (block_kv_end + begin_seq_kv) > end_seq_kv
                        ):
                            for i, j in T.Parallel(block_M, block_N):
                                dp_accum[i, j] = T.if_then_else(
                                    permuted_kv_valid(
                                        begin_seq_kv, block_kv_start, j, end_seq_kv
                                    ),
                                    dp_accum[i, j],
                                    0.0,
                                )
                        # if oob_m
                        if not has_softcap and (q_start + block_M) >= end_seq_q:
                            for i in T.Parallel(block_M):
                                delta_buffer[i] = T.if_then_else(
                                    q_start + i < end_seq_q, delta_buffer[i], 0.0
                                )
                        if has_softcap:
                            if recompute_delta:
                                for i, j in T.Parallel(block_M, block_N):
                                    dp_accum[i, j] = (
                                        (
                                            dp_accum[i, j]
                                            - delta_buffer[i] * accs_accum[i, j]
                                        )
                                        * smscale
                                        * accs_cast[i, j]
                                    )
                            else:
                                for i, j in T.Parallel(block_M, block_N):
                                    dp_accum[i, j] = dp_accum[i, j] * accs_accum[i, j]
                        else:
                            if recompute_delta:
                                for i, j in T.Parallel(block_M, block_N):
                                    dp_accum[i, j] = (
                                        dp_accum[i, j]
                                        - delta_buffer[i] * accs_accum[i, j]
                                    ) * smscale
                            else:
                                for i, j in T.Parallel(block_M, block_N):
                                    dp_accum[i, j] = (
                                        (dp_accum[i, j] - delta_buffer[i])
                                        * accs_accum[i, j]
                                        * smscale
                                    )
                        if not has_softcap and (q_start + block_M) >= end_seq_q:
                            for i, j in T.Parallel(block_M, block_N):
                                dp_accum[i, j] = T.if_then_else(
                                    q_start + i < end_seq_q, dp_accum[i, j], 0.0
                                )
                        T.barrier_wait(bar_ds_free, phase_consumer0 ^ 1)
                        T.copy(dp_accum, accs_cast)
                        for i, t in T.Parallel(block_M, 8):
                            base = t * L
                            for l in T.vectorized(L):
                                dS_shared[i, base + l] = accs_cast[i, l * 8 + t]
                        T.sync_warp()
                        T.barrier_arrive(bar_ds_ready)
                        T.barrier_wait(bar_dst_free, phase_consumer0 ^ 1)
                        for i, t in T.Parallel(block_M, 8):
                            base = t * L
                            for l in T.vectorized(L):
                                dSt_shared[i, base + l] = accs_cast[i, l * 8 + t]
                        T.sync_warp()
                        T.barrier_wait(bar_qb0_ready, phase_consumer0)
                        T.barrier_arrive(bar_dst_ready)
                        T.barrier_wait(bar_dst_ready, phase_consumer0)
                        _annotate_sqmma(dSt_shared, k_major=False, continuity=block_N)
                        _annotate_sqmma(Qb_shared_0, k_major=False, continuity=dim // 4)
                        T.gemm(
                            dSt_shared,
                            Qb_shared_0,
                            dk_accum_0,
                            wg_wait=0,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.barrier_wait(bar_qb1_ready, phase_consumer0)
                        _annotate_sqmma(Qb_shared_1, k_major=False, continuity=dim // 4)
                        T.gemm(
                            dSt_shared,
                            Qb_shared_1,
                            dk_accum_1,
                            wg_wait=0,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_doa0_free)
                        T.barrier_arrive(bar_doa1_free)
                        T.barrier_arrive(bar_dst_free)
                        T.sync_threads(1, consumer0_threads)
                        phase_consumer0 = phase_consumer0 ^ 1
                    if heads_q_eq_heads_kv:
                        T.copy(dk_accum_0, dk_cast_0)
                        T.copy(dk_accum_1, dk_cast_1)
                    if (block_kv_end + begin_seq_kv) > end_seq_kv:
                        for i, j in T.Parallel(block_N, dim // 2):
                            kv_offset = block_kv_start + i
                            if begin_seq_kv + kv_offset < end_seq_kv:
                                if heads_q_eq_heads_kv:
                                    if j < dk_dim:
                                        T.copy(
                                            dk_cast_0[i, j],
                                            kv_tile_e(
                                                dK, bz, kv_offset, begin_seq_kv, by, j
                                            ),
                                        )
                                    if dim // 2 + j < dk_dim:
                                        T.copy(
                                            dk_cast_1[i, j],
                                            kv_tile_e(
                                                dK,
                                                bz,
                                                kv_offset,
                                                begin_seq_kv,
                                                by,
                                                dim // 2 + j,
                                            ),
                                        )
                                else:
                                    if j < dk_dim:
                                        T.copy(
                                            dk_accum_0[i, j],
                                            kv_tile_e(
                                                dK, bz, kv_offset, begin_seq_kv, by, j
                                            ),
                                        )
                                    if dim // 2 + j < dk_dim:
                                        T.copy(
                                            dk_accum_1[i, j],
                                            kv_tile_e(
                                                dK,
                                                bz,
                                                kv_offset,
                                                begin_seq_kv,
                                                by,
                                                dim // 2 + j,
                                            ),
                                        )
                    else:
                        for i, j in T.Parallel(block_N, dim // 2):
                            kv_offset = block_kv_start + i
                            if heads_q_eq_heads_kv:
                                if j < dk_dim:
                                    T.copy(
                                        dk_cast_0[i, j],
                                        kv_tile_e(
                                            dK, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                                if dim // 2 + j < dk_dim:
                                    T.copy(
                                        dk_cast_1[i, j],
                                        kv_tile_e(
                                            dK,
                                            bz,
                                            kv_offset,
                                            begin_seq_kv,
                                            by,
                                            dim // 2 + j,
                                        ),
                                    )
                            else:
                                if j < dk_dim:
                                    T.copy(
                                        dk_accum_0[i, j],
                                        kv_tile_e(
                                            dK, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                                if dim // 2 + j < dk_dim:
                                    T.copy(
                                        dk_accum_1[i, j],
                                        kv_tile_e(
                                            dK,
                                            bz,
                                            kv_offset,
                                            begin_seq_kv,
                                            by,
                                            dim // 2 + j,
                                        ),
                                    )

    # if heads_kv == heads_q -> output dk dv,
    # if heads_kv != heads_q -> output dk_accum dv_accum， -> reduce to dk dv
    return flashattn_bwd_ws_kernel


# fmt: on
