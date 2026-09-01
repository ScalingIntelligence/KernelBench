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
# fmt: on


@tilelang.jit(
    out_idx=[],
    pass_configs=JIT_PASS_CONFIGS,
    verbose=True,
    compile_flags=JIT_COMPILE_FLAGS,
)
# fmt: off
def flashattn_bwd_ws_split_dq(
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
    block_M=128,
    block_N=64,
    dtype="bfloat16",
    use_static_dims=False,
):
    if dim not in (128, 256):
        raise ValueError("flashattn_bwd_ws_split_dq only supports dim == 128 or 256")

    producer_threads = 128
    consumer0_threads = 256
    consumer1_threads = 256
    threads = producer_threads + consumer0_threads + consumer1_threads
    log2e = 1.44269504
    L = block_N // 8
    batch = T.dynamic("batch")
    heads_q = T.dynamic("heads_q")
    heads_kv = heads_q if heads_q_eq_heads_kv else T.dynamic("heads_kv")
    total_seq_q = T.dynamic("total_seq_q")
    total_seq_kv = T.dynamic("total_seq_kv")
    max_seq_q = T.dynamic("max_seq_q")
    seqlen_q = T.dynamic("seqlen_q")
    seqlen_k = T.dynamic("seqlen_k")
    head_groups = 1 if heads_q_eq_heads_kv else heads_q // heads_kv
    accum_dtype = "float"
    q_stride_b = T.dynamic("q_stride_b")
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    k_stride_b = T.dynamic("k_stride_b")
    k_stride_s = T.dynamic("k_stride_s")
    k_stride_h = T.dynamic("k_stride_h")
    v_stride_b = T.dynamic("v_stride_b")
    v_stride_s = T.dynamic("v_stride_s")
    v_stride_h = T.dynamic("v_stride_h")
    do_stride_b = T.dynamic("do_stride_b")
    do_stride_s = T.dynamic("do_stride_s")
    do_stride_h = T.dynamic("do_stride_h")
    dq_stride_b = T.dynamic("dq_stride_b")
    dq_stride_s = T.dynamic("dq_stride_s")
    dq_stride_h = T.dynamic("dq_stride_h")
    qk_dim = dim if use_static_dims else T.dynamic("qk_dim")
    v_dim = dim if use_static_dims else T.dynamic("v_dim")
    dq_dim = dim if use_static_dims else T.dynamic("dq_dim")
    q_shape = (
        (total_seq_q, heads_q, qk_dim)
        if is_varlen
        else (batch, seqlen_q, heads_q, qk_dim)
    )
    dq_shape = (
        (total_seq_q, heads_q, dq_dim)
        if is_varlen
        else (batch, seqlen_q, heads_q, dq_dim)
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
    do_shape = (
        (total_seq_q, heads_q, v_dim)
        if is_varlen
        else (batch, seqlen_q, heads_q, v_dim)
    )
    q_strides = (
        (q_stride_s, q_stride_h, 1)
        if is_varlen
        else (q_stride_b, q_stride_s, q_stride_h, 1)
    )
    dq_strides = (
        (dq_stride_s, dq_stride_h, 1)
        if is_varlen
        else (dq_stride_b, dq_stride_s, dq_stride_h, 1)
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
    do_strides = (
        (do_stride_s, do_stride_h, 1)
        if is_varlen
        else (do_stride_b, do_stride_s, do_stride_h, 1)
    )
    q_type = T.StridedTensor(q_shape, q_strides, dtype)
    dq_type = T.StridedTensor(dq_shape, dq_strides, dtype)
    k_type = T.StridedTensor(k_shape, k_strides, dtype)
    v_type = T.StridedTensor(v_shape, v_strides, dtype)
    do_type = T.StridedTensor(do_shape, do_strides, dtype)
    dtype_bytes = _tilelang_dtype_nbytes(dtype)
    k_robust_bytes = cosize(k_shape, k_strides) * dtype_bytes

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

    def dq_elem(tensor, batch_idx, q_idx, begin_seq, head, dim_idx):
        if is_varlen:
            return tensor[q_idx, head, dim_idx]
        return tensor[batch_idx, q_idx - begin_seq, head, dim_idx]

    def lse_elem(tensor, batch_idx, q_idx, begin_seq, head):
        if is_varlen:
            return tensor[head, q_idx]
        return tensor[batch_idx, head, q_idx - begin_seq]

    @T.prim_func
    def flashattn_bwd_ws_split_dq_kernel(
        Q: q_type,  # type: ignore
        K: k_type,  # type: ignore
        V: v_type,  # type: ignore
        dQ: dq_type,  # type: ignore
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
        max_seq_q: T.int32,  # type: ignore
        window_size_left: T.int32,  # type: ignore
        window_size_right: T.int32,  # type: ignore
        softcap: T.float32,  # type: ignore
        smscale: T.float32,  # type: ignore
        rln2_scale: T.float32,  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(max_seq_q, block_M),
            heads_q,
            batch,
            threads=threads,
        ) as (bx, by, bz):
            T.assume(qk_dim <= dim)
            T.assume(v_dim <= dim)
            T.assume(dq_dim <= dim)
            T.assume(qk_dim % 8 == 0)
            T.assume(v_dim % 8 == 0)
            T.assume(dq_dim % 8 == 0)
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(k_stride_s % 8 == 0)
            T.assume(k_stride_h % 8 == 0)
            T.assume(v_stride_s % 8 == 0)
            T.assume(v_stride_h % 8 == 0)
            T.assume(do_stride_s % 8 == 0)
            T.assume(do_stride_h % 8 == 0)
            T.assume(dq_stride_s % 8 == 0)
            T.assume(dq_stride_h % 8 == 0)
            if not is_varlen:
                T.assume(q_stride_b % 8 == 0)
                T.assume(k_stride_b % 8 == 0)
                T.assume(v_stride_b % 8 == 0)
                T.assume(do_stride_b % 8 == 0)
                T.assume(dq_stride_b % 8 == 0)

            if is_varlen:
                k_robust_desc = T.make_robust_desc(
                    T.address_of(K[0, 0, 0]), k_robust_bytes
                )
            else:
                k_robust_desc = T.make_robust_desc(
                    T.address_of(K[0, 0, 0, 0]), k_robust_bytes
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

            bar_q_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_do_ready = T.alloc_barrier(arrive_count=producer_threads)

            bar_kt0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kt1_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_vt0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_vt1_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_k0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_k1_ready = T.alloc_barrier(arrive_count=consumer1_threads)

            bar_kt0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_kt1_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vt0_c0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vt0_producer_free = T.alloc_barrier(arrive_count=producer_threads)
            bar_vt1_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_k0_c0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vt0_c1_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_k1_free = T.alloc_barrier(arrive_count=consumer1_threads)

            bar_ds_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_ds_free = T.alloc_barrier(arrive_count=consumer1_threads)

            T.sync_threads()

            Qa_shared_0 = T.alloc_shared([block_M, dim // 2], dtype)
            Qa_shared_1 = T.alloc_shared([block_M, dim // 2], dtype)
            dOa_shared_0 = T.alloc_shared([block_M, dim // 2], dtype)
            dOa_shared_1 = T.alloc_shared([block_M, dim // 2], dtype)

            shmem_buffer0 = T.alloc_shared([block_N, dim // 2], dtype)
            shmem_buffer1 = T.alloc_shared([block_N, dim // 2], dtype)
            shmem_buffer2 = T.alloc_shared([block_M, block_N], dtype)
            shmem_buffer3 = T.alloc_shared([block_N, dim // 2], dtype)

            Kt_shared_0 = shmem_buffer0
            Kt_shared_1 = shmem_buffer1
            Vt_shared_0 = shmem_buffer0
            Vt_shared_1 = shmem_buffer1
            K_shared_0 = shmem_buffer0
            K_shared_1 = shmem_buffer3
            ds_shared = shmem_buffer2

            begin_seq_q = T.alloc_var(T.int32)
            end_seq_q = T.alloc_var(T.int32)
            begin_seq_kv = T.alloc_var(T.int32)
            end_seq_kv = T.alloc_var(T.int32)
            local_seq_q = T.alloc_var(T.int32)
            local_seq_kv = T.alloc_var(T.int32)
            causal_offset = T.alloc_var(T.int32)
            kv_loop_start = T.alloc_var(T.int32)
            kv_loop_end = T.alloc_var(T.int32)
            causal_kv_local_end = T.alloc_var(T.int32)
            block_q_start = T.alloc_var(T.int32)
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
            block_q_start = begin_seq_q + bx * block_M

            local_seq_q = end_seq_q - begin_seq_q
            local_seq_kv = end_seq_kv - begin_seq_kv
            causal_offset = local_seq_kv - local_seq_q
            kv_loop_start = begin_seq_kv
            kv_loop_end = end_seq_kv
            if is_causal:
                causal_kv_local_end = (bx + 1) * block_M + causal_offset
                causal_kv_local_end = T.if_then_else(
                    causal_kv_local_end > 0, causal_kv_local_end, 0
                )
                causal_kv_local_end = T.if_then_else(
                    causal_kv_local_end < local_seq_kv,
                    causal_kv_local_end,
                    local_seq_kv,
                )
                kv_loop_end = (
                    begin_seq_kv + T.ceildiv(causal_kv_local_end, block_N) * block_N
                )
            kv_group = T.alloc_var(T.int32)
            kv_group = by if heads_q_eq_heads_kv else by // head_groups

            tid = T.get_thread_binding()

            if tid < producer_threads:
                phase_producer = T.alloc_var(T.int32)
                phase_producer = 0
                k0_reg_buffer = T.alloc_fragment([block_N, dim // 2], dtype)
                k1_reg_buffer = T.alloc_fragment([block_N, dim // 2], dtype)
                T.copy(
                    q_block(Q, bz, block_q_start, begin_seq_q, by, 0),
                    Qa_shared_0,
                    barrier=bar_q_ready,
                )
                T.copy(
                    q_block(Q, bz, block_q_start, begin_seq_q, by, dim // 2),
                    Qa_shared_1,
                    barrier=bar_q_ready,
                )
                T.barrier_arrive(bar_q_ready)

                T.copy(
                    q_block(dO, bz, block_q_start, begin_seq_q, by, 0),
                    dOa_shared_0,
                    barrier=bar_do_ready,
                )
                T.copy(
                    q_block(dO, bz, block_q_start, begin_seq_q, by, dim // 2),
                    dOa_shared_1,
                    barrier=bar_do_ready,
                )
                T.barrier_arrive(bar_do_ready)

                # preload K
                T.barrier_wait(bar_kt0_free, 1)
                _annotate_sqmma(Kt_shared_0, k_major=True)
                T.copy(
                    kv_block(K, bz, kv_loop_start, begin_seq_kv, kv_group, 0),
                    Kt_shared_0,
                    barrier=bar_kt0_ready,
                )
                T.barrier_arrive(bar_kt0_ready)

                T.barrier_wait(bar_kt1_free, 1)
                _annotate_sqmma(Kt_shared_1, k_major=True)
                T.copy(
                    kv_block(K, bz, kv_loop_start, begin_seq_kv, kv_group, dim // 2),
                    Kt_shared_1,
                    barrier=bar_kt1_ready,
                )
                T.barrier_arrive(bar_kt1_ready)

                T.barrier_wait(bar_kt0_ready, 0)
                T.copy(Kt_shared_0, k0_reg_buffer)
                T.lma_wait()
                T.barrier_arrive(bar_vt0_producer_free)
                T.barrier_wait(bar_vt0_producer_free, 0)
                T.barrier_wait(bar_vt0_c0_free, 0)

                for kv_start in range(kv_loop_start, kv_loop_end, block_N):
                    _annotate_sqmma(Vt_shared_0, k_major=True)
                    T.copy(
                        kv_block(V, bz, kv_start, begin_seq_kv, kv_group, 0),
                        Vt_shared_0,
                        barrier=bar_vt0_ready,
                    )
                    T.barrier_arrive(bar_vt0_ready)
                    T.copy(k0_reg_buffer, k1_reg_buffer)

                    T.barrier_wait(bar_vt1_free, phase_producer)
                    _annotate_sqmma(Vt_shared_1, k_major=True)
                    T.copy(
                        kv_block(V, bz, kv_start, begin_seq_kv, kv_group, dim // 2),
                        Vt_shared_1,
                        barrier=bar_vt1_ready,
                    )
                    T.barrier_arrive(bar_vt1_ready)

                    T.barrier_wait(bar_kt0_free, phase_producer)
                    _annotate_sqmma(Kt_shared_0, k_major=True)
                    T.copy(
                        kv_block(K, bz, kv_start + block_N, begin_seq_kv, kv_group, 0),
                        Kt_shared_0,
                        barrier=bar_kt0_ready,
                    )
                    T.barrier_arrive(bar_kt0_ready)

                    T.barrier_wait(bar_kt1_free, phase_producer)
                    _annotate_sqmma(Kt_shared_1, k_major=True)
                    T.copy(
                        kv_block(
                            K, bz, kv_start + block_N, begin_seq_kv, kv_group, dim // 2
                        ),
                        Kt_shared_1,
                        barrier=bar_kt1_ready,
                    )
                    T.barrier_arrive(bar_kt1_ready)

                    T.barrier_wait(bar_kt0_ready, phase_producer ^ 1)
                    T.copy(Kt_shared_0, k0_reg_buffer)
                    T.lma_wait()
                    T.barrier_arrive(bar_vt0_producer_free)

                    T.barrier_wait(bar_k0_c0_free, phase_producer)
                    _annotate_sqmma(K_shared_0, k_major=False, continuity=dim // 4)
                    for i, j in T.Parallel(block_N, dim // 2):
                        K_shared_0[i, j] = k1_reg_buffer[perm_n(i, block_N), j]
                    T.lma_wait()
                    T.barrier_arrive(bar_k0_ready)

                    phase_producer = phase_producer ^ 1
                    T.barrier_wait(bar_vt0_producer_free, phase_producer)
                    T.barrier_wait(bar_vt0_c1_free, phase_producer ^ 1)

            elif tid < producer_threads + consumer0_threads:
                phase_consumer0 = T.alloc_var(T.int32)
                phase_consumer0 = 0
                accs_accum = T.alloc_fragment([block_M, block_N], accum_dtype)
                dp_accum = T.alloc_fragment([block_M, block_N], accum_dtype)
                accum_s_next = T.alloc_fragment([block_M, block_N], accum_dtype)
                if has_softcap:
                    softcap_deriv = T.alloc_fragment([block_M, block_N], accum_dtype)
                accs_cast = T.alloc_fragment([block_M, block_N], dtype)
                lse_buffer = T.alloc_fragment([block_M], accum_dtype)
                delta_buffer = T.alloc_fragment([block_M], accum_dtype)
                for i in T.Parallel(block_M):
                    T.copy(
                        lse_elem(Lse, bz, block_q_start + i, begin_seq_q, by),
                        lse_buffer[i],
                        src_robust_desc=lse_robust_desc,
                    )
                for i in T.Parallel(block_M):
                    lse_buffer[i] = T.if_then_else(
                        T.isfinite(lse_buffer[i]), lse_buffer[i], 2**30
                    )
                for i in T.Parallel(block_M):
                    T.copy(Delta[block_q_start + i, by], delta_buffer[i])
                T.barrier_wait(bar_q_ready, 0)
                T.barrier_wait(bar_kt0_ready, 0)
                _annotate_sqmma(Kt_shared_0, k_major=True)
                T.gemm(
                    Qa_shared_0,
                    Kt_shared_0,
                    accs_accum,
                    wg_wait=-1,
                    clear_accum=True,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.wait_wgmma(0)
                T.barrier_arrive(bar_vt0_c0_free)

                T.barrier_wait(bar_kt1_ready, 0)
                _annotate_sqmma(Kt_shared_1, k_major=True)
                T.gemm(
                    Qa_shared_1,
                    Kt_shared_1,
                    accs_accum,
                    wg_wait=-1,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.wait_wgmma(0)
                T.barrier_arrive(bar_vt1_free)
                for i in T.Parallel(block_M):
                    lse_buffer[i] = lse_buffer[i] * log2e
                for i in T.Parallel(block_M):
                    delta_buffer[i] = delta_buffer[i] * smscale
                if (bx * block_M + block_M) >= local_seq_q:
                    for i in T.Parallel(block_M):
                        lse_buffer[i] = T.if_then_else(
                            bx * block_M + i < local_seq_q,
                            lse_buffer[i],
                            2**30 * log2e,
                        )
                        delta_buffer[i] = T.if_then_else(
                            bx * block_M + i < local_seq_q,
                            delta_buffer[i],
                            0.0,
                        )

                T.barrier_wait(bar_do_ready, 0)
                for kv_start in range(kv_loop_start, kv_loop_end, block_N):
                    T.barrier_wait(bar_vt0_ready, phase_consumer0)
                    _annotate_sqmma(Vt_shared_0, k_major=True)
                    T.gemm(
                        dOa_shared_0,
                        Vt_shared_0,
                        dp_accum,
                        clear_accum=True,
                        wg_wait=-1,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    # exp
                    if (kv_start + block_N) > end_seq_kv:
                        for i, j in T.Parallel(block_M, block_N):
                            accs_accum[i, j] = T.if_then_else(
                                kv_start + j < end_seq_kv,
                                accs_accum[i, j],
                                -(2**30),
                            )
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i
                            k_idx = kv_start - begin_seq_kv + j
                            valid = q_idx + causal_offset >= k_idx
                            accs_accum[i, j] = T.if_then_else(
                                valid, accs_accum[i, j], -(2**30)
                            )
                    if has_window_left or has_window_right:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i
                            k_idx = kv_start - begin_seq_kv + j
                            valid = (
                                not has_window_left
                                or k_idx >= q_idx + causal_offset - window_size_left
                            ) and (
                                not has_window_right
                                or k_idx <= q_idx + causal_offset + window_size_right
                            )
                            accs_accum[i, j] = T.if_then_else(
                                valid, accs_accum[i, j], -(2**30)
                            )
                    if has_softcap:
                        for i, j in T.Parallel(block_M, block_N):
                            softcap_tanh = T.tanh(accs_accum[i, j] * smscale / softcap)
                            softcap_deriv[i, j] = 1.0 - softcap_tanh * softcap_tanh
                            accs_accum[i, j] = softcap_tanh * softcap
                        if (kv_start + block_N) > end_seq_kv:
                            for i, j in T.Parallel(block_M, block_N):
                                valid = kv_start + j < end_seq_kv
                                accs_accum[i, j] = T.if_then_else(
                                    valid, accs_accum[i, j], -(2**30)
                                )
                                softcap_deriv[i, j] = T.if_then_else(
                                    valid, softcap_deriv[i, j], 0.0
                                )
                        if is_causal:
                            for i, j in T.Parallel(block_M, block_N):
                                q_idx = bx * block_M + i
                                k_idx = kv_start - begin_seq_kv + j
                                valid = q_idx + causal_offset >= k_idx
                                accs_accum[i, j] = T.if_then_else(
                                    valid, accs_accum[i, j], -(2**30)
                                )
                                softcap_deriv[i, j] = T.if_then_else(
                                    valid, softcap_deriv[i, j], 0.0
                                )
                        if has_window_left or has_window_right:
                            for i, j in T.Parallel(block_M, block_N):
                                q_idx = bx * block_M + i
                                k_idx = kv_start - begin_seq_kv + j
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
                                softcap_deriv[i, j] = T.if_then_else(
                                    valid, softcap_deriv[i, j], 0.0
                                )
                        for i, j in T.Parallel(block_M, block_N):
                            accs_accum[i, j] = T.exp2(
                                accs_accum[i, j] * log2e - lse_buffer[i]
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            accs_accum[i, j] = (
                                accs_accum[i, j] * rln2_scale - lse_buffer[i]
                            )
                        for i, j in T.Parallel(block_M, block_N):
                            accs_accum[i, j] = T.exp2(accs_accum[i, j])
                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_kt0_free)

                    T.barrier_wait(bar_vt1_ready, phase_consumer0)
                    _annotate_sqmma(Vt_shared_1, k_major=True)
                    T.gemm(
                        dOa_shared_1,
                        Vt_shared_1,
                        dp_accum,
                        wg_wait=-1,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    # accum_s_next 边界处理和 causal mask
                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_kt1_free)
                    if (kv_start + block_N) > end_seq_kv:
                        for i, j in T.Parallel(block_M, block_N):
                            dp_accum[i, j] = T.if_then_else(
                                kv_start + j < end_seq_kv,
                                dp_accum[i, j],
                                0.0,
                            )
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i
                            k_idx = kv_start - begin_seq_kv + j
                            valid = q_idx + causal_offset >= k_idx
                            dp_accum[i, j] = T.if_then_else(valid, dp_accum[i, j], 0.0)
                    if has_window_left or has_window_right:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i
                            k_idx = kv_start - begin_seq_kv + j
                            valid = (
                                not has_window_left
                                or k_idx >= q_idx + causal_offset - window_size_left
                            ) and (
                                not has_window_right
                                or k_idx <= q_idx + causal_offset + window_size_right
                            )
                            dp_accum[i, j] = T.if_then_else(valid, dp_accum[i, j], 0.0)

                    T.barrier_wait(bar_kt0_ready, phase_consumer0 ^ 1)
                    _annotate_sqmma(Kt_shared_0, k_major=True)
                    T.gemm(
                        Qa_shared_0,
                        Kt_shared_0,
                        accum_s_next,
                        clear_accum=True,
                        wg_wait=-1,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(block_M, block_N):
                        dp_accum[i, j] = dp_accum[i, j] * smscale - delta_buffer[i]
                    for i, j in T.Parallel(block_M, block_N):
                        dp_accum[i, j] = dp_accum[i, j] * accs_accum[i, j]
                    if has_softcap:
                        for i, j in T.Parallel(block_M, block_N):
                            dp_accum[i, j] = dp_accum[i, j] * softcap_deriv[i, j]
                    if (bx * block_M + block_M) >= local_seq_q:
                        for i, j in T.Parallel(block_M, block_N):
                            dp_accum[i, j] = T.if_then_else(
                                bx * block_M + i < local_seq_q,
                                dp_accum[i, j],
                                0.0,
                            )
                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_k0_c0_free)

                    T.barrier_wait(bar_kt1_ready, phase_consumer0 ^ 1)
                    _annotate_sqmma(Kt_shared_1, k_major=True)
                    T.gemm(
                        Qa_shared_1,
                        Kt_shared_1,
                        accum_s_next,
                        wg_wait=-1,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    # store ds
                    T.copy(dp_accum, accs_cast)
                    T.barrier_wait(bar_ds_free, phase_consumer0 ^ 1)
                    for i, t, l in T.Parallel(block_M, 8, L):
                        ds_shared[i, t * L + l] = accs_cast[i, l * 8 + t]
                    T.lma_wait()
                    T.barrier_arrive(bar_ds_ready)

                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_vt1_free)

                    T.copy(accum_s_next, accs_accum)
                    phase_consumer0 = phase_consumer0 ^ 1

            else:
                phase_consumer1 = T.alloc_var(T.int32)
                phase_consumer1 = 0
                accum_dq0 = T.alloc_fragment([block_M, dim // 2], accum_dtype)
                accum_dq1 = T.alloc_fragment([block_M, dim // 2], accum_dtype)
                dq_cast0 = T.alloc_fragment([block_M, dim // 2], dtype)
                dq_cast1 = T.alloc_fragment([block_M, dim // 2], dtype)
                T.fill(accum_dq0, 0.0)
                T.fill(accum_dq1, 0.0)
                for kv_start in range(kv_loop_start, kv_loop_end, block_N):
                    T.barrier_wait(bar_k1_free, phase_consumer1 ^ 1)
                    # copy global K1 to shmem ,ldlms
                    for i, j in T.Parallel(block_N, dim // 2):
                        T.copy(
                            kv_elem(
                                K,
                                bz,
                                kv_start + perm_n(i, block_N),
                                begin_seq_kv,
                                kv_group,
                                dim // 2 + j,
                            ),
                            K_shared_1[i, j],
                            force_async_copy=True,
                            src_robust_desc=k_robust_desc,
                        )
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    T.barrier_arrive(bar_k1_ready)

                    _annotate_sqmma(ds_shared, k_major=True)
                    _annotate_sqmma(K_shared_0, k_major=False, continuity=dim // 4)
                    T.barrier_wait(bar_k0_ready, phase_consumer1)
                    T.barrier_wait(bar_ds_ready, phase_consumer1)
                    T.gemm(
                        ds_shared,
                        K_shared_0,
                        accum_dq0,
                        wg_wait=-1,
                        clear_accum=False,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_vt0_c1_free)

                    _annotate_sqmma(K_shared_1, k_major=False, continuity=dim // 4)
                    T.barrier_wait(bar_k1_ready, phase_consumer1)
                    T.gemm(
                        ds_shared,
                        K_shared_1,
                        accum_dq1,
                        wg_wait=-1,
                        clear_accum=False,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_k1_free)
                    T.barrier_arrive(bar_ds_free)

                    phase_consumer1 = phase_consumer1 ^ 1

                T.copy(accum_dq0, dq_cast0)
                T.copy(accum_dq1, dq_cast1)
                for i, j in T.Parallel(block_M, dim // 2):
                    if bx * block_M + i < local_seq_q:
                        if j < dq_dim:
                            T.copy(
                                dq_cast0[i, j],
                                dq_elem(dQ, bz, block_q_start + i, begin_seq_q, by, j),
                            )
                        if dim // 2 + j < dq_dim:
                            T.copy(
                                dq_cast1[i, j],
                                dq_elem(
                                    dQ,
                                    bz,
                                    block_q_start + i,
                                    begin_seq_q,
                                    by,
                                    dim // 2 + j,
                                ),
                            )

    return flashattn_bwd_ws_split_dq_kernel


# fmt: on
