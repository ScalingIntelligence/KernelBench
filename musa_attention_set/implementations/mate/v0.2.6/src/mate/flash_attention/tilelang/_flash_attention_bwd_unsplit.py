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
def flashattn_bwd_ws_unsplit(
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
    block_N=128,
    dtype="bfloat16",
    use_static_dims=False,
):
    if dim != 128:
        raise ValueError("flashattn_bwd_ws_unsplit only supports dim == 128")
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
    accum_dtype = "float"
    kv_grad_dtype = dtype if heads_q_eq_heads_kv else accum_dtype
    kv_grad_heads = heads_kv if heads_q_eq_heads_kv else heads_q
    L = block_N // 8
    consumer0_threads = 256
    consumer1_threads = 256
    producer_threads = 128
    threads = consumer0_threads + consumer1_threads + producer_threads
    consumer1_start = consumer0_threads
    producer_start = consumer0_threads + consumer1_threads
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    k_stride_s = T.dynamic("k_stride_s")
    k_stride_h = T.dynamic("k_stride_h")
    v_stride_s = T.dynamic("v_stride_s")
    v_stride_h = T.dynamic("v_stride_h")
    o_stride_s = T.dynamic("o_stride_s")
    o_stride_h = T.dynamic("o_stride_h")
    do_stride_s = T.dynamic("do_stride_s")
    do_stride_h = T.dynamic("do_stride_h")
    dk_stride_s = T.dynamic("dk_stride_s")
    dk_stride_h = T.dynamic("dk_stride_h")
    dv_stride_s = T.dynamic("dv_stride_s")
    dv_stride_h = T.dynamic("dv_stride_h")
    q_stride_b = T.dynamic("q_stride_b")
    k_stride_b = T.dynamic("k_stride_b")
    v_stride_b = T.dynamic("v_stride_b")
    o_stride_b = T.dynamic("o_stride_b")
    do_stride_b = T.dynamic("do_stride_b")
    dk_stride_b = T.dynamic("dk_stride_b")
    dv_stride_b = T.dynamic("dv_stride_b")
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
        (total_seq_kv, kv_grad_heads, dk_dim)
        if is_varlen
        else (batch, seqlen_k, kv_grad_heads, dk_dim)
    )
    dv_shape = (
        (total_seq_kv, kv_grad_heads, dv_dim)
        if is_varlen
        else (batch, seqlen_k, kv_grad_heads, dv_dim)
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
    q_type = T.StridedTensor(q_shape, q_strides, dtype)
    k_type = T.StridedTensor(k_shape, k_strides, dtype)
    v_type = T.StridedTensor(v_shape, v_strides, dtype)
    o_type = T.StridedTensor(o_shape, o_strides, dtype)
    do_type = T.StridedTensor(o_shape, do_strides, dtype)
    dk_type = T.StridedTensor(dk_shape, dk_strides, kv_grad_dtype)
    dv_type = T.StridedTensor(dv_shape, dv_strides, kv_grad_dtype)
    dtype_bytes = _tilelang_dtype_nbytes(dtype)
    k_robust_bytes = cosize(k_shape, k_strides) * dtype_bytes
    v_robust_bytes = cosize(v_shape, v_strides) * dtype_bytes

    def q_block(tensor, batch_idx, q_start, begin_seq, head):
        if is_varlen:
            return tensor[q_start : q_start + block_M, head, 0:dim]
        local_q_start = q_start - begin_seq
        return tensor[batch_idx, local_q_start : local_q_start + block_M, head, 0:dim]

    def kv_block(tensor, batch_idx, kv_start, begin_seq, head):
        if is_varlen:
            return tensor[kv_start : kv_start + block_N, head, 0:dim]
        local_kv_start = kv_start - begin_seq
        return tensor[batch_idx, local_kv_start : local_kv_start + block_N, head, 0:dim]

    def kv_elem(tensor, batch_idx, kv_idx, begin_seq, head, dim_idx):
        if is_varlen:
            return tensor[kv_idx, head, dim_idx]
        return tensor[batch_idx, kv_idx - begin_seq, head, dim_idx]

    def kv_tile_e(tensor, batch_idx, kv_offset, begin_seq, head, dim_idx):
        return kv_elem(
            tensor, batch_idx, begin_seq + kv_offset, begin_seq, head, dim_idx
        )

    def kv_tile_b(tensor, batch_idx, kv_offset, begin_seq, head):
        return kv_block(tensor, batch_idx, begin_seq + kv_offset, begin_seq, head)

    def permuted_kv_idx(begin_seq, block_start, idx):
        return begin_seq + block_start + perm_n(idx, block_N, atom_n=64)

    def permuted_kv_valid(begin_seq, block_start, idx, end_seq):
        return permuted_kv_idx(begin_seq, block_start, idx) < end_seq

    def lse_elem(tensor, batch_idx, q_idx, begin_seq, head):
        if is_varlen:
            return tensor[head, q_idx]
        return tensor[batch_idx, head, q_idx - begin_seq]

    @T.prim_func
    def flashattn_bwd_ws_unsplit_kernel(
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

            Qa_shared = T.alloc_shared([block_M, dim], dtype)
            dOb_shared = T.alloc_shared([block_M, dim], dtype)
            Qb_shared = T.alloc_shared([block_M, dim], dtype)
            dOa_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            Kt_shared = T.alloc_shared([block_N, dim], dtype)
            Vt_shared = T.alloc_shared([block_N, dim], dtype)
            Pt_shared = Qa_shared
            dSt_shared = T.alloc_shared([block_M, block_N], dtype)
            dS_shared = T.alloc_shared([block_M, block_N], dtype)

            bar_k = T.alloc_barrier(arrive_count=producer_threads)
            bar_kt = T.alloc_barrier(arrive_count=producer_threads)
            bar_vt = T.alloc_barrier(arrive_count=producer_threads)
            bar_qa_ready = T.alloc_barrier(arrive_count=producer_threads // 4)
            bar_qb_ready = T.alloc_barrier(arrive_count=producer_threads // 4)
            bar_doa_ready = T.alloc_barrier(arrive_count=producer_threads // 4)
            bar_dob_ready = T.alloc_barrier(arrive_count=producer_threads // 4)
            bar_producer_protect = T.alloc_barrier(arrive_count=producer_threads)
            bar_qa_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_qb_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_doa_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_dob_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_p_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_ds_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_dst_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_dst_free = T.alloc_barrier(arrive_count=consumer1_threads)

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

            if (block_kv_start + begin_seq_kv) < end_seq_kv:
                if tid >= producer_start:
                    phase_producer = T.alloc_var(T.int32)
                    phase_producer = 0
                    warprole = T.alloc_var(T.int32)
                    warprole = (tid - producer_start) // 32
                    T.tma_copy(
                        kv_block(
                            K, bz, begin_seq_kv + block_kv_start, begin_seq_kv, kv_group
                        ),
                        K_shared,
                        barrier=bar_k,
                    )
                    T.barrier_arrive(bar_k)
                    for i, j in T.Parallel(block_N, dim):
                        T.copy(
                            kv_elem(
                                K,
                                bz,
                                permuted_kv_idx(begin_seq_kv, block_kv_start, i),
                                begin_seq_kv,
                                kv_group,
                                j,
                            ),
                            Kt_shared[i, j],
                            force_async_copy=True,
                            src_robust_desc=k_robust_desc,
                        )
                    for i, j in T.Parallel(block_N, dim):
                        T.copy(
                            kv_elem(
                                V,
                                bz,
                                permuted_kv_idx(begin_seq_kv, block_kv_start, i),
                                begin_seq_kv,
                                kv_group,
                                j,
                            ),
                            Vt_shared[i, j],
                            force_async_copy=True,
                            src_robust_desc=v_robust_desc,
                        )
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    T.barrier_arrive(bar_kt)
                    T.barrier_arrive(bar_vt)
                    if is_causal:
                        producer_causal_offset = T.alloc_var(T.int32)
                        producer_causal_q_local_start = T.alloc_var(T.int32)
                        producer_causal_q_start = T.alloc_var(T.int32)
                        producer_causal_offset = (end_seq_kv - begin_seq_kv) - (
                            end_seq_q - begin_seq_q
                        )
                        producer_causal_q_local_start = (
                            block_kv_start - producer_causal_offset
                        )
                        producer_causal_q_local_start = T.if_then_else(
                            producer_causal_q_local_start > 0,
                            producer_causal_q_local_start,
                            0,
                        )
                        producer_causal_q_start = (
                            begin_seq_q
                            + T.floordiv(producer_causal_q_local_start, block_M)
                            * block_M
                        )
                    for q_start in range(
                        producer_causal_q_start if is_causal else begin_seq_q,
                        end_seq_q,
                        block_M,
                    ):
                        T.barrier_arrive(bar_producer_protect)
                        if warprole == 0:
                            T.barrier_wait(bar_qa_free, phase_producer ^ 1)
                            _annotate_sqmma(Qa_shared, k_major=True)
                            if tid == producer_start:
                                T.tma_copy(
                                    q_block(Q, bz, q_start, begin_seq_q, by),
                                    Qa_shared,
                                    barrier=bar_qa_ready,
                                )
                            T.barrier_arrive(bar_qa_ready)

                        if warprole == 1:
                            T.barrier_wait(bar_doa_free, phase_producer ^ 1)
                            _annotate_sqmma(dOa_shared, k_major=True)
                            if tid == producer_start + 32:
                                T.tma_copy(
                                    q_block(dO, bz, q_start, begin_seq_q, by),
                                    dOa_shared,
                                    barrier=bar_doa_ready,
                                )
                            T.barrier_arrive(bar_doa_ready)

                        if warprole == 2:
                            T.barrier_wait(bar_dob_free, phase_producer ^ 1)
                            _annotate_sqmma(dOb_shared, k_major=False, continuity=dim)
                            if tid == producer_start + 32 * 2:
                                T.tma_copy(
                                    q_block(dO, bz, q_start, begin_seq_q, by),
                                    dOb_shared,
                                    barrier=bar_dob_ready,
                                )
                            T.barrier_arrive(bar_dob_ready)

                        if warprole == 3:
                            T.barrier_wait(bar_qb_free, phase_producer ^ 1)
                            _annotate_sqmma(
                                Qb_shared, k_major=False, continuity=dim // 2
                            )
                            if tid == producer_start + 32 * 3:
                                T.tma_copy(
                                    q_block(Q, bz, q_start, begin_seq_q, by),
                                    Qb_shared,
                                    barrier=bar_qb_ready,
                                )
                            T.barrier_arrive(bar_qb_ready)
                        T.barrier_wait(bar_producer_protect, phase_producer)
                        phase_producer = phase_producer ^ 1

                elif tid >= consumer1_start:
                    phase_consumer1 = T.alloc_var(T.int32)
                    local_q_start = T.alloc_var(T.int32)
                    phase_consumer1 = 0
                    T.barrier_wait(bar_k, 0)
                    T.barrier_wait(bar_kt, 0)
                    T.barrier_wait(bar_vt, 0)
                    dk_accum = T.alloc_fragment([block_N, dim], accum_dtype)
                    dq_accum = T.alloc_fragment([block_M, dim], accum_dtype)
                    dq_store_buffer = T.alloc_fragment([block_M, dim], accum_dtype)
                    dk_cast = T.alloc_fragment([block_N, dim], dtype)
                    T.fill(dk_accum, 0.0)
                    T.barrier_wait(bar_qb_ready, phase_consumer1)
                    T.barrier_wait(bar_dst_ready, phase_consumer1)
                    _annotate_sqmma(dSt_shared, k_major=False, continuity=block_N)
                    _annotate_sqmma(Qb_shared, k_major=False, continuity=dim // 2)
                    T.gemm(
                        dSt_shared,
                        Qb_shared,
                        dk_accum,
                        wg_wait=-1,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.wait_wgmma(0)
                    T.barrier_arrive(bar_qb_free)
                    T.barrier_arrive(bar_dst_free)

                    T.barrier_wait(bar_ds_ready, phase_consumer1)
                    _annotate_sqmma(dS_shared, k_major=True)
                    T.gemm(
                        dS_shared,
                        K_shared,
                        dq_accum,
                        wg_wait=-1,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.wait_wgmma(0)
                    T.copy(dq_accum, dq_store_buffer)
                    phase_consumer1 = phase_consumer1 ^ 1

                    if is_causal:
                        consumer1_causal_offset = T.alloc_var(T.int32)
                        consumer1_causal_q_local_start = T.alloc_var(T.int32)
                        consumer1_causal_q_start = T.alloc_var(T.int32)
                        consumer1_causal_offset = (end_seq_kv - begin_seq_kv) - (
                            end_seq_q - begin_seq_q
                        )
                        consumer1_causal_q_local_start = (
                            block_kv_start - consumer1_causal_offset
                        )
                        consumer1_causal_q_local_start = T.if_then_else(
                            consumer1_causal_q_local_start > 0,
                            consumer1_causal_q_local_start,
                            0,
                        )
                        consumer1_causal_q_start = (
                            begin_seq_q
                            + T.floordiv(consumer1_causal_q_local_start, block_M)
                            * block_M
                        )
                    dq_loop_start = (
                        consumer1_causal_q_start if is_causal else begin_seq_q
                    )
                    local_q_start = dq_loop_start - begin_seq_q
                    for q_start in range(dq_loop_start, end_seq_q - block_M, block_M):
                        for i, j in T.Parallel(block_M, dim):
                            T.atomic_add(
                                dQ_accum[bz, by, local_q_start + i, j],
                                dq_store_buffer[i, j],
                            )

                        T.barrier_wait(bar_qb_ready, phase_consumer1)
                        T.barrier_wait(bar_dst_ready, phase_consumer1)
                        _annotate_sqmma(dSt_shared, k_major=False, continuity=block_N)
                        _annotate_sqmma(Qb_shared, k_major=False, continuity=dim // 2)
                        T.gemm(
                            dSt_shared,
                            Qb_shared,
                            dk_accum,
                            wg_wait=-1,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.wait_wgmma(0)

                        T.barrier_wait(bar_ds_ready, phase_consumer1)
                        _annotate_sqmma(dS_shared, k_major=True)
                        T.gemm(
                            dS_shared,
                            K_shared,
                            dq_accum,
                            wg_wait=-1,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_qb_free)
                        T.barrier_arrive(bar_dst_free)
                        T.copy(dq_accum, dq_store_buffer)
                        phase_consumer1 = phase_consumer1 ^ 1
                        local_q_start += block_M
                    for i, j in T.Parallel(block_M, dim):
                        T.atomic_add(
                            dQ_accum[bz, by, local_q_start + i, j],
                            dq_store_buffer[i, j],
                        )
                    if heads_q_eq_heads_kv:
                        T.copy(dk_accum, dk_cast)
                    if (block_kv_end + begin_seq_kv) > end_seq_kv:
                        for i, j in T.Parallel(block_N, dim):
                            kv_offset = block_kv_start + i
                            if begin_seq_kv + kv_offset < end_seq_kv and j < dk_dim:
                                if heads_q_eq_heads_kv:
                                    T.copy(
                                        dk_cast[i, j],
                                        kv_tile_e(
                                            dK, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                                else:
                                    T.copy(
                                        dk_accum[i, j],
                                        kv_tile_e(
                                            dK, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                    else:
                        for i, j in T.Parallel(block_N, dim):
                            kv_offset = block_kv_start + i
                            if j < dk_dim:
                                if heads_q_eq_heads_kv:
                                    T.copy(
                                        dk_cast[i, j],
                                        kv_tile_e(
                                            dK, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                                else:
                                    T.copy(
                                        dk_accum[i, j],
                                        kv_tile_e(
                                            dK, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )

                else:
                    phase_consumer0 = T.alloc_var(T.int32)
                    phase_consumer0 = 0
                    dv_accum = T.alloc_fragment([block_N, dim], accum_dtype)
                    accs_accum = T.alloc_fragment([block_M, block_N], accum_dtype)
                    dp_accum = T.alloc_fragment([block_M, block_N], accum_dtype)
                    accs_cast = T.alloc_fragment([block_M, block_N], dtype)
                    lse_buffer = T.alloc_fragment([block_M], accum_dtype)
                    delta_buffer = T.alloc_fragment([block_M], accum_dtype)
                    T.barrier_wait(bar_k, 0)
                    T.barrier_wait(bar_kt, 0)
                    T.barrier_wait(bar_vt, 0)
                    T.fill(dv_accum, 0.0)
                    dv_cast = T.alloc_fragment([block_N, dim], dtype)
                    if is_causal or has_window_left or has_window_right:
                        consumer0_causal_offset = T.alloc_var(T.int32)
                        consumer0_causal_offset = (end_seq_kv - begin_seq_kv) - (
                            end_seq_q - begin_seq_q
                        )
                    if is_causal:
                        consumer0_causal_q_local_start = T.alloc_var(T.int32)
                        consumer0_causal_q_local_mask_end = T.alloc_var(T.int32)
                        consumer0_causal_q_start = T.alloc_var(T.int32)
                        consumer0_causal_q_mask_end = T.alloc_var(T.int32)
                        consumer0_causal_q_local_start = (
                            block_kv_start - consumer0_causal_offset
                        )
                        consumer0_causal_q_local_start = T.if_then_else(
                            consumer0_causal_q_local_start > 0,
                            consumer0_causal_q_local_start,
                            0,
                        )
                        consumer0_causal_q_start = (
                            begin_seq_q
                            + T.floordiv(consumer0_causal_q_local_start, block_M)
                            * block_M
                        )
                        consumer0_causal_q_local_mask_end = (
                            block_kv_end - consumer0_causal_offset
                        )
                        consumer0_causal_q_local_mask_end = T.if_then_else(
                            consumer0_causal_q_local_mask_end > 0,
                            consumer0_causal_q_local_mask_end,
                            0,
                        )
                        consumer0_causal_q_mask_end = (
                            begin_seq_q
                            + T.ceildiv(consumer0_causal_q_local_mask_end, block_M)
                            * block_M
                        )
                        consumer0_causal_q_mask_end = T.if_then_else(
                            consumer0_causal_q_mask_end < end_seq_q,
                            consumer0_causal_q_mask_end,
                            end_seq_q,
                        )
                    for q_start in range(
                        consumer0_causal_q_start if is_causal else begin_seq_q,
                        end_seq_q,
                        block_M,
                    ):
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
                        for i in T.Parallel(block_M):
                            T.copy(
                                Delta[q_start + i, by],
                                delta_buffer[i],
                                src_robust_desc=delta_robust_desc,
                            )
                        if (
                            not has_window_left
                            and not has_window_right
                            and not has_softcap
                        ):
                            if (block_kv_end + begin_seq_kv) > end_seq_kv:
                                for i, j in T.Parallel(block_M, block_N):
                                    accs_accum[i, j] = T.if_then_else(
                                        permuted_kv_valid(
                                            begin_seq_kv,
                                            block_kv_start,
                                            j,
                                            end_seq_kv,
                                        ),
                                        0.0,
                                        float("-inf"),
                                    )
                            else:
                                T.fill(accs_accum, 0)
                            if is_causal:
                                if q_start < consumer0_causal_q_mask_end:
                                    for i, j in T.Parallel(block_M, block_N):
                                        q_idx = q_start - begin_seq_q + i
                                        k_idx = bx * block_N + perm_n(
                                            j, block_N, atom_n=64
                                        )
                                        valid = q_idx + consumer0_causal_offset >= k_idx
                                        accs_accum[i, j] = T.if_then_else(
                                            valid,
                                            accs_accum[i, j],
                                            float("-inf"),
                                        )
                        else:
                            T.fill(accs_accum, 0)
                        for i in T.Parallel(block_M):
                            lse_buffer[i] = lse_buffer[i] * log2e
                        for i in T.Parallel(block_M):
                            delta_buffer[i] = delta_buffer[i] * smscale
                        if (q_start + block_M) >= end_seq_q:
                            for i in T.Parallel(block_M):
                                lse_buffer[i] = T.if_then_else(
                                    q_start + i < end_seq_q,
                                    lse_buffer[i],
                                    2**30 * log2e,
                                )
                        if (q_start + block_M) >= end_seq_q:
                            for i in T.Parallel(block_M):
                                delta_buffer[i] = T.if_then_else(
                                    q_start + i < end_seq_q, delta_buffer[i], 0.0
                                )
                        T.barrier_wait(bar_qa_ready, phase_consumer0)
                        _annotate_sqmma(Qa_shared, k_major=True)
                        T.gemm(
                            Qa_shared,
                            Kt_shared,
                            accs_accum,
                            clear_accum=False,
                            wg_wait=-1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.warpgroup_commit_batch()

                        T.barrier_wait(bar_doa_ready, phase_consumer0)
                        _annotate_sqmma(dOa_shared, k_major=True)
                        T.gemm(
                            dOa_shared,
                            Vt_shared,
                            dp_accum,
                            clear_accum=True,
                            wg_wait=-1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(1)
                        T.barrier_arrive(bar_p_free)
                        # for i, j in T.Parallel(block_M, block_N):
                        #     accs_accum[i, j] = accs_accum[i, j] * rln2_scale
                        if has_softcap:
                            for i, j in T.Parallel(block_M, block_N):
                                q_idx = q_start - begin_seq_q + i
                                k_idx = bx * block_N + perm_n(j, block_N, atom_n=64)
                                valid_0 = True
                                if is_causal:
                                    valid_1 = (
                                        q_start >= consumer0_causal_q_mask_end
                                        or q_idx + consumer0_causal_offset >= k_idx
                                    )
                                else:
                                    valid_1 = valid_0
                                if has_window_left:
                                    valid_2 = valid_1 and (
                                        k_idx
                                        >= q_idx
                                        + consumer0_causal_offset
                                        - window_size_left
                                    )
                                else:
                                    valid_2 = valid_1
                                if has_window_right:
                                    valid_3 = valid_2 and (
                                        k_idx
                                        <= q_idx
                                        + consumer0_causal_offset
                                        + window_size_right
                                    )
                                else:
                                    valid_3 = valid_2
                                valid_4 = valid_3 and T.if_then_else(
                                    (block_kv_end + begin_seq_kv) > end_seq_kv,
                                    permuted_kv_valid(
                                        begin_seq_kv, block_kv_start, j, end_seq_kv
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
                                    dp_accum[i, j] * smscale - delta_buffer[i]
                                ) * (1.0 - softcap_tanh * softcap_tanh)
                                accs_accum[i, j] = T.if_then_else(
                                    valid,
                                    T.exp2(
                                        softcap_tanh * softcap * log2e - lse_buffer[i]
                                    ),
                                    0.0,
                                )
                        else:
                            if not has_window_left and not has_window_right:
                                for i, j in T.Parallel(block_M, block_N):
                                    accs_accum[i, j] = T.exp2(
                                        accs_accum[i, j] * rln2_scale - lse_buffer[i]
                                    )
                            else:
                                for i, j in T.Parallel(block_M, block_N):
                                    q_idx = q_start - begin_seq_q + i
                                    k_idx = bx * block_N + perm_n(j, block_N, atom_n=64)
                                    valid_0 = True
                                    if is_causal:
                                        valid_1 = (
                                            q_start >= consumer0_causal_q_mask_end
                                            or q_idx + consumer0_causal_offset >= k_idx
                                        )
                                    else:
                                        valid_1 = valid_0
                                    if has_window_left:
                                        valid_2 = valid_1 and (
                                            k_idx
                                            >= q_idx
                                            + consumer0_causal_offset
                                            - window_size_left
                                        )
                                    else:
                                        valid_2 = valid_1
                                    if has_window_right:
                                        valid_3 = valid_2 and (
                                            k_idx
                                            <= q_idx
                                            + consumer0_causal_offset
                                            + window_size_right
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
                                    accs_accum[i, j] = T.if_then_else(
                                        valid,
                                        T.exp2(
                                            accs_accum[i, j] * rln2_scale
                                            - lse_buffer[i]
                                        ),
                                        0.0,
                                    )
                        T.copy(accs_accum, accs_cast)
                        T.barrier_wait(bar_p_free, phase_consumer0)
                        _annotate_sqmma(Pt_shared, k_major=False, continuity=64)
                        for i, j in T.Parallel(block_M, block_N):
                            Pt_shared[i, perm_n(j, block_N, atom_n=64)] = accs_cast[
                                i, j
                            ]
                        T.lma_wait()
                        T.barrier_wait(bar_dob_ready, phase_consumer0)
                        _annotate_sqmma(Pt_shared, k_major=False, continuity=64)
                        _annotate_sqmma(dOb_shared, k_major=False, continuity=dim)
                        T.gemm(
                            Pt_shared,
                            dOb_shared,
                            dv_accum,
                            wg_wait=-1,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(1)
                        T.barrier_arrive(bar_doa_free)

                        if has_softcap:
                            for i, j in T.Parallel(block_M, block_N):
                                dp_accum[i, j] = dp_accum[i, j] * accs_accum[i, j]
                        else:
                            for i, j in T.Parallel(block_M, block_N):
                                dp_accum[i, j] = (
                                    dp_accum[i, j] * smscale - delta_buffer[i]
                                ) * accs_accum[i, j]
                        T.copy(dp_accum, accs_cast)
                        for i, j in T.Parallel(block_M, block_N):
                            dSt_shared[i, perm_n(j, block_N, atom_n=64)] = accs_cast[
                                i, j
                            ]
                        T.lma_wait()
                        T.barrier_arrive(bar_dst_ready)
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_qa_free)
                        T.barrier_arrive(bar_dob_free)
                        _annotate_sqmma(dS_shared, k_major=True)
                        for i, j in T.Parallel(block_M, block_N):
                            dS_shared[i, perm_n(j, block_N, atom_n=64)] = accs_cast[
                                i, j
                            ]
                        T.lma_wait()
                        T.barrier_arrive(bar_ds_ready)
                        phase_consumer0 = phase_consumer0 ^ 1

                    if heads_q_eq_heads_kv:
                        T.copy(dv_accum, dv_cast)
                    if (block_kv_end + begin_seq_kv) > end_seq_kv:
                        for i, j in T.Parallel(block_N, dim):
                            kv_offset = block_kv_start + i
                            if begin_seq_kv + kv_offset < end_seq_kv and j < dv_dim:
                                if heads_q_eq_heads_kv:
                                    T.copy(
                                        dv_cast[i, j],
                                        kv_tile_e(
                                            dV, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                                else:
                                    T.copy(
                                        dv_accum[i, j],
                                        kv_tile_e(
                                            dV, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                    else:
                        for i, j in T.Parallel(block_N, dim):
                            kv_offset = block_kv_start + i
                            if j < dv_dim:
                                if heads_q_eq_heads_kv:
                                    T.copy(
                                        dv_cast[i, j],
                                        kv_tile_e(
                                            dV, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                                else:
                                    T.copy(
                                        dv_accum[i, j],
                                        kv_tile_e(
                                            dV, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )

    return flashattn_bwd_ws_unsplit_kernel


# fmt: on
