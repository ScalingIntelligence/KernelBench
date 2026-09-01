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
def flashattn_bwd_ws_split_dkdv(
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
    kv_grad_heads = heads_kv if heads_q_eq_heads_kv else heads_q
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

    def lse_elem(tensor, batch_idx, q_idx, begin_seq, head):
        if is_varlen:
            return tensor[head, q_idx]
        return tensor[batch_idx, head, q_idx - begin_seq]

    @T.macro
    def store_kv_grad(
        tensor,
        accum,
        cast,
        batch_idx,
        head,
        block_start,
        block_end,
        begin_seq,
        end_seq,
        dim_start,
        grad_dim,
    ):
        if heads_q_eq_heads_kv:
            T.copy(accum, cast)
            if (block_end + begin_seq) > end_seq:
                for i, j in T.Parallel(block_N, dim // 2):
                    kv_offset = block_start + i
                    if begin_seq + kv_offset < end_seq and dim_start + j < grad_dim:
                        T.copy(
                            cast[i, j],
                            kv_tile_e(
                                tensor,
                                batch_idx,
                                kv_offset,
                                begin_seq,
                                head,
                                dim_start + j,
                            ),
                        )
            else:
                for i, j in T.Parallel(block_N, dim // 2):
                    kv_offset = block_start + i
                    if dim_start + j < grad_dim:
                        T.copy(
                            cast[i, j],
                            kv_tile_e(
                                tensor,
                                batch_idx,
                                kv_offset,
                                begin_seq,
                                head,
                                dim_start + j,
                            ),
                        )
        else:
            if (block_end + begin_seq) > end_seq:
                for i, j in T.Parallel(block_N, dim // 2):
                    kv_offset = block_start + i
                    if begin_seq + kv_offset < end_seq and dim_start + j < grad_dim:
                        T.copy(
                            accum[i, j],
                            kv_tile_e(
                                tensor,
                                batch_idx,
                                kv_offset,
                                begin_seq,
                                head,
                                dim_start + j,
                            ),
                        )
            else:
                for i, j in T.Parallel(block_N, dim // 2):
                    kv_offset = block_start + i
                    if dim_start + j < grad_dim:
                        T.copy(
                            accum[i, j],
                            kv_tile_e(
                                tensor,
                                batch_idx,
                                kv_offset,
                                begin_seq,
                                head,
                                dim_start + j,
                            ),
                        )

    @T.prim_func
    def flashattn_bwd_ws_split_dkdv_kernel(
        Q: q_type,  # type: ignore
        K: k_type,  # type: ignore
        V: v_type,  # type: ignore
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
        max_seq_kv: T.int32,
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
            if has_softcap:
                softcap_scale = T.alloc_var(T.float32)
                softcap_scale = smscale / softcap
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

            shmem_buffer0 = T.alloc_shared([block_M, dim // 2], dtype)
            Qa_shared_0 = shmem_buffer0
            dOb_shared_0 = shmem_buffer0
            Qb_shared_0 = shmem_buffer0
            dOa_shared_0 = shmem_buffer0

            shmem_buffer1 = T.alloc_shared([block_M, dim // 2], dtype)
            Qa_shared_1 = shmem_buffer1
            dOb_shared_1 = shmem_buffer1
            Qb_shared_1 = shmem_buffer1
            dOa_shared_1 = shmem_buffer1

            Kt_shared_0 = T.alloc_shared([block_N, dim // 2], dtype)
            Kt_shared_1 = T.alloc_shared([block_N, dim // 2], dtype)
            Vt_shared_0 = T.alloc_shared([block_N, dim // 2], dtype)
            Vt_shared_1 = T.alloc_shared([block_N, dim // 2], dtype)
            Pt_shared = T.alloc_shared([block_M, block_N], dtype)
            dSt_shared = T.alloc_shared([block_M, block_N], dtype)

            bar_kt = T.alloc_barrier(arrive_count=512)
            bar_vt = T.alloc_barrier(arrive_count=512)
            bar_qa0_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_qa1_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_qb0_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_qb1_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_doa0_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_doa1_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_dob0_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_dob1_ready = T.alloc_barrier(arrive_count=consumer0_threads)

            bar_qa0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_qa1_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_qb0_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_qb1_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_doa0_free = T.alloc_barrier(arrive_count=producer_threads)
            bar_doa1_free = T.alloc_barrier(arrive_count=producer_threads)
            bar_dob0_free = T.alloc_barrier(arrive_count=producer_threads)
            bar_dob1_free = T.alloc_barrier(arrive_count=producer_threads)
            bar_p_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_p_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_dst_ready = T.alloc_barrier(arrive_count=producer_threads)
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
            local_seq_q = T.alloc_var(T.int32)
            local_seq_kv = T.alloc_var(T.int32)
            causal_offset = T.alloc_var(T.int32)
            causal_q_local_mask_end = T.alloc_var(T.int32)
            causal_q_mask_end = T.alloc_var(T.int32)
            local_seq_q = end_seq_q - begin_seq_q
            local_seq_kv = end_seq_kv - begin_seq_kv
            causal_offset = local_seq_kv - local_seq_q
            q_loop_start = T.alloc_var(T.int32)
            q_loop_end = T.alloc_var(T.int32)
            q_loop_start = begin_seq_q
            q_loop_end = end_seq_q
            if is_causal or has_window_right:
                q_local_start = T.alloc_var(T.int32)
                q_local_start = block_kv_start - causal_offset
                if has_window_right:
                    q_local_start = q_local_start - window_size_right
                q_local_start = T.if_then_else(q_local_start > 0, q_local_start, 0)
                q_local_start = T.if_then_else(
                    q_local_start < local_seq_q, q_local_start, local_seq_q
                )
                q_loop_start = (
                    begin_seq_q + T.floordiv(q_local_start, block_M) * block_M
                )
            if has_window_left:
                q_local_end = T.alloc_var(T.int32)
                q_local_end = block_kv_end - causal_offset + window_size_left
                q_local_end = T.if_then_else(q_local_end > 0, q_local_end, 0)
                q_local_end = T.if_then_else(
                    q_local_end < local_seq_q, q_local_end, local_seq_q
                )
                q_loop_end = begin_seq_q + T.ceildiv(q_local_end, block_M) * block_M
                q_loop_end = T.if_then_else(
                    q_loop_end < end_seq_q, q_loop_end, end_seq_q
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
            block_kv_all_invalid = (block_kv_start + begin_seq_kv) >= end_seq_kv
            block_kv_all_valid = (block_kv_end + begin_seq_kv) <= end_seq_kv
            block_kv_partial_valid = (block_kv_start + begin_seq_kv) < end_seq_kv and (
                block_kv_end + begin_seq_kv
            ) > end_seq_kv

            # if not block_kv_all_invalid:
            if (block_kv_start + begin_seq_kv) < end_seq_kv:
                if tid < 512:
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
                    T.ptx_commit_group()
                    # T.ptx_wait_group(0)
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
                    T.ptx_commit_group()
                    # T.ptx_wait_group(0)
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
                    T.ptx_commit_group()
                    # T.ptx_wait_group(0)
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

                if tid >= producer_start:
                    # tme load k
                    phase_producer = T.alloc_var(T.int32)
                    phase_producer = 0
                    accs_accum = T.alloc_fragment([block_M, block_N], accum_dtype)
                    dp_accum = T.alloc_fragment([block_M, block_N], accum_dtype)
                    if has_softcap:
                        softcap_deriv = T.alloc_fragment(
                            [block_M, block_N], accum_dtype
                        )
                    accs_cast = T.alloc_fragment([block_M, block_N], dtype)
                    lse_buffer = T.alloc_fragment([block_M], accum_dtype)
                    delta_buffer = T.alloc_fragment([block_M], accum_dtype)
                    T.barrier_wait(bar_kt, 0)
                    T.barrier_wait(bar_vt, 0)
                    for q_start in range(q_loop_start, q_loop_end, block_M):
                        if not has_window_left:
                            T.call_extern("void", "__musa_loop_transparent_outermost")
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
                        if (block_kv_end + begin_seq_kv) > end_seq_kv:
                            for i, j in T.Parallel(block_M, block_N):
                                accs_accum[i, j] = T.if_then_else(
                                    permuted_kv_valid(
                                        begin_seq_kv, block_kv_start, j, end_seq_kv
                                    ),
                                    0,
                                    float("-inf"),
                                )
                        else:
                            T.fill(accs_accum, 0)
                        if is_causal:
                            if q_start < causal_q_mask_end:
                                for i, j in T.Parallel(block_M, block_N):
                                    q_idx = q_start - begin_seq_q + i
                                    k_idx = bx * block_N + perm_n(j, block_N)
                                    valid = q_idx + causal_offset >= k_idx
                                    accs_accum[i, j] = T.if_then_else(
                                        valid, accs_accum[i, j], float("-inf")
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
                                    valid, accs_accum[i, j], float("-inf")
                                )
                        if (q_start + block_M) >= end_seq_q:
                            for i in T.Parallel(block_M):
                                lse_buffer[i] = T.if_then_else(
                                    q_start + i < end_seq_q, lse_buffer[i], 2**30
                                )
                        if (q_start + block_M) >= end_seq_q:
                            for i in T.Parallel(block_M):
                                delta_buffer[i] = T.if_then_else(
                                    q_start + i < end_seq_q, delta_buffer[i], 0.0
                                )
                        T.barrier_wait(bar_qa0_ready, phase_producer)
                        _annotate_sqmma(Qa_shared_0, k_major=True)
                        T.gemm(
                            Qa_shared_0,
                            Kt_shared_0,
                            accs_accum,
                            wg_wait=-1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.warpgroup_commit_batch()

                        T.barrier_wait(bar_qa1_ready, phase_producer)
                        _annotate_sqmma(Qa_shared_1, k_major=True)
                        T.gemm(
                            Qa_shared_1,
                            Kt_shared_1,
                            accs_accum,
                            wg_wait=-1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(1)
                        T.barrier_arrive(bar_doa0_free)

                        for i in T.Parallel(block_M):
                            lse_buffer[i] = lse_buffer[i] * log2e
                        for i in T.Parallel(block_M):
                            delta_buffer[i] = delta_buffer[i] * smscale
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_doa1_free)

                        T.barrier_wait(bar_doa0_ready, phase_producer)
                        _annotate_sqmma(dOa_shared_0, k_major=True)
                        T.gemm(
                            dOa_shared_0,
                            Vt_shared_0,
                            dp_accum,
                            clear_accum=True,
                            wg_wait=-1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.warpgroup_commit_batch()
                        if has_softcap:
                            T.barrier_wait(bar_doa1_ready, phase_producer)
                            _annotate_sqmma(dOa_shared_1, k_major=True)
                            T.gemm(
                                dOa_shared_1,
                                Vt_shared_1,
                                dp_accum,
                                wg_wait=-1,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullCol,
                            )
                            T.warpgroup_commit_batch()
                            T.warpgroup_wait(0)
                            T.barrier_arrive(bar_dob0_free)
                            for i, j in T.Parallel(block_M, block_N):
                                softcap_tanh = T.tanh(
                                    accs_accum[i, j] * smscale / softcap
                                )
                                softcap_deriv[i, j] = 1.0 - softcap_tanh * softcap_tanh
                                accs_accum[i, j] = softcap_tanh * softcap
                            if (block_kv_end + begin_seq_kv) > end_seq_kv:
                                for i, j in T.Parallel(block_M, block_N):
                                    valid = permuted_kv_valid(
                                        begin_seq_kv, block_kv_start, j, end_seq_kv
                                    )
                                    accs_accum[i, j] = T.if_then_else(
                                        valid, accs_accum[i, j], float("-inf")
                                    )
                                    softcap_deriv[i, j] = T.if_then_else(
                                        valid, softcap_deriv[i, j], 0.0
                                    )
                            if is_causal:
                                if q_start < causal_q_mask_end:
                                    for i, j in T.Parallel(block_M, block_N):
                                        q_idx = q_start - begin_seq_q + i
                                        k_idx = bx * block_N + perm_n(j, block_N)
                                        valid = q_idx + causal_offset >= k_idx
                                        accs_accum[i, j] = T.if_then_else(
                                            valid, accs_accum[i, j], float("-inf")
                                        )
                                        softcap_deriv[i, j] = T.if_then_else(
                                            valid, softcap_deriv[i, j], 0.0
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
                                        valid, accs_accum[i, j], float("-inf")
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

                            T.barrier_wait(bar_doa1_ready, phase_producer)
                            _annotate_sqmma(dOa_shared_1, k_major=True)
                            T.gemm(
                                dOa_shared_1,
                                Vt_shared_1,
                                dp_accum,
                                wg_wait=-1,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullCol,
                            )
                            T.warpgroup_commit_batch()
                            T.warpgroup_wait(1)
                            T.barrier_arrive(bar_dob0_free)

                        T.copy(accs_accum, accs_cast)
                        T.barrier_wait(bar_p_free, phase_producer ^ 1)
                        for i, t, l in T.Parallel(block_M, 8, L):
                            Pt_shared[i, t * L + l] = accs_cast[i, l * 8 + t]
                        T.lma_wait()
                        T.barrier_arrive(bar_p_ready)
                        if not has_softcap:
                            T.warpgroup_wait(0)
                        T.barrier_arrive(bar_dob1_free)

                        for i, j in T.Parallel(block_M, block_N):
                            dp_accum[i, j] = dp_accum[i, j] * smscale - delta_buffer[i]
                        for i, j in T.Parallel(block_M, block_N):
                            dp_accum[i, j] = dp_accum[i, j] * accs_accum[i, j]
                        if has_softcap:
                            for i, j in T.Parallel(block_M, block_N):
                                dp_accum[i, j] = dp_accum[i, j] * softcap_deriv[i, j]
                        T.copy(dp_accum, accs_cast)
                        T.barrier_wait(bar_dst_free, phase_producer ^ 1)
                        for i, t, l in T.Parallel(block_M, 8, L):
                            dSt_shared[i, t * L + l] = accs_cast[i, l * 8 + t]
                        T.lma_wait()
                        T.barrier_arrive(bar_dst_ready)

                        phase_producer = phase_producer ^ 1

                elif tid >= consumer1_start:
                    # P*dO -> dV. dQ is handled by a separate split-DQ kernel.
                    # load Q_a Q_b
                    phase_consumer1 = T.alloc_var(T.int32)
                    phase_consumer1 = 0
                    dv_accum_0 = T.alloc_fragment([block_N, dim // 2], accum_dtype)
                    dv_accum_1 = T.alloc_fragment([block_N, dim // 2], accum_dtype)
                    dv_cast = T.alloc_fragment([block_N, dim // 2], dtype)
                    T.fill(dv_accum_0, 0.0)
                    T.fill(dv_accum_1, 0.0)
                    for q_start in range(q_loop_start, q_loop_end, block_M):
                        if not has_window_left:
                            T.call_extern("void", "__musa_loop_transparent_outermost")
                        T.barrier_wait(bar_qa0_free, phase_consumer1 ^ 1)
                        _annotate_sqmma(Qa_shared_0, k_major=True)
                        T.tma_copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, 0),
                            Qa_shared_0,
                            barrier=bar_qa0_ready,
                        )
                        T.barrier_arrive(bar_qa0_ready)

                        T.barrier_wait(bar_qa1_free, phase_consumer1 ^ 1)
                        _annotate_sqmma(Qa_shared_1, k_major=True)
                        T.tma_copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, dim // 2),
                            Qa_shared_1,
                            barrier=bar_qa1_ready,
                        )
                        T.barrier_arrive(bar_qa1_ready)

                        T.barrier_wait(bar_doa0_free, phase_consumer1)
                        _annotate_sqmma(dOa_shared_0, k_major=True)
                        T.tma_copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, 0),
                            dOa_shared_0,
                            barrier=bar_doa0_ready,
                        )
                        T.barrier_arrive(bar_doa0_ready)

                        T.barrier_wait(bar_doa1_free, phase_consumer1)
                        _annotate_sqmma(dOa_shared_1, k_major=True)
                        T.tma_copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, dim // 2),
                            dOa_shared_1,
                            barrier=bar_doa1_ready,
                        )
                        T.barrier_arrive(bar_doa1_ready)

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
                            wg_wait=-1,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.warpgroup_commit_batch()

                        T.barrier_wait(bar_dob1_ready, phase_consumer1)
                        _annotate_sqmma(
                            dOb_shared_1, k_major=False, continuity=dim // 4
                        )
                        T.gemm(
                            Pt_shared,
                            dOb_shared_1,
                            dv_accum_1,
                            wg_wait=-1,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.warpgroup_commit_batch()

                        T.warpgroup_wait(1)
                        T.barrier_arrive(bar_qb0_free)
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_qb1_free)
                        T.barrier_arrive(bar_p_free)
                        phase_consumer1 = phase_consumer1 ^ 1

                    store_kv_grad(
                        dV,
                        dv_accum_0,
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
                    store_kv_grad(
                        dV,
                        dv_accum_1,
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
                    T.fill(dk_accum_0, 0.0)
                    T.fill(dk_accum_1, 0.0)
                    dk_cast = T.alloc_fragment([block_N, dim // 2], dtype)
                    # Keep the dK loop opaque. Marking it transparent on MTGPU
                    # changes scheduling and corrupts dV in the sibling role.
                    for q_start in range(q_loop_start, q_loop_end, block_M):
                        T.barrier_wait(bar_dob0_free, phase_consumer0)
                        _annotate_sqmma(
                            dOb_shared_0, k_major=False, continuity=dim // 4
                        )
                        T.tma_copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, 0),
                            dOb_shared_0,
                            barrier=bar_dob0_ready,
                        )
                        T.barrier_arrive(bar_dob0_ready)

                        T.barrier_wait(bar_dob1_free, phase_consumer0)
                        _annotate_sqmma(
                            dOb_shared_1, k_major=False, continuity=dim // 4
                        )
                        T.tma_copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, dim // 2),
                            dOb_shared_1,
                            barrier=bar_dob1_ready,
                        )
                        T.barrier_arrive(bar_dob1_ready)

                        T.barrier_wait(bar_qb0_free, phase_consumer0)
                        _annotate_sqmma(Qb_shared_0, k_major=False, continuity=dim // 4)
                        T.tma_copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, 0),
                            Qb_shared_0,
                            barrier=bar_qb0_ready,
                        )
                        T.barrier_arrive(bar_qb0_ready)

                        T.barrier_wait(bar_qb1_free, phase_consumer0)
                        _annotate_sqmma(Qb_shared_1, k_major=False, continuity=dim // 4)
                        T.tma_copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, dim // 2),
                            Qb_shared_1,
                            barrier=bar_qb1_ready,
                        )
                        T.barrier_arrive(bar_qb1_ready)

                        T.barrier_wait(bar_dst_ready, phase_consumer0)
                        T.barrier_wait(bar_qb0_ready, phase_consumer0)
                        _annotate_sqmma(dSt_shared, k_major=False, continuity=block_N)
                        _annotate_sqmma(Qb_shared_0, k_major=False, continuity=dim // 4)
                        T.gemm(
                            dSt_shared,
                            Qb_shared_0,
                            dk_accum_0,
                            wg_wait=-1,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.warpgroup_commit_batch()

                        T.barrier_wait(bar_qb1_ready, phase_consumer0)
                        _annotate_sqmma(dSt_shared, k_major=False, continuity=block_N)
                        _annotate_sqmma(Qb_shared_1, k_major=False, continuity=dim // 4)
                        T.gemm(
                            dSt_shared,
                            Qb_shared_1,
                            dk_accum_1,
                            wg_wait=-1,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.warpgroup_commit_batch()

                        T.warpgroup_wait(1)
                        T.barrier_arrive(bar_qa0_free)
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_qa1_free)
                        T.barrier_arrive(bar_dst_free)

                        phase_consumer0 = phase_consumer0 ^ 1

                    store_kv_grad(
                        dK,
                        dk_accum_0,
                        dk_cast,
                        bz,
                        by,
                        block_kv_start,
                        block_kv_end,
                        begin_seq_kv,
                        end_seq_kv,
                        0,
                        dk_dim,
                    )
                    store_kv_grad(
                        dK,
                        dk_accum_1,
                        dk_cast,
                        bz,
                        by,
                        block_kv_start,
                        block_kv_end,
                        begin_seq_kv,
                        end_seq_kv,
                        dim // 2,
                        dk_dim,
                    )

    # if heads_kv == heads_q -> output dk dv,
    # if heads_kv != heads_q -> output dk_accum dv_accum， -> reduce to dk dv
    return flashattn_bwd_ws_split_dkdv_kernel


# fmt: on
