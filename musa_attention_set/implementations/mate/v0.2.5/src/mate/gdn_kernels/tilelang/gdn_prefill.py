import torch
from mate.utils import cosize
import tilelang
import tilelang.language as T

__all__ = ["fused_chunk_gdn_prefill"]


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
    compile_flags=[
        "-Od3",
        "-fno-signed-zeros",
        "-fmusa-flush-denormals-to-zero",
        "-mllvm",
        "-mtgpu-if-convert=1",
        "-mllvm",
        "-misched=mtgpu-max-ilp",
        "-mllvm",
        "-mtgpu-tiny-offset-hint=1",
        "-mllvm",
        "-mtgpu-enable-postra-sched=0",
        "-mllvm",
        "-misched-recompute-slotindex=1",
        "-mllvm",
        "-mtgpu-combine-fop-instr=1",
    ],
)
def tilelang_fused_chunk_gdn_prefill(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    is_varlen,
    is_log_space,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    raw_batch_size = T.dynamic("raw_batch_size")
    block_S = chunk_size
    assert DV == 128, "gdn_prefill tilelang kernel currently supports DV=128 only"

    if is_varlen:
        q_shape = (1, num_tokens, Hg, DK)
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        o_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
    else:
        q_shape = (batch_size, num_tokens, Hg, DK)
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        o_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
    h0_shape = (raw_batch_size if is_varlen else batch_size, H, DK, DV)
    ht_shape = (raw_batch_size, H, DK, DV)
    seqlens_shape = (raw_batch_size + 1,) if is_varlen else (batch_size + 1,)

    q_stride_b = T.dynamic("q_stride_b")
    q_stride_t = T.dynamic("q_stride_t")
    q_stride_h = T.dynamic("q_stride_h")
    k_stride_b = T.dynamic("k_stride_b")
    k_stride_t = T.dynamic("k_stride_t")
    k_stride_h = T.dynamic("k_stride_h")
    v_stride_b = T.dynamic("v_stride_b")
    v_stride_t = T.dynamic("v_stride_t")
    v_stride_h = T.dynamic("v_stride_h")
    q_strides = (q_stride_b, q_stride_t, q_stride_h, 1)
    k_strides = (k_stride_b, k_stride_t, k_stride_h, 1)
    v_strides = (v_stride_b, v_stride_t, v_stride_h, 1)

    @T.macro
    def producer_phase(i):
        return T.bitwise_xor((i // 2) % 2, 1)

    @T.macro
    def consumer_phase(i):
        return (i // 2) % 2

    @T.macro
    def perm_dv(i):
        stride = DV // 8
        return (i % 8) * stride + (i // 8)

    @T.macro
    def inv_perm_dv(i):
        stride = DV // 8
        return (i % stride) * 8 + (i // stride)

    @T.macro
    def inv_pair_dv(i):
        stride = DV // 2
        return (i % stride) * 2 + (i // stride)

    @T.macro
    def pair_dv(i):
        stride = DV // 2
        return (i % 2) * stride + (i // 2)

    @T.macro
    def qs_fragment_to_natural_dv(i):
        return inv_pair_dv(inv_perm_dv(i))

    @T.macro
    def perm_s(i):
        stride = block_S // 8
        return (i % 8) * stride + (i // 8)

    @T.prim_func
    def tilelang_fused_chunk_gdn_prefill_kernel(
        q: T.StridedTensor(q_shape, q_strides, qkva_dtype),
        k: T.StridedTensor(k_shape, k_strides, qkva_dtype),
        v: T.StridedTensor(v_shape, v_strides, qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor(seqlens_shape, dtype=seqlen_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        launch_batch_size = raw_batch_size if is_varlen else batch_size
        with T.Kernel(launch_batch_size * H, threads=1024) as (bbh,):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens

            seq_len = T.alloc_var("int32")
            seq_len = seq_end_idx - seq_start_idx
            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_len, block_S)

            q_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            scaled_k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            pa_shared = T.alloc_shared((2, block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")

            h_shared = T.alloc_shared((DK, DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            g_exp_shared = T.alloc_shared(
                (2, block_S), dtype=accum_dtype, scope="shared"
            )
            g_exp_rev_shared = T.alloc_shared(
                (2, block_S), dtype=accum_dtype, scope="shared"
            )
            h_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            uv_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            q_is_ready = T.alloc_barrier(arrive_count=[32] * 2)
            q_is_free = T.alloc_barrier(arrive_count=[640] * 2)
            k_is_ready = T.alloc_barrier(arrive_count=[32] * 2)
            k_is_free = T.alloc_barrier(arrive_count=[640] * 2)
            gb_is_ready = T.alloc_barrier(arrive_count=[64] * 2)
            gb_is_free = T.alloc_barrier(arrive_count=[768] * 2)
            v_is_ready = T.alloc_barrier(arrive_count=32)
            v_is_free = T.alloc_barrier(arrive_count=128)

            a_is_ready = T.alloc_barrier(arrive_count=[32] * 2)
            a_is_free = T.alloc_barrier(arrive_count=[128] * 2)
            inv_is_ready = T.alloc_barrier(arrive_count=[512] * 2)
            inv_is_free = T.alloc_barrier(arrive_count=[128] * 2)
            p_is_ready = T.alloc_barrier(arrive_count=[512] * 2)
            scaled_k_is_ready = T.alloc_barrier(arrive_count=512)
            vd_is_ready = T.alloc_barrier(arrive_count=128)
            vd_is_free = T.alloc_barrier(arrive_count=512)
            h_is_ready = T.alloc_barrier(arrive_count=512)
            h_is_free = T.alloc_barrier(arrive_count=256)

            b_bytes = 4 if b_dtype in (torch.float32, "float32") else 2
            b_robust_desc = T.make_robust_desc(
                T.address_of(b[batch_idx, seq_start_idx, 0]),
                cosize((seq_end_idx - seq_start_idx, H)) * b_bytes,
            )

            tx = T.get_thread_binding()

            with T.ws(0):
                w_local = T.alloc_local([4], accum_dtype)
                for i_s in T.serial(num_iters):
                    T.barrier_wait(h_is_ready, i_s % 2)
                    T.barrier_wait(k_is_ready[i_s % 2], consumer_phase(i_s))
                    # U = K @ S
                    T.gemm(
                        k_shared[i_s % 2, :, :],
                        h_shared,
                        uv_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    # W = V - g * U
                    T.barrier_wait(gb_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(v_is_ready, i_s % 2)
                    T.warpgroup_wait(0)
                    T.barrier_arrive(k_is_free[i_s % 2])
                    T.barrier_arrive(h_is_free)

                    ldg_tx = tx % 8
                    ldg_ty = tx // 8
                    for j_r in T.unroll(block_S // 16):
                        for j_p in T.unroll(2):
                            for j_q in T.unroll(2):
                                for j_u in T.vectorized(4):
                                    j_s = j_r * 16 + ldg_ty
                                    w_local[j_u] = v_shared[
                                        j_s,
                                        ldg_tx * 8 + j_p * 64 + j_q * 4 + j_u,
                                    ]
                                for j_u in T.vectorized(4):
                                    j_s = j_r * 16 + ldg_ty
                                    j_v = ((j_q * 4 + j_u) * 2 + j_p) * 8 + ldg_tx
                                    w_local[j_u] -= (
                                        g_exp_shared[i_s % 2, j_s]
                                        * uv_fragment[j_s, j_v]
                                    )
                                for j_u in T.vectorized(4):
                                    j_s = j_r * 16 + ldg_ty
                                    v_shared[
                                        j_s, ldg_tx * 8 + j_p * 64 + j_q * 4 + j_u
                                    ] = w_local[j_u]
                    T.barrier_arrive(gb_is_free[i_s % 2])

                    T.barrier_wait(inv_is_ready[i_s % 2], consumer_phase(i_s))
                    T.lma_wait()
                    # Vd = Ag @ W
                    T.gemm(
                        pa_shared[i_s % 2, :, :],
                        v_shared,
                        uv_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    T.warpgroup_wait(0)
                    T.barrier_arrive(inv_is_free[i_s % 2])
                    T.barrier_wait(vd_is_free, T.bitwise_xor(i_s % 2, 1))
                    for j_s, j_v in T.Parallel(block_S, DV):
                        v_shared[j_s, perm_dv(j_v)] = uv_fragment[j_s, j_v]
                        vd_shared[j_s, perm_dv(j_v)] = uv_fragment[j_s, j_v]
                    T.lma_wait()
                    T.barrier_arrive(vd_is_ready)

            with T.ws(1):
                for i_s in T.serial(num_iters):
                    store_left = seq_start_idx + i_s * block_S
                    valid_seqs = T.min(seq_end_idx - store_left, block_S)

                    T.barrier_wait(q_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(h_is_ready, i_s % 2)
                    T.gemm(
                        q_shared[i_s % 2, :, :],
                        h_shared,
                        o_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    T.warpgroup_wait(0)
                    T.barrier_arrive(q_is_free[i_s % 2])
                    T.barrier_arrive(h_is_free)

                    T.barrier_wait(gb_is_ready[i_s % 2], consumer_phase(i_s))
                    for j_s, j_v in T.Parallel(block_S, DV):
                        o_fragment[j_s, j_v] *= g_exp_shared[i_s % 2, j_s] * scale
                    T.barrier_arrive(gb_is_free[i_s % 2])

                    T.barrier_wait(p_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(vd_is_ready, i_s % 2)
                    # O += Pg @ Vd
                    T.gemm(
                        pa_shared[i_s % 2, :, :],
                        v_shared,
                        o_fragment,
                        clear_accum=False,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.barrier_arrive(a_is_free[i_s % 2])
                    T.barrier_arrive(v_is_free)
                    # S2[S] O.  o_fragment's DV register order is not natural;
                    # remap it while storing directly to the global output.
                    for j_s, j_v in T.Parallel(block_S, DV):
                        if j_s < valid_seqs:
                            o[batch_idx, store_left + j_s, bh, inv_perm_dv(j_v)] = (
                                o_fragment[j_s, j_v]
                            )

            with T.ws(2, 3, 4, 5):
                # Initialize S
                if use_initial_state:
                    for j_k, j_v in T.Parallel(DV, DK):
                        h_fragment[j_k, j_v] = h0[bb, bh, inv_perm_dv(j_v), j_k]
                else:
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_fragment[j_k, j_v] = 0.0
                # Main Loop
                for i_s in T.serial(num_iters):
                    valid_seqs = T.min(
                        seq_end_idx - seq_start_idx - i_s * block_S, block_S
                    )
                    T.barrier_wait(h_is_free, T.bitwise_xor(i_s % 2, 1))
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_shared[j_k, j_v] = h_fragment[j_k, j_v]
                    T.lma_wait()
                    T.barrier_arrive(h_is_ready)

                    T.barrier_wait(q_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(k_is_ready[i_s % 2], consumer_phase(i_s))
                    T.gemm(
                        q_shared[i_s % 2, :, :],
                        k_shared[i_s % 2, :, :],
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    load_left = seq_start_idx + i_s * block_S
                    load_right = load_left + block_S
                    # Load the K tile used for the state update through TMA.
                    # K can be a split-QKV strided view; TileLang emits a TMA
                    # descriptor with the tensor strides, and TMA pads tail OOB.
                    if tx == 256:
                        T.copy(
                            k[batch_idx, load_left:load_right, bhg, 0:DK],
                            scaled_k_shared,
                            barrier=scaled_k_is_ready,
                        )
                    T.barrier_arrive(scaled_k_is_ready)

                    T.barrier_wait(gb_is_ready[i_s % 2], consumer_phase(i_s))
                    # G = Lower(diag(g) @ I @ diag(1/g))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = (
                            g_shared[i_s % 2, j_s] - g_shared[i_s % 2, j_t]
                        )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = T.if_then_else(
                            g_fragment[j_s, j_t] <= 0.0,
                            g_fragment[j_s, j_t],
                            -float("inf"),
                        )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = T.exp2(
                            g_fragment[j_s, j_t] * 1.4426950408889634
                        )

                    # Ag = G * Ar * b
                    T.barrier_wait(a_is_ready[i_s % 2], consumer_phase(i_s))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        pa_shared[i_s % 2, j_s, j_t] = (
                            g_fragment[j_s, j_t]
                            * pa_shared[i_s % 2, j_s, j_t]
                            * b_shared[i_s % 2, j_t]
                        )
                    T.lma_wait()
                    T.barrier_arrive(inv_is_ready[i_s % 2])
                    T.barrier_wait(scaled_k_is_ready, i_s % 2)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        scaled_k_shared[j_s, j_k] *= g_exp_rev_shared[i_s % 2, j_s]

                    T.warpgroup_wait(0)
                    T.barrier_arrive(q_is_free[i_s % 2])
                    T.barrier_arrive(k_is_free[i_s % 2])
                    g_last_local = g_exp_shared[i_s % 2, valid_seqs - 1]
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_fragment[j_k, j_v] *= g_last_local
                    T.barrier_arrive(gb_is_free[i_s % 2])

                    T.barrier_wait(inv_is_free[i_s % 2], consumer_phase(i_s))
                    # Pg = s * G * P
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        pa_shared[i_s % 2, j_s, j_t] = (
                            g_fragment[j_s, j_t] * p_fragment[j_s, j_t] * scale
                        )
                    T.lma_wait()
                    T.barrier_arrive(p_is_ready[i_s % 2])

                    T.barrier_wait(vd_is_ready, i_s % 2)
                    T.lma_wait()
                    # Make the scaled K tile visible before the 512-thread MMA.
                    T.sync_threads(0, 512)
                    T.gemm(
                        scaled_k_shared,
                        vd_shared,
                        h_fragment,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=False,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.barrier_arrive(vd_is_free)
                # Store final S
                if store_final_state:
                    for j_k, j_v in T.Parallel(DV, DK):
                        ht[bb, bh, inv_perm_dv(j_v), j_k] = h_fragment[j_k, j_v]

            with T.ws(6):
                if tx < 800:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(q_is_free[i_s % 2], producer_phase(i_s))
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        if tx == 768:
                            T.copy(
                                q[batch_idx, load_left:load_right, bhg, 0:DK],
                                q_shared[i_s % 2, :, :],
                                barrier=q_is_ready[i_s % 2],
                            )
                        T.barrier_arrive(q_is_ready[i_s % 2])
                elif tx < 832:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(k_is_free[i_s % 2], producer_phase(i_s))
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        if tx == 800:
                            T.copy(
                                k[batch_idx, load_left:load_right, bhg, 0:DK],
                                k_shared[i_s % 2, :, :],
                                barrier=k_is_ready[i_s % 2],
                            )
                        T.barrier_arrive(k_is_ready[i_s % 2])
                elif tx < 864:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(a_is_free[i_s % 2], producer_phase(i_s))
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        if tx == 832:
                            T.copy(
                                a[batch_idx, load_left:load_right, bh, 0:block_S],
                                pa_shared[i_s % 2, :, :],
                                barrier=a_is_ready[i_s % 2],
                            )
                        T.barrier_arrive(a_is_ready[i_s % 2])
                elif tx < 896:
                    for i_s in T.serial(num_iters):
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        T.barrier_wait(v_is_free, T.bitwise_xor(i_s % 2, 1))
                        if tx == 864:
                            T.copy(
                                v[
                                    batch_idx,
                                    load_left:load_right,
                                    bh,
                                    :,
                                ],
                                v_shared,
                                barrier=v_is_ready,
                            )
                        T.barrier_arrive(v_is_ready)
            with T.ws(7):
                if tx < 960:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(gb_is_free[i_s % 2], producer_phase(i_s))
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        valid_seqs = T.min(
                            seq_end_idx - seq_start_idx - i_s * block_S, block_S
                        )
                        T.copy(
                            b[batch_idx, load_left:load_right, bh],
                            b_shared[i_s % 2, :],
                            force_async_copy=True,
                            src_robust_desc=b_robust_desc,
                        )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.sync_threads(1, 64)
                        gb_lane = tx - 896
                        gb_lane_in_warp = gb_lane % 32
                        g_prefix = T.alloc_var(accum_dtype)
                        g_peer = T.alloc_var(accum_dtype)
                        if is_log_space:
                            g_prefix = T.if_then_else(
                                gb_lane < valid_seqs,
                                g[batch_idx, load_left + gb_lane, bh],
                                -float("inf"),
                            )
                        else:
                            g_prefix = T.log(
                                T.if_then_else(
                                    gb_lane < valid_seqs,
                                    g[batch_idx, load_left + gb_lane, bh],
                                    1e-43,
                                )
                            )
                        for j_shift in T.unroll(5):
                            j_offset = 1 << j_shift
                            g_peer = T.shfl_up(g_prefix, j_offset)
                            if gb_lane_in_warp >= j_offset:
                                g_prefix += g_peer
                        g_shared[i_s % 2, gb_lane] = g_prefix
                        T.lma_wait()
                        T.sync_threads(1, 64)
                        if gb_lane >= 32:
                            g_prefix += g_shared[i_s % 2, 31]
                            g_shared[i_s % 2, gb_lane] = g_prefix
                        T.lma_wait()
                        T.sync_threads(1, 64)
                        for j_s in T.Parallel(block_S):
                            g_exp_rev_shared[i_s % 2, j_s] = (
                                g_shared[i_s % 2, valid_seqs - 1]
                                - g_shared[i_s % 2, j_s]
                            )
                        for j_s in T.Parallel(block_S):
                            g_exp_rev_shared[i_s % 2, j_s] = T.if_then_else(
                                g_exp_rev_shared[i_s % 2, j_s] <= 0,
                                g_exp_rev_shared[i_s % 2, j_s],
                                float("-inf"),
                            )

                        for j_s in T.Parallel(block_S):
                            g_exp_rev_shared[i_s % 2, j_s] = T.exp2(
                                g_exp_rev_shared[i_s % 2, j_s] * 1.4426950408889634
                            )
                            g_exp_shared[i_s % 2, j_s] = T.exp2(
                                g_shared[i_s % 2, j_s] * 1.4426950408889634
                            )
                        T.lma_wait()
                        T.barrier_arrive(gb_is_ready[i_s % 2])

    def _symbol_part(value):
        return (
            str(value)
            .replace("torch.", "")
            .replace(".", "p")
            .replace("-", "m")
            .replace(" ", "_")
        )

    symbol = (
        "tilelang_fused_chunk_gdn_prefill_kernel"
        f"_h{H}_hg{Hg}_dk{DK}_dv{DV}_cs{chunk_size}"
        f"_s{_symbol_part(scale)}"
        f"_q{_symbol_part(qkva_dtype)}"
        f"_g{_symbol_part(g_dtype)}"
        f"_b{_symbol_part(b_dtype)}"
        f"_h0{_symbol_part(h0_dtype)}"
        f"_ht{_symbol_part(ht_dtype)}"
        f"_o{_symbol_part(o_dtype)}"
        f"_seq{_symbol_part(seqlen_dtype)}"
        f"_init{int(use_initial_state)}_final{int(store_final_state)}_var{int(is_varlen)}"
        f"_log{int(is_log_space)}"
    )
    return tilelang_fused_chunk_gdn_prefill_kernel.with_attr("global_symbol", symbol)


def fused_chunk_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    output_o: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    is_log_space: bool = True,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    scale = scale or K ** (-0.5)
    assert K == V == 128
    assert chunk_size == 64
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1

    if cu_seqlens is None:
        real_batch_size = batch_size
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        seqlen_dtype = torch.int32
        is_varlen = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
        )
    h = None

    if output is None:
        o = torch.empty_like(v)
    else:
        if output.shape != v.shape:
            raise ValueError(f"Expected output shape {v.shape}, got {output.shape}.")
        if output.dtype != v.dtype:
            raise ValueError(f"Expected output dtype {v.dtype}, got {output.dtype}.")
        if output.device != v.device:
            raise ValueError(f"Expected output device {v.device}, got {output.device}.")
        o = output

    final_state_shape = (real_batch_size, H, K, V)
    if output_state is None:
        final_state = torch.empty(
            final_state_shape, dtype=torch.float32, device=k.device
        )
    else:
        if output_state.shape != final_state_shape:
            raise ValueError(
                f"Expected output_state shape {final_state_shape}, got {output_state.shape}."
            )
        if output_state.dtype != torch.float32:
            raise ValueError(
                f"Expected output_state dtype torch.float32, got {output_state.dtype}."
            )
        if output_state.device != k.device:
            raise ValueError(
                f"Expected output_state device {k.device}, got {output_state.device}."
            )
        final_state = output_state

    tilelang_fused_chunk_gdn_prefill_kernel = tilelang_fused_chunk_gdn_prefill(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        is_varlen=is_varlen,
        is_log_space=is_log_space,
    )
    tilelang_fused_chunk_gdn_prefill_kernel(
        q,
        k,
        v,
        a,
        g,
        b,
        initial_state,
        cu_seqlens,
        o,
        final_state,
    )

    if not output_final_state:
        final_state = None
    if not output_h:
        h = None
    if not output_o:
        o = None

    return o, h, final_state


if __name__ == "__main__":
    tilelang_fused_chunk_gdn_prefill_kernel = tilelang_fused_chunk_gdn_prefill(
        1,
        1,
        128,
        128,
        64,
        1,
        qkva_dtype=torch.float16,
        g_dtype=torch.float32,
        b_dtype=torch.float32,
        h0_dtype=torch.float32,
        ht_dtype=torch.float32,
        o_dtype=torch.float16,
        seqlen_dtype=torch.int32,
        accum_dtype="float32",
        use_initial_state=True,
        store_final_state=True,
        is_varlen=True,
        is_log_space=True,
    )
