from typing import Optional

import torch
import tilelang
import tilelang.language as T

__all__ = ["kkt_solve"]


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    },
    compile_flags=[
        "-Od3",
        "-fno-signed-zeros",
        "-mllvm",
        "-mtgpu-if-convert=1",
        "-mllvm",
        "-misched=mtgpu-max-ilp",
        "-mllvm",
        "-mtgpu-tiny-offset-hint=1",
        "-mllvm",
        "-misched-recompute-slotindex=1",
        "-mllvm",
        "-mtgpu-combine-fop-instr=1",
    ],
)
def tilelang_kkt_solve(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    b_dtype,
    seqlen_dtype,
    is_varlen,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)
    heads_per_k = H // Hg

    k_stride_b = T.dynamic("k_stride_b")
    k_stride_t = T.dynamic("k_stride_t")
    k_stride_h = T.dynamic("k_stride_h")
    k_strides = (k_stride_b, k_stride_t, k_stride_h, 1)

    @T.macro
    def locate_sequence(token_idx, cu_seqlens):
        lo = T.alloc_var("int32")
        hi = T.alloc_var("int32")
        mid = T.alloc_var("int32")
        lo = 0
        hi = real_batch_size
        while lo < hi:
            mid = (lo + hi) // 2
            if cu_seqlens[mid + 1] <= token_idx:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @T.macro
    def kernel_body(
        bc,
        bh,
        batch_idx,
        chunk_idx,
        body_iter,
        seq_start_idx,
        seq_end_idx,
        k,
        b,
        a,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S
        data_batch_idx = 0 if is_varlen else batch_idx
        bhg = bh // heads_per_k
        barrier_phase = body_iter % 2

        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
        a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        valid_seqs = T.min(seq_end_idx - left, block_S)

        a32d_shared = T.alloc_shared((2, 33, 32), dtype=accum_dtype)
        a32i0_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32i1_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32o1_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32o2_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)

        T.annotate_layout(
            {a32d_shared: tilelang.layout.make_linear_layout(a32d_shared)}
        )

        k_load_is_ready = T.alloc_barrier(arrive_count=128)
        diag_input_is_ready = T.alloc_barrier(arrive_count=128)
        diag_inverse_is_ready = T.alloc_barrier(arrive_count=128)
        tx = T.get_thread_binding()
        T.tma_copy(
            k[data_batch_idx, left:right, bhg, 0:DK],
            k_shared,
            barrier=k_load_is_ready,
        )
        T.barrier_arrive(k_load_is_ready)
        T.barrier_wait(k_load_is_ready, barrier_phase)
        # A = K @ K^T
        T.gemm(k_shared, k_shared, a64_fragment, transpose_B=True, clear_accum=True)
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s < j_t:
                a64_fragment[j_s, j_t] = 0.0

        # Load b
        if right <= seq_end_idx:
            for j_s in T.Parallel(block_S):
                b_shared[j_s] = b[data_batch_idx, left + j_s, bh]
        else:
            for j_s in T.Parallel(block_S):
                if left + j_s < seq_end_idx:
                    b_shared[j_s] = b[data_batch_idx, left + j_s, bh]
                else:
                    b_shared[j_s] = 0
        # Prepare the two 32x32 diagonal blocks.
        for j_s, j_t in T.Parallel(block_S, block_S):
            if (j_s // 32) == (j_t // 32):
                a32d_shared[j_s // 32, j_s % 32, j_t % 32] = T.if_then_else(
                    j_s == j_t,
                    1.0,
                    -a64_fragment[j_s, j_t] * b_shared[j_s],
                )
        T.barrier_arrive(diag_input_is_ready)
        T.barrier_wait(diag_input_is_ready, barrier_phase)

        # Invert the two 32x32 diagonal blocks directly. Each warp owns one
        # block, with one lane carrying one row.
        if tx < 64:
            diag_block = tx // 32
            diag_row_idx = tx % 32
            diag_row = T.alloc_local([32], accum_dtype)

            for k_t in T.vectorized(32):
                diag_row[k_t] = a32d_shared[diag_block, diag_row_idx, k_t]

            for src_row in T.unroll(31):
                row_scale = T.if_then_else(
                    diag_row_idx > src_row, diag_row[src_row], 0.0
                )
                for k_t in T.unroll(src_row):
                    src_row_value = T.shfl_sync(0xFFFFFFFF, diag_row[k_t], src_row, 32)
                    diag_row[k_t] += row_scale * src_row_value

            if diag_block == 0:
                for k_t in T.vectorized(32):
                    a32i0_shared[diag_row_idx, k_t] = diag_row[k_t]
            else:
                for k_t in T.vectorized(32):
                    a32i1_shared[diag_row_idx, k_t] = diag_row[k_t]

        T.barrier_arrive(diag_inverse_is_ready)
        T.barrier_wait(diag_inverse_is_ready, barrier_phase)

        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s >= 32 and j_t < 32:
                a32o1_shared[j_s - 32, j_t] = -a64_fragment[j_s, j_t] * b_shared[j_s]
        T.sync_threads()

        # Combine the two 32x32 inverse diagonal blocks into the full
        # 64x64 inverse.
        T.gemm(a32i1_shared, a32o1_shared, a32o_fragment, clear_accum=True, wg_wait=-1)
        T.warpgroup_commit_batch()
        for k_s, k_t in T.Parallel(32, 32):
            if k_s < valid_seqs:
                a[data_batch_idx, left + k_s, bh, k_t] = a32i0_shared[k_s, k_t]
        T.warpgroup_wait(0)
        T.copy(a32o_fragment, a32o2_shared)
        T.sync_threads()
        T.gemm(a32o2_shared, a32i0_shared, a32o_fragment, clear_accum=True, wg_wait=-1)
        T.warpgroup_commit_batch()
        for k_s, k_t in T.Parallel(32, 32):
            if 32 + k_s < valid_seqs:
                a[data_batch_idx, left + 32 + k_s, bh, 32 + k_t] = a32i1_shared[
                    k_s, k_t
                ]
        T.warpgroup_wait(0)
        for k_s, k_t in T.Parallel(32, 32):
            if 32 + k_s < valid_seqs:
                a[data_batch_idx, left + 32 + k_s, bh, k_t] = a32o_fragment[k_s, k_t]
        for k_s, k_t in T.Parallel(32, 32):
            if k_s < valid_seqs:
                a[data_batch_idx, left + k_s, bh, 32 + k_t] = 0

    if is_varlen:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.StridedTensor(k_shape, k_strides, qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
        ):
            launch_num_chunks = T.ceildiv(num_tokens, block_S)
            with T.Kernel(launch_num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H

                global_left = T.alloc_var("int32")
                global_right = T.alloc_var("int32")
                query_idx = T.alloc_var("int32")
                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")
                rel_idx = T.alloc_var("int32")
                chunk_left = T.alloc_var("int32")
                body_iter = T.alloc_var("int32")

                global_left = bc * block_S
                global_right = T.min(global_left + block_S, num_tokens)
                body_iter = 0

                query_idx = global_left
                batch_idx = locate_sequence(query_idx, cu_seqlens)
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]
                rel_idx = query_idx - seq_start_idx
                chunk_idx = T.ceildiv(rel_idx, block_S)
                chunk_left = seq_start_idx + chunk_idx * block_S
                while chunk_left >= seq_end_idx and seq_end_idx < num_tokens:
                    batch_idx += 1
                    seq_start_idx = seq_end_idx
                    seq_end_idx = cu_seqlens[batch_idx + 1]
                    chunk_idx = 0
                    chunk_left = seq_start_idx

                while chunk_left < global_right:
                    kernel_body(
                        bc,
                        bh,
                        batch_idx,
                        chunk_idx,
                        body_iter,
                        seq_start_idx,
                        seq_end_idx,
                        k,
                        b,
                        a,
                    )
                    body_iter += 1

                    query_idx = chunk_left + block_S
                    if query_idx >= seq_end_idx:
                        query_idx = seq_end_idx
                        while seq_end_idx <= query_idx and seq_end_idx < num_tokens:
                            batch_idx += 1
                            seq_start_idx = seq_end_idx
                            seq_end_idx = cu_seqlens[batch_idx + 1]
                        if query_idx < seq_end_idx:
                            chunk_idx = 0
                            chunk_left = seq_start_idx
                        else:
                            chunk_left = global_right
                    else:
                        chunk_idx += 1
                        chunk_left += block_S

    else:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.StridedTensor(k_shape, k_strides, qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                batch_idx = bc % data_batch_size
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bc,
                    bh,
                    batch_idx,
                    chunk_idx,
                    0,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    return tilelang_kkt_solve_kernel


def kkt_solve(
    k: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H = b.shape
    assert K == 128
    assert chunk_size == 64
    assert k.stride(-1) == 1

    if cu_seqlens is None:
        num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    a = torch.empty(
        (batch_size, num_tokens, H, chunk_size), dtype=k.dtype, device=k.device
    )

    tilelang_kkt_solve_kernel = tilelang_kkt_solve(
        H,
        Hg,
        K,
        chunk_size,
        qkva_dtype=k.dtype,
        b_dtype=b.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
    )
    if is_varlen:
        tilelang_kkt_solve_kernel(k, b, cu_seqlens, a)
    else:
        tilelang_kkt_solve_kernel(k, b, a, num_chunks)

    return a
