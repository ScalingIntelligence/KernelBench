import torch
import tilelang
import tilelang.language as T

__all__ = ["chunk_local_cumsum"]


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
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
def tilelang_chunk_local_cumsum(
    H,
    chunk_size,
    accum_dtype,
    g_dtype,
    seqlen_dtype,
    is_varlen,
    reverse,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size

    g_shape = (data_batch_size, num_tokens, H)

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
        bb,
        bc,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        g_raw,
        g_cumsum,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        g_fragment = T.alloc_fragment((H, block_S), dtype=accum_dtype)
        gT_fragment = T.alloc_fragment((block_S, H), dtype=g_dtype)
        gT_shared = T.alloc_shared((block_S, H + 1), dtype=g_dtype)

        if right <= seq_end_idx:
            T.copy(g_raw[bb, left:right, 0:H], gT_fragment)
        else:
            for j, i in T.Parallel(block_S, H):
                if left + j < seq_end_idx:
                    gT_fragment[j, i] = g_raw[bb, left + j, i]
                else:
                    gT_fragment[j, i] = 1
        T.copy(gT_fragment, gT_shared[:, :H])

        for i, j in T.Parallel(H, block_S):
            alpha = T.alloc_var(accum_dtype)
            alpha_for_log = T.alloc_var(accum_dtype)
            alpha_is_subnormal = T.alloc_var("bool")
            alpha = T.cast(gT_shared[j, i], accum_dtype)
            alpha_is_subnormal = alpha > 0.0 and alpha < 1.1754943508222875e-38
            alpha_for_log = T.if_then_else(
                alpha_is_subnormal,
                alpha * 4294967296.0,
                alpha,
            )
            g_fragment[i, j] = T.log(alpha_for_log) - T.if_then_else(
                alpha_is_subnormal,
                22.18070977791825,
                0.0,
            )

        T.cumsum(g_fragment, dim=1, reverse=reverse)

        for i, j in T.Parallel(H, block_S):
            gT_shared[j, i] = g_fragment[i, j]

        T.copy(gT_shared[:, :H], gT_fragment)
        if right <= seq_end_idx:
            T.copy(gT_fragment, g_cumsum[bb, left:right, 0:H])
        else:
            for j, i in T.Parallel(block_S, H):
                if left + j < seq_end_idx:
                    g_cumsum[bb, left + j, i] = gT_fragment[j, i]

    if is_varlen:

        @T.prim_func
        def tilelang_chunk_local_cumsum_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
        ):
            launch_num_chunks = T.ceildiv(num_tokens, block_S)
            with T.Kernel(launch_num_chunks, threads=128) as (bc,):
                bb = 0
                global_left = T.alloc_var("int32")
                global_right = T.alloc_var("int32")
                query_idx = T.alloc_var("int32")
                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")
                rel_idx = T.alloc_var("int32")
                chunk_left = T.alloc_var("int32")

                global_left = bc * block_S
                global_right = T.min(global_left + block_S, num_tokens)

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
                        bb,
                        bc,
                        batch_idx,
                        chunk_idx,
                        seq_start_idx,
                        seq_end_idx,
                        g_raw,
                        g_cumsum,
                    )

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
        def tilelang_chunk_local_cumsum_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks, threads=128) as (bc,):
                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    bc,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    g_raw,
                    g_cumsum,
                )

    return tilelang_chunk_local_cumsum_kernel


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    reverse: bool = False,
):
    batch_size, num_tokens, H = g.shape
    assert g.stride(-1) == 1

    if cu_seqlens is None:
        num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    g_cumsum = torch.empty_like(g)

    tilelang_chunk_local_cumsum_kernel = tilelang_chunk_local_cumsum(
        H,
        chunk_size,
        g_dtype=g.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
        reverse=reverse,
    )
    if is_varlen:
        tilelang_chunk_local_cumsum_kernel(g, cu_seqlens, g_cumsum)
    else:
        tilelang_chunk_local_cumsum_kernel(g, g_cumsum, num_chunks)

    return g_cumsum
