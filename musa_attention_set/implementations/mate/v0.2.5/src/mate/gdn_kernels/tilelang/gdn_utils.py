import torch
from mate.utils import tensor_cache
import tilelang
import tilelang.language as T


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    # TODO: tilelang kernel
    indices = torch.cat(
        [
            torch.arange(n)
            for n in tilelang.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tilelang.jit()
def tilelang_prepare_chunk_offsets(
    chunk_size,
    block_size,
    dtype,
):
    batch_size_plus_1 = T.dynamic("batch_size_plus_1")
    num_threads = min(max(block_size, 32), 128)

    @T.prim_func
    def tilelang_prepare_chunk_offsets_kernel(
        cu_seqlens: T.Tensor([batch_size_plus_1], dtype=dtype),
        chunk_offsets: T.Tensor([batch_size_plus_1], dtype=dtype),
    ):
        with T.Kernel(1, threads=num_threads) as (bb,):
            _batch_size = T.alloc_var("int32")
            _batch_size = batch_size_plus_1 - 1

            seqlen_start_fragment = T.alloc_fragment((block_size), dtype=dtype)
            seqlen_end_fragment = T.alloc_fragment((block_size), dtype=dtype)
            chunk_offset_fragment = T.alloc_fragment((block_size), dtype=dtype)

            T.copy(cu_seqlens[: batch_size_plus_1 - 1], seqlen_start_fragment)
            T.copy(cu_seqlens[1:], seqlen_end_fragment)

            for i in T.Parallel(block_size):
                chunk_offset_fragment[i] = (
                    seqlen_end_fragment[i] - seqlen_start_fragment[i]
                )
                chunk_offset_fragment[i] = (
                    chunk_offset_fragment[i] + chunk_size - 1
                ) // chunk_size
            T.cumsum(src=chunk_offset_fragment, dim=0)

            chunk_offsets[0] = 0
            T.copy(chunk_offset_fragment, chunk_offsets[1:])

    return tilelang_prepare_chunk_offsets_kernel


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    chunk_offsets = torch.empty_like(cu_seqlens)
    tilelang_prepare_chunk_offsets_kernel = tilelang_prepare_chunk_offsets(
        chunk_size=chunk_size,
        block_size=tilelang.next_power_of_2(cu_seqlens.shape[0] - 1),
        dtype=cu_seqlens.dtype,
    )
    tilelang_prepare_chunk_offsets_kernel(cu_seqlens, chunk_offsets)
    return chunk_offsets
