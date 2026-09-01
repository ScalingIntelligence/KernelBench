import torch
import tilelang
import tilelang.language as T

__all__ = ["gdn_l2norm_"]


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
    compile_flags=[
        "-O3",
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
def tilelang_gdn_l2norm(qkva_dtype, rows_per_block, lanes_per_row):
    batch = T.dynamic("batch")
    num_tokens = T.dynamic("num_tokens")
    num_heads = T.dynamic("num_heads")
    D = 128
    vec_size = tilelang.cdiv(D, lanes_per_row)
    num_shuffles = lanes_per_row.bit_length() - 1
    threads = rows_per_block * lanes_per_row

    x_stride_b = T.dynamic("x_stride_b")
    x_stride_t = T.dynamic("x_stride_t")
    x_stride_h = T.dynamic("x_stride_h")
    x_shape = (batch, num_tokens, num_heads, D)
    x_strides = (x_stride_b, x_stride_t, x_stride_h, 1)

    @T.prim_func
    def tilelang_gdn_l2norm_kernel(
        x: T.StridedTensor(x_shape, x_strides, qkva_dtype),
        eps: T.float32,
    ):
        total_rows = batch * num_tokens * num_heads
        num_blocks = T.ceildiv(total_rows, rows_per_block)

        with T.Kernel(num_blocks, threads=threads) as (block_idx,):
            tid = T.get_thread_binding()
            lane = tid % lanes_per_row
            row_in_block = tid // lanes_per_row
            row = block_idx * rows_per_block + row_in_block

            batch_idx = T.alloc_var("int32")
            token_idx = T.alloc_var("int32")
            head_idx = T.alloc_var("int32")
            j_d = T.alloc_var("int32")
            x_local = T.alloc_local((vec_size,), dtype="float32")
            norm2_local = T.alloc_local((1,), dtype="float32")

            if row < total_rows:
                batch_idx = row // (num_tokens * num_heads)
                token_idx = (row // num_heads) % num_tokens
                head_idx = row % num_heads
                norm2_local[0] = 0.0

                for i in T.vectorized(vec_size):
                    j_d = lane * vec_size + i
                    x_local[i] = T.cast(
                        x[batch_idx, token_idx, head_idx, j_d], "float32"
                    )
                    norm2_local[0] += x_local[i] * x_local[i]

                for offset in T.unroll(num_shuffles):
                    norm2_local[0] += T.shfl_xor(
                        norm2_local[0], (lanes_per_row // 2) >> offset
                    )

                for i in T.vectorized(vec_size):
                    j_d = lane * vec_size + i
                    x[batch_idx, token_idx, head_idx, j_d] = T.cast(
                        x_local[i] * T.rsqrt(norm2_local[0] + eps),
                        qkva_dtype,
                    )

    def _symbol_part(value):
        return (
            str(value)
            .replace("torch.", "")
            .replace(".", "p")
            .replace("-", "m")
            .replace(" ", "_")
        )

    symbol = (
        f"tilelang_gdn_l2norm_kernel_q{_symbol_part(qkva_dtype)}"
        f"_r{rows_per_block}_l{lanes_per_row}"
    )
    return tilelang_gdn_l2norm_kernel.with_attr("global_symbol", symbol)


def gdn_l2norm_(
    x: torch.Tensor,
    eps: float = 1e-6,
    rows_per_block: int = 16,
    lanes_per_row: int = 4,
) -> torch.Tensor:
    if x.dim() != 4:
        raise RuntimeError("gdn_l2norm_ expects a rank-4 tensor [B, T, H, 128].")
    if x.shape[-1] != 128:
        raise RuntimeError("gdn_l2norm_ only supports D=128.")
    if x.dtype == torch.float32:
        raise RuntimeError("gdn_l2norm_ expects fp16 or bf16 input.")
    if x.stride(-1) != 1:
        raise RuntimeError("gdn_l2norm_ requires a contiguous last dimension.")
    if lanes_per_row not in (4, 8, 16, 32):
        raise RuntimeError("gdn_l2norm_ requires lanes_per_row in {4, 8, 16, 32}.")
    if rows_per_block < 1 or rows_per_block * lanes_per_row > 1024:
        raise RuntimeError(
            "gdn_l2norm_ requires 1 <= rows_per_block * lanes_per_row <= 1024."
        )

    kernel = tilelang_gdn_l2norm(
        qkva_dtype=x.dtype,
        rows_per_block=rows_per_block,
        lanes_per_row=lanes_per_row,
    )
    kernel(x, float(eps))
    return x
