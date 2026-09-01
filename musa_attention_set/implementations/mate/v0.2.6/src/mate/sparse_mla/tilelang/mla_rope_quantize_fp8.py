import functools

import torch


_NOPE_DIM = 512
_ROPE_DIM = 64
_ROWS_PER_CTA = 4
_THREADS = 128


def _pass_configs(tilelang):
    return {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    }


@functools.lru_cache(maxsize=None)
def _mla_rope_quantize_fp8_kernel(*, is_neox: bool, pos_dtype: str, cos_sin_dtype: str):
    import tilelang
    import tilelang.language as T

    nnz = T.dynamic("nnz")
    num_heads = T.dynamic("num_heads")
    max_seq_len = T.dynamic("max_seq_len")

    q_rope_stride_n = T.dynamic("q_rope_stride_n")
    q_rope_stride_h = T.dynamic("q_rope_stride_h")
    q_nope_stride_n = T.dynamic("q_nope_stride_n")
    q_nope_stride_h = T.dynamic("q_nope_stride_h")
    q_rope_out_stride_n = T.dynamic("q_rope_out_stride_n")
    q_rope_out_stride_h = T.dynamic("q_rope_out_stride_h")
    q_nope_out_stride_n = T.dynamic("q_nope_out_stride_n")
    q_nope_out_stride_h = T.dynamic("q_nope_out_stride_h")
    k_rope_stride_n = T.dynamic("k_rope_stride_n")
    k_nope_stride_n = T.dynamic("k_nope_stride_n")
    k_rope_out_stride_n = T.dynamic("k_rope_out_stride_n")
    k_nope_out_stride_n = T.dynamic("k_nope_out_stride_n")

    pos_type = T.int32 if pos_dtype == "int32" else T.int64
    cos_sin_type = T.float32 if cos_sin_dtype == "float32" else T.bfloat16

    @tilelang.jit(
        target="musa",
        pass_configs=_pass_configs(tilelang),
    )
    def kernel(
        q_rope: T.StridedTensor[
            (nnz, num_heads, _ROPE_DIM),
            (q_rope_stride_n, q_rope_stride_h, 1),
            T.bfloat16,
        ],
        k_rope: T.StridedTensor[(nnz, _ROPE_DIM), (k_rope_stride_n, 1), T.bfloat16],
        q_nope: T.StridedTensor[
            (nnz, num_heads, _NOPE_DIM),
            (q_nope_stride_n, q_nope_stride_h, 1),
            T.bfloat16,
        ],
        k_nope: T.StridedTensor[(nnz, _NOPE_DIM), (k_nope_stride_n, 1), T.bfloat16],
        cos_sin_cache: T.Tensor[(max_seq_len, _ROPE_DIM), cos_sin_type],
        pos_ids: T.Tensor[(nnz,), pos_type],
        q_rope_out: T.StridedTensor[
            (nnz, num_heads, _ROPE_DIM),
            (q_rope_out_stride_n, q_rope_out_stride_h, 1),
            T.float8_e4m3fn,
        ],
        k_rope_out: T.StridedTensor[
            (nnz, _ROPE_DIM), (k_rope_out_stride_n, 1), T.float8_e4m3fn
        ],
        q_nope_out: T.StridedTensor[
            (nnz, num_heads, _NOPE_DIM),
            (q_nope_out_stride_n, q_nope_out_stride_h, 1),
            T.float8_e4m3fn,
        ],
        k_nope_out: T.StridedTensor[
            (nnz, _NOPE_DIM), (k_nope_out_stride_n, 1), T.float8_e4m3fn
        ],
        quant_scale_q: T.float32,
        quant_scale_kv: T.float32,
    ) -> None:
        with T.Kernel(
            T.ceildiv(nnz, _ROWS_PER_CTA), num_heads + 1, threads=_THREADS
        ) as (bx, by):
            tx = T.get_thread_binding()
            row = bx * _ROWS_PER_CTA + tx // 32
            lane = tx % 32

            if row < nnz:
                if by < num_heads:
                    for value_idx in T.vectorized(16):
                        col = lane * 16 + value_idx
                        q_nope_out[row, by, col] = (
                            T.float32(q_nope[row, by, col]) * quant_scale_q
                        )

                    q_pos = pos_ids[row]
                    q_cos = T.float32(cos_sin_cache[q_pos, lane])
                    q_sin = T.float32(cos_sin_cache[q_pos, lane + 32])
                    if is_neox:
                        q_left = T.float32(q_rope[row, by, lane])
                        q_right = T.float32(q_rope[row, by, lane + 32])
                        q_rope_out[row, by, lane] = (
                            q_left * q_cos - q_right * q_sin
                        ) * quant_scale_q
                        q_rope_out[row, by, lane + 32] = (
                            q_right * q_cos + q_left * q_sin
                        ) * quant_scale_q
                    else:
                        q_left = T.float32(q_rope[row, by, lane * 2])
                        q_right = T.float32(q_rope[row, by, lane * 2 + 1])
                        q_rope_out[row, by, lane * 2] = (
                            q_left * q_cos - q_right * q_sin
                        ) * quant_scale_q
                        q_rope_out[row, by, lane * 2 + 1] = (
                            q_right * q_cos + q_left * q_sin
                        ) * quant_scale_q
                else:
                    for value_idx in T.vectorized(16):
                        col = lane * 16 + value_idx
                        k_nope_out[row, col] = (
                            T.float32(k_nope[row, col]) * quant_scale_kv
                        )

                    k_pos = pos_ids[row]
                    k_cos = T.float32(cos_sin_cache[k_pos, lane])
                    k_sin = T.float32(cos_sin_cache[k_pos, lane + 32])
                    if is_neox:
                        k_left = T.float32(k_rope[row, lane])
                        k_right = T.float32(k_rope[row, lane + 32])
                        k_rope_out[row, lane] = (
                            k_left * k_cos - k_right * k_sin
                        ) * quant_scale_kv
                        k_rope_out[row, lane + 32] = (
                            k_right * k_cos + k_left * k_sin
                        ) * quant_scale_kv
                    else:
                        k_left = T.float32(k_rope[row, lane * 2])
                        k_right = T.float32(k_rope[row, lane * 2 + 1])
                        k_rope_out[row, lane * 2] = (
                            k_left * k_cos - k_right * k_sin
                        ) * quant_scale_kv
                        k_rope_out[row, lane * 2 + 1] = (
                            k_right * k_cos + k_left * k_sin
                        ) * quant_scale_kv

    return kernel


def run_mla_rope_quantize_fp8(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    quant_scale_q: float,
    quant_scale_kv: float,
    is_neox: bool,
) -> None:
    if pos_ids.dtype == torch.int32:
        pos_dtype = "int32"
    elif pos_ids.dtype == torch.int64:
        pos_dtype = "int64"
    else:
        raise TypeError("pos_ids must have dtype torch.int32 or torch.int64")
    if cos_sin_cache.dtype == torch.float32:
        cos_sin_dtype = "float32"
    elif cos_sin_cache.dtype == torch.bfloat16:
        cos_sin_dtype = "bfloat16"
    else:
        raise TypeError("cos_sin_cache must have dtype torch.float32 or torch.bfloat16")
    if isinstance(quant_scale_q, torch.Tensor):
        raise TypeError("quant_scale_q must be a host scalar")
    if isinstance(quant_scale_kv, torch.Tensor):
        raise TypeError("quant_scale_kv must be a host scalar")
    quant_scale_q = float(quant_scale_q)
    quant_scale_kv = float(quant_scale_kv)
    if q_rope.shape[0] == 0:
        return
    kernel = _mla_rope_quantize_fp8_kernel(
        is_neox=bool(is_neox),
        pos_dtype=pos_dtype,
        cos_sin_dtype=cos_sin_dtype,
    )
    kernel(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        cos_sin_cache,
        pos_ids,
        q_rope_out,
        k_rope_out,
        q_nope_out,
        k_nope_out,
        quant_scale_q,
        quant_scale_kv,
    )


__all__ = ["run_mla_rope_quantize_fp8"]
