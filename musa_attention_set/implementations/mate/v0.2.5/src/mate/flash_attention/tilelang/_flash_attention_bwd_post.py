# ruff: noqa
# type: ignore
import torch

from ._flash_attention_bwd_common import (
    JIT_COMPILE_FLAGS,
    JIT_PASS_CONFIGS,
    T,
    tilelang,
    tir,
)


def to_tilelang_dtype(torch_dtype):
    if torch_dtype == torch.bfloat16:
        return "bfloat16"
    if torch_dtype == torch.float16:
        return "float16"
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


@tilelang.jit(
    out_idx=[],
    pass_configs=JIT_PASS_CONFIGS,
    verbose=False,
    compile_flags=JIT_COMPILE_FLAGS,
)
def singleton_k_dv_ws(
    dim,
    is_varlen,
    is_causal=False,
    has_window_left=False,
    has_window_right=False,
    has_seqused_q=False,
    has_seqused_k=False,
    dtype="bfloat16",
    threads=256,
):
    batch = T.dynamic("batch")
    heads = T.dynamic("heads")
    total_seq_q = T.dynamic("total_seq_q")
    total_seq_kv = T.dynamic("total_seq_kv")
    seqlen_q = T.dynamic("seqlen_q")
    seqlen_k = T.dynamic("seqlen_k")
    do_stride_b = T.dynamic("do_stride_b")
    do_stride_s = T.dynamic("do_stride_s")
    do_stride_h = T.dynamic("do_stride_h")
    dv_stride_b = T.dynamic("dv_stride_b")
    dv_stride_s = T.dynamic("dv_stride_s")
    dv_stride_h = T.dynamic("dv_stride_h")
    runtime_dim = T.dynamic("runtime_dim")
    do_shape = (
        (total_seq_q, heads, runtime_dim)
        if is_varlen
        else (batch, seqlen_q, heads, runtime_dim)
    )
    dv_shape = (
        (total_seq_kv, heads, runtime_dim)
        if is_varlen
        else (batch, seqlen_k, heads, runtime_dim)
    )
    do_strides = (
        (do_stride_s, do_stride_h, 1)
        if is_varlen
        else (do_stride_b, do_stride_s, do_stride_h, 1)
    )
    dv_strides = (
        (dv_stride_s, dv_stride_h, 1)
        if is_varlen
        else (dv_stride_b, dv_stride_s, dv_stride_h, 1)
    )
    do_type = T.StridedTensor(do_shape, do_strides, dtype)
    dv_type = T.StridedTensor(dv_shape, dv_strides, dtype)

    @T.prim_func
    def singleton_k_dv_kernel(
        dO: do_type,  # type: ignore
        dV: dv_type,  # type: ignore
        cu_seq_q: T.Tensor([batch + 1], T.int32),  # type: ignore
        cu_seq_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
        seqused_q: T.Tensor([batch], T.int32),  # type: ignore
        seqused_k: T.Tensor([batch], T.int32),  # type: ignore
        window_size_left: T.int32,  # type: ignore
        window_size_right: T.int32,  # type: ignore
    ):
        with T.Kernel(heads, batch, threads=threads) as (bx, by):
            T.assume(runtime_dim <= dim)
            T.assume(runtime_dim % 8 == 0)
            T.assume(do_stride_s % 8 == 0)
            T.assume(do_stride_h % 8 == 0)
            T.assume(dv_stride_s % 8 == 0)
            T.assume(dv_stride_h % 8 == 0)
            if not is_varlen:
                T.assume(do_stride_b % 8 == 0)
                T.assume(dv_stride_b % 8 == 0)

            begin_seq_q = T.alloc_var(T.int32)
            end_seq_q = T.alloc_var(T.int32)
            begin_seq_kv = T.alloc_var(T.int32)
            end_seq_kv = T.alloc_var(T.int32)
            if is_varlen:
                begin_seq_q = cu_seq_q[by]
                end_seq_q = cu_seq_q[by + 1]
                begin_seq_kv = cu_seq_kv[by]
                end_seq_kv = cu_seq_kv[by + 1]
            else:
                begin_seq_q = by * seqlen_q
                end_seq_q = begin_seq_q + seqlen_q
                begin_seq_kv = by * seqlen_k
                end_seq_kv = begin_seq_kv + seqlen_k
            if has_seqused_q:
                end_seq_q = begin_seq_q + seqused_q[by]
            if has_seqused_k:
                end_seq_kv = begin_seq_kv + seqused_k[by]

            local_seq_q = end_seq_q - begin_seq_q
            local_seq_kv = end_seq_kv - begin_seq_kv
            causal_offset = local_seq_kv - local_seq_q
            tid = T.get_thread_binding()
            dv_accum = T.alloc_local([1], "float")
            dv_accum[0] = 0.0
            if tid < runtime_dim and local_seq_kv > 0:
                for q_local in range(local_seq_q):
                    valid = T.alloc_var(T.bool)
                    valid = True
                    if is_causal:
                        valid = q_local + causal_offset >= 0
                    if has_window_left or has_window_right:
                        valid = valid and (
                            (
                                not has_window_left
                                or 0 >= q_local + causal_offset - window_size_left
                            )
                            and (
                                not has_window_right
                                or 0 <= q_local + causal_offset + window_size_right
                            )
                        )
                    if valid:
                        if is_varlen:
                            dv_accum[0] += tir.Cast(
                                "float32", dO[begin_seq_q + q_local, bx, tid]
                            )
                        else:
                            dv_accum[0] += tir.Cast("float32", dO[by, q_local, bx, tid])
                if is_varlen:
                    dV[begin_seq_kv, bx, tid] = tir.Cast(dtype, dv_accum[0])
                else:
                    dV[by, 0, bx, tid] = tir.Cast(dtype, dv_accum[0])

    return singleton_k_dv_kernel


@tilelang.jit(
    out_idx=[],
    pass_configs=JIT_PASS_CONFIGS,
    verbose=False,
    compile_flags=JIT_COMPILE_FLAGS,
)
def pack_dq_from_accum_ws(
    dim,
    is_varlen,
    is_bhsd=False,
    has_seqused_q=False,
    dtype="bfloat16",
    accum_dtype="float",
    block_M=64,
    block_D=32,
    threads=256,
    use_strided_tensors=True,
):
    max_seq_q = T.dynamic("max_seq_q")
    batch = T.dynamic("batch")
    heads = T.dynamic("heads")
    total_seq_q = T.dynamic("total_seq_q")
    seqlen_q = T.dynamic("seqlen_q")
    dq_stride_b = T.dynamic("dq_stride_b")
    dq_stride_s = T.dynamic("dq_stride_s")
    dq_stride_h = T.dynamic("dq_stride_h")
    dq_dim = T.dynamic("dq_dim")
    num_dim_tiles = (dim + block_D - 1) // block_D
    dq_shape = (
        (total_seq_q, heads, dq_dim) if is_varlen else (batch, seqlen_q, heads, dq_dim)
    )
    dq_strides = (
        (dq_stride_s, dq_stride_h, 1)
        if is_varlen
        else (dq_stride_b, dq_stride_s, dq_stride_h, 1)
    )
    dq_type = T.StridedTensor(dq_shape, dq_strides, dtype)

    @T.prim_func
    def pack_dq_kernel(
        dQ_accum: T.Tensor([batch, heads, max_seq_q, dim], accum_dtype),  # type: ignore
        cu_seq_q: T.Tensor([batch + 1], T.int32),  # type: ignore
        seqused_q: T.Tensor([batch], T.int32),  # type: ignore
        dQ: dq_type,  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(max_seq_q, block_M),
            heads,
            batch * num_dim_tiles,
            threads=threads,
        ) as (bx, by, bz):
            T.assume(dq_dim <= dim)
            T.assume(dq_dim % 8 == 0)
            T.assume(dq_stride_s % 8 == 0)
            T.assume(dq_stride_h % 8 == 0)
            if not is_varlen:
                T.assume(dq_stride_b % 8 == 0)
            batch_idx = bz // num_dim_tiles
            dim_tile = bz % num_dim_tiles
            begin_seq_q = T.alloc_var(T.int32)
            end_seq_q = T.alloc_var(T.int32)
            if is_varlen:
                begin_seq_q = cu_seq_q[batch_idx]
                end_seq_q = cu_seq_q[batch_idx + 1]
            else:
                begin_seq_q = batch_idx * seqlen_q
                end_seq_q = begin_seq_q + seqlen_q
            if has_seqused_q:
                end_seq_q = begin_seq_q + seqused_q[batch_idx]
            local_q_start = bx * block_M
            dim_start = dim_tile * block_D
            dQ_local = T.alloc_fragment([block_M, block_D], accum_dtype)
            dQ_cast = T.alloc_fragment([block_M, block_D], dtype)
            for i, j in T.Parallel(block_M, block_D):
                q_idx = local_q_start + i
                d_idx = dim_start + j
                if q_idx < (end_seq_q - begin_seq_q) and d_idx < dq_dim:
                    T.copy(dQ_accum[batch_idx, by, q_idx, d_idx], dQ_local[i, j])
                else:
                    dQ_local[i, j] = 0.0
            T.copy(dQ_local, dQ_cast)
            for i, j in T.Parallel(block_M, block_D):
                q_idx = local_q_start + i
                d_idx = dim_start + j
                if q_idx < (end_seq_q - begin_seq_q) and d_idx < dq_dim:
                    if is_varlen:
                        T.copy(dQ_cast[i, j], dQ[begin_seq_q + q_idx, by, d_idx])
                    else:
                        T.copy(dQ_cast[i, j], dQ[batch_idx, q_idx, by, d_idx])

    return pack_dq_kernel


@tilelang.jit(
    out_idx=[],
    pass_configs=JIT_PASS_CONFIGS,
    verbose=False,
    compile_flags=JIT_COMPILE_FLAGS,
)
def compute_delta_ws(
    dim,
    is_varlen,
    is_bhsd=False,
    has_seqused_q=False,
    dtype="bfloat16",
    block_M=64,
    block_D=32,
    threads=256,
    use_strided_tensors=True,
):
    total_seq_q = T.dynamic("total_seq_q")
    heads = T.dynamic("heads")
    batch = T.dynamic("batch")
    seqlen_q = T.dynamic("seqlen_q")
    o_stride_b = T.dynamic("o_stride_b")
    o_stride_s = T.dynamic("o_stride_s")
    o_stride_h = T.dynamic("o_stride_h")
    do_stride_b = T.dynamic("do_stride_b")
    do_stride_s = T.dynamic("do_stride_s")
    do_stride_h = T.dynamic("do_stride_h")
    runtime_dim = T.dynamic("runtime_dim")
    num_queries = total_seq_q if is_varlen else batch * seqlen_q
    warps_per_block = max(1, threads // 32)
    values_per_lane = dim // 32
    q_shape = (
        (total_seq_q, heads, runtime_dim)
        if is_varlen
        else (batch, seqlen_q, heads, runtime_dim)
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
    o_type = T.StridedTensor(q_shape, o_strides, dtype)
    do_type = T.StridedTensor(q_shape, do_strides, dtype)

    def q_elem(tensor, q_idx, head, dim_idx):
        if is_varlen:
            return tensor[q_idx, head, dim_idx]
        batch_idx = q_idx // seqlen_q
        local_q_idx = q_idx - batch_idx * seqlen_q
        return tensor[batch_idx, local_q_idx, head, dim_idx]

    @T.prim_func
    def compute_delta_kernel(
        Output: o_type,  # type: ignore
        dO: do_type,  # type: ignore
        cu_seq_q: T.Tensor([batch + 1], T.int32),  # type: ignore
        seqused_q: T.Tensor([batch], T.int32),  # type: ignore
        Delta: T.Tensor([total_seq_q, heads], "float"),  # type: ignore
        max_seq_q_arg: T.int32,  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(
                batch * max_seq_q_arg if has_seqused_q else num_queries,
                warps_per_block,
            ),
            heads,
            threads=threads,
        ) as (bx, by):
            T.assume(runtime_dim <= dim)
            T.assume(runtime_dim % 8 == 0)
            T.assume(o_stride_s % 8 == 0)
            T.assume(o_stride_h % 8 == 0)
            T.assume(do_stride_s % 8 == 0)
            T.assume(do_stride_h % 8 == 0)
            if not is_varlen:
                T.assume(o_stride_b % 8 == 0)
                T.assume(do_stride_b % 8 == 0)
            tid = T.get_thread_binding()
            warp_idx = tid // 32
            lane_idx = tid % 32
            q_slot = bx * warps_per_block + warp_idx
            q_idx = T.alloc_var(T.int32)
            valid_q = T.alloc_var(T.bool)
            q_idx = q_slot
            valid_q = q_slot < num_queries
            if has_seqused_q:
                batch_idx = q_slot // max_seq_q_arg
                local_q = q_slot - batch_idx * max_seq_q_arg
                begin_seq_q = T.alloc_var(T.int32)
                begin_seq_q = 0
                q_idx = 0
                valid_q = False
                if batch_idx < batch:
                    if is_varlen:
                        begin_seq_q = cu_seq_q[batch_idx]
                    else:
                        begin_seq_q = batch_idx * seqlen_q
                    q_idx = begin_seq_q + local_q
                    valid_q = local_q < seqused_q[batch_idx]
            delta_local = T.alloc_local([1], "float")
            delta_local[0] = 0.0
            d_start = lane_idx * values_per_lane
            output_local = T.alloc_fragment([values_per_lane], dtype)
            do_local = T.alloc_fragment([values_per_lane], dtype)
            if valid_q and d_start < runtime_dim:
                for d_local in T.Parallel(values_per_lane):
                    output_local[d_local] = q_elem(Output, q_idx, by, d_start + d_local)
                    do_local[d_local] = q_elem(dO, q_idx, by, d_start + d_local)
                for d_local in range(values_per_lane):
                    delta_local[0] += tir.Cast(
                        "float32", output_local[d_local]
                    ) * tir.Cast("float32", do_local[d_local])
            delta_local[0] += T.shfl_xor(delta_local[0], 16)
            delta_local[0] += T.shfl_xor(delta_local[0], 8)
            delta_local[0] += T.shfl_xor(delta_local[0], 4)
            delta_local[0] += T.shfl_xor(delta_local[0], 2)
            delta_local[0] += T.shfl_xor(delta_local[0], 1)
            if valid_q and lane_idx == 0:
                Delta[q_idx, by] = delta_local[0]

    return compute_delta_kernel


@tilelang.jit(
    out_idx=[],
    pass_configs=JIT_PASS_CONFIGS,
    verbose=False,
    compile_flags=JIT_COMPILE_FLAGS,
)
def reduce_kv_grads_ws(
    head_groups,
    dim,
    is_varlen,
    is_bhsd=False,
    block_T=8,
    block_D=32,
    threads=256,
    use_strided_tensors=True,
    use_static_dims=False,
):
    batch = T.dynamic("batch")
    total_seq_kv = T.dynamic("total_seq_kv")
    seqlen_k = T.dynamic("seqlen_k")
    dk_dim = dim if use_static_dims else T.dynamic("dk_dim")
    dv_dim = dim if use_static_dims else T.dynamic("dv_dim")
    num_kv = total_seq_kv if is_varlen else batch * seqlen_k
    heads_kv = T.dynamic("heads_kv")
    dk_stride_b = T.dynamic("dk_stride_b")
    dk_stride_s = T.dynamic("dk_stride_s")
    dk_stride_h = T.dynamic("dk_stride_h")
    dv_stride_b = T.dynamic("dv_stride_b")
    dv_stride_s = T.dynamic("dv_stride_s")
    dv_stride_h = T.dynamic("dv_stride_h")
    kv_accum_shape = (
        (total_seq_kv, heads_kv * head_groups, dim)
        if is_varlen
        else (batch, seqlen_k, heads_kv * head_groups, dim)
    )
    dk_shape = (
        (total_seq_kv, heads_kv, dk_dim)
        if is_varlen
        else (batch, seqlen_k, heads_kv, dk_dim)
    )
    dv_shape = (
        (total_seq_kv, heads_kv, dv_dim)
        if is_varlen
        else (batch, seqlen_k, heads_kv, dv_dim)
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
    dk_type = T.StridedTensor(dk_shape, dk_strides, "float")
    dv_type = T.StridedTensor(dv_shape, dv_strides, "float")

    def kv_accum_elem(tensor, kv_idx, head, dim_idx):
        if is_varlen:
            return tensor[kv_idx, head, dim_idx]
        batch_idx = kv_idx // seqlen_k
        local_kv_idx = kv_idx - batch_idx * seqlen_k
        return tensor[batch_idx, local_kv_idx, head, dim_idx]

    def kv_grad_elem(tensor, kv_idx, head, dim_idx):
        if is_varlen:
            return tensor[kv_idx, head, dim_idx]
        batch_idx = kv_idx // seqlen_k
        local_kv_idx = kv_idx - batch_idx * seqlen_k
        return tensor[batch_idx, local_kv_idx, head, dim_idx]

    @T.prim_func
    def reduce_kv_grads_kernel(
        dK_accum: T.Tensor(kv_accum_shape, "float"),  # type: ignore
        dV_accum: T.Tensor(kv_accum_shape, "float"),  # type: ignore
        dK: dk_type,  # type: ignore
        dV: dv_type,  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(num_kv, block_T),
            heads_kv,
            T.ceildiv(dim, block_D),
            threads=threads,
        ) as (bx, by, bz):
            T.assume(dk_dim <= dim)
            T.assume(dv_dim <= dim)
            T.assume(dk_dim % 8 == 0)
            T.assume(dv_dim % 8 == 0)
            T.assume(dk_stride_s % 8 == 0)
            T.assume(dk_stride_h % 8 == 0)
            T.assume(dv_stride_s % 8 == 0)
            T.assume(dv_stride_h % 8 == 0)
            if not is_varlen:
                T.assume(dk_stride_b % 8 == 0)
                T.assume(dv_stride_b % 8 == 0)
            k_start = bx * block_T
            d_start = bz * block_D
            dK_local = T.alloc_fragment([block_T, block_D], "float")
            dK_local_odd = T.alloc_fragment([block_T, block_D], "float")
            dV_local = T.alloc_fragment([block_T, block_D], "float")
            dV_local_odd = T.alloc_fragment([block_T, block_D], "float")
            for i, j in T.Parallel(block_T, block_D):
                dK_local[i, j] = 0.0
                dK_local_odd[i, j] = 0.0
                dV_local[i, j] = 0.0
                dV_local_odd[i, j] = 0.0
            for g in T.unroll(head_groups, explicit=True):
                for i, j in T.Parallel(block_T, block_D):
                    kv_idx = k_start + i
                    d_idx = d_start + j
                    hq_idx = by * head_groups + g
                    if kv_idx < num_kv:
                        if d_idx < dk_dim:
                            if g % 2 == 0:
                                dK_local[i, j] += kv_accum_elem(
                                    dK_accum, kv_idx, hq_idx, d_idx
                                )
                            else:
                                dK_local_odd[i, j] += kv_accum_elem(
                                    dK_accum, kv_idx, hq_idx, d_idx
                                )
                        if d_idx < dv_dim:
                            if g % 2 == 0:
                                dV_local[i, j] += kv_accum_elem(
                                    dV_accum, kv_idx, hq_idx, d_idx
                                )
                            else:
                                dV_local_odd[i, j] += kv_accum_elem(
                                    dV_accum, kv_idx, hq_idx, d_idx
                                )
            for i, j in T.Parallel(block_T, block_D):
                kv_idx = k_start + i
                d_idx = d_start + j
                dK_local[i, j] += dK_local_odd[i, j]
                dV_local[i, j] += dV_local_odd[i, j]
                if kv_idx < num_kv and d_idx < dk_dim:
                    T.copy(dK_local[i, j], kv_grad_elem(dK, kv_idx, by, d_idx))
                if kv_idx < num_kv and d_idx < dv_dim:
                    T.copy(dV_local[i, j], kv_grad_elem(dV, kv_idx, by, d_idx))

    return reduce_kv_grads_kernel
