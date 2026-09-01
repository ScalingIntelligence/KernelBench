"""TileLang backend for the VK-layout GDN MTP path."""

import functools

import tilelang
import tilelang.language as T
import torch

__all__ = [
    "run_gated_delta_rule_mtp_vk_fp32",
]

_LOG2E = 1.4426950408889634
_WARP_SIZE = 32
_WARP_REDUCE_STEPS = _WARP_SIZE.bit_length() - 1
_QK_SHARED_PAD = 8
_DEFAULT_THREADS = 128
_QK_L2NORM_EPS = 1e-6
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0


def _exp2_f32(value):
    return T.exp2(value * _LOG2E)


def _dtype_name(dtype) -> str | None:
    return None if dtype is None else str(dtype).split(".")[-1]


_USE_SMEM_V = True
_NUM_STAGES = 2
_NUM_GRPS = 2


@functools.lru_cache(maxsize=128)
def _get_mtp_config(
    batch_size: int,
    seq_len: int,
    num_v_heads: int = 64,
    v_dim: int = 128,
    cache_intermediate_states: bool = False,
    state_dtype: torch.dtype | str | None = None,
) -> tuple[int, int, int]:
    """Return ``(tile_v, vec_size, ilp_rows)`` for the FP32-state VK MTP path."""

    # MUSA-tuned smem-only defaults, seeded from FlashInfer's work_units policy.
    work_units = batch_size * num_v_heads
    state_dtype_name = _dtype_name(state_dtype)
    vec_size = 4
    ilp_rows = 2

    if (
        cache_intermediate_states
        and state_dtype_name == "bfloat16"
        and seq_len == 3
        and work_units <= 16
    ):
        tile_v = 8
        ilp_rows = 1
    elif cache_intermediate_states and seq_len == 2 and work_units >= 8192:
        tile_v = 128
        ilp_rows = 8
    elif cache_intermediate_states and seq_len == 2 and work_units >= 4096:
        tile_v = 128
    elif cache_intermediate_states and seq_len == 3 and work_units >= 4096:
        tile_v = 64
        ilp_rows = 4
    elif not cache_intermediate_states and seq_len <= 2 and work_units >= 2048:
        tile_v = 128
    elif work_units <= 64:
        tile_v = 16
    elif work_units <= 128:
        tile_v = 32
    else:
        tile_v = 64

    min_tile_v = (_DEFAULT_THREADS // _WARP_SIZE) * ilp_rows * _NUM_GRPS
    tile_v = max(tile_v, min_tile_v)

    return min(tile_v, v_dim), vec_size, ilp_rows


@functools.lru_cache(maxsize=128)
def _should_disable_index_type_promotion(
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    state_dtype: torch.dtype | str,
    cache_intermediate_states: bool,
    disable_state_update: bool,
) -> bool:
    work_units = batch_size * num_v_heads
    return (
        cache_intermediate_states
        and disable_state_update
        and _dtype_name(state_dtype) == "bfloat16"
        and seq_len in (2, 3)
        and work_units <= 4096
    )


@functools.lru_cache(maxsize=32)
def _get_mtp_fp32_vk_smem_kernel(
    seq_len: int,
    qk_head: int,
    head: int,
    dim_k: int,
    dim_v: int,
    input_dtype: str,
    output_dtype: str,
    dt_bias_dtype: str,
    state_dtype: str,
    use_qk_l2norm: bool,
    cache_intermediate_states: bool,
    disable_state_update: bool,
    use_identity_state_indices: bool,
    tile_v: int,
    ilp_rows: int,
    disable_index_type_promotion: bool = False,
):
    if dim_v % tile_v != 0:
        raise ValueError(f"dim_v={dim_v} must be divisible by tile_v={tile_v}.")
    if ilp_rows not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported ilp_rows={ilp_rows}. Expected 1, 2, 4, or 8.")
    if state_dtype not in ("float32", "bfloat16"):
        raise ValueError(
            f"Unsupported state_dtype={state_dtype}. Expected float32 or bfloat16."
        )
    if _DEFAULT_THREADS % _WARP_SIZE != 0:
        raise ValueError(
            f"_DEFAULT_THREADS={_DEFAULT_THREADS} must be divisible by warp size {_WARP_SIZE}."
        )
    rows_per_state_tile = (_DEFAULT_THREADS // _WARP_SIZE) * ilp_rows * _NUM_GRPS
    if tile_v % rows_per_state_tile != 0:
        raise ValueError(
            f"tile_v={tile_v} must be divisible by rows_per_state_tile={rows_per_state_tile}."
        )

    qkv_dtype = input_dtype
    gate_batch_dtype = input_dtype
    gate_vec_dtype = "float32"
    dt_bias_vec_dtype = dt_bias_dtype
    state_storage_dtype = state_dtype
    accum_dtype = "float32"
    index_dtype = "int32"

    head_group_size = head // qk_head
    num_v_tiles = dim_v // tile_v
    vec_size = dim_k // _WARP_SIZE

    batch = T.dynamic("batch")
    pool_size = T.dynamic("pool_size")
    intermediate_pool_size = T.dynamic("intermediate_pool_size")
    intermediate_steps = T.dynamic("intermediate_steps")
    q_stride_b = T.dynamic("q_stride_b")
    q_stride_t = T.dynamic("q_stride_t")
    q_stride_h = T.dynamic("q_stride_h")
    q_stride_k = T.dynamic("q_stride_k")
    k_stride_b = T.dynamic("k_stride_b")
    k_stride_t = T.dynamic("k_stride_t")
    k_stride_h = T.dynamic("k_stride_h")
    k_stride_k = T.dynamic("k_stride_k")
    v_stride_b = T.dynamic("v_stride_b")
    v_stride_t = T.dynamic("v_stride_t")
    v_stride_h = T.dynamic("v_stride_h")
    v_stride_v = T.dynamic("v_stride_v")
    a_stride_b = T.dynamic("a_stride_b")
    a_stride_t = T.dynamic("a_stride_t")
    a_stride_h = T.dynamic("a_stride_h")
    b_stride_b = T.dynamic("b_stride_b")
    b_stride_t = T.dynamic("b_stride_t")
    b_stride_h = T.dynamic("b_stride_h")
    o_stride_b = T.dynamic("o_stride_b")
    o_stride_t = T.dynamic("o_stride_t")
    o_stride_h = T.dynamic("o_stride_h")
    o_stride_v = T.dynamic("o_stride_v")
    state_indices_stride_b = T.dynamic("state_indices_stride_b")

    q_shape = (batch, seq_len, qk_head, dim_k)
    k_shape = (batch, seq_len, qk_head, dim_k)
    v_shape = (batch, seq_len, head, dim_v)
    gate_shape = (batch, seq_len, head)
    state_indices_shape = (batch,)
    output_shape = (batch, seq_len, head, dim_v)
    q_strides = (q_stride_b, q_stride_t, q_stride_h, q_stride_k)
    k_strides = (k_stride_b, k_stride_t, k_stride_h, k_stride_k)
    v_strides = (v_stride_b, v_stride_t, v_stride_h, v_stride_v)
    a_strides = (a_stride_b, a_stride_t, a_stride_h)
    b_strides = (b_stride_b, b_stride_t, b_stride_h)
    state_indices_strides = (state_indices_stride_b,)
    output_strides = (o_stride_b, o_stride_t, o_stride_h, o_stride_v)
    intermediate_shape = (
        [intermediate_pool_size, intermediate_steps, head, dim_v, dim_k]
        if cache_intermediate_states
        else [1, 1, 1, 1, 1]
    )

    jit_pass_configs = {
        # tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
    if disable_index_type_promotion:
        jit_pass_configs[tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION] = True

    @tilelang.jit(
        pass_configs=jit_pass_configs,
        compile_flags=[
            "-O3",
            "-mllvm",
            "-misched=mtgpu-max-ilp",
            "-mllvm",
            "-mtgpu-if-convert=1",
            "-mllvm",
            "-mtgpu-tiny-offset-hint=1",
            "-mllvm",
            "-misched-recompute-slotindex=1",
        ],
    )
    def _mtp_func():
        num_warps = _DEFAULT_THREADS // _WARP_SIZE

        @T.prim_func
        def gated_deltanet_mtp_fp32_vk_smem(
            q: T.StridedTensor(q_shape, q_strides, qkv_dtype),
            k: T.StridedTensor(k_shape, k_strides, qkv_dtype),
            v: T.StridedTensor(v_shape, v_strides, qkv_dtype),
            A_log: T.Tensor([head], gate_vec_dtype),
            a: T.StridedTensor(gate_shape, a_strides, gate_batch_dtype),
            dt_bias: T.Tensor([head], dt_bias_vec_dtype),
            b: T.StridedTensor(gate_shape, b_strides, gate_batch_dtype),
            scale_value: T.float32,
            initial_state: T.Tensor(
                [pool_size, head, dim_v, dim_k], state_storage_dtype
            ),
            state_indices: T.StridedTensor(
                state_indices_shape, state_indices_strides, index_dtype
            ),
            intermediate_states: T.Tensor(intermediate_shape, state_storage_dtype),
            output: T.StridedTensor(output_shape, output_strides, output_dtype),
        ):
            with T.Kernel(batch * num_v_tiles, head, threads=_DEFAULT_THREADS) as (
                bx,
                hid,
            ):
                bid = bx // num_v_tiles
                v_tile_idx = bx % num_v_tiles
                global_v_base = v_tile_idx * tile_v

                tid = T.get_thread_binding()
                lane = tid % _WARP_SIZE
                warp = tid // _WARP_SIZE
                k_start = lane * vec_size
                qk_hid = hid // head_group_size
                if use_identity_state_indices:
                    state_slot = bid
                else:
                    state_slot = state_indices[bid]

                rows_per_group = num_warps * ilp_rows
                rows_per_state_tile = rows_per_group * _NUM_GRPS
                state_smem = T.alloc_shared(
                    [_NUM_STAGES, rows_per_state_tile, dim_k], state_storage_dtype
                )
                state_load_barrier_counts = [_DEFAULT_THREADS] * _NUM_STAGES
                state_load_mbars = T.alloc_barrier(state_load_barrier_counts)
                next_stage = T.alloc_var(T.int32)

                q_shared = T.alloc_shared(
                    [seq_len, dim_k + _QK_SHARED_PAD], accum_dtype
                )
                k_shared = T.alloc_shared(
                    [seq_len, dim_k + _QK_SHARED_PAD], accum_dtype
                )
                v_shared = T.alloc_shared([seq_len, tile_v], accum_dtype)
                output_shared = T.alloc_shared([seq_len, tile_v], output_dtype)
                alpha_shared = T.alloc_shared([seq_len], accum_dtype)
                beta_shared = T.alloc_shared([seq_len], accum_dtype)

                q_reg = T.alloc_local([vec_size], accum_dtype)
                k_reg = T.alloc_local([vec_size], accum_dtype)
                h_reg = T.alloc_local([ilp_rows * vec_size], accum_dtype)
                v_value = T.alloc_local([ilp_rows], accum_dtype)
                v_new = T.alloc_local([ilp_rows], accum_dtype)

                q_sum = T.alloc_local([1], accum_dtype)
                k_sum = T.alloc_local([1], accum_dtype)
                sum_hk = T.alloc_local([ilp_rows], accum_dtype)
                sum_hq = T.alloc_local([ilp_rows], accum_dtype)
                a_log_reg = T.alloc_local([1], accum_dtype)
                dt_bias_reg = T.alloc_local([1], accum_dtype)

                a_log_reg[0] = T.cast(A_log[hid], accum_dtype)
                dt_bias_reg[0] = T.cast(dt_bias[hid], accum_dtype)

                if state_slot >= 0:
                    T.tma_copy(
                        initial_state[state_slot, hid, global_v_base, 0],
                        state_smem[0, :, :],
                        barrier=state_load_mbars[0],
                        inner_cache_policy="cache_none",
                        outer_cache_policy="cache_persist",
                    )
                    T.mbarrier_arrive(mbarrier=state_load_mbars[0])

                    if warp == 0:
                        for t in T.serial(seq_len):
                            for i in T.vectorized(vec_size):
                                q_reg[i] = T.cast(
                                    q[bid, t, qk_hid, k_start + i], accum_dtype
                                )
                                k_reg[i] = T.cast(
                                    k[bid, t, qk_hid, k_start + i], accum_dtype
                                )

                            if use_qk_l2norm:
                                q_sum[0] = 0.0
                                k_sum[0] = 0.0
                                # change to vectorize cause cal err
                                for i in T.unroll(vec_size):
                                    q_sum[0] += q_reg[i] * q_reg[i]
                                    k_sum[0] += k_reg[i] * k_reg[i]

                                for offset in T.unroll(_WARP_REDUCE_STEPS):
                                    mask = (_WARP_SIZE // 2) >> offset
                                    q_sum[0] += T.shfl_xor(q_sum[0], mask)
                                    k_sum[0] += T.shfl_xor(k_sum[0], mask)

                                q_sum[0] = (
                                    T.rsqrt(q_sum[0] + _QK_L2NORM_EPS) * scale_value
                                )
                                k_sum[0] = T.rsqrt(k_sum[0] + _QK_L2NORM_EPS)

                                for i in T.vectorized(vec_size):
                                    q_reg[i] = q_reg[i] * q_sum[0]
                                    k_reg[i] = k_reg[i] * k_sum[0]
                            else:
                                for i in T.vectorized(vec_size):
                                    q_reg[i] = q_reg[i] * scale_value

                            for i in T.vectorized(vec_size):
                                q_shared[t, k_start + i] = q_reg[i]
                                k_shared[t, k_start + i] = k_reg[i]

                            a_val = T.cast(a[bid, t, hid], accum_dtype)
                            b_val = T.cast(b[bid, t, hid], accum_dtype)

                            x = a_val + dt_bias_reg[0]
                            beta_x = _SOFTPLUS_BETA * x
                            softplus_x = T.if_then_else(
                                beta_x <= _SOFTPLUS_THRESHOLD,
                                (1.0 / _SOFTPLUS_BETA) * T.log(1.0 + _exp2_f32(beta_x)),
                                x,
                            )
                            g_val = -_exp2_f32(a_log_reg[0]) * softplus_x
                            alpha_shared[t] = _exp2_f32(g_val)
                            beta_shared[t] = 1.0 / (1.0 + _exp2_f32(-b_val))

                    if _USE_SMEM_V:
                        for t in T.serial(seq_len):
                            if tid < tile_v:
                                v_shared[t, tid] = T.cast(
                                    v[bid, t, hid, global_v_base + tid],
                                    accum_dtype,
                                )

                    T.sync_threads()

                    for row in T.serial(0, tile_v // rows_per_state_tile):
                        row_base = row * rows_per_state_tile
                        stage = row % _NUM_STAGES
                        next_stage = (row + 1) % _NUM_STAGES
                        parity = (row // _NUM_STAGES) & 1
                        next_row_idx = row_base + rows_per_state_tile

                        if (
                            _NUM_STAGES > 1
                            and next_row_idx + rows_per_state_tile - 1 < tile_v
                        ):
                            # Multi-stage path overlaps the next state tile load
                            # with the current tile wait/compute.
                            if row > 0:
                                # The next TME load may reuse a shared stage.
                                T.sync_threads()

                            global_next_load_row = global_v_base + next_row_idx
                            T.tma_copy(
                                initial_state[state_slot, hid, global_next_load_row, 0],
                                state_smem[next_stage, :, :],
                                barrier=state_load_mbars[next_stage],
                                inner_cache_policy="cache_none",
                                outer_cache_policy="cache_persist",
                            )
                            T.mbarrier_arrive(mbarrier=state_load_mbars[next_stage])

                        T.mbarrier_wait_parity(
                            mbarrier=state_load_mbars[stage], parity=parity
                        )

                        for grp in T.serial(_NUM_GRPS):
                            row_idx = row_base + grp * rows_per_group + warp * ilp_rows
                            if row_idx + ilp_rows - 1 < tile_v:
                                for r in T.unroll(ilp_rows):
                                    local_row = (
                                        grp * rows_per_group + warp * ilp_rows + r
                                    )
                                    for i in T.vectorized(vec_size):
                                        h_reg[r * vec_size + i] = T.cast(
                                            state_smem[
                                                stage,
                                                local_row,
                                                k_start + i,
                                            ],
                                            accum_dtype,
                                        )

                                for t in T.serial(seq_len):
                                    alpha_val = alpha_shared[t]
                                    beta_val = beta_shared[t]
                                    for i in T.vectorized(vec_size):
                                        kk = k_start + i
                                        q_reg[i] = q_shared[t, kk]
                                        k_reg[i] = k_shared[t, kk]

                                    for r in T.unroll(ilp_rows):
                                        sum_hk[r] = 0.0
                                        sum_hq[r] = 0.0

                                    # to vec will cause perf regress
                                    for r in T.unroll(ilp_rows):
                                        for i in T.unroll(vec_size):
                                            h_reg[r * vec_size + i] = (
                                                h_reg[r * vec_size + i] * alpha_val
                                            )
                                            sum_hk[r] += (
                                                h_reg[r * vec_size + i] * k_reg[i]
                                            )

                                    for offset in T.unroll(_WARP_REDUCE_STEPS):
                                        mask = (_WARP_SIZE // 2) >> offset
                                        for r in T.unroll(ilp_rows):
                                            sum_hk[r] += T.shfl_xor(sum_hk[r], mask)

                                    for r in T.vectorized(ilp_rows):
                                        global_row = global_v_base + row_idx + r
                                        if _USE_SMEM_V:
                                            v_value[r] = v_shared[t, row_idx + r]
                                        else:
                                            v_value[r] = T.cast(
                                                v[bid, t, hid, global_row],
                                                accum_dtype,
                                            )
                                    for r in T.vectorized(ilp_rows):
                                        v_new[r] = (v_value[r] - sum_hk[r]) * beta_val

                                    for r in T.unroll(ilp_rows):
                                        for i in T.unroll(vec_size):
                                            h_reg[r * vec_size + i] = (
                                                h_reg[r * vec_size + i]
                                                + k_reg[i] * v_new[r]
                                            )
                                            sum_hq[r] += (
                                                h_reg[r * vec_size + i] * q_reg[i]
                                            )

                                    for offset in T.unroll(_WARP_REDUCE_STEPS):
                                        mask = (_WARP_SIZE // 2) >> offset
                                        for r in T.unroll(ilp_rows):
                                            sum_hq[r] += T.shfl_xor(sum_hq[r], mask)

                                    if lane == 0:
                                        for r in T.vectorized(ilp_rows):
                                            global_row = global_v_base + row_idx + r
                                            if _USE_SMEM_V:
                                                output_shared[t, row_idx + r] = T.cast(
                                                    sum_hq[r],
                                                    output_dtype,
                                                )
                                            else:
                                                output[bid, t, hid, global_row] = (
                                                    T.cast(
                                                        sum_hq[r],
                                                        output_dtype,
                                                    )
                                                )

                                    if cache_intermediate_states:
                                        for r in T.unroll(ilp_rows):
                                            global_row = global_v_base + row_idx + r
                                            for i in T.vectorized(vec_size):
                                                intermediate_states[
                                                    bid,
                                                    t,
                                                    hid,
                                                    global_row,
                                                    k_start + i,
                                                ] = T.cast(
                                                    h_reg[r * vec_size + i],
                                                    state_storage_dtype,
                                                )

                                if not disable_state_update:
                                    for r in T.unroll(ilp_rows):
                                        global_row = global_v_base + row_idx + r
                                        for i in T.vectorized(vec_size):
                                            initial_state[
                                                state_slot,
                                                hid,
                                                global_row,
                                                k_start + i,
                                            ] = T.cast(
                                                h_reg[r * vec_size + i],
                                                state_storage_dtype,
                                            )

                        if (
                            _NUM_STAGES == 1
                            and next_row_idx + rows_per_state_tile - 1 < tile_v
                        ):
                            # Single-stage TME reloads the same slot only after
                            # all consumers finish the current tile.
                            T.sync_threads()
                            global_next_load_row = global_v_base + next_row_idx
                            T.tma_copy(
                                initial_state[state_slot, hid, global_next_load_row, 0],
                                state_smem[stage, :, :],
                                barrier=state_load_mbars[stage],
                                inner_cache_policy="cache_none",
                                outer_cache_policy="cache_persist",
                            )
                            T.mbarrier_arrive(mbarrier=state_load_mbars[stage])
                    if _USE_SMEM_V:
                        T.sync_threads()
                        for t in T.serial(seq_len):
                            if tid < tile_v:
                                output[bid, t, hid, global_v_base + tid] = (
                                    output_shared[t, tid]
                                )
                else:
                    for row_base in range(0, tile_v, num_warps * ilp_rows):
                        row_idx = row_base + warp * ilp_rows
                        if row_idx + ilp_rows - 1 < tile_v:
                            for t in T.serial(seq_len):
                                if lane == 0:
                                    for r in T.vectorized(ilp_rows):
                                        global_row = global_v_base + row_idx + r
                                        output[bid, t, hid, global_row] = T.cast(
                                            0.0,
                                            output_dtype,
                                        )

        return gated_deltanet_mtp_fp32_vk_smem

    return _mtp_func()


def run_gated_delta_rule_mtp_vk_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    state_indices: torch.Tensor | None,
    output: torch.Tensor,
    intermediate_states_buffer: torch.Tensor | None,
    *,
    scale: float,
    disable_state_update: bool,
    use_qk_l2norm: bool,
) -> None:
    """Run the VK-layout MTP backend.

    Inputs are expected to be validated by ``mate.gdn_decode``. Negative
    ``state_indices`` entries are padding rows: output is zeroed and state is
    not read or updated for that batch entry.
    """

    B, T_len, Hq, K = q.shape
    _, HV, V, _ = state.shape
    cache_intermediate_states = intermediate_states_buffer is not None
    use_identity_state_indices = state_indices is None

    tile_v, _, ilp_rows = _get_mtp_config(
        batch_size=B,
        seq_len=T_len,
        num_v_heads=HV,
        v_dim=V,
        cache_intermediate_states=cache_intermediate_states,
        state_dtype=state.dtype,
    )

    state_indices_arg = (
        torch.empty((B,), dtype=torch.int32, device=q.device)
        if use_identity_state_indices
        else state_indices
    )

    if cache_intermediate_states:
        intermediate_arg = intermediate_states_buffer
    else:
        intermediate_arg = torch.empty(
            (1, 1, 1, 1, 1), dtype=state.dtype, device=q.device
        )

    kernel_kwargs = dict(
        seq_len=T_len,
        qk_head=Hq,
        head=HV,
        dim_k=K,
        dim_v=V,
        input_dtype=str(q.dtype).split(".")[-1],
        output_dtype=str(output.dtype).split(".")[-1],
        dt_bias_dtype=str(dt_bias.dtype).split(".")[-1],
        state_dtype=str(state.dtype).split(".")[-1],
        use_qk_l2norm=bool(use_qk_l2norm),
        disable_state_update=bool(disable_state_update),
        use_identity_state_indices=use_identity_state_indices,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
        disable_index_type_promotion=_should_disable_index_type_promotion(
            batch_size=B,
            seq_len=T_len,
            num_v_heads=HV,
            state_dtype=state.dtype,
            cache_intermediate_states=cache_intermediate_states,
            disable_state_update=disable_state_update,
        ),
    )
    kernel_fn = _get_mtp_fp32_vk_smem_kernel(
        cache_intermediate_states=cache_intermediate_states,
        **kernel_kwargs,
    )
    kernel_fn(
        q,
        k,
        v,
        A_log,
        a,
        dt_bias,
        b,
        float(scale),
        state,
        state_indices_arg,
        intermediate_arg,
        output,
    )
