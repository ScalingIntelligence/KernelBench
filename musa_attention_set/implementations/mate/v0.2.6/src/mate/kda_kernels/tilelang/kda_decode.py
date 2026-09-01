import functools
import tilelang
import tilelang.language as T
import torch


from ...execution_context import raise_complete_if_dry_run

__all__ = ["run_gated_delta_rule_decode_vk_fp32"]

_LOG2E = 1.4426950408889634
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0
_KERNEL_THREADS = 128
_STATE_TME_INNER_CACHE_POLICY = "cache_normal"
_STATE_TME_OUTER_CACHE_POLICY = "cache_none"

_AUTO_TUNED_BATCH_CONFIGS = (
    (2, (8, 8)),
    (4, (8, 4)),
    (16, (8, 2)),
)
_AUTO_TUNED_LARGE_BATCH_CONFIG = (4, 4)
# V-first D=128 varlen uses one 16-row state tile per CTA. This keeps the
# eight-way state split while removing three state TME copies/barrier pairs.
_AUTO_TUNED_VARLEN_V_FIRST_LARGE_BATCH_CONFIG = (16, 8)
_AUTO_TUNED_VARLEN_FALLBACK_LARGE_BATCH_CONFIG = (4, 8)
# BF16 V-first uses v_tile=16 to merge each pair of 2 KiB state TME tiles.
# Keeping num_blocks_per_state unchanged preserves the state rows and shared
# memory footprint per CTA while halving the number of TME copies/barriers.
_AUTO_TUNED_BF16_STATE_V_FIRST_BATCH_CONFIGS = (
    (2, (16, 8)),
    (4, (16, 4)),
    (16, (16, 2)),
    (64, (16, 4)),
)
_AUTO_TUNED_BF16_STATE_V_FIRST_LARGE_BATCH_CONFIG = (16, 2)
# BF16 V-first D=128 varlen uses one 32-row state tile per CTA once B > 16.
_AUTO_TUNED_BF16_STATE_V_FIRST_VARLEN_LARGE_BATCH_CONFIG = (32, 4)
# Preserve the previous V-first configs for dimensions other than the tuned
# D=128 shape, and for the separately tuned K-first path.
_AUTO_TUNED_BF16_STATE_FALLBACK_BATCH_CONFIGS = (
    (2, (8, 8)),
    (4, (8, 4)),
    (16, (8, 2)),
    (64, (8, 4)),
)
_AUTO_TUNED_BF16_STATE_V_FIRST_FALLBACK_LARGE_BATCH_CONFIG = (8, 2)
_AUTO_TUNED_BF16_STATE_K_FIRST_LARGE_BATCH_CONFIG = (8, 4)


def _exp2_f32(value):
    return T.exp2(value * _LOG2E)


def _make_kernel_config(v_tile: int, num_blocks_per_state: int) -> dict:
    return {
        "v_tile": v_tile,
        "num_blocks_per_state": num_blocks_per_state,
    }


def _resolve_autotuned_kernel_config(
    batch: int,
    state_dtype: str = "float32",
    state_v_first: bool = True,
    dim_v: int = 128,
    is_varlen: bool = False,
) -> dict:
    batch_configs: tuple[tuple[int, tuple[int, int]], ...]
    large_batch_config: tuple[int, int]
    if state_dtype == "bfloat16":
        if state_v_first and dim_v == 128:
            if is_varlen and batch > 16:
                return _make_kernel_config(
                    *_AUTO_TUNED_BF16_STATE_V_FIRST_VARLEN_LARGE_BATCH_CONFIG
                )
            batch_configs = _AUTO_TUNED_BF16_STATE_V_FIRST_BATCH_CONFIGS
            large_batch_config = _AUTO_TUNED_BF16_STATE_V_FIRST_LARGE_BATCH_CONFIG
        else:
            batch_configs = _AUTO_TUNED_BF16_STATE_FALLBACK_BATCH_CONFIGS
            large_batch_config = (
                _AUTO_TUNED_BF16_STATE_V_FIRST_FALLBACK_LARGE_BATCH_CONFIG
                if state_v_first
                else _AUTO_TUNED_BF16_STATE_K_FIRST_LARGE_BATCH_CONFIG
            )
    else:
        batch_configs = _AUTO_TUNED_BATCH_CONFIGS
        if is_varlen:
            large_batch_config = (
                _AUTO_TUNED_VARLEN_V_FIRST_LARGE_BATCH_CONFIG
                if state_v_first and dim_v == 128
                else _AUTO_TUNED_VARLEN_FALLBACK_LARGE_BATCH_CONFIG
            )
        else:
            large_batch_config = _AUTO_TUNED_LARGE_BATCH_CONFIG

    for max_batch, config in batch_configs:
        if batch <= max_batch:
            return _make_kernel_config(*config)
    return _make_kernel_config(*large_batch_config)


def _build_kda_decode_kernel_factory(
    qk_head: int,
    head: int,
    dim_k: int,
    dim_v: int,
    input_dtype: str,
    gate_batch_dtype: str,
    dt_bias_dtype: str,
    output_dtype: str,
    b_dtype: str,
    use_qk_l2norm: bool,
    v_tile: int,
    num_blocks_per_state: int,
    use_initial_state: bool,
    is_varlen: bool,
    inplace_final_state: bool,
    is_beta_headwise: bool,
    is_continuous_batching: bool,
    is_spec_decoding: bool,
    store_final_state: bool,
    has_dt_bias: bool,
    use_gate_in_kernel: bool,
    use_lower_bound: bool,
    apply_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
    state_dtype: str = "float32",
):
    if qk_head <= 0:
        raise ValueError("qk_head must be positive.")
    if head % qk_head != 0:
        raise ValueError(
            f"state/value heads={head} must be divisible by q/k heads={qk_head}."
        )
    if dim_k % 32 != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by 32.")
    if dim_v != dim_k:
        raise ValueError(
            f"Current decode kernel expects dim_v == dim_k, got dim_v={dim_v}, dim_k={dim_k}."
        )
    if state_dtype not in ("float32", "bfloat16"):
        raise ValueError(f"state_dtype must be float32 or bfloat16, got {state_dtype}.")
    if dim_v % v_tile != 0:
        raise ValueError(f"dim_v={dim_v} must be divisible by v_tile={v_tile}")
    if v_tile % 4 != 0:
        raise ValueError(f"v_tile={v_tile} must be divisible by 4 for row mapping.")
    head_group_size = head // qk_head
    num_v_tiles = dim_v // v_tile
    if num_blocks_per_state <= 0:
        raise ValueError("num_blocks_per_state must be positive.")
    if num_blocks_per_state > num_v_tiles:
        raise ValueError(
            f"num_blocks_per_state={num_blocks_per_state} exceeds num_v_tiles={num_v_tiles}."
        )
    if num_v_tiles % num_blocks_per_state != 0:
        raise ValueError(
            f"num_v_tiles={num_v_tiles} must be divisible by "
            f"num_blocks_per_state={num_blocks_per_state}."
        )
    num_v_tiles_per_block = (
        num_v_tiles + num_blocks_per_state - 1
    ) // num_blocks_per_state
    if num_v_tiles_per_block * v_tile > _KERNEL_THREADS:
        raise ValueError(
            f"per-block output elements={num_v_tiles_per_block * v_tile} exceeds "
            f"kernel threads={_KERNEL_THREADS}."
        )
    vec_size = dim_k // 32
    vec_size_v = dim_v // 32
    num_v_rows_per_block = num_v_tiles_per_block * v_tile
    # Localize V/beta only when the state is split across enough CTAs for the
    # old full-head producer to materially overfetch.  The two-way split used
    # by B=8/16 is faster with the original coalesced full-head load.
    use_local_v_beta = is_varlen and num_blocks_per_state >= 4

    batch = T.dynamic("batch")
    old_batch = T.dynamic("old_batch")
    seq_q = T.dynamic("seq_q")
    seq_kv = T.dynamic("seq_kv")
    seq_o = T.dynamic("seq_o")
    pool_size = T.dynamic("pool_size")

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
    a_stride_k = T.dynamic("a_stride_k")

    b_stride_b = T.dynamic("b_stride_b")
    b_stride_t = T.dynamic("b_stride_t")
    b_stride_h = T.dynamic("b_stride_h")
    b_stride_v = T.dynamic("b_stride_v")

    o_stride_b = T.dynamic("o_stride_b")
    o_stride_t = T.dynamic("o_stride_t")
    o_stride_h = T.dynamic("o_stride_h")
    o_stride_v = T.dynamic("o_stride_v")

    state_indices_stride_b = T.dynamic("state_indices_stride_b")
    state_indices_stride_t = T.dynamic("state_indices_stride_t")
    token = T.dynamic("token")

    q_shape = (batch, seq_q, qk_head, dim_k)
    k_shape = (batch, seq_kv, qk_head, dim_k)
    v_shape = (batch, seq_kv, head, dim_v)
    g_shape = (batch, seq_kv, head, dim_k)
    vk_shape = (dim_v, dim_k) if state_v_first else (dim_k, dim_v)
    state_final_shape = (
        (pool_size, head, *vk_shape)
        if inplace_final_state
        else (old_batch, head, *vk_shape)
    )
    state_final_shape = (
        (seq_q, head, *vk_shape)
        if (is_continuous_batching and not inplace_final_state)
        else state_final_shape
    )
    beta_shape = (
        (batch, seq_kv, head, dim_v) if is_beta_headwise else (batch, seq_kv, head)
    )
    o_shape = (batch, seq_o, head, dim_v)
    cu_seqlens_shape = (old_batch + 1,) if is_varlen else (1,)
    # Keep one cheap B-sized argument even when speculative decoding is off.
    # It anchors old_batch for fixed, non-continuous, in-place calls where the
    # state pool can be larger than the logical batch.
    num_accepted_tokens_shape = (old_batch,)
    state_indices_shape = (old_batch, token) if is_continuous_batching else (1, 1)

    q_strides = (q_stride_b, q_stride_t, q_stride_h, q_stride_k)
    k_strides = (k_stride_b, k_stride_t, k_stride_h, k_stride_k)
    v_strides = (v_stride_b, v_stride_t, v_stride_h, v_stride_v)
    a_strides = (a_stride_b, a_stride_t, a_stride_h, a_stride_k)
    b_strides = (
        (b_stride_b, b_stride_t, b_stride_h, b_stride_v)
        if is_beta_headwise
        else (b_stride_b, b_stride_t, b_stride_h)
    )
    o_strides = (o_stride_b, o_stride_t, o_stride_h, o_stride_v)
    state_indices_strides = (state_indices_stride_b, state_indices_stride_t)

    @tilelang.jit(
        pass_configs={
            # tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
            tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
            tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
        },
        compile_flags=[
            "-Od3",
            "-fno-signed-zeros",
            "-fmusa-flush-denormals-to-zero",
            "-mllvm",
            "-misched=mtgpu-max-ilp",
            "-mllvm",
            "-mtgpu-if-convert=1",
            "-mllvm",
            "-mtgpu-tiny-offset-hint=1",
            "-mllvm",
            "-mtgpu-combine-fop-instr=1",
            "-mllvm",
            "-mtgpu-enable-postra-sched=0",
            "-mllvm",
            "-misched-recompute-slotindex=1",
        ],
    )
    def _decode_func():
        mbarrier_list = [_KERNEL_THREADS] * num_v_tiles_per_block

        @T.prim_func
        def gated_deltanet_kda_decode(
            q: T.StridedTensor(q_shape, q_strides, input_dtype),
            k: T.StridedTensor(k_shape, k_strides, input_dtype),
            v: T.StridedTensor(v_shape, v_strides, input_dtype),
            A_log: T.Tensor([head], "float32"),
            a: T.StridedTensor(g_shape, a_strides, gate_batch_dtype),
            dt_bias: T.Tensor([head, dim_k], dt_bias_dtype),
            b: T.StridedTensor(beta_shape, b_strides, b_dtype),
            cu_seqlens: T.Tensor(cu_seqlens_shape, "int32"),
            num_accepted_tokens: T.Tensor(num_accepted_tokens_shape, "int32"),
            T_fixed_arg: T.int32,
            scale: T.float32,
            lower_bound: T.float32,
            state_indices: T.StridedTensor(
                state_indices_shape, state_indices_strides, "int32"
            ),
            state: T.Tensor([pool_size, head, *vk_shape], state_dtype),
            state_final: T.Tensor(state_final_shape, state_dtype),
            o: T.StridedTensor(o_shape, o_strides, output_dtype),
        ):
            with T.Kernel(
                old_batch * num_blocks_per_state, head, threads=_KERNEL_THREADS
            ) as (bx, hid):
                bid = bx // num_blocks_per_state
                block_inner = bx % num_blocks_per_state
                start_v_tile = block_inner * num_v_tiles_per_block
                state_load_stage = T.alloc_shared(
                    [num_v_tiles_per_block, v_tile, dim_k], state_dtype
                )
                state_load_stage2 = T.alloc_shared(
                    [num_v_tiles_per_block, dim_k, v_tile], state_dtype
                )
                output_tile = T.alloc_shared([num_v_rows_per_block], output_dtype)
                value_tile = T.alloc_shared(
                    [num_v_rows_per_block if use_local_v_beta else dim_v], "float32"
                )
                beta_tile = T.alloc_shared(
                    [num_v_rows_per_block if use_local_v_beta else dim_v], "float32"
                )

                q_reg = T.alloc_local([vec_size], "float32")
                k_reg = T.alloc_local([vec_size], "float32")
                g_reg = T.alloc_local([vec_size], "float32")
                # State storage may be BF16; keep each token's recurrent update
                # in FP32 registers, then quantize when writing the tile back.
                h_reg = T.alloc_local([vec_size], "float32")
                beta_reg = T.alloc_local(
                    [1 if use_local_v_beta else vec_size_v], "float32"
                )
                dt_bias_val = T.alloc_local([vec_size], "float32")

                sum_q = T.alloc_local([1], "float32")
                sum_k = T.alloc_local([1], "float32")
                sum_hk = T.alloc_local([1], "float32")
                sum_hq = T.alloc_local([1], "float32")
                final_state_slot = T.alloc_local([1], "int32")

                mbars = T.alloc_barrier(mbarrier_list)

                qk_hid = hid // head_group_size
                tid = T.get_thread_binding()
                lane = tid % 32
                warp = tid // 32
                k_start = lane * vec_size
                v_start = lane * vec_size_v
                if is_continuous_batching:
                    if is_spec_decoding:
                        i_t = num_accepted_tokens[bid] - 1
                    else:
                        i_t = 0
                    state_slot = state_indices[bid, i_t]
                else:
                    state_slot = bid

                if is_varlen:
                    bos = cu_seqlens[bid]
                    eos = cu_seqlens[bid + 1]
                    T_seq = eos - bos
                else:
                    T_seq = T_fixed_arg
                    bos, eos = bid * T_seq, bid * T_seq + T_seq

                if state_slot >= 0 and T_seq > 0:
                    if use_initial_state:
                        prologue_v_tile = start_v_tile
                        prologue_v_base = prologue_v_tile * v_tile
                        if state_v_first:
                            T.tma_copy(
                                state[state_slot, hid, prologue_v_base, 0],
                                state_load_stage[0, :, :],
                                barrier=mbars[0],
                                inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                            )
                        else:
                            T.tma_copy(
                                state[state_slot, hid, 0, prologue_v_base],
                                state_load_stage2[0, :, :],
                                barrier=mbars[0],
                                inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                            )
                        T.mbarrier_arrive(mbarrier=mbars[0])

                    A_decay_val = (
                        -_exp2_f32(T.cast(A_log[hid], "float32"))
                        if use_gate_in_kernel
                        else 0.0
                    )
                    for i in T.vectorized(vec_size):
                        dt_bias_val[i] = 0.0
                    if has_dt_bias:
                        for i in T.vectorized(vec_size):
                            dt_bias_val[i] = T.cast(
                                dt_bias[hid, k_start + i], "float32"
                            )

                    final_state_slot[0] = bid

                    for i_t in T.serial(0, T_seq):
                        if is_continuous_batching:
                            do_store_final_this_token = True
                            if inplace_final_state:
                                final_state_slot[0] = state_indices[bid, i_t]
                            else:
                                final_state_slot[0] = bos + i_t
                        elif store_final_state:
                            do_store_final_this_token = (
                                i_t == T_seq - 1 if is_varlen else True
                            )
                        else:
                            do_store_final_this_token = False

                        T.sync_threads()
                        if use_local_v_beta:
                            if tid < num_v_rows_per_block:
                                global_v_idx = start_v_tile * v_tile + tid
                                value_tile[tid] = T.cast(
                                    v[0, bos + i_t, hid, global_v_idx], "float32"
                                )
                        else:
                            for i in T.unroll(vec_size_v):
                                global_v_idx = v_start + i
                                value_tile[global_v_idx] = T.cast(
                                    v[0, bos + i_t, hid, global_v_idx], "float32"
                                )

                        for i in T.unroll(vec_size):
                            kk = k_start + i
                            g_reg[i] = T.cast(a[0, bos + i_t, hid, kk], "float32")

                        if use_local_v_beta:
                            if is_beta_headwise:
                                if tid < num_v_rows_per_block:
                                    global_v_idx = start_v_tile * v_tile + tid
                                    beta_reg[0] = T.cast(
                                        b[0, bos + i_t, hid, global_v_idx], "float32"
                                    )
                                    if apply_beta_sigmoid:
                                        beta_reg[0] = 1.0 / (
                                            1.0 + _exp2_f32(-beta_reg[0])
                                        )
                                        if allow_neg_eigval:
                                            beta_reg[0] *= 2
                                    beta_tile[tid] = beta_reg[0]
                            else:
                                beta_reg[0] = 0.0
                                if lane == 0:
                                    beta_reg[0] = T.cast(
                                        b[0, bos + i_t, hid], "float32"
                                    )
                                    if apply_beta_sigmoid:
                                        beta_reg[0] = 1.0 / (
                                            1.0 + _exp2_f32(-beta_reg[0])
                                        )
                                        if allow_neg_eigval:
                                            beta_reg[0] *= 2
                                beta_reg[0] = T.shfl_sync(0xFFFFFFFF, beta_reg[0], 0)
                        elif is_beta_headwise:
                            for i in T.unroll(vec_size_v):
                                global_v_idx = v_start + i
                                beta_reg[i] = T.cast(
                                    b[0, bos + i_t, hid, global_v_idx], "float32"
                                )
                                if apply_beta_sigmoid:
                                    beta_reg[i] = 1.0 / (1.0 + _exp2_f32(-beta_reg[i]))
                                    if allow_neg_eigval:
                                        beta_reg[i] *= 2
                                beta_tile[global_v_idx] = beta_reg[i]
                        else:
                            beta_reg[0] = T.cast(b[0, bos + i_t, hid], "float32")
                            if apply_beta_sigmoid:
                                beta_reg[0] = 1.0 / (1.0 + _exp2_f32(-beta_reg[0]))
                                if allow_neg_eigval:
                                    beta_reg[0] *= 2

                        for i in T.unroll(vec_size):
                            kk = k_start + i
                            if kk < dim_k:
                                q_reg[i] = T.cast(
                                    q[0, bos + i_t, qk_hid, kk], "float32"
                                )
                                k_reg[i] = T.cast(
                                    k[0, bos + i_t, qk_hid, kk], "float32"
                                )
                            else:
                                q_reg[i] = 0.0
                                k_reg[i] = 0.0
                        if use_qk_l2norm:
                            sum_q[0] = 0.0
                            sum_k[0] = 0.0
                            for i in T.unroll(vec_size):
                                sum_q[0] += q_reg[i] * q_reg[i]
                                sum_k[0] += k_reg[i] * k_reg[i]
                            for offset in T.unroll(5):
                                mask = 16 >> offset
                                sum_q[0] += T.shfl_xor(sum_q[0], mask)
                                sum_k[0] += T.shfl_xor(sum_k[0], mask)
                            inv_norm_q = T.rsqrt(sum_q[0] + 1e-6)
                            inv_norm_k = T.rsqrt(sum_k[0] + 1e-6)
                            for i in T.unroll(vec_size):
                                q_reg[i] = q_reg[i] * inv_norm_q
                                k_reg[i] = k_reg[i] * inv_norm_k

                        for i in T.unroll(vec_size):
                            q_reg[i] = q_reg[i] * scale

                        if use_gate_in_kernel:
                            if has_dt_bias:
                                for i in T.unroll(vec_size):
                                    g_reg[i] += dt_bias_val[i]
                            if use_lower_bound:
                                for i in T.unroll(vec_size):
                                    g_reg[i] = (
                                        lower_bound
                                        * 1.0
                                        / (1.0 + _exp2_f32(A_decay_val * g_reg[i]))
                                    )
                            else:
                                for i in T.unroll(vec_size):
                                    g_tmp = _SOFTPLUS_BETA * g_reg[i]
                                    softplus_x = T.if_then_else(
                                        g_tmp <= _SOFTPLUS_THRESHOLD,
                                        (1.0 / _SOFTPLUS_BETA)
                                        * T.log(1.0 + _exp2_f32(g_tmp)),
                                        g_reg[i],
                                    )
                                    g_reg[i] = A_decay_val * softplus_x
                            for i in T.unroll(vec_size):
                                g_reg[i] = _exp2_f32(g_reg[i])

                        for local_v_tile in T.serial(0, num_v_tiles_per_block):
                            global_v_tile = start_v_tile + local_v_tile
                            global_v_base = global_v_tile * v_tile

                            # Store the previous updated state tile.
                            if (
                                local_v_tile > 0
                                and do_store_final_this_token
                                and global_v_base - v_tile < dim_v
                            ):
                                global_prev_v_base = global_v_base - v_tile
                                if state_v_first:
                                    T.tma_copy(
                                        state_load_stage[local_v_tile - 1, :, :],
                                        state_final[
                                            final_state_slot[0],
                                            hid,
                                            global_prev_v_base : global_prev_v_base
                                            + v_tile,
                                            :,
                                        ],
                                        inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                        outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                                    )
                                else:
                                    T.tma_copy(
                                        state_load_stage2[local_v_tile - 1, :, :],
                                        state_final[
                                            final_state_slot[0],
                                            hid,
                                            :,
                                            global_prev_v_base : global_prev_v_base
                                            + v_tile,
                                        ],
                                        inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                        outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                                    )

                            if i_t == 0 and use_initial_state:
                                T.mbarrier_wait_parity(
                                    mbarrier=mbars[local_v_tile], parity=0
                                )
                            T.sync_threads()
                            for row_base in range(0, v_tile, 4):
                                row_idx = row_base + warp
                                local_row = local_v_tile * v_tile + row_idx
                                global_row = global_v_base + row_idx
                                value_idx = (
                                    local_row if use_local_v_beta else global_row
                                )
                                sum_hk[0] = 0.0
                                sum_hq[0] = 0.0

                                for i in T.serial(vec_size):
                                    h_reg[i] = 0.0
                                if row_idx + global_v_base < dim_v:
                                    for i in T.unroll(vec_size):
                                        if k_start + i < dim_k:
                                            if state_v_first:
                                                h_reg[i] = T.cast(
                                                    state_load_stage[
                                                        local_v_tile,
                                                        row_idx,
                                                        k_start + i,
                                                    ],
                                                    "float32",
                                                )
                                            else:
                                                h_reg[i] = T.cast(
                                                    state_load_stage2[
                                                        local_v_tile,
                                                        k_start + i,
                                                        row_idx,
                                                    ],
                                                    "float32",
                                                )
                                if i_t == 0 and use_initial_state is False:
                                    for i in T.serial(vec_size):
                                        h_reg[i] = 0.0

                                for i in T.unroll(vec_size):
                                    h_reg[i] = h_reg[i] * g_reg[i]
                                    sum_hk[0] += h_reg[i] * k_reg[i]

                                for offset in T.unroll(5):
                                    mask = 16 >> offset
                                    sum_hk[0] += T.shfl_xor(sum_hk[0], mask)

                                if is_beta_headwise:
                                    v_new = (
                                        value_tile[value_idx] - sum_hk[0]
                                    ) * beta_tile[value_idx]
                                else:
                                    v_new = (
                                        value_tile[value_idx] - sum_hk[0]
                                    ) * beta_reg[0]

                                for i in T.unroll(vec_size):
                                    h_reg[i] += k_reg[i] * v_new
                                    sum_hq[0] += h_reg[i] * q_reg[i]

                                if state_v_first:
                                    for i in T.serial(vec_size):
                                        state_load_stage[
                                            local_v_tile, row_idx, k_start + i
                                        ] = T.cast(h_reg[i], state_dtype)
                                else:
                                    for i in T.serial(vec_size):
                                        state_load_stage2[
                                            local_v_tile, k_start + i, row_idx
                                        ] = T.cast(h_reg[i], state_dtype)

                                for offset in T.unroll(5):
                                    mask = 16 >> offset
                                    sum_hq[0] += T.shfl_xor(sum_hq[0], mask)

                                o_idx = local_v_tile * v_tile + row_idx
                                if lane == 0 and o_idx < num_v_rows_per_block:
                                    output_tile[o_idx] = T.cast(sum_hq[0], output_dtype)

                            # Ensure state_load_stage[stage_idx_var] is consumed before refill.
                            T.sync_threads()
                            if i_t == 0 and use_initial_state:
                                next_local_v_tile = local_v_tile + 1
                                if next_local_v_tile < num_v_tiles_per_block:
                                    global_next_v_tile = (
                                        start_v_tile + next_local_v_tile
                                    )
                                    global_next_v_base = global_next_v_tile * v_tile
                                    if state_v_first:
                                        T.tma_copy(
                                            state[
                                                state_slot, hid, global_next_v_base, 0
                                            ],
                                            state_load_stage[local_v_tile + 1, :, :],
                                            barrier=mbars[local_v_tile + 1],
                                            inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                            outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                                        )
                                    else:
                                        T.tma_copy(
                                            state[
                                                state_slot, hid, 0, global_next_v_base
                                            ],
                                            state_load_stage2[local_v_tile + 1, :, :],
                                            barrier=mbars[local_v_tile + 1],
                                            inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                            outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                                        )
                                    T.mbarrier_arrive(mbarrier=mbars[local_v_tile + 1])

                        # Epilogue: store the last updated state tile.
                        global_prev_v_base_epi = (
                            start_v_tile + num_v_tiles_per_block - 1
                        ) * v_tile
                        if do_store_final_this_token and global_prev_v_base_epi < dim_v:
                            if state_v_first:
                                T.tma_copy(
                                    state_load_stage[num_v_tiles_per_block - 1, :, :],
                                    state_final[
                                        final_state_slot[0],
                                        hid,
                                        global_prev_v_base_epi : global_prev_v_base_epi
                                        + v_tile,
                                        :,
                                    ],
                                    inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                    outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                                )
                            else:
                                T.tma_copy(
                                    state_load_stage2[num_v_tiles_per_block - 1, :, :],
                                    state_final[
                                        final_state_slot[0],
                                        hid,
                                        :,
                                        global_prev_v_base_epi : global_prev_v_base_epi
                                        + v_tile,
                                    ],
                                    inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                    outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                                )
                        if (
                            tid < num_v_rows_per_block
                            and start_v_tile * v_tile + tid < dim_v
                        ):
                            o[0, bos + i_t, hid, start_v_tile * v_tile + tid] = (
                                output_tile[tid]
                            )
                        if do_store_final_this_token:
                            T.tma_store_wait()
                else:
                    for i_t in T.serial(0, T_seq):
                        if tid < num_v_rows_per_block:
                            o[0, bos + i_t, hid, start_v_tile * v_tile + tid] = T.cast(
                                0.0, output_dtype
                            )

        return gated_deltanet_kda_decode

    return _decode_func


@functools.lru_cache(maxsize=32)
def _get_kda_decode_kernel(
    qk_head: int,
    head: int,
    dim_k: int,
    dim_v: int,
    input_dtype: str,
    gate_batch_dtype: str,
    dt_bias_dtype: str,
    output_dtype: str,
    b_dtype: str,
    use_qk_l2norm: bool,
    v_tile: int,
    num_blocks_per_state: int,
    use_initial_state: bool,
    is_varlen: bool,
    inplace_final_state: bool,
    is_beta_headwise: bool,
    is_continuous_batching: bool,
    is_spec_decoding: bool,
    store_final_state: bool,
    has_dt_bias: bool,
    use_gate_in_kernel: bool,
    use_lower_bound: bool,
    apply_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
    state_dtype: str = "float32",
):
    return _build_kda_decode_kernel_factory(
        qk_head=qk_head,
        head=head,
        dim_k=dim_k,
        dim_v=dim_v,
        input_dtype=input_dtype,
        state_dtype=state_dtype,
        gate_batch_dtype=gate_batch_dtype,
        dt_bias_dtype=dt_bias_dtype,
        output_dtype=output_dtype,
        b_dtype=b_dtype,
        use_qk_l2norm=use_qk_l2norm,
        v_tile=v_tile,
        num_blocks_per_state=num_blocks_per_state,
        use_initial_state=use_initial_state,
        is_varlen=is_varlen,
        inplace_final_state=inplace_final_state,
        is_beta_headwise=is_beta_headwise,
        is_continuous_batching=is_continuous_batching,
        is_spec_decoding=is_spec_decoding,
        store_final_state=store_final_state,
        has_dt_bias=has_dt_bias,
        use_gate_in_kernel=use_gate_in_kernel,
        use_lower_bound=use_lower_bound,
        apply_beta_sigmoid=apply_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        state_v_first=state_v_first,
    )()


def run_gated_delta_rule_decode_vk_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
    A_log: torch.Tensor,
    g: torch.Tensor,
    dt_bias: torch.Tensor | None,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    output: torch.Tensor,
    *,
    scale: float,
    lower_bound: float,
    use_qk_l2norm: bool,
    is_varlen: bool,
    inplace_final_state: bool,
    is_beta_headwise: bool,
    is_continuous_batching: bool,
    is_spec_decoding: bool,
    store_final_state: bool,
    has_dt_bias: bool,
    use_gate_in_kernel: bool,
    use_lower_bound: bool,
    apply_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
    use_initial_state: bool,
):
    if is_continuous_batching and state_indices is not None:
        if state_indices.ndim == 1:
            state_indices = state_indices.unsqueeze(1)
        elif state_indices.ndim != 2:
            raise ValueError(
                "state_indices must have shape [B] or [B, T], "
                f"got {tuple(state_indices.shape)}."
            )

    if is_varlen:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens must not be None when is_varlen=True")

        B = cu_seqlens.numel() - 1
        _, total_tokens, Hq, K = q.shape
        _, _, HV, V = v.shape

        q_arg = q
        k_arg = k
        v_arg = v
        g_arg = g
        b_arg = b

        output_arg = output
        cu_seqlens_arg = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        T_fixed_arg = 0
    else:
        B, T_fixed_host, Hq, K = q.shape
        _, _, HV, V = v.shape
        total_tokens = B * T_fixed_host

        q_arg = q.reshape(1, total_tokens, Hq, K)
        k_arg = k.reshape(1, total_tokens, Hq, K)
        v_arg = v.reshape(1, total_tokens, HV, V)
        g_arg = g.reshape(1, total_tokens, HV, K)
        if is_beta_headwise:
            b_arg = b.reshape(1, total_tokens, HV, V)
        else:
            b_arg = b.reshape(1, total_tokens, HV)
        output_arg = output.reshape(1, total_tokens, HV, V)

        cu_seqlens_arg = torch.empty((1,), dtype=torch.int32, device=q.device)
        T_fixed_arg = int(T_fixed_host)

    input_dtype = str(q_arg.dtype).split(".")[-1]
    state_dtype = str(state.dtype).split(".")[-1]
    gate_batch_dtype = str(g_arg.dtype).split(".")[-1]

    output_dtype = str(output_arg.dtype).split(".")[-1]
    b_dtype = str(b_arg.dtype).split(".")[-1]
    if state_dtype not in ("float32", "bfloat16"):
        raise ValueError(
            f"state must have dtype torch.float32 or torch.bfloat16, got {state.dtype}."
        )
    kernel_config = _resolve_autotuned_kernel_config(
        B, state_dtype, state_v_first, V, is_varlen
    )

    needs_dummy_dt_bias = not has_dt_bias
    needs_dummy_num_accepted_tokens = not is_spec_decoding
    needs_dummy_state_indices = not is_continuous_batching

    if needs_dummy_dt_bias:
        dt_bias_arg = torch.zeros((HV, K), dtype=torch.float32, device=q.device)
    else:
        if dt_bias is None:
            raise ValueError("dt_bias must not be None when has_dt_bias=True")
        if dt_bias.ndim == 2:
            dt_bias_arg = dt_bias.to(device=q.device, dtype=dt_bias.dtype).contiguous()
        else:
            raise ValueError(f"dt_bias must be 2D, got ndim={dt_bias.ndim}")

    dt_bias_dtype = str(dt_bias_arg.dtype).split(".")[-1]

    if needs_dummy_state_indices:
        state_indices_arg = torch.empty((1, 1), dtype=torch.int32, device=q.device)
    else:
        if state_indices is None:
            raise ValueError(
                "state_indices must not be None when is_continuous_batching=True"
            )
        state_indices_arg = state_indices.to(
            device=q.device,
            dtype=torch.int32,
        ).contiguous()

    if needs_dummy_num_accepted_tokens:
        num_accepted_tokens_arg = torch.empty((B,), dtype=torch.int32, device=q.device)
    else:
        if num_accepted_tokens is None:
            raise ValueError(
                "num_accepted_tokens must not be None when is_spec_decoding=True"
            )
        num_accepted_tokens_arg = num_accepted_tokens.to(
            device=q.device,
            dtype=torch.int32,
        ).contiguous()

    A_log_arg = (
        A_log
        if A_log is not None
        else torch.zeros((HV,), dtype=torch.float32, device=q.device)
    )

    final_state_slots = None
    if is_continuous_batching:
        final_state_slots = total_tokens
    else:
        final_state_slots = B

    if inplace_final_state:
        assert state is not None
        final_state_arg = state
        final_state_result = state
    else:
        if state_v_first:
            final_state_arg = q.new_empty(
                final_state_slots, HV, V, K, dtype=state.dtype
            )
        else:
            final_state_arg = q.new_empty(
                final_state_slots, HV, K, V, dtype=state.dtype
            )
        final_state_result = final_state_arg
    if is_continuous_batching is not True and store_final_state is not True:
        final_state_result = None

    kernel_fn = _get_kda_decode_kernel(
        qk_head=Hq,
        head=HV,
        dim_k=K,
        dim_v=V,
        input_dtype=input_dtype,
        state_dtype=state_dtype,
        gate_batch_dtype=gate_batch_dtype,
        dt_bias_dtype=dt_bias_dtype,
        output_dtype=output_dtype,
        b_dtype=b_dtype,
        use_qk_l2norm=bool(use_qk_l2norm),
        v_tile=kernel_config["v_tile"],
        num_blocks_per_state=kernel_config["num_blocks_per_state"],
        use_initial_state=bool(use_initial_state),
        is_varlen=bool(is_varlen),
        inplace_final_state=bool(inplace_final_state),
        is_beta_headwise=bool(is_beta_headwise),
        is_continuous_batching=bool(is_continuous_batching),
        is_spec_decoding=bool(is_spec_decoding),
        store_final_state=bool(store_final_state),
        has_dt_bias=bool(has_dt_bias),
        use_gate_in_kernel=bool(use_gate_in_kernel),
        use_lower_bound=bool(use_lower_bound),
        apply_beta_sigmoid=bool(apply_beta_sigmoid),
        allow_neg_eigval=bool(allow_neg_eigval),
        state_v_first=bool(state_v_first),
    )

    # FakeTensor dry runs resolve and compile the selected specialization, but
    # must stop before TileLang attempts to launch it with null data pointers.
    raise_complete_if_dry_run()

    kernel_fn(
        q_arg,
        k_arg,
        v_arg,
        A_log_arg,
        g_arg,
        dt_bias_arg,
        b_arg,
        cu_seqlens_arg,
        num_accepted_tokens_arg,
        T_fixed_arg,
        float(scale),
        float(lower_bound),
        state_indices_arg,
        state,
        final_state_arg,
        output_arg,
    )
    if not is_varlen:
        output_arg = output.reshape(B, T_fixed_host, HV, V)
    return output_arg, final_state_result
