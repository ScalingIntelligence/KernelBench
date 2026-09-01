"""TileLang backend for the active GDN unified decode path.

This module intentionally stays backend-oriented: it builds and runs the active
VK-layout, FP32-state decode kernel and assumes API-level validation is handled
by ``mate.gdn_decode``.
"""

import functools

import tilelang
import tilelang.language as T
import torch

__all__ = ["run_gated_delta_rule_decode_vk_fp32"]

_LOG2E = 1.4426950408889634
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0
_KERNEL_THREADS = 128

_STATE_TME_INNER_CACHE_POLICY = "cache_normal"
_STATE_TME_OUTER_CACHE_POLICY = "cache_none"

# Tuned on S5000 from benchmarks/perf_all.log with continuous batch ranges.
_AUTO_TUNED_BATCH_CONFIGS = (
    (2, (8, 1, 8)),
    (4, (8, 1, 4)),
    (16, (8, 2, 2)),
)
_AUTO_TUNED_LARGE_BATCH_CONFIG = (4, 1, 8)
_LOW_PARALLEL_SPLIT_CTA_THRESHOLD = 256
_LOW_PARALLEL_SPLIT_CONFIG = (8, 1, 4)


def _exp2_f32(value):
    return T.exp2(value * _LOG2E)


def _make_kernel_config(v_tile: int, stage: int, num_blocks_per_state: int) -> dict:
    return {
        "v_tile": v_tile,
        "stage": stage,
        "num_blocks_per_state": num_blocks_per_state,
    }


def _should_use_low_parallel_split(
    batch: int, head: int, num_blocks_per_state: int
) -> bool:
    if num_blocks_per_state >= _LOW_PARALLEL_SPLIT_CONFIG[2]:
        return False
    return batch * head * num_blocks_per_state < _LOW_PARALLEL_SPLIT_CTA_THRESHOLD


@functools.lru_cache(maxsize=64)
def _resolve_autotuned_kernel_config_tuple(
    batch: int, head: int
) -> tuple[int, int, int]:
    for max_batch, config in _AUTO_TUNED_BATCH_CONFIGS:
        if batch <= max_batch:
            break
    else:
        config = _AUTO_TUNED_LARGE_BATCH_CONFIG

    _, _, num_blocks_per_state = config
    if _should_use_low_parallel_split(batch, head, num_blocks_per_state):
        config = _LOW_PARALLEL_SPLIT_CONFIG
    return config


def _resolve_autotuned_kernel_config(batch: int, head: int) -> dict:
    config = _resolve_autotuned_kernel_config_tuple(batch, head)
    return _make_kernel_config(*config)


def _build_decode_fp32_vk_kernel_factory(
    qk_head: int,
    head: int,
    dim_k: int,
    dim_v: int,
    input_dtype: str,
    gate_batch_dtype: str,
    dt_bias_dtype: str,
    output_dtype: str,
    use_qk_l2norm: bool,
    v_tile: int,
    num_blocks_per_state: int,
    stage: int,
    use_identity_state_indices: bool,
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
    if dim_v % v_tile != 0:
        raise ValueError(f"dim_v={dim_v} must be divisible by v_tile={v_tile}")
    if v_tile % 4 != 0:
        raise ValueError(f"v_tile={v_tile} must be divisible by 4 for row mapping.")
    if input_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"Unsupported input_dtype={input_dtype}")
    if gate_batch_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"Unsupported gate_batch_dtype={gate_batch_dtype}")
    if dt_bias_dtype not in ("float32", "bfloat16"):
        raise ValueError(f"Unsupported dt_bias_dtype={dt_bias_dtype}")
    if output_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"Unsupported output_dtype={output_dtype}")

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
    num_v_tiles_per_block = num_v_tiles // num_blocks_per_state
    if stage <= 0:
        raise ValueError("stage must be positive.")
    if stage > num_v_tiles_per_block:
        raise ValueError(
            f"stage={stage} exceeds num_v_tiles_per_block={num_v_tiles_per_block}."
        )
    if num_v_tiles_per_block * v_tile > _KERNEL_THREADS:
        raise ValueError(
            f"per-block output elements={num_v_tiles_per_block * v_tile} exceeds "
            f"kernel threads={_KERNEL_THREADS}."
        )
    vec_size = dim_k // 32

    batch = T.dynamic("batch")
    pool_size = T.dynamic("pool_size")
    q_stride_b = T.dynamic("q_stride_b")
    q_stride_h = T.dynamic("q_stride_h")
    q_stride_k = T.dynamic("q_stride_k")
    k_stride_b = T.dynamic("k_stride_b")
    k_stride_h = T.dynamic("k_stride_h")
    k_stride_k = T.dynamic("k_stride_k")
    v_stride_b = T.dynamic("v_stride_b")
    v_stride_h = T.dynamic("v_stride_h")
    v_stride_v = T.dynamic("v_stride_v")
    a_stride_b = T.dynamic("a_stride_b")
    a_stride_h = T.dynamic("a_stride_h")
    b_stride_b = T.dynamic("b_stride_b")
    b_stride_h = T.dynamic("b_stride_h")
    o_stride_b = T.dynamic("o_stride_b")
    o_stride_h = T.dynamic("o_stride_h")
    o_stride_v = T.dynamic("o_stride_v")
    state_indices_stride_b = T.dynamic("state_indices_stride_b")

    q_shape = (batch, qk_head, dim_k)
    k_shape = (batch, qk_head, dim_k)
    v_shape = (batch, head, dim_v)
    gate_shape = (batch, head)
    o_shape = (batch, head, dim_v)
    state_indices_shape = (batch,)
    q_strides = (q_stride_b, q_stride_h, q_stride_k)
    k_strides = (k_stride_b, k_stride_h, k_stride_k)
    v_strides = (v_stride_b, v_stride_h, v_stride_v)
    a_strides = (a_stride_b, a_stride_h)
    b_strides = (b_stride_b, b_stride_h)
    o_strides = (o_stride_b, o_stride_h, o_stride_v)
    state_indices_strides = (state_indices_stride_b,)

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
    def _decode_func():
        mbarrier_list = [_KERNEL_THREADS] * stage

        @T.prim_func
        def gated_deltanet_decode_fp32_vk(
            q: T.StridedTensor(q_shape, q_strides, input_dtype),
            k: T.StridedTensor(k_shape, k_strides, input_dtype),
            v: T.StridedTensor(v_shape, v_strides, input_dtype),
            A_log: T.Tensor([head], "float32"),
            a: T.StridedTensor(gate_shape, a_strides, gate_batch_dtype),
            dt_bias: T.Tensor([head], dt_bias_dtype),
            b: T.StridedTensor(gate_shape, b_strides, gate_batch_dtype),
            scale: T.float32,
            state_indices: T.StridedTensor(
                state_indices_shape, state_indices_strides, "int32"
            ),
            state: T.Tensor([pool_size, head, dim_v, dim_k], "float32"),
            o: T.StridedTensor(o_shape, o_strides, output_dtype),
        ):
            with T.Kernel(
                batch * num_blocks_per_state, head, threads=_KERNEL_THREADS
            ) as (bx, hid):
                bid = bx // num_blocks_per_state
                block_inner = bx % num_blocks_per_state
                start_v_tile = block_inner * num_v_tiles_per_block

                state_load_stage = T.alloc_shared([stage, v_tile, dim_k], "float32")
                output_tile = T.alloc_shared(
                    [num_v_tiles_per_block * v_tile], output_dtype
                )
                value_tile = T.alloc_shared([dim_v], "float32")
                state_store_stage = T.alloc_shared([2, v_tile, dim_k], "float32")

                q_reg = T.alloc_local([vec_size], "float32")
                k_reg = T.alloc_local([vec_size], "float32")
                h_reg = T.alloc_local([vec_size], "float32")

                sum_q = T.alloc_local([1], "float32")
                sum_k = T.alloc_local([1], "float32")
                alpha_reg = T.alloc_local([1], "float32")
                beta_reg = T.alloc_local([1], "float32")
                sum_hk = T.alloc_local([1], "float32")
                sum_hq = T.alloc_local([1], "float32")
                stage_idx_var = T.alloc_var(T.int32)

                mbars = T.alloc_barrier(mbarrier_list)

                qk_hid = hid // head_group_size
                tid = T.get_thread_binding()
                lane = tid % 32
                warp = tid // 32
                k_start = lane * vec_size
                if use_identity_state_indices:
                    state_slot = bid
                else:
                    state_slot = state_indices[bid]

                if state_slot >= 0:
                    # Prologue: preload state tiles into stage buffers.
                    prefetch_count = min(stage, num_v_tiles_per_block)
                    for i in T.serial(prefetch_count):
                        prologue_v_tile = start_v_tile + i
                        prologue_v_base = prologue_v_tile * v_tile
                        T.tma_copy(
                            state[state_slot, hid, prologue_v_base, 0],
                            state_load_stage[i, :, :],
                            barrier=mbars[i],
                            inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                            outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                        )
                        T.mbarrier_arrive(mbarrier=mbars[i])

                    alpha_reg[0] = 0.0
                    beta_reg[0] = 0.0
                    if lane == 0:
                        A_log_val = T.cast(A_log[hid], "float32")
                        a_val = T.cast(a[bid, hid], "float32")
                        dt_bias_val = T.cast(dt_bias[hid], "float32")
                        b_val = T.cast(b[bid, hid], "float32")

                        x = a_val + dt_bias_val
                        beta_x = _SOFTPLUS_BETA * x
                        softplus_x = T.if_then_else(
                            beta_x <= _SOFTPLUS_THRESHOLD,
                            (1.0 / _SOFTPLUS_BETA) * T.log(1.0 + _exp2_f32(beta_x)),
                            x,
                        )
                        g_val = -_exp2_f32(A_log_val) * softplus_x
                        beta_reg[0] = 1.0 / (1.0 + _exp2_f32(-b_val))
                        alpha_reg[0] = _exp2_f32(g_val)

                    mask = 0xFFFFFFFF
                    alpha_reg[0] = T.shfl_sync(mask, alpha_reg[0], 0)
                    beta_reg[0] = T.shfl_sync(mask, beta_reg[0], 0)

                    for i in T.vectorized(vec_size):
                        kk = k_start + i
                        value_tile[kk] = T.cast(v[bid, hid, kk], "float32")

                    for i in T.vectorized(vec_size):
                        kk = k_start + i
                        q_reg[i] = T.cast(q[bid, qk_hid, kk], "float32")
                        k_reg[i] = T.cast(k[bid, qk_hid, kk], "float32")

                    if use_qk_l2norm:
                        sum_q[0] = 0.0
                        sum_k[0] = 0.0
                        for i in T.serial(vec_size):
                            sum_q[0] += q_reg[i] * q_reg[i]
                            sum_k[0] += k_reg[i] * k_reg[i]
                        for offset in T.unroll(5):
                            mask = 16 >> offset
                            sum_q[0] += T.shfl_xor(sum_q[0], mask)
                            sum_k[0] += T.shfl_xor(sum_k[0], mask)
                        inv_norm_q = T.rsqrt(sum_q[0] + 1e-6)
                        inv_norm_k = T.rsqrt(sum_k[0] + 1e-6)
                        for i in T.vectorized(vec_size):
                            q_reg[i] = q_reg[i] * inv_norm_q
                            k_reg[i] = k_reg[i] * inv_norm_k

                    for i in T.vectorized(vec_size):
                        q_reg[i] = q_reg[i] * scale

                    T.sync_threads()

                    for local_v_tile in T.serial(0, num_v_tiles_per_block):
                        stage_idx_var = local_v_tile % stage
                        parity = (local_v_tile // stage) & 1
                        global_v_tile = start_v_tile + local_v_tile
                        global_v_base = global_v_tile * v_tile

                        # Store the previous updated state tile.
                        if local_v_tile > 0:
                            # Keep one TME store overlapped with compute, but wait before
                            # reusing its shared-memory stage two iterations later.
                            if local_v_tile > 1:
                                T.tma_store_wait()
                            global_prev_v_base = global_v_base - v_tile
                            T.tma_copy(
                                state_store_stage[(local_v_tile - 1) % 2, :, :],
                                state[
                                    state_slot,
                                    hid,
                                    global_prev_v_base : global_prev_v_base + v_tile,
                                    :,
                                ],
                                inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                            )
                            # Keep the CTA in step after the issuing warp commits the store.
                            T.sync_threads()

                        T.mbarrier_wait_parity(
                            mbarrier=mbars[stage_idx_var], parity=parity
                        )

                        for row_base in range(0, v_tile, 4):
                            row_idx = row_base + warp
                            global_row = global_v_base + row_idx
                            sum_hk[0] = 0.0
                            sum_hq[0] = 0.0

                            for i in T.vectorized(vec_size):
                                h_reg[i] = T.cast(
                                    state_load_stage[
                                        stage_idx_var, row_idx, k_start + i
                                    ],
                                    "float32",
                                )

                            for i in T.unroll(vec_size):
                                h_reg[i] = h_reg[i] * alpha_reg[0]
                                sum_hk[0] += h_reg[i] * k_reg[i]

                            for offset in T.unroll(5):
                                mask = 16 >> offset
                                sum_hk[0] += T.shfl_xor(sum_hk[0], mask)

                            v_new = (value_tile[global_row] - sum_hk[0]) * beta_reg[0]

                            for i in T.unroll(vec_size):
                                h_reg[i] += k_reg[i] * v_new
                                sum_hq[0] += h_reg[i] * q_reg[i]

                            # Stage updated state for the next TME store.
                            for i in T.vectorized(vec_size):
                                state_store_stage[
                                    local_v_tile % 2, row_idx, k_start + i
                                ] = h_reg[i]

                            for offset in T.unroll(5):
                                mask = 16 >> offset
                                sum_hq[0] += T.shfl_xor(sum_hq[0], mask)

                            o_idx = local_v_tile * v_tile + row_idx
                            if lane == 0 and o_idx < num_v_tiles_per_block * v_tile:
                                output_tile[o_idx] = sum_hq[0]

                        # Ensure state_load_stage[stage_idx_var] is consumed before refill.
                        T.sync_threads()
                        next_local_v_tile = local_v_tile + stage
                        if next_local_v_tile < num_v_tiles_per_block:
                            global_next_v_tile = start_v_tile + next_local_v_tile
                            global_next_v_base = global_next_v_tile * v_tile
                            T.tma_copy(
                                state[state_slot, hid, global_next_v_base, 0],
                                state_load_stage[stage_idx_var, :, :],
                                barrier=mbars[stage_idx_var],
                                inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                            )
                            T.mbarrier_arrive(mbarrier=mbars[stage_idx_var])

                    # Epilogue: store the last updated state tile.
                    global_prev_v_base_epi = (
                        start_v_tile + num_v_tiles_per_block - 1
                    ) * v_tile
                    T.tma_copy(
                        state_store_stage[(num_v_tiles_per_block - 1) % 2, :, :],
                        state[
                            state_slot,
                            hid,
                            global_prev_v_base_epi : global_prev_v_base_epi + v_tile,
                            :,
                        ],
                        inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                        outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                    )

                    if tid < num_v_tiles_per_block * v_tile:
                        o[bid, hid, start_v_tile * v_tile + tid] = output_tile[tid]

                    T.tma_store_wait()
                else:
                    if tid < num_v_tiles_per_block * v_tile:
                        o[bid, hid, start_v_tile * v_tile + tid] = T.cast(
                            0.0, output_dtype
                        )

        return gated_deltanet_decode_fp32_vk

    return _decode_func


@functools.lru_cache(maxsize=32)
def _get_decode_fp32_vk_kernel(
    qk_head: int,
    head: int,
    dim_k: int,
    dim_v: int,
    input_dtype: str,
    gate_batch_dtype: str,
    dt_bias_dtype: str,
    output_dtype: str,
    use_qk_l2norm: bool,
    v_tile: int,
    num_blocks_per_state: int,
    stage: int,
    use_identity_state_indices: bool,
):
    return _build_decode_fp32_vk_kernel_factory(
        qk_head=qk_head,
        head=head,
        dim_k=dim_k,
        dim_v=dim_v,
        input_dtype=input_dtype,
        gate_batch_dtype=gate_batch_dtype,
        dt_bias_dtype=dt_bias_dtype,
        output_dtype=output_dtype,
        use_qk_l2norm=use_qk_l2norm,
        v_tile=v_tile,
        num_blocks_per_state=num_blocks_per_state,
        stage=stage,
        use_identity_state_indices=use_identity_state_indices,
    )()


def run_gated_delta_rule_decode_vk_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    *,
    scale: float,
    use_qk_l2norm: bool,
):
    """Run the active VK-layout FP32-state decode backend.

    Inputs are expected to be validated by ``mate.gdn_decode.gated_delta_rule_decode``.
    Writes final output dtype in-kernel while updating ``state`` in-place.
    """
    B, _, Hq, K = q.shape
    _, _, HV, V = v.shape
    kernel_config = _resolve_autotuned_kernel_config(B, HV)
    use_identity_state_indices = state_indices is None

    q_arg = q.squeeze(1)
    k_arg = k.squeeze(1)
    v_arg = v.squeeze(1)
    a_arg = a.squeeze(1)
    b_arg = b.squeeze(1)
    output_arg = output.squeeze(1) if output.dim() == 4 else output
    input_dtype = str(q.dtype).split(".")[-1]
    gate_batch_dtype = str(a.dtype).split(".")[-1]
    dt_bias_dtype = str(dt_bias.dtype).split(".")[-1]
    output_dtype = str(output_arg.dtype).split(".")[-1]
    state_indices_arg = (
        torch.empty((B,), dtype=torch.int32, device=q.device)
        if use_identity_state_indices
        else state_indices
    )

    kernel_fn = _get_decode_fp32_vk_kernel(
        qk_head=Hq,
        head=HV,
        dim_k=K,
        dim_v=V,
        input_dtype=input_dtype,
        gate_batch_dtype=gate_batch_dtype,
        dt_bias_dtype=dt_bias_dtype,
        output_dtype=output_dtype,
        use_qk_l2norm=bool(use_qk_l2norm),
        v_tile=kernel_config["v_tile"],
        num_blocks_per_state=kernel_config["num_blocks_per_state"],
        stage=kernel_config["stage"],
        use_identity_state_indices=use_identity_state_indices,
    )

    kernel_fn(
        q_arg,
        k_arg,
        v_arg,
        A_log,
        a_arg,
        dt_bias,
        b_arg,
        float(scale),
        state_indices_arg,
        state,
        output_arg,
    )
