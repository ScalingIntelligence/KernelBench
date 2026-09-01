# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from tvm import tir
# tilelang.disable_cache()


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither MUSA nor CUDA is available")


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
    verbose=True,
    compile_flags=[
        #"-Od3",
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        "-fno-strict-aliasing",
        "-mllvm",
        "-misched=mtgpu-max-ilp",
        "-mllvm",
        "-mtgpu-tiny-offset-hint=1",
        "-mllvm",
        "-misched-recompute-slotindex=1",
        #"-mllvm",
        #"-mtgpu-combine-instr-with-burst=1",
        "-mllvm",
        "-mtgpu-combine-fop-instr=1",
    ])
def sparse_attention_fwd_kernel_v2(
    num_heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    block_H=64,
    block_I=64,
    threads=640,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert (topk %
            block_I == 0), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = num_heads // kv_group
    q_shape = [seq_len, num_heads, dim + tail_dim]
    kv_shape = [seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [seq_len, num_heads, dim]
    lse_shape = [seq_len, num_heads]
    indices_shape = [seq_len, kv_group, topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), block_H)
    if padded_H != H:
        assert kv_group == 1
    kv_latent_dtype = "float8_e4m3"
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim
    L = block_I // 8
    dim_bytes = 656
    if head_kv > block_H:
        assert head_kv % block_H == 0, "head_kv should be a multiple of block_H"
        REPLICATE_H = head_kv // block_H
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else block_H
    PV_MMA_N = 64 if block_H == 32 else 128
    @T.prim_func
    def main(
            Q: T.Tensor([seq_len, num_heads, dim + tail_dim], dtype),  # type: ignore
            KV: T.Tensor([seq_len_kv, kv_group, dim_bytes], kv_latent_dtype),  # type: ignore
            K_pe: T.Tensor([seq_len_kv, kv_group, dim_bytes // 2], dtype),  # type: ignore
            Quant_scales: T.Tensor([seq_len_kv, kv_group, dim_bytes // 4],
                                   T.float32),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            debug_K: T.Tensor([seq_len_kv, kv_group, 512], kv_latent_dtype),  # type: ignore
    ):
        with T.Kernel(
                seq_len * REPLICATE_H, kv_group, threads=threads) as (
                    bx,
                    by,
                ):
            KV_shared_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_r = T.alloc_shared([BI, D // 2], dtype)
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)

            V_shared_0 = T.alloc_shared([BI, D // 4], dtype)
            V_shared_1 = T.alloc_shared([BI, D // 4], dtype)

            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sum_exp_inv_shared = T.alloc_shared([H_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([H_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([BI], "bool", scope="shared")
            kv_indices = T.alloc_shared([BI], "int32", scope="shared")
            Quant_shared = T.alloc_shared([BI, 4], "float32")

            bar_q = T.alloc_barrier(arrive_count=512)
            bar_indices_ready = T.alloc_barrier(arrive_count=128)
            bar_kv0_ready = T.alloc_barrier(arrive_count=128)
            bar_kv1_ready = T.alloc_barrier(arrive_count=128)
            bar_kv0_lma_read_ready = T.alloc_barrier(arrive_count=256)
            bar_kv1_lma_read_ready = T.alloc_barrier(arrive_count=256)
            bar_kv0_quant_ready = T.alloc_barrier(arrive_count=256)
            bar_kv1_quant_ready = T.alloc_barrier(arrive_count=256)

            # bar_kv1_read_ready = T.alloc_barrier(arrive_count=256)
            bar_kv0_free = T.alloc_barrier(arrive_count=256)
            bar_kv1_free = T.alloc_barrier(arrive_count=256)

            bar_vl0_ready = T.alloc_barrier(arrive_count=256)
            bar_vl1_ready = T.alloc_barrier(arrive_count=256)
            bar_vr0_ready = T.alloc_barrier(arrive_count=256)
            bar_vr1_ready = T.alloc_barrier(arrive_count=256)
            bar_vl0_free = T.alloc_barrier(arrive_count=256)
            bar_vl1_free = T.alloc_barrier(arrive_count=256)

            bar_p_ready = T.alloc_barrier(arrive_count=256)
            bar_final = T.alloc_barrier(arrive_count=256)

            q_robust_desc = T.make_robust_desc(
                T.address_of(Q[0, 0, 0]), (seq_len * num_heads * (dim + tail_dim)) * 2)
            kv_robust_desc = T.make_robust_desc(
                T.address_of(KV[0, 0, 0]), (seq_len_kv * kv_group * (dim_bytes)))

            T.sync_threads()

            mask = T.alloc_fragment([BI], "bool")

            g_i = by
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block
            tid = T.get_thread_binding()

            if tid < 512:
                T.copy(
                    Q[s_i, H0:H1, 0:D // 2],
                    Q_shared_l,
                    force_async_copy=True,
                    src_robust_desc=q_robust_desc)
                T.copy(
                    Q[s_i, H0:H1, D // 2:D],
                    Q_shared_r,
                    force_async_copy=True,
                    src_robust_desc=q_robust_desc)
                T.copy(
                    Q[s_i, H0:H1, D:],
                    Q_tail_shared,
                    force_async_copy=True,
                    src_robust_desc=q_robust_desc)

                tir.call_extern("void", "__musa_memcpy_g2s_commit_group")
                tir.call_extern("void", "__musa_memcpy_g2s_wait_group", 0)
                T.barrier_arrive(bar_q)
                T.barrier_wait(bar_q, 0)

            if tid < 256:
                # consumer 0
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_inv = T.alloc_fragment([H_per_block], accum_dtype)
                alpha_local = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                acc_s_cast = T.alloc_fragment([H_per_block, BI], dtype)
                acc_o_l_0 = T.alloc_fragment([H_per_block, D // 4], accum_dtype)
                acc_o_l_1 = T.alloc_fragment([H_per_block, D // 4], accum_dtype)
                kv_reg_l = T.alloc_local([64], dtype)
                kv_reg_l_fp16 = T.view(kv_reg_l, [64], T.float16)
                kv_reg_l_bf16_load = T.alloc_local([32], T.bfloat16)
                kv_reg_l_fp8 = T.view(kv_reg_l_bf16_load, [64], kv_latent_dtype)
                quant_local_l = T.alloc_local([2, 2], T.float32)
                ldg_tx = (tid) % 8
                ldg_ty = (tid) // 8
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))
                T.fill(acc_o_l_0, 0)
                T.fill(acc_o_l_1, 0)

                for i_i in range(T.ceildiv(topk, block_I)):                    
                    T.barrier_wait(bar_kv0_ready, (i_i & 1))

                    T.copy(Quant_shared[ldg_ty, 0:2], quant_local_l[0, :])
                    T.copy(Quant_shared[ldg_ty + 32, 0:2], quant_local_l[1, :])

                    T.annotate_layout(
                        {
                            KV_shared_l[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared_l[:, :], k_major=True)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    # KV_l fp8 quant
                    # load FP8 form SMEM
                    # U 01 -> 0-127
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(4):
                                kv_reg_l_bf16_load[r * 16 + u * 4 + v] = KV_shared_l[
                                    (ldg_ty + r * 32),
                                    64 * u + ldg_tx * 8 + v,
                                ]
                    tir.call_extern("void", "__musa_lma_wait")

                    # CVT FP8 -> FP16, KV_l_0
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + u * 8 + v
                                kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]

                    # FOP.MUL FP16 * scale > BF16.  KV_l_0
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + u * 8 + v
                                kv_reg_l[idx] = T.Cast("bfloat16",
                                                    kv_reg_l_fp16[idx] * quant_local_l[r, 0])
                    
                    # STS BF16 KV_l_0
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                KV_shared_l[ldg_ty + r * 32,
                                            64 * u + ldg_tx * 8 + v] = kv_reg_l[r * 32 + u * 8 + v]

                    # load FP8 form SMEM
                    # U 23 -> 128-255
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(4):
                                kv_reg_l_bf16_load[r * 16 + (u+2) * 4 + v] = KV_shared_l[
                                    (ldg_ty + r * 32),
                                    64 * (u+2) + ldg_tx * 8 + v,
                                ]
                    tir.call_extern("void", "__musa_lma_wait")
                    # T.barrier_arrive(bar_kv0_lma_read_ready)
                    # CVT FP8 -> FP16, KV_l_1
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + (u+2) * 8 + v
                                kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]
                    
                    # FOP.MUL FP16 * scale > BF16.  KV_l_1
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + (u+2) * 8 + v
                                kv_reg_l[idx] = T.Cast("bfloat16",
                                                    kv_reg_l_fp16[idx] * quant_local_l[r, 1])

                    
                    # T.barrier_wait(bar_kv0_lma_read_ready, i_i & 1)
                    # STS BF16 KV_l_1
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                KV_shared_l[ldg_ty + r * 32,
                                            64 * (u+2) + ldg_tx * 8 + v] = kv_reg_l[r * 32 + (u+2) * 8 + v]
                    
                    tir.call_extern("void", "__musa_lma_wait")
                    T.barrier_arrive(bar_kv0_quant_ready)

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[bi_i % 8 * 8 + bi_i // 8], 0,
                                                          -(2**30))

                    T.barrier_wait(bar_kv0_quant_ready, (i_i & 1))
                    T.gemm(
                        Q_shared_l[:, :],
                        KV_shared_l[:, :],
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)
                    tir.call_extern("void", "__musa_tce_commit_group")
                    tir.call_extern("void", "__musa_tce_wait_group", 0)
                    T.barrier_arrive(bar_kv0_free)

                    T.barrier_wait(bar_kv1_quant_ready, (i_i & 1))
                    T.annotate_layout(
                        {
                            KV_shared_r[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared_r[:, :], k_major=True)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    T.gemm(
                        Q_shared_r,
                        KV_shared_r[:, :],
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)
                    tir.call_extern("void", "__musa_tce_commit_group")
                    tir.call_extern("void", "__musa_tce_wait_group", 0)
                    

                    T.annotate_layout(
                        {
                            K_tail_shared[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    K_tail_shared[:, :], k_major=True)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared[:, :],
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)
                    tir.call_extern("void", "__musa_tce_commit_group")
                    tir.call_extern("void", "__musa_tce_wait_group", 0)
                    T.barrier_arrive(bar_kv1_free)

                    # online softmax
                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i_prev[h_i], m_i[h_i])
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)

                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 4):
                        acc_o_l_0[h_i, d_i] *= alpha_local[h_i]
                        acc_o_l_1[h_i, d_i] *= alpha_local[h_i]

                    T.copy(alpha_local, alpha_shared)
                    T.copy(acc_s, acc_s_cast)
                    for i, t in T.Parallel(H_per_block, 8):
                        base = t * L
                        for l in T.vectorized(L):
                            S_shared[i, base + l] = acc_s_cast[i, l * 8 + t]

                    tir.call_extern("void", "__musa_lma_wait")
                    T.barrier_arrive(bar_p_ready)

                    T.annotate_layout(
                        {
                            V_shared_0[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    V_shared_0[:, :], continuity=PV_MMA_N, k_major=False)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    # STS 2 V Buf 0
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                V_shared_0[
                                    ((ldg_ty + r * 32) % 8) * (block_I // 8) +
                                    (ldg_ty + r * 32) // 8,
                                    64 * u + ldg_tx * 8 + v,
                                ] = kv_reg_l[r * 32 + u * 8 + v]
                    tir.call_extern("void", "__musa_lma_wait")
                    T.barrier_arrive(bar_vl0_ready)
                    T.barrier_wait(bar_vl0_ready, (i_i & 1))

                    T.gemm(S_shared, V_shared_0, acc_o_l_0, policy=T.GemmWarpPolicy.FullRow)
                    tir.call_extern("void", "__musa_tce_commit_group")
                    T.annotate_layout(
                        {
                            V_shared_1[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    V_shared_1[:, :], continuity=PV_MMA_N, k_major=False)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    # STS 2 V Buf 1
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                V_shared_1[
                                    ((ldg_ty + r * 32) % 8) * (block_I // 8) +
                                    (ldg_ty + r * 32) // 8,
                                    64 * u + ldg_tx * 8 + v,
                                ] = kv_reg_l[r * 32 + (u + 2) * 8 + v]

                    tir.call_extern("void", "__musa_tce_wait_group", 0)
                    T.barrier_arrive(bar_vl0_free)

                    tir.call_extern("void", "__musa_lma_wait")
                    T.barrier_arrive(bar_vl1_ready)
                    T.barrier_wait(bar_vl1_ready, ((i_i) & 1))

                    T.gemm(
                        S_shared,
                        V_shared_1,
                        acc_o_l_1,
                        transpose_B=False,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)
                    tir.call_extern("void", "__musa_tce_commit_group")
                    tir.call_extern("void", "__musa_tce_wait_group", 0)
                    T.barrier_arrive(bar_vl1_free)

                for h_i in T.Parallel(H_per_block):
                    sumexp_inv[h_i] = 1 / sumexp[h_i]
                for h_i in T.Parallel(H_per_block):
                    sum_exp_inv_shared[h_i] = sumexp_inv[h_i]
                T.barrier_arrive(bar_final)
                for h_i, d_i in T.Parallel(H_per_block, D // 4):
                    acc_o_l_0[h_i, d_i] *= sumexp_inv[h_i]
                    acc_o_l_1[h_i, d_i] *= sumexp_inv[h_i]

                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

                T.copy(acc_o_l_0, Output[s_i, H0:H1, 0:D // 4])
                T.copy(acc_o_l_1, Output[s_i, H0:H1, D // 4:D // 2])
            elif tid >= 256 and tid < 512:
                # consumer 1
                acc_o_r_0 = T.alloc_fragment([H_per_block, D // 4], accum_dtype)
                acc_o_r_1 = T.alloc_fragment([H_per_block, D // 4], accum_dtype)
                kv_reg_r = T.alloc_local([64], dtype)
                kv_reg_r_fp16 = T.view(kv_reg_r, [64], T.float16)

                kv_reg_r_fp8 = T.alloc_local([64], kv_latent_dtype)
                kv_reg_r_bf16 = T.view(kv_reg_r_fp8, [32], T.bfloat16)

                quant_local_r = T.alloc_local([2, 2], T.float32)
                T.fill(acc_o_r_0, 0)
                T.fill(acc_o_r_1, 0)

                ldg_tx = (tid - 256) % 8
                ldg_ty = (tid - 256) // 8

                for i_i in range(T.ceildiv(topk, block_I)):
                    T.barrier_wait(bar_kv1_ready, (i_i & 1))

                    T.copy(Quant_shared[ldg_ty, 2:4], quant_local_r[0, :])
                    T.copy(Quant_shared[ldg_ty + 32, 2:4], quant_local_r[1, :])

                    T.annotate_layout(
                        {
                            KV_shared_r[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared_r[:, :], k_major=True)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    # KV_r fp8 quant
                    # load FP8 form SMEM
                    # U 01 -> 0-127
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(4):
                                kv_reg_r_bf16[r * 16 + u * 4 + v] = KV_shared_r[
                                    ldg_ty + r * 32,
                                    64 * u + ldg_tx * 8 + v,
                                ]
                    tir.call_extern("void", "__musa_lma_wait")

                    # CVT FP8 -> FP16, KV_l_0
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + u * 8 + v
                                kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]


                    # FOP.MUL FP16 * scale > BF16.  KV_l_0
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + u * 8 + v
                                kv_reg_r[idx] = T.Cast("bfloat16",
                                                    kv_reg_r_fp16[idx] * quant_local_r[r, 0])


                    # STS BF16 KV_r_0
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                KV_shared_r[ldg_ty + r * 32,
                                            64 * u + ldg_tx * 8 + v] = kv_reg_r[r * 32 + u * 8 + v]

                    # load FP8 form SMEM
                    # U 23 -> 128-255
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(4):
                                kv_reg_r_bf16[r * 16 + (u+2) * 4 + v] = KV_shared_r[
                                    (ldg_ty + r * 32),
                                    64 * (u+2) + ldg_tx * 8 + v,
                                ]
                    tir.call_extern("void", "__musa_lma_wait")
                    # T.barrier_arrive(bar_kv1_lma_read_ready)
                    # CVT FP8 -> FP16, KV_l_1
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + (u+2) * 8 + v
                                kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]
                    
                    # FOP.MUL FP16 * scale > BF16.  KV_l_1
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + (u+2) * 8 + v
                                kv_reg_r[idx] = T.Cast("bfloat16",
                                                    kv_reg_r_fp16[idx] * quant_local_r[r, 1])


                    # T.barrier_wait(bar_kv1_lma_read_ready, i_i & 1)
                    # STS BF16 KV_r_1
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                KV_shared_r[ldg_ty + r * 32,
                                            64 * (u+2) + ldg_tx * 8 + v] = kv_reg_r[r * 32 + (u+2) * 8 + v]
                    
                    tir.call_extern("void", "__musa_lma_wait")
                    T.barrier_arrive(bar_kv1_quant_ready)

                    T.barrier_wait(bar_vl0_free, ((i_i) & 1))
                    # STS 2 VR Buf 0
                    T.annotate_layout(
                        {
                            V_shared_0[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    V_shared_0[:, :], continuity=PV_MMA_N, k_major=False)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                V_shared_0[
                                    ((ldg_ty + r * 32) % 8) * (block_I // 8) +
                                    (ldg_ty + r * 32) // 8,
                                    64 * u + ldg_tx * 8 + v,
                                ] = kv_reg_r[r * 32 + u * 8 + v]

                    tir.call_extern("void", "__musa_lma_wait")
                    T.barrier_arrive(bar_vr0_ready)
                    T.barrier_wait(bar_vr0_ready, ((i_i) & 1))

                    # compute v4-v7
                    T.barrier_wait(bar_p_ready, (i_i & 1))
                    for h_i, d_i in T.Parallel(H_per_block, D // 4):
                        acc_o_r_0[h_i, d_i] *= alpha_shared[h_i]
                        acc_o_r_1[h_i, d_i] *= alpha_shared[h_i]

                    # bar arrive & wait
                    T.gemm(S_shared, V_shared_0, acc_o_r_0, policy=T.GemmWarpPolicy.FullRow)
                    tir.call_extern("void", "__musa_tce_commit_group")

                    T.barrier_wait(bar_vl1_free, ((i_i) & 1))
                    # STS 2 V Buf 1
                    T.annotate_layout(
                        {
                            V_shared_1[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    V_shared_1[:, :], continuity=PV_MMA_N, k_major=False)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                V_shared_1[
                                    ((ldg_ty + r * 32) % 8) * (block_I // 8) +
                                    (ldg_ty + r * 32) // 8,
                                    64 * u + ldg_tx * 8 + v,
                                ] = kv_reg_r[r * 32 + (u + 2) * 8 + v]
                    tir.call_extern("void", "__musa_lma_wait")
                    tir.call_extern("void", "__musa_tce_wait_group", 0)
                    T.barrier_arrive(bar_vr1_ready)
                    T.barrier_wait(bar_vr1_ready, ((i_i) & 1))

                    # compute v4-v7
                    T.gemm(S_shared, V_shared_1, acc_o_r_1, policy=T.GemmWarpPolicy.FullRow)
                    tir.call_extern("void", "__musa_tce_commit_group")
                    tir.call_extern("void", "__musa_tce_wait_group", 0)

                T.barrier_wait(bar_final, 0)
                for h_i, d_i in T.Parallel(H_per_block, D // 4):
                    acc_o_r_0[h_i, d_i] *= sum_exp_inv_shared[h_i]
                    acc_o_r_1[h_i, d_i] *= sum_exp_inv_shared[h_i]

                T.copy(acc_o_r_0, Output[s_i, H0:H1, D // 2:D // 2 + D // 4])
                T.copy(acc_o_r_1, Output[s_i, H0:H1, D // 2 + D // 4:D])
            elif tid >= 512:
                mask_local = T.alloc_local([4], "bool")
                indices_local = T.alloc_local([4], indices_dtype)

                kperm_mask_local = T.alloc_local([4], "bool")
                kperm_indices_local = T.alloc_local([4], "int32")

                # producer: 128 ldg_ty 16
                ldg_tx = (tid - 512) % 8
                ldg_ty = (tid - 512) // 8

                ldg_scale_tx = (tid - 512) % 2
                ldg_scale_ty = (tid - 512) // 2

                for i_i in range(T.ceildiv(topk, block_I)):
                    # LOAD Indices
                    for r in T.unroll(4):
                        kperm_indices_local[r] = Indices[s_i, g_i,
                                                         (i_i) * block_I + ((r * 16 + ldg_ty) % 8) *
                                                         (block_I // 8) + (r * 16 + ldg_ty) // 8]
                    for r in T.unroll(4):
                        kperm_mask_local[
                            r] = kperm_indices_local[r] >= 0 and kperm_indices_local[r] < seq_len_kv
                        kperm_indices_local[r] = T.if_then_else(
                            kperm_mask_local[r], kperm_indices_local[r],
                            (seq_len_kv * kv_group * (dim_bytes)) * 2)
                    
                    if ldg_tx == 0:
                        for r in T.unroll(4):
                            is_kv_valid[((r * 16 + ldg_ty) % 8) * (block_I // 8) +
                                        (r * 16 + ldg_ty) // 8] = kperm_mask_local[r]
                            kv_indices[((r * 16 + ldg_ty) % 8) * (block_I // 8) +
                                        (r * 16 + ldg_ty) // 8] = kperm_indices_local[r]

                    T.barrier_wait(bar_kv0_free, (i_i & 1) ^ 1)

                    # load k0-k3
                    # T.annotate_layout(
                    #         { KV_shared_l[:, :]: tilelang.layout.make_sqmma_swizzled_layout(KV_shared_l[:, :], k_major=True) },
                    #         allow_reannotation=True,
                    #         allow_buffer_region=True)
                    T.annotate_layout(
                        {
                            KV_shared_l[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared_l[:, :], k_major=True)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    for r in T.unroll(4):
                        for u in T.unroll(4):
                            for v in T.vectorized(4):
                                pass
                                T.copy(
                                    K_pe[kperm_indices_local[r], g_i, 32 * u + ldg_tx * 4 + v],
                                    KV_shared_l[r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v],
                                    force_async_copy=True,
                                    src_robust_desc=kv_robust_desc,
                                )
                    
                    tir.call_extern("void", "__musa_lma_wait")
                    T.barrier_arrive(bar_indices_ready)
                    T.barrier_wait(bar_indices_ready, (i_i & 1))
                    
                    for c in T.vectorized(2):
                        T.copy(Quant_scales[
                            kv_indices[ldg_scale_ty],
                            g_i,
                            128 + ldg_scale_tx * 2 + c,
                        ],
                        Quant_shared[ldg_scale_ty, ldg_scale_tx * 2 + c],
                        src_robust_desc=kv_robust_desc,
                        )

                    tir.call_extern("void", "__musa_memcpy_g2s_commit_group")
                    tir.call_extern("void", "__musa_memcpy_g2s_wait_group", 0)
                    T.barrier_arrive(bar_kv0_ready)

                    T.barrier_wait(bar_kv1_free, (i_i & 1) ^ 1)

                    # load k rope
                    T.annotate_layout(
                        {
                            K_tail_shared[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    K_tail_shared[:, :], k_major=True)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    for r in T.unroll(4):
                        for v in T.vectorized(8):
                            pass
                            T.copy(
                                K_pe[kperm_indices_local[r], g_i, D // 2 + 8 + ldg_tx * 8 + v],
                                K_tail_shared[r * 16 + ldg_ty, ldg_tx * 8 + v],
                                force_async_copy=True,
                                src_robust_desc=kv_robust_desc,
                            )
                    
                    # load k4-k7
                    # T.annotate_layout(
                    #         { KV_shared_r[:, :]: tilelang.layout.make_sqmma_swizzled_layout(KV_shared_r[:, :], k_major=True) },
                    #         allow_reannotation=True,
                    #         allow_buffer_region=True)
                    T.annotate_layout(
                        {
                            KV_shared_r[:, :]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared_r[:, :], k_major=True)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True)
                    for r in T.unroll(4):
                        for u in T.unroll(4):
                            for v in T.vectorized(4):
                                pass
                                T.copy(
                                    K_pe[kperm_indices_local[r], g_i,
                                         D // 4 + 32 * u + ldg_tx * 4 + v],
                                    KV_shared_r[r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v],
                                    force_async_copy=True,
                                    src_robust_desc=kv_robust_desc,
                                )
                    tir.call_extern("void", "__musa_memcpy_g2s_commit_group")
                    tir.call_extern("void", "__musa_memcpy_g2s_wait_group", 0)
                    T.barrier_arrive(bar_kv1_ready)



    return main


def sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    threads=512,
    verbose=False,
):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    seq_len, heads, dim_plus_tail_dim = q.shape
    seq_len_kv, kv_group, _ = kv.shape
    #  In FP8+sparse mode, each token's KV cache is 656 Bytes, structured as:
    #         - The shape of the tensor `k_cache` is (num_blocks*page_block_size*num_heads_k, head_dim), and num_heads_k must be 1.
    #         - First 512 bytes: The "quantized NoPE" part, containing 512 float8_e4m3 values.
    #         - Next 16 bytes: Scale factors, containing 4 float32 values. The first float32 is the scale for the first 128 float8_e4m3 values, the second for the next 128, and so on.
    #         - Last 128 bytes: The "RoPE" part, containing 64 bfloat16 values. This part is not quantized for accuracy.
    # assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v
    dim_bytes = 656
    assert kv.shape[-1] == dim_bytes
    tail_dim = dim_plus_tail_dim - dim
    _, _, topk = indices.shape
    assert indices.shape == (seq_len, kv_group, topk)

    # kernel = sparse_attention_fwd_kernel_v1(
    threads = 640
    kernel = sparse_attention_fwd_kernel_v2(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=kv_group,
        sm_scale=sm_scale,
        threads=threads,
    )
    if verbose:
        kernel.show_source()
    kv_latent_f8 = kv.view(torch.float8_e4m3fn)
    k_rope = kv.view(torch.bfloat16)
    scales = kv.view(torch.float32)
    # out [S_q, H, D]
    out = kernel(q, kv_latent_f8, k_rope, scales, indices)
    return out


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(0, 1)
    sq, h, dim_q = q.shape
    sk, g, _ = kv.shape

    # assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    # compressed_casual_mask = torch.arange(
    #     0, sq, dtype=torch.int32, device=q.device).view(-1, 1) >= torch.arange(
    #         1 - 1, sk * 1, 1, dtype=torch.int32, device=q.device).view(1, -1)

    indices_clamped = torch.where(indices < 0, sk, indices)
    mask = q.new_zeros(g_index, sq, sk + 1, dtype=torch.bool).scatter(2, indices_clamped.long(), 1)
    mask = mask[..., :-1]
    # mask = mask & compressed_casual_mask.view(1, sq, sk)
    mask[:, :1 - 1, 0] = True
    mask = mask.view(g_index, 1, sq, sk)

    q = q.view(sq, g, -1, dim_q)
    score = torch.einsum("mghd,ngd->ghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(g_index, h_index, sq, sk)
    p = p.view(g, -1, sq, sk)
    o = torch.einsum("ghmn,ngd->mghd", p.type(v.dtype), v)
    o = o.reshape(sq, h, dim_v)
    return o.to(torch.bfloat16), score


def test_sparse_mla_fwd(
    B=128,
    S=4096,
    SKV=8192,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    check_correctness=True,
    perf_test=False,
    threads=512,
):
    torch.random.manual_seed(0)
    device = get_test_device()
    q = torch.randn((S, H, DQK), dtype=dtype, device=device)
    kv = torch.randn((SKV, HKV, DQK), dtype=dtype, device=device)
    # kv = torch.ones((SKV, HKV, DQK), dtype=dtype, device=device).requires_grad_(True) *0.1

    indices = torch.full((S, HKV, topk), -1, dtype=torch.int32, device=device)
    for t in range(S):
        for h in range(HKV):
            i_i = torch.randperm(SKV, device=device)[:topk]
            # i_i = torch.arange(0, SKV, device=device)[:topk]
            # i_i = torch.range(0, topk-1, device=device)
            indices[t, h, :len(i_i)] = i_i
    # form input
    quant_scales = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    quant_scales = quant_scales.view(1, 1, 4)
    quant_scales = quant_scales.repeat_interleave(SKV, dim=0)
    quant_scales = quant_scales.repeat_interleave(HKV, dim=1)
    k_latent_fp8 = kv[..., :DV].to(torch.float8_e4m3fn).contiguous().view(SKV, HKV, DV)
    k_pe = kv[..., DV:].to(torch.bfloat16).contiguous().view(SKV, HKV, DQK - DV)
    k_cache_bytes = torch.cat(
        [k_latent_fp8.view(torch.uint8),
         quant_scales.view(torch.uint8),
         k_pe.view(torch.uint8)],
        dim=-1).contiguous()

    tl_out, tl_debug_out = sparse_mla_fwd_interface(
        q, k_cache_bytes, indices, threads=threads, verbose=True)
    # torch.testing.assert_close(tl_debug_out.view(SKV, HKV, DV).to(torch.float32), k_latent_fp8.view(SKV, HKV, DV).to(torch.float32), rtol=1e-2, atol=1e-2)
    tl_out_2, tl_debug_out_2 = sparse_mla_fwd_interface(q, k_cache_bytes, indices, threads=threads)
    # print(tl_debug_out)
    # print(k_latent_fp8)
    if check_correctness:
        # otherwise may cause out of memory
        torch.testing.assert_close(tl_out_2, tl_out, rtol=1e-7, atol=1e-7)
        k_scales = quant_scales.repeat_interleave(128, dim=-1)
        k_latent_fp32 = k_latent_fp8.to(torch.float32) * k_scales
        k_latent_fp32[k_latent_fp32 != k_latent_fp32] = 0.0
        k_latent_bf16 = k_latent_fp32.to(torch.bfloat16)
        kv_ref = torch.cat([k_latent_bf16, k_pe], dim=-1).contiguous()
        ref_out, ref_debug = ref_sparse_mla_fwd_interface(q, kv_ref, indices)
        # torch.testing.assert_close(ref_debug.view(-1).to(torch.float32), tl_debug_out.view(-1).to(torch.float32), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
        print("assert_tensors_similar passed")

    def fn():
        return sparse_mla_fwd_interface(q, k_cache_bytes, indices, threads=threads)

    if perf_test:
        from tilelang.profiler import do_bench

        ms = do_bench(
            fn,
            _n_warmup=5,
            _n_repeat=10,
        )
        print(f"Average time: {ms:.3f} ms")
        print("fwd io bandwidth = ", (S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
        print("fwd tflops = ", (S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)
        # IO bandwidth calculation (bytes transferred)
        # Q input:  S * H * DQK * 2 (bf16)
        # KV input:  S * HKV * topk * DQK * 2 (bf16, read D once + D_tail once)
        # Indices:  S * HKV * topk * 4 (int32)
        # Output:  S * H * DV * 2 (bf16)
        io_bytes = (S * H * DQK * 2 + S * HKV * topk * 656 + S * HKV * topk * 4 + S * H * DV * 2)
        total_flops = S * (DQK + DV) * topk * 2 * H
        bandwidth_tbps = io_bytes / (ms * 1e-3) / 1e12
        tflops = total_flops / ms * 1e-9
        print(f"[PERF] case=sparse_mla_fwd_sglang_v1 device={device} "
              f"params= S={S},SKV={SKV},H={H},HKV={HKV},DQK={DQK},DV={DV},"
              f"topk={topk}")
        print(f"[PERF] avg_time_ms={ms:.3f} bandwidth_TBps={bandwidth_tbps:.6f} "
              f"tflops={tflops:.6f}")


if __name__ == "__main__":
    test_sparse_mla_fwd(
        B=1,
        S=196,  #1024,
        SKV=8192,  #1024,
        H=128,  #64,
        HKV=1,
        DQK=576,
        DV=512,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=True,
        perf_test=True,
        threads=512,
    )

