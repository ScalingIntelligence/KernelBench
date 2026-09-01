# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from tvm import tir

tilelang.disable_cache()


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
    compile_flags=[
        # "-Od3",
        "-fmusa-flush-denormals-to-zero",
        "-mllvm",
        "-misched=mtgpu-max-ilp",
        "-mllvm",
        "-mtgpu-if-convert=1",
        "-mllvm",
        "-mtgpu-tiny-offset-hint=1",
        "-mllvm",
        "-mtgpu-enable-postra-sched=0",
        "-mllvm",
        "-misched-recompute-slotindex=1",
        "-mllvm",
        "-mtgpu-combine-instr-with-burst=1",
        "-mllvm",
        "-mtgpu-combine-fop-instr=1",
        "-fno-signed-zeros",
        "-fno-strict-aliasing",
        "-mllvm",
        "-mtgpu-load-cluster-mutation=1",
        "-mllvm",
        "--num-dwords-of-load-in-mutation=64",
    ],
)
def sparse_attention_fwd_kernel_v1(
    num_heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=0,
    threads=512,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
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
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 64)
    if padded_H != H:
        assert kv_group == 1
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim
    L = block_I // 8

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(
                seq_len * REPLICATE_H, kv_group, threads=threads) as (
                    bx,
                    by,
                ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            acc_s_cast = T.alloc_fragment([H_per_block, BI], dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            bar_q = T.alloc_barrier(arrive_count=threads)
            bar_k_ready = T.alloc_barrier(arrive_count=threads)
            bar_p_ready = T.alloc_barrier(arrive_count=threads)
            bar_v_ready = T.alloc_barrier(arrive_count=threads)
            q_robust_desc = T.make_robust_desc(
                T.address_of(Q[0, 0, 0]), (seq_len * num_heads * (dim + tail_dim)) * 2)
            kv_robust_desc = T.make_robust_desc(
                T.address_of(KV[0, 0, 0]),
                (seq_len_kv * kv_group * (dim + tail_dim)) * 2,
            )

            T.sync_threads()
            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            mask = T.alloc_fragment([BI], "bool")
            mask_local = T.alloc_local([1], "bool")
            kperm_indices_local = T.alloc_local([1], "int32")
            kperm_mask_local = T.alloc_local([1], "bool")
            indices_local = T.alloc_local([1], indices_dtype)

            g_i = by
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(
                Q[s_i, H0:H1, :D],
                Q_shared,
                force_async_copy=True,
                src_robust_desc=q_robust_desc,
            )
            T.copy(
                Q[s_i, H0:H1, D:],
                Q_tail_shared,
                force_async_copy=True,
                src_robust_desc=q_robust_desc,
            )

            tir.call_extern("void", "__musa_memcpy_g2s_commit_group")
            tir.call_extern("void", "__musa_memcpy_g2s_wait_group", 0)
            T.barrier_arrive(bar_q)
            T.barrier_wait(bar_q, 0)
            tid = T.get_thread_binding()
            ldg_tx = tid % 8
            ldg_ty = tid // 8

            for i_i in range(NI):
                indices_local[0] = Indices[s_i, g_i, i_i * block_I + ldg_ty]
                mask_local[0] = indices_local[0] >= 0 and indices_local[0] < seq_len_kv
                indices_local[0] = T.if_then_else(
                    mask_local[0],
                    indices_local[0],
                    (seq_len_kv * kv_group * (dim + tail_dim)) * 2 + 1,
                )
                kperm_indices_local[0] = Indices[
                    s_i,
                    g_i,
                    i_i * block_I + ((ldg_ty) % 8) * (block_I // 8) + (ldg_ty) // 8,
                ]
                kperm_mask_local[0] = (
                    kperm_indices_local[0] >= 0 and kperm_indices_local[0] < seq_len_kv)
                kperm_indices_local[0] = T.if_then_else(
                    kperm_mask_local[0],
                    kperm_indices_local[0],
                    (seq_len_kv * kv_group * (dim + tail_dim)) * 2 + 1,
                )

                for bi_i in T.Parallel(BI):
                    mask[bi_i] = (
                        Indices[s_i, g_i, i_i * BI + bi_i % 8 * 8 + bi_i // 8] >= 0 and
                        Indices[s_i, g_i, i_i * BI + bi_i % 8 * 8 + bi_i // 8] < seq_len_kv)

                T.annotate_layout(
                    {
                        KV_shared:
                            tilelang.layout.make_sqmma_swizzled_layout(KV_shared, k_major=True)
                    },
                    allow_reannotation=True,
                )

                for u in T.unroll(8):
                    for v in T.vectorized(8):
                        T.copy(
                            KV[
                                kperm_indices_local[0],
                                g_i,
                                64 * u + ldg_tx * 8 + v,
                            ],
                            KV_shared[ldg_ty, 64 * u + ldg_tx * 8 + v],
                            force_async_copy=True,
                            src_robust_desc=kv_robust_desc,
                        )
                tir.call_extern("void", "__musa_memcpy_g2s_commit_group")

                for v in T.vectorized(8):
                    T.copy(
                        KV[
                            kperm_indices_local[0],
                            g_i,
                            D + ldg_tx * 8 + v,
                        ],
                        K_tail_shared[ldg_ty, ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=kv_robust_desc,
                    )

                tir.call_extern("void", "__musa_memcpy_g2s_commit_group")
                tir.call_extern("void", "__musa_memcpy_g2s_wait_group", 1)
                tir.call_extern("void", "__musa_memcpy_g2s_wait_group", 0)
                T.barrier_arrive(bar_k_ready)

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -(2**30))

                T.barrier_wait(bar_k_ready, (i_i & 1))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.wait_wgmma(0)

                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.wait_wgmma(0)

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, acc_s_cast)
                for i, t in T.Parallel(H_per_block, 8):
                    base = t * L
                    for l in T.vectorized(L):
                        S_shared[i, base + l] = acc_s_cast[i, l * 8 + t]

                T.annotate_layout(
                    {
                        KV_shared:
                            tilelang.layout.make_sqmma_swizzled_layout(
                                KV_shared, continuity=64, k_major=False)
                    },
                    allow_reannotation=True,
                )
                for u in T.unroll(8):
                    for v in T.vectorized(8):
                        T.copy(
                            KV[
                                indices_local[0],
                                g_i,
                                64 * u + ldg_tx * 8 + v,
                            ],
                            KV_shared[ldg_ty, 64 * u + ldg_tx * 8 + v],
                            force_async_copy=True,
                            src_robust_desc=kv_robust_desc,
                        )
                tir.call_extern("void", "__musa_memcpy_g2s_commit_group")
                tir.call_extern("void", "__musa_memcpy_g2s_wait_group", 0)
                tir.call_extern("void", "__musa_lma_wait")
                T.barrier_arrive(bar_v_ready)
                T.barrier_wait(bar_v_ready, (i_i & 1))
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            sumexp_inv = T.alloc_fragment([H_per_block], accum_dtype)
            for h_i in T.Parallel(H_per_block):
                sumexp_inv[h_i] = 1 / sumexp[h_i]
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] *= sumexp_inv[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[s_i, H0:H1, :])
            T.copy(sumexp, Lse[s_i, H0:H1])

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

    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    _, _, topk = indices.shape
    assert indices.shape == (seq_len, kv_group, topk)

    kernel = sparse_attention_fwd_kernel_v1(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_causal=is_casual,
        threads=threads,
    )
    if verbose:
        kernel.show_source()

    out = kernel(q, kv, indices)
    return out


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, d_v=512):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(0, 1)
    sq, h, dim_q = q.shape
    sk, g, _ = kv.shape

    dim = d_v
    k = kv
    v = kv[..., :dim]

    _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(
        0, sq, dtype=torch.int32, device=q.device).view(-1, 1) >= torch.arange(
            1 - 1, sk * 1, 1, dtype=torch.int32, device=q.device).view(1, -1)

    indices_clamped = torch.where(indices < 0, sk, indices)
    mask = q.new_zeros(g_index, sq, sk + 1, dtype=torch.bool).scatter(2, indices_clamped.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, sq, sk)
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
    return o.to(torch.bfloat16)


def test_sparse_mla_fwd_v1(
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
    q = torch.randn((S, H, DQK), dtype=dtype, device=device).requires_grad_(True)
    kv = torch.randn((SKV, HKV, DQK), dtype=dtype, device=device).requires_grad_(True)

    indices = torch.full((S, HKV, topk), -1, dtype=torch.int32, device=device)
    for t in range(S):
        for h in range(HKV):
            i_i = torch.randperm(max(1, t), device=device)[:topk]
            indices[t, h, :len(i_i)] = i_i

    tl_out, _ = sparse_mla_fwd_interface(
        q,
        kv,
        indices,
        d_v=DV,
        threads=threads,
        verbose=False,
    )

    if check_correctness:
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)
        torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
        print("assert_tensors_similar passed")

    def fn():
        return sparse_mla_fwd_interface(
            q,
            kv,
            indices,
            d_v=DV,
            threads=threads,
        )

    if perf_test:
        from tilelang.profiler import do_bench

        ms = do_bench(
            fn,
            rep=10,
            warmup=2,
        )
        print(f"Average time: {ms:.3f} ms")
        print("fwd io bandwidth = ", (S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
        print("fwd tflops = ", (S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)
        io_bytes = (
            S * H * DQK * 2 + S * HKV * topk * DQK * 2 + S * HKV * topk * 4 + S * H * DV * 2)
        total_flops = S * (DQK + DV) * topk * 2 * H
        bandwidth_tbps = io_bytes / (ms * 1e-3) / 1e12
        tflops = total_flops / ms * 1e-9
        print(f"[PERF] case=sparse_mla_fwd_pipelined_v1 device={device} "
              f"params= S={S},SKV={SKV},H={H},HKV={HKV},DQK={DQK},DV={DV},"
              f"topk={topk}")
        print(f"[PERF] avg_time_ms={ms:.3f} bandwidth_TBps={bandwidth_tbps:.6f} "
              f"tflops={tflops:.6f}")


if __name__ == "__main__":
    test_sparse_mla_fwd_v1(
        S=896,
        SKV=4096,
        H=128,
        HKV=1,
        DQK=576,
        DV=512,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=True,
        perf_test=True,
        threads=512,
    )
