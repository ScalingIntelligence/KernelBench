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
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER:
            True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:
            True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:
            True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: 
            True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: 
            True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST:
            True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: 
            True,
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
def sparse_attention_fwd_kernel_v1(
    num_heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    block_I=64,
    threads=512,
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
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 64)
    if padded_H != H:
        assert kv_group == 1
    kv_latent_dtype = "float8_e4m3"
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim
    L = block_I // 8 
    dim_bytes = 656
    dv_slice_num = 8
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
            Q: T.Tensor([seq_len, num_heads, dim + tail_dim], dtype),  # type: ignore
            KV: T.Tensor([seq_len_kv, kv_group, dim_bytes], kv_latent_dtype),  # type: ignore
            K_pe: T.Tensor([seq_len_kv, kv_group, dim_bytes//2], dtype),  # type: ignore
            Quant_scales: T.Tensor([seq_len_kv, kv_group, dim_bytes//4], T.float32),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            debug_S: T.Tensor([num_heads, seq_len, topk], T.float32),  # type: ignore
    ):
        with T.Kernel(
                seq_len, num_heads//H_per_block ,kv_group, threads=threads) as (
                    bx,
                    by,
                    bz,
                ):
            Q_shared = T.alloc_shared([dv_slice_num, H_per_block, D//dv_slice_num], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([dv_slice_num, BI, D//dv_slice_num], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            indices_local = T.alloc_local([1], "int32")
            mask_local = T.alloc_local([1], "bool")
            kperm_indices_local = T.alloc_local([1], "int32")
            kperm_mask_local = T.alloc_local([1], "bool")
            
            acc_o_0 = T.alloc_fragment([H_per_block, D//dv_slice_num], accum_dtype)
            acc_o_1 = T.alloc_fragment([H_per_block, D//dv_slice_num], accum_dtype)
            acc_o_2 = T.alloc_fragment([H_per_block, D//dv_slice_num], accum_dtype)
            acc_o_3 = T.alloc_fragment([H_per_block, D//dv_slice_num], accum_dtype)
            acc_o_4 = T.alloc_fragment([H_per_block, D//dv_slice_num], accum_dtype)
            acc_o_5 = T.alloc_fragment([H_per_block, D//dv_slice_num], accum_dtype)
            acc_o_6 = T.alloc_fragment([H_per_block, D//dv_slice_num], accum_dtype)
            acc_o_7 = T.alloc_fragment([H_per_block, D//dv_slice_num], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            acc_s_cast = T.alloc_fragment([H_per_block, BI], dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            quant_local = T.alloc_local([4], T.float32)
            KV_reg_FP32 = T.alloc_local([8], T.float32)
            KV_reg_FP8 = T.alloc_local([8], kv_latent_dtype)
            KV_reg_FP16 = T.alloc_local([8], T.float16)
            KV_reg_BF16 = T.alloc_local([8], T.bfloat16)
            T.fill(acc_o_0, 0)
            T.fill(acc_o_1, 0)
            T.fill(acc_o_2, 0)
            T.fill(acc_o_3, 0)
            T.fill(acc_o_4, 0)
            T.fill(acc_o_5, 0)
            T.fill(acc_o_6, 0)
            T.fill(acc_o_7, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -2**30 )  
            q_robust_desc = T.make_robust_desc(T.address_of(Q[0,0,0]), (seq_len * num_heads * (dim + tail_dim)) * 2)
            kv_robust_desc = T.make_robust_desc(T.address_of(KV[0,0,0]), (seq_len_kv * kv_group * (dim_bytes)))
            tid = T.get_thread_binding()


            for i in T.unroll(dv_slice_num):
                T.copy(Q[bx, by*H_per_block:(by+1)*H_per_block, (i * (D // dv_slice_num)):((i + 1) * (D // dv_slice_num))], Q_shared[i,:,:])
            T.copy(Q[bx, by*H_per_block:(by+1)*H_per_block, D:], Q_tail_shared)
            T.sync_threads()
            tx = tid * 8 % 64
            ty = tid * 8 // 64
            ty_perm = (ty % 8) * (BI // 8) + ty // 8
            for i_i in T.Pipelined(T.ceildiv(topk, BI), num_stages=0):  
                indices_local[0] = Indices[bx, bz, i_i * BI + ty]
                mask_local[0] = indices_local[0]>=0
                indices_local[0] = T.if_then_else(mask_local[0], indices_local[0], 0)

                kperm_indices_local[0] = Indices[bx, bz, i_i * BI + ty_perm]
                kperm_mask_local[0] = kperm_indices_local[0]>=0
                kperm_indices_local[0] = T.if_then_else(kperm_mask_local[0], kperm_indices_local[0], 0)
                # quant_local[0] = quant_scales[kperm_indices_local[0] ,tx//2]
                T.copy(Quant_scales[kperm_indices_local[0], bz, D//4:D//4+4],quant_local)
                for i in T.unroll(dv_slice_num):
                    T.annotate_layout(
                        {
                            KV_shared[i,:,:]:
                                tilelang.layout.make_sqmma_swizzled_layout(KV_shared[i,:,:], k_major=True)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                if kperm_mask_local[0]:
                    for a_i in T.unroll(8):
                        for b_i in T.vectorized(8):
                            KV_reg_FP8[b_i]=KV[kperm_indices_local[0], bz, a_i*64+tx+b_i]
                        for b_i in T.vectorized(8):
                            KV_reg_FP16[b_i]=KV_reg_FP8[b_i]
                        for b_i in T.vectorized(8):
                            KV_reg_FP32[b_i]=KV_reg_FP16[b_i]
                        for b_i in T.vectorized(8):
                            KV_reg_FP32[b_i]=KV_reg_FP32[b_i]*quant_local[a_i//2]
                        for b_i in T.vectorized(8):
                            KV_reg_BF16[b_i]=KV_reg_FP32[b_i]
                        for b_i in T.vectorized(8):
                            KV_shared[a_i, ty, tx+b_i] = KV_reg_BF16[b_i]
                    for b_i in T.vectorized(8):
                        KV_reg_BF16[b_i] = K_pe[kperm_indices_local[0], bz, tx+b_i+264]
                    for b_i in T.vectorized(8):
                        K_tail_shared[ty, tx+b_i] = KV_reg_BF16[b_i]
                else:
                    for a_i in T.unroll(8):
                        for b_i in T.vectorized(8):
                            KV_shared[a_i, ty, tx+b_i] = 0
                    for b_i in T.vectorized(8):
                        K_tail_shared[ty, tx+b_i] = 0

                T.sync_threads()
                T.gemm(
                    Q_shared[0,:,:],
                    KV_shared[0,:,:],
                    acc_s,
                    clear_accum=True,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i in T.unroll(dv_slice_num-1):
                    T.gemm(
                        Q_shared[i+1,:,:],
                        KV_shared[i+1,:,:],
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for h_i, n_ii in T.Parallel(H_per_block, BI):
                    acc_s[h_i, n_ii] = T.if_then_else(Indices[bx, bz, i_i * BI + n_ii%8 * 8 + n_ii//8]>=0, acc_s[h_i, n_ii], -T.infinity(acc_s.dtype))

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i]- m_i[h_i])*sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i]*sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D//dv_slice_num):
                    acc_o_0[h_i, d_i] = acc_o_0[h_i, d_i] * alpha[h_i]
                    acc_o_1[h_i, d_i] = acc_o_1[h_i, d_i] * alpha[h_i]
                    acc_o_2[h_i, d_i] = acc_o_2[h_i, d_i] * alpha[h_i]
                    acc_o_3[h_i, d_i] = acc_o_3[h_i, d_i] * alpha[h_i]
                    acc_o_4[h_i, d_i] = acc_o_4[h_i, d_i] * alpha[h_i]
                    acc_o_5[h_i, d_i] = acc_o_5[h_i, d_i] * alpha[h_i]
                    acc_o_6[h_i, d_i] = acc_o_6[h_i, d_i] * alpha[h_i]
                    acc_o_7[h_i, d_i] = acc_o_7[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, acc_s_cast)
                for i, t in T.Parallel(H_per_block, 8):
                    base = t * L
                    for l in T.vectorized(L):
                        S_shared[i, base + l] = acc_s_cast[i, l * 8 + t]
                for i in T.unroll(dv_slice_num):
                    T.annotate_layout(
                        {
                            KV_shared[i,:,:]:
                                tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared[i,:,:], continuity=64, k_major=False)
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                T.copy(Quant_scales[indices_local[0], bz, D//4:D//4+4],quant_local)
                if mask_local[0]:
                    for a_i in T.unroll(8):
                        for b_i in T.vectorized(8):
                            KV_reg_FP8[b_i]=KV[indices_local[0], bz, a_i*64+tx+b_i]
                        for b_i in T.vectorized(8):
                            KV_reg_FP16[b_i]=KV_reg_FP8[b_i]
                        for b_i in T.vectorized(8):
                            KV_reg_FP32[b_i]=KV_reg_FP16[b_i]
                        for b_i in T.vectorized(8):
                            KV_reg_FP32[b_i]=KV_reg_FP32[b_i]*quant_local[a_i//2]
                        for b_i in T.vectorized(8):
                            KV_reg_BF16[b_i]=KV_reg_FP32[b_i]
                        for b_i in T.vectorized(8):
                            KV_shared[a_i, ty,tx+b_i] = KV_reg_BF16[b_i]
                else:
                    for a_i in T.unroll(8):
                        for b_i in T.vectorized(8):
                            KV_shared[a_i, ty,tx+b_i] = 0 
                T.sync_threads()
                # for i in T.unroll(dv_slice_num):
                #     T.gemm(S_shared, KV_shared[i,:,:], acc_o[i,:,:], policy=T.GemmWarpPolicy.FullRow)
                T.gemm(S_shared, KV_shared[0,:,:], acc_o_0, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(S_shared, KV_shared[1,:,:], acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(S_shared, KV_shared[2,:,:], acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(S_shared, KV_shared[3,:,:], acc_o_3, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(S_shared, KV_shared[4,:,:], acc_o_4, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(S_shared, KV_shared[5,:,:], acc_o_5, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(S_shared, KV_shared[6,:,:], acc_o_6, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(S_shared, KV_shared[7,:,:], acc_o_7, policy=T.GemmWarpPolicy.FullRow)
            # Rescale
            for h_i in T.Parallel(H_per_block):
                sumexp_i[h_i] = 1.0 / (sumexp[h_i]+1e-7)
            for h_i, d_i in T.Parallel(H_per_block, D//dv_slice_num):
                # for s_i in T.unroll(dv_slice_num):
                #     acc_o[s_i, h_i, d_i] *= sumexp_i[h_i]
                acc_o_0[h_i, d_i] *= sumexp_i[h_i]
                acc_o_1[h_i, d_i] *= sumexp_i[h_i]
                acc_o_2[h_i, d_i] *= sumexp_i[h_i]
                acc_o_3[h_i, d_i] *= sumexp_i[h_i]
                acc_o_4[h_i, d_i] *= sumexp_i[h_i]
                acc_o_5[h_i, d_i] *= sumexp_i[h_i]
                acc_o_6[h_i, d_i] *= sumexp_i[h_i]
                acc_o_7[h_i, d_i] *= sumexp_i[h_i]
            T.copy(acc_o_0, Output[bx, by*H_per_block:(by+1)*H_per_block, 64*0:64*1])
            T.copy(acc_o_1, Output[bx, by*H_per_block:(by+1)*H_per_block, 64*1:64*2])
            T.copy(acc_o_2, Output[bx, by*H_per_block:(by+1)*H_per_block, 64*2:64*3])
            T.copy(acc_o_3, Output[bx, by*H_per_block:(by+1)*H_per_block, 64*3:64*4])
            T.copy(acc_o_4, Output[bx, by*H_per_block:(by+1)*H_per_block, 64*4:64*5])
            T.copy(acc_o_5, Output[bx, by*H_per_block:(by+1)*H_per_block, 64*5:64*6])
            T.copy(acc_o_6, Output[bx, by*H_per_block:(by+1)*H_per_block, 64*6:64*7])
            T.copy(acc_o_7, Output[bx, by*H_per_block:(by+1)*H_per_block, 64*7:64*8])      


            
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

    threads=512
    kernel = sparse_attention_fwd_kernel_v1(
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
    kv_latent_f8  = kv.view(torch.float8_e4m3fn)
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
    mask = q.new_zeros(
        g_index, sq, sk + 1, dtype=torch.bool).scatter(2, indices_clamped.long(), 1)
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
    S=1,
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
    total_q = B*S
    q = torch.randn((B, S, H, DQK), dtype=dtype, device=device)
    kv = torch.randn((SKV, HKV, DQK), dtype=dtype, device=device)
    indices = torch.full((B, S, HKV, topk), -1, dtype=torch.int32, device=device)
    for i in range(B):
        for s in range(S):
            for h in range(HKV):
                cur_abs_indices = torch.randperm(SKV, device="cpu")[:topk]
                if len(cur_abs_indices) < topk:
                    cur_abs_indices = torch.cat([cur_abs_indices, torch.full((topk-len(cur_abs_indices),), -1, device="cpu")])
                indices[i, s, h, :] = cur_abs_indices
    q = q.view(total_q, H, DQK)
    indices = indices.view(total_q, HKV ,topk)
    # form input
    quant_scales = torch.tensor([1.0,1.0,1.0,1.0], dtype=torch.float32, device=device)
    quant_scales = quant_scales.view(1, 1, 4)
    quant_scales = quant_scales.repeat_interleave(SKV,dim=0)
    quant_scales = quant_scales.repeat_interleave(HKV,dim=1)
    k_latent_fp8 = kv[..., :DV].to(torch.float8_e4m3fn).contiguous().view(SKV, HKV, DV)
    k_pe = kv[..., DV:].to(torch.bfloat16).contiguous().view(SKV, HKV, DQK-DV)
    k_cache_bytes = torch.cat([k_latent_fp8.view(torch.uint8), quant_scales.view(torch.uint8) , k_pe.view(torch.uint8)], dim=-1).contiguous()
    softmax_scale = 0.1352337788608801
    tl_out, tl_debug_out  = sparse_mla_fwd_interface(q, k_cache_bytes, indices, sm_scale=softmax_scale)


    if check_correctness:
        # otherwise may cause out of memory
        k_scales = quant_scales.repeat_interleave(128, dim=-1)
        k_latent_fp32 = k_latent_fp8.to(torch.float32) * k_scales
        k_latent_fp32[k_latent_fp32!=k_latent_fp32] = 0.0
        k_latent_bf16 = k_latent_fp32.to(torch.bfloat16)
        kv_ref = torch.cat([k_latent_bf16, k_pe],dim=-1).contiguous()
        ref_out,ref_debug = ref_sparse_mla_fwd_interface(q, kv_ref,indices,sm_scale=softmax_scale)
        # torch.testing.assert_close(ref_debug.view(-1).to(torch.float32), tl_debug_out.view(-1).to(torch.float32), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
        print("assert_tensors_similar passed")

    def fn():
        return sparse_mla_fwd_interface(q, k_cache_bytes, indices, sm_scale=softmax_scale)
    if perf_test:
        from tilelang.profiler import do_bench
        
        ms = do_bench(
            fn,
            rep=10,
            warmup=2,
        )
        print(f"Average time: {ms:.3f} ms")
        # IO bandwidth calculation (bytes transferred)
        # Q input:  S * H * DQK * 2 (bf16)
        # kcache bytes input:  S * HKV * topk * 656 
        # Indices:  S * HKV * topk * 4 (int32)
        # Output:  S * H * DV * 2 (bf16)
        io_bytes = (
            B * S * H * DQK * 2 +  B * S * HKV * topk * 656 +  B * S * HKV * topk * 4 +
            B * S * H * DV * 2)
        total_flops =  B * S * (DQK + DV) * topk * 2 * H
        bandwidth_tbps = io_bytes / (ms * 1e-3) / 1e12
        tflops = total_flops / ms * 1e-9
        print(f"[PERF] case=sparse_mla_fwd_sglang_v1 device={device} "
            f"params= S={S},SKV={SKV},H={H},HKV={HKV},DQK={DQK},DV={DV},"
            f"topk={topk}")
        print(f"[PERF] avg_time_ms={ms:.3f} bandwidth_TBps={bandwidth_tbps:.6f} "
            f"tflops={tflops:.6f}")


if __name__ == "__main__":
    test_sparse_mla_fwd(
        B=256,
        S=2,  #1024,
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

