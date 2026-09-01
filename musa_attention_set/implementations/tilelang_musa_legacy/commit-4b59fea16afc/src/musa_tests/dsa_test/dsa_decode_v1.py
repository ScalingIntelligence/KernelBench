import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse
from tilelang.profiler import do_bench
import math


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    },
    verbose = True,
    compile_flags=[
        "-fmusa-flush-denormals-to-zero",
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
        "-O2",
    ]
)
def mla_decode_tilelang(batch, h_q, h_kv, topk, max_seqlen_pad, dv, dpe, block_N, block_H, num_split, block_size, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = (dv + dpe) ** -0.5
    scale = float(softmax_scale * 1.44269504)  # log2(e)
    log2scale = float(softmax_scale * 1.44269504)
    dtype = T.bfloat16
    accum_dtype = T.float32
    L = block_N // 8 
    kv_group_num = h_q // h_kv
    VALID_BLOCK_H = min(block_H, kv_group_num)
    quant_scale_group = 128
    scale_group_num = dv // quant_scale_group
    kv_latent_dtype = "float8_e4m3"

    assert dv % quant_scale_group == 0, "dv must be a multiple of quant_scale_group"
    assert h_kv == 1, "h_kv must be 1"
    assert block_size >= block_N and block_size % block_N == 0, "block_size must be larger than block_N and a multiple of block_N"
    dim_bytes = 656

    @T.prim_func
    def dsa_decode(
        Q: T.Tensor([batch, s_q, h_q, dv+dpe], dtype), # type: ignore
        KV: T.Tensor([batch * max_seqlen_pad, dim_bytes], kv_latent_dtype), # type: ignore
        K_pe: T.Tensor([batch * max_seqlen_pad, dim_bytes//2], dtype), # type: ignore
        quant_scales: T.Tensor([batch * max_seqlen_pad, dim_bytes//4], T.float32), # type: ignore
        indices: T.Tensor([batch, s_q, topk], T.int32), # type: ignore
        Output: T.Tensor([batch, s_q, h_q, dv], dtype), # type: ignore
    ):
        with T.Kernel(batch, h_q // min(block_H, kv_group_num), s_q, threads=512) as (bz, by, bx):
            Q_shared = T.alloc_shared([block_H, dv], dtype)
            Q_tail_shared = T.alloc_shared([block_H, dpe], dtype)
            KV_shared = T.alloc_shared([block_N, dv], dtype)
            K_tail_shared = T.alloc_shared([block_N, dpe], dtype)
            indices_local = T.alloc_local([1], "int32")
            mask_local = T.alloc_local([1], "bool")
            kperm_indices_local = T.alloc_local([1], "int32")
            kperm_mask_local = T.alloc_local([1], "bool")
            acc_o = T.alloc_fragment([block_H, dv], accum_dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            sumexp = T.alloc_fragment([block_H], accum_dtype)
            sumexp_i = T.alloc_fragment([block_H], accum_dtype)
            alpha = T.alloc_fragment([block_H], accum_dtype)
            m_i = T.alloc_fragment([block_H], accum_dtype)
            m_i_prev = T.alloc_fragment([block_H], accum_dtype)
            quant_local = T.alloc_local([1], T.float32)
            KV_reg_FP32 = T.alloc_local([16], T.float32)
            KV_reg_FP8 = T.alloc_local([16], kv_latent_dtype)
            KV_reg_BF16 = T.alloc_local([16], T.bfloat16)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -2**30 )  # avoid -inf - inf to cause nan


            T.copy(Q[bz, bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, 0:dv], Q_shared)
            T.copy(Q[bz, bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, dv:], Q_tail_shared)
            T.sync_threads()
            tid = T.get_thread_binding()
            tx = tid % 8
            ty = tid // 8
            

            for i_i in T.Pipelined(T.ceildiv(topk, block_N), num_stages=0):
                # load layout 64*512 in 512 threads, per 64 dim  

                indices_local[0] = indices[bz, bx, i_i * block_N + ty]
                mask_local[0] = indices_local[0]>=0
                indices_local[0] = T.if_then_else(mask_local[0], indices_local[0], 0)

                kperm_indices_local[0] = indices[bz, bx, i_i * block_N + (ty % 8) * (block_N // 8) + ty // 8]
                kperm_mask_local[0] = kperm_indices_local[0]>=0
                kperm_indices_local[0] = T.if_then_else(kperm_mask_local[0], kperm_indices_local[0], 0)
                quant_local[0] = quant_scales[kperm_indices_local[0], dv//4+tx//2]
                T.annotate_layout(
                    {
                        KV_shared:
                            tilelang.layout.make_sqmma_swizzled_layout(KV_shared, k_major=True)
                    },
                    allow_reannotation=True,
                )
                for a_i in T.serial(4):
                    if kperm_mask_local[0]:
                        for b_i in T.vectorized(16):
                            KV_reg_FP8[b_i]=KV[kperm_indices_local[0], tx*64+a_i*16+b_i]
                    else:
                        for b_i in T.vectorized(16):
                            KV_reg_FP8[b_i]=0
                    for b_i in T.vectorized(16):
                        KV_reg_FP32[b_i]=KV_reg_FP8[b_i]
                    for b_i in T.vectorized(16):
                        KV_reg_FP32[b_i]=KV_reg_FP32[b_i]*quant_local[0]
                    for b_i in T.vectorized(16):
                        KV_reg_BF16[b_i]=KV_reg_FP32[b_i]
                    for b_i in T.vectorized(16):
                        KV_shared[ty, tx*64+a_i*16+b_i] = KV_reg_BF16[b_i]
                if kperm_mask_local[0]:
                    for b_i in T.vectorized(8):
                        KV_reg_BF16[b_i] = K_pe[kperm_indices_local[0], dv//2+scale_group_num*2+tx*8+b_i]
                        K_tail_shared[ty, tx*8+b_i] = KV_reg_BF16[b_i]
                else:
                    for b_i in T.vectorized(8):
                        K_tail_shared[ty, tx*8+b_i] = 0

                T.sync_threads()
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    clear_accum=True,
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
                for h_i, n_ii in T.Parallel(block_H, block_N):
                    acc_s[h_i, n_ii] = T.if_then_else(indices[bz, bx, i_i * block_N + n_ii%8 * 8 + n_ii//8]>=0, acc_s[h_i, n_ii], -2**30)

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=True)
                
                # for i in T.Parallel(block_H):
                #     m_i_prev[i] = T.if_then_else(m_i[i] < m_i_prev[i], 0, m_i_prev[i])
                # for i in T.Parallel(block_H):
                #     m_i[i] *= log2scale
                for h_i in T.Parallel(block_H):
                    if m_i[h_i]>-2**30:
                        alpha[h_i] = T.exp2((m_i_prev[h_i]- m_i[h_i])*log2scale)
                    else:
                        alpha[h_i] = 1.0
                for h_i, bi_i in T.Parallel(block_H, block_N):
                    if m_i[h_i]>-2**30:
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * log2scale - m_i[h_i]*log2scale)
                    else:
                        acc_s[h_i, bi_i] = 0.0
                T.reduce_sum(acc_s, sumexp_i, dim=1)  
                for h_i in T.Parallel(block_H):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(block_H, dv):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, acc_s_cast)
                for i, t in T.Parallel(block_H, 8):
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
                quant_local[0] = quant_scales[indices_local[0], dv//4 + tx//2]
                for a_i in T.serial(4):
                    if mask_local[0]:
                        for b_i in T.vectorized(16):
                            KV_reg_FP8[b_i]=KV[indices_local[0], tx*64+a_i*16+b_i]
                    else:
                        for b_i in T.vectorized(16):
                            KV_reg_FP8[b_i]=0
                    for b_i in T.vectorized(16):
                        KV_reg_FP32[b_i]=KV_reg_FP8[b_i]
                    for b_i in T.vectorized(16):
                        KV_reg_FP32[b_i]=KV_reg_FP32[b_i]*quant_local[0]
                    for b_i in T.vectorized(16):
                        KV_reg_BF16[b_i]=KV_reg_FP32[b_i]
                    for b_i in T.vectorized(16):
                        KV_shared[ty, tx*64+a_i*16+b_i] = KV_reg_BF16[b_i]
                T.sync_threads()
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i in T.Parallel(block_H):
                sumexp_i[h_i] = 1.0 / (sumexp[h_i]+1e-7)
            for h_i, d_i in T.Parallel(block_H, dv):
                acc_o[h_i, d_i] *= sumexp_i[h_i]
            # for h_i in T.Parallel(block_H):
            #     sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * log2scale

            T.copy(acc_o, Output[bz, bx,  by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, 0:dv])

    return dsa_decode




@torch.inference_mode()
def run_torch_sparse_mla(q, indices, kv_flat, k_pe_flat, quant_scales, h_q, h_kv, dv, dpe):
    """
    q: [B, Sq, H, D]
    indices: [B, Sq, TopK] (Physical absolute indices)
    kv_flat: [Total, DV] fp8
    k_pe_flat: [Total, DPE]
    quant_scales : [Total, scale_group_num]
    """
    B, Sq, H, D = q.shape
    topk = indices.shape[-1]

    q_nope = q[..., :dv].float()  # [B, Sq, H, dv]
    q_pe = q[..., dv:].float()    # [B, Sq, H, dpe]
    
    scale_group = 128
    group_num = dv // scale_group

    # indices: [B, Sq, TopK] -> reshape to [B*Sq*TopK]
    flat_indices = indices.reshape(-1)
    safe_indices = torch.where(flat_indices >= 0, flat_indices, 0)

    kv_flat = kv_float = kv_flat[..., :dv].float()

    # Gather kv: [B*Sq*TopK, dv/dpe]
    k_nope = torch.index_select(kv_float, 0, safe_indices.long())  # [B*Sq*TopK, dv]
    k_pe = torch.index_select(k_pe_flat.float(), 0, safe_indices.long())  # [B*Sq*TopK, dpe]
    k_scales = torch.index_select(quant_scales, 0, safe_indices)  # [B*Sq*TopK, group_num]
    
    # Reshape: [B, Sq, TopK, dv/dpe]
    k_nope = k_nope.view(B, Sq, topk, dv)
    k_pe = k_pe.view(B, Sq, topk, dpe)
    k_scales = k_scales.view(B, Sq, topk, group_num).float()
    k_scales = k_scales.repeat_interleave(scale_group, dim=-1)  # [B, Sq, TopK, dv]
    k_nope = k_nope * k_scales
    v = k_nope
    
    # Mask: [B, Sq, TopK] -> [B, Sq, 1, TopK] for broadcasting
    invalid_mask = (indices < 0)  # [B, Sq, TopK]
    
    # Compute Scores: einsum over dv/dpe dimension
    # q_nope/q_pe: [B, Sq, H, dv/dpe], k_nope/k_pe: [B, Sq, TopK, dv/dpe]
    # result: [B, Sq, H, TopK]
    attn_nope = torch.einsum("bsqd,bskd->bsqk", q_nope, k_nope)
    attn_pe = torch.einsum("bsqd,bskd->bsqk", q_pe, k_pe)
    
    scores = (attn_nope + attn_pe) * (D ** -0.5)
    scores = scores.masked_fill(invalid_mask.unsqueeze(2), float('-inf'))  # [B, Sq, H, TopK]
    
    # Softmax over TopK dimension
    lse = scores.logsumexp(dim=-1, keepdim=True)  # [B, Sq, H, 1]
    probs = torch.exp(scores - lse)  # [B, Sq, H, TopK]
    
    # Output: probs [B, Sq, H, TopK] @ v [B, Sq, TopK, dv] -> [B, Sq, H, dv]
    out = torch.einsum("bsqk,bskd->bsqd", probs, v)
    return out.to(q.dtype)


def flashmla(
    q,
    kcache_bytes,
    indices, 
    page_table,
    seqlen_kvs,
    head_dim_latent,
    tile_scheduler_metadata,
    num_splits,
    sm_scale,
    causal = False,
    is_fp8_kvcache = True,
):
    #  In FP8+sparse mode, each token's KV cache is 656 Bytes, structured as:
    #         - The shape of the tensor `k_cache` is (num_blocks*page_block_size*num_heads_k, head_dim), and num_heads_k must be 1.
    #         - First 512 bytes: The "quantized NoPE" part, containing 512 float8_e4m3 values.
    #         - Next 16 bytes: Scale factors, containing 4 float32 values. The first float32 is the scale for the first 128 float8_e4m3 values, the second for the next 128, and so on.
    #         - Last 128 bytes: The "RoPE" part, containing 64 bfloat16 values. This part is not quantized for accuracy.
    # kv_latent_f8 = kcache_bytes[..., :512].contiguous()     
    # scales = kcache_bytes[..., 512:528].contiguous()       
    # k_rope = kcache_bytes[..., 528:656].contiguous()  
    kv_latent_f8  = kcache_bytes.view(torch.float8_e4m3fn)
    k_rope = kcache_bytes.view(torch.bfloat16)
    scales = kcache_bytes.view(torch.float32)
    print(kv_latent_f8.shape, scales.shape, k_rope.shape)
    h_q = q.shape[1]
    BLOCK_N = 64
    BLOCK_H = min(64, h_q // h_kv)
    kernel = mla_decode_tilelang(b, h_q, h_kv, topk, max_seqlen_pad, dv, dpe, BLOCK_N, BLOCK_H, 1, block_size, sm_scale)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    out = profiler.func(
        q.view(-1, s_q, h_q, dv+dpe),
        kv_latent_f8,
        k_rope,
        scales,
        indices,
    )
    return out


    
    

def run_tilelang_mla(
        q, 
        block_table, 
        blocked_k, 
        max_seqlen_pad, 
        block_size, 
        b, 
        s_q, 
        cache_seqlens, 
        h_q, 
        h_kv, 
        topk, 
        d, 
        dv, 
        causal, 
        dtype):
    assert d > dv, "mla with rope dim should be larger than no rope dim"
    blocked_k_nope, blocked_k_pe = blocked_k[..., :dv].contiguous(), blocked_k[..., dv:].contiguous()
    dpe = d - dv
    blocked_k_nope = blocked_k_nope.view(-1, dv)
    blocked_k_pe = blocked_k_pe.view(-1,dpe)
    total_seqlen = blocked_k_nope.shape[0]


    BLOCK_N = 64
    BLOCK_H = min(64, h_q // h_kv)
    softmax_scale = d**-0.5
    def generate_topk_idx(
        batch: int,
        seqlen_q: int,
        topk: int,
        seqlen_kvs: torch.Tensor,
        is_all_indices_invalid: bool,
        page_table: torch.Tensor,
    ):
        page_size: int = 64
        block_table_cpu = page_table.cpu()
        abs_indices = torch.empty(batch, seqlen_q, topk, dtype=torch.int32, device="cpu")
        indices_in_kvcache = torch.empty(
            batch, seqlen_q, topk, dtype=torch.int32, device="cpu"
        )
        for i in range(batch):
            # Generate indices
            for j in range(seqlen_q):
                cur_abs_indices = torch.randperm(int(seqlen_kvs[i].item()), device="cpu")[:topk]
                # cur_abs_indices = torch.arange(int(seqlen_kvs[i].item()), device="cpu")[:topk]
                cur_blocked_indices = block_table_cpu[
                    i, cur_abs_indices // page_size
                ] * page_size + (cur_abs_indices % page_size)
                if len(cur_abs_indices) < topk:
                    pad_len = topk - len(cur_abs_indices)
                    cur_abs_indices = torch.cat(
                        [cur_abs_indices, torch.full((pad_len,), -1, device="cpu")]
                    )
                    cur_blocked_indices = torch.cat(
                        [cur_blocked_indices, torch.full((pad_len,), -1, device="cpu")]
                    )

                # Mask KV
                # perm = torch.randperm(topk, device="cpu")
                # cur_abs_indices = cur_abs_indices[perm]
                # cur_blocked_indices = cur_blocked_indices[perm]

                # Fill it with invalid indices if needed
                if is_all_indices_invalid:
                    cur_abs_indices.fill_(-1)
                    cur_blocked_indices.fill_(-1)

                abs_indices[i, j, :] = cur_abs_indices
                indices_in_kvcache[i, j, :] = cur_blocked_indices
        return indices_in_kvcache, abs_indices
    _, indices = generate_topk_idx(
        b,
        s_q,
        topk,
        cache_seqlens,
        is_all_indices_invalid=False,
        page_table=block_table,
    )  # b, q, topk
    print(indices.shape)
    print(indices)
    indices = indices.to(q.device)

    total_k = blocked_k_nope
    quant_scales = torch.ones((total_seqlen,4), dtype=torch.float32, device=q.device)
    kv_latent = blocked_k_nope.view(-1, dv).to(torch.float8_e4m3fn)
    kv_latent = kv_latent.view(torch.uint8)
    bytes_rope = blocked_k_pe.view(torch.uint8) 
    bytes_scales = quant_scales.view(torch.uint8) 
    k_cache_bytes = torch.cat([kv_latent, bytes_scales, bytes_rope], dim=-1).contiguous() 
    def flash_mla_tilelang():
        out = flashmla(
            q.view(-1, h_q, dv+dpe),
            k_cache_bytes,
            indices,
            block_table,
            cache_seqlens,
            dv,
            None,
            1,
            softmax_scale,
            False,
            True
        )
        return out.view([b, s_q, h_q, dv])

    out_flash = flash_mla_tilelang()
    t = do_bench(flash_mla_tilelang)
    out_ref = run_torch_sparse_mla(q, indices, blocked_k_nope.view(-1, dv).to(torch.float8_e4m3fn).to(dtype), blocked_k_pe.view(-1, dpe), quant_scales,h_q, h_kv, dv, dpe)
    torch.testing.assert_close(out_flash, out_ref, rtol=0.01, atol=0.01)
    print("All close")
    return out_flash, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--h_q", type=int, default=128, help="q heads number")
    parser.add_argument("--h_kv", type=int, default=1, help="kv heads number")
    parser.add_argument("--cache_seqlen", type=int, default=256, help="kv cache context length")
    parser.add_argument("--d", type=int, default=576, help="query/key head dim, d = dv + dpe")
    parser.add_argument("--dv", type=int, default=512, help="value head dim")
    parser.add_argument("--topk", type=int, default=2048, help="top-k attention")
    args = parser.parse_args()
    b, h_q, h_kv, cache_seqlen, d, dv, topk = args.batch, args.h_q, args.h_kv, args.cache_seqlen, args.d, args.dv, args.topk
    device = "musa"
    dtype = torch.bfloat16

    s_q = 2  # for decode, s_q = 1 or 2(mtp)
    block_size = 64
    # cache_seqlens = torch.tensor([cache_seqlen + 2 * i for i in range(b)], dtype=torch.int32, device=device)
    cache_seqlens = torch.full((b,), cache_seqlen, dtype=torch.int32, device=device)
    dpe = d - dv
    causal = True

    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen / 256) * 256

    total_flops = s_q * b * topk * h_q * d * 2

    q = torch.randn(b, s_q, h_q, d, dtype=dtype, device=device)
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32, device=device).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=dtype, device=device)
    out_flash, latency = run_tilelang_mla(
        q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, topk, d, dv, causal, dtype
    )

    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))