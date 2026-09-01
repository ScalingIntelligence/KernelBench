import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()
tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

FP8 = "float8_e4m3"
FP32 = "float32"


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither MUSA nor CUDA is available")


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def fp8_index_kernel(h: int, d: int):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 128
    blk_n2 = 128

    @T.prim_func
    def fp8_index_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        o: T.Tensor[(b, m, n), FP32],
    ) -> None:
        with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
            q_smem = T.alloc_shared((h, d), FP8)
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            q_s_frag = T.alloc_fragment(h, FP32)
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                k_smem = T.alloc_shared((blk_n2, d), FP8)
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                k_s_frag = T.alloc_fragment(blk_n2, FP32)
                T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                logits = T.alloc_fragment((blk_n2, h), FP32)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n2, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)

                for i3_n in T.Parallel(blk_n2):
                    logits_sum[i3_n] *= k_s_frag[i3_n]

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return fp8_index_kernel_


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    return fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)


def fp8_index_torch(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    q_f = q.float()
    k_f = k.float()
    logits = torch.einsum("bnd,bmhd->bmnh", k_f, q_f)
    logits = torch.relu(logits) * q_s.unsqueeze(2)
    out = logits.sum(dim=-1) * k_s.unsqueeze(1)
    return out


def test_fp8_index(
    B: int = 64,
    M: int = 256,
    H: int = 32,
    D: int = 32,
    N: int = 1024,
    check_correctness: bool = True,
):
    torch.random.manual_seed(0)
    device = get_test_device()

    q_fp32 = torch.randn(B, M, H, D, device=device, dtype=torch.float32)
    k_fp32 = torch.randn(B, N, D, device=device, dtype=torch.float32)
    q = q_fp32.to(torch.float8_e4m3fn).contiguous()
    k = k_fp32.to(torch.float8_e4m3fn).contiguous()
    q_s = torch.rand(B, M, H, device=device, dtype=torch.float32).contiguous()
    k_s = torch.rand(B, N, device=device, dtype=torch.float32).contiguous()

    o = fp8_index(q, q_s, k, k_s)

    if check_correctness:
        o_ref = fp8_index_torch(q, q_s, k, k_s)
        torch.testing.assert_close(
            o.to(torch.float32), o_ref.to(torch.float32), rtol=1.25e-2, atol=1.25e-2)
        print("assert_tensors_similar passed")

    def fn():
        return fp8_index(q, q_s, k, k_s)

    from tilelang.profiler import do_bench

    ms = do_bench(fn, rep=100, warmup=250)
    io_bytes = (q.numel() * 1 + k.numel() * 1 + q_s.numel() * 4 + k_s.numel() * 4 + o.numel() * 4)
    total_flops = B * M * N * H * D * 2
    bandwidth_tbps = io_bytes / (ms * 1e-3) / 1e12
    tflops = total_flops / ms * 1e-9
    print(f"[PERF] case=fp8_index device={device} "
          f"params=B={B},M={M},N={N},H={H},D={D}")
    print(f"[PERF] avg_time_ms={ms:.3f} bandwidth_TBps={bandwidth_tbps:.6f} "
          f"tflops={tflops:.6f}")


if __name__ == "__main__":
    test_fp8_index(
        B=64,
        M=640,
        H=32,
        D=32,
        N=2560,
        check_correctness=True,
    )
