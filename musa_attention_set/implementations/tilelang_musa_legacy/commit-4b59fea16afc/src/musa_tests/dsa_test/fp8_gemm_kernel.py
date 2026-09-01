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
BF16 = "bfloat16"
FP32 = "float32"


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither MUSA nor CUDA is available")


@tilelang.jit(pass_configs=pass_configs)
def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype="float32"):
    assert out_dtype in [BF16, "float32"]

    M = T.symbolic("M")
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), FP32],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), FP32],
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
                    bx,
                    by,
                ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            Scale_C_shared = T.alloc_shared((block_M), FP32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                Scale_B = scales_b[bx * block_N // group_size, k]
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp8_gemm_kernel_


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor,
             b_s: torch.Tensor) -> torch.Tensor:
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous")
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.bfloat16)
    kernel = fp8_gemm_kernel(N, K)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c


def fp8_gemm_torch(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    K = a.size(-1)
    a2d = a.view(-1, K).float()
    a_s2d = a_s.view(-1, a_s.shape[-1]).float()
    b2d = b.float()
    N = b2d.shape[0]

    out = torch.zeros((a2d.shape[0], N), device=a.device, dtype=torch.float32)
    k_groups = (K + group_size - 1) // group_size

    for kg in range(k_groups):
        k0 = kg * group_size
        k1 = min((kg + 1) * group_size, K)

        a_blk = a2d[:, k0:k1]
        b_blk = b2d[:, k0:k1]

        sa = a_s2d[:, kg].unsqueeze(1)
        sb_group = b_s[:, kg]
        sb = sb_group.repeat_interleave(group_size)[:N].unsqueeze(0)

        out += (a_blk @ b_blk.t()) * (sa * sb)

    return out.view(*a.shape[:-1], N)


def test_fp8_gemm(
    M: int = 512,
    N: int = 1024,
    K: int = 1024,
    check_correctness: bool = True,
):
    torch.random.manual_seed(0)
    device = get_test_device()

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)
    a = a_fp32.to(torch.float8_e4m3fn).contiguous()
    b = b_fp32.to(torch.float8_e4m3fn).contiguous()

    group_size = 128
    a_s = torch.rand(M, (K + group_size - 1) // group_size, device=device, dtype=torch.float32)
    b_s = torch.rand(
        (N + group_size - 1) // group_size, (K + group_size - 1) // group_size,
        device=device,
        dtype=torch.float32)

    c = fp8_gemm(a, a_s, b, b_s)

    if check_correctness:
        c_ref = fp8_gemm_torch(a, a_s, b, b_s, group_size=group_size)
        torch.testing.assert_close(c.float(), c_ref.float(), rtol=2e-2, atol=2e-2)
        print("assert_tensors_similar passed")

    def fn():
        return fp8_gemm(a, a_s, b, b_s)

    from tilelang.profiler import do_bench

    ms = do_bench(fn, rep=100, warmup=250)
    io_bytes = (a.numel() * 1 + b.numel() * 1 + a_s.numel() * 4 + b_s.numel() * 4 + c.numel() * 2)
    total_flops = 2 * M * N * K
    bandwidth_tbps = io_bytes / (ms * 1e-3) / 1e12
    tflops = total_flops / ms * 1e-9
    print(f"[PERF] case=fp8_gemm device={device} "
          f"params=M={M},N={N},K={K},group_size={group_size}")
    print(f"[PERF] avg_time_ms={ms:.3f} bandwidth_TBps={bandwidth_tbps:.6f} "
          f"tflops={tflops:.6f}")


if __name__ == "__main__":
    test_fp8_gemm(
        M=4096,
        N=4096,
        K=4096,
        check_correctness=True,
    )
