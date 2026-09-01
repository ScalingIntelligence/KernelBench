from typing import Optional, Tuple

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

BF16 = "bfloat16"
FP8 = "float8_e4m3"
FP32 = "float32"


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither MUSA nor CUDA is available")


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(N, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False):
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2
    blk_m = 32
    group_size = 128

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(
                T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
                    pid_m,
                    pid_n,
                ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), scale_dtype)
            s_local = T.alloc_fragment((blk_m,), scale_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = s_local[i]
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(x: torch.Tensor,
              block_size: int = 128,
              scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (x.size(-1) % block_size == 0
           ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    N = x.size(-1)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    return y, s


def act_quant_torch(
    x: torch.Tensor,
    block_size: int = 128,
    round_scale: bool = False,
    fp8_min: float = -448.0,
    fp8_max: float = 448.0,
):
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.shape[-1] % block_size == 0, "Last dim must be divisible by block_size"

    orig_shape = x.shape
    N = x.shape[-1]
    M = x.numel() // N
    x2d = x.view(M, N)

    n_groups = N // block_size
    x_blocks = x2d.view(M, n_groups, block_size).to(torch.float32)
    amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-4)

    inv_max = 1.0 / fp8_max
    if round_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax * inv_max)))
    else:
        scale = amax * inv_max
    scale = scale.to(torch.float32)

    y_blocks = torch.clamp(x_blocks / scale.unsqueeze(-1), fp8_min, fp8_max)
    y = y_blocks.to(torch.float8_e4m3fn).view(*orig_shape)
    s = scale.view(*orig_shape[:-1], n_groups)
    return y, s


def test_act_quant(
    batch_size: int = 512,
    hidden_dim: int = 8192,
    dtype=torch.bfloat16,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    check_correctness: bool = True,
):
    torch.random.manual_seed(0)
    device = get_test_device()
    x = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device).contiguous()

    y, s = act_quant(x, block_size=block_size, scale_fmt=scale_fmt)

    if check_correctness:
        y_ref, s_ref = act_quant_torch(
            x, block_size=block_size, round_scale=(scale_fmt is not None))
        torch.testing.assert_close(y.float(), y_ref.float(), rtol=0.0, atol=0.0)
        torch.testing.assert_close(s, s_ref, rtol=1e-2, atol=1e-2)
        print("assert_tensors_similar passed")

    def fn():
        return act_quant(x, block_size=block_size, scale_fmt=scale_fmt)

    from tilelang.profiler import do_bench

    ms = do_bench(fn, rep=100, warmup=250)
    io_bytes = x.numel() * 2 + y.numel() * 1 + s.numel() * 4
    # Approximate op count for quantize path (abs/max/scale/div/clamp style ops).
    total_flops = 4 * x.numel()
    bandwidth_tbps = io_bytes / (ms * 1e-3) / 1e12
    tflops = total_flops / ms * 1e-9
    print(f"[PERF] case=act_quant device={device} "
          f"params=batch_size={batch_size},hidden_dim={hidden_dim},dtype={dtype},"
          f"block_size={block_size},scale_fmt={scale_fmt}")
    print(f"[PERF] avg_time_ms={ms:.3f} bandwidth_TBps={bandwidth_tbps:.6f} "
          f"tflops={tflops:.6f}")


if __name__ == "__main__":
    test_act_quant(
        batch_size=2048,
        hidden_dim=16384,
        dtype=torch.bfloat16,
        block_size=128,
        scale_fmt=None,
        check_correctness=True,
    )
