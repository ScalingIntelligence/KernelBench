import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse

tilelang.disable_cache()


def make_k_perm_layout(K, block_N):
    l = block_N // 8

    def perm(n):
        # permute within each block_N tile
        tile = n // block_N
        in_tile = n % block_N
        return tile * block_N + (in_tile % 8) * l + (in_tile // 8)

    return T.Layout(K.shape, lambda b, h, n, d: [b, h, perm(n), d])


TARGET = "musa"
DEVICE = "musa"


def dual_gemm(
    batch,
    heads,
    seq_q,
    seq_kv,
    dim,
    is_causal,
    block_M=256,
    block_N=64,
    num_stages=1,
    threads=512,
):
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = "float16"
    accum_dtype = "float"
    L = block_N // 8  # 8 means 8 threads, L means Local Elem

    @T.macro
    def MMA0(
        K: T.Tensor(kv_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
        T.clear(acc_s)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        P_shared: T.SharedBuffer([block_M, block_N], dtype),
        V_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
        T.gemm(P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
                T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (
                    bx,
                    by,
                    bz,
                ):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            P_shared = T.alloc_shared([block_M, block_N], dtype)
            P_linear = T.alloc_shared([block_M, block_N], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)

            T.annotate_layout({
                K: make_k_perm_layout(K, block_N),
            })

            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)

            for k in T.Pipelined(T.ceildiv(seq_kv, block_N), num_stages=num_stages):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                # T.copy(acc_s, P_shared)

                for i, t in T.Parallel(block_M, 8):
                    base = t * L
                    for l in T.vectorized(L):
                        P_linear[i, base + l] = acc_s[i, l * 8 + t]
                T.copy(P_linear, P_shared)

                MMA1(V, P_shared, V_shared, acc_o, k, by, bz)

            T.copy(acc_o, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

    return main


def ref_program(Q, K, V, is_causal):
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    # attention_weights = scores
    output = torch.einsum("bhqk,bhkd->bhqd", scores, V)

    return output


def main(
    batch: int = 1,
    heads: int = 1,
    seq_q: int = 64,
    seq_kv: int = 64,
    dim: int = 128,
    is_causal: bool = False,
    tune: bool = False,
    verbose: bool = False,
):

    print(
        f" batch:{batch}, heads: {heads}, seq_q: {seq_q}, seq_kv:{seq_kv}, dim:{dim}, is_causal:{is_causal}"
    )

    program = dual_gemm(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        is_causal,
        block_M=256,
        block_N=64,
        num_stages=1,
        threads=512,
    )
    dtype = "float16"
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
    }

    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target=TARGET,
        execution_backend="cython",
        verbose=verbose,
        pass_configs=pass_configs,
    )

    if verbose:
        print(kernel.get_kernel_source())

    q = torch.rand(batch, heads, seq_q, dim, device=DEVICE, dtype=getattr(torch, dtype))
    k = torch.rand(batch, heads, seq_kv, dim, device=DEVICE, dtype=getattr(torch, dtype))
    v = torch.rand(batch, heads, seq_kv, dim, device=DEVICE, dtype=getattr(torch, dtype))

    output = kernel(q, k, v)
    ref_output = ref_program(q, k, v, is_causal)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("All checks pass.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=1, help="heads")
    parser.add_argument("--seq_q", type=int, default=64, help="query sequence length")
    parser.add_argument("--seq_kv", type=int, default=64, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    main(
        args.batch,
        args.heads,
        args.seq_q,
        args.seq_kv,
        args.dim,
        args.is_causal,
        args.tune,
        args.verbose,
    )
