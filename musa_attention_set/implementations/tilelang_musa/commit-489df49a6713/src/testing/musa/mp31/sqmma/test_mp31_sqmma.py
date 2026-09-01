import torch
import pytest
import tilelang
import tilelang.language as T

tilelang.disable_cache()

TORCH_DTYPE_TO_TILELANG = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
}

TOLERANCE_BY_DTYPE = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (7.9e-3, 7.9e-3),
    torch.int8: (0.0, 0.0),
    torch.uint8: (0.0, 0.0),
    torch.float8_e4m3fn: (1.25e-1, 1.25e-1),
    torch.float8_e5m2: (2.5e-1, 2.5e-1),
}

ACCUM_DTYPE = {
    torch.float16: "float32",
    torch.bfloat16: "float32",
    torch.int8: "int32",
    torch.uint8: "int32",
    torch.float8_e4m3fn: "float32",
    torch.float8_e5m2: "float32",
}

OUTPUT_DTYPE = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int8: "int32",
    torch.uint8: "int32",
    torch.float8_e4m3fn: "float32",
    torch.float8_e5m2: "float32",
}


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads,
    dtype,
    out_dtype,
    accum_dtype,
    trans_A=False,
    trans_B=True,
):
    if trans_A and trans_B:

        @T.prim_func
        def gemm(
            A: T.Tensor((K, M), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_K, block_M), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[k * block_K, by * block_M], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.sqmma_gemm(A_shared, B_shared, C_local, transpose_A=True, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    if trans_A:

        @T.prim_func
        def gemm(
            A: T.Tensor((K, M), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_K, block_M), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[k * block_K, by * block_M], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.sqmma_gemm(A_shared, B_shared, C_local, transpose_A=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    if trans_B:

        @T.prim_func
        def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.sqmma_gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.sqmma_gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


elem_type_list = list(TORCH_DTYPE_TO_TILELANG)
size_list = [(256, 256, 256)]
threads_block_list = [
    (128, 64, 64, 128),
    (512, 128, 128, 128),
]
test_params = [
    (
        elem_type,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        threads,
        trans_A,
        trans_B,
    )
    for elem_type in elem_type_list
    for (M, N, K) in size_list
    for (threads, BLOCK_M, BLOCK_N, BLOCK_K) in threads_block_list
    for (trans_A, trans_B) in [(False, False), (True, True)]
    if M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0
]


@pytest.mark.parametrize(
    "elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, threads, trans_A, trans_B",
    test_params,
)
def test_mm_kernel(
    elem_type,
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    threads,
    trans_A,
    trans_B,
):
    device = "musa"
    a_shape = (K, M) if trans_A else (M, K)
    b_shape = (N, K) if trans_B else (K, N)
    A = torch.randn(a_shape, dtype=torch.float16, device=device).to(elem_type)
    B = torch.randn(b_shape, dtype=torch.float16, device=device).to(elem_type)
    program = matmul(
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        threads,
        trans_A=trans_A,
        trans_B=trans_B,
        dtype=TORCH_DTYPE_TO_TILELANG[elem_type],
        out_dtype=OUTPUT_DTYPE[elem_type],
        accum_dtype=ACCUM_DTYPE[elem_type],
    )
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
        verbose=True,
    )
    logical_A = A.T if trans_A else A
    logical_B = B.T if trans_B else B
    rtol, atol = TOLERANCE_BY_DTYPE[elem_type]
    if elem_type in (torch.int8, torch.uint8):
        ref_out = torch.matmul(logical_A.cpu().int(), logical_B.cpu().int())
    else:
        ref_out = torch.matmul(logical_A.cpu().float(), logical_B.cpu().float()).to(getattr(torch, OUTPUT_DTYPE[elem_type]))
    C = kernel(A, B)
    torch.testing.assert_close(
        C.cpu().to(torch.float32),
        ref_out.cpu().to(torch.float32),
        rtol=rtol,
        atol=atol,
    )
    print(
        f"elem_type={elem_type}, M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, "
        f"BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, threads={threads}, "
        f"trans_A={trans_A}, trans_B={trans_B}"
    )
    print("Pass")
