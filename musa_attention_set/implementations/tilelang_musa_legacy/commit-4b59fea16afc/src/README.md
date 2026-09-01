

# Tilelang_MUSA

## Deprecated

This repository is no longer maintained or used. The codebase has been migrated to a new repository and all future development will continue there.

Please use the new repository for the latest source code, issues, and updates:

[Tilelang MUSA](https://github.com/tile-ai/tilelang-musa)

---

Tile Language (**tile-lang**) is a concise domain-specific language designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). By employing a Pythonic syntax with an underlying compiler infrastructure on top of [TVM](https://tvm.apache.org/), tile-lang allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.

Tilelang MUSA is a deeply adapted version of Tile Language for the MUSA platform and Moore Threads GPUs. It modifies the compilation pipeline (including Passes and Codegen) to target the MUSA toolchain and generate MUSA C that invokes `tl`-namespace templates under `src/tl_templates/musa`. It supports almost all Tilelang instructions, except for a small subset of NVIDIA hardware-specific instructions.

<img src=./images/MatmulExample.png />

## New

### 0.1.6.post2+musa3

- Add a high-performance DSA kernel implementation.
- Supports the vast majority of Tilelang's features, over 95% of the test cases passed.

### 0.1.8+musa1

tilelang_musa0.1.8+musa1 is largely complete and will be officially released once further refinements are made.

## Tested Devices

Tilelang MUSA already supports S5000, S4000, and M1000.

## OP Implementation Examples

**tile-lang** provides the building blocks to implement a wide variety of operators. Some examples include:

- [Matrix Multiplication](./musa_tests/basic/mm_mma_stage_num3.py)
- [Flash Attention](./musa_tests/flash_attention/example_mha_fwd_bhsd.py)
- [Dequantization GEMM](./examples/dequantize_gemm/)
- [Flash Linear Attention](./examples/linear_attention/)
- [Flash MLA Decoding](./examples/deepseek_mla/)
- [Native Sparse Attention](./examples/deepseek_nsa/)

Within the `examples` directory, you will also find additional complex kernels—such as convolutions, forward/backward passes for FlashAttention, more operators will continuously be added.

## Build from Source

We recommend using a virtual environment.

```
conda create -n tilelang-dev python=3.10
conda activate tilelang-dev
```

Install Tilelang_MUSA

```
git clone --recursive https://github.com/MooreThreads/tilelang_musa.git
cd tilelang
pip install -r ./requirements-dev.txt
export USE_MUSA=1
pip install -e . -v --no-build-isolation
```

## Quick Start

In this section, you'll learn how to write and execute a straightforward GEMM (matrix multiplication) kernel using tile-lang, followed by techniques for layout optimizations, pipelining, and L2-cache–friendly swizzling.

### GEMM Example with Annotations (Layout, L2 Cache Swizzling, and Pipelining, etc.)

Below is an example that demonstrates more advanced features: layout annotation, parallelized copy, and swizzle for improved L2 cache locality. This snippet shows how to adapt your kernel to maximize performance on complex hardware.

```python
import tilelang
import tilelang.language as T

# @tilelang.jit(target="musa")
# if not specified, it will be inferred from the input tensors during compile time
@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def matmul_relu_kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Enable rasterization for better L2 cache locality (Optional)
            # T.use_swizzle(panel_size=4, order='col')

            # Clear local accumulation
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                # This is a sugar syntax for parallelized copy
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Copy tile of B
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # Perform a tile-level GEMM on the shared buffers
                # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
                T.gemm(A_shared, B_shared, C_local)

            # relu
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_relu_kernel


M = 1024  # M = T.dynamic("m") if you want to use dynamic shape
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 64

# 1. Define the kernel (matmul) and compile/lower it into an executable module
matmul_relu_kernel = matmul(M, N, K, block_M, block_N, block_K)

# 3. Test the kernel in Python with PyTorch data
import torch

# Create random input tensors on the GPU
a = torch.randn(M, K, device="musa", dtype=torch.float16)
b = torch.randn(K, N, device="musa", dtype=torch.float16)
c = torch.empty(M, N, device="musa", dtype=torch.float16)

# Run the kernel through the Profiler
matmul_relu_kernel(a, b, c)

print(c)
# Reference multiplication using PyTorch
ref_c = torch.relu(a @ b)

# Validate correctness
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel output matches PyTorch reference.")

# 4. Retrieve and inspect the generated MUSA source (optional)
# musa_source = jit_kernel.get_kernel_source()
# print("Generated MUSA kernel:\n", musa_source)

# 5.Profile latency with kernel
profiler = matmul_relu_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

latency = profiler.do_bench()

print(f"Latency: {latency} ms")
```

## More Docs

See the official [Tilelang documentation](https://tilelang.com/) and the [Tilelang MUSA Programming Guide](docs/tilelang_musa_programming_guide.md).

## Acknowledgments

We would like to express our gratitude to the [Tilelang](https://github.com/tile-ai/tilelang) community and [TVM](https://github.com/apache/tvm) community for their invaluable contributions.
