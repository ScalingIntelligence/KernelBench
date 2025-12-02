import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# ThunderKittens header-only library path (set via environment variable)
# Default to /root/ThunderKittens for Modal containers, or use THUNDERKITTENS_PATH env var
TK_PATH = os.environ.get("THUNDERKITTENS_PATH", os.environ.get("THUNDERKITTENS_ROOT", "/root/ThunderKittens"))

# C++ source: function declaration for binding
elementwise_add_cpp_source = """
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);
"""

# CUDA source: ThunderKittens kernel implementation
# 
# IMPORTANT ThunderKittens API notes:
# 1. Define KITTENS_HOPPER before including kittens.cuh for H100/Hopper GPUs
# 2. Operations like load, store, zero, mma_AB are NOT free functions!
#    They are static member functions inside kittens::group<N> template struct.
# 3. Create an alias like: using warp = kittens::group<1>;
# 4. Then call: warp::load(...), warp::zero(...), etc.
#
elementwise_add_cuda_source = """
// IMPORTANT: Define KITTENS_HOPPER before including ThunderKittens headers for H100/Hopper GPUs
// This enables FP8 types and Hopper-specific features
#define KITTENS_HOPPER

#include <torch/extension.h>
#include <cuda_runtime.h>

// Include ThunderKittens headers
#include "kittens.cuh"

// ThunderKittens namespace and group aliases
// Operations are accessed through these group types, NOT as free functions
using namespace kittens;
using warp = kittens::group<1>;  // For single-warp operations (32 threads)
// For multi-warp operations, use: using warpgroup = kittens::group<4>;

// Constants for tile dimensions
constexpr int TILE_DIM = 16;

// ThunderKittens elementwise add kernel using shared memory tiles
// This example demonstrates the ThunderKittens API pattern
__global__ void tk_elementwise_add_kernel(const float* __restrict__ a_ptr, 
                                           const float* __restrict__ b_ptr, 
                                           float* __restrict__ out_ptr, 
                                           int rows, int cols) {
    // For simple element-wise ops, we use a straightforward approach
    // ThunderKittens shines for matrix ops with tiles, but here we show basic pattern
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    // Grid-stride loop for simple element-wise addition
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

// Alternative: ThunderKittens tiled version for larger matrices
// Shows proper usage of ThunderKittens tile types and group operations
// Uncomment and adapt for matrix operations:
/*
__global__ void tk_matmul_kernel(const bf16* A, const bf16* B, bf16* C, 
                                  int M, int N, int K) {
    // Define aliases for the group - THIS IS REQUIRED for ThunderKittens ops
    using warpgroup = kittens::group<4>;  // 4 warps = 128 threads
    
    // ThunderKittens register tiles for accumulation
    rt_fl<16, 16> acc;  // 16x16 float register tile
    
    // Shared memory tiles  
    extern __shared__ alignment_dummy __shm[];
    st_bf<16, 16> (&a_smem)[2] = *reinterpret_cast<st_bf<16, 16>(*)[2]>(__shm);
    st_bf<16, 16> (&b_smem)[2] = *reinterpret_cast<st_bf<16, 16>(*)[2]>(__shm + sizeof(st_bf<16,16>)*2);
    
    // Initialize accumulator to zero - NOTE: use warpgroup:: prefix!
    warpgroup::zero(acc);
    
    // Main loop would go here with:
    // warpgroup::load(a_smem[...], ...);   // Load from global to shared
    // warpgroup::mma_AB(acc, a_tile, b_tile);  // Matrix multiply-accumulate
    // warpgroup::store(C_ptr, acc, ...);  // Store result
}
*/

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "Input tensor a must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "Input tensor b must be on CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    
    auto out = torch::empty_like(a);
    int rows = a.size(0);
    int cols = a.numel() / rows;

    const int block_size = 256;
    const int num_blocks = (a.numel() + block_size - 1) / block_size;

    tk_elementwise_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        out.data_ptr<float>(), 
        rows, cols
    );

    return out;
}
"""

# Compile the ThunderKittens kernel inline
elementwise_add = load_inline(
    name="elementwise_add_tk",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_cuda_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_include_paths=[
        TK_PATH,
        os.path.join(TK_PATH, "include"),
    ],
    extra_cflags=["-std=c++20", "-O3"],
    extra_cuda_cflags=[
        "-std=c++20",
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler", "-fPIC",
        "-DNDEBUG",
        "-DKITTENS_HOPPER",
    ],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
