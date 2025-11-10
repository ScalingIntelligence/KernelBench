import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source (includes both kernel and C++ wrapper)
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define TILE_DIM for shared memory tiling. A common choice is 32.
#define TILE_DIM 32

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    // Shared memory for tiles of A and B
    // Using volatile to prevent compiler from optimizing away redundant reads/writes
    // which can happen across __syncthreads() in some older CUDA versions/compilers,
    // though generally not strictly necessary for modern PyTorch/CUDA.
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    // Thread coordinates within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global coordinates for the C element this thread computes
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float Cvalue = 0.0f; // Accumulator for C[row][col], stored in a register

    // Loop over the K dimension (inner product)
    // Each iteration processes a TILE_DIM x TILE_DIM block of the K dimension
    for (int k_block_offset = 0; k_block_offset < N; k_block_offset += TILE_DIM) {
        // Load tiles of A and B from global memory into shared memory.
        // Each thread loads one element.
        // sA[ty][tx] loads A[row][k_block_offset + tx]
        // sB[ty][tx] loads B[k_block_offset + ty][col]
        
        // Ensure global memory accesses are within bounds of A and B
        // For A: row index `row`, column index `k_block_offset + tx`
        if (row < N && (k_block_offset + tx) < N) {
            sA[ty][tx] = A[row * N + (k_block_offset + tx)];
        } else {
            sA[ty][tx] = 0.0f; // Pad with zero if out of bounds (handles non-multiples of TILE_DIM)
        }

        // For B: row index `k_block_offset + ty`, column index `col`
        if ((k_block_offset + ty) < N && col < N) {
            sB[ty][tx] = B[(k_block_offset + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f; // Pad with zero if out of bounds
        }

        // Synchronize to ensure all shared memory loads are complete before computation begins
        __syncthreads();

        // Perform dot product using the loaded shared memory tiles
        for (int k_inner = 0; k_inner < TILE_DIM; ++k_inner) {
            Cvalue += sA[ty][k_inner] * sB[k_inner][tx];
        }

        // Synchronize before loading the next set of tiles to avoid race conditions
        __syncthreads();
    }

    // Write the accumulated result back to global memory
    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation (optional, but good practice)
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors.");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices.");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions mismatch for multiplication.");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1) && A.size(0) == B.size(0), 
                "Matrices must be square and of the same size.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat && B.scalar_type() == torch::kFloat, 
                "Inputs must be float32 (torch.float).");

    int N = A.size(0);
    // Create output tensor C on the same device and with the same dtype as A
    auto C = torch::zeros({N, N}, A.options());

    // Define grid and block dimensions for the kernel launch
    dim3 block_size(TILE_DIM, TILE_DIM);
    dim3 grid_size((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // Launch the CUDA kernel
    matmul_tiled_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}
"""

# C++ function declaration for load_inline
cpp_source = "torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
# The 'name' argument should be unique if multiple extensions are loaded
custom_matmul_extension = load_inline(
    name="custom_matmul_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_tiled_cuda"],
    verbose=False,  # Set to True for debugging compilation issues
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom tiled CUDA kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the loaded custom CUDA function as an attribute
        self.custom_matmul_op = custom_matmul_extension.matmul_tiled_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom tiled CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        # The custom CUDA kernel expects CUDA tensors, ensure inputs are on GPU.
        # This is typically handled by the calling context, but good to be aware.
        return self.custom_matmul_op(A, B)
