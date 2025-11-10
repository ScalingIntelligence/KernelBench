import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for tiled matrix multiplication
# TILE_SIZE is set to 32 for optimal performance on many GPUs.
# Each thread block computes a TILE_SIZE x TILE_SIZE tile of the output matrix C.
# Each thread within a block computes one element of that tile.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Block row and column indices in the output matrix C
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column indices within the block
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Accumulator for the C element computed by this thread
    float Cvalue = 0;

    // Shared memory for A and B tiles
    // TILE_SIZE x TILE_SIZE for each matrix
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Iterate over the K dimension (inner dimension of matrix multiplication)
    // N / TILE_SIZE gives the number of tiles along the K dimension
    for (int m = 0; m < N / TILE_SIZE; ++m) {
        // Load a tile of A into shared memory sA
        // The element A[row_A][col_A] is loaded by thread (row, col)
        // row_A = blockRow * TILE_SIZE + row
        // col_A = m * TILE_SIZE + col
        sA[row][col] = A[(blockRow * TILE_SIZE + row) * N + (m * TILE_SIZE + col)];
        
        // Load a tile of B into shared memory sB
        // The element B[row_B][col_B] is loaded by thread (row, col)
        // row_B = m * TILE_SIZE + row
        // col_B = blockCol * TILE_SIZE + col
        sB[row][col] = B[(m * TILE_SIZE + row) * N + (blockCol * TILE_SIZE + col)];
        
        // Synchronize threads to ensure all shared memory loads are complete
        __syncthreads();

        // Perform the dot product for the current tiles
        // Each thread computes one element of the C tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += sA[row][k] * sB[k][col];
        }
        
        // Synchronize threads to ensure all writes to shared memory are visible
        // for the next iteration, if needed (though not strictly for this kernel)
        __syncthreads();
    }

    // Write the accumulated Cvalue to the global memory output matrix C
    // The element C[row_C][col_C] is written by thread (row, col)
    // row_C = blockRow * TILE_SIZE + row
    // col_C = blockCol * TILE_SIZE + col
    C[(blockRow * TILE_SIZE + row) * N + (blockCol * TILE_SIZE + col)] = Cvalue;
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are contiguous and on CUDA device
    A = A.contiguous();
    B = B.contiguous();

    int N = A.size(0);
    // Create output tensor C, initialized to zeros, with same options as A
    auto C = torch::zeros({N, N}, A.options());

    // Define grid and block dimensions
    // Each block computes a TILE_SIZE x TILE_SIZE tile of C
    // Grid dimensions: (N / TILE_SIZE, N / TILE_SIZE)
    dim3 grid(N / TILE_SIZE, N / TILE_SIZE);
    // Block dimensions: (TILE_SIZE, TILE_SIZE)
    dim3 block(TILE_SIZE, TILE_SIZE);

    // Launch the kernel
    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    // Check for CUDA errors (optional, but good for debugging)
    // CUDA_POST_KERNEL_CHECK; 

    return C;
}
"""

# C++ function declaration
cpp_source = "torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
custom_matmul_module = load_inline(
    name="custom_matmul_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["custom_matmul_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom CUDA kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_matmul = custom_matmul_module.custom_matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.custom_matmul(A, B)
