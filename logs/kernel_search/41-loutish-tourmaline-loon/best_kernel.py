import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source (includes both kernel definition and C++ wrapper)
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define BLOCK_SIZE for shared memory tiling. A common and effective size is 32.
#define BLOCK_SIZE 32

// CUDA kernel for tiled matrix multiplication (C = A * B)
// Each thread block computes a BLOCK_SIZE x BLOCK_SIZE tile of the output matrix C.
// Threads within a block cooperatively load sub-tiles of A and B into shared memory
// to perform the partial matrix multiplication for their assigned C tile.
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    // Shared memory for storing tiles of A and B.
    // As stores a BLOCK_SIZE x BLOCK_SIZE tile of matrix A.
    // Bs stores a BLOCK_SIZE x BLOCK_SIZE tile of matrix B.
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate the global row and column indices for the current thread's element in C.
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float Cvalue = 0.0f; // Accumulator for the C[row][col] element.

    // Loop over the tiles along the K dimension (the inner product dimension).
    // This loop ensures all necessary intermediate products are calculated.
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE gives the ceiling division for number of tiles.
    for (int tile_idx = 0; tile_idx < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile_idx) {
        // Calculate global indices for loading data from global memory into shared memory.
        // Each thread loads one element for As and one for Bs.
        int A_global_row = row;
        int A_global_col = tile_idx * BLOCK_SIZE + threadIdx.x; // K index for A
        int B_global_row = tile_idx * BLOCK_SIZE + threadIdx.y; // K index for B
        int B_global_col = col;

        // Load data from global memory to shared memory.
        // Perform boundary checks for inputs to avoid out-of-bounds access
        // and pad with zeros if outside the matrix dimensions.
        As[threadIdx.y][threadIdx.x] = (A_global_row < N && A_global_col < N) ? A[A_global_row * N + A_global_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (B_global_row < N && B_global_col < N) ? B[B_global_row * N + B_global_col] : 0.0f;

        // Synchronize threads to ensure all shared memory data is loaded
        // before any thread starts computation using that shared data.
        __syncthreads();

        // Perform the dot product for the current tile.
        // Each thread calculates a partial sum for C[row][col] using the shared memory tiles.
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize threads to ensure all shared memory reads are complete
        // before the next iteration potentially loads new data into shared memory.
        __syncthreads();
    }

    // Write the accumulated result to global memory, with a boundary check
    // to ensure we only write within the valid matrix dimensions.
    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

// C++ wrapper function that prepares data and launches the CUDA kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure input tensors are on the CUDA device and are contiguous in memory.
    // This is crucial for direct data_ptr access in the kernel.
    A = A.to(torch::kCUDA).contiguous();
    B = B.to(torch::kCUDA).contiguous();

    // Get the matrix dimension N. We assume square matrices of size N x N.
    int N = A.size(0);

    // Create the output tensor C on the CUDA device, initialized to zeros.
    // It will have the same dimensions, data type, and device as the input tensor A.
    auto C = torch::zeros({N, N}, A.options());

    // Define thread block dimensions. Using BLOCK_SIZE from the kernel for consistency.
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    
    // Calculate grid dimensions. Each block processes a BLOCK_SIZE x BLOCK_SIZE tile of C.
    // The ceiling division `(N + BLOCK_SIZE - 1) / BLOCK_SIZE` ensures enough blocks
    // are launched to cover the entire matrix, even if N is not a multiple of BLOCK_SIZE.
    dim3 grid_dim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the CUDA kernel with the specified grid and block dimensions.
    // Pass raw pointers to the tensor data and the matrix dimension N.
    matmul_tiled_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), // Pointer to the start of matrix A's data
        B.data_ptr<float>(), // Pointer to the start of matrix B's data
        C.data_ptr<float>(), // Pointer to the start of matrix C's data
        N                    // The dimension N of the square matrices
    );

    // Return the resulting output tensor.
    return C;
}
"""

# C++ function declaration. This is required by load_inline.
cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile and load the custom CUDA kernel using torch.utils.cpp_extension.load_inline.
matmul_custom_op = load_inline(
    name="matmul_custom_tiled",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_cuda"],
    verbose=False,  # Set to True for detailed compilation output, useful for debugging.
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces the standard torch.matmul with a custom tiled CUDA kernel
    for square matrix multiplication (C = A * B).
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled custom operation module as a member of the nn.Module.
        self.matmul_op = matmul_custom_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom tiled CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        # Call the C++ wrapper function (matmul_cuda) from the loaded custom operation module.
        return self.matmul_op.matmul_cuda(A, B)
