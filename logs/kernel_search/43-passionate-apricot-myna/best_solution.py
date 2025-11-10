import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source (includes both kernel and C++ wrapper)
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// TILE_SIZE defines the dimensions of the shared memory tiles
// and the number of threads per dimension in a block.
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread index within the block
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Shared memory for A_sub and B_sub tiles
    // These will store TILE_SIZE x TILE_SIZE sub-matrices from A and B
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // C_value is used to accumulate the element C[row][col]
    // that is computed by the current thread
    float C_value = 0.0f;

    // Calculate the global row and column of the C element computed by this thread
    int row = blockRow * TILE_SIZE + threadRow;
    int col = blockCol * TILE_SIZE + threadCol;

    // Loop over the K dimension (inner product) in steps of TILE_SIZE
    // Each iteration processes a pair of tiles from A and B
    for (int k_tile = 0; k_tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
        // Load A_sub from global memory to shared memory sA
        // Each thread loads one element.
        // The element is from A[row][k_tile*TILE_SIZE + threadCol]
        int globalA_row = row;
        int globalA_col = k_tile * TILE_SIZE + threadCol;
        if (globalA_row < N && globalA_col < N) {
            sA[threadRow][threadCol] = A[globalA_row * N + globalA_col];
        } else {
            // Pad with zero if out of bounds (handles non-divisible N, though N=4096 is divisible by TILE_SIZE=32)
            sA[threadRow][threadCol] = 0.0f; 
        }

        // Load B_sub from global memory to shared memory sB
        // Each thread loads one element.
        // The element is from B[k_tile*TILE_SIZE + threadRow][col]
        int globalB_row = k_tile * TILE_SIZE + threadRow;
        int globalB_col = col;
        if (globalB_row < N && globalB_col < N) {
            sB[threadRow][threadCol] = B[globalB_row * N + globalB_col];
        } else {
            // Pad with zero if out of bounds
            sB[threadRow][threadCol] = 0.0f;
        }

        // Synchronize to make sure all data is loaded into shared memory
        __syncthreads();

        // Perform the matrix multiplication for the current tiles
        // Each thread computes a partial sum for its C_value
        for (int i = 0; i < TILE_SIZE; ++i) {
            C_value += sA[threadRow][i] * sB[i][threadCol];
        }

        // Synchronize to make sure all computations are done
        // before the next iteration (new data loaded into shared memory)
        __syncthreads();
    }

    // Write the final result from C_value to global memory C
    if (row < N && col < N) {
        C[row * N + col] = C_value;
    }
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and of float type
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat, "Input A must be a float tensor");
    TORCH_CHECK(B.dtype() == torch::kFloat, "Input B must be a float tensor");

    // Ensure inputs are contiguous for direct pointer access
    A = A.contiguous();
    B = B.contiguous();
    
    // Basic shape checks
    TORCH_CHECK(A.dim() == 2, "Input A must be 2D");
    TORCH_CHECK(B.dim() == 2, "Input B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "Input A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "Input B must be square");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions of A and B must match for matrix multiplication");

    int N = A.size(0); // Assuming square matrices N x N

    // Allocate output tensor C with the same options (device, dtype) as A
    auto C = torch::empty({N, N}, A.options());

    // Calculate grid and block dimensions for the tiled kernel
    // Block dimensions are TILE_SIZE x TILE_SIZE threads
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    // Grid dimensions are (N/TILE_SIZE) x (N/TILE_SIZE) blocks
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matmul_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
"""

# C++ function declaration for load_inline
cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
custom_matmul_module = load_inline(
    name="custom_matmul",
    cpp_sources=cpp_source,  # C++ function declaration
    cuda_sources=cuda_source,  # CUDA kernel implementation and C++ wrapper
    functions=["matmul_cuda"],  # The C++ function name to expose to Python
    verbose=True,  # Enable verbose output for compilation details
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom CUDA kernel with tiled shared memory approach.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled CUDA function as an attribute
        self.custom_matmul = custom_matmul_module.matmul_cuda

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
