import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// TILE_DIM defines the tile size for shared memory and thread block dimensions.
#define TILE_DIM 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Block index in the grid
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread index within the block
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Calculate the global row and column for the C element this thread is responsible for
    int C_row = blockRow * TILE_DIM + threadRow;
    int C_col = blockCol * TILE_DIM + threadCol;

    // Declare shared memory for tiles of A and B
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    float Cvalue = 0.0f; // Accumulator for the C element

    // Iterate over the 'k' dimension (inner product dimension) in tiles
    // Each iteration processes a pair of A and B tiles
    for (int tile = 0; tile < (N + TILE_DIM - 1) / TILE_DIM; ++tile) {
        // Load a tile of A from global memory to shared memory
        // Each thread loads one element from A
        int k_global_A = tile * TILE_DIM + threadCol;
        if (C_row < N && k_global_A < N) {
            sA[threadRow][threadCol] = A[C_row * N + k_global_A];
        } else {
            sA[threadRow][threadCol] = 0.0f; // Pad with zeros for out-of-bounds access
        }

        // Load a tile of B from global memory to shared memory
        // Each thread loads one element from B
        int k_global_B = tile * TILE_DIM + threadRow;
        if (k_global_B < N && C_col < N) {
            sB[threadRow][threadCol] = B[k_global_B * N + C_col];
        } else {
            sB[threadRow][threadCol] = 0.0f; // Pad with zeros for out-of-bounds access
        }
        
        __syncthreads(); // Synchronize to ensure all shared memory loads are complete

        // Perform the matrix multiplication for the current tiles
        // Each thread computes its part of the dot product
        for (int k = 0; k < TILE_DIM; ++k) {
            Cvalue += sA[threadRow][k] * sB[k][threadCol];
        }
        
        __syncthreads(); // Synchronize to ensure all threads finish using current sA/sB before next load
    }

    // Write the accumulated result from register to global memory
    if (C_row < N && C_col < N) {
        C[C_row * N + C_col] = Cvalue;
    }
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor matmul_cuda_wrapper(torch::Tensor A, torch::Tensor B) {
    // Basic input validation
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square matrices");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");

    int N = A.size(0);

    // Create the output tensor on the same device as inputs
    auto C = torch::empty({N, N}, A.options());

    // Define block and grid dimensions
    // Each block processes a TILE_DIM x TILE_DIM sub-matrix of C
    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // Launch the CUDA kernel
    matmul_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}
"""

cpp_source = "torch::Tensor matmul_cuda_wrapper(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code using torch.utils.cpp_extension.load_inline
matmul_custom = load_inline(
    name="matmul_custom_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_cuda_wrapper"],
    verbose=True,  # Set to True to see compilation output
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom tiled CUDA kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled custom operation as a member
        self.matmul_op = matmul_custom.matmul_cuda_wrapper

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.matmul_op(A, B)
