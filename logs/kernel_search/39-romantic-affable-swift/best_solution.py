import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for tiled matrix multiplication
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled matrix multiplication kernel
// C = A * B
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    // Block and thread indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    // TILE_SIZE must match blockDim.x and blockDim.y
    const int TILE_SIZE = 32; 

    // Global row and column of the C element computed by this thread
    int C_row = blockRow * TILE_SIZE + row;
    int C_col = blockCol * TILE_SIZE + col;

    float C_value = 0.0f; // Accumulator for the C element

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over the K dimension (inner product)
    // Each iteration processes one "sub-tile" multiplication
    for (int tile_idx = 0; tile_idx < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {
        // Load tiles from global memory to shared memory
        // Each thread loads one element into As and one into Bs
        int A_global_row = blockRow * TILE_SIZE + row;
        int A_global_col = tile_idx * TILE_SIZE + col;
        int B_global_row = tile_idx * TILE_SIZE + row;
        int B_global_col = blockCol * TILE_SIZE + col;

        // Load element for As
        if (A_global_row < N && A_global_col < N) {
            As[row][col] = A[A_global_row * N + A_global_col];
        } else {
            As[row][col] = 0.0f; // Pad with zeros if out of bounds
        }

        // Load element for Bs
        if (B_global_row < N && B_global_col < N) {
            Bs[row][col] = B[B_global_row * N + B_global_col];
        } else {
            Bs[row][col] = 0.0f; // Pad with zeros if out of bounds
        }

        __syncthreads(); // Wait for all threads in the block to load their tile

        // Perform dot product using shared memory tiles
        // Each thread computes one element of C_value for the current sub-tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            C_value += As[row][k] * Bs[k][col];
        }

        __syncthreads(); // Wait for all threads to finish computation for this sub-tile
    }

    // Write the accumulated result to global memory
    if (C_row < N && C_col < N) {
        C[C_row * N + C_col] = C_value;
    }
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors.");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous.");
    TORCH_CHECK(A.dtype() == torch::kFloat && B.dtype() == torch::kFloat, "Inputs must be float32.");
    
    int N = A.size(0);
    TORCH_CHECK(A.size(1) == N && B.size(0) == N && B.size(1) == N, 
                "Inputs must be square matrices of the same size.");

    // Create output tensor on the same device and with the same dtype as inputs
    auto C = torch::zeros({N, N}, A.options());

    // Kernel launch configuration
    const int TILE_SIZE = 32; // This must match the TILE_SIZE in the kernel
    dim3 block_dim(TILE_SIZE, TILE_SIZE); // 2D block for 2D matrix operations
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE); // 2D grid

    // Launch the kernel
    matmul_tiled_kernel<<<grid_dim, block_dim>>>(
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
# The name 'matmul_custom_extension' should be unique
_matmul_custom = load_inline(
    name="matmul_custom_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_cuda"],  # The C++ function name to expose
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom tiled CUDA kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled CUDA function
        self.matmul_op = _matmul_custom.matmul_cuda

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
