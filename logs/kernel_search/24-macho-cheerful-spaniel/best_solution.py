import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code for tiled matrix multiplication
cuda_source = """
#define BLOCK_SIZE 32 // Define the tile size for shared memory and thread block dimensions

extern "C" __global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    // Block index in X and Y dimensions
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index within the block in X and Y dimensions
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate the global row and column of the C element this thread is responsible for
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Accumulator for the C element
    float C_value = 0;

    // Declare shared memory for tiles of A and B
    // These tiles will be loaded from global memory once per iteration and reused by all threads in the block
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over the "strips" of A and B matrices
    // Each iteration processes a BLOCK_SIZE-wide strip from A and B
    for (int k = 0; k < N; k += BLOCK_SIZE) {
        // Load a tile of A into shared memory sA
        // Each thread loads one element from global memory A.
        // The access pattern A[row][k + tx] is coalesced across threads in a warp for a given row.
        if (row < N && (k + tx) < N) {
            sA[ty][tx] = A[row * N + (k + tx)];
        } else {
            // Pad with zero if out of bounds (important for correctness if N is not a multiple of BLOCK_SIZE)
            sA[ty][tx] = 0.0f; 
        }
        
        // Load a tile of B into shared memory sB
        // Each thread loads one element from global memory B.
        // The access pattern B[k + ty][col] is coalesced across threads in a warp for a given column.
        if ((k + ty) < N && col < N) {
            sB[ty][tx] = B[(k + ty) * N + col];
        } else {
            // Pad with zero if out of bounds
            sB[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all threads in the block have loaded their data into shared memory
        __syncthreads();

        // Perform the dot product of the current shared memory tiles
        // Each thread computes its partial sum for C_value
        #pragma unroll // Hint to the compiler to unroll this loop for potentially better performance
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            C_value += sA[ty][i] * sB[i][tx];
        }

        // Synchronize to ensure all threads have completed their partial sums before the next iteration
        // This prevents reading stale shared memory data if next iteration starts loading new tiles
        __syncthreads();
    }

    // Write the final accumulated result to global memory C
    // Only write if the current thread's calculated C element is within the matrix bounds
    if (row < N && col < N) {
        C[row * N + col] = C_value;
    }
}
"""

# Global variable to store the compiled CUDA module.
# This ensures the kernel is compiled only once.
_matmul_cuda_module = None


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom tiled CUDA kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        global _matmul_cuda_module
        if _matmul_cuda_module is None:
            # Compile the CUDA kernel using torch.utils.cpp_extension.load_inline
            _matmul_cuda_module = load_inline(
                name="matmul_tiled_extension",
                cpp_sources=[""],  # No C++ sources needed for this pure CUDA kernel
                cuda_sources=[cuda_source],
                functions=["matmul_tiled_kernel"],
                verbose=False,
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                ],  # Optimization flags
            )
        # Store the compiled kernel function as an attribute
        self.matmul_kernel = _matmul_cuda_module.matmul_tiled_kernel

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        # Ensure tensors are on CUDA, are float32, and are contiguous for kernel compatibility
        A = A.to(torch.float32).contiguous().cuda()
        B = B.to(torch.float32).contiguous().cuda()

        N = A.shape[0]  # Get the dimension N from the input tensor A

        # Allocate the output tensor C on the CUDA device
        C = torch.empty((N, N), dtype=torch.float32, device="cuda")

        # Define grid and block dimensions for kernel launch
        BLOCK_SIZE = 32  # Must match the #define in the CUDA source
        grid_dim_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_dim_y = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_dim = (grid_dim_x, grid_dim_y)  # Grid dimensions (blocks per grid)
        block_dim = (BLOCK_SIZE, BLOCK_SIZE)  # Block dimensions (threads per block)

        # Launch the custom CUDA kernel
        # [grid_dim, block_dim] specifies the launch configuration
        self.matmul_kernel[grid_dim, block_dim](A, B, C, N)

        return C
