import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#define TILE_SIZE 32

__global__ void matmul_tiled_fp32(const float* A, const float* B, float* C, int N) {
    // Shared memory for tiles of A and B
    // Each block loads two TILE_SIZE x TILE_SIZE tiles into shared memory.
    // Total shared memory per block: 2 * TILE_SIZE * TILE_SIZE * sizeof(float)
    // For TILE_SIZE=32, this is 2 * 32 * 32 * 4 bytes = 8192 bytes, well within typical limits.
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Thread coordinates within the block
    int tx = threadIdx.x; // Column index within the block (0 to TILE_SIZE-1)
    int ty = threadIdx.y; // Row index within the block (0 to TILE_SIZE-1)

    // Global row and column for the current thread's C element
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float C_value = 0.0f; // Accumulator for the C[row][col] element

    // Loop over tiles along the K dimension (inner dimension of matrix multiplication)
    // (N + TILE_SIZE - 1) / TILE_SIZE calculates ceil(N / TILE_SIZE)
    for (int k_tile_idx = 0; k_tile_idx < (N + TILE_SIZE - 1) / TILE_SIZE; ++k_tile_idx) {
        // Calculate the starting index for the current K-tile
        int k_start = k_tile_idx * TILE_SIZE;

        // Load a tile from matrix A into shared memory sA
        // Each thread (ty, tx) loads one element sA[ty][tx]
        // This element comes from A[row][k_start + tx]
        if (row < N && (k_start + tx) < N) {
            sA[ty][tx] = A[row * N + (k_start + tx)];
        } else {
            sA[ty][tx] = 0.0f; // Pad with zeros if outside matrix A bounds
        }

        // Load a tile from matrix B into shared memory sB
        // Each thread (ty, tx) loads one element sB[ty][tx]
        // This element comes from B[k_start + ty][col]
        if ((k_start + ty) < N && col < N) {
            sB[ty][tx] = B[(k_start + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f; // Pad with zeros if outside matrix B bounds
        }

        // Synchronize threads within the block.
        // Ensures all threads have finished loading their part of sA and sB
        // before any thread starts using them for computation.
        __syncthreads();

        // Perform the dot product for the C_value using the shared memory tiles
        // C_value += sA[ty][i] * sB[i][tx]
        for (int i = 0; i < TILE_SIZE; ++i) {
            C_value += sA[ty][i] * sB[i][tx];
        }

        // Synchronize threads within the block again.
        // Ensures all threads have finished their computation on the current shared tiles
        // before the next iteration loads new data into sA and sB.
        __syncthreads();
    }

    // Store the final accumulated result into global memory C
    // Only write if the global row and column are within the matrix bounds.
    if (row < N && col < N) {
        C[row * N + col] = C_value;
    }
}
"""

# Compile and load the CUDA kernel
matmul_extension = load_inline(
    name="matmul_extension",
    cuda_sources=[cuda_source],
    functions=["matmul_tiled_fp32"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],  # Aggressive optimization flags
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom tiled CUDA kernel with shared memory.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the loaded CUDA kernel function
        self.matmul_op = matmul_extension.matmul_tiled_fp32

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        # Ensure inputs are on CUDA device and are float32
        A = A.to(torch.device("cuda"), dtype=torch.float32)
        B = B.to(torch.device("cuda"), dtype=torch.float32)

        N = A.shape[0]  # Assuming A and B are square matrices of shape (N, N)

        # Create an output tensor on the CUDA device with the same shape and dtype as A
        C = torch.empty((N, N), device=A.device, dtype=A.dtype)

        # Define TILE_SIZE (must match the one in the kernel source)
        TILE_SIZE = 32

        # Calculate grid and block dimensions for the kernel launch
        # Grid dimensions: Number of blocks needed to cover the N x N output matrix
        grid_dim_x = (N + TILE_SIZE - 1) // TILE_SIZE
        grid_dim_y = (N + TILE_SIZE - 1) // TILE_SIZE

        # Block dimensions: TILE_SIZE x TILE_SIZE threads per block
        block_dim_x = TILE_SIZE
        block_dim_y = TILE_SIZE

        # Launch the custom CUDA kernel
        # The 'grid' and 'block' arguments specify the dimensions for the kernel launch.
        self.matmul_op(
            A, B, C, N, grid=(grid_dim_x, grid_dim_y), block=(block_dim_x, block_dim_y)
        )

        return C
