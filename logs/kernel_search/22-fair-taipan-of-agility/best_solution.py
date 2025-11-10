import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for tiled matrix multiplication
cuda_source = """
#define TILE_SIZE 32

__global__ void matmul_tiled_kernel(float* C, const float* A, const float* B, int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column of C that this thread is responsible for
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Accumulator for the C element
    float Csub = 0;

    // Declare shared memory for tiles of A and B
    // As stores a TILE_SIZE x TILE_SIZE tile of matrix A
    // Bs stores a TILE_SIZE x TILE_SIZE tile of matrix B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over the K dimension (inner product dimension)
    // The loop iterates over "blocks" of K, each block being TILE_SIZE wide
    for (int k_block = 0; k_block < (N + TILE_SIZE - 1) / TILE_SIZE; ++k_block) {
        // Calculate the column index in A and row index in B for global memory access
        int k_idx_A = k_block * TILE_SIZE + tx; // A's column index
        int k_idx_B = k_block * TILE_SIZE + ty; // B's row index

        // Load a tile of A into shared memory
        // Each thread loads one element of A_tile
        if (row < N && k_idx_A < N) {
            As[ty][tx] = A[row * N + k_idx_A];
        } else {
            As[ty][tx] = 0.0f; // Pad with zeros if out of bounds
        }

        // Load a tile of B into shared memory
        // Each thread loads one element of B_tile
        if (k_idx_B < N && col < N) {
            Bs[ty][tx] = B[k_idx_B * N + col];
        } else {
            Bs[ty][tx] = 0.0f; // Pad with zeros if out of bounds
        }

        // Synchronize threads to ensure all shared memory loads are complete
        __syncthreads();

        // Perform the dot product for the current tile
        // Each thread computes one element of the Csub accumulator
        for (int m = 0; m < TILE_SIZE; ++m) {
            Csub += As[ty][m] * Bs[m][tx];
        }

        // Synchronize threads to ensure all computations using the current shared memory tiles are done
        // before the next iteration loads new data into shared memory
        __syncthreads();
    }

    // Write the accumulated Csub to global memory
    if (row < N && col < N) {
        C[row * N + col] = Csub;
    }
}
"""


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom CUDA kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Compile the CUDA kernel
        self.matmul_kernel = load_inline(
            name="matmul_tiled_kernel_module",
            cpp_sources="""
            #include <torch/extension.h>
            #include <vector>

            // Forward declaration of the CUDA kernel
            void matmul_tiled_kernel_launcher(
                float* C, const float* A, const float* B, int N,
                dim3 grid_dim, dim3 block_dim
            );

            torch::Tensor matmul_tiled_forward(
                torch::Tensor A, torch::Tensor B
            ) {
                // Ensure inputs are contiguous and on CUDA device
                A = A.contiguous();
                B = B.contiguous();

                int N = A.size(0);
                torch::Tensor C = torch::zeros({N, N}, A.options());

                const int TILE_SIZE = 32;
                dim3 block_dim(TILE_SIZE, TILE_SIZE);
                dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

                matmul_tiled_kernel_launcher(
                    C.data_ptr<float>(),
                    A.data_ptr<float>(),
                    B.data_ptr<float>(),
                    N,
                    grid_dim,
                    block_dim
                );

                return C;
            }
            """,
            cuda_sources=cuda_source,
            functions=["matmul_tiled_forward"],
            with_cuda=True,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            build_directory="/tmp/matmul_tiled_kernel_build",  # Specify a build directory
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        if not (A.is_cuda and B.is_cuda):
            raise ValueError("Inputs must be CUDA tensors.")
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            raise ValueError("Inputs must be float32 tensors.")
        if A.shape != B.shape or len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Inputs must be square matrices of the same size.")

        return self.matmul_kernel.matmul_tiled_forward(A, B)
