import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code for tiled matrix multiplication
cuda_source = """
#define TILE_SIZE 32

__global__ void matmul_kernel(float* C, const float* A, const float* B, int N) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within the block
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // C sub-matrix element this thread is responsible for
    int C_row = blockRow * TILE_SIZE + threadRow;
    int C_col = blockCol * TILE_SIZE + threadCol;

    // Accumulator for C[C_row][C_col]
    float Pvalue = 0.0f;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over the K dimension (inner product)
    // Each iteration loads a TILE_SIZE x TILE_SIZE sub-matrix of A and B
    // into shared memory, then computes a partial product.
    for (int k_idx = 0; k_idx < N; k_idx += TILE_SIZE) {
        // Load A sub-matrix into shared memory
        // Each thread loads one element of As
        // As[threadRow][threadCol] corresponds to A[C_row][k_idx + threadCol]
        if (C_row < N && (k_idx + threadCol) < N) {
            As[threadRow][threadCol] = A[C_row * N + (k_idx + threadCol)];
        } else {
            As[threadRow][threadCol] = 0.0f; // Handle out-of-bounds with zero padding
        }


        // Load B sub-matrix into shared memory
        // Each thread loads one element of Bs
        // Bs[threadRow][threadCol] corresponds to B[k_idx + threadRow][C_col]
        if ((k_idx + threadRow) < N && C_col < N) {
            Bs[threadRow][threadCol] = B[(k_idx + threadRow) * N + C_col];
        } else {
            Bs[threadRow][threadCol] = 0.0f; // Handle out-of-bounds with zero padding
        }

        // Synchronize to ensure all shared memory loads are complete
        __syncthreads();

        // Perform computation using shared memory
        // Each thread computes a partial sum for Pvalue
        for (int i = 0; i < TILE_SIZE; ++i) {
            Pvalue += As[threadRow][i] * Bs[i][threadCol];
        }

        // Synchronize to ensure all computation using current shared memory tiles is complete
        __syncthreads();
    }

    // Write the final result to global memory, if C_row and C_col are within bounds
    if (C_row < N && C_col < N) {
        C[C_row * N + C_col] = Pvalue;
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
        self.matmul_cuda = load_inline(
            name="matmul_cuda_module",
            cpp_sources="""
            #include <torch/extension.h>
            #include <vector>

            // Forward declaration of the CUDA kernel
            __global__ void matmul_kernel(float* C, const float* A, const float* B, int N);

            torch::Tensor matmul_forward_cuda(
                torch::Tensor A,
                torch::Tensor B) {
                
                // Ensure inputs are on CUDA and are float32
                AT_ASSERTM(A.is_cuda() && B.is_cuda(), "Tensors must be CUDA tensors!");
                AT_ASSERTM(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous!");
                AT_ASSERTM(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Tensors must be float32!");

                int N = A.size(0);
                AT_ASSERTM(A.size(1) == N && B.size(0) == N && B.size(1) == N, "Input matrices must be square and of the same size N x N!");

                // Create output tensor
                torch::Tensor C = torch::empty_like(A);

                // Kernel launch parameters
                const int TILE_SIZE = 32;
                dim3 dimBlock(TILE_SIZE, TILE_SIZE);
                
                // Calculate grid dimensions, ensuring it covers the entire matrix
                dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

                matmul_kernel<<<dimGrid, dimBlock>>>(
                    C.data_ptr<float>(),
                    A.data_ptr<float>(),
                    B.data_ptr<float>(),
                    N
                );

                // Check for CUDA errors
                AT_CUDA_CHECK(cudaGetLastError());

                return C;
            }
            """,
            cuda_sources=cuda_source,
            functions=["matmul_forward_cuda"],
            with_cuda=True,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
            ],  # Add optimization flags for better performance
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using a custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        # Ensure inputs are on the correct device and contiguous for the kernel
        A_cuda = A.to("cuda").contiguous()
        B_cuda = B.to("cuda").contiguous()

        # Call the custom CUDA kernel wrapper
        C_cuda = self.matmul_cuda.matmul_forward_cuda(A_cuda, B_cuda)

        return C_cuda
