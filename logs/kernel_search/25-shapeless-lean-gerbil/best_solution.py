import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for tiled matrix multiplication
# This kernel computes C = A * B using a tiled approach with shared memory.
# Each thread block computes a BLOCK_SIZE x BLOCK_SIZE tile of the output matrix C.
# Each thread within a block computes one element of the C tile by accumulating
# products from corresponding tiles of A and B loaded into shared memory.
cuda_kernel = """
#define BLOCK_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Determine the row and column of the C tile being computed by this block.
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Determine the row and column of the thread within its block.
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Calculate the global row and column index for the element C[globalRow][globalCol]
    // that this thread is responsible for computing.
    int globalRow = blockRow * BLOCK_SIZE + row;
    int globalCol = blockCol * BLOCK_SIZE + col;

    // Accumulator for the C[globalRow][globalCol] element.
    float Cvalue = 0.0f;

    // Declare shared memory for tiles of A and B.
    // These tiles will be loaded from global memory once per iteration over the K dimension
    // and then reused by all threads in the block.
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over the K dimension in steps of BLOCK_SIZE.
    // Each iteration processes a "strip" of A and B that contributes to the C tile.
    for (int k_tile = 0; k_tile < N / BLOCK_SIZE; ++k_tile) {
        // Load the current tiles of A and B from global memory into shared memory.
        // Each thread loads one element.
        // sA[row][col] gets A[globalRow][current_k_col_in_A_tile]
        // sB[row][col] gets B[current_k_row_in_B_tile][globalCol]
        sA[row][col] = A[globalRow * N + (k_tile * BLOCK_SIZE + col)];
        sB[row][col] = B[(k_tile * BLOCK_SIZE + row) * N + globalCol];

        // Synchronize threads to ensure all shared memory loads are complete
        // before any thread starts using the shared data.
        __syncthreads();

        // Perform the dot product for the current tiles.
        // Each thread computes its part of the Cvalue using the shared A and B tiles.
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            Cvalue += sA[row][i] * sB[i][col];
        }

        // Synchronize threads to ensure all computations for the current tile
        // are finished before loading the next tile into shared memory.
        __syncthreads();
    }

    // Store the final accumulated result to global memory, if within bounds.
    // (The bounds check is technically not needed if N is a multiple of BLOCK_SIZE,
    // but is good practice for robustness).
    if (globalRow < N && globalCol < N) {
        C[globalRow * N + globalCol] = Cvalue;
    }
}
"""


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom CUDA kernel with tiled shared memory approach.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Compile the CUDA kernel using torch.utils.cpp_extension.load_inline
        self.custom_matmul_cuda = load_inline(
            name="custom_matmul_cuda",
            cpp_sources="""
            #include <torch/extension.h>
            #include <vector>

            // Forward declaration of the CUDA kernel
            void matmul_kernel(const float* A, const float* B, float* C, int N);

            // C++ wrapper function to call the CUDA kernel
            torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
                // Ensure inputs are contiguous in memory and reside on the CUDA device.
                // This is crucial for direct pointer access in CUDA kernels.
                A = A.contiguous();
                B = B.contiguous();

                // Get matrix dimension N (assuming square matrices A(N,N) and B(N,N)).
                int N = A.size(0);
                
                // Create an output tensor C with the same dimensions and options (dtype, device) as A.
                torch::Tensor C = torch::empty({N, N}, A.options());

                // Define BLOCK_SIZE, must match the kernel's #define.
                const int BLOCK_SIZE = 32;
                
                // Calculate grid and block dimensions for the kernel launch.
                // Each block will handle a BLOCK_SIZE x BLOCK_SIZE tile of C.
                dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
                dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

                // Launch the CUDA kernel.
                matmul_kernel<<<dimGrid, dimBlock>>>(
                    A.data_ptr<float>(), // Pointer to the data of matrix A
                    B.data_ptr<float>(), // Pointer to the data of matrix B
                    C.data_ptr<float>(), // Pointer to the data of output matrix C
                    N                    // Matrix dimension N
                );

                // Check for any CUDA errors that might have occurred during kernel execution.
                AT_CUDA_CHECK(cudaGetLastError());

                return C;
            }
            """,
            cuda_sources=cuda_kernel,
            functions=["matmul_forward"],
            with_cuda=True,
            # Aggressive optimization flags for the CUDA compiler
            extra_cuda_cflags=["-O3", "--use_fast_math"],
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
        # Ensure input tensors are on the CUDA device before passing them to the kernel.
        A = A.cuda()
        B = B.cuda()

        # Call the compiled custom CUDA matrix multiplication function.
        return self.custom_matmul_cuda.matmul_forward(A, B)
