import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUSTOM_CUDA_CODE = """
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Determine the global row and column for the current thread's C element
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Declare shared memory for tiles of A and B
    // TILE_SIZE x TILE_SIZE float matrix for A and B
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float Cvalue = 0; // Accumulator for the C element computed by this thread

    // Loop over the "inner" dimension (k-dimension) in tiles
    // Each iteration processes a block of TILE_SIZE elements along the k-dimension
    for (int k_tile_idx = 0; k_tile_idx < (N + TILE_SIZE - 1) / TILE_SIZE; ++k_tile_idx) {
        // Calculate the global k-index for loading from global memory
        int k_global_A = k_tile_idx * TILE_SIZE + threadIdx.x; // Column index for A
        int k_global_B = k_tile_idx * TILE_SIZE + threadIdx.y; // Row index for B

        // Load a tile of A into shared memory (sA)
        // Each thread loads one element: sA[threadIdx.y][threadIdx.x] corresponds to A[row][k_global_A]
        if (row < N && k_global_A < N) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + k_global_A];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f; // Pad with zeros if out of bounds
        }

        // Load a tile of B into shared memory (sB)
        // Each thread loads one element: sB[threadIdx.y][threadIdx.x] corresponds to B[k_global_B][col]
        if (k_global_B < N && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[k_global_B * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f; // Pad with zeros if out of bounds
        }
        
        // Synchronize all threads in the block to ensure both shared memory tiles (sA, sB) are fully loaded
        // before any thread starts using them for computation.
        __syncthreads();

        // Perform the dot product for the current tile
        // Each thread computes one element of the C tile using the loaded shared memory tiles
        for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
            Cvalue += sA[threadIdx.y][k_local] * sB[k_local][threadIdx.x];
        }
        
        // Synchronize all threads in the block before loading the next tile from global memory.
        // This ensures that all computations using the current shared memory tiles are complete
        // before new data overwrites them in the next k_tile_idx iteration.
        __syncthreads();
    }

    // Write the accumulated result (Cvalue) to global memory if the current thread's (row, col)
    // is within the bounds of the output matrix C.
    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}
"""


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom CUDA kernel with tiled matrix multiplication.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Compile the CUDA kernel and its C++ wrapper
        self.custom_matmul_kernel_module = load_inline(
            name="custom_matmul_kernel_module",
            cpp_sources=[
                """
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>

            // Forward declaration of the CUDA kernel
            __global__ void matmul_kernel(const float* A, const float* B, float* C, int N);

            torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
                // Ensure inputs are on CUDA and are float32
                TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
                TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
                TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
                TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
                TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
                TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square matrices");
                TORCH_CHECK(A.size(0) == B.size(0), "Input matrices must have the same size");

                const int N = A.size(0);
                // Create output tensor C with the same options as A (device, dtype)
                torch::Tensor C = torch::empty({N, N}, A.options());

                const int TILE_SIZE = 32;
                // Configure thread block dimensions (TILE_SIZE x TILE_SIZE threads)
                dim3 block_dim(TILE_SIZE, TILE_SIZE);
                // Configure grid dimensions (number of blocks needed to cover the N x N matrix)
                dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

                // Launch the CUDA kernel
                matmul_kernel<<<grid_dim, block_dim>>>(
                    A.data_ptr<float>(),
                    B.data_ptr<float>(),
                    C.data_ptr<float>(),
                    N
                );
                // Check for any CUDA errors after kernel launch
                AT_CUDA_CHECK(cudaGetLastError());
                return C;
            }
            """
            ],
            cuda_sources=[CUSTOM_CUDA_CODE],
            functions=["matmul_forward"],
            verbose=False,  # Set to True for debugging compilation issues
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
        # Ensure inputs are on the GPU and are float32, and contiguous for direct pointer access
        A = A.to(device="cuda", dtype=torch.float32).contiguous()
        B = B.to(device="cuda", dtype=torch.float32).contiguous()

        # Call the compiled custom CUDA kernel via its C++ wrapper
        return self.custom_matmul_kernel_module.matmul_forward(A, B)
