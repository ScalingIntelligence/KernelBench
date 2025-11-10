import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source (includes both kernel and C++ wrapper)
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define the tile size for shared memory matrix multiplication
#define TILE_SIZE 32

// CUDA kernel for square matrix multiplication using shared memory tiling
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Shared memory for tiles of A and B
    // TILE_SIZE x TILE_SIZE floats = 32 * 32 * 4 bytes = 4096 bytes per shared array.
    // Total shared memory per block = 2 * 4096 = 8192 bytes. This is well within typical limits (e.g., 48KB).
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    // Global row and column index for the current thread
    // These determine the element of C this thread is responsible for.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0.0f; // Accumulator for C[row][col]

    // Loop over the 'inner' dimension (k) in steps of TILE_SIZE
    // Each iteration processes one pair of tiles from A and B
    for (int k_idx = 0; k_idx < N; k_idx += TILE_SIZE) {
        // Load A tile into shared memory
        // Each thread loads one element of s_A.
        // s_A[threadIdx.y][threadIdx.x] will hold A[row][k_idx + threadIdx.x]
        // The condition (k_idx + threadIdx.x) < N handles potential padding if N is not a multiple of TILE_SIZE.
        if (row < N && (k_idx + threadIdx.x) < N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * N + (k_idx + threadIdx.x)];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f; // Pad with zeros for out-of-bounds reads
        }

        // Load B tile into shared memory
        // Each thread loads one element of s_B.
        // s_B[threadIdx.y][threadIdx.x] will hold B[k_idx + threadIdx.y][col]
        // The condition (k_idx + threadIdx.y) < N handles potential padding if N is not a multiple of TILE_SIZE.
        if (col < N && (k_idx + threadIdx.y) < N) {
            s_B[threadIdx.y][threadIdx.x] = B[(k_idx + threadIdx.y) * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f; // Pad with zeros for out-of-bounds reads
        }

        __syncthreads(); // Wait for all threads in the block to load their tile elements into shared memory

        // Perform the dot product for the current tiles
        // Each thread computes its part of Cvalue using the shared memory tiles.
        for (int i = 0; i < TILE_SIZE; ++i) {
            Cvalue += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads(); // Wait for all threads to finish computing with shared memory before loading next tiles
    }

    // Store the computed value to global memory
    // No boundary check needed for C[row][col] as grid_size ensures row < N and col < N
    // for threads that actually write to C.
    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor.");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32.");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32.");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices.");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions mismatch for multiplication.");
    TORCH_CHECK(A.size(0) == B.size(1), "This kernel currently supports only square matrices of same size N x N.");

    int N = A.size(0); // Assuming square matrices N x N
    TORCH_CHECK(A.size(1) == N && B.size(0) == N && B.size(1) == N, "Input matrices must be square and of the same size N x N.");

    // Allocate output tensor C on the same device as A.
    auto C = torch::empty({N, N}, A.options());

    // Define grid and block dimensions for the kernel launch
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    // Calculate number of blocks needed to cover the N x N output matrix
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    // Launch the CUDA kernel
    matmul_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(), // Pointer to A's data
        B.data_ptr<float>(), // Pointer to B's data
        C.data_ptr<float>(), // Pointer to C's data
        N                    // Size of the matrix
    );

    // Check for any CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in matmul_kernel: %s\\n", cudaGetErrorString(err));
        TORCH_CHECK(false, "CUDA kernel launch failed.");
    }

    return C;
}
"""

# C++ function declaration for load_inline
cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code extension
matmul_extension = load_inline(
    name="matmul_extension",  # Unique name for the compiled extension
    cpp_sources=cpp_source,  # C++ function declarations
    cuda_sources=cuda_source,  # CUDA kernel implementation and C++ wrapper
    functions=["matmul_cuda"],  # List of C++ functions to expose to Python
    verbose=True,  # Enable verbose output during compilation
)


class ModelNew(nn.Module):
    """
    Optimized model that replaces torch.matmul with a custom tiled CUDA kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Store the compiled CUDA extension's matmul_cuda function
        self.matmul_op = matmul_extension.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        # Call the custom CUDA kernel through the loaded extension
        return self.matmul_op(A, B)
