import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code for tiled matrix multiplication
cuda_kernel_code = """
#define BLOCK_SIZE 32 // Define tile size for shared memory and thread block dimensions

__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    // Determine the global row and column for the current thread's C element
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Accumulator for the C[row][col] element
    float C_value = 0;

    // Shared memory for tiles of A and B
    // Using BLOCK_SIZE x BLOCK_SIZE for simplicity and typical efficiency.
    // Padding (e.g., BLOCK_SIZE + 1) could be used to prevent bank conflicts,
    // but for BLOCK_SIZE = 32, it's often not strictly necessary as bank conflicts
    // are less likely with this access pattern and modern hardware.
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over the tiles in the K-dimension (inner product dimension)
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE calculates ceil(N / BLOCK_SIZE)
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile_idx = 0; tile_idx < numTiles; ++tile_idx) {
        // Calculate global column index for A and global row index for B
        int k_global = tile_idx * BLOCK_SIZE + threadIdx.x;

        // Load sub-tile of A into shared memory
        // Each thread loads one element. Threads in a row load elements for As[threadIdx.y][threadIdx.x]
        // ensuring coalesced access to global memory A.
        if (row < N && k_global < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + k_global];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f; // Pad with zeros for out-of-bounds access
        }

        // Calculate global row index for B and global column index for A (same k_global logic)
        k_global = tile_idx * BLOCK_SIZE + threadIdx.y; // Different k_global for B loading
                                                        // B needs row-major access to its tile
        // Load sub-tile of B into shared memory
        // Each thread loads one element. Threads in a column load elements for Bs[threadIdx.y][threadIdx.x]
        // ensuring coalesced access to global memory B.
        if (k_global < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[k_global * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f; // Pad with zeros for out-of-bounds access
        }

        // Synchronize threads within the block after loading shared memory tiles
        // This ensures all data is available before computation begins.
        __syncthreads();

        // Perform dot product for the current tile
        // Each thread computes one element of the output tile.
        // It iterates over the K-dimension (BLOCK_SIZE elements) of the shared memory tiles.
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            C_value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize threads within the block before loading the next shared memory tiles
        // This ensures all computation for the current tile is done before shared memory is overwritten.
        __syncthreads();
    }

    // Write the accumulated result to global memory
    if (row < N && col < N) {
        C[row * N + col] = C_value;
    }
}
"""

# Define the C++ wrapper code to interface with PyTorch
cpp_wrapper_code = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare the CUDA kernel (must match the signature in cuda_kernel_code)
__global__ void matmul_kernel(float* A, float* B, float* C, int N);

// C++ function that will be exposed to Python
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Matrices must be 2D.");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square for this kernel.");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match for matrix multiplication.");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors.");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32.");

    int N = A.size(0); // Assuming square matrices N x N

    // Allocate the output tensor C on the GPU with the same options (device, dtype) as A
    torch::Tensor C = torch::empty({N, N}, A.options());

    // Get raw pointers to the tensor data
    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Define block and grid dimensions for the kernel launch
    const int BLOCK_SIZE = 32; // Must match the #define in the CUDA kernel
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Launch the CUDA kernel
    matmul_kernel<<<dimGrid, dimBlock>>>(A_ptr, B_ptr, C_ptr, N);

    // Optional: Check for CUDA errors. Good for debugging.
    // cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error in matmul_forward: " << cudaGetErrorString(err) << std::endl;
    // }

    return C;
}
"""


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom CUDA kernel with a tiled shared memory approach.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Compile the CUDA kernel and C++ wrapper only once during initialization.
        # This creates a Python callable `custom_matmul_op` that executes our CUDA code.
        self.custom_matmul_op = load_inline(
            name="custom_matmul_module",
            cpp_sources=cpp_wrapper_code,
            cuda_sources=cuda_kernel_code,
            functions=["matmul_forward"],  # Name of the C++ function to expose
            with_cuda=True,
            extra_cuda_cflags=["-O3", "--use_fast_math"],  # Optimization flags for nvcc
        ).matmul_forward

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        # Ensure inputs are on the CUDA device and are contiguous.
        # `contiguous()` is important for ensuring proper memory layout for raw pointer access.
        # `torch.rand` typically produces contiguous tensors, but it's good practice to ensure.
        A_cuda = A.contiguous().cuda()
        B_cuda = B.contiguous().cuda()

        # Call the compiled custom CUDA operation
        return self.custom_matmul_op(A_cuda, B_cuda)
