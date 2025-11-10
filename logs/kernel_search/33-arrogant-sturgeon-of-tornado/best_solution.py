import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define BLOCK_SIZE for the tiled matrix multiplication kernel
BLOCK_SIZE = 32

# CUDA kernel source code for tiled matrix multiplication
cuda_source = f"""
const int BLOCK_SIZE = {BLOCK_SIZE};

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {{
    // Block and thread indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Global row and column for C
    int row = blockRow * BLOCK_SIZE + threadRow;
    int col = blockCol * BLOCK_SIZE + threadCol;

    float Cvalue = 0.0f;

    // Shared memory for A and B sub-matrices (dynamically allocated)
    extern __shared__ float shared_mem[];
    float* As = (float*)shared_mem;
    float* Bs = (float*)(shared_mem + BLOCK_SIZE * BLOCK_SIZE);

    // Loop over the K dimension (N / BLOCK_SIZE stages)
    int num_blocks_k = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int k_block = 0; k_block < num_blocks_k; ++k_block) {{
        // Load sub-matrix of A into shared memory
        // Each thread loads one element from global memory to shared memory
        int A_global_row = blockRow * BLOCK_SIZE + threadRow;
        int A_global_col = k_block * BLOCK_SIZE + threadCol;
        
        // Handle boundary conditions: pad with zeros if out of bounds
        if (A_global_row < N && A_global_col < N) {{
            As[threadRow * BLOCK_SIZE + threadCol] = A[A_global_row * N + A_global_col];
        }} else {{
            As[threadRow * BLOCK_SIZE + threadCol] = 0.0f; 
        }}

        // Load sub-matrix of B into shared memory
        // Each thread loads one element from global memory to shared memory
        int B_global_row = k_block * BLOCK_SIZE + threadRow;
        int B_global_col = blockCol * BLOCK_SIZE + threadCol;
        
        // Handle boundary conditions: pad with zeros if out of bounds
        if (B_global_row < N && B_global_col < N) {{
            Bs[threadRow * BLOCK_SIZE + threadCol] = B[B_global_row * N + B_global_col];
        }} else {{
            Bs[threadRow * BLOCK_SIZE + threadCol] = 0.0f;
        }}

        __syncthreads(); // Wait for all threads to load their data into shared memory

        // Perform the dot product for the current sub-matrices in shared memory
        for (int k = 0; k < BLOCK_SIZE; ++k) {{
            Cvalue += As[threadRow * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + threadCol];
        }}

        __syncthreads(); // Wait for all threads to finish computing with shared data
    }}

    // Store the final result in C to global memory
    if (row < N && col < N) {{
        C[row * N + col] = Cvalue;
    }}
}}
"""

# Compile and load the CUDA kernel
# The 'extra_cuda_cflags' are used for optimization (e.g., -O3 for aggressive optimization)
matmul_module = load_inline(
    name="matmul_extension",
    cuda_sources=[cuda_source],
    functions=["matmul_kernel"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom tiled CUDA kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Get the compiled kernel function from the loaded module
        self.matmul_kernel_func = matmul_module.matmul_kernel

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        # Ensure inputs are on CUDA, float32, 2D, square, and have matching shapes
        if not (A.is_cuda and B.is_cuda):
            raise ValueError("Inputs must be on CUDA.")
        if not (A.dtype == torch.float32 and B.dtype == torch.float32):
            raise ValueError("Inputs must be float32.")
        if not (A.dim() == 2 and B.dim() == 2):
            raise ValueError("Inputs must be 2D matrices.")
        if not (A.shape == B.shape and A.shape[0] == A.shape[1]):
            raise ValueError("Input matrices must be square and have the same shape.")

        N = A.shape[0]
        # Allocate output tensor on the same device as inputs
        C = torch.empty_like(A)

        # Calculate grid and block dimensions for kernel launch
        grid_dim_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_dim_y = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_dim = (grid_dim_x, grid_dim_y)
        block_dim = (BLOCK_SIZE, BLOCK_SIZE)

        # Calculate dynamic shared memory size required by the kernel
        # 2 * BLOCK_SIZE * BLOCK_SIZE floats for As and Bs sub-matrices
        shared_mem_size = (
            2 * BLOCK_SIZE * BLOCK_SIZE * torch.finfo(torch.float32).bits // 8
        )

        # Launch the custom CUDA kernel
        # .contiguous() is used to ensure tensors are in a row-major memory layout,
        # which the C++ kernel expects for optimal performance.
        self.matmul_kernel_func[grid_dim, block_dim, shared_mem_size](
            A.contiguous(), B.contiguous(), C, N
        )

        return C
