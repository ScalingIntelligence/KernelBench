import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());
    
    const int block_size = 16;
    const int num_blocks_x = (N + block_size - 1) / block_size;
    const int num_blocks_y = (M + block_size - 1) / block_size;

    matmul_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
    );

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name='matmul',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=['matmul_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) using a custom CUDA kernel
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using custom CUDA kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul.matmul_cuda(A, B)

M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(M, K).cuda()
    B = torch.randn(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed