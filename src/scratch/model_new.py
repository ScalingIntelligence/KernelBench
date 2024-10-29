import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matmul + relu
matmul_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_relu_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = max(sum, 0.0f);
    }
}

torch::Tensor matmul_relu_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    const int block_size = 16;
    const int num_blocks_x = (M + block_size - 1) / block_size;
    const int num_blocks_y = (N + block_size - 1) / block_size;

    matmul_relu_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matmul_relu_cpp_source = "torch::Tensor matmul_relu_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matmul + relu
matmul_relu = load_inline(
    name='matmul_relu',
    cpp_sources=matmul_relu_cpp_source,
    cuda_sources=matmul_relu_source,
    functions=['matmul_relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for sum + max + mean
sum_max_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_max_mean_kernel(const float* x, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = x[i];
        float max_val = x[i];
        for (int j = i + 1; j < N; ++j) {
            sum += x[j];
            max_val = max(max_val, x[j]);
        }
        out[i] = (sum / N) * max_val;
    }
}

torch::Tensor sum_max_mean_cuda(torch::Tensor x) {
    auto N = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    sum_max_mean_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), N);

    return out;
}
"""

sum_max_mean_cpp_source = "torch::Tensor sum_max_mean_cuda(torch::Tensor x);"

# Compile the inline CUDA code for sum + max + mean
sum_max_mean = load_inline(
    name='sum_max_mean',
    cpp_sources=sum_max_mean_cpp_source,
    cuda_sources=sum_max_mean_source,
    functions=['sum_max_mean_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for logsumexp
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void logsumexp_kernel(const float* x, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float max_val = x[i];
        for (int j = i + 1; j < N; ++j) {
            max_val = max(max_val, x[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += exp(x[j] - max_val);
        }
        out[i] = log(sum) + max_val;
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor x) {
    auto N = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    logsumexp_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), N);

    return out;
}
"""

logsumexp_cpp_source = "torch::Tensor logsumexp_cuda(torch::Tensor x);"

# Compile the inline CUDA code for logsumexp
logsumexp = load_inline(
    name='logsumexp',
    cpp_sources=logsumexp_cpp_source,
    cuda_sources=logsumexp_source,
    functions=['logsumexp_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)


class ModelNew(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication + ReLU
        - Summation + Max + Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.matmul_relu = matmul_relu
        self.sum_max_mean = sum_max_mean
        self.logsumexp = logsumexp

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.matmul_relu.matmul_relu_cuda(x, self.linear.weight.t())  # (batch_size, out_features)
        x = self.sum_max_mean.sum_max_mean_cuda(x) # (batch_size, 1)
        x = self.logsumexp.logsumexp_cuda(x) # (batch_size, 1)
        x = self.logsumexp.logsumexp_cuda(x) # (batch_size, 1)
        return x