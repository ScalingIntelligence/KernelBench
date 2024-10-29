import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for combined matmul and sum operation
matmul_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_sum_kernel(const float* x, const float* weight, float* out, int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < out_features; ++j) {
            for (int i = 0; i < in_features; ++i) {
                sum += x[idx * in_features + i] * weight[j * in_features + i];
            }
        }
        out[idx] = sum;
    }
}

torch::Tensor matmul_sum_cuda(torch::Tensor x, torch::Tensor weight) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);
    auto out = torch::zeros({batch_size, 1}, x.dtype());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    matmul_sum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, out_features);

    return out;
}
"""

matmul_sum_cpp_source = "torch::Tensor matmul_sum_cuda(torch::Tensor x, torch::Tensor weight);"

# Compile the inline CUDA code for matmul_sum
matmul_sum = load_inline(
    name='matmul_sum',
    cpp_sources=matmul_sum_cpp_source,
    cuda_sources=matmul_sum_source,
    functions=['matmul_sum_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for combined max and average pooling
max_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_avg_pool_kernel(const float* x, float* out, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        out[idx] = x[idx]; // assuming input is already reduced to (batch_size, 1)
    }
}

torch::Tensor max_avg_pool_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    max_avg_pool_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size);

    return out;
}
"""

max_avg_pool_cpp_source = "torch::Tensor max_avg_pool_cuda(torch::Tensor x);"

# Compile the inline CUDA code for max_avg_pool
max_avg_pool = load_inline(
    name='max_avg_pool',
    cpp_sources=max_avg_pool_cpp_source,
    cuda_sources=max_avg_pool_source,
    functions=['max_avg_pool_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.matmul_sum = matmul_sum
        self.max_avg_pool = max_avg_pool

    def forward(self, x):
        x = self.matmul_sum.matmul_sum_cuda(x, self.linear.weight.T)
        x = self.max_avg_pool.max_avg_pool_cuda(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        return x