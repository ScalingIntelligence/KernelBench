import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matmul+sum+max
matmul_sum_max_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_sum_max_kernel(const float* a, const float* b, float* out, int batch_size, int in_features, int out_features) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        float max_val = -INFINITY;
        for (int i = 0; i < out_features; i++) {
            float sum = 0;
            for (int j = 0; j < in_features; j++) {
                sum += a[batch_idx * in_features + j] * b[i * in_features + j];
            }
            max_val = max(max_val, sum);
        }
        out[batch_idx] = max_val;
    }
}

torch::Tensor matmul_sum_max_cuda(torch::Tensor a, torch::Tensor b, int batch_size, int in_features, int out_features) {
    auto out = torch::zeros({batch_size}, dtype=torch::float32, device=a.device);
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    matmul_sum_max_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, out_features);

    return out.view(-1, 1);
}
"""

matmul_sum_max_cpp_source = "torch::Tensor matmul_sum_max_cuda(torch::Tensor a, torch::Tensor b, int batch_size, int in_features, int out_features);"

# Compile the inline CUDA code for matmul+sum+max
matmul_sum_max = load_inline(
    name='matmul_sum_max',
    cpp_sources=matmul_sum_max_cpp_source,
    cuda_sources=matmul_sum_max_source,
    functions=['matmul_sum_max_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for log_sum_exp
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void logsumexp_kernel(const float* a, float* out, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        float max_val = a[batch_idx];
        float sum = 0;
        for (int i = 0; i < batch_size; i++) {
            sum += exp(a[i] - max_val);
        }
        out[batch_idx] = log(sum) + max_val;
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor a, int batch_size) {
    auto out = torch::zeros({batch_size}, dtype=torch::float32, device=a.device);
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    logsumexp_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), batch_size);

    return out.view(-1, 1);
}
"""

logsumexp_cpp_source = "torch::Tensor logsumexp_cuda(torch::Tensor a, int batch_size);"

# Compile the inline CUDA code for log_sum_exp
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
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.matmul_sum_max = matmul_sum_max
        self.logsumexp = logsumexp

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.matmul_sum_max.matmul_sum_max_cuda(x, self.linear.weight.T, x.shape[0], x.shape[1], self.linear.weight.shape[0])
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.logsumexp.logsumexp_cuda(x.squeeze(), x.shape[0])
        x = self.logsumexp.logsumexp_cuda(x.squeeze(), x.shape[0])
        return x