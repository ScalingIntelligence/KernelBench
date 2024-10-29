import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Mish activation function
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void mish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float exp_x = expf(x[idx]);
        float exp_2x = expf(2 * x[idx]);
        float exp_3x = expf(3 * x[idx]);
        out[idx] = x[idx] * (1 + exp_x) / (1 + exp_x + exp_2x + exp_3x);
    }
}

torch::Tensor mish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

mish_cpp_source = "torch::Tensor mish_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Mish activation function
mish = load_inline(
    name='mish',
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=['mish_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies custom Mish, and another custom Mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.mish = mish

    def forward(self, x):
        x = self.conv(x)
        x = self.mish.mish_cuda(x)
        x = self.mish.mish_cuda(x)
        return x