import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution + ReLU
conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_relu_kernel(float* output, const float* input, const float* weight, const float* bias, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int out_h = blockIdx.x;
    int out_w = threadIdx.x;
    int batch = blockIdx.y;
    int out_channel = blockIdx.z;

    if (out_h < height && out_w < width) {
        float sum = 0.0f;
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_h = out_h + kh - kernel_size / 2;
                    int in_w = out_w + kw - kernel_size / 2;
                    if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        sum += input[batch * in_channels * height * width + in_channel * height * width + in_h * width + in_w] * 
                               weight[out_channel * in_channels * kernel_size * kernel_size + in_channel * kernel_size * kernel_size + kh * kernel_size + kw];
                    }
                }
            }
        }
        float val = sum + bias[out_channel];
        output[batch * out_channels * height * width + out_channel * height * width + out_h * width + out_w] = max(0.0f, val);
    }
}

torch::Tensor conv_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    auto output = torch::zeros({batch_size, out_channels, height, width}, torch::kFloat32).cuda();

    const int block_size = 16;
    const dim3 num_blocks(height, batch_size, out_channels);

    conv_relu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), 
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        batch_size, 
        in_channels, 
        out_channels, 
        height, 
        width, 
        kernel_size
    );

    return output;
}
"""

conv_relu_cpp_source = "torch::Tensor conv_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size);"

# Compile the inline CUDA code for convolution + ReLU
conv_relu = load_inline(
    name="conv_relu",
    cpp_sources=conv_relu_cpp_source,
    cuda_sources=conv_relu_source,
    functions=["conv_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.conv_relu = conv_relu

    def forward(self, x):
        batch_size, _, height, width = x.shape
        x = self.conv_relu.conv_relu_cuda(
            x,
            self.weight,
            self.bias,
            batch_size,
            self.weight.shape[1],
            self.weight.shape[0],
            height,
            width,
            self.weight.shape[2],
        )
        x = x + self.bias
        return x
