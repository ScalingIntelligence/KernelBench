import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_d, const int in_h, const int in_w,
    const int out_channels,
    const int kD, const int kH, const int kW,
    const int out_d, const int out_h, const int out_w
) {
    // Each thread corresponds to one element in the output tensor
    int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elems = batch_size * out_channels * out_d * out_h * out_w;
    if (out_index >= total_out_elems) return;

    // Compute indices in the output tensor (b, c_out, d, h, w)
    int w_out = out_index % out_w;
    int temp_idx = out_index / out_w;
    int h_out = temp_idx % out_h;
    temp_idx /= out_h;
    int d_out = temp_idx % out_d;
    temp_idx /= out_d;
    int c_out = temp_idx % out_channels;
    int b = temp_idx / out_channels;

    // Compute starting position in input
    float value = 0.0;
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int d_in = d_out + kd;
                    int h_in = h_out + kh;
                    int w_in = w_out + kw;
                    int in_idx = b * in_channels * in_d * in_h * in_w
                                 + c_in * in_d * in_h * in_w
                                 + d_in * in_h * in_w
                                 + h_in * in_w
                                 + w_in;
                    int wt_idx = c_out * in_channels * kD * kH * kW
                                 + c_in * kD * kH * kW
                                 + kd * kH * kW
                                 + kh * kW
                                 + kw;
                    value += input[in_idx] * weight[wt_idx];
                }
            }
        }
    }
    // Add bias
    value += bias[c_out];
    output[out_index] = value;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    // Shapes
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_d = input.size(2);
    const int in_h = input.size(3);
    const int in_w = input.size(4);

    const int out_channels = weight.size(0);
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    // Assuming stride=1, padding=0, dilation=1
    const int out_d = in_d - kD + 1;
    const int out_h = in_h - kH + 1;
    const int out_w = in_w - kW + 1;

    auto output = torch::empty(
        {batch_size, out_channels, out_d, out_h, out_w},
        input.options()
    );

    int total_out_elems = batch_size * out_channels * out_d * out_h * out_w;
    const int block_size = 256;
    const int grid_size = (total_out_elems + block_size - 1) / block_size;

    conv3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_d, in_h, in_w,
        out_channels,
        kD, kH, kW,
        out_d, out_h, out_w
    );
    // Synchronize to check for errors
    cudaDeviceSynchronize();
    return output;
}
'''

conv3d_cpp_declaration = r'''
torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);
'''

conv3d_operator = load_inline(
    name="conv3d_operator",
    cpp_sources=conv3d_cpp_declaration,
    cuda_sources=conv3d_cuda_source,
    functions=["conv3d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution via a custom CUDA kernel,
    applies Group Normalization, then computes the mean.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        # Replace nn.Conv3d with custom parameters + custom kernel
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = conv3d_operator.conv3d_cuda(x, self.weight, self.bias)
        x = self.group_norm(x)
        x = x.mean(dim=[1, 2, 3, 4])
        return x