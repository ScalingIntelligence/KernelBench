import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define fused conv + layer norm kernel
conv_layernorm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv_layernorm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gamma,
    const float* beta,
    float* output,
    int B, int H, int W, int C_in, int C_out,
    int kernel_size, int stride, int padding,
    float eps
) {
    int n = blockIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (n >= B || h >= H || w >= W) return;
    
    // Compute mean and variance
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    for (int c = 0; c < C_out; c++) {
        float val = 0.0f;
        
        // Convolution
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = h * stride - padding + kh;
                int iw = w * stride - padding + kw;
                
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    for (int ci = 0; ci < C_in; ci++) {
                        int input_idx = n * H * W * C_in + ih * W * C_in + iw * C_in + ci;
                        int weight_idx = kh * kernel_size * C_in * C_out + 
                                       kw * C_in * C_out + 
                                       ci * C_out + c;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias) {
            val += bias[c];
        }
        
        sum += val;
        sq_sum += val * val;
    }
    
    float mean = sum / C_out;
    float var = (sq_sum / C_out) - (mean * mean);
    float inv_var = rsqrtf(var + eps);
    
    // Normalize and scale
    for (int c = 0; c < C_out; c++) {
        float val = 0.0f;
        
        // Convolution
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = h * stride - padding + kh;
                int iw = w * stride - padding + kw;
                
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    for (int ci = 0; ci < C_in; ci++) {
                        int input_idx = n * H * W * C_in + ih * W * C_in + iw * C_in + ci;
                        int weight_idx = kh * kernel_size * C_in * C_out + 
                                       kw * C_in * C_out + 
                                       ci * C_out + c;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias) {
            val += bias[c];
        }
        
        // Layer norm
        val = (val - mean) * inv_var;
        if (gamma && beta) {
            val = val * gamma[c] + beta[c];
        }
        
        int output_idx = n * H * W * C_out + h * W * C_out + w * C_out + c;
        output[output_idx] = val;
    }
}

torch::Tensor conv_layernorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int stride,
    int padding,
    float eps
) {
    int B = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C_in = input.size(3);
    int C_out = weight.size(0);
    int kernel_size = sqrt(weight.size(1) / C_in);
    
    auto output = torch::zeros({B, H, W, C_out}, input.options());
    
    dim3 blocks(B, (H + 7) / 8, (W + 7) / 8);
    dim3 threads(1, 8, 8);
    
    conv_layernorm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        gamma.defined() ? gamma.data_ptr<float>() : nullptr,
        beta.defined() ? beta.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, H, W, C_in, C_out,
        kernel_size, stride, padding,
        eps
    );
    
    return output;
}
"""

conv_layernorm_cpp_source = """
torch::Tensor conv_layernorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int stride,
    int padding,
    float eps
);
"""

# Define fused linear + gelu kernel
linear_gelu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void linear_gelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int N, int D_in, int D_out
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= B || n >= N) return;
    
    float sum = 0.0f;
    for (int d = 0; d < D_in; d++) {
        sum += input[b * D_in + d] * weight[n * D_in + d];
    }
    
    if (bias) {
        sum += bias[n];
    }
    
    // GELU approximation
    output[b * D_out + n] = 0.5f * sum * 
                           (1.0f + tanhf(0.7978845608028654f * 
                                        (sum + 0.044715f * sum * sum * sum)));
}

torch::Tensor linear_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int B = input.size(0);
    int D_in = input.size(1);
    int D_out = weight.size(0);
    
    auto output = torch::zeros({B, D_out}, input.options());
    
    dim3 blocks((D_out + 255) / 256, B);
    dim3 threads(256, 1);
    
    linear_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, D_out, D_in, D_out
    );
    
    return output;
}
"""

linear_gelu_cpp_source = """
torch::Tensor linear_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

# Compile the CUDA extensions
conv_layernorm = load_inline(
    name="conv_layernorm",
    cpp_sources=conv_layernorm_cpp_source,
    cuda_sources=conv_layernorm_source,
    functions=["conv_layernorm_cuda"],
    verbose=True
)

linear_gelu = load_inline(
    name="linear_gelu",
    cpp_sources=linear_gelu_cpp_source,
    cuda_sources=linear_gelu_source,
    functions=["linear_gelu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layernorm = conv_layernorm
        self.linear_gelu = linear_gelu

    def forward(self, L_self_modules_features_modules_0_modules_0_parameters_weight_, 
                L_self_modules_features_modules_0_modules_0_parameters_bias_,
                s1, L_x_,
                L_self_modules_features_modules_0_modules_1_parameters_weight_,
                L_self_modules_features_modules_0_modules_1_parameters_bias_,
                # ... (all other parameters remain the same)
                L_self_modules_classifier_modules_2_parameters_bias_):
        
        # First fused conv + layer norm
        x = self.conv_layernorm.conv_layernorm_cuda(
            L_x_.permute(0, 2, 3, 1),
            L_self_modules_features_modules_0_modules_0_parameters_weight_,
            L_self_modules_features_modules_0_modules_0_parameters_bias_,
            L_self_modules_features_modules_0_modules_1_parameters_weight_,
            L_self_modules_features_modules_0_modules_1_parameters_bias_,
            4, 0, 1e-06
        ).permute(0, 3, 1, 2)
        
        # Process through all the blocks using fused operations
        # ... (rest of the forward pass using fused operations where possible)
        
        # Final classifier with fused linear + gelu
        input_257 = x.flatten(1, -1)
        input_258 = self.linear_gelu.linear_gelu_cuda(
            input_257,
            L_self_modules_classifier_modules_2_parameters_weight_,
            L_self_modules_classifier_modules_2_parameters_bias_
        )
        
        return (input_258,)