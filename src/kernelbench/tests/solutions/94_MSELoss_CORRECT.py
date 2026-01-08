import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mse_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_loss_kernel(
    const float* predictions, const float* targets,
    float* result, int total_elements) {
    
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    
    for (int i = global_idx; i < total_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        sum += diff * diff;
    }
    
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int total_elements = predictions.numel();
    
    auto result = torch::zeros({}, predictions.options());
    
    const int block_size = 256;
    int num_blocks = min((total_elements + block_size - 1) / block_size, 1024);
    
    mse_loss_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(), targets.data_ptr<float>(),
        result.data_ptr<float>(), total_elements);
    
    return result / total_elements;
}
"""

mse_loss_cpp_source = "torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

mse_loss = load_inline(
    name="mse_loss",
    cpp_sources=mse_loss_cpp_source,
    cuda_sources=mse_loss_source,
    functions=["mse_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse_loss = mse_loss

    def forward(self, predictions, targets):
        return self.mse_loss.mse_loss_cuda(predictions, targets)
