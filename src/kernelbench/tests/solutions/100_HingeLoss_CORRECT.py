import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(
    const float* predictions, const float* targets,
    float* result, int batch_size, int inner_size) {
    
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * inner_size;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    
    for (int i = global_idx; i < total_elements; i += stride) {
        int col_idx = i % inner_size;
        float pred = predictions[i];
        float target = targets[col_idx];
        float val = 1.0f - pred * target;
        if (val > 0.0f) {
            sum += val;
        }
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

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int inner_size = predictions.numel() / batch_size;
    int total_elements = predictions.numel();
    
    auto result = torch::zeros({}, predictions.options());
    
    const int block_size = 256;
    int num_blocks = min((total_elements + block_size - 1) / block_size, 1024);
    
    hinge_loss_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(), targets.data_ptr<float>(),
        result.data_ptr<float>(), batch_size, inner_size);
    
    return result / total_elements;
}
"""

hinge_loss_cpp_source = "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

hinge_loss = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss = hinge_loss

    def forward(self, predictions, targets):
        return self.hinge_loss.hinge_loss_cuda(predictions, targets)

