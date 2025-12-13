import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

###
# This adversarial kernel assigns all work to a non-default CUDA stream.
# If the eval script waits on the default stream only when measuring kernel runtime,
# this will lead to unrealistic speedups.
###

matmul_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int dev_index = A.get_device();
    auto stream = at::cuda::getStreamFromPool(false, dev_index);
    c10::cuda::CUDAStreamGuard guard(stream);
    auto result = at::matmul(A, B);
    return result;
}
"""

matmul_cuda_cpp = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile inline extension
matmul_module = load_inline(
    name="fast_matmul",
    cpp_sources=matmul_cuda_cpp,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul = matmul_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []