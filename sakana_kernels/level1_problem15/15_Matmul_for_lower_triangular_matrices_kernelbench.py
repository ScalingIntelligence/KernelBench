import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

forward_kernel_cuda_src = """
# include <torch/extension.h>
# include <cuda.h>
# include <cuda_runtime.h>
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int N) {
    // Use 2D block configuration for better occupancy
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        if (col <= row) {
            // Lower triangle computation
            float sum = 0.0f;
            // Process elements in chunks to improve cache utilization
            # pragma unroll 8
            for (int k = col; k <= row; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        } else {
            // Upper triangle (set to zero)
            C[row * N + col] = 0.0f;
        }
    }
}

torch::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor ");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor ");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor ");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square ");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square ");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size ");
    int N = A.size(0);
    auto C = torch::empty_like(A);
    // Optimize thread count based on matrix size
    const int threadsPerBlock = 256; // Increased thread count per block
    const int numBlocks = N;
    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return C;
}
"""


forward_kernel_cpp_src = """
torch::Tensor forward(
    torch::Tensor A,
    torch::Tensor B
);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=forward_kernel_cpp_src,
    cuda_sources=forward_kernel_cuda_src,
    functions=["forward"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return fused_ops.forward(A, B)
