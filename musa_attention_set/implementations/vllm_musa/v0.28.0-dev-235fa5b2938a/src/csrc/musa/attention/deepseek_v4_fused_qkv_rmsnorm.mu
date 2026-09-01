#include <cstdint>
#include <limits>
#include <tuple>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace {

constexpr int kBlock = 256;

template <int BLOCK_X>
__device__ __forceinline__ float block_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  if constexpr (BLOCK_X <= 32) {
    return value;
  }

  __shared__ float shared[BLOCK_X / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < (BLOCK_X / 32) ? shared[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int offset = (BLOCK_X / 32) >> 1; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    if (threadIdx.x == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ float load_float(float value) {
  return value;
}

__device__ __forceinline__ float load_float(__half value) {
  return __half2float(value);
}

__device__ __forceinline__ float load_float(__mt_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T store_float(float value);

template <>
__device__ __forceinline__ __half store_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __mt_bfloat16 store_float<__mt_bfloat16>(
    float value) {
  return __float2bfloat16(value);
}

template <typename T, typename W>
__global__ void deepseek_v4_fused_q_kv_rmsnorm_kernel(
    const T* __restrict__ q, T* __restrict__ q_out,
    const W* __restrict__ q_weight, int64_t q_in_stride,
    int64_t q_out_stride, const T* __restrict__ kv,
    T* __restrict__ kv_out, const W* __restrict__ kv_weight,
    int64_t kv_in_stride, int64_t kv_out_stride, float eps,
    int64_t num_tokens, int64_t q_size, int64_t kv_size) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const int task = blockIdx.y;
  if (token >= num_tokens) {
    return;
  }

  const T* row_in;
  T* row_out;
  const W* weight;
  int64_t size;
  if (task == 0) {
    row_in = q + token * q_in_stride;
    row_out = q_out + token * q_out_stride;
    weight = q_weight;
    size = q_size;
  } else {
    row_in = kv + token * kv_in_stride;
    row_out = kv_out + token * kv_out_stride;
    weight = kv_weight;
    size = kv_size;
  }

  float partial = 0.0f;
  for (int64_t dim = threadIdx.x; dim < size; dim += blockDim.x) {
    const float value = load_float(row_in[dim]);
    partial += value * value;
  }

  const float total = block_reduce_sum<kBlock>(partial);
  const float inv_rms = rsqrtf(total / static_cast<float>(size) + eps);

  for (int64_t dim = threadIdx.x; dim < size; dim += blockDim.x) {
    const float value =
        load_float(row_in[dim]) * inv_rms * load_float(weight[dim]);
    row_out[dim] = store_float<T>(value);
  }
}

template <typename T, typename W>
void launch_deepseek_v4_fused_q_kv_rmsnorm(
    const torch::Tensor& q, torch::Tensor& q_out,
    const torch::Tensor& q_weight, const torch::Tensor& kv,
    torch::Tensor& kv_out, const torch::Tensor& kv_weight, double eps,
    musaStream_t stream) {
  const int64_t num_tokens = q.size(0);
  if (num_tokens == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(num_tokens), 2);
  const dim3 block(kBlock);
  deepseek_v4_fused_q_kv_rmsnorm_kernel<T, W><<<grid, block, 0, stream>>>(
      static_cast<const T*>(q.data_ptr()), static_cast<T*>(q_out.data_ptr()),
      static_cast<const W*>(q_weight.data_ptr()),
      static_cast<int64_t>(q.stride(0)),
      static_cast<int64_t>(q_out.stride(0)),
      static_cast<const T*>(kv.data_ptr()), static_cast<T*>(kv_out.data_ptr()),
      static_cast<const W*>(kv_weight.data_ptr()),
      static_cast<int64_t>(kv.stride(0)),
      static_cast<int64_t>(kv_out.stride(0)), static_cast<float>(eps),
      num_tokens, static_cast<int64_t>(q.size(1)),
      static_cast<int64_t>(kv.size(1)));
}

template <typename T>
void dispatch_weight_dtype(const torch::Tensor& q, torch::Tensor& q_out,
                           const torch::Tensor& q_weight,
                           const torch::Tensor& kv, torch::Tensor& kv_out,
                           const torch::Tensor& kv_weight, double eps,
                           musaStream_t stream) {
  if (q_weight.scalar_type() == at::ScalarType::Float) {
    launch_deepseek_v4_fused_q_kv_rmsnorm<T, float>(
        q, q_out, q_weight, kv, kv_out, kv_weight, eps, stream);
  } else if (q_weight.scalar_type() == at::ScalarType::Half) {
    launch_deepseek_v4_fused_q_kv_rmsnorm<T, __half>(
        q, q_out, q_weight, kv, kv_out, kv_weight, eps, stream);
  } else if (q_weight.scalar_type() == at::ScalarType::BFloat16) {
    launch_deepseek_v4_fused_q_kv_rmsnorm<T, __mt_bfloat16>(
        q, q_out, q_weight, kv, kv_out, kv_weight, eps, stream);
  } else {
    TORCH_CHECK(false, "q_weight/kv_weight must be float, fp16, or bf16");
  }
}

void check_inputs(const torch::Tensor& q, const torch::Tensor& kv,
                  const torch::Tensor& q_weight,
                  const torch::Tensor& kv_weight) {
  TORCH_CHECK(q.device().is_privateuseone(), "q must be a MUSA tensor");
  TORCH_CHECK(kv.device().is_privateuseone(), "kv must be a MUSA tensor");
  TORCH_CHECK(q_weight.device().is_privateuseone(),
              "q_weight must be a MUSA tensor");
  TORCH_CHECK(kv_weight.device().is_privateuseone(),
              "kv_weight must be a MUSA tensor");
  TORCH_CHECK(q.dim() == 2, "q must be 2-D");
  TORCH_CHECK(kv.dim() == 2, "kv must be 2-D");
  TORCH_CHECK(q.size(0) == kv.size(0), "q/kv token dimension mismatch");
  TORCH_CHECK(q.stride(1) == 1, "q last dimension must be contiguous");
  TORCH_CHECK(kv.stride(1) == 1, "kv last dimension must be contiguous");
  TORCH_CHECK(q_weight.dim() == 1, "q_weight must be 1-D");
  TORCH_CHECK(kv_weight.dim() == 1, "kv_weight must be 1-D");
  TORCH_CHECK(q_weight.is_contiguous(), "q_weight must be contiguous");
  TORCH_CHECK(kv_weight.is_contiguous(), "kv_weight must be contiguous");
  TORCH_CHECK(q_weight.size(0) == q.size(1), "q_weight size mismatch");
  TORCH_CHECK(kv_weight.size(0) == kv.size(1), "kv_weight size mismatch");
  TORCH_CHECK(q.scalar_type() == kv.scalar_type(), "q/kv dtype mismatch");
  TORCH_CHECK(q_weight.scalar_type() == kv_weight.scalar_type(),
              "q_weight/kv_weight dtype mismatch");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Half ||
                  q.scalar_type() == at::ScalarType::BFloat16,
              "q/kv must be fp16 or bf16");
  TORCH_CHECK(q.size(0) <= std::numeric_limits<int>::max(),
              "num_tokens exceeds launch grid limit");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_fused_q_kv_rmsnorm(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& q_weight, const torch::Tensor& kv_weight,
    double eps) {
  check_inputs(q, kv, q_weight, kv_weight);

  auto q_out = torch::empty_like(q);
  auto kv_out = torch::empty_like(kv);
  const c10::musa::OptionalMUSAGuard guard(device_of(q));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();

  if (q.scalar_type() == at::ScalarType::Half) {
    dispatch_weight_dtype<__half>(
        q, q_out, q_weight, kv, kv_out, kv_weight, eps, stream);
  } else {
    dispatch_weight_dtype<__mt_bfloat16>(
        q, q_out, q_weight, kv, kv_out, kv_weight, eps, stream);
  }

  const musaError_t err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_fused_q_kv_rmsnorm launch failed: ",
              musaGetErrorString(err));
  return {q_out, kv_out};
}
