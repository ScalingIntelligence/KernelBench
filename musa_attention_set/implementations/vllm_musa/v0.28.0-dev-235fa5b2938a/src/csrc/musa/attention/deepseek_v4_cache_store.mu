#include <cmath>
#include <cstdint>

#include <musa_bf16.h>
#include <musa_fp8.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

constexpr int64_t kNopeDim = 448;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kHeadDim = kNopeDim + kRopeDim;
constexpr int64_t kTokenDataBytes = kNopeDim + kRopeDim * 2;
constexpr int64_t kTokenScaleBytes = 8;
constexpr int64_t kQuantBlockSize = 64;
constexpr int kQNormThreads = 256;

constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;

__device__ __forceinline__ int64_t load_index(const void* ptr, int kind,
                                              int64_t idx) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t*>(ptr)[idx]);
  }
  return static_cast<int64_t>(static_cast<const int64_t*>(ptr)[idx]);
}

__device__ __forceinline__ float warp_reduce_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 16));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 8));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 4));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return value;
}


__global__ void deepseek_v4_qnorm_rope_kernel(
    __mt_bfloat16* __restrict__ q, const void* __restrict__ positions,
    int position_kind, const float* __restrict__ cos_sin_cache, float eps,
    int64_t num_tokens, int64_t num_heads) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const int64_t head = static_cast<int64_t>(blockIdx.y);
  if (token >= num_tokens || head >= num_heads) {
    return;
  }

  __shared__ float reduce[kQNormThreads];
  const int tid = threadIdx.x;
  __mt_bfloat16* row = q + (token * num_heads + head) * kHeadDim;

  float partial = 0.0f;
  for (int64_t dim = tid; dim < kHeadDim; dim += blockDim.x) {
    const float value = __bfloat162float(row[dim]);
    partial += value * value;
  }
  reduce[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }

  const float norm = rsqrtf(reduce[0] / static_cast<float>(kHeadDim) + eps);
  for (int64_t dim = tid; dim < kNopeDim; dim += blockDim.x) {
    row[dim] = __float2bfloat16(__bfloat162float(row[dim]) * norm);
  }

  const int64_t pos = load_index(positions, position_kind, token);
  const float* cos_ptr = cos_sin_cache + pos * kRopeDim;
  const float* sin_ptr = cos_ptr + kRopeDim / 2;
  for (int64_t pair = tid; pair < kRopeDim / 2; pair += blockDim.x) {
    const int64_t even_dim = kNopeDim + pair * 2;
    const int64_t odd_dim = even_dim + 1;
    const float even = __bfloat162float(row[even_dim]) * norm;
    const float odd = __bfloat162float(row[odd_dim]) * norm;
    const float c = cos_ptr[pair];
    const float s = sin_ptr[pair];
    row[even_dim] = __float2bfloat16(even * c - odd * s);
    row[odd_dim] = __float2bfloat16(even * s + odd * c);
  }
}

__global__ void deepseek_v4_kv_rope_pack_kernel(
    const __mt_bfloat16* __restrict__ kv, uint8_t* __restrict__ cache,
    const void* __restrict__ slots, int slot_kind,
    const void* __restrict__ positions, int position_kind,
    const float* __restrict__ cos_sin_cache, int64_t num_tokens,
    int64_t num_blocks, int64_t block_size, int64_t block_stride) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  if (token >= num_tokens) {
    return;
  }

  const int64_t slot = load_index(slots, slot_kind, token);
  if (slot < 0 || slot >= num_blocks * block_size) {
    return;
  }

  __shared__ float abs_values[kQuantBlockSize];
  __shared__ int scale_exponents[kTokenScaleBytes];

  const int tid = threadIdx.x;
  const int64_t block_idx = slot / block_size;
  const int64_t pos_in_block = slot - block_idx * block_size;
  uint8_t* block_ptr = cache + block_idx * block_stride;
  uint8_t* token_ptr = block_ptr + pos_in_block * kTokenDataBytes;
  uint8_t* scale_ptr = block_ptr + block_size * kTokenDataBytes +
                       pos_in_block * kTokenScaleBytes;
  const __mt_bfloat16* input = kv + token * kHeadDim;

  for (int qblock = 0; qblock < kNopeDim / kQuantBlockSize; ++qblock) {
    const int64_t start = qblock * kQuantBlockSize;
    if (tid < kQuantBlockSize) {
      const float value = __bfloat162float(input[start + tid]);
      abs_values[tid] = fabsf(value);
    }
    __syncthreads();

    for (int stride = kQuantBlockSize / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        abs_values[tid] = fmaxf(abs_values[tid], abs_values[tid + stride]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float amax = fmaxf(abs_values[0], 1.0e-4f);
      const int exponent =
          static_cast<int>(ceilf(log2f(amax / 448.0f)));
      scale_exponents[qblock] = exponent;
      const int scale_byte = max(0, min(255, exponent + 127));
      scale_ptr[qblock] = static_cast<uint8_t>(scale_byte);
    }
    __syncthreads();

    if (tid < kQuantBlockSize) {
      const float value = __bfloat162float(input[start + tid]);
      const float scaled = fminf(fmaxf(value * exp2f(-scale_exponents[qblock]),
                                      -448.0f),
                                448.0f);
      const __mt_fp8_e4m3 packed(scaled);
      token_ptr[start + tid] = packed.__x;
    }
    __syncthreads();
  }

  if (tid == 0) {
    scale_ptr[kTokenScaleBytes - 1] = 0;
  }

  const int64_t pos = load_index(positions, position_kind, token);
  const float* cos_ptr = cos_sin_cache + pos * kRopeDim;
  const float* sin_ptr = cos_ptr + kRopeDim / 2;
  __mt_bfloat16* rope_ptr =
      reinterpret_cast<__mt_bfloat16*>(token_ptr + kNopeDim);
  for (int64_t pair = tid; pair < kRopeDim / 2; pair += blockDim.x) {
    const int64_t even_dim = kNopeDim + pair * 2;
    const int64_t odd_dim = even_dim + 1;
    const float even = __bfloat162float(input[even_dim]);
    const float odd = __bfloat162float(input[odd_dim]);
    const float c = cos_ptr[pair];
    const float s = sin_ptr[pair];
    rope_ptr[pair * 2] = __float2bfloat16(even * c - odd * s);
    rope_ptr[pair * 2 + 1] = __float2bfloat16(even * s + odd * c);
  }
}

__global__ void deepseek_v4_store_sparse_kv_kernel(
    const __mt_bfloat16* __restrict__ normed, uint8_t* __restrict__ cache,
    const void* __restrict__ slots, const bool* __restrict__ write_mask,
    int index_kind, int64_t num_tokens, int64_t num_blocks, int64_t block_size,
    int64_t block_stride) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  if (token >= num_tokens) {
    return;
  }

  const int64_t slot = load_index(slots, index_kind, token);
  if (!write_mask[token] || slot < 0 || slot >= num_blocks * block_size) {
    return;
  }

  __shared__ float abs_values[kQuantBlockSize];
  __shared__ int scale_exponents[kTokenScaleBytes];

  const int tid = threadIdx.x;
  const int64_t block_idx = slot / block_size;
  const int64_t pos_in_block = slot - block_idx * block_size;
  uint8_t* block_ptr = cache + block_idx * block_stride;
  uint8_t* token_ptr = block_ptr + pos_in_block * kTokenDataBytes;
  uint8_t* scale_ptr = block_ptr + block_size * kTokenDataBytes +
                       pos_in_block * kTokenScaleBytes;
  const __mt_bfloat16* input = normed + token * kHeadDim;

  for (int qblock = 0; qblock < kNopeDim / kQuantBlockSize; ++qblock) {
    const int64_t start = qblock * kQuantBlockSize;
    if (tid < kQuantBlockSize) {
      const float value = __bfloat162float(input[start + tid]);
      abs_values[tid] = fabsf(value);
    }
    __syncthreads();

    for (int stride = kQuantBlockSize / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        abs_values[tid] = fmaxf(abs_values[tid], abs_values[tid + stride]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float amax = fmaxf(abs_values[0], 1.0e-4f);
      const int exponent =
          static_cast<int>(ceilf(log2f(amax / 448.0f)));
      scale_exponents[qblock] = exponent;
      const int scale_byte = max(0, min(255, exponent + 127));
      scale_ptr[qblock] = static_cast<uint8_t>(scale_byte);
    }
    __syncthreads();

    if (tid < kQuantBlockSize) {
      const float value = __bfloat162float(input[start + tid]);
      const float scaled = fminf(fmaxf(value * exp2f(-scale_exponents[qblock]),
                                      -448.0f),
                                448.0f);
      const __mt_fp8_e4m3 packed(scaled);
      token_ptr[start + tid] = packed.__x;
    }
    __syncthreads();
  }

  if (tid == 0) {
    scale_ptr[kTokenScaleBytes - 1] = 0;
  }
  if (tid < kRopeDim * 2) {
    const uint8_t* rope_bytes =
        reinterpret_cast<const uint8_t*>(input + kNopeDim);
    token_ptr[kNopeDim + tid] = rope_bytes[tid];
  }
}

int index_kind(const torch::Tensor& tensor) {
  if (tensor.scalar_type() == torch::kInt32) {
    return kIndexInt32;
  }
  if (tensor.scalar_type() == torch::kInt64) {
    return kIndexInt64;
  }
  TORCH_CHECK(false, "slot_mapping must be int32 or int64");
}

__global__ void deepseek_v4_qnorm_rope_kv_pack_fused_kernel(
    __mt_bfloat16* __restrict__ q, const __mt_bfloat16* __restrict__ kv,
    uint8_t* __restrict__ cache, const void* __restrict__ slots,
    int slot_kind, const void* __restrict__ positions, int position_kind,
    const float* __restrict__ cos_sin_cache, float eps, int64_t num_tokens,
    int64_t num_heads, int64_t num_slots, int64_t num_blocks,
    int64_t block_size, int64_t block_stride) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const int64_t head = static_cast<int64_t>(blockIdx.y);
  if (token >= num_tokens || head >= num_heads) {
    return;
  }

  __shared__ float reduce[kQNormThreads];
  const int tid = threadIdx.x;
  __mt_bfloat16* row = q + (token * num_heads + head) * kHeadDim;

  float partial = 0.0f;
  for (int64_t dim = tid; dim < kHeadDim; dim += blockDim.x) {
    const float value = __bfloat162float(row[dim]);
    partial += value * value;
  }
  reduce[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }

  const float norm = rsqrtf(reduce[0] / static_cast<float>(kHeadDim) + eps);
  for (int64_t dim = tid; dim < kNopeDim; dim += blockDim.x) {
    row[dim] = __float2bfloat16(__bfloat162float(row[dim]) * norm);
  }

  const int64_t pos = load_index(positions, position_kind, token);
  const float* cos_ptr = cos_sin_cache + pos * kRopeDim;
  const float* sin_ptr = cos_ptr + kRopeDim / 2;
  for (int64_t pair = tid; pair < kRopeDim / 2; pair += blockDim.x) {
    const int64_t even_dim = kNopeDim + pair * 2;
    const int64_t odd_dim = even_dim + 1;
    const float even = __bfloat162float(row[even_dim]) * norm;
    const float odd = __bfloat162float(row[odd_dim]) * norm;
    const float c = cos_ptr[pair];
    const float s = sin_ptr[pair];
    row[even_dim] = __float2bfloat16(even * c - odd * s);
    row[odd_dim] = __float2bfloat16(even * s + odd * c);
  }

  const bool kv_pack_block = head == 0 && token < num_slots;
  if (!kv_pack_block) {
    return;
  }

  const int64_t slot = load_index(slots, slot_kind, token);
  if (slot < 0 || slot >= num_blocks * block_size) {
    return;
  }

  const int64_t block_idx = slot / block_size;
  const int64_t pos_in_block = slot - block_idx * block_size;
  uint8_t* block_ptr = cache + block_idx * block_stride;
  uint8_t* token_ptr = block_ptr + pos_in_block * kTokenDataBytes;
  uint8_t* scale_ptr = block_ptr + block_size * kTokenDataBytes +
                       pos_in_block * kTokenScaleBytes;
  const __mt_bfloat16* input = kv + token * kHeadDim;

  // One warp owns each 64-element FP8 group. The previous implementation
  // processed the seven groups serially with a block-wide reduction and
  // barriers for every group. Keep the same scale rule and cache layout while
  // running all seven groups in parallel and using warp shuffles.
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (warp < kNopeDim / kQuantBlockSize) {
    const int64_t start = static_cast<int64_t>(warp) * kQuantBlockSize;
    const float x0 = __bfloat162float(input[start + lane * 2]);
    const float x1 = __bfloat162float(input[start + lane * 2 + 1]);
    const float amax =
        fmaxf(warp_reduce_max(fmaxf(fabsf(x0), fabsf(x1))), 1.0e-4f);
    const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
    const float inv_scale = exp2f(-exponent);
    const float scaled0 = fminf(fmaxf(x0 * inv_scale, -448.0f), 448.0f);
    const float scaled1 = fminf(fmaxf(x1 * inv_scale, -448.0f), 448.0f);
    const __mt_fp8_e4m3 packed0(scaled0);
    const __mt_fp8_e4m3 packed1(scaled1);
    reinterpret_cast<uint16_t*>(token_ptr + start)[lane] =
        static_cast<uint16_t>(packed0.__x) |
        (static_cast<uint16_t>(packed1.__x) << 8);
    if (lane == 0) {
      const int scale_byte = max(0, min(255, exponent + 127));
      scale_ptr[warp] = static_cast<uint8_t>(scale_byte);
    }
  }

  __mt_bfloat16* rope_ptr =
      reinterpret_cast<__mt_bfloat16*>(token_ptr + kNopeDim);
  if (warp == 7) {
    if (lane == 0) {
      scale_ptr[kTokenScaleBytes - 1] = 0;
    }
    const int64_t even_dim = kNopeDim + lane * 2;
    const int64_t odd_dim = even_dim + 1;
    const float even = __bfloat162float(input[even_dim]);
    const float odd = __bfloat162float(input[odd_dim]);
    const float c = cos_ptr[lane];
    const float s = sin_ptr[lane];
    rope_ptr[lane * 2] = __float2bfloat16(even * c - odd * s);
    rope_ptr[lane * 2 + 1] = __float2bfloat16(even * s + odd * c);
  }
}

}  // namespace

void deepseek_v4_qnorm_rope_kv_insert(
    torch::Tensor& q,
    const torch::Tensor& kv,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size) {
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(kv.scalar_type() == torch::kBFloat16, "kv must be bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8,
              "kv_cache must be uint8");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat32,
              "cos_sin_cache must be float32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
  TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  TORCH_CHECK(q.device() == kv.device() && q.device() == kv_cache.device() &&
                  q.device() == slot_mapping.device() &&
                  q.device() == positions.device() &&
                  q.device() == cos_sin_cache.device(),
              "all tensors must be on the same device");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kHeadDim, "q shape [N, H, 512]");
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == kHeadDim,
              "kv shape [N, 512]");
  TORCH_CHECK(kv.size(0) == q.size(0), "q and kv row counts must match");
  TORCH_CHECK(positions.dim() == 1 && positions.numel() == q.size(0),
              "positions must be [N]");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");
  TORCH_CHECK(slot_mapping.numel() <= q.size(0),
              "slot_mapping must not exceed q row count");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRopeDim,
              "cos_sin_cache shape [max_pos, 64]");
  TORCH_CHECK(kv_cache.dim() >= 3,
              "kv_cache must be [blocks, block_size, ...bytes...]");
  TORCH_CHECK(kv_cache.size(1) == cache_block_size,
              "cache_block_size must match kv_cache.size(1)");
  TORCH_CHECK(kv_cache.stride(-1) == 1,
              "kv_cache byte dimension must be contiguous");
  const int64_t logical_block_bytes =
      cache_block_size * (kTokenDataBytes + kTokenScaleBytes);
  TORCH_CHECK(kv_cache.stride(0) >= logical_block_bytes,
              "kv_cache block stride is too small");
  if (q.size(0) == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(q));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  const dim3 q_grid(static_cast<unsigned int>(q.size(0)),
                    static_cast<unsigned int>(q.size(1)));
  const dim3 q_block(kQNormThreads);
  // The fused q-norm/RoPE/KV-pack kernel is the validated DeepSeek-V4 path.
  // Keep the fallback kernels available for source-level reuse, but do not
  // make production behavior depend on an A/B-only process environment flag.
  deepseek_v4_qnorm_rope_kv_pack_fused_kernel<<<q_grid, q_block, 0, stream>>>(
      static_cast<__mt_bfloat16*>(q.data_ptr()),
      static_cast<const __mt_bfloat16*>(kv.data_ptr()),
      static_cast<uint8_t*>(kv_cache.data_ptr()), slot_mapping.data_ptr(),
      index_kind(slot_mapping), positions.data_ptr(), index_kind(positions),
      static_cast<const float*>(cos_sin_cache.data_ptr()),
      static_cast<float>(eps), q.size(0), q.size(1), slot_mapping.numel(),
      kv_cache.size(0), cache_block_size, kv_cache.stride(0));
  auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_qnorm_rope_kv_pack_fused launch failed: ",
              musaGetErrorString(err));
}

void deepseek_v4_store_sparse_kv(
    const torch::Tensor& normed,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& write_mask) {
  TORCH_CHECK(normed.scalar_type() == torch::kBFloat16,
              "normed must be bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8,
              "kv_cache must be uint8");
  TORCH_CHECK(write_mask.scalar_type() == torch::kBool,
              "write_mask must be bool");
  TORCH_CHECK(normed.is_contiguous(), "normed must be contiguous");
  TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
  TORCH_CHECK(write_mask.is_contiguous(), "write_mask must be contiguous");
  TORCH_CHECK(normed.device() == kv_cache.device() &&
                  normed.device() == slot_mapping.device() &&
                  normed.device() == write_mask.device(),
              "all tensors must be on the same device");
  TORCH_CHECK(normed.dim() == 2 && normed.size(1) == kHeadDim,
              "normed must be [num_tokens, 512]");
  TORCH_CHECK(kv_cache.dim() >= 2, "kv_cache must include block dimension");
  TORCH_CHECK(kv_cache.size(0) > 0 && kv_cache.size(1) > 0,
              "kv_cache must have non-empty blocks");
  TORCH_CHECK(slot_mapping.numel() == normed.size(0),
              "slot_mapping must have one entry per normed row");
  TORCH_CHECK(write_mask.numel() == normed.size(0),
              "write_mask must have one entry per normed row");
  TORCH_CHECK(kv_cache.stride(-1) == 1,
              "kv_cache byte dimension must be contiguous");
  const int64_t logical_block_bytes =
      kv_cache.size(1) * (kTokenDataBytes + kTokenScaleBytes);
  TORCH_CHECK(kv_cache.stride(0) >= logical_block_bytes,
              "kv_cache block stride is too small");

  if (normed.size(0) == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(normed));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  const dim3 grid(static_cast<unsigned int>(normed.size(0)));
  const dim3 block(128);
  deepseek_v4_store_sparse_kv_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(normed.data_ptr()),
      static_cast<uint8_t*>(kv_cache.data_ptr()), slot_mapping.data_ptr(),
      static_cast<const bool*>(write_mask.data_ptr()), index_kind(slot_mapping),
      normed.size(0), kv_cache.size(0), kv_cache.size(1), kv_cache.stride(0));
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess, "deepseek_v4_store_sparse_kv launch failed: ",
              musaGetErrorString(err));
}
