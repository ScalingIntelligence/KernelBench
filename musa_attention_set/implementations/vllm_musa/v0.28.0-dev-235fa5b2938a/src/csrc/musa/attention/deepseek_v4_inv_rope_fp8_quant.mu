#include <cmath>
#include <cstdint>
#include <tuple>

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
constexpr int64_t kQuantGroupSize = 128;
constexpr int64_t kChunksPerHead = kHeadDim / kQuantGroupSize;
constexpr int kThreads = 128;
constexpr float kFp8Max = 448.0f;
constexpr float kFp8Min = -448.0f;
constexpr float kScaleEps = 1.0e-10f;

constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;

__device__ __forceinline__ int64_t load_index(const void* ptr, int kind,
                                              int64_t idx) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t*>(ptr)[idx]);
  }
  return static_cast<int64_t>(static_cast<const int64_t*>(ptr)[idx]);
}

__device__ __forceinline__ float reduce_max_128(float* shared, int tid,
                                                float value) {
  shared[tid] = value;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  return shared[0];
}

__device__ __forceinline__ float load_rotated_or_raw(
    const __mt_bfloat16* __restrict__ input, int64_t dim,
    const float* __restrict__ cos_ptr, const float* __restrict__ sin_ptr) {
  if (dim < kNopeDim) {
    return __bfloat162float(input[dim]);
  }

  const int64_t rope_offset = dim - kNopeDim;
  const int64_t pair = rope_offset / 2;
  const int64_t even_dim = kNopeDim + pair * 2;
  const int64_t odd_dim = even_dim + 1;
  const float even = __bfloat162float(input[even_dim]);
  const float odd = __bfloat162float(input[odd_dim]);
  const float c = cos_ptr[pair];
  const float s = sin_ptr[pair];
  if ((rope_offset & 1) == 0) {
    return even * c + odd * s;
  }
  return odd * c - even * s;
}

template <bool TMA_ALIGNED_SCALES>
__global__ void deepseek_v4_inv_rope_fp8_quant_kernel(
    const __mt_bfloat16* __restrict__ o, int64_t o_stride_token,
    int64_t o_stride_head, const void* __restrict__ positions,
    int position_kind, const float* __restrict__ cos_sin_cache,
    int64_t cache_stride_pos, uint8_t* __restrict__ fp8_out,
    int64_t fp8_stride_group, int64_t fp8_stride_token, void* __restrict__ scale,
    int64_t scale_stride_group, int64_t scale_stride_k, int64_t num_tokens,
    int64_t heads_per_group) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const int64_t global_head = static_cast<int64_t>(blockIdx.y);
  const int tid = threadIdx.x;

  const int64_t group = global_head / heads_per_group;
  const int64_t head_in_group = global_head - group * heads_per_group;
  const int64_t chunk_base = head_in_group * kChunksPerHead;

  if (token >= num_tokens) {
    if constexpr (TMA_ALIGNED_SCALES) {
      if (tid == 0) {
        auto* packed = static_cast<int32_t*>(scale);
        packed[group * scale_stride_group + token +
               head_in_group * scale_stride_k] = 0;
      }
    } else if (tid < kChunksPerHead) {
      auto* scale_f = static_cast<float*>(scale);
      const int64_t chunk = chunk_base + tid;
      scale_f[group * scale_stride_group + token + chunk * scale_stride_k] =
          0.0f;
    }
    return;
  }

  __shared__ float reduce[kThreads];
  __shared__ float values[kQuantGroupSize];
  __shared__ uint32_t scale_bytes[kChunksPerHead];

  const int64_t pos = load_index(positions, position_kind, token);
  const float* cos_ptr = cos_sin_cache + pos * cache_stride_pos;
  const float* sin_ptr = cos_ptr + kRopeDim / 2;
  const __mt_bfloat16* input =
      o + token * o_stride_token + global_head * o_stride_head;
  uint8_t* output = fp8_out + group * fp8_stride_group +
                    token * fp8_stride_token + head_in_group * kHeadDim;

#pragma unroll
  for (int64_t chunk = 0; chunk < kChunksPerHead; ++chunk) {
    const int64_t dim = chunk * kQuantGroupSize + tid;
    const float value = load_rotated_or_raw(input, dim, cos_ptr, sin_ptr);
    values[tid] = value;

    const float absmax = reduce_max_128(reduce, tid, fabsf(value));
    const float scale_raw = fmaxf(absmax / kFp8Max, kScaleEps);
    const float scale_pow2 = exp2f(ceilf(log2f(scale_raw)));

    if (tid == 0) {
      if constexpr (TMA_ALIGNED_SCALES) {
        const uint32_t bits = __float_as_uint(scale_pow2);
        scale_bytes[chunk] = (bits >> 23u) & 0xffu;
      } else {
        auto* scale_f = static_cast<float*>(scale);
        const int64_t out_chunk = chunk_base + chunk;
        scale_f[group * scale_stride_group + token +
                out_chunk * scale_stride_k] = scale_pow2;
      }
    }
    __syncthreads();

    float q = values[tid] / scale_pow2;
    q = fminf(fmaxf(q, kFp8Min), kFp8Max);
    const __mt_fp8_e4m3 packed(q);
    output[dim] = packed.__x;
    __syncthreads();
  }

  if constexpr (TMA_ALIGNED_SCALES) {
    if (tid == 0) {
      uint32_t packed = 0;
#pragma unroll
      for (int64_t chunk = 0; chunk < kChunksPerHead; ++chunk) {
        packed |= scale_bytes[chunk] << (chunk * 8);
      }
      auto* scale_i = static_cast<int32_t*>(scale);
      scale_i[group * scale_stride_group + token +
              head_in_group * scale_stride_k] =
          static_cast<int32_t>(packed);
    }
  }
}

// The FP32-scale path has no cross-chunk packing dependency. Give each
// 128-element quantization chunk its own block so the four chunks in a
// DeepSeek-V4 head can execute concurrently. Only real tokens are launched;
// token 0 also clears the at-most-three scale-padding slots required by the
// downstream aligned layout.
__global__ void deepseek_v4_inv_rope_fp8_quant_chunk_kernel(
    const __mt_bfloat16* __restrict__ o, int64_t o_stride_token,
    int64_t o_stride_head, const void* __restrict__ positions,
    int position_kind, const float* __restrict__ cos_sin_cache,
    int64_t cache_stride_pos, uint8_t* __restrict__ fp8_out,
    int64_t fp8_stride_group, int64_t fp8_stride_token,
    float* __restrict__ scale, int64_t scale_stride_group,
    int64_t scale_stride_token, int64_t scale_stride_k, int64_t num_tokens,
    int64_t aligned_tokens, int64_t heads_per_group) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const int64_t global_head = static_cast<int64_t>(blockIdx.y);
  const int64_t chunk = static_cast<int64_t>(blockIdx.z);
  const int tid = threadIdx.x;

  const int64_t group = global_head / heads_per_group;
  const int64_t head_in_group = global_head - group * heads_per_group;
  const int64_t out_chunk = head_in_group * kChunksPerHead + chunk;
  const int64_t dim = chunk * kQuantGroupSize + tid;

  const int64_t pos = load_index(positions, position_kind, token);
  const float* cos_ptr = cos_sin_cache + pos * cache_stride_pos;
  const float* sin_ptr = cos_ptr + kRopeDim / 2;
  const __mt_bfloat16* input =
      o + token * o_stride_token + global_head * o_stride_head;
  uint8_t* output = fp8_out + group * fp8_stride_group +
                    token * fp8_stride_token + head_in_group * kHeadDim;

  const float value = load_rotated_or_raw(input, dim, cos_ptr, sin_ptr);
  __shared__ float reduce[kThreads];
  const float absmax = reduce_max_128(reduce, tid, fabsf(value));
  const float scale_raw = fmaxf(absmax / kFp8Max, kScaleEps);
  const float scale_pow2 = exp2f(ceilf(log2f(scale_raw)));

  if (tid == 0) {
    scale[group * scale_stride_group + token * scale_stride_token +
          out_chunk * scale_stride_k] = scale_pow2;
  }

  float q = value / scale_pow2;
  q = fminf(fmaxf(q, kFp8Min), kFp8Max);
  const __mt_fp8_e4m3 packed(q);
  output[dim] = packed.__x;

  const int64_t padding = aligned_tokens - num_tokens;
  if (token == 0 && tid < padding) {
    scale[group * scale_stride_group +
          (num_tokens + tid) * scale_stride_token +
          out_chunk * scale_stride_k] = 0.0f;
  }
}

int index_kind(const torch::Tensor& tensor, const char* name) {
  if (tensor.scalar_type() == torch::kInt32) {
    return kIndexInt32;
  }
  if (tensor.scalar_type() == torch::kInt64) {
    return kIndexInt64;
  }
  TORCH_CHECK(false, name, " must be int32 or int64");
}

int64_t tma_aligned_tokens(int64_t num_tokens) {
  return ((num_tokens + 3) / 4) * 4;
}

void check_musa_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_privateuseone(), name,
              " must be a MUSA tensor");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_fused_inv_rope_fp8_quant(
    const torch::Tensor& o, const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache, int64_t n_groups,
    int64_t heads_per_group, int64_t nope_dim, int64_t rope_dim,
    int64_t quant_group_size, bool tma_aligned_scales) {
  check_musa_tensor(o, "o");
  check_musa_tensor(positions, "positions");
  check_musa_tensor(cos_sin_cache, "cos_sin_cache");
  TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bfloat16");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat32,
              "cos_sin_cache must be float32");
  TORCH_CHECK(o.dim() == 3, "o must have shape [tokens, heads, head_dim]");
  TORCH_CHECK(o.size(2) == kHeadDim, "DeepSeek-V4 head dim must be 512");
  TORCH_CHECK(o.stride(2) == 1, "o last dimension must be contiguous");
  TORCH_CHECK(positions.dim() == 1 && positions.numel() == o.size(0),
              "positions must be [tokens]");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRopeDim,
              "cos_sin_cache must have shape [max_positions, 64]");
  TORCH_CHECK(cos_sin_cache.stride(1) == 1,
              "cos_sin_cache last dimension must be contiguous");
  TORCH_CHECK(o.device() == positions.device() &&
                  o.device() == cos_sin_cache.device(),
              "o, positions, and cos_sin_cache must be on the same device");
  TORCH_CHECK(nope_dim == kNopeDim && rope_dim == kRopeDim &&
                  quant_group_size == kQuantGroupSize,
              "only DeepSeek-V4 dimensions 448/64 with group_size=128 are "
              "supported");
  TORCH_CHECK(heads_per_group > 0 && n_groups > 0,
              "n_groups and heads_per_group must be positive");
  TORCH_CHECK(o.size(1) == n_groups * heads_per_group,
              "num heads must equal n_groups * heads_per_group");

  const int64_t num_tokens = o.size(0);
  const int64_t num_heads = o.size(1);
  const int64_t d = heads_per_group * kHeadDim;
  const int64_t num_scale_blocks = d / kQuantGroupSize;
  const int64_t aligned_tokens = tma_aligned_tokens(num_tokens);

  auto fp8_buf =
      torch::empty({n_groups, num_tokens, d},
                   o.options().dtype(torch::kFloat8_e4m3fn));

  torch::Tensor scale_storage;
  torch::Tensor scale_buf;
  if (tma_aligned_scales) {
    const int64_t packed_sf_k = (num_scale_blocks + 3) / 4;
    scale_storage = torch::empty({n_groups * packed_sf_k * aligned_tokens},
                                 o.options().dtype(torch::kInt32));
    scale_buf =
        scale_storage.as_strided({n_groups, num_tokens, packed_sf_k},
                                 {packed_sf_k * aligned_tokens, 1,
                                  aligned_tokens});
  } else {
    scale_storage = torch::empty({n_groups * num_scale_blocks * aligned_tokens},
                                 o.options().dtype(torch::kFloat32));
    // Keep each group-local [token, block] scale matrix contiguous.  The
    // downstream DeepSeek-V4 o-proj consumes one group at a time, so this
    // layout preserves the public [token, group, block] view while avoiding a
    // device copy for every layer and decode step.
    scale_buf =
        scale_storage.as_strided({n_groups, num_tokens, num_scale_blocks},
                                 {num_scale_blocks * aligned_tokens,
                                  num_scale_blocks, 1});
  }

  if (num_tokens == 0 || num_heads == 0) {
    return std::make_tuple(fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1));
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(o));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  const dim3 grid(static_cast<unsigned int>(aligned_tokens),
                  static_cast<unsigned int>(num_heads));
  const dim3 block(kThreads);

  if (tma_aligned_scales) {
    deepseek_v4_inv_rope_fp8_quant_kernel<true>
        <<<grid, block, 0, stream>>>(
            static_cast<const __mt_bfloat16*>(o.data_ptr()), o.stride(0),
            o.stride(1), positions.data_ptr(), index_kind(positions, "positions"),
            static_cast<const float*>(cos_sin_cache.data_ptr()),
            cos_sin_cache.stride(0), static_cast<uint8_t*>(fp8_buf.data_ptr()),
            fp8_buf.stride(0), fp8_buf.stride(1), scale_buf.data_ptr(),
            scale_buf.stride(0), scale_buf.stride(2), num_tokens,
            heads_per_group);
  } else {
    const dim3 chunk_grid(static_cast<unsigned int>(num_tokens),
                          static_cast<unsigned int>(num_heads),
                          static_cast<unsigned int>(kChunksPerHead));
    deepseek_v4_inv_rope_fp8_quant_chunk_kernel<<<chunk_grid, block, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(o.data_ptr()), o.stride(0),
        o.stride(1), positions.data_ptr(), index_kind(positions, "positions"),
        static_cast<const float*>(cos_sin_cache.data_ptr()),
        cos_sin_cache.stride(0), static_cast<uint8_t*>(fp8_buf.data_ptr()),
        fp8_buf.stride(0), fp8_buf.stride(1),
        static_cast<float*>(scale_buf.data_ptr()), scale_buf.stride(0),
        scale_buf.stride(1), scale_buf.stride(2), num_tokens, aligned_tokens,
        heads_per_group);
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_fused_inv_rope_fp8_quant launch failed: ",
              musaGetErrorString(err));

  return std::make_tuple(fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1));
}
