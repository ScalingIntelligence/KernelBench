#include <cfloat>
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

constexpr int64_t kHeadDim = 128;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kNopeDim = kHeadDim - kRopeDim;
constexpr int64_t kStateWidth = 2 * kHeadDim;
constexpr int64_t kStateRowWidth = 2 * kStateWidth;
constexpr int64_t kStateBlockSize = 4;
constexpr int64_t kCompressRatio = 4;
constexpr int64_t kCompressRows = 2 * kCompressRatio;
constexpr int64_t kTokenValueBytes = kHeadDim;
constexpr int64_t kTokenScaleBytes = sizeof(float);
constexpr int64_t kWarpsPerBlock = 4;
constexpr int64_t kThreadsPerBlock = 32 * kWarpsPerBlock;
constexpr int64_t kMaxDecodeRows = 128;
constexpr float kFp8Max = 448.0f;
constexpr float kLog2E = 1.4426950408889634f;

constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;

__device__ __forceinline__ int64_t load_index(const void* ptr, int kind,
                                              int64_t idx) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t*>(ptr)[idx]);
  }
  return static_cast<const int64_t*>(ptr)[idx];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  value += __shfl_xor_sync(0xffffffffu, value, 16);
  value += __shfl_xor_sync(0xffffffffu, value, 8);
  value += __shfl_xor_sync(0xffffffffu, value, 4);
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value;
}

__device__ __forceinline__ uint32_t warp_reduce_max_u32(uint32_t value) {
  uint32_t peer = __shfl_xor_sync(0xffffffffu, value, 16);
  value = value > peer ? value : peer;
  peer = __shfl_xor_sync(0xffffffffu, value, 8);
  value = value > peer ? value : peer;
  peer = __shfl_xor_sync(0xffffffffu, value, 4);
  value = value > peer ? value : peer;
  peer = __shfl_xor_sync(0xffffffffu, value, 2);
  value = value > peer ? value : peer;
  peer = __shfl_xor_sync(0xffffffffu, value, 1);
  return value > peer ? value : peer;
}

__device__ __forceinline__ float weight_to_float(float value) { return value; }

__device__ __forceinline__ float weight_to_float(__mt_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float clamp_fp8(float value) {
  return fminf(fmaxf(value, -kFp8Max), kFp8Max);
}

template <typename WeightT, int kKvBlockSize>
__global__ __launch_bounds__(kThreadsPerBlock, 4)
void deepseek_v4_c4_indexer_compressor_kernel(
    const float* __restrict__ state_cache, int64_t state_stride0,
    int64_t state_stride1, const void* __restrict__ token_to_req_indices,
    int token_to_req_kind, const void* __restrict__ positions,
    int position_kind, const void* __restrict__ state_slot_mapping,
    int state_slot_kind, const void* __restrict__ block_table,
    int block_table_kind, int64_t block_table_stride,
    const WeightT* __restrict__ rms_norm_weight,
    const float* __restrict__ cos_sin_cache, int64_t cos_sin_stride,
    uint8_t* __restrict__ kv_cache, int64_t kv_cache_stride0,
    const void* __restrict__ kv_slot_mapping, int kv_slot_kind, float rms_eps,
    int64_t num_tokens, int64_t num_state_blocks, int64_t num_reqs,
    int64_t max_blocks_per_req, int64_t num_kv_blocks) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t token =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (token >= num_tokens) {
    return;
  }

  // Preserve the Triton PAD and compression-boundary semantics. Every lane in
  // a warp observes the same branch, so early returns do not break reductions.
  const int64_t state_slot =
      load_index(state_slot_mapping, state_slot_kind, token);
  if (state_slot < 0 ||
      state_slot >= num_state_blocks * kStateBlockSize) {
    return;
  }
  const int64_t position = load_index(positions, position_kind, token);
  if (position < kCompressRatio - 1 ||
      (position + 1) % kCompressRatio != 0) {
    return;
  }
  const int64_t req_idx =
      load_index(token_to_req_indices, token_to_req_kind, token);
  if (req_idx < 0 || req_idx >= num_reqs) {
    return;
  }

  // One lane owns four adjacent head dimensions. The eight C4 rows are kept
  // in registers; unlike the Triton program this packs four logical programs
  // into a single 128-thread block without using shared memory or barriers.
  float kv[kCompressRows][4];
  float score[kCompressRows][4];
  const int64_t first_position = position - kCompressRows + 1;

#pragma unroll
  for (int row = 0; row < kCompressRows; ++row) {
    const int64_t source_position = first_position + row;
    bool valid = source_position >= 0;
    int64_t physical_block = 0;
    const int64_t logical_block = source_position / kStateBlockSize;
    if (valid) {
      valid = logical_block >= 0 && logical_block < max_blocks_per_req;
    }
    if (valid) {
      physical_block = load_index(
          block_table, block_table_kind,
          req_idx * block_table_stride + logical_block);
      valid = physical_block >= 0 && physical_block < num_state_blocks;
    }

    float4 kv_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 score_vec =
        make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
    if (valid) {
      const int64_t block_offset = source_position % kStateBlockSize;
      const int64_t head_offset = row >= kCompressRatio ? kHeadDim : 0;
      const float* row_ptr = state_cache + physical_block * state_stride0 +
                             block_offset * state_stride1 + head_offset +
                             lane * 4;
      kv_vec = *reinterpret_cast<const float4*>(row_ptr);
      score_vec = *reinterpret_cast<const float4*>(row_ptr + kStateWidth);
    }
    kv[row][0] = kv_vec.x;
    kv[row][1] = kv_vec.y;
    kv[row][2] = kv_vec.z;
    kv[row][3] = kv_vec.w;
    score[row][0] = score_vec.x;
    score[row][1] = score_vec.y;
    score[row][2] = score_vec.z;
    score[row][3] = score_vec.w;
  }

  // Stable per-dimension softmax and weighted reduction. The exp2 form follows
  // the optimized SGLang C4 kernel and matches the fast-math lowering used by
  // MUSA Triton for exp.
  float max0 = score[0][0];
  float max1 = score[0][1];
  float max2 = score[0][2];
  float max3 = score[0][3];
#pragma unroll
  for (int row = 1; row < kCompressRows; ++row) {
    max0 = fmaxf(max0, score[row][0]);
    max1 = fmaxf(max1, score[row][1]);
    max2 = fmaxf(max2, score[row][2]);
    max3 = fmaxf(max3, score[row][3]);
  }
  float denominator0 = 0.0f;
  float denominator1 = 0.0f;
  float denominator2 = 0.0f;
  float denominator3 = 0.0f;
  float numerator0 = 0.0f;
  float numerator1 = 0.0f;
  float numerator2 = 0.0f;
  float numerator3 = 0.0f;
#pragma unroll
  for (int row = 0; row < kCompressRows; ++row) {
    const float exp0 = exp2f((score[row][0] - max0) * kLog2E);
    const float exp1 = exp2f((score[row][1] - max1) * kLog2E);
    const float exp2 = exp2f((score[row][2] - max2) * kLog2E);
    const float exp3 = exp2f((score[row][3] - max3) * kLog2E);
    numerator0 += kv[row][0] * exp0;
    numerator1 += kv[row][1] * exp1;
    numerator2 += kv[row][2] * exp2;
    numerator3 += kv[row][3] * exp3;
    denominator0 += exp0;
    denominator1 += exp1;
    denominator2 += exp2;
    denominator3 += exp3;
  }
  float compressed[4] = {
      numerator0 / denominator0,
      numerator1 / denominator1,
      numerator2 / denominator2,
      numerator3 / denominator3,
  };

  float sum_of_squares = 0.0f;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    sum_of_squares += compressed[elem] * compressed[elem];
  }
  sum_of_squares = warp_reduce_sum(sum_of_squares);
  const float norm_factor =
      rsqrtf(sum_of_squares / static_cast<float>(kHeadDim) + rms_eps);

  float output[4];
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    output[elem] = compressed[elem] * norm_factor *
                   weight_to_float(rms_norm_weight[lane * 4 + elem]);
  }

  // vLLM's indexer cache contract applies GPT-J interleaved RoPE to the final
  // 64 dimensions. Do not import SGLang's subsequent Hadamard transform: its
  // query/indexer contract differs and vLLM's eager reference has no Hadamard.
  if (lane >= kNopeDim / 4) {
    const int64_t pair_base = (lane - kNopeDim / 4) * 2;
    const int64_t compressed_position =
        (position / kCompressRatio) * kCompressRatio;
    const float* cos_ptr =
        cos_sin_cache + compressed_position * cos_sin_stride;
    const float* sin_ptr = cos_ptr + kRopeDim / 2;

    const float even0 = output[0];
    const float odd0 = output[1];
    const float even1 = output[2];
    const float odd1 = output[3];
    const float cos0 = cos_ptr[pair_base];
    const float sin0 = sin_ptr[pair_base];
    const float cos1 = cos_ptr[pair_base + 1];
    const float sin1 = sin_ptr[pair_base + 1];
    output[0] = even0 * cos0 - odd0 * sin0;
    output[1] = odd0 * cos0 + even0 * sin0;
    output[2] = even1 * cos1 - odd1 * sin1;
    output[3] = odd1 * cos1 + even1 * sin1;
  }

  // Match the existing kernel's one bf16 rounding point before absmax/FP8.
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    output[elem] = __bfloat162float(__float2bfloat16(output[elem]));
  }

  // Positive finite float bit patterns preserve numeric ordering. The latest
  // SGLang C4 decode path uses this integer max to reduce the absmax latency.
  uint32_t local_absmax_bits =
      __float_as_uint(output[0]) & 0x7fffffffu;
#pragma unroll
  for (int elem = 1; elem < 4; ++elem) {
    const uint32_t abs_bits = __float_as_uint(output[elem]) & 0x7fffffffu;
    local_absmax_bits =
        local_absmax_bits > abs_bits ? local_absmax_bits : abs_bits;
  }
  const float absmax = fmaxf(
      1.0e-4f, __uint_as_float(warp_reduce_max_u32(local_absmax_bits)));
  const int exponent = static_cast<int>(ceilf(log2f(absmax / kFp8Max)));
  const float scale = exp2f(static_cast<float>(exponent));
  const float inv_scale = exp2f(static_cast<float>(-exponent));

  const int64_t kv_slot = load_index(kv_slot_mapping, kv_slot_kind, token);
  if (kv_slot < 0 || kv_slot >= num_kv_blocks * kKvBlockSize) {
    return;
  }
  const int64_t kv_block_idx = kv_slot / kKvBlockSize;
  const int64_t kv_pos_in_block = kv_slot % kKvBlockSize;
  uint8_t* cache_block = kv_cache + kv_block_idx * kv_cache_stride0;
  uint8_t* value_ptr =
      cache_block + kv_pos_in_block * kTokenValueBytes;
  uint8_t* scale_ptr = cache_block + kKvBlockSize * kTokenValueBytes +
                       kv_pos_in_block * kTokenScaleBytes;

  const float4 quant_input =
      make_float4(clamp_fp8(output[0] * inv_scale),
                  clamp_fp8(output[1] * inv_scale),
                  clamp_fp8(output[2] * inv_scale),
                  clamp_fp8(output[3] * inv_scale));
  const __mt_fp8x4_e4m3 packed(quant_input);
  reinterpret_cast<uint32_t*>(value_ptr)[lane] =
      static_cast<uint32_t>(packed.__x);
  if (lane == 0) {
    *reinterpret_cast<float*>(scale_ptr) = scale;
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

void check_same_device(const torch::Tensor& reference,
                       const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device() == reference.device(), name,
              " must be on the same device as state_cache");
}

template <typename WeightT, int kKvBlockSize>
void launch_c4_indexer_compressor(
    const torch::Tensor& state_cache,
    const torch::Tensor& token_to_req_indices,
    const torch::Tensor& positions,
    const torch::Tensor& state_slot_mapping,
    const torch::Tensor& block_table,
    const torch::Tensor& rms_norm_weight,
    const torch::Tensor& cos_sin_cache, torch::Tensor& kv_cache,
    const torch::Tensor& kv_slot_mapping, float rms_eps,
    musaStream_t stream) {
  const int64_t num_tokens = state_slot_mapping.numel();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(
      (num_tokens + kWarpsPerBlock - 1) / kWarpsPerBlock));
  deepseek_v4_c4_indexer_compressor_kernel<WeightT, kKvBlockSize>
      <<<grid, block, 0, stream>>>(
          static_cast<const float*>(state_cache.data_ptr()),
          state_cache.stride(0), state_cache.stride(1),
          token_to_req_indices.data_ptr(),
          index_kind(token_to_req_indices, "token_to_req_indices"),
          positions.data_ptr(), index_kind(positions, "positions"),
          state_slot_mapping.data_ptr(),
          index_kind(state_slot_mapping, "state_slot_mapping"),
          block_table.data_ptr(), index_kind(block_table, "block_table"),
          block_table.stride(0),
          static_cast<const WeightT*>(rms_norm_weight.data_ptr()),
          static_cast<const float*>(cos_sin_cache.data_ptr()),
          cos_sin_cache.stride(0), static_cast<uint8_t*>(kv_cache.data_ptr()),
          kv_cache.stride(0), kv_slot_mapping.data_ptr(),
          index_kind(kv_slot_mapping, "kv_slot_mapping"), rms_eps, num_tokens,
          state_cache.size(0), block_table.size(0), block_table.size(1),
          kv_cache.size(0));
}

}  // namespace

void deepseek_v4_c4_indexer_compress_cache(
    const torch::Tensor& state_cache,
    const torch::Tensor& token_to_req_indices,
    const torch::Tensor& positions,
    const torch::Tensor& state_slot_mapping,
    const torch::Tensor& block_table,
    const torch::Tensor& rms_norm_weight,
    const torch::Tensor& cos_sin_cache, torch::Tensor& kv_cache,
    const torch::Tensor& kv_slot_mapping, double rms_eps,
    int64_t state_block_size, int64_t state_width,
    int64_t kv_block_size) {
  TORCH_CHECK(state_cache.scalar_type() == torch::kFloat32,
              "state_cache must be float32");
  TORCH_CHECK(state_cache.dim() == 3 &&
                  state_cache.size(1) == kStateBlockSize &&
                  state_cache.size(2) == kStateRowWidth,
              "state_cache must have shape [num_blocks, 4, 512]");
  TORCH_CHECK(state_cache.stride(2) == 1 &&
                  state_cache.stride(1) % 4 == 0 &&
                  state_cache.stride(0) % 4 == 0,
              "state_cache rows must support aligned float4 loads");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state_cache.data_ptr()) % 16 == 0,
              "state_cache must be 16-byte aligned");
  TORCH_CHECK(state_block_size == kStateBlockSize,
              "C4 indexer compressor requires state block size 4");
  TORCH_CHECK(state_width == kStateWidth,
              "C4 indexer compressor requires state width 256");
  TORCH_CHECK(state_slot_mapping.numel() > 0 &&
                  state_slot_mapping.numel() <= kMaxDecodeRows,
              "C4 native path supports 1..128 decode rows");

  const int64_t num_tokens = state_slot_mapping.numel();
  TORCH_CHECK(token_to_req_indices.numel() >= num_tokens,
              "token_to_req_indices does not cover every decode row");
  TORCH_CHECK(positions.numel() >= num_tokens,
              "positions does not cover every decode row");
  TORCH_CHECK(kv_slot_mapping.numel() >= num_tokens,
              "kv_slot_mapping does not cover every decode row");
  TORCH_CHECK(token_to_req_indices.is_contiguous() &&
                  positions.is_contiguous() &&
                  state_slot_mapping.is_contiguous() &&
                  kv_slot_mapping.is_contiguous(),
              "index tensors must be contiguous");
  TORCH_CHECK(block_table.dim() == 2 && block_table.stride(1) == 1 &&
                  block_table.size(0) > 0 && block_table.size(1) > 0,
              "block_table must be a non-empty row-contiguous 2D tensor");

  TORCH_CHECK(rms_norm_weight.dim() == 1 &&
                  rms_norm_weight.numel() == kHeadDim &&
                  rms_norm_weight.is_contiguous(),
              "rms_norm_weight must be contiguous with shape [128]");
  TORCH_CHECK(rms_norm_weight.scalar_type() == torch::kFloat32 ||
                  rms_norm_weight.scalar_type() == torch::kBFloat16,
              "rms_norm_weight must be float32 or bfloat16");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat32 &&
                  cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.size(1) == kRopeDim &&
                  cos_sin_cache.stride(1) == 1,
              "cos_sin_cache must be float32 [max_position, 64]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8 && kv_cache.dim() >= 2 &&
                  kv_cache.size(0) > 0 && kv_cache.size(1) == kv_block_size &&
                  kv_cache.stride(-1) == 1,
              "kv_cache must be a uint8 paged cache with the requested block size");
  TORCH_CHECK(kv_block_size == 64 || kv_block_size == 256,
              "C4 native path supports KV block size 64 or 256");
  TORCH_CHECK(kv_cache.stride(0) >=
                  kv_block_size * (kTokenValueBytes + kTokenScaleBytes),
              "kv_cache page stride is smaller than the 132-byte token layout");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(kv_cache.data_ptr()) % 4 == 0 &&
                  kv_cache.stride(0) % 4 == 0,
              "kv_cache must support aligned fp8x4 and float scale stores");

  check_same_device(state_cache, token_to_req_indices,
                    "token_to_req_indices");
  check_same_device(state_cache, positions, "positions");
  check_same_device(state_cache, state_slot_mapping, "state_slot_mapping");
  check_same_device(state_cache, block_table, "block_table");
  check_same_device(state_cache, rms_norm_weight, "rms_norm_weight");
  check_same_device(state_cache, cos_sin_cache, "cos_sin_cache");
  check_same_device(state_cache, kv_cache, "kv_cache");
  check_same_device(state_cache, kv_slot_mapping, "kv_slot_mapping");

  const at::musa::OptionalMUSAGuard device_guard(device_of(state_cache));
  musaStream_t stream = at::musa::getCurrentMUSAStream();

#define LAUNCH_C4_INDEXER(WEIGHT_T, PAGE_SIZE)                                \
  launch_c4_indexer_compressor<WEIGHT_T, PAGE_SIZE>(                          \
      state_cache, token_to_req_indices, positions, state_slot_mapping,       \
      block_table, rms_norm_weight, cos_sin_cache, kv_cache, kv_slot_mapping, \
      static_cast<float>(rms_eps), stream)

  if (rms_norm_weight.scalar_type() == torch::kFloat32) {
    if (kv_block_size == 64) {
      LAUNCH_C4_INDEXER(float, 64);
    } else {
      LAUNCH_C4_INDEXER(float, 256);
    }
  } else {
    if (kv_block_size == 64) {
      LAUNCH_C4_INDEXER(__mt_bfloat16, 64);
    } else {
      LAUNCH_C4_INDEXER(__mt_bfloat16, 256);
    }
  }

#undef LAUNCH_C4_INDEXER

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_c4_indexer_compress_cache launch failed: ",
              musaGetErrorString(err));
}
