#include <cmath>
#include <cstdint>
#include <limits>

#include <musa_bf16.h>
#include <musa_fp8.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kNumHeads = 64;
constexpr int64_t kScaleBytes = 4;
constexpr int64_t kMaxSeqLen = 4096;
constexpr int64_t kMaxTopK = 512;
constexpr int64_t kMaxCandidates = 1024;
constexpr int kThreads = 256;
constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;

__device__ __forceinline__ float dequant_fp8_e4m3(uint8_t byte) {
  __mt_fp8_e4m3 packed;
  packed.__x = byte;
  return static_cast<float>(packed);
}

__device__ __forceinline__ int64_t load_index(const void *ptr, int kind,
                                              int64_t idx) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t *>(ptr)[idx]);
  }
  return static_cast<int64_t>(static_cast<const int64_t *>(ptr)[idx]);
}

int index_kind(const torch::Tensor &tensor, const char *name) {
  if (tensor.scalar_type() == torch::kInt32) {
    return kIndexInt32;
  }
  if (tensor.scalar_type() == torch::kInt64) {
    return kIndexInt64;
  }
  TORCH_CHECK(false, name, " must be int32 or int64");
}

void check_musa_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.device().is_privateuseone(), name,
              " must be a MUSA tensor");
}

void check_same_device(const torch::Tensor &a, const torch::Tensor &b,
                       const char *b_name) {
  TORCH_CHECK(a.device() == b.device(), b_name, " must be on the same device");
}

__device__ __forceinline__ bool better_pair(float lhs_value, int lhs_index,
                                            float rhs_value, int rhs_index) {
  return lhs_value > rhs_value ||
         (lhs_value == rhs_value &&
          (rhs_index < 0 || (lhs_index >= 0 && lhs_index < rhs_index)));
}

__device__ __forceinline__ void compare_swap_score_pair(
    float *__restrict__ scores, int *__restrict__ score_indices, int lhs,
    int rhs, bool descending) {
  const float lhs_value = scores[lhs];
  const int lhs_index = score_indices[lhs];
  const float rhs_value = scores[rhs];
  const int rhs_index = score_indices[rhs];
  const bool should_swap =
      (descending && better_pair(rhs_value, rhs_index, lhs_value, lhs_index)) ||
      (!descending && better_pair(lhs_value, lhs_index, rhs_value, rhs_index));
  if (should_swap) {
    scores[lhs] = rhs_value;
    score_indices[lhs] = rhs_index;
    scores[rhs] = lhs_value;
    score_indices[rhs] = lhs_index;
  }
}

template <typename OutT>
__global__ void deepseek_v4_indexer_topk_decode_kernel(
    const uint8_t *__restrict__ q_quant, int64_t q_stride0, int64_t q_stride1,
    int64_t q_stride2, const uint8_t *__restrict__ kv_cache,
    int64_t num_blocks, int64_t block_size, int64_t block_stride,
    const float *__restrict__ weights, int64_t weights_stride0,
    int64_t weights_stride1, const void *__restrict__ seq_lens,
    int seq_lens_kind, int64_t seq_lens_stride,
    const void *__restrict__ block_table, int block_table_kind,
    int64_t block_table_stride, OutT *__restrict__ topk_indices,
    int64_t topk_stride0, int64_t topk_stride1, int64_t rows, int64_t topk) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float scores[kMaxSeqLen];
  __shared__ float reduce_values[kThreads];
  __shared__ int reduce_indices[kThreads];

  const int tid = threadIdx.x;
  const int64_t raw_seq_len =
      load_index(seq_lens, seq_lens_kind, row * seq_lens_stride);
  const int64_t seq_len =
      raw_seq_len < kMaxSeqLen ? (raw_seq_len > 0 ? raw_seq_len : 0)
                               : kMaxSeqLen;

  for (int64_t pos = tid; pos < kMaxSeqLen; pos += blockDim.x) {
    float score = -INFINITY;
    if (pos < seq_len) {
      const int64_t logical_block = pos / block_size;
      const int64_t pos_in_block = pos - logical_block * block_size;
      const int64_t physical_block = load_index(
          block_table, block_table_kind, row * block_table_stride + logical_block);

      if (physical_block >= 0 && physical_block < num_blocks &&
          pos_in_block >= 0 && pos_in_block < block_size) {
        const uint8_t *block_ptr = kv_cache + physical_block * block_stride;
        const uint8_t *k_ptr = block_ptr + pos_in_block * kHeadDim;
        const uint8_t *scale_ptr =
            block_ptr + block_size * kHeadDim + pos_in_block * kScaleBytes;
        const float k_scale = *reinterpret_cast<const float *>(scale_ptr);

        float accum = 0.0f;
        for (int64_t head = 0; head < kNumHeads; ++head) {
          const uint8_t *q_ptr =
              q_quant + row * q_stride0 + head * q_stride1;
          float dot = 0.0f;
#pragma unroll 4
          for (int64_t dim = 0; dim < kHeadDim; ++dim) {
            dot += dequant_fp8_e4m3(q_ptr[dim * q_stride2]) *
                   dequant_fp8_e4m3(k_ptr[dim]);
          }
          accum += fmaxf(dot, 0.0f) *
                   weights[row * weights_stride0 + head * weights_stride1];
        }
        score = accum * k_scale;
      }
    }
    scores[pos] = score;
  }
  __syncthreads();

  const int64_t effective_topk =
      topk < seq_len ? (topk < kMaxTopK ? topk : kMaxTopK)
                     : (seq_len < kMaxTopK ? seq_len : kMaxTopK);

  for (int64_t rank = 0; rank < effective_topk; ++rank) {
    float local_value = -INFINITY;
    int local_index = -1;
    for (int64_t pos = tid; pos < seq_len; pos += blockDim.x) {
      const float value = scores[pos];
      if (better_pair(value, static_cast<int>(pos), local_value, local_index)) {
        local_value = value;
        local_index = static_cast<int>(pos);
      }
    }
    reduce_values[tid] = local_value;
    reduce_indices[tid] = local_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other_value = reduce_values[tid + stride];
        const int other_index = reduce_indices[tid + stride];
        if (better_pair(other_value, other_index, reduce_values[tid],
                        reduce_indices[tid])) {
          reduce_values[tid] = other_value;
          reduce_indices[tid] = other_index;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      const int selected = reduce_indices[0];
      topk_indices[row * topk_stride0 + rank * topk_stride1] =
          static_cast<OutT>(selected);
      if (selected >= 0) {
        scores[selected] = -INFINITY;
      }
    }
    __syncthreads();
  }

  for (int64_t rank = effective_topk + tid; rank < topk; rank += blockDim.x) {
    topk_indices[row * topk_stride0 + rank * topk_stride1] =
        static_cast<OutT>(-1);
  }
}

template <typename OutT>
__global__ void deepseek_v4_indexer_topk_prefill_kernel(
    const uint8_t *__restrict__ q_quant, int64_t q_stride0, int64_t q_stride1,
    int64_t q_stride2, const uint8_t *__restrict__ kv_cache,
    int64_t num_blocks, int64_t block_size, int64_t block_stride,
    const float *__restrict__ weights, int64_t weights_stride0,
    int64_t weights_stride1, const void *__restrict__ block_table,
    int block_table_kind, int64_t block_table_stride,
    const void *__restrict__ cu_seq_lens, int cu_seq_lens_kind,
    int64_t cu_seq_lens_stride, const void *__restrict__ token_to_seq,
    int token_to_seq_kind, int64_t token_to_seq_stride,
    const void *__restrict__ cu_seqlen_ks, int cu_seqlen_ks_kind,
    int64_t cu_seqlen_ks_stride, const void *__restrict__ cu_seqlen_ke,
    int cu_seqlen_ke_kind, int64_t cu_seqlen_ke_stride,
    OutT *__restrict__ topk_indices, int64_t topk_stride0,
    int64_t topk_stride1, int64_t rows, int64_t total_seq_lens,
    int64_t topk) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float scores[kMaxSeqLen];
  __shared__ float reduce_values[kThreads];
  __shared__ int reduce_indices[kThreads];

  const int tid = threadIdx.x;
  const int64_t row_start =
      load_index(cu_seqlen_ks, cu_seqlen_ks_kind, row * cu_seqlen_ks_stride);
  const int64_t row_end =
      load_index(cu_seqlen_ke, cu_seqlen_ke_kind, row * cu_seqlen_ke_stride);
  const int64_t raw_row_len = row_end - row_start;
  const int64_t row_len =
      raw_row_len < kMaxSeqLen ? (raw_row_len > 0 ? raw_row_len : 0)
                               : kMaxSeqLen;

  for (int64_t rel_pos = tid; rel_pos < kMaxSeqLen; rel_pos += blockDim.x) {
    float score = -INFINITY;
    if (rel_pos < row_len) {
      const int64_t abs_pos = row_start + rel_pos;
      if (abs_pos >= 0 && abs_pos < total_seq_lens) {
        const int64_t req_idx = load_index(
            token_to_seq, token_to_seq_kind, abs_pos * token_to_seq_stride);
        const int64_t req_start = load_index(
            cu_seq_lens, cu_seq_lens_kind, req_idx * cu_seq_lens_stride);
        const int64_t local_pos = abs_pos - req_start;
        const int64_t logical_block = local_pos / block_size;
        const int64_t pos_in_block = local_pos - logical_block * block_size;
        const int64_t physical_block =
            load_index(block_table, block_table_kind,
                       req_idx * block_table_stride + logical_block);

        if (physical_block >= 0 && physical_block < num_blocks &&
            pos_in_block >= 0 && pos_in_block < block_size) {
          const uint8_t *block_ptr = kv_cache + physical_block * block_stride;
          const uint8_t *k_ptr = block_ptr + pos_in_block * kHeadDim;
          const uint8_t *scale_ptr =
              block_ptr + block_size * kHeadDim + pos_in_block * kScaleBytes;
          const float k_scale = *reinterpret_cast<const float *>(scale_ptr);

          float accum = 0.0f;
          for (int64_t head = 0; head < kNumHeads; ++head) {
            const uint8_t *q_ptr =
                q_quant + row * q_stride0 + head * q_stride1;
            float dot = 0.0f;
#pragma unroll 4
            for (int64_t dim = 0; dim < kHeadDim; ++dim) {
              dot += dequant_fp8_e4m3(q_ptr[dim * q_stride2]) *
                     dequant_fp8_e4m3(k_ptr[dim]);
            }
            accum += fmaxf(dot, 0.0f) *
                     weights[row * weights_stride0 + head * weights_stride1];
          }
          score = accum * k_scale;
        }
      }
    }
    scores[rel_pos] = score;
  }
  __syncthreads();

  const int64_t effective_topk =
      topk < row_len ? (topk < kMaxTopK ? topk : kMaxTopK)
                     : (row_len < kMaxTopK ? row_len : kMaxTopK);

  for (int64_t rank = 0; rank < effective_topk; ++rank) {
    float local_value = -INFINITY;
    int local_index = -1;
    for (int64_t rel_pos = tid; rel_pos < row_len; rel_pos += blockDim.x) {
      const float value = scores[rel_pos];
      if (better_pair(value, static_cast<int>(rel_pos), local_value,
                      local_index)) {
        local_value = value;
        local_index = static_cast<int>(rel_pos);
      }
    }
    reduce_values[tid] = local_value;
    reduce_indices[tid] = local_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other_value = reduce_values[tid + stride];
        const int other_index = reduce_indices[tid + stride];
        if (better_pair(other_value, other_index, reduce_values[tid],
                        reduce_indices[tid])) {
          reduce_values[tid] = other_value;
          reduce_indices[tid] = other_index;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      const int selected = reduce_indices[0];
      topk_indices[row * topk_stride0 + rank * topk_stride1] =
          static_cast<OutT>(selected);
      if (selected >= 0) {
        scores[selected] = -INFINITY;
      }
    }
    __syncthreads();
  }

  for (int64_t rank = effective_topk + tid; rank < topk; rank += blockDim.x) {
    topk_indices[row * topk_stride0 + rank * topk_stride1] =
        static_cast<OutT>(-1);
  }
}

template <typename OutT>
__global__ void deepseek_v4_indexer_topk_prefill_q_cache_kernel(
    const uint8_t *__restrict__ q_quant, int64_t q_stride0, int64_t q_stride1,
    int64_t q_stride2, const uint8_t *__restrict__ kv_cache,
    int64_t num_blocks, int64_t block_size, int64_t block_stride,
    const float *__restrict__ weights, int64_t weights_stride0,
    int64_t weights_stride1, const void *__restrict__ block_table,
    int block_table_kind, int64_t block_table_stride,
    const void *__restrict__ cu_seq_lens, int cu_seq_lens_kind,
    int64_t cu_seq_lens_stride, const void *__restrict__ token_to_seq,
    int token_to_seq_kind, int64_t token_to_seq_stride,
    const void *__restrict__ cu_seqlen_ks, int cu_seqlen_ks_kind,
    int64_t cu_seqlen_ks_stride, const void *__restrict__ cu_seqlen_ke,
    int cu_seqlen_ke_kind, int64_t cu_seqlen_ke_stride,
    OutT *__restrict__ topk_indices, int64_t topk_stride0,
    int64_t topk_stride1, int64_t rows, int64_t total_seq_lens,
    int64_t topk) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ __mt_bfloat16 q_deq[kNumHeads * kHeadDim];
  __shared__ float weight_cache[kNumHeads];
  __shared__ float scores[kMaxSeqLen];
  __shared__ float reduce_values[kThreads];
  __shared__ int reduce_indices[kThreads];

  const int tid = threadIdx.x;
  for (int64_t elem = tid; elem < kNumHeads * kHeadDim; elem += blockDim.x) {
    const int64_t head = elem / kHeadDim;
    const int64_t dim = elem - head * kHeadDim;
    const uint8_t *q_ptr = q_quant + row * q_stride0 + head * q_stride1;
    q_deq[elem] = __float2bfloat16(dequant_fp8_e4m3(q_ptr[dim * q_stride2]));
  }
  for (int64_t head = tid; head < kNumHeads; head += blockDim.x) {
    weight_cache[head] =
        weights[row * weights_stride0 + head * weights_stride1];
  }
  __syncthreads();

  const int64_t row_start =
      load_index(cu_seqlen_ks, cu_seqlen_ks_kind, row * cu_seqlen_ks_stride);
  const int64_t row_end =
      load_index(cu_seqlen_ke, cu_seqlen_ke_kind, row * cu_seqlen_ke_stride);
  const int64_t raw_row_len = row_end - row_start;
  const int64_t row_len =
      raw_row_len < kMaxSeqLen ? (raw_row_len > 0 ? raw_row_len : 0)
                               : kMaxSeqLen;

  for (int64_t rel_pos = tid; rel_pos < kMaxSeqLen; rel_pos += blockDim.x) {
    float score = -INFINITY;
    if (rel_pos < row_len) {
      const int64_t abs_pos = row_start + rel_pos;
      if (abs_pos >= 0 && abs_pos < total_seq_lens) {
        const int64_t req_idx = load_index(
            token_to_seq, token_to_seq_kind, abs_pos * token_to_seq_stride);
        const int64_t req_start = load_index(
            cu_seq_lens, cu_seq_lens_kind, req_idx * cu_seq_lens_stride);
        const int64_t local_pos = abs_pos - req_start;
        const int64_t logical_block = local_pos / block_size;
        const int64_t pos_in_block = local_pos - logical_block * block_size;
        const int64_t physical_block =
            load_index(block_table, block_table_kind,
                       req_idx * block_table_stride + logical_block);

        if (physical_block >= 0 && physical_block < num_blocks &&
            pos_in_block >= 0 && pos_in_block < block_size) {
          const uint8_t *block_ptr = kv_cache + physical_block * block_stride;
          const uint8_t *k_ptr = block_ptr + pos_in_block * kHeadDim;
          const uint8_t *scale_ptr =
              block_ptr + block_size * kHeadDim + pos_in_block * kScaleBytes;
          const float k_scale = *reinterpret_cast<const float *>(scale_ptr);

          float accum = 0.0f;
          for (int64_t head = 0; head < kNumHeads; ++head) {
            const __mt_bfloat16 *q_ptr = q_deq + head * kHeadDim;
            float dot = 0.0f;
#pragma unroll 4
            for (int64_t dim = 0; dim < kHeadDim; ++dim) {
              dot += __bfloat162float(q_ptr[dim]) *
                     dequant_fp8_e4m3(k_ptr[dim]);
            }
            accum += fmaxf(dot, 0.0f) * weight_cache[head];
          }
          score = accum * k_scale;
        }
      }
    }
    scores[rel_pos] = score;
  }
  __syncthreads();

  const int64_t effective_topk =
      topk < row_len ? (topk < kMaxTopK ? topk : kMaxTopK)
                     : (row_len < kMaxTopK ? row_len : kMaxTopK);

  for (int64_t rank = 0; rank < effective_topk; ++rank) {
    float local_value = -INFINITY;
    int local_index = -1;
    for (int64_t rel_pos = tid; rel_pos < row_len; rel_pos += blockDim.x) {
      const float value = scores[rel_pos];
      if (better_pair(value, static_cast<int>(rel_pos), local_value,
                      local_index)) {
        local_value = value;
        local_index = static_cast<int>(rel_pos);
      }
    }
    reduce_values[tid] = local_value;
    reduce_indices[tid] = local_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other_value = reduce_values[tid + stride];
        const int other_index = reduce_indices[tid + stride];
        if (better_pair(other_value, other_index, reduce_values[tid],
                        reduce_indices[tid])) {
          reduce_values[tid] = other_value;
          reduce_indices[tid] = other_index;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      const int selected = reduce_indices[0];
      topk_indices[row * topk_stride0 + rank * topk_stride1] =
          static_cast<OutT>(selected);
      if (selected >= 0) {
        scores[selected] = -INFINITY;
      }
    }
    __syncthreads();
  }

  for (int64_t rank = effective_topk + tid; rank < topk; rank += blockDim.x) {
    topk_indices[row * topk_stride0 + rank * topk_stride1] =
        static_cast<OutT>(-1);
  }
}

template <typename OutT>
__global__ void deepseek_v4_indexer_topk_prefill_q_cache_blockselect_kernel(
    const uint8_t *__restrict__ q_quant, int64_t q_stride0, int64_t q_stride1,
    int64_t q_stride2, const uint8_t *__restrict__ kv_cache,
    int64_t num_blocks, int64_t block_size, int64_t block_stride,
    const float *__restrict__ weights, int64_t weights_stride0,
    int64_t weights_stride1, const void *__restrict__ block_table,
    int block_table_kind, int64_t block_table_stride,
    const void *__restrict__ cu_seq_lens, int cu_seq_lens_kind,
    int64_t cu_seq_lens_stride, const void *__restrict__ token_to_seq,
    int token_to_seq_kind, int64_t token_to_seq_stride,
    const void *__restrict__ cu_seqlen_ks, int cu_seqlen_ks_kind,
    int64_t cu_seqlen_ks_stride, const void *__restrict__ cu_seqlen_ke,
    int cu_seqlen_ke_kind, int64_t cu_seqlen_ke_stride,
    OutT *__restrict__ topk_indices, int64_t topk_stride0,
    int64_t topk_stride1, int64_t rows, int64_t total_seq_lens,
    int64_t topk) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ __mt_bfloat16 q_deq[kNumHeads * kHeadDim];
  __shared__ float weight_cache[kNumHeads];
  __shared__ float scores[kMaxSeqLen];
  __shared__ int score_indices[kMaxSeqLen];

  const int tid = threadIdx.x;
  for (int64_t elem = tid; elem < kNumHeads * kHeadDim; elem += blockDim.x) {
    const int64_t head = elem / kHeadDim;
    const int64_t dim = elem - head * kHeadDim;
    const uint8_t *q_ptr = q_quant + row * q_stride0 + head * q_stride1;
    q_deq[elem] = __float2bfloat16(dequant_fp8_e4m3(q_ptr[dim * q_stride2]));
  }
  for (int64_t head = tid; head < kNumHeads; head += blockDim.x) {
    weight_cache[head] =
        weights[row * weights_stride0 + head * weights_stride1];
  }
  __syncthreads();

  const int64_t row_start =
      load_index(cu_seqlen_ks, cu_seqlen_ks_kind, row * cu_seqlen_ks_stride);
  const int64_t row_end =
      load_index(cu_seqlen_ke, cu_seqlen_ke_kind, row * cu_seqlen_ke_stride);
  const int64_t raw_row_len = row_end - row_start;
  const int64_t row_len =
      raw_row_len < kMaxSeqLen ? (raw_row_len > 0 ? raw_row_len : 0)
                               : kMaxSeqLen;

  for (int64_t rel_pos = tid; rel_pos < kMaxSeqLen; rel_pos += blockDim.x) {
    float score = -INFINITY;
    if (rel_pos < row_len) {
      const int64_t abs_pos = row_start + rel_pos;
      if (abs_pos >= 0 && abs_pos < total_seq_lens) {
        const int64_t req_idx = load_index(
            token_to_seq, token_to_seq_kind, abs_pos * token_to_seq_stride);
        const int64_t req_start = load_index(
            cu_seq_lens, cu_seq_lens_kind, req_idx * cu_seq_lens_stride);
        const int64_t local_pos = abs_pos - req_start;
        const int64_t logical_block = local_pos / block_size;
        const int64_t pos_in_block = local_pos - logical_block * block_size;
        const int64_t physical_block =
            load_index(block_table, block_table_kind,
                       req_idx * block_table_stride + logical_block);

        if (physical_block >= 0 && physical_block < num_blocks &&
            pos_in_block >= 0 && pos_in_block < block_size) {
          const uint8_t *block_ptr = kv_cache + physical_block * block_stride;
          const uint8_t *k_ptr = block_ptr + pos_in_block * kHeadDim;
          const uint8_t *scale_ptr =
              block_ptr + block_size * kHeadDim + pos_in_block * kScaleBytes;
          const float k_scale = *reinterpret_cast<const float *>(scale_ptr);

          float accum = 0.0f;
          for (int64_t head = 0; head < kNumHeads; ++head) {
            const __mt_bfloat16 *q_ptr = q_deq + head * kHeadDim;
            float dot = 0.0f;
#pragma unroll 4
            for (int64_t dim = 0; dim < kHeadDim; ++dim) {
              dot += __bfloat162float(q_ptr[dim]) *
                     dequant_fp8_e4m3(k_ptr[dim]);
            }
            accum += fmaxf(dot, 0.0f) * weight_cache[head];
          }
          score = accum * k_scale;
        }
      }
    }
    scores[rel_pos] = score;
    score_indices[rel_pos] = static_cast<int>(rel_pos);
  }
  __syncthreads();

  for (int size = 2; size <= kMaxSeqLen; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int pos = tid; pos < kMaxSeqLen; pos += blockDim.x) {
        const int other = pos ^ stride;
        if (other > pos) {
          const bool descending = (pos & size) == 0;
          const float pos_value = scores[pos];
          const int pos_index = score_indices[pos];
          const float other_value = scores[other];
          const int other_index = score_indices[other];
          const bool should_swap =
              (descending &&
               better_pair(other_value, other_index, pos_value, pos_index)) ||
              (!descending &&
               better_pair(pos_value, pos_index, other_value, other_index));
          if (should_swap) {
            scores[pos] = other_value;
            score_indices[pos] = other_index;
            scores[other] = pos_value;
            score_indices[other] = pos_index;
          }
        }
      }
      __syncthreads();
    }
  }

  const int64_t effective_topk =
      topk < row_len ? (topk < kMaxTopK ? topk : kMaxTopK)
                     : (row_len < kMaxTopK ? row_len : kMaxTopK);
  for (int64_t rank = tid; rank < topk; rank += blockDim.x) {
    const int selected = rank < effective_topk ? score_indices[rank] : -1;
    topk_indices[row * topk_stride0 + rank * topk_stride1] =
        static_cast<OutT>(selected);
  }
}

template <typename OutT, bool HoistMergeBarriers>
__global__ void deepseek_v4_indexer_topk_prefill_q_cache_partialsort_kernel(
    const uint8_t *__restrict__ q_quant, int64_t q_stride0, int64_t q_stride1,
    int64_t q_stride2, const uint8_t *__restrict__ kv_cache,
    int64_t num_blocks, int64_t block_size, int64_t block_stride,
    const float *__restrict__ weights, int64_t weights_stride0,
    int64_t weights_stride1, const void *__restrict__ block_table,
    int block_table_kind, int64_t block_table_stride,
    const void *__restrict__ cu_seq_lens, int cu_seq_lens_kind,
    int64_t cu_seq_lens_stride, const void *__restrict__ token_to_seq,
    int token_to_seq_kind, int64_t token_to_seq_stride,
    const void *__restrict__ cu_seqlen_ks, int cu_seqlen_ks_kind,
    int64_t cu_seqlen_ks_stride, const void *__restrict__ cu_seqlen_ke,
    int cu_seqlen_ke_kind, int64_t cu_seqlen_ke_stride,
    OutT *__restrict__ topk_indices, int64_t topk_stride0,
    int64_t topk_stride1, int64_t rows, int64_t total_seq_lens,
    int64_t topk, bool full_row_shortcut) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ __mt_bfloat16 q_deq[kNumHeads * kHeadDim];
  __shared__ float weight_cache[kNumHeads];
  __shared__ float scores[kMaxSeqLen];
  __shared__ int score_indices[kMaxSeqLen];

  const int tid = threadIdx.x;
  const int64_t row_start =
      load_index(cu_seqlen_ks, cu_seqlen_ks_kind, row * cu_seqlen_ks_stride);
  const int64_t row_end =
      load_index(cu_seqlen_ke, cu_seqlen_ke_kind, row * cu_seqlen_ke_stride);
  const int64_t raw_row_len = row_end - row_start;
  const int64_t row_len =
      raw_row_len < kMaxSeqLen ? (raw_row_len > 0 ? raw_row_len : 0)
                               : kMaxSeqLen;

  if (full_row_shortcut && row_len <= topk) {
    for (int64_t rank = tid; rank < topk; rank += blockDim.x) {
      const int64_t selected = rank < row_len ? rank : -1;
      topk_indices[row * topk_stride0 + rank * topk_stride1] =
          static_cast<OutT>(selected);
    }
    return;
  }

  for (int64_t elem = tid; elem < kNumHeads * kHeadDim; elem += blockDim.x) {
    const int64_t head = elem / kHeadDim;
    const int64_t dim = elem - head * kHeadDim;
    const uint8_t *q_ptr = q_quant + row * q_stride0 + head * q_stride1;
    q_deq[elem] = __float2bfloat16(dequant_fp8_e4m3(q_ptr[dim * q_stride2]));
  }
  for (int64_t head = tid; head < kNumHeads; head += blockDim.x) {
    weight_cache[head] =
        weights[row * weights_stride0 + head * weights_stride1];
  }
  __syncthreads();

  for (int64_t rel_pos = tid; rel_pos < kMaxSeqLen; rel_pos += blockDim.x) {
    float score = -INFINITY;
    if (rel_pos < row_len) {
      const int64_t abs_pos = row_start + rel_pos;
      if (abs_pos >= 0 && abs_pos < total_seq_lens) {
        const int64_t req_idx = load_index(
            token_to_seq, token_to_seq_kind, abs_pos * token_to_seq_stride);
        const int64_t req_start = load_index(
            cu_seq_lens, cu_seq_lens_kind, req_idx * cu_seq_lens_stride);
        const int64_t local_pos = abs_pos - req_start;
        const int64_t logical_block = local_pos / block_size;
        const int64_t pos_in_block = local_pos - logical_block * block_size;
        const int64_t physical_block =
            load_index(block_table, block_table_kind,
                       req_idx * block_table_stride + logical_block);

        if (physical_block >= 0 && physical_block < num_blocks &&
            pos_in_block >= 0 && pos_in_block < block_size) {
          const uint8_t *block_ptr = kv_cache + physical_block * block_stride;
          const uint8_t *k_ptr = block_ptr + pos_in_block * kHeadDim;
          const uint8_t *scale_ptr =
              block_ptr + block_size * kHeadDim + pos_in_block * kScaleBytes;
          const float k_scale = *reinterpret_cast<const float *>(scale_ptr);

          float accum = 0.0f;
          for (int64_t head = 0; head < kNumHeads; ++head) {
            const __mt_bfloat16 *q_ptr = q_deq + head * kHeadDim;
            float dot = 0.0f;
#pragma unroll 4
            for (int64_t dim = 0; dim < kHeadDim; ++dim) {
              dot += __bfloat162float(q_ptr[dim]) *
                     dequant_fp8_e4m3(k_ptr[dim]);
            }
            accum += fmaxf(dot, 0.0f) * weight_cache[head];
          }
          score = accum * k_scale;
        }
      }
    }
    scores[rel_pos] = score;
    score_indices[rel_pos] = static_cast<int>(rel_pos);
  }
  __syncthreads();

  if (row_len == kMaxSeqLen && topk == kMaxTopK) {
    for (int size = 2; size <= kMaxTopK; size <<= 1) {
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        for (int pos = tid; pos < kMaxSeqLen; pos += blockDim.x) {
          const int other = pos ^ stride;
          if (other > pos) {
            compare_swap_score_pair(scores, score_indices, pos, other,
                                    (pos & size) == 0);
          }
        }
        __syncthreads();
      }
    }

    if constexpr (HoistMergeBarriers) {
      for (int merge_size = kMaxTopK << 1; merge_size <= kMaxSeqLen;
           merge_size <<= 1) {
        const int half = merge_size >> 1;
        for (int offset = tid; offset < kMaxTopK; offset += blockDim.x) {
          for (int group_start = 0; group_start < kMaxSeqLen;
               group_start += merge_size) {
            compare_swap_score_pair(scores, score_indices, group_start + offset,
                                    group_start + half + offset, true);
          }
        }
        __syncthreads();

        for (int stride = kMaxTopK >> 1; stride > 0; stride >>= 1) {
          for (int offset = tid; offset < kMaxTopK; offset += blockDim.x) {
            const int other = offset ^ stride;
            if (other > offset) {
              for (int group_start = 0; group_start < kMaxSeqLen;
                   group_start += merge_size) {
                const bool output_descending = (group_start & merge_size) == 0;
                compare_swap_score_pair(scores, score_indices,
                                        group_start + offset,
                                        group_start + other, output_descending);
              }
            }
          }
          __syncthreads();
        }
      }
    } else {
      for (int merge_size = kMaxTopK << 1; merge_size <= kMaxSeqLen;
           merge_size <<= 1) {
        const int half = merge_size >> 1;
        for (int group_start = 0; group_start < kMaxSeqLen;
             group_start += merge_size) {
          for (int offset = tid; offset < kMaxTopK; offset += blockDim.x) {
            compare_swap_score_pair(scores, score_indices, group_start + offset,
                                    group_start + half + offset, true);
          }
          __syncthreads();

          const bool output_descending = (group_start & merge_size) == 0;
          for (int stride = kMaxTopK >> 1; stride > 0; stride >>= 1) {
            for (int offset = tid; offset < kMaxTopK; offset += blockDim.x) {
              const int other = offset ^ stride;
              if (other > offset) {
                compare_swap_score_pair(scores, score_indices,
                                        group_start + offset,
                                        group_start + other, output_descending);
              }
            }
            __syncthreads();
          }
        }
      }
    }
  } else {
    for (int size = 2; size <= kMaxSeqLen; size <<= 1) {
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        for (int pos = tid; pos < kMaxSeqLen; pos += blockDim.x) {
          const int other = pos ^ stride;
          if (other > pos) {
            compare_swap_score_pair(scores, score_indices, pos, other,
                                    (pos & size) == 0);
          }
        }
        __syncthreads();
      }
    }
  }

  const int64_t effective_topk =
      topk < row_len ? (topk < kMaxTopK ? topk : kMaxTopK)
                     : (row_len < kMaxTopK ? row_len : kMaxTopK);
  for (int64_t rank = tid; rank < topk; rank += blockDim.x) {
    const int selected = rank < effective_topk ? score_indices[rank] : -1;
    topk_indices[row * topk_stride0 + rank * topk_stride1] =
        static_cast<OutT>(selected);
  }
}

template <typename OutT>
__global__ void deepseek_v4_indexer_rerank_prefill_kernel(
    const uint8_t *__restrict__ q_quant, int64_t q_stride0, int64_t q_stride1,
    int64_t q_stride2, const uint8_t *__restrict__ kv_cache,
    int64_t num_blocks, int64_t block_size, int64_t block_stride,
    const float *__restrict__ weights, int64_t weights_stride0,
    int64_t weights_stride1, const void *__restrict__ block_table,
    int block_table_kind, int64_t block_table_stride,
    const void *__restrict__ cu_seq_lens, int cu_seq_lens_kind,
    int64_t cu_seq_lens_stride, const void *__restrict__ token_to_seq,
    int token_to_seq_kind, int64_t token_to_seq_stride,
    const void *__restrict__ cu_seqlen_ks, int cu_seqlen_ks_kind,
    int64_t cu_seqlen_ks_stride, const void *__restrict__ cu_seqlen_ke,
    int cu_seqlen_ke_kind, int64_t cu_seqlen_ke_stride,
    const void *__restrict__ candidate_abs_indices, int candidate_kind,
    int64_t candidate_stride0, int64_t candidate_stride1,
    OutT *__restrict__ topk_indices, int64_t topk_stride0,
    int64_t topk_stride1, int64_t rows, int64_t total_seq_lens,
    int64_t candidate_width, int64_t topk, bool full_row_shortcut) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ __mt_bfloat16 q_deq[kNumHeads * kHeadDim];
  __shared__ float weight_cache[kNumHeads];
  __shared__ float scores[kMaxCandidates];
  __shared__ int score_indices[kMaxCandidates];

  const int tid = threadIdx.x;
  const int64_t row_start =
      load_index(cu_seqlen_ks, cu_seqlen_ks_kind, row * cu_seqlen_ks_stride);
  const int64_t row_end =
      load_index(cu_seqlen_ke, cu_seqlen_ke_kind, row * cu_seqlen_ke_stride);
  const int64_t raw_row_len = row_end - row_start;
  const int64_t row_len =
      raw_row_len < kMaxSeqLen ? (raw_row_len > 0 ? raw_row_len : 0)
                               : kMaxSeqLen;

  if (full_row_shortcut && row_len <= topk) {
    for (int64_t rank = tid; rank < topk; rank += blockDim.x) {
      const int64_t selected = rank < row_len ? rank : -1;
      topk_indices[row * topk_stride0 + rank * topk_stride1] =
          static_cast<OutT>(selected);
    }
    return;
  }

  for (int64_t elem = tid; elem < kNumHeads * kHeadDim; elem += blockDim.x) {
    const int64_t head = elem / kHeadDim;
    const int64_t dim = elem - head * kHeadDim;
    const uint8_t *q_ptr = q_quant + row * q_stride0 + head * q_stride1;
    q_deq[elem] = __float2bfloat16(dequant_fp8_e4m3(q_ptr[dim * q_stride2]));
  }
  for (int64_t head = tid; head < kNumHeads; head += blockDim.x) {
    weight_cache[head] =
        weights[row * weights_stride0 + head * weights_stride1];
  }
  __syncthreads();

  for (int64_t slot = tid; slot < kMaxCandidates; slot += blockDim.x) {
    float score = -INFINITY;
    int rel_index = static_cast<int>(kMaxSeqLen + slot);
    if (slot < candidate_width && slot < kMaxCandidates) {
      const int64_t abs_pos =
          load_index(candidate_abs_indices, candidate_kind,
                     row * candidate_stride0 + slot * candidate_stride1);
      const int64_t rel_pos = abs_pos - row_start;
      if (rel_pos >= 0 && rel_pos < row_len && abs_pos >= 0 &&
          abs_pos < total_seq_lens) {
        const int64_t req_idx = load_index(
            token_to_seq, token_to_seq_kind, abs_pos * token_to_seq_stride);
        const int64_t req_start = load_index(
            cu_seq_lens, cu_seq_lens_kind, req_idx * cu_seq_lens_stride);
        const int64_t local_pos = abs_pos - req_start;
        const int64_t logical_block = local_pos / block_size;
        const int64_t pos_in_block = local_pos - logical_block * block_size;
        const int64_t physical_block =
            load_index(block_table, block_table_kind,
                       req_idx * block_table_stride + logical_block);

        if (physical_block >= 0 && physical_block < num_blocks &&
            pos_in_block >= 0 && pos_in_block < block_size) {
          const uint8_t *block_ptr = kv_cache + physical_block * block_stride;
          const uint8_t *k_ptr = block_ptr + pos_in_block * kHeadDim;
          const uint8_t *scale_ptr =
              block_ptr + block_size * kHeadDim + pos_in_block * kScaleBytes;
          const float k_scale = *reinterpret_cast<const float *>(scale_ptr);

          float accum = 0.0f;
          for (int64_t head = 0; head < kNumHeads; ++head) {
            const __mt_bfloat16 *q_ptr = q_deq + head * kHeadDim;
            float dot = 0.0f;
#pragma unroll 4
            for (int64_t dim = 0; dim < kHeadDim; ++dim) {
              dot += __bfloat162float(q_ptr[dim]) *
                     dequant_fp8_e4m3(k_ptr[dim]);
            }
            accum += fmaxf(dot, 0.0f) * weight_cache[head];
          }
          score = accum * k_scale;
          rel_index = static_cast<int>(rel_pos);
        }
      }
    }
    scores[slot] = score;
    score_indices[slot] = rel_index;
  }
  __syncthreads();

  for (int size = 2; size <= kMaxCandidates; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int pos = tid; pos < kMaxCandidates; pos += blockDim.x) {
        const int other = pos ^ stride;
        if (other > pos) {
          compare_swap_score_pair(scores, score_indices, pos, other,
                                  (pos & size) == 0);
        }
      }
      __syncthreads();
    }
  }

  const int64_t effective_topk =
      topk < row_len ? (topk < kMaxTopK ? topk : kMaxTopK)
                     : (row_len < kMaxTopK ? row_len : kMaxTopK);
  for (int64_t rank = tid; rank < topk; rank += blockDim.x) {
    const int candidate = rank < effective_topk ? score_indices[rank] : -1;
    const int selected =
        candidate >= 0 && candidate < row_len ? candidate : -1;
    topk_indices[row * topk_stride0 + rank * topk_stride1] =
        static_cast<OutT>(selected);
  }
}

template <typename OutT>
void launch_indexer_topk_decode(const torch::Tensor &q_quant,
                                const torch::Tensor &kv_cache,
                                const torch::Tensor &weights,
                                const torch::Tensor &seq_lens,
                                const torch::Tensor &block_table,
                                torch::Tensor &topk_indices, int64_t topk,
                                musaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(q_quant.size(0)));
  const dim3 block(kThreads);
  deepseek_v4_indexer_topk_decode_kernel<OutT><<<grid, block, 0, stream>>>(
      static_cast<const uint8_t *>(q_quant.data_ptr()), q_quant.stride(0),
      q_quant.stride(1), q_quant.stride(2),
      static_cast<const uint8_t *>(kv_cache.data_ptr()), kv_cache.size(0),
      kv_cache.size(1), kv_cache.stride(0),
      static_cast<const float *>(weights.data_ptr()), weights.stride(0),
      weights.stride(1), seq_lens.data_ptr(), index_kind(seq_lens, "seq_lens"),
      seq_lens.stride(0), block_table.data_ptr(),
      index_kind(block_table, "block_table"), block_table.stride(0),
      static_cast<OutT *>(topk_indices.data_ptr()), topk_indices.stride(0),
      topk_indices.stride(1), q_quant.size(0), topk);
}

template <typename OutT>
void launch_indexer_topk_prefill(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &block_table,
    const torch::Tensor &cu_seq_lens, const torch::Tensor &token_to_seq,
    const torch::Tensor &cu_seqlen_ks, const torch::Tensor &cu_seqlen_ke,
    torch::Tensor &topk_indices, int64_t topk, bool use_q_cache,
    musaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(q_quant.size(0)));
  const dim3 block(kThreads);
  // The q-cache + block-select + partial-sort pipeline is the validated
  // DeepSeek-V4 prefill path.  Keep it as a compile-time/default decision so
  // production dispatch cannot silently fall back when an A/B env is absent.
  const bool use_blockselect = use_q_cache;
  const bool use_partialsort = use_blockselect;
  constexpr bool use_partialsort_merge_barrier = true;
  const bool use_full_row_shortcut = use_partialsort;
  if (use_partialsort) {
    if (use_partialsort_merge_barrier) {
      deepseek_v4_indexer_topk_prefill_q_cache_partialsort_kernel<OutT, true>
          <<<grid, block, 0, stream>>>(
              static_cast<const uint8_t *>(q_quant.data_ptr()), q_quant.stride(0),
              q_quant.stride(1), q_quant.stride(2),
              static_cast<const uint8_t *>(kv_cache.data_ptr()), kv_cache.size(0),
              kv_cache.size(1), kv_cache.stride(0),
              static_cast<const float *>(weights.data_ptr()), weights.stride(0),
              weights.stride(1), block_table.data_ptr(),
              index_kind(block_table, "block_table"), block_table.stride(0),
              cu_seq_lens.data_ptr(), index_kind(cu_seq_lens, "cu_seq_lens"),
              cu_seq_lens.stride(0), token_to_seq.data_ptr(),
              index_kind(token_to_seq, "token_to_seq"), token_to_seq.stride(0),
              cu_seqlen_ks.data_ptr(), index_kind(cu_seqlen_ks, "cu_seqlen_ks"),
              cu_seqlen_ks.stride(0), cu_seqlen_ke.data_ptr(),
              index_kind(cu_seqlen_ke, "cu_seqlen_ke"), cu_seqlen_ke.stride(0),
              static_cast<OutT *>(topk_indices.data_ptr()),
              topk_indices.stride(0), topk_indices.stride(1), q_quant.size(0),
              token_to_seq.numel(), topk, use_full_row_shortcut);
    } else {
      deepseek_v4_indexer_topk_prefill_q_cache_partialsort_kernel<OutT, false>
          <<<grid, block, 0, stream>>>(
              static_cast<const uint8_t *>(q_quant.data_ptr()), q_quant.stride(0),
              q_quant.stride(1), q_quant.stride(2),
              static_cast<const uint8_t *>(kv_cache.data_ptr()), kv_cache.size(0),
              kv_cache.size(1), kv_cache.stride(0),
              static_cast<const float *>(weights.data_ptr()), weights.stride(0),
              weights.stride(1), block_table.data_ptr(),
              index_kind(block_table, "block_table"), block_table.stride(0),
              cu_seq_lens.data_ptr(), index_kind(cu_seq_lens, "cu_seq_lens"),
              cu_seq_lens.stride(0), token_to_seq.data_ptr(),
              index_kind(token_to_seq, "token_to_seq"), token_to_seq.stride(0),
              cu_seqlen_ks.data_ptr(), index_kind(cu_seqlen_ks, "cu_seqlen_ks"),
              cu_seqlen_ks.stride(0), cu_seqlen_ke.data_ptr(),
              index_kind(cu_seqlen_ke, "cu_seqlen_ke"), cu_seqlen_ke.stride(0),
              static_cast<OutT *>(topk_indices.data_ptr()),
              topk_indices.stride(0), topk_indices.stride(1), q_quant.size(0),
              token_to_seq.numel(), topk, use_full_row_shortcut);
    }
  } else if (use_blockselect) {
    deepseek_v4_indexer_topk_prefill_q_cache_blockselect_kernel<OutT>
        <<<grid, block, 0, stream>>>(
            static_cast<const uint8_t *>(q_quant.data_ptr()), q_quant.stride(0),
            q_quant.stride(1), q_quant.stride(2),
            static_cast<const uint8_t *>(kv_cache.data_ptr()), kv_cache.size(0),
            kv_cache.size(1), kv_cache.stride(0),
            static_cast<const float *>(weights.data_ptr()), weights.stride(0),
            weights.stride(1), block_table.data_ptr(),
            index_kind(block_table, "block_table"), block_table.stride(0),
            cu_seq_lens.data_ptr(), index_kind(cu_seq_lens, "cu_seq_lens"),
            cu_seq_lens.stride(0), token_to_seq.data_ptr(),
            index_kind(token_to_seq, "token_to_seq"), token_to_seq.stride(0),
            cu_seqlen_ks.data_ptr(), index_kind(cu_seqlen_ks, "cu_seqlen_ks"),
            cu_seqlen_ks.stride(0), cu_seqlen_ke.data_ptr(),
            index_kind(cu_seqlen_ke, "cu_seqlen_ke"), cu_seqlen_ke.stride(0),
            static_cast<OutT *>(topk_indices.data_ptr()),
            topk_indices.stride(0), topk_indices.stride(1), q_quant.size(0),
            token_to_seq.numel(), topk);
  } else if (use_q_cache) {
    deepseek_v4_indexer_topk_prefill_q_cache_kernel<OutT>
        <<<grid, block, 0, stream>>>(
            static_cast<const uint8_t *>(q_quant.data_ptr()), q_quant.stride(0),
            q_quant.stride(1), q_quant.stride(2),
            static_cast<const uint8_t *>(kv_cache.data_ptr()), kv_cache.size(0),
            kv_cache.size(1), kv_cache.stride(0),
            static_cast<const float *>(weights.data_ptr()), weights.stride(0),
            weights.stride(1), block_table.data_ptr(),
            index_kind(block_table, "block_table"), block_table.stride(0),
            cu_seq_lens.data_ptr(), index_kind(cu_seq_lens, "cu_seq_lens"),
            cu_seq_lens.stride(0), token_to_seq.data_ptr(),
            index_kind(token_to_seq, "token_to_seq"), token_to_seq.stride(0),
            cu_seqlen_ks.data_ptr(), index_kind(cu_seqlen_ks, "cu_seqlen_ks"),
            cu_seqlen_ks.stride(0), cu_seqlen_ke.data_ptr(),
            index_kind(cu_seqlen_ke, "cu_seqlen_ke"), cu_seqlen_ke.stride(0),
            static_cast<OutT *>(topk_indices.data_ptr()),
            topk_indices.stride(0), topk_indices.stride(1), q_quant.size(0),
            token_to_seq.numel(), topk);
  } else {
    deepseek_v4_indexer_topk_prefill_kernel<OutT><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t *>(q_quant.data_ptr()), q_quant.stride(0),
        q_quant.stride(1), q_quant.stride(2),
        static_cast<const uint8_t *>(kv_cache.data_ptr()), kv_cache.size(0),
        kv_cache.size(1), kv_cache.stride(0),
        static_cast<const float *>(weights.data_ptr()), weights.stride(0),
        weights.stride(1), block_table.data_ptr(),
        index_kind(block_table, "block_table"), block_table.stride(0),
        cu_seq_lens.data_ptr(), index_kind(cu_seq_lens, "cu_seq_lens"),
        cu_seq_lens.stride(0), token_to_seq.data_ptr(),
        index_kind(token_to_seq, "token_to_seq"), token_to_seq.stride(0),
        cu_seqlen_ks.data_ptr(), index_kind(cu_seqlen_ks, "cu_seqlen_ks"),
        cu_seqlen_ks.stride(0), cu_seqlen_ke.data_ptr(),
        index_kind(cu_seqlen_ke, "cu_seqlen_ke"), cu_seqlen_ke.stride(0),
        static_cast<OutT *>(topk_indices.data_ptr()), topk_indices.stride(0),
        topk_indices.stride(1), q_quant.size(0), token_to_seq.numel(), topk);
  }
}

template <typename OutT>
void launch_indexer_rerank_prefill(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &block_table,
    const torch::Tensor &cu_seq_lens, const torch::Tensor &token_to_seq,
    const torch::Tensor &cu_seqlen_ks, const torch::Tensor &cu_seqlen_ke,
    const torch::Tensor &candidate_abs_indices, torch::Tensor &topk_indices,
    int64_t topk, musaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(q_quant.size(0)));
  const dim3 block(kThreads);
  constexpr bool use_full_row_shortcut = true;
  deepseek_v4_indexer_rerank_prefill_kernel<OutT><<<grid, block, 0, stream>>>(
      static_cast<const uint8_t *>(q_quant.data_ptr()), q_quant.stride(0),
      q_quant.stride(1), q_quant.stride(2),
      static_cast<const uint8_t *>(kv_cache.data_ptr()), kv_cache.size(0),
      kv_cache.size(1), kv_cache.stride(0),
      static_cast<const float *>(weights.data_ptr()), weights.stride(0),
      weights.stride(1), block_table.data_ptr(),
      index_kind(block_table, "block_table"), block_table.stride(0),
      cu_seq_lens.data_ptr(), index_kind(cu_seq_lens, "cu_seq_lens"),
      cu_seq_lens.stride(0), token_to_seq.data_ptr(),
      index_kind(token_to_seq, "token_to_seq"), token_to_seq.stride(0),
      cu_seqlen_ks.data_ptr(), index_kind(cu_seqlen_ks, "cu_seqlen_ks"),
      cu_seqlen_ks.stride(0), cu_seqlen_ke.data_ptr(),
      index_kind(cu_seqlen_ke, "cu_seqlen_ke"), cu_seqlen_ke.stride(0),
      candidate_abs_indices.data_ptr(),
      index_kind(candidate_abs_indices, "candidate_abs_indices"),
      candidate_abs_indices.stride(0), candidate_abs_indices.stride(1),
      static_cast<OutT *>(topk_indices.data_ptr()), topk_indices.stride(0),
      topk_indices.stride(1), q_quant.size(0), token_to_seq.numel(),
      candidate_abs_indices.size(1), topk, use_full_row_shortcut);
}

} // namespace

void deepseek_v4_indexer_topk_decode(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &seq_lens,
    const torch::Tensor &block_table, torch::Tensor &topk_indices,
    int64_t topk) {
  check_musa_tensor(q_quant, "q_quant");
  check_same_device(q_quant, kv_cache, "kv_cache");
  check_same_device(q_quant, weights, "weights");
  check_same_device(q_quant, seq_lens, "seq_lens");
  check_same_device(q_quant, block_table, "block_table");
  check_same_device(q_quant, topk_indices, "topk_indices");

  TORCH_CHECK(q_quant.scalar_type() == torch::kFloat8_e4m3fn,
              "q_quant must be float8_e4m3fn");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8,
              "kv_cache must be uint8");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32,
              "weights must be float32");
  TORCH_CHECK(q_quant.dim() == 3, "q_quant must be [rows, 64, 128]");
  TORCH_CHECK(q_quant.size(1) == kNumHeads && q_quant.size(2) == kHeadDim,
              "q_quant must be [rows, 64, 128]");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == q_quant.size(0) &&
                  weights.size(1) == kNumHeads,
              "weights must be [rows, 64]");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) >= q_quant.size(0),
              "seq_lens must be 1-D with at least one entry per row");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) >= q_quant.size(0),
              "block_table must be 2-D with at least one row per query");
  TORCH_CHECK(topk_indices.dim() == 2 &&
                  topk_indices.size(0) >= q_quant.size(0),
              "topk_indices must be 2-D with at least one row per query");
  TORCH_CHECK(topk >= 0 && topk <= topk_indices.size(1),
              "topk must fit topk_indices width");
  TORCH_CHECK(topk <= kMaxTopK, "topk > 512 is not supported");
  TORCH_CHECK(kv_cache.dim() >= 2 && kv_cache.size(1) > 0,
              "kv_cache must include a non-empty block dimension");
  TORCH_CHECK(block_table.size(1) * kv_cache.size(1) <= kMaxSeqLen,
              "deepseek_v4_indexer_topk_decode supports max sequence length "
              "4096");
  TORCH_CHECK(kv_cache.stride(-1) == 1,
              "kv_cache byte dimension must be contiguous");
  TORCH_CHECK(kv_cache.stride(0) >=
                  kv_cache.size(1) * (kHeadDim + kScaleBytes),
              "kv_cache block stride is too small for indexer FP8 layout");
  TORCH_CHECK(q_quant.stride(2) == 1, "q_quant last dimension must be contiguous");
  TORCH_CHECK(topk_indices.stride(1) == 1,
              "topk_indices last dimension must be contiguous");
  index_kind(seq_lens, "seq_lens");
  index_kind(block_table, "block_table");

  if (q_quant.size(0) == 0 || topk == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(q_quant));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  if (topk_indices.scalar_type() == torch::kInt32) {
    launch_indexer_topk_decode<int32_t>(
        q_quant, kv_cache, weights, seq_lens, block_table, topk_indices, topk,
        stream);
  } else if (topk_indices.scalar_type() == torch::kInt64) {
    launch_indexer_topk_decode<int64_t>(
        q_quant, kv_cache, weights, seq_lens, block_table, topk_indices, topk,
        stream);
  } else {
    TORCH_CHECK(false, "topk_indices must be int32 or int64");
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_indexer_topk_decode launch failed: ",
              musaGetErrorString(err));
}

void deepseek_v4_indexer_topk_prefill(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &block_table,
    const torch::Tensor &cu_seq_lens, const torch::Tensor &token_to_seq,
    const torch::Tensor &cu_seqlen_ks, const torch::Tensor &cu_seqlen_ke,
    torch::Tensor &topk_indices, int64_t topk) {
  check_musa_tensor(q_quant, "q_quant");
  check_same_device(q_quant, kv_cache, "kv_cache");
  check_same_device(q_quant, weights, "weights");
  check_same_device(q_quant, block_table, "block_table");
  check_same_device(q_quant, cu_seq_lens, "cu_seq_lens");
  check_same_device(q_quant, token_to_seq, "token_to_seq");
  check_same_device(q_quant, cu_seqlen_ks, "cu_seqlen_ks");
  check_same_device(q_quant, cu_seqlen_ke, "cu_seqlen_ke");
  check_same_device(q_quant, topk_indices, "topk_indices");

  TORCH_CHECK(q_quant.scalar_type() == torch::kFloat8_e4m3fn,
              "q_quant must be float8_e4m3fn");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8,
              "kv_cache must be uint8");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32,
              "weights must be float32");
  TORCH_CHECK(q_quant.dim() == 3, "q_quant must be [rows, 64, 128]");
  TORCH_CHECK(q_quant.size(1) == kNumHeads && q_quant.size(2) == kHeadDim,
              "q_quant must be [rows, 64, 128]");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == q_quant.size(0) &&
                  weights.size(1) == kNumHeads,
              "weights must be [rows, 64]");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-D");
  TORCH_CHECK(cu_seq_lens.dim() == 1, "cu_seq_lens must be 1-D");
  TORCH_CHECK(token_to_seq.dim() == 1, "token_to_seq must be 1-D");
  TORCH_CHECK(cu_seqlen_ks.dim() == 1 &&
                  cu_seqlen_ks.size(0) >= q_quant.size(0),
              "cu_seqlen_ks must be 1-D with at least one entry per row");
  TORCH_CHECK(cu_seqlen_ke.dim() == 1 &&
                  cu_seqlen_ke.size(0) >= q_quant.size(0),
              "cu_seqlen_ke must be 1-D with at least one entry per row");
  TORCH_CHECK(topk_indices.dim() == 2 &&
                  topk_indices.size(0) >= q_quant.size(0),
              "topk_indices must be 2-D with at least one row per query");
  TORCH_CHECK(topk >= 0 && topk <= topk_indices.size(1),
              "topk must fit topk_indices width");
  TORCH_CHECK(topk <= kMaxTopK, "topk > 512 is not supported");
  TORCH_CHECK(kv_cache.dim() >= 2 && kv_cache.size(1) > 0,
              "kv_cache must include a non-empty block dimension");
  TORCH_CHECK(kv_cache.stride(-1) == 1,
              "kv_cache byte dimension must be contiguous");
  TORCH_CHECK(kv_cache.stride(0) >=
                  kv_cache.size(1) * (kHeadDim + kScaleBytes),
              "kv_cache block stride is too small for indexer FP8 layout");
  TORCH_CHECK(q_quant.stride(2) == 1, "q_quant last dimension must be contiguous");
  TORCH_CHECK(topk_indices.stride(1) == 1,
              "topk_indices last dimension must be contiguous");
  index_kind(block_table, "block_table");
  index_kind(cu_seq_lens, "cu_seq_lens");
  index_kind(token_to_seq, "token_to_seq");
  index_kind(cu_seqlen_ks, "cu_seqlen_ks");
  index_kind(cu_seqlen_ke, "cu_seqlen_ke");

  if (q_quant.size(0) == 0 || topk == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(q_quant));
  constexpr bool use_q_cache = true;
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  if (topk_indices.scalar_type() == torch::kInt32) {
    launch_indexer_topk_prefill<int32_t>(
        q_quant, kv_cache, weights, block_table, cu_seq_lens, token_to_seq,
        cu_seqlen_ks, cu_seqlen_ke, topk_indices, topk, use_q_cache, stream);
  } else if (topk_indices.scalar_type() == torch::kInt64) {
    launch_indexer_topk_prefill<int64_t>(
        q_quant, kv_cache, weights, block_table, cu_seq_lens, token_to_seq,
        cu_seqlen_ks, cu_seqlen_ke, topk_indices, topk, use_q_cache, stream);
  } else {
    TORCH_CHECK(false, "topk_indices must be int32 or int64");
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_indexer_topk_prefill launch failed: ",
              musaGetErrorString(err));
}

void deepseek_v4_indexer_rerank_prefill(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &block_table,
    const torch::Tensor &cu_seq_lens, const torch::Tensor &token_to_seq,
    const torch::Tensor &cu_seqlen_ks, const torch::Tensor &cu_seqlen_ke,
    const torch::Tensor &candidate_abs_indices, torch::Tensor &topk_indices,
    int64_t topk) {
  check_musa_tensor(q_quant, "q_quant");
  check_same_device(q_quant, kv_cache, "kv_cache");
  check_same_device(q_quant, weights, "weights");
  check_same_device(q_quant, block_table, "block_table");
  check_same_device(q_quant, cu_seq_lens, "cu_seq_lens");
  check_same_device(q_quant, token_to_seq, "token_to_seq");
  check_same_device(q_quant, cu_seqlen_ks, "cu_seqlen_ks");
  check_same_device(q_quant, cu_seqlen_ke, "cu_seqlen_ke");
  check_same_device(q_quant, candidate_abs_indices, "candidate_abs_indices");
  check_same_device(q_quant, topk_indices, "topk_indices");

  TORCH_CHECK(q_quant.scalar_type() == torch::kFloat8_e4m3fn,
              "q_quant must be float8_e4m3fn");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8,
              "kv_cache must be uint8");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32,
              "weights must be float32");
  TORCH_CHECK(q_quant.dim() == 3, "q_quant must be [rows, 64, 128]");
  TORCH_CHECK(q_quant.size(1) == kNumHeads && q_quant.size(2) == kHeadDim,
              "q_quant must be [rows, 64, 128]");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == q_quant.size(0) &&
                  weights.size(1) == kNumHeads,
              "weights must be [rows, 64]");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-D");
  TORCH_CHECK(cu_seq_lens.dim() == 1, "cu_seq_lens must be 1-D");
  TORCH_CHECK(token_to_seq.dim() == 1, "token_to_seq must be 1-D");
  TORCH_CHECK(cu_seqlen_ks.dim() == 1 &&
                  cu_seqlen_ks.size(0) >= q_quant.size(0),
              "cu_seqlen_ks must be 1-D with at least one entry per row");
  TORCH_CHECK(cu_seqlen_ke.dim() == 1 &&
                  cu_seqlen_ke.size(0) >= q_quant.size(0),
              "cu_seqlen_ke must be 1-D with at least one entry per row");
  TORCH_CHECK(candidate_abs_indices.dim() == 2 &&
                  candidate_abs_indices.size(0) >= q_quant.size(0),
              "candidate_abs_indices must be 2-D with at least one row per query");
  TORCH_CHECK(candidate_abs_indices.size(1) <= kMaxCandidates,
              "candidate_abs_indices width > 1024 is not supported");
  TORCH_CHECK(topk_indices.dim() == 2 &&
                  topk_indices.size(0) >= q_quant.size(0),
              "topk_indices must be 2-D with at least one row per query");
  TORCH_CHECK(topk >= 0 && topk <= topk_indices.size(1),
              "topk must fit topk_indices width");
  TORCH_CHECK(topk <= candidate_abs_indices.size(1),
              "candidate_abs_indices width must be at least topk");
  TORCH_CHECK(topk <= kMaxTopK, "topk > 512 is not supported");
  TORCH_CHECK(kv_cache.dim() >= 2 && kv_cache.size(1) > 0,
              "kv_cache must include a non-empty block dimension");
  TORCH_CHECK(kv_cache.stride(-1) == 1,
              "kv_cache byte dimension must be contiguous");
  TORCH_CHECK(kv_cache.stride(0) >=
                  kv_cache.size(1) * (kHeadDim + kScaleBytes),
              "kv_cache block stride is too small for indexer FP8 layout");
  TORCH_CHECK(q_quant.stride(2) == 1, "q_quant last dimension must be contiguous");
  TORCH_CHECK(candidate_abs_indices.stride(1) == 1,
              "candidate_abs_indices last dimension must be contiguous");
  TORCH_CHECK(topk_indices.stride(1) == 1,
              "topk_indices last dimension must be contiguous");
  index_kind(block_table, "block_table");
  index_kind(cu_seq_lens, "cu_seq_lens");
  index_kind(token_to_seq, "token_to_seq");
  index_kind(cu_seqlen_ks, "cu_seqlen_ks");
  index_kind(cu_seqlen_ke, "cu_seqlen_ke");
  index_kind(candidate_abs_indices, "candidate_abs_indices");

  if (q_quant.size(0) == 0 || topk == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(q_quant));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  if (topk_indices.scalar_type() == torch::kInt32) {
    launch_indexer_rerank_prefill<int32_t>(
        q_quant, kv_cache, weights, block_table, cu_seq_lens, token_to_seq,
        cu_seqlen_ks, cu_seqlen_ke, candidate_abs_indices, topk_indices, topk,
        stream);
  } else if (topk_indices.scalar_type() == torch::kInt64) {
    launch_indexer_rerank_prefill<int64_t>(
        q_quant, kv_cache, weights, block_table, cu_seq_lens, token_to_seq,
        cu_seqlen_ks, cu_seqlen_ke, candidate_abs_indices, topk_indices, topk,
        stream);
  } else {
    TORCH_CHECK(false, "topk_indices must be int32 or int64");
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_indexer_rerank_prefill launch failed: ",
              musaGetErrorString(err));
}
