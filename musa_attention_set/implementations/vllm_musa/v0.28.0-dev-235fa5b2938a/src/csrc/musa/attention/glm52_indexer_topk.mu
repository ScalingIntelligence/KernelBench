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

// GLM-5.2's lightning indexer shape.  Unlike the DeepSeek-V4 kernel, GLM uses
// 32 index heads and keeps 2048 tokens.  8192 scores plus the 2048-candidate
// sorting scratch fit in shared memory on mp_31 when the Q cache is aliased
// with the candidate scratch after scoring.
constexpr int64_t kHeadDim = 128;
constexpr int64_t kNumHeads = 32;
constexpr int64_t kScaleBytes = 4;
constexpr int64_t kMaxSeqLen = 8192;
constexpr int64_t kMaxTopK = 2048;
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

__device__ __forceinline__ uint32_t ordered_float(float value) {
  // Indexer scores are expected to be finite.  Treat a defensive NaN as the
  // lowest possible score rather than allowing it to poison radix ordering.
  if (isnan(value)) {
    value = -INFINITY;
  }
  // Normalize signed zero so equality and radix ordering use the same tie set.
  if (value == 0.0f) {
    value = 0.0f;
  }
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
}

union SelectionScratch {
  __mt_bfloat16 q_deq[kNumHeads * kHeadDim];
  struct {
    float scores[kMaxTopK];
    int indices[kMaxTopK];
  } candidates;
};

template <typename OutT>
__device__ __forceinline__ void fill_all_indices(
    OutT *__restrict__ topk_indices, int64_t topk_stride0,
    int64_t topk_stride1, int64_t row, int64_t row_len, int64_t topk) {
  for (int64_t rank = threadIdx.x; rank < topk; rank += blockDim.x) {
    topk_indices[row * topk_stride0 + rank * topk_stride1] =
        static_cast<OutT>(rank < row_len ? rank : -1);
  }
}

// Select and sort the largest topk entries from shared-memory scores.  The
// four radix passes are adapted from dashboard submission 10185, which passed
// the S5000 streaming-aware indexer benchmark through topk=2048.  After the
// threshold is known, only the selected 2048 candidates are bitonic-sorted so
// the output retains the descending-score / ascending-index contract used by
// the existing exact fallback.
template <typename OutT>
__device__ void radix_select_scores(
    float *__restrict__ scores, SelectionScratch *__restrict__ scratch,
    int *__restrict__ histogram, uint32_t *__restrict__ prefix,
    int *__restrict__ rank_in_bucket, int *__restrict__ tie_cutoff,
    int *__restrict__ candidate_count, OutT *__restrict__ topk_indices,
    int64_t topk_stride0, int64_t topk_stride1, int64_t row,
    int64_t row_len, int64_t topk) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    *prefix = 0U;
    *rank_in_bucket = static_cast<int>(topk);
    *tie_cutoff = -1;
    *candidate_count = 0;
  }
  __syncthreads();

#pragma unroll
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int bin = tid; bin < 256; bin += blockDim.x) {
      histogram[bin] = 0;
    }
    __syncthreads();

    const uint32_t current_prefix = *prefix;
    const uint32_t high_mask =
        shift == 24 ? 0U : (0xFFFFFFFFU << (shift + 8));
    for (int64_t pos = tid; pos < row_len; pos += blockDim.x) {
      const uint32_t key = ordered_float(scores[pos]);
      if ((key & high_mask) == (current_prefix & high_mask)) {
        atomicAdd(&histogram[(key >> shift) & 0xFFU], 1);
      }
    }
    __syncthreads();

    if (tid == 0) {
      int kth = *rank_in_bucket;
      uint32_t next_prefix = *prefix;
      for (int bin = 255; bin >= 0; --bin) {
        if (histogram[bin] >= kth) {
          next_prefix |= static_cast<uint32_t>(bin) << shift;
          *prefix = next_prefix;
          break;
        }
        kth -= histogram[bin];
      }
      *rank_in_bucket = kth;
    }
    __syncthreads();
  }

  const uint32_t threshold = *prefix;
  if (tid == 0) {
    int greater = 0;
    for (int64_t pos = 0; pos < row_len; ++pos) {
      greater += ordered_float(scores[pos]) > threshold ? 1 : 0;
    }
    int ties_needed = static_cast<int>(topk) - greater;
    int cutoff = -1;
    for (int64_t pos = 0; pos < row_len && ties_needed > 0; ++pos) {
      if (ordered_float(scores[pos]) == threshold) {
        cutoff = static_cast<int>(pos);
        --ties_needed;
      }
    }
    *tie_cutoff = cutoff;
  }

  float *candidate_scores = scratch->candidates.scores;
  int *candidate_indices = scratch->candidates.indices;
  for (int slot = tid; slot < kMaxTopK; slot += blockDim.x) {
    candidate_scores[slot] = -INFINITY;
    candidate_indices[slot] = -1;
  }
  __syncthreads();

  const int cutoff = *tie_cutoff;
  for (int64_t pos = tid; pos < row_len; pos += blockDim.x) {
    const uint32_t key = ordered_float(scores[pos]);
    if (key > threshold || (key == threshold && pos <= cutoff)) {
      const int slot = atomicAdd(candidate_count, 1);
      if (slot < kMaxTopK) {
        candidate_scores[slot] = scores[pos];
        candidate_indices[slot] = static_cast<int>(pos);
      }
    }
  }
  __syncthreads();

  for (int size = 2; size <= kMaxTopK; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int pos = tid; pos < kMaxTopK; pos += blockDim.x) {
        const int other = pos ^ stride;
        if (other > pos) {
          compare_swap_score_pair(candidate_scores, candidate_indices, pos,
                                  other, (pos & size) == 0);
        }
      }
      __syncthreads();
    }
  }

  for (int64_t rank = tid; rank < topk; rank += blockDim.x) {
    topk_indices[row * topk_stride0 + rank * topk_stride1] =
        static_cast<OutT>(candidate_indices[rank]);
  }
}

template <typename OutT>
__global__ void sparse_indexer_fill_all_kernel(
    const void *__restrict__ lengths, int lengths_kind, int64_t lengths_stride,
    OutT *__restrict__ topk_indices, int64_t topk_stride0,
    int64_t topk_stride1, int64_t rows, int64_t topk) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t raw_len =
      load_index(lengths, lengths_kind, row * lengths_stride);
  const int64_t row_len = raw_len > 0 ? raw_len : 0;
  fill_all_indices(topk_indices, topk_stride0, topk_stride1, row, row_len,
                   topk);
}

template <typename OutT>
__global__ void sparse_indexer_topk_kernel(
    const float *__restrict__ logits, int64_t logits_stride0,
    int64_t logits_stride1, int64_t columns, const void *__restrict__ row_starts,
    int row_starts_kind, int64_t row_starts_stride,
    const void *__restrict__ row_ends, int row_ends_kind,
    int64_t row_ends_stride, OutT *__restrict__ topk_indices,
    int64_t topk_stride0, int64_t topk_stride1, int64_t rows, int64_t topk,
    bool starts_at_zero) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ int histogram[256];
  __shared__ uint32_t prefix;
  __shared__ int rank_in_bucket;
  __shared__ int tie_cutoff;
  __shared__ int greater_count;
  __shared__ int minimum_tie_index;

  const int tid = threadIdx.x;
  int64_t row_start = starts_at_zero
                          ? 0
                          : load_index(row_starts, row_starts_kind,
                                       row * row_starts_stride);
  int64_t row_end = load_index(row_ends, row_ends_kind, row * row_ends_stride);
  row_start = row_start < 0 ? 0 : (row_start > columns ? columns : row_start);
  row_end = row_end < row_start ? row_start
                               : (row_end > columns ? columns : row_end);
  const int64_t row_len = row_end - row_start;
  if (row_len <= topk) {
    fill_all_indices(topk_indices, topk_stride0, topk_stride1, row, row_len,
                     topk);
    return;
  }

  if (tid == 0) {
    prefix = 0U;
    rank_in_bucket = static_cast<int>(topk);
    tie_cutoff = -1;
    greater_count = 0;
    minimum_tie_index = INT32_MAX;
  }
  __syncthreads();

  const float *row_logits = logits + row * logits_stride0;
#pragma unroll
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int bin = tid; bin < 256; bin += blockDim.x) {
      histogram[bin] = 0;
    }
    __syncthreads();

    const uint32_t current_prefix = prefix;
    const uint32_t high_mask =
        shift == 24 ? 0U : (0xFFFFFFFFU << (shift + 8));
    for (int64_t pos = row_start + tid; pos < row_end; pos += blockDim.x) {
      const uint32_t key = ordered_float(row_logits[pos * logits_stride1]);
      if ((key & high_mask) == (current_prefix & high_mask)) {
        atomicAdd(&histogram[(key >> shift) & 0xFFU], 1);
      }
    }
    __syncthreads();

    if (tid == 0) {
      int kth = rank_in_bucket;
      uint32_t next_prefix = prefix;
      for (int bin = 255; bin >= 0; --bin) {
        if (histogram[bin] >= kth) {
          next_prefix |= static_cast<uint32_t>(bin) << shift;
          prefix = next_prefix;
          break;
        }
        kth -= histogram[bin];
      }
      rank_in_bucket = kth;
    }
    __syncthreads();
  }

  const uint32_t threshold = prefix;
  int local_greater = 0;
  int local_ties = 0;
  int local_min_tie = INT32_MAX;
  for (int64_t pos = row_start + tid; pos < row_end; pos += blockDim.x) {
    const uint32_t key = ordered_float(row_logits[pos * logits_stride1]);
    if (key > threshold) {
      ++local_greater;
    } else if (key == threshold) {
      ++local_ties;
      local_min_tie = min(local_min_tie, static_cast<int>(pos));
    }
  }
  if (local_greater > 0) {
    atomicAdd(&greater_count, local_greater);
  }
  if (local_ties > 0) {
    atomicMin(&minimum_tie_index, local_min_tie);
  }
  __syncthreads();

  if (tid == 0) {
    int ties_needed = static_cast<int>(topk) - greater_count;
    if (ties_needed <= 0) {
      tie_cutoff = -1;
    } else if (ties_needed == 1) {
      tie_cutoff = minimum_tie_index;
    } else {
      // Large exact-tie groups are uncommon for weighted indexer scores.  Keep
      // a deterministic low-index fallback without penalizing the unique-key
      // fast path with a serial scan.
      int cutoff = -1;
      for (int64_t pos = row_start; pos < row_end && ties_needed > 0; ++pos) {
        if (ordered_float(row_logits[pos * logits_stride1]) == threshold) {
          cutoff = static_cast<int>(pos);
          --ties_needed;
        }
      }
      tie_cutoff = cutoff;
    }
  }
  __syncthreads();

  const int cutoff = tie_cutoff;
  int selected_count = 0;
  for (int64_t pos = row_start + tid; pos < row_end; pos += blockDim.x) {
    const float score = row_logits[pos * logits_stride1];
    const uint32_t key = ordered_float(score);
    if (key > threshold || (key == threshold && pos <= cutoff)) {
      ++selected_count;
    }
  }
  histogram[tid] = selected_count;
  __syncthreads();

  if (tid == 0) {
    int prefix_sum = 0;
    for (int lane = 0; lane < kThreads; ++lane) {
      const int lane_count = histogram[lane];
      histogram[lane] = prefix_sum;
      prefix_sum += lane_count;
    }
  }
  __syncthreads();

  // Sparse MLA consumes an index set, not a score-sorted list.  Compact each
  // thread's strided selections into a disjoint output range computed by the
  // block-wide prefix above.  This avoids contending on one atomic counter.
  int output_slot = histogram[tid];
  for (int64_t pos = row_start + tid; pos < row_end; pos += blockDim.x) {
    const uint32_t key = ordered_float(row_logits[pos * logits_stride1]);
    if (key > threshold || (key == threshold && pos <= cutoff)) {
      if (output_slot < topk) {
        topk_indices[row * topk_stride0 +
                     static_cast<int64_t>(output_slot) * topk_stride1] =
            static_cast<OutT>(pos - row_start);
      }
      ++output_slot;
    }
  }
}

template <typename OutT>
__global__ void glm52_indexer_topk_decode_kernel(
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
  __shared__ SelectionScratch scratch;
  __shared__ float weight_cache[kNumHeads];
  __shared__ int histogram[256];
  __shared__ uint32_t prefix;
  __shared__ int rank_in_bucket;
  __shared__ int tie_cutoff;
  __shared__ int candidate_count;

  const int tid = threadIdx.x;
  const int64_t raw_seq_len =
      load_index(seq_lens, seq_lens_kind, row * seq_lens_stride);
  const int64_t seq_len = raw_seq_len > 0 ? raw_seq_len : 0;
  if (seq_len <= topk) {
    fill_all_indices(topk_indices, topk_stride0, topk_stride1, row, seq_len,
                     topk);
    return;
  }
  if (seq_len > kMaxSeqLen) {
    // The host dispatcher prevents this path.  Keep the output visibly invalid
    // if metadata changes race a caller rather than silently truncating scores.
    fill_all_indices(topk_indices, topk_stride0, topk_stride1, row, 0, topk);
    return;
  }

  for (int64_t elem = tid; elem < kNumHeads * kHeadDim;
       elem += blockDim.x) {
    const int64_t head = elem / kHeadDim;
    const int64_t dim = elem - head * kHeadDim;
    const uint8_t *q_ptr = q_quant + row * q_stride0 + head * q_stride1;
    scratch.q_deq[elem] =
        __float2bfloat16(dequant_fp8_e4m3(q_ptr[dim * q_stride2]));
  }
  for (int64_t head = tid; head < kNumHeads; head += blockDim.x) {
    weight_cache[head] =
        weights[row * weights_stride0 + head * weights_stride1];
  }
  __syncthreads();

  for (int64_t pos = tid; pos < seq_len; pos += blockDim.x) {
    float score = -INFINITY;
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
        const __mt_bfloat16 *q_ptr = scratch.q_deq + head * kHeadDim;
        float dot = 0.0f;
#pragma unroll 4
        for (int64_t dim = 0; dim < kHeadDim; ++dim) {
          dot += __bfloat162float(q_ptr[dim]) * dequant_fp8_e4m3(k_ptr[dim]);
        }
        accum += fmaxf(dot, 0.0f) * weight_cache[head];
      }
      score = accum * k_scale;
    }
    scores[pos] = score;
  }
  __syncthreads();

  radix_select_scores(scores, &scratch, histogram, &prefix, &rank_in_bucket,
                      &tie_cutoff, &candidate_count, topk_indices,
                      topk_stride0, topk_stride1, row, seq_len, topk);
}

template <typename OutT>
__global__ void glm52_indexer_topk_prefill_kernel(
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
  __shared__ SelectionScratch scratch;
  __shared__ float weight_cache[kNumHeads];
  __shared__ int histogram[256];
  __shared__ uint32_t prefix;
  __shared__ int rank_in_bucket;
  __shared__ int tie_cutoff;
  __shared__ int candidate_count;

  const int tid = threadIdx.x;
  const int64_t row_start =
      load_index(cu_seqlen_ks, cu_seqlen_ks_kind, row * cu_seqlen_ks_stride);
  const int64_t row_end =
      load_index(cu_seqlen_ke, cu_seqlen_ke_kind, row * cu_seqlen_ke_stride);
  const int64_t raw_row_len = row_end - row_start;
  const int64_t row_len = raw_row_len > 0 ? raw_row_len : 0;
  if (row_len <= topk) {
    fill_all_indices(topk_indices, topk_stride0, topk_stride1, row, row_len,
                     topk);
    return;
  }
  if (row_len > kMaxSeqLen) {
    fill_all_indices(topk_indices, topk_stride0, topk_stride1, row, 0, topk);
    return;
  }

  for (int64_t elem = tid; elem < kNumHeads * kHeadDim;
       elem += blockDim.x) {
    const int64_t head = elem / kHeadDim;
    const int64_t dim = elem - head * kHeadDim;
    const uint8_t *q_ptr = q_quant + row * q_stride0 + head * q_stride1;
    scratch.q_deq[elem] =
        __float2bfloat16(dequant_fp8_e4m3(q_ptr[dim * q_stride2]));
  }
  for (int64_t head = tid; head < kNumHeads; head += blockDim.x) {
    weight_cache[head] =
        weights[row * weights_stride0 + head * weights_stride1];
  }
  __syncthreads();

  for (int64_t rel_pos = tid; rel_pos < row_len; rel_pos += blockDim.x) {
    float score = -INFINITY;
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
          const __mt_bfloat16 *q_ptr = scratch.q_deq + head * kHeadDim;
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
    scores[rel_pos] = score;
  }
  __syncthreads();

  radix_select_scores(scores, &scratch, histogram, &prefix, &rank_in_bucket,
                      &tie_cutoff, &candidate_count, topk_indices,
                      topk_stride0, topk_stride1, row, row_len, topk);
}

template <typename OutT>
void launch_fill_all(const torch::Tensor &lengths, torch::Tensor &topk_indices,
                     int64_t topk, musaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(lengths.numel()));
  const dim3 block(kThreads);
  sparse_indexer_fill_all_kernel<OutT><<<grid, block, 0, stream>>>(
      lengths.data_ptr(), index_kind(lengths, "lengths"), lengths.stride(-1),
      static_cast<OutT *>(topk_indices.data_ptr()), topk_indices.stride(0),
      topk_indices.stride(1), lengths.numel(), topk);
}

template <typename OutT>
void launch_sparse_indexer_topk(const torch::Tensor &logits,
                                const torch::Tensor &row_starts,
                                const torch::Tensor &row_ends,
                                torch::Tensor &topk_indices, int64_t topk,
                                musaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(logits.size(0)));
  const dim3 block(kThreads);
  sparse_indexer_topk_kernel<OutT><<<grid, block, 0, stream>>>(
      static_cast<const float *>(logits.data_ptr()), logits.stride(0),
      logits.stride(1), logits.size(1), row_starts.data_ptr(),
      index_kind(row_starts, "row_starts"), row_starts.stride(0),
      row_ends.data_ptr(), index_kind(row_ends, "row_ends"),
      row_ends.stride(0), static_cast<OutT *>(topk_indices.data_ptr()),
      topk_indices.stride(0), topk_indices.stride(1), logits.size(0), topk,
      false);
}

template <typename OutT>
void launch_sparse_indexer_topk_decode(const torch::Tensor &logits,
                                       const torch::Tensor &seq_lens,
                                       torch::Tensor &topk_indices,
                                       int64_t topk, musaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(logits.size(0)));
  const dim3 block(kThreads);
  sparse_indexer_topk_kernel<OutT><<<grid, block, 0, stream>>>(
      static_cast<const float *>(logits.data_ptr()), logits.stride(0),
      logits.stride(1), logits.size(1), nullptr, kIndexInt32, 0,
      seq_lens.data_ptr(), index_kind(seq_lens, "seq_lens"),
      seq_lens.stride(0), static_cast<OutT *>(topk_indices.data_ptr()),
      topk_indices.stride(0), topk_indices.stride(1), logits.size(0), topk,
      true);
}

template <typename OutT>
void launch_glm52_decode(const torch::Tensor &q_quant,
                         const torch::Tensor &kv_cache,
                         const torch::Tensor &weights,
                         const torch::Tensor &seq_lens,
                         const torch::Tensor &block_table,
                         torch::Tensor &topk_indices, int64_t topk,
                         musaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(q_quant.size(0)));
  const dim3 block(kThreads);
  glm52_indexer_topk_decode_kernel<OutT><<<grid, block, 0, stream>>>(
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
void launch_glm52_prefill(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &block_table,
    const torch::Tensor &cu_seq_lens, const torch::Tensor &token_to_seq,
    const torch::Tensor &cu_seqlen_ks, const torch::Tensor &cu_seqlen_ke,
    torch::Tensor &topk_indices, int64_t topk, musaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(q_quant.size(0)));
  const dim3 block(kThreads);
  glm52_indexer_topk_prefill_kernel<OutT><<<grid, block, 0, stream>>>(
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

void check_glm52_common(const torch::Tensor &q_quant,
                        const torch::Tensor &kv_cache,
                        const torch::Tensor &weights,
                        const torch::Tensor &topk_indices, int64_t topk) {
  check_musa_tensor(q_quant, "q_quant");
  check_same_device(q_quant, kv_cache, "kv_cache");
  check_same_device(q_quant, weights, "weights");
  check_same_device(q_quant, topk_indices, "topk_indices");
  TORCH_CHECK(q_quant.scalar_type() == torch::kFloat8_e4m3fn,
              "q_quant must be float8_e4m3fn");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8,
              "kv_cache must be uint8");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32,
              "weights must be float32");
  TORCH_CHECK(q_quant.dim() == 3 && q_quant.size(1) == kNumHeads &&
                  q_quant.size(2) == kHeadDim,
              "q_quant must be [rows, 32, 128]");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == q_quant.size(0) &&
                  weights.size(1) == kNumHeads,
              "weights must be [rows, 32]");
  TORCH_CHECK(topk_indices.dim() == 2 &&
                  topk_indices.size(0) >= q_quant.size(0),
              "topk_indices must have at least one row per query");
  TORCH_CHECK(topk >= 0 && topk <= topk_indices.size(1),
              "topk must fit topk_indices width");
  TORCH_CHECK(topk <= kMaxTopK, "topk > 2048 is not supported");
  TORCH_CHECK(kv_cache.dim() >= 2 && kv_cache.size(1) > 0,
              "kv_cache must include a non-empty block dimension");
  TORCH_CHECK(kv_cache.stride(-1) == 1,
              "kv_cache byte dimension must be contiguous");
  TORCH_CHECK(kv_cache.stride(0) >=
                  kv_cache.size(1) * (kHeadDim + kScaleBytes),
              "kv_cache block stride is too small for indexer FP8 layout");
  TORCH_CHECK(q_quant.stride(2) == 1,
              "q_quant last dimension must be contiguous");
  TORCH_CHECK(topk_indices.stride(1) == 1,
              "topk_indices last dimension must be contiguous");
}

} // namespace

void sparse_indexer_fill_all(const torch::Tensor &lengths,
                             torch::Tensor &topk_indices, int64_t topk) {
  check_musa_tensor(lengths, "lengths");
  check_same_device(lengths, topk_indices, "topk_indices");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be 1-D");
  TORCH_CHECK(topk_indices.dim() == 2 &&
                  topk_indices.size(0) >= lengths.numel(),
              "topk_indices must have at least one row per length");
  TORCH_CHECK(topk >= 0 && topk <= topk_indices.size(1),
              "topk must fit topk_indices width");
  index_kind(lengths, "lengths");
  if (lengths.numel() == 0 || topk == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(lengths));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  if (topk_indices.scalar_type() == torch::kInt32) {
    launch_fill_all<int32_t>(lengths, topk_indices, topk, stream);
  } else if (topk_indices.scalar_type() == torch::kInt64) {
    launch_fill_all<int64_t>(lengths, topk_indices, topk, stream);
  } else {
    TORCH_CHECK(false, "topk_indices must be int32 or int64");
  }
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess, "sparse_indexer_fill_all launch failed: ",
              musaGetErrorString(err));
}

void sparse_indexer_topk(const torch::Tensor &logits,
                         const torch::Tensor &row_starts,
                         const torch::Tensor &row_ends,
                         torch::Tensor &topk_indices, int64_t topk) {
  check_musa_tensor(logits, "logits");
  check_same_device(logits, row_starts, "row_starts");
  check_same_device(logits, row_ends, "row_ends");
  check_same_device(logits, topk_indices, "topk_indices");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D");
  TORCH_CHECK(row_starts.dim() == 1 && row_starts.numel() >= logits.size(0),
              "row_starts must have at least one entry per logit row");
  TORCH_CHECK(row_ends.dim() == 1 && row_ends.numel() >= logits.size(0),
              "row_ends must have at least one entry per logit row");
  TORCH_CHECK(topk_indices.dim() == 2 &&
                  topk_indices.size(0) >= logits.size(0),
              "topk_indices must have at least one output row per logit row");
  TORCH_CHECK(topk >= 0 && topk <= topk_indices.size(1),
              "topk must fit topk_indices width");
  TORCH_CHECK(topk <= kMaxTopK, "topk > 2048 is not supported");
  TORCH_CHECK(logits.stride(1) == 1,
              "logits last dimension must be contiguous");
  TORCH_CHECK(topk_indices.stride(1) == 1,
              "topk_indices last dimension must be contiguous");
  index_kind(row_starts, "row_starts");
  index_kind(row_ends, "row_ends");
  if (logits.size(0) == 0 || topk == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(logits));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  if (topk_indices.scalar_type() == torch::kInt32) {
    launch_sparse_indexer_topk<int32_t>(logits, row_starts, row_ends,
                                        topk_indices, topk, stream);
  } else if (topk_indices.scalar_type() == torch::kInt64) {
    launch_sparse_indexer_topk<int64_t>(logits, row_starts, row_ends,
                                        topk_indices, topk, stream);
  } else {
    TORCH_CHECK(false, "topk_indices must be int32 or int64");
  }
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess, "sparse_indexer_topk launch failed: ",
              musaGetErrorString(err));
}

void sparse_indexer_topk_decode(const torch::Tensor &logits,
                                const torch::Tensor &seq_lens,
                                torch::Tensor &topk_indices, int64_t topk) {
  check_musa_tensor(logits, "logits");
  check_same_device(logits, seq_lens, "seq_lens");
  check_same_device(logits, topk_indices, "topk_indices");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.numel() >= logits.size(0),
              "seq_lens must have at least one entry per logit row");
  TORCH_CHECK(topk_indices.dim() == 2 &&
                  topk_indices.size(0) >= logits.size(0),
              "topk_indices must have at least one output row per logit row");
  TORCH_CHECK(topk >= 0 && topk <= topk_indices.size(1),
              "topk must fit topk_indices width");
  TORCH_CHECK(topk <= kMaxTopK, "topk > 2048 is not supported");
  TORCH_CHECK(logits.stride(1) == 1,
              "logits last dimension must be contiguous");
  TORCH_CHECK(topk_indices.stride(1) == 1,
              "topk_indices last dimension must be contiguous");
  index_kind(seq_lens, "seq_lens");
  if (logits.size(0) == 0 || topk == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(logits));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  if (topk_indices.scalar_type() == torch::kInt32) {
    launch_sparse_indexer_topk_decode<int32_t>(logits, seq_lens,
                                               topk_indices, topk, stream);
  } else if (topk_indices.scalar_type() == torch::kInt64) {
    launch_sparse_indexer_topk_decode<int64_t>(logits, seq_lens,
                                               topk_indices, topk, stream);
  } else {
    TORCH_CHECK(false, "topk_indices must be int32 or int64");
  }
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "sparse_indexer_topk_decode launch failed: ",
              musaGetErrorString(err));
}

void glm52_indexer_topk_decode(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &seq_lens,
    const torch::Tensor &block_table, torch::Tensor &topk_indices,
    int64_t topk) {
  check_glm52_common(q_quant, kv_cache, weights, topk_indices, topk);
  check_same_device(q_quant, seq_lens, "seq_lens");
  check_same_device(q_quant, block_table, "block_table");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) >= q_quant.size(0),
              "seq_lens must be 1-D with at least one entry per row");
  TORCH_CHECK(block_table.dim() == 2 &&
                  block_table.size(0) >= q_quant.size(0),
              "block_table must be 2-D with at least one row per query");
  TORCH_CHECK(block_table.size(1) * kv_cache.size(1) <= kMaxSeqLen,
              "glm52_indexer_topk_decode supports max sequence length 8192");
  index_kind(seq_lens, "seq_lens");
  index_kind(block_table, "block_table");
  if (q_quant.size(0) == 0 || topk == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(q_quant));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  if (topk_indices.scalar_type() == torch::kInt32) {
    launch_glm52_decode<int32_t>(q_quant, kv_cache, weights, seq_lens,
                                 block_table, topk_indices, topk, stream);
  } else if (topk_indices.scalar_type() == torch::kInt64) {
    launch_glm52_decode<int64_t>(q_quant, kv_cache, weights, seq_lens,
                                 block_table, topk_indices, topk, stream);
  } else {
    TORCH_CHECK(false, "topk_indices must be int32 or int64");
  }
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "glm52_indexer_topk_decode launch failed: ",
              musaGetErrorString(err));
}

void glm52_indexer_topk_prefill(
    const torch::Tensor &q_quant, const torch::Tensor &kv_cache,
    const torch::Tensor &weights, const torch::Tensor &block_table,
    const torch::Tensor &cu_seq_lens, const torch::Tensor &token_to_seq,
    const torch::Tensor &cu_seqlen_ks, const torch::Tensor &cu_seqlen_ke,
    torch::Tensor &topk_indices, int64_t topk) {
  check_glm52_common(q_quant, kv_cache, weights, topk_indices, topk);
  check_same_device(q_quant, block_table, "block_table");
  check_same_device(q_quant, cu_seq_lens, "cu_seq_lens");
  check_same_device(q_quant, token_to_seq, "token_to_seq");
  check_same_device(q_quant, cu_seqlen_ks, "cu_seqlen_ks");
  check_same_device(q_quant, cu_seqlen_ke, "cu_seqlen_ke");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-D");
  TORCH_CHECK(cu_seq_lens.dim() == 1, "cu_seq_lens must be 1-D");
  TORCH_CHECK(token_to_seq.dim() == 1, "token_to_seq must be 1-D");
  TORCH_CHECK(cu_seqlen_ks.dim() == 1 &&
                  cu_seqlen_ks.size(0) >= q_quant.size(0),
              "cu_seqlen_ks must have at least one entry per row");
  TORCH_CHECK(cu_seqlen_ke.dim() == 1 &&
                  cu_seqlen_ke.size(0) >= q_quant.size(0),
              "cu_seqlen_ke must have at least one entry per row");
  TORCH_CHECK(token_to_seq.numel() <= kMaxSeqLen,
              "glm52_indexer_topk_prefill supports chunks up to 8192 tokens");
  index_kind(block_table, "block_table");
  index_kind(cu_seq_lens, "cu_seq_lens");
  index_kind(token_to_seq, "token_to_seq");
  index_kind(cu_seqlen_ks, "cu_seqlen_ks");
  index_kind(cu_seqlen_ke, "cu_seqlen_ke");
  if (q_quant.size(0) == 0 || topk == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(q_quant));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  if (topk_indices.scalar_type() == torch::kInt32) {
    launch_glm52_prefill<int32_t>(
        q_quant, kv_cache, weights, block_table, cu_seq_lens, token_to_seq,
        cu_seqlen_ks, cu_seqlen_ke, topk_indices, topk, stream);
  } else if (topk_indices.scalar_type() == torch::kInt64) {
    launch_glm52_prefill<int64_t>(
        q_quant, kv_cache, weights, block_table, cu_seq_lens, token_to_seq,
        cu_seqlen_ks, cu_seqlen_ke, topk_indices, topk, stream);
  } else {
    TORCH_CHECK(false, "topk_indices must be int32 or int64");
  }
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "glm52_indexer_topk_prefill launch failed: ",
              musaGetErrorString(err));
}
