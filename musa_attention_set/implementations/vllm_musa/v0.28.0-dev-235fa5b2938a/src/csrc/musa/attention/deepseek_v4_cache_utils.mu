#include <algorithm>
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
constexpr int64_t kTokenDataBytes = kNopeDim + kRopeDim * 2;
constexpr int64_t kTokenScaleBytes = 8;
constexpr int64_t kQuantBlockSize = 64;
constexpr int64_t kSparsePrefillTopKAlignment = 128;

constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;

__device__ __forceinline__ int64_t load_index(const void *ptr, int kind,
                                              int64_t idx) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t *>(ptr)[idx]);
  }
  return static_cast<int64_t>(static_cast<const int64_t *>(ptr)[idx]);
}

template <typename T>
__device__ __forceinline__ int64_t load_typed_index(const T *ptr, int64_t idx) {
  return static_cast<int64_t>(ptr[idx]);
}

__device__ __forceinline__ float dequant_fp8_e4m3(uint8_t byte,
                                                  uint8_t encoded_scale) {
  __mt_fp8_e4m3 packed;
  packed.__x = byte;
  return static_cast<float>(packed) *
         exp2f(static_cast<float>(encoded_scale) - 127.0f);
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

int64_t padded_topk(int64_t topk, int64_t window_size) {
  const int64_t raw = topk + window_size;
  return ((raw + kSparsePrefillTopKAlignment - 1) /
          kSparsePrefillTopKAlignment) *
         kSparsePrefillTopKAlignment;
}

void check_musa_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.device().is_privateuseone(), name,
              " must be a MUSA tensor");
}

void check_same_device(const torch::Tensor &a, const torch::Tensor &b,
                       const char *b_name) {
  TORCH_CHECK(a.device() == b.device(), b_name, " must be on the same device");
}

__global__ void deepseek_v4_dequantize_and_gather_k_cache_kernel(
    __mt_bfloat16 *__restrict__ out, int64_t out_stride0, int64_t out_stride1,
    const uint8_t *__restrict__ k_cache, const void *__restrict__ seq_lens,
    int seq_lens_kind, const void *__restrict__ gather_lens,
    int gather_lens_kind, const void *__restrict__ block_table,
    int block_table_kind, int64_t block_table_stride, int64_t num_reqs,
    int64_t num_blocks, int64_t block_size, int64_t block_stride,
    int64_t offset) {
  const int64_t req_idx = static_cast<int64_t>(blockIdx.x);
  if (req_idx >= num_reqs) {
    return;
  }

  const int64_t seq_len = load_index(seq_lens, seq_lens_kind, req_idx);
  const int64_t gather_len =
      gather_lens == nullptr
          ? seq_len
          : load_index(gather_lens, gather_lens_kind, req_idx);
  const int64_t start_pos = seq_len - gather_len;

  for (int64_t i = threadIdx.x; i < gather_len; i += blockDim.x) {
    const int64_t pos = start_pos + i;
    const int64_t block_in_seq = pos / block_size;
    const int64_t pos_in_block = pos - block_in_seq * block_size;
    const int64_t physical_block_idx =
        load_index(block_table, block_table_kind,
                   req_idx * block_table_stride + block_in_seq);
    if (physical_block_idx < 0 || physical_block_idx >= num_blocks) {
      continue;
    }

    const uint8_t *block_ptr = k_cache + physical_block_idx * block_stride;
    const uint8_t *token_ptr = block_ptr + pos_in_block * kTokenDataBytes;
    const uint8_t *scale_ptr = block_ptr + block_size * kTokenDataBytes +
                               pos_in_block * kTokenScaleBytes;
    __mt_bfloat16 *output =
        out + req_idx * out_stride0 + (offset + i) * out_stride1;

    for (int64_t qblock = 0; qblock < kNopeDim / kQuantBlockSize; ++qblock) {
      const int64_t start = qblock * kQuantBlockSize;
      const uint8_t scale = scale_ptr[qblock];
      for (int64_t j = 0; j < kQuantBlockSize; ++j) {
        output[start + j] =
            __float2bfloat16(dequant_fp8_e4m3(token_ptr[start + j], scale));
      }
    }

    const __mt_bfloat16 *rope =
        reinterpret_cast<const __mt_bfloat16 *>(token_ptr + kNopeDim);
    for (int64_t j = 0; j < kRopeDim; ++j) {
      output[kNopeDim + j] = rope[j];
    }
  }
}

template <typename TopKType, typename ReqType, typename BlockType>
__global__ void deepseek_v4_compute_global_topk_indices_and_lens_kernel(
    TopKType *__restrict__ global_topk_indices, int32_t *__restrict__ topk_lens,
    const TopKType *__restrict__ topk_indices,
    const ReqType *__restrict__ token_to_req_indices,
    const BlockType *__restrict__ block_table,
    const bool *__restrict__ is_valid_token, int64_t topk_stride,
    int64_t block_table_stride, int64_t num_tokens, int64_t topk,
    int64_t block_size) {
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
  if (token_idx >= num_tokens) {
    return;
  }

  __shared__ int counts[256];
  int count = 0;
  const int64_t req_idx = load_typed_index(token_to_req_indices, token_idx);

  for (int64_t i = threadIdx.x; i < topk; i += blockDim.x) {
    const int64_t local_idx =
        load_typed_index(topk_indices, token_idx * topk_stride + i);
    int64_t slot_id = -1;
    if (local_idx >= 0) {
      const int64_t block_idx = local_idx / block_size;
      const int64_t block_offset = local_idx - block_idx * block_size;
      const int64_t block_number = load_typed_index(
          block_table, req_idx * block_table_stride + block_idx);
      slot_id = block_number * block_size + block_offset;
      count += 1;
    }
    global_topk_indices[token_idx * topk_stride + i] =
        static_cast<TopKType>(slot_id);
  }

  counts[threadIdx.x] = count;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      counts[threadIdx.x] += counts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    topk_lens[token_idx] = is_valid_token[token_idx] ? counts[0] : 0;
  }
}

template <typename TopKType>
__global__ void deepseek_v4_combine_topk_swa_indices_kernel(
    int32_t *__restrict__ combined_indices, int32_t *__restrict__ combined_lens,
    const TopKType *__restrict__ topk_indices,
    const void *__restrict__ query_start_loc, int query_start_loc_kind,
    const void *__restrict__ seq_lens, int seq_lens_kind,
    const void *__restrict__ gather_lens, int gather_lens_kind,
    int64_t combined_stride, int64_t topk_stride, int64_t num_reqs,
    int64_t combined_topk, int64_t window_size, int64_t compress_ratio,
    int64_t topk, int64_t m, int64_t n) {
  const int64_t batch_idx = static_cast<int64_t>(blockIdx.x);
  if (batch_idx >= num_reqs) {
    return;
  }

  const int64_t base = load_index(query_start_loc, query_start_loc_kind, 0);
  const int64_t query_start =
      load_index(query_start_loc, query_start_loc_kind, batch_idx) - base;
  const int64_t query_end =
      load_index(query_start_loc, query_start_loc_kind, batch_idx + 1) - base;
  const int64_t query_len = query_end - query_start;
  const int64_t seq_len = load_index(seq_lens, seq_lens_kind, batch_idx);
  const int64_t gather_len =
      load_index(gather_lens, gather_lens_kind, batch_idx);
  const int64_t start_pos = seq_len - query_len;
  const int64_t gather_start = seq_len - gather_len;
  const int64_t req_offset = m * batch_idx;

  for (int64_t token_idx = query_start + threadIdx.x; token_idx < query_end;
       token_idx += blockDim.x) {
    int32_t *combined_row = combined_indices + token_idx * combined_stride;
    for (int64_t i = 0; i < combined_topk; ++i) {
      combined_row[i] = -1;
    }

    const int64_t token_idx_in_query = token_idx - query_start;
    const int64_t pos = start_pos + token_idx_in_query;
    const int64_t raw_topk_len = (pos + 1) / compress_ratio;
    const int64_t topk_len = raw_topk_len < topk ? raw_topk_len : topk;
    const int64_t swa_len = (pos + 1) < window_size ? (pos + 1) : window_size;

    for (int64_t i = 0; i < topk_len; ++i) {
      combined_row[i] = static_cast<int32_t>(
          load_typed_index(topk_indices, token_idx * topk_stride + i) +
          req_offset);
    }
    for (int64_t i = 0; i < swa_len; ++i) {
      combined_row[topk_len + i] = static_cast<int32_t>(
          req_offset + n + i + pos - swa_len + 1 - gather_start);
    }
    combined_lens[token_idx] = static_cast<int32_t>(topk_len + swa_len);
  }
}

template <typename TopKType, typename ReqType, typename BlockType>
std::tuple<torch::Tensor, torch::Tensor>
launch_compute_global_topk_indices_and_lens(
    const torch::Tensor &topk_indices,
    const torch::Tensor &token_to_req_indices, const torch::Tensor &block_table,
    int64_t block_size, const torch::Tensor &is_valid_token,
    musaStream_t stream) {
  auto global_topk_indices = torch::empty_like(topk_indices);
  auto topk_lens = torch::empty({topk_indices.size(0)},
                                topk_indices.options().dtype(torch::kInt32));
  if (topk_indices.size(0) == 0) {
    return std::make_tuple(global_topk_indices, topk_lens);
  }

  constexpr int kThreads = 256;
  deepseek_v4_compute_global_topk_indices_and_lens_kernel<<<
      static_cast<unsigned int>(topk_indices.size(0)), kThreads, 0, stream>>>(
      static_cast<TopKType *>(global_topk_indices.data_ptr()),
      static_cast<int32_t *>(topk_lens.data_ptr()),
      static_cast<const TopKType *>(topk_indices.data_ptr()),
      static_cast<const ReqType *>(token_to_req_indices.data_ptr()),
      static_cast<const BlockType *>(block_table.data_ptr()),
      static_cast<const bool *>(is_valid_token.data_ptr()),
      topk_indices.stride(0), block_table.stride(0), topk_indices.size(0),
      topk_indices.size(1), block_size);
  return std::make_tuple(global_topk_indices, topk_lens);
}

template <typename TopKType>
std::tuple<torch::Tensor, torch::Tensor> launch_combine_topk_swa_indices(
    const torch::Tensor &topk_indices, const torch::Tensor &query_start_loc,
    const torch::Tensor &seq_lens, const torch::Tensor &gather_lens,
    int64_t window_size, int64_t compress_ratio, int64_t topk, int64_t m,
    int64_t n, musaStream_t stream) {
  const int64_t combined_topk = padded_topk(topk, window_size);
  auto combined_indices =
      torch::empty({topk_indices.size(0), combined_topk},
                   topk_indices.options().dtype(torch::kInt32));
  auto combined_lens = torch::empty(
      {topk_indices.size(0)}, topk_indices.options().dtype(torch::kInt32));
  if (topk_indices.size(0) == 0) {
    return std::make_tuple(combined_indices, combined_lens);
  }

  constexpr int kThreads = 256;
  deepseek_v4_combine_topk_swa_indices_kernel<<<
      static_cast<unsigned int>(seq_lens.size(0)), kThreads, 0, stream>>>(
      static_cast<int32_t *>(combined_indices.data_ptr()),
      static_cast<int32_t *>(combined_lens.data_ptr()),
      static_cast<const TopKType *>(topk_indices.data_ptr()),
      query_start_loc.data_ptr(),
      index_kind(query_start_loc, "query_start_loc"), seq_lens.data_ptr(),
      index_kind(seq_lens, "seq_lens"), gather_lens.data_ptr(),
      index_kind(gather_lens, "gather_lens"), combined_indices.stride(0),
      topk_indices.stride(0), seq_lens.size(0), combined_topk, window_size,
      compress_ratio, topk, m, n);
  return std::make_tuple(combined_indices, combined_lens);
}

} // namespace

void deepseek_v4_dequantize_and_gather_k_cache(
    torch::Tensor &out, const torch::Tensor &k_cache,
    const torch::Tensor &seq_lens,
    const c10::optional<torch::Tensor> &gather_lens,
    const torch::Tensor &block_table, int64_t block_size, int64_t offset) {
  check_musa_tensor(out, "out");
  check_same_device(out, k_cache, "k_cache");
  check_same_device(out, seq_lens, "seq_lens");
  check_same_device(out, block_table, "block_table");
  if (gather_lens.has_value()) {
    check_same_device(out, *gather_lens, "gather_lens");
  }
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(k_cache.scalar_type() == torch::kUInt8, "k_cache must be uint8");
  TORCH_CHECK(out.dim() == 3 && out.size(2) == kHeadDim,
              "out must be [num_reqs, max_tokens, 512]");
  TORCH_CHECK(k_cache.dim() >= 2, "k_cache must include block dimension");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1-D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-D");
  TORCH_CHECK(block_table.size(0) == seq_lens.size(0),
              "block_table and seq_lens batch mismatch");
  TORCH_CHECK(out.size(0) == seq_lens.size(0),
              "out and seq_lens batch mismatch");
  TORCH_CHECK(k_cache.stride(-1) == 1,
              "k_cache byte dimension must be contiguous");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(k_cache.stride(0) >=
                  block_size * kTokenDataBytes + block_size * kTokenScaleBytes,
              "k_cache block stride is too small");
  if (gather_lens.has_value()) {
    TORCH_CHECK(gather_lens->dim() == 1, "gather_lens must be 1-D");
    TORCH_CHECK(gather_lens->size(0) == seq_lens.size(0),
                "gather_lens and seq_lens batch mismatch");
  }

  if (seq_lens.size(0) == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(out));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  constexpr int kWorkers = 128;
  deepseek_v4_dequantize_and_gather_k_cache_kernel<<<
      static_cast<unsigned int>(seq_lens.size(0)), kWorkers, 0, stream>>>(
      static_cast<__mt_bfloat16 *>(out.data_ptr()), out.stride(0),
      out.stride(1), static_cast<const uint8_t *>(k_cache.data_ptr()),
      seq_lens.data_ptr(), index_kind(seq_lens, "seq_lens"),
      gather_lens.has_value() ? gather_lens->data_ptr() : nullptr,
      gather_lens.has_value() ? index_kind(*gather_lens, "gather_lens") : 0,
      block_table.data_ptr(), index_kind(block_table, "block_table"),
      block_table.stride(0), seq_lens.size(0), k_cache.size(0), block_size,
      k_cache.stride(0), offset);
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_dequantize_and_gather_k_cache launch failed: ",
              musaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor>
deepseek_v4_compute_global_topk_indices_and_lens(
    const torch::Tensor &topk_indices,
    const torch::Tensor &token_to_req_indices, const torch::Tensor &block_table,
    int64_t block_size, const torch::Tensor &is_valid_token) {
  check_musa_tensor(topk_indices, "topk_indices");
  check_same_device(topk_indices, token_to_req_indices, "token_to_req_indices");
  check_same_device(topk_indices, block_table, "block_table");
  check_same_device(topk_indices, is_valid_token, "is_valid_token");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2-D");
  TORCH_CHECK(token_to_req_indices.dim() == 1,
              "token_to_req_indices must be 1-D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-D");
  TORCH_CHECK(is_valid_token.dim() == 1, "is_valid_token must be 1-D");
  TORCH_CHECK(token_to_req_indices.size(0) >= topk_indices.size(0),
              "token_to_req_indices is shorter than topk_indices rows");
  TORCH_CHECK(is_valid_token.size(0) >= topk_indices.size(0),
              "is_valid_token is shorter than topk_indices rows");
  TORCH_CHECK(is_valid_token.scalar_type() == torch::kBool,
              "is_valid_token must be bool");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  index_kind(token_to_req_indices, "token_to_req_indices");
  index_kind(block_table, "block_table");

  const at::musa::OptionalMUSAGuard device_guard(device_of(topk_indices));
  musaStream_t stream = at::musa::getCurrentMUSAStream();

#define DISPATCH_BLOCK(TOPK_CPP, REQ_CPP, BLOCK_CPP)                           \
  return launch_compute_global_topk_indices_and_lens<TOPK_CPP, REQ_CPP,        \
                                                     BLOCK_CPP>(               \
      topk_indices, token_to_req_indices, block_table, block_size,             \
      is_valid_token, stream)

  if (topk_indices.scalar_type() == torch::kInt32 &&
      token_to_req_indices.scalar_type() == torch::kInt32 &&
      block_table.scalar_type() == torch::kInt32) {
    DISPATCH_BLOCK(int32_t, int32_t, int32_t);
  }
  if (topk_indices.scalar_type() == torch::kInt32 &&
      token_to_req_indices.scalar_type() == torch::kInt64 &&
      block_table.scalar_type() == torch::kInt32) {
    DISPATCH_BLOCK(int32_t, int64_t, int32_t);
  }
  if (topk_indices.scalar_type() == torch::kInt64 &&
      token_to_req_indices.scalar_type() == torch::kInt32 &&
      block_table.scalar_type() == torch::kInt32) {
    DISPATCH_BLOCK(int64_t, int32_t, int32_t);
  }
  if (topk_indices.scalar_type() == torch::kInt64 &&
      token_to_req_indices.scalar_type() == torch::kInt64 &&
      block_table.scalar_type() == torch::kInt32) {
    DISPATCH_BLOCK(int64_t, int64_t, int32_t);
  }
  if (topk_indices.scalar_type() == torch::kInt32 &&
      token_to_req_indices.scalar_type() == torch::kInt64 &&
      block_table.scalar_type() == torch::kInt64) {
    DISPATCH_BLOCK(int32_t, int64_t, int64_t);
  }
  if (topk_indices.scalar_type() == torch::kInt64 &&
      token_to_req_indices.scalar_type() == torch::kInt64 &&
      block_table.scalar_type() == torch::kInt64) {
    DISPATCH_BLOCK(int64_t, int64_t, int64_t);
  }

#undef DISPATCH_BLOCK
  TORCH_CHECK(false, "unsupported dtype combination for "
                     "deepseek_v4_compute_global_topk_indices_and_lens");
}

std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_combine_topk_swa_indices(
    const torch::Tensor &topk_indices, const torch::Tensor &query_start_loc,
    const torch::Tensor &seq_lens, const torch::Tensor &gather_lens,
    int64_t window_size, int64_t compress_ratio, int64_t topk, int64_t m,
    int64_t n) {
  check_musa_tensor(topk_indices, "topk_indices");
  check_same_device(topk_indices, query_start_loc, "query_start_loc");
  check_same_device(topk_indices, seq_lens, "seq_lens");
  check_same_device(topk_indices, gather_lens, "gather_lens");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2-D");
  TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be 1-D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1-D");
  TORCH_CHECK(gather_lens.dim() == 1, "gather_lens must be 1-D");
  TORCH_CHECK(query_start_loc.size(0) == seq_lens.size(0) + 1,
              "query_start_loc must have num_reqs + 1 entries");
  TORCH_CHECK(gather_lens.size(0) == seq_lens.size(0),
              "gather_lens and seq_lens batch mismatch");
  TORCH_CHECK(window_size >= 0, "window_size must be non-negative");
  TORCH_CHECK(compress_ratio > 0, "compress_ratio must be positive");
  TORCH_CHECK(topk >= 0, "topk must be non-negative");
  TORCH_CHECK(m >= 0 && n >= 0, "M and N must be non-negative");
  TORCH_CHECK(topk <= topk_indices.size(1),
              "topk exceeds topk_indices row width");
  index_kind(query_start_loc, "query_start_loc");
  index_kind(seq_lens, "seq_lens");
  index_kind(gather_lens, "gather_lens");

  const at::musa::OptionalMUSAGuard device_guard(device_of(topk_indices));
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  if (topk_indices.scalar_type() == torch::kInt32) {
    return launch_combine_topk_swa_indices<int32_t>(
        topk_indices, query_start_loc, seq_lens, gather_lens, window_size,
        compress_ratio, topk, m, n, stream);
  }
  if (topk_indices.scalar_type() == torch::kInt64) {
    return launch_combine_topk_swa_indices<int64_t>(
        topk_indices, query_start_loc, seq_lens, gather_lens, window_size,
        compress_ratio, topk, m, n, stream);
  }

  TORCH_CHECK(false, "topk_indices must be int32 or int64 for "
                     "deepseek_v4_combine_topk_swa_indices");
}
