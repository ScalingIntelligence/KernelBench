#include <cmath>
#include <cstdint>
#include <limits>
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
constexpr int kThreads = 256;

constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;

__device__ __forceinline__ int64_t load_index(const void *ptr, int kind,
                                              int64_t idx) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t *>(ptr)[idx]);
  }
  return static_cast<int64_t>(static_cast<const int64_t *>(ptr)[idx]);
}

__device__ __forceinline__ float load_bf16(const __mt_bfloat16 *ptr,
                                           int64_t idx) {
  return __bfloat162float(ptr[idx]);
}

__device__ __forceinline__ float dequant_fp8_e4m3(uint8_t byte,
                                                  uint8_t encoded_scale) {
  __mt_fp8_e4m3 packed;
  packed.__x = byte;
  return static_cast<float>(packed) *
         exp2f(static_cast<float>(encoded_scale) - 127.0f);
}

__device__ __forceinline__ float load_packed_cache_value(
    const uint8_t *__restrict__ cache, int64_t num_blocks, int64_t block_size,
    int64_t block_stride, int64_t slot, int64_t dim) {
  if (slot < 0 || slot >= num_blocks * block_size) {
    return 0.0f;
  }
  const int64_t block_idx = slot / block_size;
  const int64_t pos_in_block = slot - block_idx * block_size;
  const uint8_t *block_ptr = cache + block_idx * block_stride;
  const uint8_t *token_ptr = block_ptr + pos_in_block * kTokenDataBytes;
  if (dim < kNopeDim) {
    const uint8_t *scale_ptr =
        block_ptr + block_size * kTokenDataBytes +
        pos_in_block * kTokenScaleBytes;
    return dequant_fp8_e4m3(token_ptr[dim],
                            scale_ptr[dim / kQuantBlockSize]);
  }
  const __mt_bfloat16 *rope =
      reinterpret_cast<const __mt_bfloat16 *>(token_ptr + kNopeDim);
  return load_bf16(rope, dim - kNopeDim);
}

__device__ void process_sparse_slot(
    int64_t slot, const uint8_t *__restrict__ cache, int64_t num_blocks,
    int64_t block_size, int64_t block_stride, const __mt_bfloat16 *q_ptr,
    int64_t q_dim_stride, float softmax_scale, float *__restrict__ acc,
    float *__restrict__ kv_vec, float *__restrict__ reduce,
    float *__restrict__ denom_m, float *__restrict__ denom_l,
    float *__restrict__ key_m, float *__restrict__ key_l,
    float *__restrict__ alpha, float *__restrict__ beta) {
  if (slot < 0 || slot >= num_blocks * block_size) {
    return;
  }

  float partial = 0.0f;
  for (int64_t dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
    const float q_val = load_bf16(q_ptr, dim * q_dim_stride);
    const float kv_val =
        load_packed_cache_value(cache, num_blocks, block_size, block_stride,
                                slot, dim);
    kv_vec[dim] = kv_val;
    partial += q_val * kv_val;
  }
  reduce[threadIdx.x] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float logit = reduce[0] * softmax_scale;

    if (*key_l == 0.0f) {
      *key_m = logit;
      *key_l = 1.0f;
    } else {
      const float new_key_m = fmaxf(*key_m, logit);
      *key_l = *key_l * expf(*key_m - new_key_m) + expf(logit - new_key_m);
      *key_m = new_key_m;
    }

    if (*denom_l == 0.0f) {
      *alpha = 0.0f;
      *beta = 1.0f;
      *denom_m = logit;
      *denom_l = 1.0f;
    } else {
      const float new_m = fmaxf(*denom_m, logit);
      *alpha = expf(*denom_m - new_m);
      *beta = expf(logit - new_m);
      *denom_l = *denom_l * *alpha + *beta;
      *denom_m = new_m;
    }
  }
  __syncthreads();

  for (int64_t dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
    acc[dim] = acc[dim] * *alpha + kv_vec[dim] * *beta;
  }
  __syncthreads();
}

__global__ void deepseek_v4_sparse_flashmla_decode_kernel(
    const __mt_bfloat16 *__restrict__ q, int64_t q_stride0,
    int64_t q_stride1, int64_t q_stride2, int64_t q_stride3,
    __mt_bfloat16 *__restrict__ out, int64_t out_stride0,
    int64_t out_stride1, int64_t out_stride2, int64_t out_stride3,
    float *__restrict__ lse, const uint8_t *__restrict__ k_cache,
    int64_t k_num_blocks, int64_t k_block_size, int64_t k_block_stride,
    const void *__restrict__ indices, int index_kind, int64_t indices_stride0,
    int64_t indices_stride1, const void *__restrict__ topk_length,
    int topk_length_kind, int64_t topk, const uint8_t *__restrict__ extra_cache,
    int64_t extra_num_blocks, int64_t extra_block_size,
    int64_t extra_block_stride, const void *__restrict__ extra_indices,
    int extra_index_kind, int64_t extra_indices_stride0,
    int64_t extra_indices_stride1, const void *__restrict__ extra_topk_length,
    int extra_topk_length_kind, int64_t extra_topk, const float *attn_sink,
    int64_t batch, int64_t seq_len, int64_t num_heads, float softmax_scale) {
  const int64_t query_idx = static_cast<int64_t>(blockIdx.x);
  const int64_t head_idx = static_cast<int64_t>(blockIdx.y);
  if (query_idx >= batch * seq_len || head_idx >= num_heads) {
    return;
  }

  __shared__ float acc[kHeadDim];
  __shared__ float kv_vec[kHeadDim];
  __shared__ float reduce[kThreads];
  __shared__ float denom_m;
  __shared__ float denom_l;
  __shared__ float key_m;
  __shared__ float key_l;
  __shared__ float alpha;
  __shared__ float beta;

  const int64_t b = query_idx / seq_len;
  const int64_t s = query_idx - b * seq_len;
  const __mt_bfloat16 *q_ptr =
      q + b * q_stride0 + s * q_stride1 + head_idx * q_stride2;
  __mt_bfloat16 *out_ptr =
      out + b * out_stride0 + s * out_stride1 + head_idx * out_stride2;

  for (int64_t dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
    acc[dim] = 0.0f;
  }
  if (threadIdx.x == 0) {
    const float sink = attn_sink == nullptr ? -INFINITY : attn_sink[head_idx];
    if (isfinite(sink)) {
      denom_m = sink;
      denom_l = 1.0f;
    } else {
      denom_m = -INFINITY;
      denom_l = 0.0f;
    }
    key_m = -INFINITY;
    key_l = 0.0f;
    alpha = 0.0f;
    beta = 0.0f;
  }
  __syncthreads();

  int64_t main_len = topk;
  if (topk_length != nullptr) {
    const int64_t raw_len = load_index(topk_length, topk_length_kind, query_idx);
    main_len = raw_len < topk ? raw_len : topk;
    main_len = main_len > 0 ? main_len : 0;
  }
  for (int64_t i = 0; i < main_len; ++i) {
    const int64_t slot = load_index(indices, index_kind,
                                    query_idx * indices_stride0 +
                                        i * indices_stride1);
    process_sparse_slot(slot, k_cache, k_num_blocks, k_block_size,
                        k_block_stride, q_ptr, q_stride3, softmax_scale, acc,
                        kv_vec, reduce, &denom_m, &denom_l, &key_m, &key_l,
                        &alpha, &beta);
  }

  if (extra_cache != nullptr && extra_indices != nullptr) {
    int64_t extra_len = extra_topk;
    if (extra_topk_length != nullptr) {
      const int64_t raw_len =
          load_index(extra_topk_length, extra_topk_length_kind, query_idx);
      extra_len = raw_len < extra_topk ? raw_len : extra_topk;
      extra_len = extra_len > 0 ? extra_len : 0;
    }
    for (int64_t i = 0; i < extra_len; ++i) {
      const int64_t slot = load_index(extra_indices, extra_index_kind,
                                      query_idx * extra_indices_stride0 +
                                          i * extra_indices_stride1);
      process_sparse_slot(slot, extra_cache, extra_num_blocks, extra_block_size,
                          extra_block_stride, q_ptr, q_stride3, softmax_scale,
                          acc, kv_vec, reduce, &denom_m, &denom_l, &key_m,
                          &key_l, &alpha, &beta);
    }
  }

  const float inv_l = denom_l == 0.0f ? 0.0f : 1.0f / denom_l;
  for (int64_t dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
    const float value = acc[dim] * inv_l;
    out_ptr[dim * out_stride3] = __float2bfloat16(value);
  }
  if (threadIdx.x == 0) {
    const float key_lse =
        key_l == 0.0f ? INFINITY : key_m + logf(fmaxf(key_l, 1.0e-30f));
    lse[b * num_heads * seq_len + head_idx * seq_len + s] = key_lse;
  }
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

void check_packed_cache(const torch::Tensor &cache, const char *name) {
  check_musa_tensor(cache, name);
  TORCH_CHECK(cache.scalar_type() == torch::kUInt8, name, " must be uint8");
  TORCH_CHECK(cache.dim() == 4 || cache.dim() == 3, name,
              " must be [blocks, block, 1, bytes] or [blocks, block, bytes]");
  if (cache.dim() == 4) {
    TORCH_CHECK(cache.size(2) == 1, name,
                " must use one KV head for DeepSeek-V4 MLA");
  }
  TORCH_CHECK(cache.size(0) > 0 && cache.size(1) > 0, name,
              " must have non-empty blocks");
  TORCH_CHECK(cache.stride(-1) == 1, name,
              " byte dimension must be contiguous");
  const int64_t logical_block_bytes =
      cache.size(1) * (kTokenDataBytes + kTokenScaleBytes);
  TORCH_CHECK(cache.stride(0) >= logical_block_bytes, name,
              " block stride is smaller than packed fp8_ds_mla payload");
}

void check_flat_indices(const torch::Tensor &indices, const char *name,
                        int64_t num_queries) {
  check_musa_tensor(indices, name);
  TORCH_CHECK(indices.dim() == 2, name, " must be flattened to [queries, topk]");
  TORCH_CHECK(indices.size(0) == num_queries, name,
              " first dimension must match query count");
  TORCH_CHECK(indices.is_contiguous(), name, " must be contiguous");
  index_kind(indices, name);
}

void check_optional_lengths(const c10::optional<torch::Tensor> &lengths,
                            const char *name, int64_t num_queries) {
  if (!lengths.has_value()) {
    return;
  }
  const torch::Tensor &tensor = lengths.value();
  check_musa_tensor(tensor, name);
  TORCH_CHECK(tensor.numel() == num_queries, name,
              " must contain one length per query");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  index_kind(tensor, name);
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> deepseek_v4_sparse_flashmla_decode(
    const torch::Tensor &q, const torch::Tensor &k_cache,
    const torch::Tensor &indices,
    const c10::optional<torch::Tensor> &topk_length,
    const c10::optional<torch::Tensor> &attn_sink,
    const c10::optional<torch::Tensor> &extra_k_cache,
    const c10::optional<torch::Tensor> &extra_indices,
    const c10::optional<torch::Tensor> &extra_topk_length, torch::Tensor &out,
    double softmax_scale) {
  check_musa_tensor(q, "q");
  check_musa_tensor(out, "out");
  check_packed_cache(k_cache, "k_cache");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(q.dim() == 4, "q must be [batch, seq, heads, 512]");
  TORCH_CHECK(out.dim() == 4, "out must be [batch, seq, heads, 512]");
  TORCH_CHECK(q.size(3) == kHeadDim && out.size(3) == kHeadDim,
              "DeepSeek-V4 sparse FlashMLA decode requires head dim 512");
  TORCH_CHECK(q.size(0) == out.size(0) && q.size(1) == out.size(1) &&
                  q.size(2) == out.size(2),
              "q and out leading dimensions must match");
  TORCH_CHECK(q.stride(3) == 1, "q last dimension must be contiguous");
  TORCH_CHECK(out.stride(3) == 1, "out last dimension must be contiguous");
  TORCH_CHECK(q.device() == out.device() && q.device() == k_cache.device() &&
                  q.device() == indices.device(),
              "q, out, k_cache, and indices must be on the same device");

  const int64_t batch = q.size(0);
  const int64_t seq_len = q.size(1);
  const int64_t num_heads = q.size(2);
  const int64_t num_queries = batch * seq_len;
  check_flat_indices(indices, "indices", num_queries);
  check_optional_lengths(topk_length, "topk_length", num_queries);

  const bool has_extra_cache = extra_k_cache.has_value();
  const bool has_extra_indices = extra_indices.has_value();
  TORCH_CHECK(has_extra_cache == has_extra_indices,
              "extra_k_cache and extra_indices must be provided together");
  if (has_extra_cache) {
    check_packed_cache(extra_k_cache.value(), "extra_k_cache");
    check_flat_indices(extra_indices.value(), "extra_indices", num_queries);
    TORCH_CHECK(extra_k_cache.value().device() == q.device() &&
                    extra_indices.value().device() == q.device(),
                "extra sparse tensors must be on the same device");
    check_optional_lengths(extra_topk_length, "extra_topk_length", num_queries);
  }
  if (attn_sink.has_value()) {
    check_musa_tensor(attn_sink.value(), "attn_sink");
    TORCH_CHECK(attn_sink.value().scalar_type() == torch::kFloat32,
                "attn_sink must be float32");
    TORCH_CHECK(attn_sink.value().numel() >= num_heads,
                "attn_sink must include one value per query head");
    TORCH_CHECK(attn_sink.value().is_contiguous(),
                "attn_sink must be contiguous");
  }

  auto lse = torch::empty({batch, num_heads, seq_len},
                          q.options().dtype(torch::kFloat32));
  if (num_queries == 0 || num_heads == 0) {
    return std::make_tuple(out, lse);
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(q));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  const dim3 grid(static_cast<unsigned int>(num_queries),
                  static_cast<unsigned int>(num_heads));
  const dim3 block(kThreads);

  const torch::Tensor *extra_cache_ptr =
      has_extra_cache ? &extra_k_cache.value() : nullptr;
  const torch::Tensor *extra_indices_ptr =
      has_extra_indices ? &extra_indices.value() : nullptr;

  deepseek_v4_sparse_flashmla_decode_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __mt_bfloat16 *>(q.data_ptr()), q.stride(0),
      q.stride(1), q.stride(2), q.stride(3),
      static_cast<__mt_bfloat16 *>(out.data_ptr()), out.stride(0),
      out.stride(1), out.stride(2), out.stride(3),
      static_cast<float *>(lse.data_ptr()),
      static_cast<const uint8_t *>(k_cache.data_ptr()), k_cache.size(0),
      k_cache.size(1), k_cache.stride(0), indices.data_ptr(),
      index_kind(indices, "indices"), indices.stride(0), indices.stride(1),
      topk_length.has_value() ? topk_length.value().data_ptr() : nullptr,
      topk_length.has_value() ? index_kind(topk_length.value(), "topk_length")
                              : kIndexInt32,
      indices.size(1),
      extra_cache_ptr == nullptr
          ? nullptr
          : static_cast<const uint8_t *>(extra_cache_ptr->data_ptr()),
      extra_cache_ptr == nullptr ? 0 : extra_cache_ptr->size(0),
      extra_cache_ptr == nullptr ? 0 : extra_cache_ptr->size(1),
      extra_cache_ptr == nullptr ? 0 : extra_cache_ptr->stride(0),
      extra_indices_ptr == nullptr ? nullptr : extra_indices_ptr->data_ptr(),
      extra_indices_ptr == nullptr ? kIndexInt32
                                   : index_kind(*extra_indices_ptr,
                                                "extra_indices"),
      extra_indices_ptr == nullptr ? 0 : extra_indices_ptr->stride(0),
      extra_indices_ptr == nullptr ? 0 : extra_indices_ptr->stride(1),
      extra_topk_length.has_value() ? extra_topk_length.value().data_ptr()
                                    : nullptr,
      extra_topk_length.has_value()
          ? index_kind(extra_topk_length.value(), "extra_topk_length")
          : kIndexInt32,
      extra_indices_ptr == nullptr ? 0 : extra_indices_ptr->size(1),
      attn_sink.has_value() ? static_cast<const float *>(attn_sink.value().data_ptr())
                            : nullptr,
      batch, seq_len, num_heads, static_cast<float>(softmax_scale));
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_sparse_flashmla_decode launch failed: ",
              musaGetErrorString(err));
  return std::make_tuple(out, lse);
}
