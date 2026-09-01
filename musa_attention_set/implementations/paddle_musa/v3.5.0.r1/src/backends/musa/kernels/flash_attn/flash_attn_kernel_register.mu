// Reserved. Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modifications:
// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [register musa backend]

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "glog/logging.h"  // For VLOG()
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/utils/none.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/tril_triu_kernel.h"

#include "kernels/kernels_utils.h"
#include "kernels/musa_context.h"

#include <musa_runtime.h>

COMMON_DECLARE_int32(flash_attn_version);

using ::musa::dnn::ScaledDotProductAttention;
using ::musa::dnn::MemoryHandler;
using muTensor = ::musa::dnn::Tensor;

namespace phi {
template <typename Context>
static std::pair<uint64_t, uint64_t> GenerateRNGState(
    const Context& dev_ctx,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const std::string& rng_name,
    const int64_t batch_size,
    const int64_t num_heads) {
  if (fixed_seed_offset.get_ptr()) {
    const int64_t* fixed_seed_offset_data =
        fixed_seed_offset.get_ptr()->data<int64_t>();
    uint64_t seed = static_cast<uint64_t>(fixed_seed_offset_data[0]);
    uint64_t offset = static_cast<uint64_t>(fixed_seed_offset_data[1]);
    return std::make_pair(seed, offset);
  } else {
    uint64_t inc = batch_size * num_heads * 32;
    std::pair<uint64_t, uint64_t> seed_offset_pair;
    if (rng_name != "") {
      auto gen = phi::GetRandomSeedGenerator(rng_name);
      seed_offset_pair = gen->IncrementOffset(inc);
    } else {
      auto* gen = dev_ctx.GetGenerator();
      seed_offset_pair = gen->IncrementOffset(inc);
    }
    return seed_offset_pair;
  }
}

template <typename OutT>
struct ZeroFunctor {
  __device__ __forceinline__ OutT operator()() const {
    return static_cast<OutT>(0);
  }
};




template <typename T, int BLOCKDIM_X, int BLOCKDIM_Y>
__global__ void unpad_lse_kernel(
    const T* input,          // [B, H, L]  (BHL, contiguous)
    T* output,               // [T, H]     (TH, contiguous)
    const int* accum_lens,   // [B+1] prefix-sum (cu_seqlens), accum_lens[B] = T
    int max_seq,             // L
    int num_heads,           // H
    int batch_size           // B
) {
    const int b = (int)blockIdx.z;
    const int t = (int)blockIdx.x * BLOCKDIM_Y + (int)threadIdx.y;
    const int h = (int)blockIdx.y * BLOCKDIM_X + (int)threadIdx.x;

    if (b >= batch_size || h >= num_heads) return;

    const int start  = accum_lens[b];
    const int end    = accum_lens[b + 1];
    const int seqlen = end - start;

    if (t >= seqlen) return;
    if (t >= max_seq) return;

    const int64_t in_idx =
        ((int64_t)b * (int64_t)num_heads + (int64_t)h) * (int64_t)max_seq + (int64_t)t;

    const int64_t out_idx =
        ((int64_t)(start + t) * (int64_t)num_heads) + (int64_t)h;

    output[out_idx] = input[in_idx];
}

template <class T, typename Context>
void launch_lse_unpad_kernel(
    const Context& dev_ctx,
    const T* input,            // [B,H,L]
    T* output,                 // [T,H]
    const int* accum_lens,     // [B+1]
    int max_seq,               // L
    int num_heads,             // H
    int batch_size             // B
) {
    auto stream = dev_ctx.stream();

    constexpr int BX = 32;  // heads tile
    constexpr int BY = 32;  // tokens tile
    dim3 block(BX, BY);

    dim3 grid(
        (max_seq + BY - 1) / BY,           // token tiles (over L; per-b seqlen 在 kernel 内裁掉)
        (num_heads + BX - 1) / BX,         // head tiles
        batch_size
    );

    unpad_lse_kernel<T, BX, BY><<<grid, block, 0, stream>>>(
        input, output, accum_lens, max_seq, num_heads, batch_size
    );
}

template <typename T, typename Context>
void FlashAttnBaseKernel(
    const Context& dev_ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const paddle::optional<DenseTensor>& startend_row_indices,
    float dropout,
    float scale,
    bool causal,
    bool return_softmax,
    bool is_test,
    bool is_bhsd,
    const std::string& rng_name,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
  // q, k, v [N, L, H, E]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      4,
      common::errors::InvalidArgument(
          "flash_attn receive input with dim "
          "[batch_size, seq_len, num_heads, head_dim]"));

  PADDLE_ENFORCE_EQ(
      k.dims().size(),
      4,
      common::errors::InvalidArgument(
          "flash_attn receive input with dim "
          "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      v.dims().size(),
      4,
      common::errors::InvalidArgument(
          "flash_attn receive input with dim "
          "[batch_size, seq_len, num_heads, head_dim]"));

  auto mask = attn_mask;
//   if (causal) {
//     PADDLE_ENFORCE_EQ(
//         !attn_mask.is_initialized(),
//         true,
//         phi::errors::InvalidArgument(
//             "MUSA SDPA: Explicit attn_mask should not be set "
//             "when is_causal=True"));
//   }

  if (is_bhsd) {
    const auto N   = dims[0];  // batch
    const auto H_q = dims[1];  // num_heads
    const auto L   = dims[2];  // seq_len
    const auto E   = dims[3];  // head_dim

    const auto S   = k.dims()[2];
    const auto E_v = v.dims()[3];

    const auto mask_type = ParseMaskType(mask, causal, N, H_q, L, S);

    auto musa_q = CreateMUTensor(q);
    auto musa_k = CreateMUTensor(k);
    auto musa_v = CreateMUTensor(v);

    out->Resize({N, H_q, L, E});
    dev_ctx.template Alloc<T>(out);
    
    auto musa_out = CreateMUTensor(*out);

    muTensor musa_mask;
    if (HasMask(mask_type)) {
        const auto& m = mask.get();
        musa_mask = CreateMUTensor(m);
    } else {
        musa_mask = muTensor();
    }

    DenseTensorMeta lse_meta(
        phi::DataType::FLOAT32,
        phi::make_ddim({N, H_q, L}),  // [N, H, L]
        q.layout());
    softmax_lse->set_meta(lse_meta);
    dev_ctx.template Alloc<float>(softmax_lse);
    auto musa_lse = CreateMUTensor(*softmax_lse);

    auto& h = GetMudnnHandle<Context>(dev_ctx);
    ::musa::dnn::ScaledDotProductAttention sdpa;

    MUDNN_CHECK(sdpa.SetCausal(causal), "SetCausal");
    MUDNN_CHECK(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
    MUDNN_CHECK(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
    MUDNN_CHECK(sdpa.SetMaskMode(IsPadMask(mask_type)), "SetMaskMode");
    if (scale > 0.0) {
        MUDNN_CHECK(sdpa.SetScale(scale), "SetScale");
    }
    
    DenseTensor contig_dropout_mask;
    DenseTensorMeta drop_meta(
        phi::DataType::BOOL,
        phi::make_ddim({0}),
        q.layout());
    contig_dropout_mask.set_meta(drop_meta);
    dev_ctx.template Alloc<bool>(&contig_dropout_mask);

    seed_offset->Resize({2});
    int64_t* seed_offset_data =
        dev_ctx.template HostAlloc<int64_t>(seed_offset);
    seed_offset_data[0] = 0;
    seed_offset_data[1] = 0;

    if (dropout > 0.0f) {
        contig_dropout_mask.Resize(phi::make_ddim({N, H_q, L, S}));
        dev_ctx.template Alloc<bool>(&contig_dropout_mask);
        auto seedOffsetPair =
            GenerateRNGState(dev_ctx, fixed_seed_offset, "", N, H_q);

        seed_offset_data[0] = static_cast<int64_t>(seedOffsetPair.first);
        seed_offset_data[1] = static_cast<int64_t>(seedOffsetPair.second);

        MUDNN_CHECK(sdpa.SetDropoutP(dropout), "SetDropoutP");
        MUDNN_CHECK(sdpa.SetTraining(true), "SetTraining");

        MUDNN_CHECK(
            sdpa.SetSeed(seedOffsetPair.first,
                        seedOffsetPair.second,
                        /*dropoutmode*/ 0,
                        nullptr,
                        nullptr,
                        /*graph_offset*/ 0),
            "SetSeed");
    }
    auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

    auto place = dev_ctx.GetPlace();
    ::musa::dnn::MemoryMaintainer maintainer =
        [place](size_t bytes) { return PaddleInternalMemAlloc(bytes, place); };

    MUDNN_CHECK(
        sdpa.RunFlash(
            h,
            musa_out,          // out_bhsd: [N, H, L, E_v]
            musa_lse,          // [N, H, L]
            musa_q,            // q_bhsd:  [N, H, L, E]
            musa_k,            // k_bhsd:  [N, H, S, E]
            musa_v,            // v_bhsd:  [N, H, S, E_v]
            musa_mask,         
            musa_dropout_mask,
            maintainer),
        "Run SDPA Flash FWD.");

    return;
  }

  const auto N   = dims[0];  // batch
  const auto L   = dims[1];  // seq_len
  const auto H_q = dims[2];  // num_heads
  const auto E   = dims[3];  // head_dim

  const auto S   = k.dims()[1];
  const auto E_v = v.dims()[3];

  const auto mask_type = ParseMaskType(mask, causal, N, H_q, L, S);

  DenseTensor q_bhsd;
  TransposeMudnnStyle<T, Context>(dev_ctx, q, &q_bhsd);
  DenseTensor k_bhsd;
  TransposeMudnnStyle<T, Context>(dev_ctx, k, &k_bhsd);
  DenseTensor v_bhsd;
  TransposeMudnnStyle<T, Context>(dev_ctx, v, &v_bhsd);

  auto musa_q = CreateMUTensor(q_bhsd);
  auto musa_k = CreateMUTensor(k_bhsd);
  auto musa_v = CreateMUTensor(v_bhsd);
  
  DenseTensor out_bhsd;
  DenseTensorMeta out_bhsd_meta(
      q.dtype(), phi::make_ddim({N, H_q, L, E_v}), q.layout());
  out_bhsd.set_meta(out_bhsd_meta);
  dev_ctx.template Alloc<T>(&out_bhsd);
  auto musa_out = CreateMUTensor(out_bhsd);

  muTensor musa_mask;
  if (HasMask(mask_type)) {
    const auto& m = mask.get();
    musa_mask = CreateMUTensor(m);
  } else {
    musa_mask = muTensor();
  }

  DenseTensorMeta lse_meta(
      phi::DataType::FLOAT32,
      phi::make_ddim({N, H_q, L}),  // [N, H, L]
      q.layout());
  softmax_lse->set_meta(lse_meta);
  dev_ctx.template Alloc<float>(softmax_lse);
  auto musa_lse = CreateMUTensor(*softmax_lse);

  auto& h = GetMudnnHandle<Context>(dev_ctx);
  ::musa::dnn::ScaledDotProductAttention sdpa;

  MUDNN_CHECK(sdpa.SetCausal(causal), "SetCausal");
  MUDNN_CHECK(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
  MUDNN_CHECK(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  MUDNN_CHECK(sdpa.SetMaskMode(IsPadMask(mask_type)), "SetMaskMode");
  if (scale > 0.0) {
    MUDNN_CHECK(sdpa.SetScale(scale), "SetScale");
  }

  DenseTensor contig_dropout_mask;
  DenseTensorMeta drop_meta(
      phi::DataType::BOOL,
      phi::make_ddim({0}),
      q.layout());
  contig_dropout_mask.set_meta(drop_meta);
  dev_ctx.template Alloc<bool>(&contig_dropout_mask);

  seed_offset->Resize({2});
  int64_t* seed_offset_data =
      dev_ctx.template HostAlloc<int64_t>(seed_offset);
  seed_offset_data[0] = 0;
  seed_offset_data[1] = 0;

  if (dropout > 0.0f) {
    contig_dropout_mask.Resize(phi::make_ddim({N, H_q, L, S}));
    dev_ctx.template Alloc<bool>(&contig_dropout_mask);
    auto seedOffsetPair =
        GenerateRNGState(dev_ctx, fixed_seed_offset, "", N, H_q);

    seed_offset_data[0] = static_cast<int64_t>(seedOffsetPair.first);
    seed_offset_data[1] = static_cast<int64_t>(seedOffsetPair.second);

    MUDNN_CHECK(sdpa.SetDropoutP(dropout), "SetDropoutP");
    MUDNN_CHECK(sdpa.SetTraining(true), "SetTraining");

    MUDNN_CHECK(
        sdpa.SetSeed(seedOffsetPair.first,
                      seedOffsetPair.second,
                      /*dropoutmode*/ 0,
                      nullptr,
                      nullptr,
                      /*graph_offset*/ 0),
        "SetSeed");
  }
  auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

  auto place = dev_ctx.GetPlace();
  ::musa::dnn::MemoryMaintainer maintainer =
      [place](size_t bytes) { return PaddleInternalMemAlloc(bytes, place); };

  MUDNN_CHECK(
      sdpa.RunFlash(
          h,
          musa_out,          // out_bhsd: [N, H, L, E_v]
          musa_lse,          // [N, H, L]
          musa_q,            // q_bhsd:  [N, H, L, E]
          musa_k,            // k_bhsd:  [N, H, S, E]
          musa_v,            // v_bhsd:  [N, H, S, E_v]
          musa_mask,         
          musa_dropout_mask,
          maintainer),
      "Run SDPA Flash FWD.");

  out->Resize({N, L, H_q, E_v});
  dev_ctx.template Alloc<T>(out);

  std::vector<int> axis_back = {0, 2, 1, 3};  // [N,H,L,E] -> [N,L,H,E]
  phi::TransposeKernel<T, Context>(dev_ctx, out_bhsd, axis_back, out);
  VLOG(1) << "q in forwad = " << q.dims();
  VLOG(1) << "k in forwad = " << k.dims();
  VLOG(1) << "v in forwad = " << v.dims();
  VLOG(1) << "out in forwad = " << out->dims();
}

template <typename T, typename Context>
void FlashAttnKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     const paddle::optional<DenseTensor>& fixed_seed_offset,
                     const paddle::optional<DenseTensor>& attn_mask,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     bool is_test,
                     const std::string& rng_name,
                     DenseTensor* out,
                     DenseTensor* softmax,
                     DenseTensor* softmax_lse,
                     DenseTensor* seed_offset) {
  if (q.numel() == 0 || k.numel() == 0 || v.numel() == 0) {
    if (out) {
      Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(out->dims())), 0, out);
    }
    if (softmax) {
      Full<T, Context>(dev_ctx,
                       phi::IntArray(common::vectorize(softmax->dims())),
                       0,
                       softmax);
    }
    if (softmax_lse) {
      Full<T, Context>(dev_ctx,
                       phi::IntArray(common::vectorize(softmax_lse->dims())),
                       0,
                       softmax_lse);
    }
    if (seed_offset) {
      Full<T, Context>(dev_ctx,
                       phi::IntArray(common::vectorize(seed_offset->dims())),
                       0,
                       seed_offset);
    }
    return;
  }
  FlashAttnBaseKernel<T, Context>(dev_ctx,
                                  q,
                                  k,
                                  v,
                                  fixed_seed_offset,
                                  attn_mask,
                                  paddle::none,
                                  dropout,
                                  0.0,
                                  causal,
                                  return_softmax,
                                  is_test,
                                  false,
                                  rng_name,
                                  out,
                                  softmax,
                                  softmax_lse,
                                  seed_offset);
}

template <typename T, typename Context>
void FlashMaskKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     const DenseTensor& startend_row_indices,
                     const paddle::optional<DenseTensor>& fixed_seed_offset,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     bool is_test,
                     const std::string& rng_name,
                     DenseTensor* out,
                     DenseTensor* softmax,
                     DenseTensor* softmax_lse,
                     DenseTensor* seed_offset) {
  FlashAttnBaseKernel<T, Context>(dev_ctx,
                                  q,
                                  k,
                                  v,
                                  fixed_seed_offset,
                                  paddle::none,
                                  startend_row_indices,
                                  dropout,
                                  0.0,
                                  causal,
                                  return_softmax,
                                  is_test,
                                  false,
                                  rng_name,
                                  out,
                                  softmax,
                                  softmax_lse,
                                  seed_offset);
}

template <typename T, typename Context>
void FlashAttnUnpaddedBaseKernel(
    const Context& dev_ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const Scalar& max_seqlen_q_,
    const Scalar& max_seqlen_k_,
    float scale,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset,
    bool varlen_padded) {
  if (!out->IsInitialized()) dev_ctx.template Alloc<T>(out);
  if (varlen_padded) {
    std::vector<const DenseTensor*> inputs{};
    std::vector<DenseTensor*> outputs{out};

    phi::funcs::ElementwiseKernel<T>(
        dev_ctx, inputs, &outputs, ZeroFunctor<T>());
  }

  PADDLE_ENFORCE_EQ(
    attn_mask.is_initialized(),
    false,
    common::errors::InvalidArgument(
        "FlashAttnUnpaddedBaseKernel expects attn_mask "
        "to be empty. It will build padding mask internally."));

  auto stream = dev_ctx.stream();

  // q, k, v [total_q/k/v, num_heads, head_dim]
  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      3,
      common::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                      "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      k.dims().size(),
      3,
      common::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                      "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      v.dims().size(),
      3,
      common::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                      "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      out->dims().size(),
      3,
      common::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                      "[total_seq_len, num_heads, head_dim]"));

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t sum_seq = dims[0];
  const int64_t num_heads = dims[1];
  const int64_t head_size = dims[2];
  const int64_t num_heads_k = k.dims()[1];

  int max_seqlen_q = max_seqlen_q_.to<int64_t>();
  int max_seqlen_k = max_seqlen_k_.to<int64_t>();

  DenseTensor q_pad;
  DenseTensorMeta q_pad_meta(
      q.dtype(), phi::make_ddim({batch_size, num_heads, max_seqlen_q, head_size}), q.layout());
  q_pad.set_meta(q_pad_meta);
  dev_ctx.template Alloc<T>(&q_pad);

  DenseTensor k_pad;
  DenseTensorMeta k_pad_meta(
      k.dtype(), phi::make_ddim({batch_size, num_heads_k, max_seqlen_k, head_size}), k.layout());
  k_pad.set_meta(k_pad_meta);
  dev_ctx.template Alloc<T>(&k_pad);

  DenseTensor v_pad;
  DenseTensorMeta v_pad_meta(
      v.dtype(), phi::make_ddim({batch_size, num_heads_k, max_seqlen_k, head_size}), v.layout());
  v_pad.set_meta(v_pad_meta);
  dev_ctx.template Alloc<T>(&v_pad);

  DenseTensor out_pad;
  DenseTensorMeta out_pad_meta(
      out->dtype(), phi::make_ddim({batch_size, num_heads, max_seqlen_q, head_size}), out->layout());
  out_pad.set_meta(out_pad_meta);
  dev_ctx.template Alloc<T>(&out_pad);

  paddle::optional<DenseTensor> place_holder;
  VarlenFaSeqlenPad<T, Context>(dev_ctx, q, k, v, paddle::none, paddle::none, q_pad, k_pad, v_pad, place_holder, place_holder,
    cu_seqlens_q, cu_seqlens_k, sum_seq, max_seqlen_q, batch_size);
  
  DenseTensor padding_mask;
  DenseTensorMeta mask_meta(
      q.dtype(),
      phi::make_ddim({batch_size, 1, max_seqlen_q, max_seqlen_k}),
      phi::DataLayout::NCHW);
  padding_mask.set_meta(mask_meta);
  dev_ctx.template Alloc<T>(&padding_mask);

  // launch kernel
  const int B = static_cast<int>(batch_size);
  const int max_q = static_cast<int>(max_seqlen_q);
  const int max_k = static_cast<int>(max_seqlen_k);

  dim3 block(16, 16, 1);
  dim3 grid((max_k + block.x - 1) / block.x,
            (max_q + block.y - 1) / block.y,
            B);

  auto* mask_ptr = padding_mask.data<T>();
  float negInf = -1e9f;

  build_padding_mask_kernel<T><<<grid, block, 0, stream>>>(
      cu_seqlens_q.data<int>(), cu_seqlens_k.data<int>(),
      mask_ptr, B, max_q, max_k, negInf);

  paddle::optional<DenseTensor> local_attn_mask(padding_mask);

  FlashAttnBaseKernel<T, Context>(dev_ctx,
                     q_pad,
                     k_pad,
                     v_pad,
                     fixed_seed_offset,
                     local_attn_mask,
                     paddle::none,
                     dropout,
                     scale,
                     causal,
                     return_softmax,
                     is_test,
                     true,
                     rng_name,
                     &out_pad,
                     softmax,
                     softmax_lse,
                     seed_offset);
    
  out->Resize({sum_seq, num_heads, head_size});
  dev_ctx.template Alloc<T>(out);
  VarlenFaSeqlenUnpad<T, Context>(dev_ctx, out_pad, out, cu_seqlens_q, 
        sum_seq, max_seqlen_q, head_size, num_heads, batch_size);
}

template <typename T, typename Context>
void FlashAttnUnpaddedKernel(
    const Context& dev_ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const Scalar& max_seqlen_q,
    const Scalar& max_seqlen_k,
    float scale,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
  FlashAttnUnpaddedBaseKernel<T>(dev_ctx,
                                 q,
                                 k,
                                 v,
                                 cu_seqlens_q,
                                 cu_seqlens_k,
                                 fixed_seed_offset,
                                 attn_mask,
                                 max_seqlen_q,
                                 max_seqlen_k,
                                 scale,
                                 dropout,
                                 causal,
                                 return_softmax,
                                 is_test,
                                 rng_name,
                                 out,
                                 softmax,
                                 softmax_lse,
                                 seed_offset,
                                 false /*varlen_padded*/);
}


}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(flash_attn_unpadded,
                   musa,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedKernel,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(5).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

// PD_CUSTOM_KERNEL_REGISTER(flash_attn_varlen_qkvpacked,
//                    musa,
//                    ALL_LAYOUT,
//                    phi::FlashAttnVarlenQKVPackedKernel,
//                    phi::float16,
//                    phi::bfloat16) {
//   kernel->InputAt(3).SetBackend(
//       phi::Backend::ALL_BACKEND);  // fixed_seed_offset
// }

PD_CUSTOM_KERNEL_REGISTER(flash_attn,
                   musa,
                   ALL_LAYOUT,
                   phi::FlashAttnKernel,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

// PD_CUSTOM_KERNEL_REGISTER(flash_attn_qkvpacked,
//                    musa,
//                    ALL_LAYOUT,
//                    phi::FlashAttnQKVPackedKernel,
//                    phi::float16,
//                    phi::bfloat16) {
//   kernel->InputAt(1).SetBackend(
//       phi::Backend::ALL_BACKEND);  // fixed_seed_offset
// }

PD_CUSTOM_KERNEL_REGISTER(flashmask_attention,
                   musa,
                   ALL_LAYOUT,
                   phi::FlashMaskKernel,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(4).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}
