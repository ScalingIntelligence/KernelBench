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

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

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
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"

#include "kernels/kernels_utils.h"
#include "kernels/musa_context.h"

COMMON_DECLARE_int32(flash_attn_version);

using ::musa::dnn::ScaledDotProductAttention;
using ::musa::dnn::MemoryHandler;

namespace phi {

template <typename T, typename Context>
void FlashAttnGradBaseKernel(
    const Context& dev_ctx,
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const paddle::optional<DenseTensor>& startend_row_indices,
    const DenseTensor& dout,
    float dropout,
    float scale,
    bool causal,
    bool is_bhsd,
    DenseTensor* dq,
    DenseTensor* dk,
    DenseTensor* dv) {

  if (is_bhsd) {
    const auto& dims = query.dims();
    const auto N = dims[0];
    const auto H_q = dims[1];
    const auto L = dims[2];

    const auto H_k = key.dims()[1];
    const auto S = key.dims()[2];

    const auto E_v = value.dims()[3];

    auto musa_q = CreateMUTensor(query);
    auto musa_k = CreateMUTensor(key);
    auto musa_v = CreateMUTensor(value);

    DenseTensorMeta dq_meta(
        query.dtype(), phi::make_ddim({N, H_q, L, E_v}), query.layout());
    dq->set_meta(dq_meta);
    dev_ctx.template Alloc<T>(dq);

    DenseTensorMeta dk_meta(
        key.dtype(), phi::make_ddim({N, H_k, S, E_v}), key.layout());
    dk->set_meta(dk_meta);
    dev_ctx.template Alloc<T>(dk);

    DenseTensorMeta dv_meta(
        value.dtype(), phi::make_ddim({N, H_k, S, E_v}), value.layout());
    dv->set_meta(dv_meta);
    dev_ctx.template Alloc<T>(dv);
    
    auto musa_grad_q = CreateMUTensor(*dq);
    auto musa_grad_k = CreateMUTensor(*dk);
    auto musa_grad_v = CreateMUTensor(*dv);

    if (!dout.meta().is_contiguous()) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Non-contiguous dout tensor is not supported in Paddle-MUSA attn backward."));
    }
    auto musa_grad_output = CreateMUTensor(dout);

    if (!softmax_lse.meta().is_contiguous()) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Non-contiguous softmax_lse tensor is not supported in Paddle-MUSA attn backward."));
    }
    auto musa_logsumexp = CreateMUTensor(softmax_lse);

    auto musa_output = CreateMUTensor(out);

    auto& h = GetMudnnHandle<Context>(dev_ctx);
    ::musa::dnn::ScaledDotProductAttention sdpa;

    const auto mask_type = ParseMaskType(attn_mask, causal, N, H_q, L, S);
    muTensor musa_mask;
    if (HasMask(mask_type)) {
      const auto& m = attn_mask.get();
      musa_mask = CreateMUTensor(m);
    } else {
      musa_mask = muTensor();
    }

    phi::DenseTensor contig_dropout_mask;
    phi::DenseTensorMeta drop_meta(
        phi::DataType::BOOL,
        phi::make_ddim({0}),
        query.layout());
    contig_dropout_mask.set_meta(drop_meta);
    dev_ctx.template Alloc<bool>(&contig_dropout_mask);
    if (dropout > 0.0) {
      contig_dropout_mask.Resize(phi::make_ddim({N, H_q, L, S}));
      dev_ctx.template Alloc<bool>(&contig_dropout_mask);
      MUDNN_CHECK(sdpa.SetDropoutP(dropout), "SetDropoutP");
      MUDNN_CHECK(sdpa.SetTraining(true), "SetTraining");

      const auto* seed_offset_data = seed_offset.data<int64_t>();
      // currently does not support cuda graph, so graph offset sets to 0
      MUDNN_CHECK(sdpa.SetSeed(seed_offset_data[0], seed_offset_data[1],
        /*dropoutmode*/ 0, nullptr, nullptr, /*graph_offset*/ 0), "SetSeed");
    }
    auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

    MUDNN_CHECK(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
    MUDNN_CHECK(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
    MUDNN_CHECK(sdpa.SetTraining(true), "SetTraining");
    MUDNN_CHECK(sdpa.SetCausal(causal), "SetCausal");
    MUDNN_CHECK(sdpa.SetMaskMode(IsPadMask(mask_type)), "SetMaskMode");
    if (scale > 0) {
      MUDNN_CHECK(sdpa.SetScale(scale), "SetScale");
    }

    auto place = dev_ctx.GetPlace();

    ::musa::dnn::MemoryMaintainer maintainer =
        [place](size_t bytes) {
          return PaddleInternalMemAlloc(bytes, place);
        };

    MUDNN_CHECK(
        sdpa.RunFlashBwd(
            h,
            musa_grad_q,
            musa_grad_k,
            musa_grad_v,
            musa_grad_output,
            musa_q,
            musa_k,
            musa_v,
            musa_mask,
            musa_output,
            musa_logsumexp,
            musa_dropout_mask,
            maintainer),
        "Run SDPA Flash BWD.");

    return;
  }

  const auto& dims = query.dims();
  const auto N = dims[0];
  const auto L = dims[1];
  const auto H_q = dims[2];

  const auto S = key.dims()[1];
  const auto H_k = key.dims()[2];

  const auto E_v = value.dims()[3];

  DenseTensor q_bhsd;
  TransposeMudnnStyle<T, Context>(dev_ctx, query, &q_bhsd);
  DenseTensor k_bhsd;
  TransposeMudnnStyle<T, Context>(dev_ctx, key, &k_bhsd);
  DenseTensor v_bhsd;
  TransposeMudnnStyle<T, Context>(dev_ctx, value, &v_bhsd);

  auto musa_q = CreateMUTensor(q_bhsd);
  auto musa_k = CreateMUTensor(k_bhsd);
  auto musa_v = CreateMUTensor(v_bhsd);

  DenseTensorMeta dq_bhsd_meta(
      query.dtype(), phi::make_ddim({N, H_q, L, E_v}), query.layout());
  DenseTensor dq_bhsd;
  dq_bhsd.set_meta(dq_bhsd_meta);
  dev_ctx.template Alloc<T>(&dq_bhsd);
  auto musa_grad_q = CreateMUTensor(dq_bhsd);

  DenseTensorMeta dk_bhsd_meta(
      key.dtype(), phi::make_ddim({N, H_k, S, E_v}), query.layout());
  DenseTensor dk_bhsd;
  dk_bhsd.set_meta(dk_bhsd_meta);
  dev_ctx.template Alloc<T>(&dk_bhsd);
  auto musa_grad_k = CreateMUTensor(dk_bhsd);

  DenseTensorMeta dv_bhsd_meta(
      value.dtype(), phi::make_ddim({N, H_k, S, E_v}), query.layout());
  DenseTensor dv_bhsd;
  dv_bhsd.set_meta(dv_bhsd_meta);
  dev_ctx.template Alloc<T>(&dv_bhsd);
  auto musa_grad_v = CreateMUTensor(dv_bhsd);

  if (!dout.meta().is_contiguous()) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Non-contiguous dout tensor is not supported in Paddle-MUSA attn backward."));
  }
  DenseTensor dout_bhsd;
  TransposeMudnnStyle<T, Context>(dev_ctx, dout, &dout_bhsd);
  auto musa_grad_output = CreateMUTensor(dout_bhsd);

  if (!softmax_lse.meta().is_contiguous()) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Non-contiguous softmax_lse tensor is not supported in Paddle-MUSA attn backward."));
  }
  auto musa_logsumexp = CreateMUTensor(softmax_lse);

  DenseTensor out_bhsd;
  TransposeMudnnStyle<T, Context>(dev_ctx, out, &out_bhsd);
  auto musa_output = CreateMUTensor(out_bhsd);

  auto& h = GetMudnnHandle<Context>(dev_ctx);
  ::musa::dnn::ScaledDotProductAttention sdpa;

  const auto mask_type = ParseMaskType(attn_mask, causal, N, H_q, L, S);
  muTensor musa_mask;
  if (HasMask(mask_type)) {
    const auto& m = attn_mask.get();
    musa_mask = CreateMUTensor(m);
  } else {
    musa_mask = muTensor();
  }

  phi::DenseTensor contig_dropout_mask;
  phi::DenseTensorMeta drop_meta(
      phi::DataType::BOOL,
      phi::make_ddim({0}),
      query.layout());
  contig_dropout_mask.set_meta(drop_meta);
  dev_ctx.template Alloc<bool>(&contig_dropout_mask);
  if (dropout > 0.0) {
    contig_dropout_mask.Resize(phi::make_ddim({N, H_q, L, S}));
    dev_ctx.template Alloc<bool>(&contig_dropout_mask);
    MUDNN_CHECK(sdpa.SetDropoutP(dropout), "SetDropoutP");
    MUDNN_CHECK(sdpa.SetTraining(true), "SetTraining");

    const auto* seed_offset_data = seed_offset.data<int64_t>();
    // currently does not support cuda graph, so graph offset sets to 0
    MUDNN_CHECK(sdpa.SetSeed(seed_offset_data[0], seed_offset_data[1],
      /*dropoutmode*/ 0, nullptr, nullptr, /*graph_offset*/ 0), "SetSeed");
  }
  auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

  MUDNN_CHECK(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
  MUDNN_CHECK(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  MUDNN_CHECK(sdpa.SetTraining(true), "SetTraining");
  MUDNN_CHECK(sdpa.SetCausal(causal), "SetCausal");
  MUDNN_CHECK(sdpa.SetMaskMode(IsPadMask(mask_type)), "SetMaskMode");
  if (scale > 0) {
    MUDNN_CHECK(sdpa.SetScale(scale), "SetScale");
  }

  auto place = dev_ctx.GetPlace();

  ::musa::dnn::MemoryMaintainer maintainer =
      [place](size_t bytes) {
        return PaddleInternalMemAlloc(bytes, place);
      };

  MUDNN_CHECK(
      sdpa.RunFlashBwd(
          h,
          musa_grad_q,
          musa_grad_k,
          musa_grad_v,
          musa_grad_output,
          musa_q,
          musa_k,
          musa_v,
          musa_mask,
          musa_output,
          musa_logsumexp,
          musa_dropout_mask,
          maintainer),
      "Run SDPA Flash BWD.");

  DenseTensorMeta dq_meta(
      query.dtype(), phi::make_ddim({N, L, H_q, E_v}), query.layout());
  dq->set_meta(dq_meta);
  dev_ctx.template Alloc<T>(dq);

  DenseTensorMeta dk_meta(
      key.dtype(), phi::make_ddim({N, S, H_k, E_v}), key.layout());
  dk->set_meta(dk_meta);
  dev_ctx.template Alloc<T>(dk);

  DenseTensorMeta dv_meta(
      value.dtype(), phi::make_ddim({N, S, H_k, E_v}), value.layout());
  dv->set_meta(dv_meta);
  dev_ctx.template Alloc<T>(dv);

  std::vector<int> axis_back = {0, 2, 1, 3};  // [N,H,L,E] -> [N,L,H,E]
  phi::TransposeKernel<T, Context>(dev_ctx, dq_bhsd, axis_back, dq);
  phi::TransposeKernel<T, Context>(dev_ctx, dk_bhsd, axis_back, dk);
  phi::TransposeKernel<T, Context>(dev_ctx, dv_bhsd, axis_back, dv);
}

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& dev_ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const DenseTensor& out,
                         const DenseTensor& softmax_lse,
                         const DenseTensor& seed_offset,
                         const paddle::optional<DenseTensor>& attn_mask,
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
//   if (dq) {
//     dev_ctx.template Alloc<T>(dq);
//   }
//   if (dk) {
//     dev_ctx.template Alloc<T>(dk);
//   }
//   if (dv) {
//     dev_ctx.template Alloc<T>(dv);
//   }
  if (dout.numel() == 0) {
    if (dq)
      Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(dq->dims())), 0, dq);
    if (dk)
      Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(dk->dims())), 0, dk);
    if (dv)
      Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(dv->dims())), 0, dv);
    return;
  }
  FlashAttnGradBaseKernel<T, Context>(dev_ctx,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      seed_offset,
                                      attn_mask,
                                      paddle::none,
                                      dout,
                                      dropout,
                                      0.0,
                                      causal,
                                      false,
                                      dq,
                                      dk,
                                      dv);
}

template <typename T, typename Context>
void FlashMaskGradKernel(const Context& dev_ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const DenseTensor& startend_row_indices,
                         const DenseTensor& out,
                         const DenseTensor& softmax_lse,
                         const DenseTensor& seed_offset,
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
//   if (dq) {
//     dev_ctx.template Alloc<T>(dq);
//   }
//   if (dk) {
//     dev_ctx.template Alloc<T>(dk);
//   }
//   if (dv) {
//     dev_ctx.template Alloc<T>(dv);
//   }
  FlashAttnGradBaseKernel<T, Context>(dev_ctx,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      seed_offset,
                                      paddle::none,
                                      startend_row_indices,
                                      dout,
                                      dropout,
                                      0.0,
                                      causal,
                                      false,
                                      dq,
                                      dk,
                                      dv);
}

template <typename T, typename Context>
void FlashAttnUnpaddedGradBaseKernel(
    const Context& dev_ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const DenseTensor& dout,
    const Scalar& max_seqlen_q_,
    const Scalar& max_seqlen_k_,
    float scale,
    float dropout,
    bool causal,
    DenseTensor* dq,
    DenseTensor* dk,
    DenseTensor* dv,
    bool varlen_padded) {
  // q,k,v [total_*, num_heads, head_dim]
  auto dims = q.dims();

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t sum_seq = dims[0];
  const int64_t num_heads = dims[1];
  const int64_t head_size_og = dout.dims()[2];
  const int64_t head_size = dims[2];
  const int64_t total_k = k.dims()[0];
  const int64_t num_heads_k = k.dims()[1];

  const auto stream = dev_ctx.stream();

  // TODO(umiswing): add shape check
  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      common::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));
  
  PADDLE_ENFORCE_EQ(
    attn_mask.is_initialized(),
    false,
    common::errors::InvalidArgument(
        "FlashAttnUnpaddedGradBaseKernel expects attn_mask "
        "to be empty. It will build padding mask internally."));

  int64_t max_seqlen_q = max_seqlen_q_.to<int64_t>();
  int64_t max_seqlen_k = max_seqlen_k_.to<int64_t>();

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
      out.dtype(), phi::make_ddim({batch_size, num_heads, max_seqlen_q, head_size}), out.layout());
  out_pad.set_meta(out_pad_meta);
  dev_ctx.template Alloc<T>(&out_pad);

  DenseTensor dout_pad;
  DenseTensorMeta dout_pad_meta(
      dout.dtype(), phi::make_ddim({batch_size, num_heads, max_seqlen_q, head_size}), dout.layout());
  dout_pad.set_meta(dout_pad_meta);
  dev_ctx.template Alloc<T>(&dout_pad);

  paddle::optional<DenseTensor> optional_dout(dout);
  paddle::optional<DenseTensor> optional_dout_pad(dout_pad);
  paddle::optional<DenseTensor> optional_out(out);
  paddle::optional<DenseTensor> optional_out_pad(out_pad);
  VarlenFaSeqlenPad<T, Context>(dev_ctx, q, k, v, optional_dout, optional_out, q_pad, k_pad, v_pad, optional_dout_pad, optional_out_pad,
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

  DenseTensor dq_pad;
  DenseTensor dk_pad;
  DenseTensor dv_pad;
  FlashAttnGradBaseKernel<T, Context>(dev_ctx,
                                      q_pad,
                                      k_pad,
                                      v_pad,
                                      out_pad,
                                      softmax_lse,
                                      seed_offset,
                                      local_attn_mask,
                                      paddle::none,
                                      dout_pad,
                                      dropout,
                                      scale,
                                      causal,
                                      true,
                                      &dq_pad,
                                      &dk_pad,
                                      &dv_pad);

  dq->Resize({sum_seq, num_heads, head_size});
  dev_ctx.template Alloc<T>(dq);
  dk->Resize({total_k, num_heads_k, head_size});
  dev_ctx.template Alloc<T>(dk);
  dv->Resize({total_k, num_heads_k, head_size});
  dev_ctx.template Alloc<T>(dv);

  VarlenFaSeqlenUnpad<T, Context>(dev_ctx, dq_pad, dq, cu_seqlens_q, 
        sum_seq, max_seqlen_q, head_size, num_heads, batch_size);
  VarlenFaSeqlenUnpad<T, Context>(dev_ctx, dk_pad, dk, cu_seqlens_k, 
        total_k, max_seqlen_k, head_size, num_heads_k, batch_size);
  VarlenFaSeqlenUnpad<T, Context>(dev_ctx, dv_pad, dv, cu_seqlens_k, 
        total_k, max_seqlen_k, head_size, num_heads_k, batch_size);
}

template <typename T, typename Context>
void FlashAttnUnpaddedGradKernel(const Context& dev_ctx,
                                 const DenseTensor& q,
                                 const DenseTensor& k,
                                 const DenseTensor& v,
                                 const DenseTensor& cu_seqlens_q,
                                 const DenseTensor& cu_seqlens_k,
                                 const DenseTensor& out,
                                 const DenseTensor& softmax_lse,
                                 const DenseTensor& seed_offset,
                                 const paddle::optional<DenseTensor>& attn_mask,
                                 const DenseTensor& dout,
                                 const Scalar& max_seqlen_q,
                                 const Scalar& max_seqlen_k,
                                 float scale,
                                 float dropout,
                                 bool causal,
                                 DenseTensor* dq,
                                 DenseTensor* dk,
                                 DenseTensor* dv) {
  // if (dq) {
  //   dev_ctx.template Alloc<T>(dq);
  // }
  // if (dk) {
  //   dev_ctx.template Alloc<T>(dk);
  // }
  // if (dv) {
  //   dev_ctx.template Alloc<T>(dv);
  // }
  FlashAttnUnpaddedGradBaseKernel<T>(dev_ctx,
                                     q,
                                     k,
                                     v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     out,
                                     softmax_lse,
                                     seed_offset,
                                     attn_mask,
                                     dout,
                                     max_seqlen_q,
                                     max_seqlen_k,
                                     scale,
                                     dropout,
                                     causal,
                                     dq,
                                     dk,
                                     dv,
                                     false /*varlen_padded*/);
}

} // namespace phi

PD_CUSTOM_KERNEL_REGISTER(flash_attn_unpadded_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedGradKernel,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(7).SetBackend(phi::Backend::CPU);  // seed_offset
}

// PD_CUSTOM_KERNEL_REGISTER(flash_attn_varlen_qkvpacked_grad,
//                    musa,
//                    ALL_LAYOUT,
//                    phi::FlashAttnVarlenQKVPackedGradKernel,
//                    phi::float16,
//                    phi::bfloat16) {
//   kernel->InputAt(5).SetBackend(phi::Backend::CPU);  // seed_offset
// }

PD_CUSTOM_KERNEL_REGISTER(flash_attn_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::FlashAttnGradKernel,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::CPU);  // seed_offset
}

// PD_CUSTOM_KERNEL_REGISTER(flash_attn_qkvpacked_grad,
//                    musa,
//                    ALL_LAYOUT,
//                    phi::FlashAttnQKVPackedGradKernel,
//                    phi::float16,
//                    phi::bfloat16) {
//   kernel->InputAt(3).SetBackend(phi::Backend::CPU);  // seed_offset
// }

PD_CUSTOM_KERNEL_REGISTER(flashmask_attention_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::FlashMaskGradKernel,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(6).SetBackend(phi::Backend::CPU);  // seed_offset
}
