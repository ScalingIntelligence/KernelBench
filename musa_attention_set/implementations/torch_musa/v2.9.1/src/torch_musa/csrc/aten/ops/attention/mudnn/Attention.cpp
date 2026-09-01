#include <ATen/ExpandUtils.h>
#include <ATen/TensorOperators.h>
#include <c10/core/Scalar.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_flash_attention_backward_native.h>
#include <ATen/ops/_flash_attention_forward_native.h>
#include <ATen/ops/_native_multi_head_attention_cpu_dispatch.h>
#include <ATen/ops/_native_multi_head_attention_native.h>
#include <ATen/ops/_transform_bias_rescale_qkv_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#endif

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/attention/mudnn/SDPUtils.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/Numeric.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _FlashAttnForward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& cumulative_sequence_length_q,
    const std::optional<Tensor>& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<Tensor>& _seqused_k,
    const std::optional<Tensor>& _alibi_slopes) {
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3000)
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();

  TORCH_CHECK(
      !_alibi_slopes.has_value(),
      "_flash_attention_forward doesn't support alibi_slopes for MUSA backend");
  TORCH_CHECK(
      !window_size_left.has_value(),
      "_flash_attention_forward doesn't support window_size_left for MUSA backend");
  TORCH_CHECK(
      !window_size_right.has_value(),
      "_flash_attention_forward doesn't support window_size_right for MUSA backend");
  TORCH_CHECK(
      cumulative_sequence_length_q.has_value() ==
          cumulative_sequence_length_k.has_value(),
      "cumulative_sequence_length_q and cumulative_sequence_length_k must be both set or both not set");

  Tensor output, logsumexp, rng_state, debug_attn_mask;
  if (cumulative_sequence_length_q.has_value()) {
    std::tie(output, logsumexp, debug_attn_mask) = MuDNNFlashVarlenFwd(
        query,
        key,
        value,
        cumulative_sequence_length_q.value(),
        cumulative_sequence_length_k.value(),
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        dropout_p,
        softmax_scale,
        is_causal);
  } else {
    std::tie(output, logsumexp, debug_attn_mask, rng_state) = MuDNNFlashSDPAFwd(
        query,
        key,
        value,
        /*attn_mask=*/std::nullopt,
        dropout_p,
        is_causal,
        return_debug_mask,
        softmax_scale);
  }
  return std::make_tuple(
      std::move(output),
      std::move(logsumexp),
      std::move(rng_state),
      Tensor(),
      std::move(debug_attn_mask));

#else
  TORCH_CHECK(false, "_flash_attention_forward was not enabled for build.")
  return std::make_tuple(Tensor(), Tensor(), Tensor(), Tensor(), Tensor());
#endif
}

std::tuple<Tensor, Tensor, Tensor> _FlashAttnBackward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const Tensor& rng_state,
    const Tensor& unused,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right) {
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3000)
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  //  CUDA code assumes that dout is contiguous
  auto contiguous_grad_out = grad_out.contiguous();
  auto contiguous_out = out.contiguous();

  TORCH_CHECK(
      !window_size_left.has_value(),
      "_flash_attention_forward doesn't support window_size_left for MUSA backend");
  TORCH_CHECK(
      !window_size_right.has_value(),
      "_flash_attention_forward doesn't support window_size_right for MUSA backend");

  // We check the whether the cumulative_sequence_length_q is defined
  // in order to determine whether we are using varlen or dense backward
  if (cumulative_sequence_length_q.defined()) {
    auto dq = at::empty_like(query);
    auto dk = at::empty_like(key);
    auto dv = at::empty_like(value);
    // Varlen backward
    auto [dQuery, dKey, dValue, dSoftmax] = MuDNNFlashVarlenBwd(
        contiguous_grad_out,
        query,
        key,
        value,
        contiguous_out,
        logsumexp,
        dq,
        dk,
        dv,
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        dropout_p,
        softmax_scale,
        is_causal);
    return std::make_tuple(
        std::move(dQuery), std::move(dKey), std::move(dValue));
  } else {
    // Dense backward
    auto [dQuery, dKey, dValue, dSoftmax] = MuDNNFlashSDPABwd(
        contiguous_grad_out,
        query,
        key,
        value,
        contiguous_out,
        logsumexp,
        /*dropout_mask=*/Tensor(),
        rng_state,
        dropout_p,
        is_causal,
        /*attn_mask=*/std::nullopt,
        softmax_scale);
    return std::make_tuple(
        std::move(dQuery), std::move(dKey), std::move(dValue));
  }
#else
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.");
  return std::make_tuple(Tensor(), Tensor(), Tensor());
#endif
}

// clang-format off
std::tuple<Tensor, Tensor> _NativeMultiHeadAttention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const int64_t num_head,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const std::optional<Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    const std::optional<int64_t> mask_type) {
  // query shape: [B, T, D]
  // qkv_weight shape: [3 * D, D]

  TORCH_CHECK(
      !mask || !query.is_nested(),
      "NestedTensor with mask is not supported yet");
  const auto D = embed_dim;
  TORCH_CHECK(
      query.dim() == 3,
      "expected 3-D `query`, got ",
      query.dim(),
      "-D tensor");
  TORCH_CHECK(
      query.is_nested() || query.sizes()[2] == embed_dim,
      "passed-in embed_dim ",
      embed_dim,
      " didn't match last dim of query ",
      query.sizes()[2]);
  TORCH_CHECK(
      key.dim() == 3,
      "expected 3-D `key`, got ",
      key.dim(),
      "-D tensor");
  TORCH_CHECK(
      value.dim() == 3,
      "expected 3-D `value`, got ",
      value.dim(),
      "-D tensor");
  TORCH_CHECK(
      query.is_nested() || key.is_nested() || value.is_nested() ||
          (query.sizes() == key.sizes() && key.sizes() == value.sizes()),
      "expected `query`/`key`/`value` shapes to match");
  TORCH_CHECK(
      qkv_weight.dim() == 2,
      "expected 2-D `qkv_weight`, got ",
      qkv_weight.dim(),
      "-D tensor");
  TORCH_CHECK(
      D * 3 == qkv_weight.sizes()[0],
      "expected `qkv_weight` first dim to be 3x embed_dim");
  TORCH_CHECK(
      D == qkv_weight.sizes()[1],
      "expected `qkv_weight` second dim to be embed_Dim");
  TORCH_CHECK(
      qkv_bias.dim() == 1,
      "expected 1-D `qkv_bias`, got ",
      qkv_bias.dim(),
      "-D tensor");
  TORCH_CHECK(
      qkv_bias.sizes()[0] == 3 * D,
      "expected `qkv_bias` first dim and first dim of query to be equal");
  TORCH_CHECK(D % num_head == 0, "`embed_dim` must divide evenly by `num_heads`");

  const auto dim_per_head = D / num_head;
  const bool dispatch_common = (query.is_nested() || key.is_nested() || value.is_nested());

  if ((!dispatch_common) && (query.is_same(key) && key.is_same(value)) && !need_weights) {
    auto q = query.view({query.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto k = key.view({key.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto v = value.view({value.size(0), -1, num_head, dim_per_head}).transpose(1, 2);

    sdp::sdp_params kernel_params{q, k, v, mask, 0.0, false, false};
    auto backend = sdp::select_backend(kernel_params);
    // strides from packed projection for nested tensors when seq_len is 1 will be
    // and will trigger a contiguous call in the kernel, so we prevent this
    // bool no_seq_len_1_nested = query.is_nested() ? check_for_seq_len_1_nested_tensor(kernel_params, false) : true;
    // The API for transformer_encoder is a mask of shape (Batch_Size, Seq_len_q)
    // For mem-eff attention this will cause the expand call to error
    // For now I am going to turn of that path not have to deal with all the annoying
    // Mask type shape grossness
    if (!mask.has_value() && (backend == sdp::SDPBackend::flash_attention)) {
      auto x = at::linear(query, qkv_weight, qkv_bias);
      auto chunks = x.chunk(3, -1);
      auto x_size_0 = x.size(0);

      chunks[0] = (chunks[0].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[1] = (chunks[1].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[2] = (chunks[2].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      auto y = at::scaled_dot_product_attention(
          chunks[0], chunks[1], chunks[2], mask, 0.0, false, std::nullopt);

      auto past_sdp = y.transpose(1, 2).reshape({x_size_0, -1, embed_dim});
      return std::make_tuple(
          at::linear(past_sdp, proj_weight, proj_bias), Tensor());
    }
    // Returned math or error lets not use it
  }

  return at::cpu::_native_multi_head_attention(
      query,
      key,
      value,
      embed_dim,
      num_head,
      qkv_weight,
      qkv_bias,
      proj_weight,
      proj_bias,
      mask,
      need_weights,
      average_attn_weights,
      mask_type);
}
// clang-format on

// compute q = (q + q_bias) / sqrt(dim_per_head), k = k + k_bias, v = v + v_bias
// [B, T, 3 * D]
// [3 * D]
// [3, B, NH, T, DH]
std::tuple<Tensor, Tensor, Tensor> _TransformBiasRescaleQKV(
    const Tensor& qkv_,
    const Tensor& qkv_bias,
    const int64_t num_head) {
  // TORCH_CHECK(!qkv.is_nested(), "Not support nested q/k/v tensors.");
  auto qkv_proxy = qkv_.is_nested()
      ? c10::MaybeOwned<Tensor>::owned(qkv_.to_padded_tensor(0))
      : c10::MaybeOwned<Tensor>::borrowed(qkv_);
  const auto& qkv = *qkv_proxy;

  auto B = qkv.size(0);
  auto T = qkv.size(1);
  auto _3D = qkv_bias.size(0);
  auto D = _3D / 3;
  TORCH_CHECK(D % num_head == 0);
  const auto dim_per_head = D / num_head;

  auto temp = qkv + qkv_bias;
  temp = temp.view({B, T, 3, D});
  temp = temp.view({B, T, 3, num_head, dim_per_head});
  temp = temp.permute({2, 0, 3, 1, 4}).contiguous();
  auto chunks = temp.chunk(3, 0);

  const auto inv =
      c10::Scalar(1.0 / std::sqrt(static_cast<double>(dim_per_head)));
  chunks[0] = chunks[0].squeeze(0);
  chunks[0].mul_(inv);

  chunks[1] = chunks[1].squeeze(0);
  chunks[2] = chunks[2].squeeze(0);

  return std::make_tuple(
      std::move(chunks[0]), std::move(chunks[1]), std::move(chunks[2]));
}

} // namespace at::musa