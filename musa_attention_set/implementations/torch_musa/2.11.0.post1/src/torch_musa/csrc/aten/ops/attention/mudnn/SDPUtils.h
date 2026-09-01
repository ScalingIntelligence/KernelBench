#ifndef TORCH_MUSA_CSRC_ATEN_OPS_ATTENTION_MUDNN_SDPUTILS_H_
#define TORCH_MUSA_CSRC_ATEN_OPS_ATTENTION_MUDNN_SDPUTILS_H_

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_sdp_choice_native.h>
#endif
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/musa/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"

namespace at::musa {
std::tuple<at::Tensor, at::Tensor, at::Tensor> MuDNNFlashVarlenFwd(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    double dropout_p,
    std::optional<double> scale,
    bool is_causal);

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor>
MuDNNFlashVarlenBwd(
    at::Tensor& grad_output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& output,
    const at::Tensor& logsumexp,
    at::Tensor& grad_q,
    at::Tensor& grad_k,
    at::Tensor& grad_v,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    double dropout_p,
    std::optional<double> scale,
    bool is_causal);

std::tuple<Tensor, Tensor, Tensor> MuDNNMathSDPAFwd(
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale);

std::tuple<Tensor, Tensor, Tensor, Tensor> MuDNNMathSDPABwd(
    const Tensor& _grad_output,
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const Tensor& _output,
    const Tensor& _attn_weights,
    const Tensor& _dropout_mask,
    bool is_causal,
    const std::optional<Tensor>& attn_mask,
    std::optional<double> scale);

std::tuple<Tensor, Tensor, Tensor, Tensor> MuDNNFlashSDPAFwd(
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale);

std::tuple<Tensor, Tensor, Tensor, Tensor> MuDNNFlashSDPABwd(
    const Tensor& _grad_output,
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const Tensor& _output,
    const Tensor& _logsumexp,
    const Tensor& _dropout_mask,
    const Tensor& rng_state,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& attn_mask,
    std::optional<double> scale);

} // namespace at::musa

namespace sdp {

inline bool check_musa_arch(const sdp_params& params, bool is_debug) {
  // Check that the gpu is capable of running flash attention
  const auto device_prop = at::musa::getCurrentDeviceProperties();
  const auto arch_major = device_prop->major;
  const auto arch_minor = device_prop->minor;
  const bool enable_flash_attn =
      (arch_major > 2) || (arch_major == 2 && arch_minor > 1);

  if (!enable_flash_attn) {
    if (is_debug) {
      TORCH_WARN(
          "Flash attention only supports architecture with mp version >= 2.2, "
          "but now attempts to run on a mp ",
          arch_major,
          ".",
          arch_minor,
          " gpu.");
    }
    return false;
  }
  return true;
}

inline bool check_musa_attention_input(
    const sdp_params& params,
    bool is_debug) {
  const int64_t qk_head_dim = params.query.size(-1);
  const int64_t v_head_dim = params.value.size(-1);

  const auto arch = at::musa::getMUSAArch();
  const bool fwd_only = (!input_requires_grad(params));

  const bool is_head_dim_qkv_128 = (qk_head_dim <= 128 && v_head_dim <= 128);
  const bool is_head_dim_qkv_160 = (qk_head_dim == 160 && v_head_dim == 160);
  const bool is_head_dim_qkv_256 = (qk_head_dim == 256 && v_head_dim == 256);
  // const bool is_head_dim_qkv_384 = (qk_head_dim == 384 && v_head_dim == 384);
  const bool is_head_dim_qkv_512 = (qk_head_dim == 512 && v_head_dim == 512);
  const bool is_head_dim_qk192_v128 = (qk_head_dim == 192 && v_head_dim == 128);

  bool enable = is_head_dim_qkv_128;
  if (arch == 220) {
    if (fwd_only) {
      enable = enable || is_head_dim_qkv_160;
    } else {
      enable = enable && (qk_head_dim >= 64 && v_head_dim >= 64);
    }
  } else if (arch == 310) {
    enable = enable || is_head_dim_qkv_160;
    enable = enable || is_head_dim_qkv_256;
    // enable = enable || is_head_dim_qkv_384;
    enable = enable || is_head_dim_qkv_512;
    enable = enable || is_head_dim_qk192_v128;
  }

  if (params.dropout > 0.0 && is_head_dim_qkv_128) {
    if (is_debug) {
      TORCH_WARN(
          "Flash SDPA does not support dropout with qk_head_dim: ",
          qk_head_dim,
          " v_head_dim: ",
          v_head_dim);
    }
    return false;
  }

  if (!enable) {
    if (is_debug) {
      TORCH_WARN(
          "Unsupported qk_head_dim: ",
          qk_head_dim,
          " v_head_dim: ",
          v_head_dim,
          " for FlashAttention in MUSA backend");
    }
    return false;
  }
  return true;
}

inline bool check_musa_tensor_shapes(const sdp_params& params, bool is_debug) {
  const auto query_dim = params.query.dim();
  const auto qkv_same_dim =
      (query_dim == params.key.dim() && query_dim == params.value.dim());
  const auto valid_dim = (query_dim == 4 || query_dim == 3);

  if (!(qkv_same_dim && valid_dim)) {
    if (is_debug) {
      TORCH_WARN(
          "All fused kernels requires query, key and value to be 3(no batch) ",
          "or 4(with batch) dimensional, but got Query dim: ",
          query_dim,
          ", Key dim: ",
          params.key.dim(),
          ", Value dim: ",
          params.value.dim(),
          " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_musa_attn_mask(const sdp_params& params, bool is_debug) {
  auto attn_mask = params.attn_mask;
  if (!attn_mask.has_value()) {
    return true;
  }
  if (attn_mask.value().requires_grad()) {
    if (is_debug) {
      TORCH_WARN(
          "MUSA Flash SDPA does not support calculate attn_mask gradient.");
    }
    return false;
  }
  return true;
}

inline bool use_flash_attention(const sdp_params& params) {
  using SDPParamsCheckFunc = bool (*)(const sdp_params&, bool);
  constexpr int conditions_num = 5;
  constexpr std::array<SDPParamsCheckFunc, conditions_num> conditions{
      {check_runtime_disabled_flash,
       check_musa_tensor_shapes,
       check_musa_attention_input,
       check_musa_attn_mask,
       check_musa_arch}};
  auto res = std::all_of(
      conditions.begin(), conditions.end(), [&params](SDPParamsCheckFunc func) {
        return func(params, true);
      });
  if (!res) {
    return false;
  }
  static const std::array<at::ScalarType, 2> musa_allowed_dtypes{
      at::kHalf, at::kBFloat16};
  return check_tensor_dtype(params, musa_allowed_dtypes, true);
}

inline SDPBackend select_backend(const sdp_params& params) {
  const auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP()) {
    return SDPBackend::error;
  }
  if (use_flash_attention(params)) {
    return SDPBackend::flash_attention;
  } else if (ctx.userEnabledMathSDP()) {
    return SDPBackend::math;
  } else {
    TORCH_CHECK(false, "Invalid backend configuration for musa!")
  }
}

} // namespace sdp

#endif // TORCH_MUSA_CSRC_ATEN_OPS_ATTENTION_MUDNN_SDPUTILS_H_
