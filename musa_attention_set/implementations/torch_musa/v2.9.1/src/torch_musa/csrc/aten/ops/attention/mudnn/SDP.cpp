#include <ATen/ExpandUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_dot_product_attention_flash_musa_backward_native.h>
#include <ATen/ops/_scaled_dot_product_attention_flash_musa_native.h>
#include <ATen/ops/_scaled_dot_product_attention_math_musa_backward_native.h>
#include <ATen/ops/_scaled_dot_product_attention_math_musa_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_cpu_dispatch.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/sum_musa_dispatch.h>
#endif

#include <mudnn.h>

#include "torch_musa/csrc/aten/mudnn/ScaledDotProductAttention.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/MudnnUtils.h"
#include "torch_musa/csrc/aten/utils/Numeric.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

namespace {

// Since MUSA not perform the `convert_boolean_attn_mask` preprocessing work,
// floating/boolean masks will be distinguished as different types.
// clang-format off
enum class MaskType {
  NONE = 0,
  CAUSAL = 1,
  // Assume that full mask shape is [N, H, L, S]
  // Floating masks
  FLT_LS         = 100, // 2D
  FLT_NS         = 101, // key padding
  FLT_HLS_BCAST  = 102, // 3-dim no batch broadcast
  FLT_HLS_FULL   = 103, // 3-dim no batch no broadcast
  FLT_NHLS_BCAST = 104, // 4-dim broadcast
  FLT_NHLS_FULL  = 105, // no broadcast
  // Binary masks
  BIN_LS         = 200, // 2D
  BIN_NS         = 201, // key padding
  BIN_HLS_BCAST  = 202, // 3-dim no batch broadcast
  BIN_HLS_FULL   = 203, // 3-dim no batch no broadcast
  BIN_NHLS_BCAST = 204, // 4-dim broadcast
  BIN_NHLS_FULL  = 205, // no broadcast
};
// clang-format on

MaskType ParseMaskType(
    const std::optional<Tensor>& mask,
    bool is_causal,
    int64_t N,
    int64_t H_q,
    int64_t L,
    int64_t S) {
  if (is_causal) {
    return MaskType::CAUSAL;
  }
  if (!mask.has_value() || !mask->defined()) {
    return MaskType::NONE;
  }

  const auto ndim = mask->dim();
  const auto shape = mask->sizes();
  const auto is_binary_mask = (mask->scalar_type() == ScalarType::Bool);

  if (ndim == 2) {
    if (shape[0] == L && shape[1] == S) {
      return is_binary_mask ? MaskType::BIN_LS : MaskType::FLT_LS;
    }
    if (shape[0] == N && shape[1] == S) {
      return is_binary_mask ? MaskType::BIN_NS : MaskType::FLT_NS;
    }
    TORCH_CHECK(
        false, "MUSA SPDA: shape of 2D attn_mask should be [L, S] or [N, S].");
  }

  if (ndim == 3) {
    if (shape == IntArrayRef({H_q, L, S})) {
      return is_binary_mask ? MaskType::BIN_HLS_FULL : MaskType::FLT_HLS_FULL;
    }
    TORCH_CHECK(
        at::is_expandable_to(shape, {H_q, L, S}),
        "MUSA SPDA: shape of 3D attn_mask should be expandable to [H_q, L, S].");
    return is_binary_mask ? MaskType::BIN_HLS_BCAST : MaskType::FLT_HLS_BCAST;
  }

  if (ndim == 4) {
    if (shape == IntArrayRef({N, H_q, L, S})) {
      return is_binary_mask ? MaskType::BIN_NHLS_FULL : MaskType::FLT_NHLS_FULL;
    }
    TORCH_CHECK(
        at::is_expandable_to(shape, {N, H_q, L, S}),
        "MUSA SPDA: ",
        "shape of 4D attn_mask should be expandable to [N, H_q, L, S].");
    return is_binary_mask ? MaskType::BIN_NHLS_BCAST : MaskType::FLT_NHLS_BCAST;
  }

  TORCH_CHECK(false, "MUSA SPDA: attn_mask should be 2/3/4D Tensor.");
}

bool HasMask(MaskType m) noexcept {
  return (m != MaskType::NONE) && (m != MaskType::CAUSAL);
}

bool IsPadMask(MaskType m) noexcept {
  return (m == MaskType::FLT_NS) || (m == MaskType::BIN_NS);
}

// dim-3 no batch, needs unsqueeze/squeeze
bool IsNoBatchMask(MaskType m) noexcept {
  return (m == MaskType::BIN_HLS_FULL) || (m == MaskType::FLT_HLS_FULL) ||
      (m == MaskType::BIN_HLS_BCAST) || (m == MaskType::FLT_HLS_BCAST);
}

bool IsValidGradMask(MaskType m) noexcept {
  const auto val = static_cast<int64_t>(m);
  constexpr auto low = static_cast<int64_t>(MaskType::FLT_LS);
  constexpr auto high = static_cast<int64_t>(MaskType::FLT_NHLS_FULL);
  return (val >= low) && (val <= high);
}

Tensor CalMaskGrad(
    const Tensor& contig_grad_mask_full,
    const Tensor& mask,
    MaskType mask_type) {
  const auto full_sizes = contig_grad_mask_full.sizes();
  const auto N = full_sizes[0];
  const auto H_q = full_sizes[1];
  const auto L = full_sizes[2];
  const auto S = full_sizes[3];
  switch (mask_type) {
    case MaskType::FLT_LS: {
      Tensor grad = mask.new_empty({L, S});
      at::musa::sum_out(grad, contig_grad_mask_full, {0, 1});
      return grad;
    }
    case MaskType::FLT_NS: {
      Tensor grad = mask.new_empty({N, S});
      at::musa::sum_out(grad, contig_grad_mask_full, {1, 2});
      return grad;
    }
    case MaskType::FLT_HLS_FULL: {
      Tensor grad = mask.new_empty({H_q, L, S});
      at::musa::sum_out(grad, contig_grad_mask_full, {0});
      return grad;
    }
    case MaskType::FLT_NHLS_FULL: {
      return contig_grad_mask_full;
    }
    case MaskType::FLT_HLS_BCAST: {
      const auto mask_sizes = mask.sizes();
      DimVector grad_shape{1};
      grad_shape.insert(grad_shape.end(), mask_sizes.begin(), mask_sizes.end());
      Tensor grad = mask.new_empty(grad_shape);
      DimVector axes;
      if (N != 1) {
        axes.push_back(0);
      }
      const int64_t full_dim = contig_grad_mask_full.dim();
      for (int64_t i = 1; i < full_dim; ++i) {
        if (full_sizes[i] != mask_sizes[i]) {
          axes.push_back(i);
        }
      }
      at::musa::sum_out(grad, contig_grad_mask_full, axes, true);
      return grad.squeeze(0);
    }
    case MaskType::FLT_NHLS_BCAST: {
      const auto mask_sizes = mask.sizes();
      Tensor grad = mask.new_empty(mask_sizes);
      DimVector axes;
      const int64_t full_dim = contig_grad_mask_full.dim();
      for (int64_t i = 0; i < full_dim; ++i) {
        if (full_sizes[i] != mask_sizes[i]) {
          axes.push_back(i);
        }
      }
      at::musa::sum_out(grad, contig_grad_mask_full, axes, true);
      return grad;
    }
    default:
      break;
  }
  return Tensor();
}

void CheckScale(const Tensor& query, std::optional<double> scale) {
#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 4020)
  if (scale.has_value()) {
    using FLT = FloatingPoint<double>;
    using SFT = c10::SymFloat;
    const auto expect_scale = (SFT(1.0) / (SFT(query.sym_size(-1)).sqrt()));
    const auto almost_equal =
        FLT(expect_scale.expect_float()).AlmostEquals(FLT(scale.value()));
    if (!almost_equal) {
      TORCH_CHECK(
          false,
          "Not support explicit `scale` parameter which is ",
          "not equal to default `1/sqrt(query.size(-1)).");
    }
    TORCH_WARN_ONCE(
        "Detected explicit `scale` parameter which is almost equal to default ",
        "`1/sqrt(query.size(-1)), SDPA will treat it as `scale=None`. If this ",
        "is the expected behavior, you can set it manually to avoid this warning.");
  }
#endif
}

void MaybeSetScale(
    ::musa::dnn::ScaledDotProductAttention& sdpa,
    std::optional<double> scale) {
#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION >= 4020)
  if (scale.has_value()) {
    CHECK_MUDNN_STATUS(sdpa.SetScale(scale.value()), "SetScale");
  }
#endif
}

void MathCheckTensorShapes(const Tensor& q, const Tensor& k, const Tensor& v) {
  const auto q_dim = q.dim();
  const auto k_dim = k.dim();
  const auto v_dim = v.dim();
  TORCH_CHECK(
      (q_dim == k_dim) && (q_dim == v_dim) && (q_dim == 3 || q_dim == 4),
      "SDPA on MUSA requires query, key and value to be 3(no batch) ",
      "or 4(with batch) dimensional, but got Query dim: ",
      q_dim,
      ", Key dim: ",
      k_dim,
      ", Value dim: ",
      v_dim,
      " instead.");
}

using Proxy = typename c10::MaybeOwned<Tensor>;
Proxy AffineBatch(const Tensor& orig, bool no_batch) {
  if (no_batch) {
    return Proxy::owned(orig.unsqueeze(0));
  }
  return Proxy::borrowed(orig);
}

Tensor PackRNGState(double dropout_p) {
  using CType = std::uintptr_t;
  constexpr auto SType = c10::CppTypeToScalarType<CType>::value;

  // Always return a defined tensor so downstream consumers (autograd,
  // FakeTensor, DTensor sharding) see a stable schema regardless of
  // whether dropout is actually active.
  if (!(dropout_p > 0.0)) {
    auto state_tensor = at::cpu::empty(
        {2}, SType, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    auto* rng_state = state_tensor.data_ptr<CType>();
    rng_state[0] = 0;
    rng_state[1] = 0;
    return state_tensor;
  }

  auto gen = get_generator_or_default<MUSAGeneratorImpl>(
      std::nullopt, musa::detail::getDefaultMUSAGenerator());
  PhiloxMusaState rng_engine_inputs;
  constexpr uint64_t counter_offset = 4;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_musa_state(counter_offset);
  }

  auto state_tensor = at::cpu::empty(
      {rng_engine_inputs.captured_ ? 3 : 2},
      SType,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt);
  auto* rng_state = state_tensor.data_ptr<CType>();

  if (rng_engine_inputs.captured_) {
    rng_state[0] = reinterpret_cast<CType>(rng_engine_inputs.seed_.ptr);
    rng_state[1] = reinterpret_cast<CType>(rng_engine_inputs.offset_.ptr);
    rng_state[2] = static_cast<CType>(rng_engine_inputs.offset_intragraph_);
  } else {
    rng_state[0] = static_cast<CType>(rng_engine_inputs.seed_.val);
    rng_state[1] = static_cast<CType>(rng_engine_inputs.offset_.val);
  }

  return state_tensor;
}

void UnpackRNGStateToMUDNN(
    ::musa::dnn::ScaledDotProductAttention& sdpa,
    const Tensor& state_tensor) {
  if (!state_tensor.defined()) {
    return;
  }
  using CType = std::uintptr_t;
  auto* rng_state = state_tensor.data_ptr<CType>();

  if (state_tensor.numel() == 2) {
    CHECK_MUDNN_STATUS(
        sdpa.SetSeed(
            static_cast<uint64_t>(rng_state[0]),
            static_cast<uint64_t>(rng_state[1]),
            0, /* kHostSeed */
            nullptr,
            nullptr,
            0),
        "SetSeed");
  } else {
    CHECK_MUDNN_STATUS(
        sdpa.SetSeed(
            0,
            0,
            1, /* kDeviceSeed  */
            reinterpret_cast<const uint64_t*>(rng_state[0]),
            reinterpret_cast<const uint64_t*>(rng_state[1]),
            static_cast<uint64_t>(rng_state[2])),
        "SetSeed");
  }
}

} // anonymous namespace

// query shape: [N, H_q, L, E]
// key   shape: [N, H, S, E]
// value shape: [N, H, S, E_v]
std::tuple<Tensor, Tensor, Tensor> MuDNNMathSDPAFwd(
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  const c10::musa::MUSAGuard device_guard(_query.device());

  MathCheckTensorShapes(_query, _key, _value);
  const auto no_batch = (_query.dim() == 3);
  Proxy batch_q = AffineBatch(_query, no_batch);
  Proxy batch_k = AffineBatch(_key, no_batch);
  Proxy batch_v = AffineBatch(_value, no_batch);
  const auto& query = (*batch_q);
  const auto& key = (*batch_k);
  const auto& value = (*batch_v);

  const auto N = query.size(0);
  const auto H_q = query.size(1);
  const auto L = query.size(2);
  const auto E = query.size(3);

  const auto S = key.size(2);

  const auto E_v = value.size(3);

  auto mask = attn_mask; // remove const
  CheckScale(query, scale);
  const auto query_opt = query.options();
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "MUSA SPDA: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "MUSA SPDA: Nested tensors for query / key are not supported when is_causal=True");
    mask = at::ones({L, S}, query_opt.dtype(at::kBool));
    mask->tril_();
  }
  const auto mask_type = ParseMaskType(mask, is_causal, N, H_q, L, S);

  const auto contig_q = query.contiguous();
  const auto contig_k = key.contiguous();
  const auto contig_v = value.contiguous();

  auto musa_q = CreateMUTensor(contig_q);
  auto musa_k = CreateMUTensor(contig_k);
  auto musa_v = CreateMUTensor(contig_v);

  auto contig_mask =
      mask.has_value() ? mask->contiguous() : at::empty({0}, query_opt);
  if (IsNoBatchMask(mask_type) && no_batch) {
    contig_mask = contig_mask.unsqueeze(0);
  }
  // muDNN requires the mask.dim[2] equals seq_q_len when mask.dim is 4:
  if (contig_mask.dim() == 4 && contig_mask.size(2) == 1 && L != 1) {
    contig_mask = contig_mask.expand({-1, -1, L, -1}).contiguous();
  }
  auto musa_mask = CreateMUTensor(contig_mask);

  auto contig_output =
      at::empty({N, H_q, L, E_v}, query_opt, at::MemoryFormat::Contiguous);
  auto musa_out = CreateMUTensor(contig_output);

  auto contig_attn_weights =
      at::empty({N, H_q, L, S}, query_opt, at::MemoryFormat::Contiguous);
  auto musa_attn_weights = CreateMUTensor(contig_attn_weights);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  CHECK_MUDNN_STATUS(
      sdpa.SetComputeMode(
          musa::GetMatmulComputeModeFromCtx(query.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetMaskMode(IsPadMask(mask_type)), "SetMaskMode");

  MaybeSetScale(sdpa, scale);

  auto contig_dropout_mask = at::empty({0}, query_opt.dtype(at::kBool));
  if (dropout_p > 0.0) {
    contig_dropout_mask.resize_({N, H_q, L, S});
    CHECK_MUDNN_STATUS(sdpa.SetDropoutP(dropout_p), "SetDropoutP");
    CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
  }
  auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

  CHECK_MUDNN_STATUS(
      sdpa.RunMath(
          h,
          musa_out,
          musa_attn_weights,
          musa_q,
          musa_k,
          musa_v,
          musa_mask,
          musa_dropout_mask,
          at::musa::InternalMemAlloc),
      "Run SDPA Math FWD.");

  if (no_batch) {
    contig_output.squeeze_(0);
    contig_attn_weights.squeeze_(0);
    contig_dropout_mask.squeeze_(0);
  }
  return std::make_tuple(
      contig_output, contig_attn_weights, contig_dropout_mask);
}

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
    std::optional<double> scale) {
  const c10::musa::MUSAGuard device_guard(_query.device());

  const auto no_batch = (_query.dim() == 3);
  Proxy batch_q = AffineBatch(_query, no_batch);
  Proxy batch_k = AffineBatch(_key, no_batch);
  Proxy batch_v = AffineBatch(_value, no_batch);
  Proxy batch_go = AffineBatch(_grad_output, no_batch);
  Proxy batch_o = AffineBatch(_output, no_batch);
  Proxy batch_aw = AffineBatch(_attn_weights, no_batch);
  Proxy batch_dm = AffineBatch(_dropout_mask, no_batch);

  const auto& query = (*batch_q);
  const auto& key = (*batch_k);
  const auto& value = (*batch_v);
  const auto& grad_output = (*batch_go);
  const auto& output = (*batch_o);
  const auto& attn_weights = (*batch_aw);
  const auto& dropout_mask = (*batch_dm);

  const auto contig_q = query.contiguous();
  const auto contig_k = key.contiguous();
  const auto contig_v = value.contiguous();

  auto musa_q = CreateMUTensor(contig_q);
  auto musa_k = CreateMUTensor(contig_k);
  auto musa_v = CreateMUTensor(contig_v);

  auto contig_grad_q = at::empty_like(contig_q, query.options());
  auto musa_grad_q = CreateMUTensor(contig_grad_q);

  auto contig_grad_k = at::empty_like(contig_k, key.options());
  auto musa_grad_k = CreateMUTensor(contig_grad_k);

  auto contig_grad_v = at::empty_like(contig_v, value.options());
  auto musa_grad_v = CreateMUTensor(contig_grad_v);

  const auto contig_grad_out = grad_output.contiguous();
  auto musa_grad_out = CreateMUTensor(contig_grad_out);

  const auto contig_attn_weights = attn_weights.contiguous();
  auto musa_attn_weights = CreateMUTensor(contig_attn_weights);

  auto contig_grad_attn_weights = at::empty_like(contig_attn_weights);
  auto musa_grad_attn_weights = CreateMUTensor(contig_grad_attn_weights);

  const auto contig_dropout_mask = dropout_mask.contiguous();
  auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  const auto N = query.size(0);
  const auto H_q = query.size(1);
  const auto L = query.size(2);

  const auto S = key.size(2);

  const auto E_v = value.size(3);

  const auto mask_type = ParseMaskType(attn_mask, is_causal, N, H_q, L, S);

  CHECK_MUDNN_STATUS(
      sdpa.SetComputeMode(
          musa::GetMatmulComputeModeFromCtx(query.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
  MaybeSetScale(sdpa, scale);

  CHECK_MUDNN_STATUS(
      sdpa.RunMathBwd(
          h,
          musa_grad_q,
          musa_grad_k,
          musa_grad_v,
          musa_grad_attn_weights,
          musa_grad_out,
          musa_q,
          musa_k,
          musa_v,
          musa_attn_weights,
          musa_dropout_mask,
          at::musa::InternalMemAlloc),
      "Run SDPA Math BWD.");

  const bool mask_need_grad =
      IsValidGradMask(mask_type) && attn_mask->requires_grad();
  auto attn_mask_grad = mask_need_grad
      ? CalMaskGrad(contig_grad_attn_weights, *attn_mask, mask_type)
      : Tensor();

  if (no_batch) {
    contig_grad_q.squeeze_(0);
    contig_grad_k.squeeze_(0);
    contig_grad_v.squeeze_(0);
  }

  return std::make_tuple(
      contig_grad_q, contig_grad_k, contig_grad_v, attn_mask_grad);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> MuDNNFlashSDPAFwd(
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  const c10::musa::MUSAGuard device_guard(_query.device());

  const auto no_batch = (_query.dim() == 3);
  Proxy batch_q = AffineBatch(_query, no_batch);
  Proxy batch_k = AffineBatch(_key, no_batch);
  Proxy batch_v = AffineBatch(_value, no_batch);
  const auto& query = (*batch_q);
  const auto& key = (*batch_k);
  const auto& value = (*batch_v);

  auto mask = attn_mask; // remove const
  CheckScale(query, scale);
  const auto query_opt = query.options();
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "MUSA SPDA: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "MUSA SPDA: Nested tensors for query / key are not supported when is_causal=True");
  }

  const auto N = query.size(0);
  const auto H_q = query.size(1);
  const auto L = query.size(2);
  const auto E = query.size(3);

  const auto S = key.size(2);

  const auto E_v = value.size(3);

  const auto mask_type = ParseMaskType(mask, is_causal, N, H_q, L, S);

  auto musa_q = CreateMUTensor(query);
  auto musa_k = CreateMUTensor(key);
  auto musa_v = CreateMUTensor(value);

  auto output =
      at::empty({N, L, H_q, E_v}, query_opt, at::MemoryFormat::Contiguous)
          .transpose(1, 2);
  auto musa_out = CreateMUTensor(output);

  auto contig_mask =
      HasMask(mask_type) ? mask->contiguous() : at::empty({0}, query_opt);
  if (IsNoBatchMask(mask_type) && no_batch) {
    contig_mask = contig_mask.unsqueeze(0);
  }
  // muDNN requires the mask.dim[2] equals seq_q_len when mask.dim is 4:
  if (contig_mask.dim() == 4 && contig_mask.size(2) == 1 && L != 1) {
    contig_mask = contig_mask.expand({-1, -1, L, -1}).contiguous();
  }
  auto musa_mask = CreateMUTensor(contig_mask);

  auto contig_logsumexp = at::empty(
      {N, H_q, L}, query_opt.dtype(at::kFloat), at::MemoryFormat::Contiguous);
  auto musa_lse = CreateMUTensor(contig_logsumexp);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  CHECK_MUDNN_STATUS(sdpa.SetCausal(is_causal), "SetCausal");
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetMaskMode(IsPadMask(mask_type)), "SetMaskMode");
  MaybeSetScale(sdpa, scale);

  auto contig_dropout_mask = at::empty({0}, query_opt.dtype(at::kFloat));
  auto rng_state = PackRNGState(dropout_p);
  if (dropout_p > 0.0) {
    CHECK_MUDNN_STATUS(sdpa.SetDropoutP(dropout_p), "SetDropoutP");
    CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
    if (return_debug_mask) {
      contig_dropout_mask.resize_({N, H_q, L, S});
    }
    UnpackRNGStateToMUDNN(sdpa, rng_state);
  }
  auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

  CHECK_MUDNN_STATUS(
      sdpa.RunFlash(
          h,
          musa_out,
          musa_lse,
          musa_q,
          musa_k,
          musa_v,
          musa_mask,
          musa_dropout_mask,
          at::musa::InternalMemAlloc),
      "Run SDPA Flash FWD.");

  if (no_batch) {
    output.squeeze_(0);
    contig_logsumexp.squeeze_(0);
    if (dropout_p > 0.0 && return_debug_mask) {
      contig_dropout_mask.squeeze_(0);
    }
  }
  return std::make_tuple(
      output, contig_logsumexp, contig_dropout_mask, rng_state);
}

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
    std::optional<double> scale) {
  (void)_dropout_mask;
  const c10::musa::MUSAGuard device_guard(_query.device());

  const auto no_batch = (_query.dim() == 3);
  Proxy batch_q = AffineBatch(_query, no_batch);
  Proxy batch_k = AffineBatch(_key, no_batch);
  Proxy batch_v = AffineBatch(_value, no_batch);
  Proxy batch_go = AffineBatch(_grad_output, no_batch);
  Proxy batch_o = AffineBatch(_output, no_batch);
  Proxy batch_lse = AffineBatch(_logsumexp, no_batch);

  const auto& query = (*batch_q);
  const auto& key = (*batch_k);
  const auto& value = (*batch_v);
  const auto& grad_output = (*batch_go);
  const auto& output = (*batch_o);
  const auto& logsumexp = (*batch_lse);

  auto musa_q = CreateMUTensor(query);
  auto musa_k = CreateMUTensor(key);
  auto musa_v = CreateMUTensor(value);

  auto grad_q = at::empty_like(query, query.options());
  auto musa_grad_q = CreateMUTensor(grad_q);

  auto grad_k = at::empty_like(key, key.options());
  auto musa_grad_k = CreateMUTensor(grad_k);

  auto grad_v = at::empty_like(value, value.options());
  auto musa_grad_v = CreateMUTensor(grad_v);

  const auto reformatted_grad_output = ContiguousIfZeroInStrides(grad_output);
  auto musa_grad_output = CreateMUTensor(reformatted_grad_output);

  const auto contig_logsumexp = logsumexp.contiguous();
  auto musa_logsumexp = CreateMUTensor(contig_logsumexp);

  auto musa_output = CreateMUTensor(output);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  const auto N = query.size(0);
  const auto H_q = query.size(1);
  const auto L = query.size(2);

  const auto S = key.size(2);

  const auto E_v = value.size(3);

  const auto mask_type = ParseMaskType(attn_mask, is_causal, N, H_q, L, S);
  auto contig_mask = HasMask(mask_type) ? attn_mask->contiguous()
                                        : at::empty({0}, query.options());
  if (IsNoBatchMask(mask_type) && no_batch) {
    contig_mask = contig_mask.unsqueeze(0);
  }
  auto musa_mask = CreateMUTensor(contig_mask);

  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
  CHECK_MUDNN_STATUS(sdpa.SetDropoutP(dropout_p), "SetDropoutP");
  CHECK_MUDNN_STATUS(sdpa.SetCausal(is_causal), "SetCausal");
  CHECK_MUDNN_STATUS(sdpa.SetMaskMode(IsPadMask(mask_type)), "SetMaskMode");
  MaybeSetScale(sdpa, scale);

  if (dropout_p > 0.0) {
    UnpackRNGStateToMUDNN(sdpa, rng_state);
  }
  auto musa_dropout_mask = CreateEmptyMUTensor();

  CHECK_MUDNN_STATUS(
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
          at::musa::InternalMemAlloc),
      "Run SDPA Flash BWD.");

  if (no_batch) {
    grad_q.squeeze_(0);
    grad_k.squeeze_(0);
    grad_v.squeeze_(0);
  }

  return std::make_tuple(grad_q, grad_k, grad_v, Tensor());
}

// query shape: [total_seq_q, H, D]
// key   shape: [total_seq_kv, H, D]
// value shape: [total_seq_kv, H, D_v]
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
    bool is_causal) {
  const c10::musa::MUSAGuard device_guard(query.device());

  TORCH_CHECK(
      at::musa::getMUSAArch() != 220,
      "MUSA Flash Attention Varlen (variable-length / nested tensors) is not "
      "supported on mp22 (arch 2.2). Please run on mp31 (arch 3.1) or newer.");

  TORCH_CHECK(query.dim() == 3, "query's layout should be [total_seq_q, H, D]");
  TORCH_CHECK(key.dim() == 3, "key's layout should be [total_seq_q, H, D]");
  TORCH_CHECK(value.dim() == 3, "value's layout should be [total_seq_q, H, D]");

  CheckScale(query, scale);
  const auto query_opt = query.options();
  if (is_causal) {
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "MUSA SPDA: Nested tensors for query / key are not supported "
        "when is_causal=True");
  }

  const int S_q = query.size(0);
  const int H = query.size(1);
  const int D = query.size(2);

  const int S_kv = key.size(0);
  const int D_v = value.size(2);
  const int B = cu_seqlens_q.numel() - 1;

  at::musa::muTensor musa_q = at::musa::CreateMUTensor(query);
  at::musa::muTensor musa_k = at::musa::CreateMUTensor(key);
  at::musa::muTensor musa_v = at::musa::CreateMUTensor(value);
  at::musa::muTensor musa_cu_seqlens_q = at::musa::CreateMUTensor(cu_seqlens_q);
  at::musa::muTensor musa_cu_seqlens_k = at::musa::CreateMUTensor(cu_seqlens_k);

  at::Tensor output =
      at::empty({S_q, H, D_v}, query_opt, at::MemoryFormat::Contiguous);
  at::musa::muTensor musa_out = at::musa::CreateMUTensor(output);

  at::Tensor contig_logsumexp = at::empty(
      {B, H, max_seqlen_q},
      query_opt.dtype(at::kFloat),
      at::MemoryFormat::Contiguous);
  at::musa::muTensor musa_lse = at::musa::CreateMUTensor(contig_logsumexp);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H * D_v), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetBatchSize(B), "SetBatchSize");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetMaxSeqlenQ(max_seqlen_q), "SetMaxSeqlenQ");
  CHECK_MUDNN_STATUS(sdpa.SetMaxSeqlenK(max_seqlen_k), "SetMaxSeqlenK");
  CHECK_MUDNN_STATUS(sdpa.SetCausal(is_causal), "SetCausal");
  MaybeSetScale(sdpa, scale);

  at::Tensor contig_dropout_mask = at::empty({0}, query_opt.dtype(at::kBool));
  if (dropout_p > 0.0) {
    contig_dropout_mask.resize_({B, S_q, H, D});
    CHECK_MUDNN_STATUS(sdpa.SetDropoutP(dropout_p), "SetDropoutP");
    CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
  }
  at::musa::muTensor musa_dropout_mask =
      at::musa::CreateMUTensor(contig_dropout_mask);

  CHECK_MUDNN_STATUS(
      sdpa.RunFlashVarlen(
          h,
          musa_out,
          musa_lse,
          musa_q,
          musa_k,
          musa_v,
          /*attn_mask=*/at::musa::muTensor(),
          musa_dropout_mask,
          musa_cu_seqlens_q,
          musa_cu_seqlens_k,
          at::musa::InternalMemAlloc),
      "Run SDPA Flash Varlen FWD.");

  return std::make_tuple(output, contig_logsumexp, contig_dropout_mask);
}

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
    bool is_causal) {
  const c10::musa::MUSAGuard device_guard(query.device());

  TORCH_CHECK(
      at::musa::getMUSAArch() != 220,
      "MUSA Flash Attention Varlen (variable-length / nested tensors) is not "
      "supported on mp22 (arch 2.2). Please run on mp31 (arch 3.1) or newer.");

  at::musa::muTensor musa_q = at::musa::CreateMUTensor(query);
  at::musa::muTensor musa_k = at::musa::CreateMUTensor(key);
  at::musa::muTensor musa_v = at::musa::CreateMUTensor(value);
  at::musa::muTensor musa_grad_q = at::musa::CreateMUTensor(grad_q);
  at::musa::muTensor musa_grad_k = at::musa::CreateMUTensor(grad_k);
  at::musa::muTensor musa_grad_v = at::musa::CreateMUTensor(grad_v);
  at::musa::muTensor musa_cu_seqlens_q = at::musa::CreateMUTensor(cu_seqlens_q);
  at::musa::muTensor musa_cu_seqlens_k = at::musa::CreateMUTensor(cu_seqlens_k);

  const at::Tensor reformatted_grad_output =
      at::musa::ContiguousIfZeroInStrides(grad_output);
  at::musa::muTensor musa_grad_output =
      at::musa::CreateMUTensor(reformatted_grad_output);

  const at::Tensor contig_logsumexp = logsumexp.contiguous();
  at::musa::muTensor musa_logsumexp =
      at::musa::CreateMUTensor(contig_logsumexp);

  at::musa::muTensor musa_output = at::musa::CreateMUTensor(output);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  const int S_q = query.size(0);
  const int H = query.size(1);
  const int D = query.size(2);
  const int S_kv = key.size(0);
  const int D_v = value.size(2);
  const int B = cu_seqlens_q.numel() - 1;

  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H * D_v), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetBatchSize(B), "SetBatchSize");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetMaxSeqlenQ(max_seqlen_q), "SetMaxSeqlenQ");
  CHECK_MUDNN_STATUS(sdpa.SetMaxSeqlenK(max_seqlen_k), "SetMaxSeqlenK");
  CHECK_MUDNN_STATUS(sdpa.SetCausal(is_causal), "SetCausal");
  CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
  MaybeSetScale(sdpa, scale);

  CHECK_MUDNN_STATUS(
      sdpa.RunFlashVarlenBwd(
          h,
          musa_grad_q,
          musa_grad_k,
          musa_grad_v,
          musa_grad_output,
          musa_q,
          musa_k,
          musa_v,
          /*mask=*/at::musa::muTensor(),
          musa_output,
          musa_logsumexp,
          /*dropout_mask=*/at::musa::muTensor(),
          musa_cu_seqlens_q,
          musa_cu_seqlens_k,
          at::musa::InternalMemAlloc),
      "Run SDPA Flash BWD.");

  return {grad_q, grad_k, grad_v, at::Tensor()};
}
} // namespace at::musa
