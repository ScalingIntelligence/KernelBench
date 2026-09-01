#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_sdp_choice_native.h>
#endif

#include <ATen/native/transformers/attention.h>
#include <c10/util/Exception.h>

#include "torch_musa/csrc/aten/ops/attention/mudnn/SDPUtils.h"

namespace at {
namespace musa {

int64_t _fused_sdp_choice_musa(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  const sdp::sdp_params params{
      query, key, value, attn_mask, dropout_p, is_causal, enable_gqa};
  const auto backend = sdp::select_backend(params);
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention in musa",
        "This is likely due to turning off both the math kernel and the flash kernel.");
  }
  return static_cast<int64_t>(backend);
}

} // namespace musa

namespace native {

REGISTER_MUSA_DISPATCH(_fused_sdp_choice_stub, &musa::_fused_sdp_choice_musa);

} // namespace native

} // namespace at
