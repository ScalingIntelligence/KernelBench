#include <mudnn_xmma.h>
#include <musa.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cstdint>
#include <optional>
#include <tuple>

#include "gemm_mudnn_utils.hpp"

namespace mate::gemm::mudnn {

struct BMMArgs {
  ffi::TensorView                a;
  ffi::TensorView                b;
  ffi::TensorView                d;
  std::optional<ffi::TensorView> c;
  std::optional<ffi::TensorView> scale_a;
  std::optional<ffi::TensorView> scale_b;
  std::tuple<int64_t, int64_t>   recipe_a;
  std::tuple<int64_t, int64_t>   recipe_b;
  bool                           trans_a;
  bool                           trans_b;
  bool                           fixed_scale_layout;
};

namespace {

void validate_bmm_quant_mode(TensorQuantMode quant_mode_a,
                             TensorQuantMode quant_mode_b,
                             int             granularity_m,
                             int             granularity_n,
                             int             granularity_k) {
  const bool tensor_tensor = quant_mode_a == TensorQuantMode::TENSOR && quant_mode_b == TensorQuantMode::TENSOR &&
                             granularity_m == -1 && granularity_n == -1 && granularity_k == -1;
  const bool channel_tensor = quant_mode_a == TensorQuantMode::CHANNEL && quant_mode_b == TensorQuantMode::TENSOR &&
                              granularity_m == 1 && granularity_n == -1 && granularity_k == -1;
  const bool channel_channel = quant_mode_a == TensorQuantMode::CHANNEL && quant_mode_b == TensorQuantMode::CHANNEL &&
                               granularity_m == 1 && granularity_n == 1 && granularity_k == -1;
  const bool group_group = quant_mode_a == TensorQuantMode::GROUP && quant_mode_b == TensorQuantMode::GROUP &&
                           granularity_m == 1 && granularity_n == 1 && granularity_k == 128;
  const bool group_block = quant_mode_a == TensorQuantMode::GROUP && quant_mode_b == TensorQuantMode::BLOCK &&
                           granularity_m == 1 && granularity_n == 128 && granularity_k == 128;
  if (!tensor_tensor && !channel_tensor && !channel_channel && !group_group && !group_block) {
    TVM_FFI_THROW(ValueError) << "FP8 bmm got unsupported recipe_a and recipe_b";
  }
}

void run_bmm_mudnn(const BMMArgs&                  args,
                   TensorMajor                     major_a,
                   TensorMajor                     major_b,
                   int                             batch,
                   int                             m,
                   int                             n,
                   int                             k,
                   const musa::dnn::MatMulLtParam& lt_param) {
  ffi::MUSADeviceGuard device_guard(args.a.device().device_id);
  const int64_t        a_dim0 = major_a == TensorMajor::K ? m : k;
  const int64_t        a_dim1 = major_a == TensorMajor::K ? k : m;
  const int64_t        b_dim0 = major_b == TensorMajor::K ? n : k;
  const int64_t        b_dim1 = major_b == TensorMajor::K ? k : n;

  auto a = make_mudnn_tensor(args.a.data_ptr(),
                             args.a.dtype(),
                             {batch, a_dim0, a_dim1},
                             {args.a.stride(0), args.a.stride(1), args.a.stride(2)});
  auto b = make_mudnn_tensor(args.b.data_ptr(),
                             args.b.dtype(),
                             {batch, b_dim0, b_dim1},
                             {args.b.stride(0), args.b.stride(1), args.b.stride(2)});
  auto d = make_mudnn_tensor(
      args.d.data_ptr(), args.d.dtype(), {batch, m, n}, {args.d.stride(0), args.d.stride(1), args.d.stride(2)});
  musa::dnn::Tensor c;
  if (args.c.has_value()) {
    c = make_mudnn_tensor(args.c.value().data_ptr(),
                          args.c.value().dtype(),
                          {batch, m, n},
                          {args.c.value().stride(0), args.c.value().stride(1), args.c.value().stride(2)});
  }

  musa::dnn::Handle handle(args.a.device().device_id);
  init_mudnn_handle(handle, get_stream(args.a.device()));
  run_mudnn_lt_matmul(handle, d, a, b, c, args.c.has_value(), major_a, major_b, lt_param);
}

void bmm_16bit(const BMMArgs& args) {
  check_mp31(args.a.device(), "bmm");
  CHECK_MUSA(args.a);
  CHECK_MUSA(args.b);
  CHECK_MUSA(args.d);
  CHECK_DEVICE(args.a, args.b);
  CHECK_DEVICE(args.a, args.d);
  CHECK_DIM(3, args.a);
  CHECK_DIM(3, args.b);
  CHECK_DIM(3, args.d);
  if (args.c.has_value()) {
    CHECK_MUSA(args.c.value());
    CHECK_DEVICE(args.a, args.c.value());
    CHECK_DIM(3, args.c.value());
    TVM_FFI_ICHECK_EQ(args.c.value().stride(-1), 1) << "c must be contiguous at the last dimension";
  }
  TVM_FFI_ICHECK_EQ(args.d.stride(-1), 1) << "d must be contiguous at the last dimension";

  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(args.a.dtype())) << "a must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(args.b.dtype())) << "b must be bf16 or fp16";
  TVM_FFI_ICHECK(dtype_equal(args.a.dtype(), args.b.dtype())) << "a and b must have the same dtype";
  TVM_FFI_ICHECK(is_bf16_or_fp16_or_fp32_dtype(args.d.dtype())) << "d must be bf16, fp16 or fp32";
  TVM_FFI_ICHECK(dtype_equal(args.d.dtype(), args.a.dtype()) || dtype_equal(args.d.dtype(), dl_float32))
      << "d must have the same dtype as a and b, or be fp32";
  if (args.c.has_value()) {
    TVM_FFI_ICHECK(dtype_equal(args.c.value().dtype(), args.d.dtype())) << "c must have the same dtype as d";
  }
  const TensorMajor major_a = args.trans_a ? TensorMajor::MN : TensorMajor::K;
  const TensorMajor major_b = args.trans_b ? TensorMajor::K : TensorMajor::MN;
  const int         batch   = static_cast<int>(args.a.size(0));
  const int         m       = static_cast<int>(major_a == TensorMajor::K ? args.a.size(1) : args.a.size(2));
  const int         k       = static_cast<int>(major_a == TensorMajor::K ? args.a.size(2) : args.a.size(1));
  const int         n       = static_cast<int>(major_b == TensorMajor::K ? args.b.size(1) : args.b.size(2));
  const int         b_k     = static_cast<int>(major_b == TensorMajor::K ? args.b.size(2) : args.b.size(1));

  TVM_FFI_ICHECK_EQ(args.b.size(0), batch);
  TVM_FFI_ICHECK_EQ(b_k, k);
  TVM_FFI_ICHECK_EQ(args.d.size(0), batch);
  TVM_FFI_ICHECK_EQ(args.d.size(1), m);
  TVM_FFI_ICHECK_EQ(args.d.size(2), n);
  if (args.c.has_value()) {
    TVM_FFI_ICHECK_EQ(args.c.value().size(0), batch);
    TVM_FFI_ICHECK_EQ(args.c.value().size(1), m);
    TVM_FFI_ICHECK_EQ(args.c.value().size(2), n);
  }

  if (common::gemm_early_return(batch, m, n, k, args.d, args.c.has_value())) {
    return;
  }

  TVM_FFI_ICHECK_EQ(args.a.stride(-1), 1) << "a must be contiguous in the declared major dimension";
  TVM_FFI_ICHECK_EQ(args.b.stride(-1), 1) << "b must be contiguous in the declared major dimension";

  musa::dnn::MatMulLtParam lt_param;
  run_bmm_mudnn(args, major_a, major_b, batch, m, n, k, lt_param);
}

struct ScaleStrides {
  int64_t mn;
  int64_t k;
};

ScaleStrides validate_scale_shape(const ffi::TensorView& scale,
                                  const char*            name,
                                  int                    batch,
                                  int                    mn,
                                  int                    k,
                                  int                    granularity_mn,
                                  int                    granularity_k,
                                  bool                   fixed_scale_layout) {
  const bool expects_scalar = granularity_mn == -1 && granularity_k == -1;
  if (expects_scalar) {
    if (scale.ndim() != 0) {
      TVM_FFI_THROW(ValueError) << name << " must be a scalar tensor";
    }
    return {0, 0};
  }

  if (scale.ndim() != 3) {
    TVM_FFI_THROW(ValueError) << name << " must be a 3D tensor";
  }
  const int mn_extent = granularity_mn == -1 ? 1 : mutlass::ceil_div(mn, granularity_mn);
  const int k_extent  = granularity_k == -1 ? 1 : mutlass::ceil_div(k, granularity_k);
  if (scale.size(0) != batch) {
    TVM_FFI_THROW(ValueError) << name << " batch dimension mismatch";
  }
  const bool k_major_shape  = scale.size(1) == mn_extent && scale.size(2) == k_extent;
  const bool mn_major_shape = scale.size(1) == k_extent && scale.size(2) == mn_extent;
  if (!k_major_shape && !mn_major_shape) {
    TVM_FFI_THROW(ValueError) << name << " must have K-major shape (" << batch << ", " << mn_extent << ", " << k_extent
                              << ") or MN-major shape (" << batch << ", " << k_extent << ", " << mn_extent << ")";
  }
  const bool physical_mn_major = mn_major_shape && (!k_major_shape || fixed_scale_layout);
  return physical_mn_major ? ScaleStrides{scale.stride(2), scale.stride(1)}
                           : ScaleStrides{scale.stride(1), scale.stride(2)};
}

musa::dnn::Tensor make_mudnn_scale_tensor(
    const ffi::TensorView& scale, int mn_extent, int k_extent, const ScaleStrides& strides, bool fixed_scale_layout) {
  const int64_t dim1    = fixed_scale_layout ? k_extent : mn_extent;
  const int64_t dim2    = fixed_scale_layout ? mn_extent : k_extent;
  const int64_t stride1 = fixed_scale_layout ? strides.k : strides.mn;
  const int64_t stride2 = fixed_scale_layout ? strides.mn : strides.k;
  return make_mudnn_tensor(
      scale.data_ptr(), dl_float32, {scale.size(0), dim1, dim2}, {scale.stride(0), stride1, stride2});
}

void bmm_8bit(const BMMArgs& args) {
  TVM_FFI_ICHECK(args.scale_a.has_value()) << "bmm_8bit requires scale_a";
  TVM_FFI_ICHECK(args.scale_b.has_value()) << "bmm_8bit requires scale_b";
  const ffi::TensorView& scale_a = args.scale_a.value();
  const ffi::TensorView& scale_b = args.scale_b.value();

  check_mp31(args.a.device(), "bmm");
  CHECK_MUSA(args.a);
  CHECK_MUSA(args.b);
  CHECK_MUSA(scale_a);
  CHECK_MUSA(scale_b);
  CHECK_MUSA(args.d);
  CHECK_DEVICE(args.a, args.b);
  CHECK_DEVICE(args.a, scale_a);
  CHECK_DEVICE(args.a, scale_b);
  CHECK_DEVICE(args.a, args.d);
  CHECK_DIM(3, args.a);
  CHECK_DIM(3, args.b);
  CHECK_DIM(3, args.d);
  if (args.c.has_value()) {
    CHECK_MUSA(args.c.value());
    CHECK_DEVICE(args.a, args.c.value());
    CHECK_DIM(3, args.c.value());
    TVM_FFI_ICHECK_EQ(args.c.value().stride(-1), 1) << "c must be contiguous at the last dimension";
  }
  TVM_FFI_ICHECK_EQ(args.d.stride(-1), 1) << "d must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(scale_a.dtype(), dl_float32) << "scale_a must be float32";
  TVM_FFI_ICHECK_EQ(scale_b.dtype(), dl_float32) << "scale_b must be float32";

  TVM_FFI_ICHECK(is_fp8_dtype(args.a.dtype())) << "a must be fp8";
  TVM_FFI_ICHECK(is_fp8_dtype(args.b.dtype())) << "b must be fp8";
  TVM_FFI_ICHECK(is_bf16_or_fp16_or_fp32_dtype(args.d.dtype())) << "d must be bf16, fp16 or fp32";
  if (args.c.has_value()) {
    TVM_FFI_ICHECK(dtype_equal(args.c.value().dtype(), args.d.dtype())) << "c must have the same dtype as d";
    TVM_FFI_ICHECK(dtype_equal(args.d.dtype(), dl_float32)) << "FP8 bmm with c only supports fp32 d";
  }
  const int granularity_m   = static_cast<int>(std::get<0>(args.recipe_a));
  const int granularity_n   = static_cast<int>(std::get<0>(args.recipe_b));
  const int granularity_k_a = static_cast<int>(std::get<1>(args.recipe_a));
  const int granularity_k_b = static_cast<int>(std::get<1>(args.recipe_b));
  TVM_FFI_ICHECK_EQ(granularity_k_a, granularity_k_b) << "recipe_a and recipe_b must use matching K granularity";
  const int             granularity_k = granularity_k_a;
  const TensorQuantMode quant_mode_a  = get_tensor_quant_mode(granularity_m, granularity_k);
  const TensorQuantMode quant_mode_b  = get_tensor_quant_mode(granularity_n, granularity_k);
  validate_bmm_quant_mode(quant_mode_a, quant_mode_b, granularity_m, granularity_n, granularity_k);
  const bool group_or_block_scaled =
      (quant_mode_a == TensorQuantMode::GROUP || quant_mode_a == TensorQuantMode::BLOCK) &&
      (quant_mode_b == TensorQuantMode::GROUP || quant_mode_b == TensorQuantMode::BLOCK);
  TVM_FFI_ICHECK(!(group_or_block_scaled && dtype_equal(args.a.dtype(), dl_float8_e4m3fn) &&
                   dtype_equal(args.b.dtype(), dl_float8_e5m2)))
      << "FP8 bmm group/block scaling does not support E4M3 a with E5M2 b";

  const TensorMajor major_a = args.trans_a ? TensorMajor::MN : TensorMajor::K;
  const TensorMajor major_b = args.trans_b ? TensorMajor::K : TensorMajor::MN;
  const int         batch   = static_cast<int>(args.a.size(0));
  const int         m       = static_cast<int>(major_a == TensorMajor::K ? args.a.size(1) : args.a.size(2));
  const int         k       = static_cast<int>(major_a == TensorMajor::K ? args.a.size(2) : args.a.size(1));
  const int         n       = static_cast<int>(major_b == TensorMajor::K ? args.b.size(1) : args.b.size(2));
  const int         b_k     = static_cast<int>(major_b == TensorMajor::K ? args.b.size(2) : args.b.size(1));

  TVM_FFI_ICHECK_EQ(args.b.size(0), batch);
  TVM_FFI_ICHECK_EQ(b_k, k);
  TVM_FFI_ICHECK_EQ(args.d.size(0), batch);
  TVM_FFI_ICHECK_EQ(args.d.size(1), m);
  TVM_FFI_ICHECK_EQ(args.d.size(2), n);
  if (args.c.has_value()) {
    TVM_FFI_ICHECK_EQ(args.c.value().size(0), batch);
    TVM_FFI_ICHECK_EQ(args.c.value().size(1), m);
    TVM_FFI_ICHECK_EQ(args.c.value().size(2), n);
  }

  const bool has_non_scalar_scale = scale_a.ndim() != 0 || scale_b.ndim() != 0;
  if (has_non_scalar_scale && (args.trans_a || !args.trans_b) && !args.fixed_scale_layout) {
    TVM_FFI_THROW(ValueError) << "non-NT FP8 bmm scales require fixed_scale_layout=true";
  }
  const ScaleStrides scale_a_strides =
      validate_scale_shape(scale_a, "scale_a", batch, m, k, granularity_m, granularity_k, args.fixed_scale_layout);
  const ScaleStrides scale_b_strides =
      validate_scale_shape(scale_b, "scale_b", batch, n, k, granularity_n, granularity_k, args.fixed_scale_layout);

  if (common::gemm_early_return(batch, m, n, k, args.d, args.c.has_value())) {
    return;
  }

  TVM_FFI_ICHECK_EQ(args.a.stride(-1), 1) << "a must be contiguous in the declared major dimension";
  TVM_FFI_ICHECK_EQ(args.b.stride(-1), 1) << "b must be contiguous in the declared major dimension";

  ffi::MUSADeviceGuard     device_guard(args.a.device().device_id);
  musa::dnn::MatMulLtParam lt_param;
  if (quant_mode_a == TensorQuantMode::TENSOR && quant_mode_b == TensorQuantMode::TENSOR) {
    // Apply one scalar scale to each complete operand.
    TVM_FFI_ICHECK_EQ(scale_a.ndim(), 0) << "scale_a must be a scalar tensor";
    TVM_FFI_ICHECK_EQ(scale_b.ndim(), 0) << "scale_b must be a scalar tensor";
    musa::dnn::Tensor mudnn_scale_a = make_mudnn_scalar_tensor(scale_a.data_ptr(), dl_float32);
    musa::dnn::Tensor mudnn_scale_b = make_mudnn_scalar_tensor(scale_b.data_ptr(), dl_float32);
    MATE_MUDNN_STATUS_CHECK(lt_param.SetScale(mudnn_scale_a, mudnn_scale_b, musa::dnn::Tensor{}, musa::dnn::Tensor{}));
  } else if (quant_mode_a == TensorQuantMode::CHANNEL && quant_mode_b == TensorQuantMode::TENSOR) {
    // Apply one scale per A row and one scalar scale to B.
    TVM_FFI_ICHECK_EQ(scale_b.ndim(), 0) << "scale_b must be a scalar tensor";
    musa::dnn::Tensor mudnn_scale_a = make_mudnn_scale_tensor(scale_a, m, 1, scale_a_strides, args.fixed_scale_layout);
    musa::dnn::Tensor mudnn_scale_b = make_mudnn_scalar_tensor(scale_b.data_ptr(), dl_float32);
    MATE_MUDNN_STATUS_CHECK(lt_param.SetScale(
        mudnn_scale_a, mudnn_scale_b, musa::dnn::Tensor{}, musa::dnn::Tensor{}, 0, args.fixed_scale_layout));
  } else if (quant_mode_a == TensorQuantMode::CHANNEL && quant_mode_b == TensorQuantMode::CHANNEL) {
    // Apply independent row-wise scales to A and B.
    musa::dnn::Tensor mudnn_scale_a = make_mudnn_scale_tensor(scale_a, m, 1, scale_a_strides, args.fixed_scale_layout);
    musa::dnn::Tensor mudnn_scale_b = make_mudnn_scale_tensor(scale_b, n, 1, scale_b_strides, args.fixed_scale_layout);
    MATE_MUDNN_STATUS_CHECK(lt_param.SetScale(
        mudnn_scale_a, mudnn_scale_b, musa::dnn::Tensor{}, musa::dnn::Tensor{}, 0, args.fixed_scale_layout));
  } else {
    // Apply K-grouped scales; B uses N granularity 1 for GROUP and 128 for BLOCK.
    const int scale_a_m = mutlass::ceil_div(m, granularity_m);
    const int scale_a_k = mutlass::ceil_div(k, granularity_k);
    const int scale_b_n = mutlass::ceil_div(n, granularity_n);
    const int scale_b_k = mutlass::ceil_div(k, granularity_k);

    musa::dnn::Tensor mudnn_scale_a =
        make_mudnn_scale_tensor(scale_a, scale_a_m, scale_a_k, scale_a_strides, args.fixed_scale_layout);
    musa::dnn::Tensor mudnn_scale_b =
        make_mudnn_scale_tensor(scale_b, scale_b_n, scale_b_k, scale_b_strides, args.fixed_scale_layout);
    MATE_MUDNN_STATUS_CHECK(lt_param.SetScale(mudnn_scale_a,
                                              mudnn_scale_b,
                                              musa::dnn::Tensor{},
                                              musa::dnn::Tensor{},
                                              granularity_k,
                                              args.fixed_scale_layout));
  }
  run_bmm_mudnn(args, major_a, major_b, batch, m, n, k, lt_param);
}

}  // namespace

void bmm(ffi::TensorView                     a,
         ffi::TensorView                     b,
         ffi::TensorView                     d,
         std::optional<ffi::TensorView>      c,
         std::optional<ffi::TensorView>      scale_a,
         std::optional<ffi::TensorView>      scale_b,
         const std::tuple<int64_t, int64_t>& recipe_a,
         const std::tuple<int64_t, int64_t>& recipe_b,
         bool                                trans_a,
         bool                                trans_b,
         bool                                fixed_scale_layout) {
  const BMMArgs args{a, b, d, c, scale_a, scale_b, recipe_a, recipe_b, trans_a, trans_b, fixed_scale_layout};
  if (is_bf16_or_fp16_dtype(args.a.dtype()) && is_bf16_or_fp16_dtype(args.b.dtype())) {
    bmm_16bit(args);
    return;
  }
  if (is_fp8_dtype(args.a.dtype()) && is_fp8_dtype(args.b.dtype())) {
    bmm_8bit(args);
    return;
  }
  TVM_FFI_THROW(ValueError) << "bmm got unsupported input dtypes";
}

}  // namespace mate::gemm::mudnn

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm, mate::gemm::mudnn::bmm);
