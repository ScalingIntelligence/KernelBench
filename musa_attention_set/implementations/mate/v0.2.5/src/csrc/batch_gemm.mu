#include <mudnn_xmma.h>
#include <musa.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <optional>
#include <string>
#include <tuple>

#include "gemm_mudnn_utils.hpp"

struct BatchMatMulScaledParam {
  static constexpr int DIM_ABD = 3;

  TensorMajor major_a;
  TensorMajor major_b;
  TensorMajor major_scale_a;
  TensorMajor major_scale_b;

  MatMulScalingMode scale_mode;

  DLDataType type_a;
  DLDataType type_b;
  DLDataType type_c;
  DLDataType type_d;

  int m;
  int n;
  int k;
  int nr_batch;

  int scale_a_m;
  int scale_a_k;
  int scale_b_n;
  int scale_b_k;

  int scale_granularity_m;
  int scale_granularity_n;
  int scale_granularity_k;
  int quant_tile;

  int64_t stride_a[DIM_ABD];
  int64_t stride_b[DIM_ABD];
  int64_t stride_c[DIM_ABD];
  int64_t stride_d[DIM_ABD];

  int64_t stride_scale_a[DIM_ABD];
  int64_t stride_scale_b[DIM_ABD];

  void* p_a;
  void* p_b;
  void* p_c;
  void* p_scale_a;
  void* p_scale_b;
  void* p_d;

  bool has_c;
};

namespace {

namespace gemm_common = mate::gemm::common;
namespace gemm_mudnn  = mate::gemm::mudnn;

MatMulScalingMode get_scaling_mode_bmm_fp8(const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk) {
  const int scale_granularity_m = static_cast<int>(std::get<0>(scale_granularity_mnk));
  const int scale_granularity_n = static_cast<int>(std::get<1>(scale_granularity_mnk));
  const int scale_granularity_k = static_cast<int>(std::get<2>(scale_granularity_mnk));

  if (scale_granularity_k == -1) {
    if (scale_granularity_m == -1 && scale_granularity_n == -1) {
      return MatMulScalingMode::TENSOR_TENSOR;
    }
    if (scale_granularity_m == 1 && scale_granularity_n == -1) {
      return MatMulScalingMode::CHANNEL_TENSOR;
    }
  }

  if (scale_granularity_k == 128 && scale_granularity_m == 1 &&
      (scale_granularity_n == 128 || scale_granularity_n == 1)) {
    return MatMulScalingMode::GROUP_BLOCK;
  }

  TVM_FFI_THROW(ValueError) << "bmm_fp8 got unsupported scale_granularity_mnk";
}

void run_bmm_mudnn_lt(const BatchMatMulScaledParam&   param,
                      musa::dnn::Handle&              handle,
                      const musa::dnn::MatMulLtParam& lt_param) {
  const int64_t a_dim0    = param.major_a == TensorMajor::K ? param.m : param.k;
  const int64_t a_dim1    = param.major_a == TensorMajor::K ? param.k : param.m;
  const int64_t a_stride0 = param.major_a == TensorMajor::K ? param.stride_a[1] : param.stride_a[2];
  const int64_t a_stride1 = param.major_a == TensorMajor::K ? param.stride_a[2] : param.stride_a[1];
  const int64_t b_dim0    = param.major_b == TensorMajor::K ? param.n : param.k;
  const int64_t b_dim1    = param.major_b == TensorMajor::K ? param.k : param.n;
  const int64_t b_stride0 = param.major_b == TensorMajor::K ? param.stride_b[1] : param.stride_b[2];
  const int64_t b_stride1 = param.major_b == TensorMajor::K ? param.stride_b[2] : param.stride_b[1];
  auto          a         = make_mudnn_tensor(
      param.p_a, param.type_a, {param.nr_batch, a_dim0, a_dim1}, {param.stride_a[0], a_stride0, a_stride1});
  auto b = make_mudnn_tensor(
      param.p_b, param.type_b, {param.nr_batch, b_dim0, b_dim1}, {param.stride_b[0], b_stride0, b_stride1});
  auto              d = make_mudnn_tensor(param.p_d,
                             param.type_d,
                                          {param.nr_batch, param.m, param.n},
                                          {param.stride_d[0], param.stride_d[1], param.stride_d[2]});
  musa::dnn::Tensor c;
  if (param.has_c) {
    c = make_mudnn_tensor(param.p_c,
                          param.type_c,
                          {param.nr_batch, param.m, param.n},
                          {param.stride_c[0], param.stride_c[1], param.stride_c[2]});
  }
  gemm_mudnn::run_mudnn_lt_matmul(handle, d, a, b, c, param.has_c, param.major_a, param.major_b, lt_param);
}

void bmm_fp8_run(const BatchMatMulScaledParam& param, musa::dnn::Handle& handle) {
  musa::dnn::Tensor scale_a;
  if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    scale_a = make_mudnn_tensor(param.p_scale_a,
                                dl_float32,
                                {param.nr_batch, param.m, 1},
                                {param.stride_scale_a[0], param.stride_scale_a[1], param.stride_scale_a[2]});
  } else {
    scale_a = make_mudnn_scalar_tensor(param.p_scale_a, dl_float32);
  }

  musa::dnn::MatMulLtParam lt_param;
  if (param.scale_mode == MatMulScalingMode::GROUP_BLOCK) {
    const int64_t scale_a_dim0 = param.major_scale_a == TensorMajor::K ? param.scale_a_m : param.scale_a_k;
    const int64_t scale_a_dim1 = param.major_scale_a == TensorMajor::K ? param.scale_a_k : param.scale_a_m;
    const int64_t scale_a_stride0 =
        param.major_scale_a == TensorMajor::K ? param.stride_scale_a[1] : param.stride_scale_a[2];
    const int64_t scale_a_stride1 =
        param.major_scale_a == TensorMajor::K ? param.stride_scale_a[2] : param.stride_scale_a[1];
    scale_a = make_mudnn_tensor(param.p_scale_a,
                                dl_float32,
                                {param.nr_batch, scale_a_dim0, scale_a_dim1},
                                {param.stride_scale_a[0], scale_a_stride0, scale_a_stride1});

    const int64_t scale_b_dim0 = param.major_scale_b == TensorMajor::K ? param.scale_b_n : param.scale_b_k;
    const int64_t scale_b_dim1 = param.major_scale_b == TensorMajor::K ? param.scale_b_k : param.scale_b_n;
    const int64_t scale_b_stride0 =
        param.major_scale_b == TensorMajor::K ? param.stride_scale_b[1] : param.stride_scale_b[2];
    const int64_t scale_b_stride1 =
        param.major_scale_b == TensorMajor::K ? param.stride_scale_b[2] : param.stride_scale_b[1];
    auto       scale_b            = make_mudnn_tensor(param.p_scale_b,
                                     dl_float32,
                                                      {param.nr_batch, scale_b_dim0, scale_b_dim1},
                                                      {param.stride_scale_b[0], scale_b_stride0, scale_b_stride1});
    const bool fixed_scale_layout = param.major_scale_a == TensorMajor::MN;
    MATE_MUDNN_STATUS_CHECK(lt_param.SetScale(
        scale_a, scale_b, musa::dnn::Tensor{}, musa::dnn::Tensor{}, param.quant_tile, fixed_scale_layout));
  } else {
    musa::dnn::Tensor scale_b = make_mudnn_scalar_tensor(param.p_scale_b, dl_float32);
    MATE_MUDNN_STATUS_CHECK(lt_param.SetScale(scale_a, scale_b, musa::dnn::Tensor{}, musa::dnn::Tensor{}));
  }
  run_bmm_mudnn_lt(param, handle, lt_param);
}

void bmm_fp16_run(const BatchMatMulScaledParam& param, musa::dnn::Handle& handle) {
  musa::dnn::MatMulLtParam lt_param;
  run_bmm_mudnn_lt(param, handle, lt_param);
}

void dispatch_bmm_backend(
    const BatchMatMulScaledParam& param, const std::string& backend, int device_id, musaStream_t stream, bool is_fp8) {
  gemm_mudnn::validate_mudnn_backend(backend, is_fp8 ? "bmm_fp8" : "bmm_fp16");
  musa::dnn::Handle handle(device_id);
  gemm_mudnn::init_mudnn_handle(handle, stream);
  if (is_fp8) {
    bmm_fp8_run(param, handle);
  } else {
    bmm_fp16_run(param, handle);
  }
}

TensorMajor parse_a_major(const std::string& major_a_mode) {
  if (major_a_mode == "K") {
    return TensorMajor::K;
  }
  if (major_a_mode == "M") {
    return TensorMajor::MN;
  }
  TVM_FFI_THROW(ValueError) << "major_a_mode must be 'K' or 'M'";
}

TensorMajor parse_b_major(const std::string& major_b_mode) {
  if (major_b_mode == "K") {
    return TensorMajor::K;
  }
  if (major_b_mode == "N") {
    return TensorMajor::MN;
  }
  TVM_FFI_THROW(ValueError) << "major_b_mode must be 'N' or 'K'";
}

TensorMajor infer_group_scale_major(ffi::TensorView scale,
                                    int             batch,
                                    int             rows,
                                    int             k_blocks,
                                    const char*     tensor_name,
                                    int64_t&        row_stride,
                                    int64_t&        k_stride) {
  CHECK_DIM(3, scale);
  TVM_FFI_ICHECK_EQ(scale.size(0), batch) << tensor_name << " batch dimension mismatch";
  if (scale.size(1) == rows && scale.size(2) == k_blocks) {
    row_stride = scale.stride(1);
    k_stride   = scale.stride(2);
    return TensorMajor::K;
  }
  if (scale.size(1) == k_blocks && scale.size(2) == rows) {
    row_stride = scale.stride(2);
    k_stride   = scale.stride(1);
    return TensorMajor::MN;
  }
  TVM_FFI_THROW(ValueError) << tensor_name << " has invalid group scale shape";
}

}  // namespace

void bmm_fp8(ffi::TensorView                              a,
             ffi::TensorView                              b,
             ffi::TensorView                              scale_a,
             ffi::TensorView                              scale_b,
             ffi::TensorView                              d,
             const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk,
             const std::string&                           backend,
             std::optional<ffi::TensorView>               c,
             const std::string&                           major_a_mode,
             const std::string&                           major_b_mode) {
  check_mp31(a.device(), "bmm_fp8");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(scale_a);
  CHECK_MUSA(scale_b);
  CHECK_MUSA(d);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, scale_a);
  CHECK_DEVICE(a, scale_b);
  CHECK_DEVICE(a, d);
  CHECK_DIM(3, a);
  CHECK_DIM(3, b);
  CHECK_DIM(3, d);
  if (c.has_value()) {
    CHECK_MUSA(c.value());
    CHECK_DEVICE(a, c.value());
    CHECK_DIM(3, c.value());
    TVM_FFI_ICHECK_EQ(c.value().stride(-1), 1) << "c must be contiguous at the last dimension";
  }
  TVM_FFI_ICHECK_EQ(d.stride(-1), 1) << "d must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(scale_a.dtype(), dl_float32) << "scale_a must be float32";
  TVM_FFI_ICHECK_EQ(scale_b.dtype(), dl_float32) << "scale_b must be float32";

  const auto scale_mode = get_scaling_mode_bmm_fp8(scale_granularity_mnk);

  BatchMatMulScaledParam param{};
  param.major_a    = parse_a_major(major_a_mode);
  param.major_b    = parse_b_major(major_b_mode);
  param.scale_mode = scale_mode;
  param.type_a     = a.dtype();
  param.type_b     = b.dtype();
  param.type_d     = d.dtype();
  param.has_c      = c.has_value();
  if (param.has_c) {
    param.type_c = c.value().dtype();
  }

  TVM_FFI_ICHECK(is_fp8_dtype(param.type_a)) << "a must be fp8";
  TVM_FFI_ICHECK(is_fp8_dtype(param.type_b)) << "b must be fp8";
  TVM_FFI_ICHECK(is_bf16_or_fp16_or_fp32_dtype(param.type_d)) << "d must be bf16, fp16 or fp32";
  if (param.has_c) {
    TVM_FFI_ICHECK(dtype_equal(param.type_c, param.type_d)) << "c must have the same dtype as d";
    TVM_FFI_ICHECK_EQ(param.type_d.code, kDLFloat) << "bmm_fp8 with c only supports fp32 d";
    TVM_FFI_ICHECK_EQ(param.type_d.bits, 32) << "bmm_fp8 with c only supports fp32 d";
  }

  param.nr_batch = static_cast<int>(a.size(0));
  param.m        = static_cast<int>(param.major_a == TensorMajor::K ? a.size(1) : a.size(2));
  param.k        = static_cast<int>(param.major_a == TensorMajor::K ? a.size(2) : a.size(1));
  param.n        = static_cast<int>(param.major_b == TensorMajor::K ? b.size(1) : b.size(2));
  const int b_k  = static_cast<int>(param.major_b == TensorMajor::K ? b.size(2) : b.size(1));

  TVM_FFI_ICHECK_EQ(b.size(0), param.nr_batch);
  TVM_FFI_ICHECK_EQ(b_k, param.k);
  TVM_FFI_ICHECK_EQ(d.size(0), param.nr_batch);
  TVM_FFI_ICHECK_EQ(d.size(1), param.m);
  TVM_FFI_ICHECK_EQ(d.size(2), param.n);
  if (param.has_c) {
    TVM_FFI_ICHECK_EQ(c.value().size(0), param.nr_batch);
    TVM_FFI_ICHECK_EQ(c.value().size(1), param.m);
    TVM_FFI_ICHECK_EQ(c.value().size(2), param.n);
  }

  if (gemm_common::gemm_early_return(param.m, param.n, param.k, d)) {
    return;
  }

  param.scale_granularity_m = static_cast<int>(std::get<0>(scale_granularity_mnk));
  param.scale_granularity_n = static_cast<int>(std::get<1>(scale_granularity_mnk));
  param.scale_granularity_k = static_cast<int>(std::get<2>(scale_granularity_mnk));
  param.quant_tile          = param.scale_mode == MatMulScalingMode::GROUP_BLOCK ? param.scale_granularity_k : 0;

  param.stride_a[0] = a.stride(0);
  param.stride_a[1] = param.major_a == TensorMajor::K ? a.stride(1) : a.stride(2);
  param.stride_a[2] = param.major_a == TensorMajor::K ? a.stride(2) : a.stride(1);
  param.stride_b[0] = b.stride(0);
  param.stride_b[1] = param.major_b == TensorMajor::K ? b.stride(1) : b.stride(2);
  param.stride_b[2] = param.major_b == TensorMajor::K ? b.stride(2) : b.stride(1);
  param.stride_d[0] = d.stride(0);
  param.stride_d[1] = d.stride(1);
  param.stride_d[2] = d.stride(2);
  if (param.has_c) {
    param.stride_c[0] = c.value().stride(0);
    param.stride_c[1] = c.value().stride(1);
    param.stride_c[2] = c.value().stride(2);
  }
  TVM_FFI_ICHECK_EQ(param.major_a == TensorMajor::K ? param.stride_a[2] : param.stride_a[1], 1)
      << "a must be contiguous in the declared major dimension";
  TVM_FFI_ICHECK_EQ(param.major_b == TensorMajor::K ? param.stride_b[2] : param.stride_b[1], 1)
      << "b must be contiguous in the declared major dimension";

  if (scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    CHECK_DIM(3, scale_a);
    TVM_FFI_ICHECK_EQ(scale_a.size(0), param.nr_batch);
    TVM_FFI_ICHECK_EQ(scale_a.size(1), param.m);
    TVM_FFI_ICHECK_EQ(scale_a.size(2), 1);
    TVM_FFI_ICHECK_EQ(scale_b.ndim(), 0) << "scale_b must be a scalar tensor";
  } else if (scale_mode == MatMulScalingMode::GROUP_BLOCK) {
    const int scale_a_m = mutlass::ceil_div(param.m, param.scale_granularity_m);
    const int scale_a_k = mutlass::ceil_div(param.k, param.scale_granularity_k);
    const int scale_b_n = mutlass::ceil_div(param.n, param.scale_granularity_n);
    const int scale_b_k = mutlass::ceil_div(param.k, param.scale_granularity_k);

    param.scale_a_m = scale_a_m;
    param.scale_a_k = scale_a_k;
    param.scale_b_n = scale_b_n;
    param.scale_b_k = scale_b_k;

    const TensorMajor physical_scale_a       = infer_group_scale_major(scale_a,
                                                                 param.nr_batch,
                                                                 param.scale_a_m,
                                                                 param.scale_a_k,
                                                                 "scale_a",
                                                                 param.stride_scale_a[1],
                                                                 param.stride_scale_a[2]);
    const TensorMajor physical_scale_b       = infer_group_scale_major(scale_b,
                                                                 param.nr_batch,
                                                                 param.scale_b_n,
                                                                 param.scale_b_k,
                                                                 "scale_b",
                                                                 param.stride_scale_b[1],
                                                                 param.stride_scale_b[2]);
    const bool        use_fixed_scale_layout = param.major_a == TensorMajor::MN || param.major_b == TensorMajor::MN ||
                                        physical_scale_a == TensorMajor::MN || physical_scale_b == TensorMajor::MN;
    param.major_scale_a = use_fixed_scale_layout ? TensorMajor::MN : TensorMajor::K;
    param.major_scale_b = param.major_scale_a;
  } else {
    TVM_FFI_ICHECK_EQ(scale_a.ndim(), 0) << "scale_a must be a scalar tensor";
    TVM_FFI_ICHECK_EQ(scale_b.ndim(), 0) << "scale_b must be a scalar tensor";
  }

  if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    param.stride_scale_a[0] = scale_a.stride(0);
    param.stride_scale_a[1] = scale_a.stride(1);
    param.stride_scale_a[2] = scale_a.stride(2);
  } else if (param.scale_mode == MatMulScalingMode::GROUP_BLOCK) {
    param.stride_scale_a[0] = scale_a.stride(0);
    param.stride_scale_b[0] = scale_b.stride(0);
  }

  param.p_a       = a.data_ptr();
  param.p_b       = b.data_ptr();
  param.p_c       = param.has_c ? c.value().data_ptr() : nullptr;
  param.p_scale_a = scale_a.data_ptr();
  param.p_scale_b = scale_b.data_ptr();
  param.p_d       = d.data_ptr();

  ffi::MUSADeviceGuard device_guard(a.device().device_id);
  dispatch_bmm_backend(param, backend, a.device().device_id, get_stream(a.device()), true);
}

void bmm_fp16(ffi::TensorView                a,
              ffi::TensorView                b,
              ffi::TensorView                d,
              std::optional<ffi::TensorView> c,
              const std::string&             backend) {
  check_mp31(a.device(), "bmm_fp16");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(d);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, d);
  CHECK_DIM(3, a);
  CHECK_DIM(3, b);
  CHECK_DIM(3, d);
  if (c.has_value()) {
    CHECK_MUSA(c.value());
    CHECK_DEVICE(a, c.value());
    CHECK_DIM(3, c.value());
    TVM_FFI_ICHECK_EQ(c.value().stride(-1), 1) << "c must be contiguous at the last dimension";
  }
  TVM_FFI_ICHECK_EQ(a.stride(2), 1) << "a must be contiguous at k dimension";
  TVM_FFI_ICHECK_EQ(b.stride(1), 1) << "b must be contiguous at k dimension";
  TVM_FFI_ICHECK_EQ(d.stride(-1), 1) << "d must be contiguous at the last dimension";

  BatchMatMulScaledParam param{};
  param.major_a = TensorMajor::K;
  param.major_b = TensorMajor::K;
  param.type_a  = a.dtype();
  param.type_b  = b.dtype();
  param.type_d  = d.dtype();
  param.has_c   = c.has_value();
  if (param.has_c) {
    param.type_c = c.value().dtype();
  }

  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(param.type_a)) << "a must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(param.type_b)) << "b must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_or_fp32_dtype(param.type_d)) << "d must be bf16, fp16 or fp32";
  if (param.has_c) {
    TVM_FFI_ICHECK(dtype_equal(param.type_c, param.type_d)) << "c must have the same dtype as d";
  }

  param.nr_batch = static_cast<int>(a.size(0));
  param.m        = static_cast<int>(a.size(1));
  param.n        = static_cast<int>(b.size(2));
  param.k        = static_cast<int>(a.size(2));

  TVM_FFI_ICHECK_EQ(b.size(0), param.nr_batch);
  TVM_FFI_ICHECK_EQ(b.size(1), param.k);
  TVM_FFI_ICHECK_EQ(d.size(0), param.nr_batch);
  TVM_FFI_ICHECK_EQ(d.size(1), param.m);
  TVM_FFI_ICHECK_EQ(d.size(2), param.n);
  if (param.has_c) {
    TVM_FFI_ICHECK_EQ(c.value().size(0), param.nr_batch);
    TVM_FFI_ICHECK_EQ(c.value().size(1), param.m);
    TVM_FFI_ICHECK_EQ(c.value().size(2), param.n);
  }

  if (gemm_common::gemm_early_return(param.m, param.n, param.k, d)) {
    return;
  }

  param.stride_a[0] = a.stride(0);
  param.stride_a[1] = a.stride(1);
  param.stride_a[2] = a.stride(2);
  param.stride_b[0] = b.stride(0);
  param.stride_b[1] = b.stride(2);
  param.stride_b[2] = b.stride(1);
  if (param.has_c) {
    param.stride_c[0] = c.value().stride(0);
    param.stride_c[1] = c.value().stride(1);
    param.stride_c[2] = c.value().stride(2);
  }
  param.stride_d[0] = d.stride(0);
  param.stride_d[1] = d.stride(1);
  param.stride_d[2] = d.stride(2);

  param.p_a = a.data_ptr();
  param.p_b = b.data_ptr();
  param.p_c = param.has_c ? c.value().data_ptr() : nullptr;
  param.p_d = d.data_ptr();

  ffi::MUSADeviceGuard device_guard(a.device().device_id);
  dispatch_bmm_backend(param, backend, a.device().device_id, get_stream(a.device()), false);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_fp8, bmm_fp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_fp16, bmm_fp16);
