#include <mudnn_xmma.h>
#include <musa.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <string>
#include <tuple>

#include "gemm_mudnn_utils.hpp"

struct MatMulFP8ScaledParam {
  static constexpr int DIM_ABD = 2;

  TensorMajor major_a;
  TensorMajor major_b;
  TensorMajor major_scale_a;
  TensorMajor major_scale_b;

  MatMulScalingMode scale_mode;

  DLDataType type_a;
  DLDataType type_b;
  DLDataType type_d;

  int m;
  int n;
  int k;

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
  int64_t stride_d[DIM_ABD];
  int64_t stride_scale_a[DIM_ABD];
  int64_t stride_scale_b[DIM_ABD];

  void* p_a;
  void* p_b;
  void* p_scale_a;
  void* p_scale_b;
  void* p_d;
};

namespace {

namespace gemm_common = mate::gemm::common;
namespace gemm_mudnn  = mate::gemm::mudnn;

MatMulScalingMode get_scaling_mode_gemm_fp8(const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk) {
  const int scale_granularity_m = static_cast<int>(std::get<0>(scale_granularity_mnk));
  const int scale_granularity_n = static_cast<int>(std::get<1>(scale_granularity_mnk));
  const int scale_granularity_k = static_cast<int>(std::get<2>(scale_granularity_mnk));

  if (scale_granularity_k == -1) {
    if (scale_granularity_m == 1 && scale_granularity_n == -1) {
      return MatMulScalingMode::CHANNEL_TENSOR;
    }
    if (scale_granularity_m == 1 && scale_granularity_n == 1) {
      return MatMulScalingMode::CHANNEL_CHANNEL;
    }
  }

  if (scale_granularity_k == 128 && scale_granularity_m == 1 && scale_granularity_n == 128) {
    return MatMulScalingMode::GROUP_BLOCK;
  }

  TVM_FFI_THROW(ValueError) << "gemm_fp8_nt_groupwise got unsupported scale_granularity_mnk";
}

void gemm_fp8_mudnn_run(const MatMulFP8ScaledParam& param, musa::dnn::Handle& handle) {
  const int64_t a_dim0 = param.major_a == TensorMajor::K ? param.m : param.k;
  const int64_t a_dim1 = param.major_a == TensorMajor::K ? param.k : param.m;
  const int64_t b_dim0 = param.major_b == TensorMajor::K ? param.n : param.k;
  const int64_t b_dim1 = param.major_b == TensorMajor::K ? param.k : param.n;
  auto a = make_mudnn_tensor(param.p_a, param.type_a, {a_dim0, a_dim1}, {param.stride_a[0], param.stride_a[1]});
  auto b = make_mudnn_tensor(param.p_b, param.type_b, {b_dim0, b_dim1}, {param.stride_b[0], param.stride_b[1]});
  auto d = make_mudnn_tensor(param.p_d, param.type_d, {param.m, param.n}, {param.stride_d[0], param.stride_d[1]});

  const int64_t scale_a_dim0 = param.major_scale_a == TensorMajor::K ? param.scale_a_m : param.scale_a_k;
  const int64_t scale_a_dim1 = param.major_scale_a == TensorMajor::K ? param.scale_a_k : param.scale_a_m;
  auto          scale_a      = make_mudnn_tensor(
      param.p_scale_a, dl_float32, {scale_a_dim0, scale_a_dim1}, {param.stride_scale_a[0], param.stride_scale_a[1]});

  musa::dnn::Tensor scale_b;
  if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    scale_b = make_mudnn_scalar_tensor(param.p_scale_b, dl_float32);
  } else {
    const int64_t scale_b_dim0 = param.major_scale_b == TensorMajor::K ? param.scale_b_n : param.scale_b_k;
    const int64_t scale_b_dim1 = param.major_scale_b == TensorMajor::K ? param.scale_b_k : param.scale_b_n;
    scale_b                    = make_mudnn_tensor(
        param.p_scale_b, dl_float32, {scale_b_dim0, scale_b_dim1}, {param.stride_scale_b[0], param.stride_scale_b[1]});
  }

  musa::dnn::MatMulLtParam lt_param;
  const bool               is_scale_mn_major = param.major_scale_a == TensorMajor::MN;
  MATE_MUDNN_STATUS_CHECK(lt_param.SetScale(
      scale_a, scale_b, musa::dnn::Tensor{}, musa::dnn::Tensor{}, param.quant_tile, is_scale_mn_major));
  gemm_mudnn::run_mudnn_lt_matmul(handle, d, a, b, param.major_a, param.major_b, lt_param);
}

void dispatch_backend(const MatMulFP8ScaledParam& param,
                      const std::string&          backend,
                      int                         device_id,
                      musaStream_t                stream) {
  gemm_mudnn::validate_mudnn_backend(backend, "gemm_fp8_nt_groupwise");
  musa::dnn::Handle handle(device_id);
  gemm_mudnn::init_mudnn_handle(handle, stream);
  gemm_fp8_mudnn_run(param, handle);
}

}  // namespace

void gemm_fp8_nt_groupwise(ffi::TensorView                              a,
                           ffi::TensorView                              b,
                           ffi::TensorView                              scale_a,
                           ffi::TensorView                              scale_b,
                           const std::string&                           scale_major_mode,
                           int64_t                                      mma_sm,
                           const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk,
                           ffi::TensorView                              d,
                           const std::string&                           backend) {
  check_mp31(a.device(), "gemm_fp8_nt_groupwise");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(scale_a);
  CHECK_MUSA(scale_b);
  CHECK_MUSA(d);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, scale_a);
  CHECK_DEVICE(a, scale_b);
  CHECK_DEVICE(a, d);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(2, d);
  CHECK_CONTIGUOUS(a);
  CHECK_CONTIGUOUS(b);
  TVM_FFI_ICHECK_EQ(d.stride(-1), 1) << "d must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(scale_a.dtype(), dl_float32) << "scale_a must be float32";
  TVM_FFI_ICHECK_EQ(scale_b.dtype(), dl_float32) << "scale_b must be float32";

  if (mma_sm != 1) {
    TVM_FFI_THROW(ValueError) << "gemm_fp8_nt_groupwise only supports mma_sm=1";
  }

  MatMulFP8ScaledParam param{};
  param.major_a = TensorMajor::K;
  param.major_b = TensorMajor::K;
  param.type_a  = a.dtype();
  param.type_b  = b.dtype();
  param.type_d  = d.dtype();

  TVM_FFI_ICHECK(is_fp8_dtype(param.type_a)) << "a must be fp8";
  TVM_FFI_ICHECK(is_fp8_dtype(param.type_b)) << "b must be fp8";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(param.type_d)) << "d must be bf16 or fp16";

  param.m = static_cast<int>(a.size(0));
  param.n = static_cast<int>(b.size(0));
  param.k = static_cast<int>(a.size(1));

  if (gemm_common::gemm_early_return(param.m, param.n, param.k, d)) {
    return;
  }

  param.scale_granularity_m = static_cast<int>(std::get<0>(scale_granularity_mnk));
  param.scale_granularity_n = static_cast<int>(std::get<1>(scale_granularity_mnk));
  param.scale_granularity_k = static_cast<int>(std::get<2>(scale_granularity_mnk));
  param.scale_mode          = get_scaling_mode_gemm_fp8(scale_granularity_mnk);
  param.quant_tile =
      (param.scale_mode == MatMulScalingMode::CHANNEL_CHANNEL || param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR)
          ? 0
          : param.scale_granularity_k;

  CHECK_CONTIGUOUS(scale_a);
  if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    TVM_FFI_ICHECK_EQ(scale_b.ndim(), 0) << "scale_b must be a scalar tensor";
  } else {
    CHECK_CONTIGUOUS(scale_b);
  }

  TVM_FFI_ICHECK_EQ(b.size(1), param.k);
  TVM_FFI_ICHECK_EQ(d.size(0), param.m);
  TVM_FFI_ICHECK_EQ(d.size(1), param.n);

  param.stride_a[0]       = a.stride(0);
  param.stride_a[1]       = a.stride(1);
  param.stride_b[0]       = b.stride(0);
  param.stride_b[1]       = b.stride(1);
  param.stride_d[0]       = d.stride(0);
  param.stride_d[1]       = d.stride(1);
  param.stride_scale_a[0] = scale_a.stride(0);
  param.stride_scale_a[1] = scale_a.stride(1);

  if (param.scale_mode != MatMulScalingMode::CHANNEL_TENSOR) {
    param.stride_scale_b[0] = scale_b.stride(0);
    param.stride_scale_b[1] = scale_b.stride(1);
  }

  if (scale_major_mode == "K") {
    param.major_scale_a = TensorMajor::K;
    param.major_scale_b = TensorMajor::K;

    if (param.scale_mode == MatMulScalingMode::CHANNEL_CHANNEL) {
      TVM_FFI_ICHECK_EQ(scale_a.size(0), param.m);
      TVM_FFI_ICHECK_EQ(scale_a.size(1), 1);
      TVM_FFI_ICHECK_EQ(scale_b.size(0), param.n);
      TVM_FFI_ICHECK_EQ(scale_b.size(1), 1);
    } else if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
      TVM_FFI_ICHECK_EQ(scale_a.size(0), param.m);
      TVM_FFI_ICHECK_EQ(scale_a.size(1), 1);
    } else {
      TVM_FFI_ICHECK_EQ(scale_a.size(0), param.m);
      TVM_FFI_ICHECK_EQ(scale_a.size(1), mutlass::ceil_div(param.k, param.scale_granularity_k));
      TVM_FFI_ICHECK_EQ(scale_b.size(0), mutlass::ceil_div(param.n, param.scale_granularity_n));
      TVM_FFI_ICHECK_EQ(scale_b.size(1), mutlass::ceil_div(param.k, param.scale_granularity_k));
    }

    param.scale_a_m = static_cast<int>(scale_a.size(0));
    param.scale_a_k = static_cast<int>(scale_a.size(1));
    if (param.scale_mode != MatMulScalingMode::CHANNEL_TENSOR) {
      param.scale_b_n = static_cast<int>(scale_b.size(0));
      param.scale_b_k = static_cast<int>(scale_b.size(1));
    } else {
      param.scale_b_n = 0;
      param.scale_b_k = 0;
    }
  } else if (scale_major_mode == "MN") {
    param.major_scale_a = TensorMajor::MN;
    param.major_scale_b = TensorMajor::MN;

    if (param.scale_mode == MatMulScalingMode::CHANNEL_CHANNEL) {
      TVM_FFI_ICHECK_EQ(scale_a.size(0), 1);
      TVM_FFI_ICHECK_EQ(scale_a.size(1), param.m);
      TVM_FFI_ICHECK_EQ(scale_b.size(0), 1);
      TVM_FFI_ICHECK_EQ(scale_b.size(1), param.n);
    } else if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
      TVM_FFI_ICHECK_EQ(scale_a.size(0), 1);
      TVM_FFI_ICHECK_EQ(scale_a.size(1), param.m);
    } else {
      TVM_FFI_ICHECK_EQ(scale_a.size(0), mutlass::ceil_div(param.k, param.scale_granularity_k));
      TVM_FFI_ICHECK_EQ(scale_a.size(1), param.m);
      TVM_FFI_ICHECK_EQ(scale_b.size(0), mutlass::ceil_div(param.k, param.scale_granularity_k));
      TVM_FFI_ICHECK_EQ(scale_b.size(1), mutlass::ceil_div(param.n, param.scale_granularity_n));
    }

    param.scale_a_m = static_cast<int>(scale_a.size(1));
    param.scale_a_k = static_cast<int>(scale_a.size(0));
    if (param.scale_mode != MatMulScalingMode::CHANNEL_TENSOR) {
      param.scale_b_n = static_cast<int>(scale_b.size(1));
      param.scale_b_k = static_cast<int>(scale_b.size(0));
    } else {
      param.scale_b_n = 0;
      param.scale_b_k = 0;
    }
  } else {
    TVM_FFI_THROW(ValueError) << "gemm_fp8_nt_groupwise got unsupported scale_major_mode: " << scale_major_mode;
  }

  param.p_a       = a.data_ptr();
  param.p_b       = b.data_ptr();
  param.p_scale_a = scale_a.data_ptr();
  param.p_scale_b = scale_b.data_ptr();
  param.p_d       = d.data_ptr();

  ffi::MUSADeviceGuard device_guard(a.device().device_id);
  dispatch_backend(param, backend, a.device().device_id, get_stream(a.device()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_fp8_nt_groupwise, gemm_fp8_nt_groupwise);
