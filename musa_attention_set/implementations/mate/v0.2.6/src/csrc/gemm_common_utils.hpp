#pragma once

#include "mudnn_utils.hpp"
#include "op_utils.hpp"

namespace mate::gemm::common {

inline void fill_zero_tensor(ffi::TensorView tensor) {
  if (tensor.numel() == 0) {
    return;
  }
  CHECK_MUSA(tensor);
  ffi::MUSADeviceGuard device_guard(tensor.device().device_id);
  fill_mudnn_tensor(tensor.device(),
                    get_stream(tensor.device()),
                    tensor.data_ptr(),
                    tensor.dtype(),
                    tensor.shape(),
                    tensor.strides(),
                    0.0);
}

inline bool gemm_early_return(int batch, int m, int n, int k, ffi::TensorView out, bool has_c) {
  if (batch == 0 || m == 0 || n == 0) {
    return true;
  }
  if (k == 0 && !has_c) {
    fill_zero_tensor(out);
    return true;
  }
  return false;
}

}  // namespace mate::gemm::common
