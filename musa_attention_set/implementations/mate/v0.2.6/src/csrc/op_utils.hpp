#pragma once

// Canonical include for FFI operator entrypoints.
#include "mate_utils.hpp"
#include "tvm_ffi_utils.hpp"

inline void check_mp31(DLDevice device, const char* func_name) {
  musaDeviceProp dprops{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, device.device_id));
  TVM_FFI_ICHECK(dprops.major > 3 || (dprops.major == 3 && dprops.minor >= 1))
      << func_name << " requires MP31 or above";
}
