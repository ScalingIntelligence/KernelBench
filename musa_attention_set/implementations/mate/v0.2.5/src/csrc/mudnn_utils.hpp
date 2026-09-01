#pragma once

#include <mudnn.h>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include "dtype_utils.hpp"
#include "mate_utils.hpp"
#include "tvm/ffi/container/tensor.h"
#include "tvm/ffi/error.h"
#include "tvm/ffi/extra/c_env_api.h"
#include "tvm/ffi/extra/musa/device_guard.h"

inline const char* mudnn_status_to_string(musa::dnn::Status status) {
  switch (status) {
    case musa::dnn::Status::SUCCESS:
      return "SUCCESS";
    case musa::dnn::Status::INVALID_PARAMETER:
      return "INVALID_PARAMETER";
    case musa::dnn::Status::NOT_INITIALIZED:
      return "NOT_INITIALIZED";
    case musa::dnn::Status::ALLOC_FAILED:
      return "ALLOC_FAILED";
    case musa::dnn::Status::NOT_SUPPORTED:
      return "NOT_SUPPORTED";
    case musa::dnn::Status::INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case musa::dnn::Status::ARCH_MISMATCH:
      return "ARCH_MISMATCH";
    case musa::dnn::Status::EXECUTION_FAILED:
      return "EXECUTION_FAILED";
    default:
      return "UNKNOWN";
  }
}

#ifndef MATE_MUDNN_STATUS_CHECK
#define MATE_MUDNN_STATUS_CHECK(error)                                                        \
  do {                                                                                        \
    musa::dnn::Status __err = (error);                                                        \
    if (__err != musa::dnn::Status::SUCCESS) {                                                \
      TVM_FFI_THROW(RuntimeError) << "muDNN Error: " << mudnn_status_to_string(__err) << " (" \
                                  << static_cast<int>(__err) << ")";                          \
    }                                                                                         \
  } while (0)
#endif

inline musa::dnn::Tensor::Type dl_dtype_to_mudnn_type(DLDataType dtype) {
  if (dtype.lanes != 1) {
    throw std::runtime_error("dl_dtype_to_mudnn_type only supports lane=1 dtypes");
  }

  switch (dtype.code) {
    case kDLFloat:
      if (dtype.bits == 16) return musa::dnn::Tensor::Type::HALF;
      if (dtype.bits == 32) return musa::dnn::Tensor::Type::FLOAT;
      if (dtype.bits == 64) return musa::dnn::Tensor::Type::DOUBLE;
      break;
    case kDLBfloat:
      if (dtype.bits == 16) return musa::dnn::Tensor::Type::BFLOAT16;
      break;
    case kDLFloat8_e4m3fn:
      if (dtype.bits == 8) return musa::dnn::Tensor::Type::FP8_E4M3;
      break;
    case kDLFloat8_e5m2:
      if (dtype.bits == 8) return musa::dnn::Tensor::Type::FP8_E5M2;
      break;
    case kDLInt:
      if (dtype.bits == 8) return musa::dnn::Tensor::Type::INT8;
      if (dtype.bits == 16) return musa::dnn::Tensor::Type::INT16;
      if (dtype.bits == 32) return musa::dnn::Tensor::Type::INT32;
      if (dtype.bits == 64) return musa::dnn::Tensor::Type::INT64;
      break;
    case kDLUInt:
      if (dtype.bits == 8) return musa::dnn::Tensor::Type::UINT8;
      if (dtype.bits == 16) return musa::dnn::Tensor::Type::UINT16;
      if (dtype.bits == 32) return musa::dnn::Tensor::Type::UINT32;
      if (dtype.bits == 64) return musa::dnn::Tensor::Type::UINT64;
      break;
    case kDLBool:
      if (dtype.bits == 8) return musa::dnn::Tensor::Type::BOOL;
      break;
    default:
      break;
  }

  throw std::runtime_error("dl_dtype_to_mudnn_type got unsupported DLPack dtype");
}

inline musa::dnn::Tensor::Type to_mudnn_tensor_type(DLDataType dtype) {
  return dl_dtype_to_mudnn_type(dtype);
}

inline auto make_mudnn_tensor(void*                          ptr,
                              DLDataType                     datatype,
                              std::initializer_list<int64_t> shape,
                              std::initializer_list<int64_t> stride) {
  musa::dnn::Tensor tensor;

  tensor.SetType(dl_dtype_to_mudnn_type(datatype));
  tensor.SetAddr(ptr);
  tensor.SetNdInfo(shape, stride);

  return tensor;
}

inline auto make_mudnn_scalar_tensor(void* ptr, DLDataType datatype) {
  musa::dnn::Tensor tensor;

  tensor.SetType(dl_dtype_to_mudnn_type(datatype));
  tensor.SetAddr(ptr);
  tensor.SetNdInfo({1}, {1});

  return tensor;
}

inline auto make_mudnn_tensor(void* ptr, DLDataType datatype, tvm::ffi::ShapeView shape, tvm::ffi::ShapeView stride) {
  musa::dnn::Tensor tensor;

  TVM_FFI_ICHECK_EQ(shape.size(), stride.size()) << "shape and stride must have the same rank";
  tensor.SetType(dl_dtype_to_mudnn_type(datatype));
  tensor.SetAddr(ptr);
  MATE_MUDNN_STATUS_CHECK(tensor.SetNdInfo(static_cast<int64_t>(shape.size()), shape.data(), stride.data()));

  return tensor;
}

inline void fill_mudnn_tensor(musa::dnn::Handle& handle, musa::dnn::Tensor& tensor, DLDataType dtype, double value) {
  musa::dnn::Fill fill;
  if (dtype.code == kDLInt || dtype.code == kDLUInt || dtype.code == kDLBool) {
    MATE_MUDNN_STATUS_CHECK(fill.SetValue(static_cast<int64_t>(value)));
  } else {
    MATE_MUDNN_STATUS_CHECK(fill.SetValue(value));
  }
  MATE_MUDNN_STATUS_CHECK(fill.Run(handle, tensor));
}

inline void fill_mudnn_tensor(DLDevice                       device,
                              musaStream_t                   stream,
                              void*                          ptr,
                              DLDataType                     dtype,
                              std::initializer_list<int64_t> shape,
                              std::initializer_list<int64_t> stride,
                              double                         value) {
  TVM_FFI_ICHECK_EQ(device.device_type, kDLExtDev) << "fill_mudnn_tensor expects a MUSA device";
  tvm::ffi::MUSADeviceGuard device_guard(device.device_id);
  musa::dnn::Handle         handle(device.device_id);
  MATE_MUDNN_STATUS_CHECK(handle.SetStream(stream));
  auto tensor = make_mudnn_tensor(ptr, dtype, shape, stride);
  fill_mudnn_tensor(handle, tensor, dtype, value);
}

inline void fill_mudnn_tensor(DLDevice            device,
                              musaStream_t        stream,
                              void*               ptr,
                              DLDataType          dtype,
                              tvm::ffi::ShapeView shape,
                              tvm::ffi::ShapeView stride,
                              double              value) {
  TVM_FFI_ICHECK_EQ(device.device_type, kDLExtDev) << "fill_mudnn_tensor expects a MUSA device";
  tvm::ffi::MUSADeviceGuard device_guard(device.device_id);
  musa::dnn::Handle         handle(device.device_id);
  MATE_MUDNN_STATUS_CHECK(handle.SetStream(stream));
  auto tensor = make_mudnn_tensor(ptr, dtype, shape, stride);
  fill_mudnn_tensor(handle, tensor, dtype, value);
}

inline musa::dnn::MemoryHandler mudnn_internal_mem_alloc(size_t size) {
  if (size == 0) {
    return musa::dnn::MemoryHandler();
  }

  TVM_FFI_ICHECK_LE(size, static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      << "muDNN workspace size exceeds int64_t range";

  int device_id = 0;
  MATE_MUSA_RUNTIME_CHECK(musaGetDevice(&device_id));

  DLDevice device{kDLExtDev, device_id};
  // muDNN expects an untyped workspace buffer. Keep the env-owned byte tensor
  // alive through the deleter instead of falling back to raw device alloc/free.
  tvm::ffi::Tensor workspace = tvm::ffi::Tensor::FromEnvAlloc(
      TVMFFIEnvTensorAlloc, tvm::ffi::Shape{static_cast<int64_t>(size)}, dl_uint8, device);

  return musa::dnn::MemoryHandler(workspace.data_ptr(), [workspace](void* ptr) { (void)ptr; });
}
