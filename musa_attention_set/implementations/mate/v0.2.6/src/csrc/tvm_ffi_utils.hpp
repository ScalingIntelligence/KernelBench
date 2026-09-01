#pragma once
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/musa/device_guard.h>
#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/function.h>

#include <initializer_list>

#include "dlpack/dlpack.h"
#include "dtype_utils.hpp"

using tvm::ffi::Tensor;
using tvm::ffi::TensorView;
namespace ffi = tvm::ffi;

#define FFI_CHECK(expr, msg) TVM_FFI_ICHECK(expr) << msg;
#define CHECK_MUSA(x) TVM_FFI_ICHECK(x.device().device_type == kDLExtDev) << #x " must be a MUSA tensor";
#define CHECK_CONTIGUOUS(x) TVM_FFI_ICHECK(x.IsContiguous()) << #x " must be contiguous";
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TVM_FFI_ICHECK_EQ(x.stride(-1), 1) \
  #x "must be contiguous at last dimension";
#define CHECK_INPUT(x) \
  CHECK_MUSA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_TYPE(x, st) TVM_FFI_ICHECK_EQ(x.dtype(), st) << "Inconsistency of Tensor type: " #x;
#define CHECK_INPUT_AND_TYPE(x, st) \
  CHECK_MUSA(x);                    \
  CHECK_CONTIGUOUS(x);              \
  CHECK_INPUT_TYPE(x, st)
#define CHECK_HAS_VALUE(x) TVM_FFI_ICHECK(x.has_value()) << #x " must have value";
#define CHECK_HAS_VALUE_WITH_MSG(x, msg) TVM_FFI_ICHECK(x.has_value()) << msg;
#define CHECK_MAYBE_INPUT_TYPE(maybe_x, st) \
  if (maybe_x.has_value()) {                \
    CHECK_INPUT_TYPE(maybe_x.value(), st);  \
  }
#define CHECK_MAYBE_INPUT_TYPES(maybe_x, st1, st2)                                   \
  if (maybe_x.has_value()) {                                                         \
    TVM_FFI_ICHECK(maybe_x.value().dtype() == st1 || maybe_x.value().dtype() == st2) \
        << "Inconsistency of Tensor type: " #maybe_x " must be " #st1 " or " #st2;   \
  }
#define CHECK_SAME_DTYPE(x, y) \
  TVM_FFI_ICHECK(x.dtype() == y.dtype()) << "Inconsistency of Tensor type: " #x " dtype must match " #y " dtype";
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_MUSA(x);                           \
  CHECK_LAST_DIM_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TVM_FFI_ICHECK_EQ(x.ndim(), d) << #x " must be a " #d "D tensor";
#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)
#define CHECK_DEVICE(a, b)                                           \
  TVM_FFI_ICHECK_EQ(a.device().device_type, b.device().device_type); \
  TVM_FFI_ICHECK_EQ(a.device().device_id, b.device().device_id);

inline musaStream_t get_current_stream() {
  int device;
  musaGetDevice(&device);
  return static_cast<musaStream_t>(TVMFFIEnvGetStream(kDLExtDev, device));
}

inline musaStream_t get_stream(DLDevice device) {
  return static_cast<musaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}

inline int64_t get_element_size(ffi::Tensor x) {
  return (x.dtype().bits * x.dtype().lanes) / 8;
}

inline int64_t get_element_size(ffi::TensorView x) {
  return (x.dtype().bits * x.dtype().lanes) / 8;
}

inline ffi::Tensor alloc_tensor(tvm::ffi::Shape shape, DLDataType dtype, DLDevice device) {
  return ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, shape, dtype, device);
}

inline void expect_shape(ffi::TensorView tensor, std::initializer_list<int64_t> expected, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), static_cast<int32_t>(expected.size())) << name << " has unexpected rank";
  int32_t dim = 0;
  for (int64_t value : expected) {
    TVM_FFI_ICHECK_EQ(tensor.size(dim), value) << name << " has unexpected shape at dim " << dim;
    ++dim;
  }
}

inline void check_shape(const tvm::ffi::Tensor& a, const tvm::ffi::Tensor& b, const char* a_name, const char* b_name) {
  TVM_FFI_ICHECK_EQ(a.ndim(), b.ndim()) << a_name << ".ndim() and " << b_name << ".ndim() mismatch";
  for (int i = 0; i < a.ndim(); ++i) {
    TVM_FFI_ICHECK_EQ(a.size(i), b.size(i))
        << a_name << ".size(" << i << ") and " << b_name << ".size(" << i << ") mismatch";
  }
}

inline void check_shape(const tvm::ffi::TensorView& a,
                        const tvm::ffi::TensorView& b,
                        const char*                 a_name,
                        const char*                 b_name) {
  TVM_FFI_ICHECK_EQ(a.ndim(), b.ndim()) << a_name << ".ndim() and " << b_name << ".ndim() mismatch";
  for (int i = 0; i < a.ndim(); ++i) {
    TVM_FFI_ICHECK_EQ(a.size(i), b.size(i))
        << a_name << ".size(" << i << ") and " << b_name << ".size(" << i << ") mismatch";
  }
}

inline void check_shape_at_dims(const ffi::TensorView&         tensor,
                                const ffi::TensorView&         ref,
                                std::initializer_list<int32_t> dims,
                                const char*                    name,
                                const char*                    ref_name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), ref.ndim()) << name << " rank must match " << ref_name;
  for (int32_t dim : dims) {
    TVM_FFI_ICHECK_GE(dim, 0);
    TVM_FFI_ICHECK_LT(dim, tensor.ndim()) << name << " invalid dim " << dim;
    TVM_FFI_ICHECK_EQ(tensor.size(dim), ref.size(dim)) << name << " shape must match " << ref_name << " at dim " << dim;
  }
}

struct StridedTensorView {
  ffi::Shape      shape;
  ffi::Shape      strides;
  DLTensor        prototype;
  ffi::TensorView view;

  static DLTensor make_prototype(ffi::TensorView           base,
                                 ffi::ShapeView            shape,
                                 ffi::ShapeView            strides,
                                 std::optional<DLDataType> dtype_override,
                                 std::optional<int64_t>    byte_offset) {
    TVMFFIAny any{};
    ffi::TypeTraits<ffi::TensorView>::CopyToAnyView(base, &any);
    auto prototype    = *static_cast<DLTensor*>(any.v_ptr);
    prototype.shape   = const_cast<int64_t*>(shape.data());
    prototype.ndim    = static_cast<int>(shape.size());
    prototype.strides = const_cast<int64_t*>(strides.data());
    TVM_FFI_ICHECK_EQ(shape.size(), strides.size());
    if (dtype_override.has_value()) {
      prototype.dtype = dtype_override.value();
    }

    const int64_t byte_offset_i64 = byte_offset.value_or(0);
    TVM_FFI_ICHECK_GE(byte_offset_i64, 0);
    if (byte_offset_i64 != 0) {
      prototype.data        = reinterpret_cast<void*>(reinterpret_cast<char*>(prototype.data) + byte_offset_i64);
      prototype.byte_offset = 0;
    }
    return prototype;
  }

  StridedTensorView(ffi::TensorView        base,
                    ffi::Shape             shape_,
                    ffi::Shape             strides_,
                    ffi::Optional<int64_t> element_offset = ffi::Optional<int64_t>())  // NOLINT(*)
      : shape(std::move(shape_)),
        strides(std::move(strides_)),
        prototype(make_prototype(base,
                                 shape,
                                 strides,
                                 std::nullopt,
                                 element_offset.has_value()
                                     ? std::optional<int64_t>(element_offset.value() * get_element_size(base))
                                     : std::nullopt)),
        view(&prototype) {
  }

  StridedTensorView(ffi::TensorView        base,
                    ffi::Shape             shape_,
                    ffi::Shape             strides_,
                    DLDataType             dtype_override,
                    ffi::Optional<int64_t> byte_offset = ffi::Optional<int64_t>())  // NOLINT(*)
      : shape(std::move(shape_)),
        strides(std::move(strides_)),
        prototype(make_prototype(base,
                                 shape,
                                 strides,
                                 std::optional<DLDataType>(dtype_override),
                                 byte_offset.has_value() ? std::optional<int64_t>(byte_offset.value()) : std::nullopt)),
        view(&prototype) {
  }

  operator ffi::TensorView() const {
    return view;
  }  // NOLINT(*)
};
