#pragma once

#include <mutlass/fast_math.h>
#include <tvm/ffi/extra/musa/base.h>

#include <cstdint>
#include <cstdio>
#include <mute/int_tuple.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include "musa.h"
#include "musa_runtime.h"

#ifndef MATE_MUSA_DRIVER_CHECK
#define MATE_MUSA_DRIVER_CHECK(error)                                                                       \
  do {                                                                                                      \
    MUresult __err = (error);                                                                               \
    if (__err != MUSA_SUCCESS) {                                                                            \
      const char* __err_name = "unknown";                                                                   \
      const char* __err_str  = "unknown";                                                                   \
      if (muGetErrorName(__err, &__err_name) != MUSA_SUCCESS) {                                             \
        __err_name = "unknown";                                                                             \
      }                                                                                                     \
      if (muGetErrorString(__err, &__err_str) != MUSA_SUCCESS) {                                            \
        __err_str = "unknown";                                                                              \
      }                                                                                                     \
      TVM_FFI_THROW(RuntimeError) << "MUSA Driver Error: " << __err_name << " (" << static_cast<int>(__err) \
                                  << "): " << __err_str;                                                    \
    }                                                                                                       \
  } while (0)
#endif

#ifndef MATE_MUSA_RUNTIME_CHECK
#define MATE_MUSA_RUNTIME_CHECK(error) TVM_FFI_CHECK_MUSA_ERROR(error)
#endif

#ifndef MATE_CHECK_MUSA_KERNEL_LAUNCH
#define MATE_CHECK_MUSA_KERNEL_LAUNCH() MATE_MUSA_RUNTIME_CHECK(musaGetLastError())
#endif

#ifndef MATE_DEVICE_ASSERT
#define MATE_DEVICE_ASSERT(cond)                                           \
  do {                                                                     \
    if (not(cond)) {                                                       \
      printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
      __trap();                                                            \
    }                                                                      \
  } while (0)
#endif

enum class TensorMajor {

  MN = 0,
  K

};  // enum class TensorMajor

struct TmeDesc {
  MUtensorDescriptor desc{};

  TmeDesc() = default;

  static uint64_t desc_byte_stride(uint64_t stride, int element_size) {
    TVM_FFI_ICHECK_GT(element_size, 0) << "TME element size must be positive";
    return stride * static_cast<uint64_t>(element_size);
  }

  template <typename T>
  static uint64_t checked_desc_extent(T value, const char* name) {
    if constexpr (std::is_signed_v<T>) {
      TVM_FFI_ICHECK_GE(value, static_cast<T>(0)) << name << " must be non-negative";
    }
    return static_cast<uint64_t>(value);
  }

  TmeDesc(const std::vector<uint64_t>& dims,
          const std::vector<uint64_t>& strides,
          void*                        ptr,
          MUtensorDescriptorDataType   tensor_dtype,
          MUtensorDescriptorInterleave interleave        = MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE,
          uint64_t                     oob_constant_fill = 0) {
    auto rank = dims.size();

    TVM_FFI_ICHECK_GT(rank, 0) << "TmeDesc dims must be non-empty";
    TVM_FFI_ICHECK_LE(rank, 5) << "TmeDesc rank must be <= 5";
    TVM_FFI_ICHECK_EQ(strides.size(), rank - 1) << "TmeDesc strides must have rank - 1 entries";

    uint64_t  desc_strides[4]{};
    const int element_size = get_data_type_size(tensor_dtype);

    for (size_t i = 0; i < rank - 1; ++i) {
      desc_strides[i] = desc_byte_stride(strides[i], element_size);
    }
    MATE_MUSA_DRIVER_CHECK(muTensorDescriptorEncode(
        &desc, tensor_dtype, rank, ptr, dims.data(), desc_strides, interleave, oob_constant_fill));
  }

  template <class Dim, class Stride>
  TmeDesc(const Dim&                   dim,
          const Stride&                stride,
          void*                        ptr,
          MUtensorDescriptorDataType   tensor_dtype,
          MUtensorDescriptorInterleave interleave        = MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE,
          uint64_t                     oob_constant_fill = 0) {
    constexpr int dim_rank    = mute::rank(Dim{});
    constexpr int stride_rank = mute::rank(Stride{});

    static_assert(int(dim_rank) <= 5, "TmeDesc Dim must be <= 5");
    static_assert(int(dim_rank) > 0, "TmeDesc Dim must be > 0");
    static_assert(int(dim_rank) == int(stride_rank) + 1, "TmeDesc Dim and Stride must match!");

    uint64_t  desc_dims[dim_rank]{};
    uint64_t  desc_strides[stride_rank]{};
    const int element_size = get_data_type_size(tensor_dtype);
    mute::for_each(mute::make_seq<dim_rank>{},
                   [&](auto i) { desc_dims[i] = checked_desc_extent(mute::get<i>(dim), "TME dim"); });
    mute::for_each(mute::make_seq<stride_rank>{}, [&](auto i) {
      const auto stride_elem = checked_desc_extent(mute::get<i>(stride), "TME stride");
      desc_strides[i]        = desc_byte_stride(stride_elem, element_size);
    });

    MATE_MUSA_DRIVER_CHECK(muTensorDescriptorEncode(
        &desc, tensor_dtype, dim_rank, ptr, desc_dims, desc_strides, interleave, oob_constant_fill));
  }

  static int get_data_type_size(MUtensorDescriptorDataType data_type) {
    switch (data_type) {
      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT64:
      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT64:
      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT64:
        return 8;

      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32:
      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT32:
      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32:
        return 4;

      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT16:
      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16:
      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT16:
      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16:
        return 2;

      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8:
      case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8:
        return 1;

      default:
        throw std::runtime_error("get_data_type_size() get Unsupported MUtensorDescriptorDataType");
    }
  }
};  // struct TmeDesc

struct Arch {
  int major;
  int minor;

  bool is_mp31() const {
    return major == 3 && minor == 1;
  }
};

enum class TensorQuantMode {
  NO_QUANT = 0,
  TENSOR,
  CHANNEL,
  BLOCK,
  GROUP,
};  // enum class TensorQuantMode

enum class MatMulScalingMode {

  // TODO: replace by TensorQuantMode

  // Mat A + Mat B
  // for example
  // GROUP_BLOCK means Mat A is per group scaling and Mat B is per block scaling

  TENSOR_TENSOR,
  CHANNEL_TENSOR,
  CHANNEL_CHANNEL,
  GROUP_BLOCK,

};  // struct MatMulScalingMode

template <class T>
auto get_fast_div_mod(T val) {
  if (val != static_cast<T>(0)) {
    return mutlass::FastDivmod(val);
  }
  return mutlass::FastDivmod();
}
