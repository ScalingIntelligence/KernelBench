#pragma once

#include <musa.h>

#include <cstdint>
#include <stdexcept>

#include "dlpack/dlpack.h"

inline constexpr int64_t encode_dlpack_dtype(DLDataType dtype) {
  return (static_cast<int64_t>(dtype.code) << 16) | (static_cast<int64_t>(dtype.bits) << 8) |
         static_cast<int64_t>(dtype.lanes);
}

constexpr DLDataType dl_uint8            = DLDataType{kDLUInt, 8, 1};
constexpr DLDataType dl_uint16           = DLDataType{kDLUInt, 16, 1};
constexpr DLDataType dl_uint32           = DLDataType{kDLUInt, 32, 1};
constexpr DLDataType dl_uint64           = DLDataType{kDLUInt, 64, 1};
constexpr DLDataType dl_int4             = DLDataType{kDLInt, 4, 1};
constexpr DLDataType dl_int8             = DLDataType{kDLInt, 8, 1};
constexpr DLDataType dl_int16            = DLDataType{kDLInt, 16, 1};
constexpr DLDataType dl_int32            = DLDataType{kDLInt, 32, 1};
constexpr DLDataType dl_int64            = DLDataType{kDLInt, 64, 1};
constexpr DLDataType dl_float16          = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType dl_float32          = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_float64          = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType dl_float8_e4m3fn    = DLDataType{kDLFloat8_e4m3fn, 8, 1};
constexpr DLDataType dl_float8_e5m2      = DLDataType{kDLFloat8_e5m2, 8, 1};
constexpr DLDataType dl_float4_e2m1fn    = DLDataType{kDLFloat4_e2m1fn, 4, 1};
constexpr DLDataType dl_float4_e2m1fn_x2 = DLDataType{kDLFloat4_e2m1fn, 4, 2};
constexpr DLDataType dl_bfloat16         = DLDataType{kDLBfloat, 16, 1};
constexpr DLDataType dl_bool             = DLDataType{kDLBool, 8, 1};

constexpr int64_t uint8_code         = encode_dlpack_dtype(dl_uint8);
constexpr int64_t int4_code          = encode_dlpack_dtype(dl_int4);
constexpr int64_t int8_code          = encode_dlpack_dtype(dl_int8);
constexpr int64_t int32_code         = encode_dlpack_dtype(dl_int32);
constexpr int64_t int64_code         = encode_dlpack_dtype(dl_int64);
constexpr int64_t float16_code       = encode_dlpack_dtype(dl_float16);
constexpr int64_t bfloat16_code      = encode_dlpack_dtype(dl_bfloat16);
constexpr int64_t float32_code       = encode_dlpack_dtype(dl_float32);
constexpr int64_t float8_e4m3fn_code = encode_dlpack_dtype(dl_float8_e4m3fn);
constexpr int64_t float8_e5m2_code   = encode_dlpack_dtype(dl_float8_e5m2);
constexpr int64_t float4_e2m1fn_code = encode_dlpack_dtype(dl_float4_e2m1fn);

inline bool dtype_equal(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

inline bool is_bf16_or_fp16_dtype(DLDataType dtype) {
  const int64_t code = encode_dlpack_dtype(dtype);
  return code == float16_code || code == bfloat16_code;
}

inline bool is_bf16_or_fp16_or_fp32_dtype(DLDataType dtype) {
  const int64_t code = encode_dlpack_dtype(dtype);
  return code == float16_code || code == bfloat16_code || code == float32_code;
}

inline bool is_fp8_dtype(DLDataType dtype) {
  const int64_t code = encode_dlpack_dtype(dtype);
  return code == float8_e4m3fn_code || code == float8_e5m2_code;
}

inline bool is_8bit_dtype(DLDataType dtype) {
  const int64_t code = encode_dlpack_dtype(dtype);
  return is_fp8_dtype(dtype) || code == int8_code || code == uint8_code;
}

inline int dl_dtype_size(DLDataType dtype) {
  if (dtype.lanes <= 0) {
    throw std::runtime_error("dl_dtype_size got invalid lanes");
  }
  const int bits = dtype.bits * dtype.lanes;
  if (bits % 8 != 0) {
    throw std::runtime_error("dl_dtype_size only supports byte-aligned dtypes");
  }
  return bits / 8;
}

inline MUtensorDescriptorDataType dl_dtype_to_tme_type(DLDataType dtype) {
  if (dtype.lanes != 1) {
    throw std::runtime_error("dl_dtype_to_tme_type only supports lane=1 dtypes");
  }

  switch (dtype.code) {
    case kDLFloat:
      if (dtype.bits == 64) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT64;
      if (dtype.bits == 32) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32;
      if (dtype.bits == 16) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT16;
      break;
    case kDLBfloat:
      if (dtype.bits == 16) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16;
      break;
    case kDLFloat8_e4m3fn:
      if (dtype.bits == 8) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8;
      break;
    case kDLFloat8_e5m2:
      if (dtype.bits == 8) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8;
      break;
    case kDLInt:
      if (dtype.bits == 64) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT64;
      if (dtype.bits == 32) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT32;
      if (dtype.bits == 16) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT16;
      if (dtype.bits == 8) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8;
      break;
    case kDLUInt:
      if (dtype.bits == 64) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT64;
      if (dtype.bits == 32) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32;
      if (dtype.bits == 16) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;
      if (dtype.bits == 8) return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
      break;
    default:
      break;
  }

  throw std::runtime_error("dl_dtype_to_tme_type got unsupported DLPack dtype");
}
