#pragma once

#include <cstdint>
#include <type_traits>

namespace mega_moe::math {

template <typename dtype_t = void>
__host__ __device__ dtype_t* advance_ptr(void* ptr, const uint64_t num_bytes) {
  return reinterpret_cast<dtype_t*>(static_cast<uint8_t*>(ptr) + num_bytes);
}

template <typename T>
__host__ __device__ constexpr T constexpr_ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__host__ __device__ constexpr T constexpr_align(T a, T b) {
  return constexpr_ceil_div(a, b) * b;
}

template <typename T>
__host__ __device__ T ceil_div(const T& a, const T& b) {
  return (a + b - 1) / b;
}

template <typename T>
__host__ __device__ T align(const T& a, const T& b) {
  return ceil_div(a, b) * b;
}

}  // namespace mega_moe::math

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC)                                      \
  {                                                                                                  \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                                                \
    auto          __src       = (SRC);                                                               \
    auto          __dst       = (DST);                                                               \
    using RawValueType        = typename std::remove_reference<decltype(*(__src + 0))>::type;        \
    using ValueType           = typename std::remove_cv<RawValueType>::type;                         \
    ValueType unrolled_values[(UNROLL_FACTOR)];                                                      \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {         \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] =       \
          *(__src + __i + __j * 32);                                                                 \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) * (__dst + __i + __j * 32) = \
          unrolled_values[__j];                                                                      \
    }                                                                                                \
    for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += 32)              \
      *(__dst + __i) = *(__src + __i);                                                               \
  }

namespace mega_moe {

__device__ __forceinline__ uint32_t warp_reduce_add_u32(uint32_t value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ uint32_t warp_reduce_min_u32(uint32_t value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const uint32_t other = __shfl_xor_sync(0xffffffff, value, offset);
    value                = value < other ? value : other;
  }
  return value;
}

__device__ __forceinline__ float octet_reduce_max(float value, uint32_t lane_idx) {
  const uint32_t mask = 0xffu << ((lane_idx / 8) * 8);
  value               = fmaxf(value, __shfl_xor_sync(mask, value, 4));
  value               = fmaxf(value, __shfl_xor_sync(mask, value, 2));
  value               = fmaxf(value, __shfl_xor_sync(mask, value, 1));
  return value;
}

__device__ __forceinline__ uint32_t atomic_add_shared_u32(uint32_t* ptr, uint32_t val) {
  return atomicAdd(reinterpret_cast<uint32_t*>(__musa_ptr_gen_to_shared(ptr)), val);
}

__device__ __forceinline__ uint32_t atomic_add_global_u32(uint32_t* ptr, uint32_t val) {
  return atomicAdd(reinterpret_cast<uint32_t*>(__musa_ptr_gen_to_global(ptr)), val);
}

__device__ __forceinline__ unsigned long long atomic_add_global_u64(unsigned long long* ptr, unsigned long long val) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(__musa_ptr_gen_to_global(ptr)), val);
}

template <typename T, typename U>
__device__ __forceinline__ void st_relaxed_sys_global(const T* ptr, U val) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Global control store supports only 32-bit or 64-bit words");
  auto* global_ptr = __musa_ptr_gen_to_global(const_cast<T*>(ptr));
  if constexpr (sizeof(T) == 8) {
    atomicExch(reinterpret_cast<unsigned long long*>(global_ptr), static_cast<unsigned long long>(val));
  } else if constexpr (std::is_signed<T>::value) {
    atomicExch(reinterpret_cast<int*>(global_ptr), static_cast<int>(val));
  } else {
    atomicExch(reinterpret_cast<unsigned int*>(global_ptr), static_cast<unsigned int>(val));
  }
}

template <typename T>
__device__ __forceinline__ T ld_volatile_global(const T* ptr) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Global control load supports only 32-bit or 64-bit words");
  auto* global_ptr = __musa_ptr_gen_to_global(const_cast<T*>(ptr));
  if constexpr (sizeof(T) == 8) {
    return static_cast<T>(atomicCAS(reinterpret_cast<unsigned long long*>(global_ptr), 0ull, 0ull));
  } else if constexpr (std::is_signed<T>::value) {
    return static_cast<T>(atomicCAS(reinterpret_cast<int*>(global_ptr), 0, 0));
  } else {
    return static_cast<T>(atomicCAS(reinterpret_cast<unsigned int*>(global_ptr), 0u, 0u));
  }
}

constexpr float kSGLangSwiGLUFP8AmaxFloor = 1e-10f;
constexpr float kFinfoAmaxE4M3            = 448.0f;

__forceinline__ __device__ float clamp_fp8_e4m3(float x) {
  return fminf(fmaxf(x, -kFinfoAmaxE4M3), kFinfoAmaxE4M3);
}

}  // namespace mega_moe
