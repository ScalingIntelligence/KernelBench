#pragma once

#include <cstdint>

namespace tl {

TL_DEVICE void cp_async_commit() {}

template <int N> TL_DEVICE void cp_async_wait() {
  // MTCC has no corresponding MUSA C API; this must use the builtin.
  __musa_memcpy_g2s_wait();
}

template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr,
                           void const *const global_ptr) {
  // MTCC has no corresponding MUSA C API; this must use the builtin.
  __musa_memcpy_g2s((void _AS3 *)smem_addr,
                    (void const _AS1 *)global_ptr, N, 0);
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *const global_ptr,
                                       bool condition) {
  if (condition) {
    cp_async_gs<N>(smem_addr, global_ptr);
  }
}

// Global memory load/store helpers used by T.ldg*/T.stg* codegen. MUSA
// supports vector pointer access for these POD vector types; callers are
// responsible for providing naturally aligned addresses.
TL_DEVICE uint32_t load_global_32(const void *ptr) {
  return *reinterpret_cast<const uint32_t *>(ptr);
}

TL_DEVICE uint2 load_global_64(const void *ptr) {
  return *reinterpret_cast<const uint2 *>(ptr);
}

TL_DEVICE uint4 load_global_128(const void *ptr) {
  return *reinterpret_cast<const uint4 *>(ptr);
}

TL_DEVICE longlong4 load_global_256(const longlong4 *ptr) { return *ptr; }

TL_DEVICE ulonglong4 load_global_256(const ulonglong4 *ptr) { return *ptr; }

template <typename T> TL_DEVICE ulonglong4 load_global_256(const T *ptr) {
  return *reinterpret_cast<const ulonglong4 *>(ptr);
}

TL_DEVICE uint32_t load_global_32_conditional(const void *ptr, bool pred) {
  return pred ? load_global_32(ptr) : uint32_t{};
}

TL_DEVICE uint2 load_global_64_conditional(const void *ptr, bool pred) {
  return pred ? load_global_64(ptr) : uint2{};
}

TL_DEVICE uint4 load_global_128_conditional(const void *ptr, bool pred) {
  return pred ? load_global_128(ptr) : uint4{};
}

TL_DEVICE longlong4 load_global_256_conditional(const longlong4 *ptr,
                                                bool pred) {
  return pred ? load_global_256(ptr) : longlong4{};
}

TL_DEVICE ulonglong4 load_global_256_conditional(const ulonglong4 *ptr,
                                                 bool pred) {
  return pred ? load_global_256(ptr) : ulonglong4{};
}

template <typename T>
TL_DEVICE ulonglong4 load_global_256_conditional(const T *ptr, bool pred) {
  return pred ? load_global_256(ptr) : ulonglong4{};
}

TL_DEVICE void store_global_32(void *ptr, uint32_t value) {
  *reinterpret_cast<uint32_t *>(ptr) = value;
}

TL_DEVICE void store_global_64(void *ptr, uint2 value) {
  *reinterpret_cast<uint2 *>(ptr) = value;
}

TL_DEVICE void store_global_128(void *ptr, uint4 value) {
  *reinterpret_cast<uint4 *>(ptr) = value;
}

TL_DEVICE void store_global_256(void *ptr, const longlong4 &value) {
  *reinterpret_cast<longlong4 *>(ptr) = value;
}

TL_DEVICE void store_global_256(void *ptr, const ulonglong4 &value) {
  *reinterpret_cast<ulonglong4 *>(ptr) = value;
}

template <typename T>
TL_DEVICE void store_global_256(void *ptr, const T &value) {
  const ulonglong4 &value_u64 = *reinterpret_cast<const ulonglong4 *>(&value);
  *reinterpret_cast<ulonglong4 *>(ptr) = value_u64;
}

TL_DEVICE void store_global_32_conditional(void *ptr, uint32_t value,
                                           bool pred) {
  if (pred) {
    store_global_32(ptr, value);
  }
}

TL_DEVICE void store_global_64_conditional(void *ptr, uint2 value, bool pred) {
  if (pred) {
    store_global_64(ptr, value);
  }
}

TL_DEVICE void store_global_128_conditional(void *ptr, uint4 value, bool pred) {
  if (pred) {
    store_global_128(ptr, value);
  }
}

TL_DEVICE void store_global_256_conditional(void *ptr, const longlong4 &value,
                                            bool pred) {
  if (pred) {
    store_global_256(ptr, value);
  }
}

TL_DEVICE void store_global_256_conditional(void *ptr, const ulonglong4 &value,
                                            bool pred) {
  if (pred) {
    store_global_256(ptr, value);
  }
}

template <typename T>
TL_DEVICE void store_global_256_conditional(void *ptr, const T &value,
                                            bool pred) {
  if (pred) {
    store_global_256(ptr, value);
  }
}

} // namespace tl
