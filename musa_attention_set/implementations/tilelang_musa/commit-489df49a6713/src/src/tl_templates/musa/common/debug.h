#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <musa_runtime.h>

#define TL_DEVICE __forceinline__ __device__

template <typename T> struct PrintTraits {
  static __device__ void print_var(const char *msg, T value) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
           "dtype=unknown value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, (const void *)&value);
  }

  static __device__ void print_buffer(const char *msg, const char *buf_name,
                                      int index, T value) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
           "buffer=%s, index=%d, dtype=unknown value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, buf_name, index, (const void *)&value);
  }
};

#define DEFINE_PRINT_TRAIT(TYPE, NAME, FORMAT, CAST_TYPE)                      \
  template <> struct PrintTraits<TYPE> {                                       \
    static __device__ void print_var(const char *msg, TYPE value) {            \
      printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "        \
             "dtype=" NAME " value=" FORMAT "\n",                            \
             msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,             \
             threadIdx.y, threadIdx.z, (CAST_TYPE)value);                      \
    }                                                                          \
    static __device__ void print_buffer(const char *msg,                       \
                                        const char *buf_name, int index,       \
                                        TYPE value) {                          \
      printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "        \
             "buffer=%s, index=%d, dtype=" NAME " value=" FORMAT "\n",        \
             msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,             \
             threadIdx.y, threadIdx.z, buf_name, index, (CAST_TYPE)value);     \
    }                                                                          \
  }

DEFINE_PRINT_TRAIT(char, "char", "%d", int);
DEFINE_PRINT_TRAIT(signed char, "signed char", "%d", int);
DEFINE_PRINT_TRAIT(unsigned char, "unsigned char", "%u", unsigned int);
DEFINE_PRINT_TRAIT(short, "short", "%d", int);
DEFINE_PRINT_TRAIT(unsigned short, "unsigned short", "%u", unsigned int);
DEFINE_PRINT_TRAIT(int, "int", "%d", int);
DEFINE_PRINT_TRAIT(unsigned int, "uint", "%u", unsigned int);
DEFINE_PRINT_TRAIT(long, "long", "%ld", long);
DEFINE_PRINT_TRAIT(unsigned long, "ulong", "%lu", unsigned long);
DEFINE_PRINT_TRAIT(long long, "long long", "%lld", long long);
DEFINE_PRINT_TRAIT(unsigned long long, "unsigned long long", "%llu",
                   unsigned long long);
DEFINE_PRINT_TRAIT(float, "float", "%f", float);
DEFINE_PRINT_TRAIT(double, "double", "%lf", double);

#if defined(TL_MUSA_ENABLE_FP16)
DEFINE_PRINT_TRAIT(half, "half", "%f", float);
#endif

#if defined(TL_MUSA_ENABLE_BF16)
DEFINE_PRINT_TRAIT(mt_bfloat16, "mt_bfloat16", "%f", float);
#endif

#undef DEFINE_PRINT_TRAIT

template <> struct PrintTraits<bool> {
  static __device__ void print_var(const char *msg, bool value) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
           "dtype=bool value=%s\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, value ? "true" : "false");
  }

  static __device__ void print_buffer(const char *msg, const char *buf_name,
                                      int index, bool value) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
           "buffer=%s, index=%d, dtype=bool value=%s\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, buf_name, index, value ? "true" : "false");
  }
};

template <typename T> struct PrintTraits<T *> {
  static __device__ void print_var(const char *msg, T *value) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
           "dtype=pointer value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, (void *)value);
  }

  static __device__ void print_buffer(const char *msg, const char *buf_name,
                                      int index, T *value) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
           "buffer=%s, index=%d, dtype=pointer value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, buf_name, index, (void *)value);
  }
};

template <typename T> __device__ void debug_print_var(const char *msg, T value) {
  PrintTraits<T>::print_var(msg, value);
}

template <typename T>
__device__ void debug_print_buffer_value(const char *msg, const char *buf_name,
                                         int index, T value) {
  PrintTraits<T>::print_buffer(msg, buf_name, index, value);
}

template <>
__device__ void debug_print_buffer_value<uint16_t>(const char *msg,
                                                   const char *buf_name,
                                                   int index, uint16_t value) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
         "buffer=%s, index=%d, dtype=uint16_t value=%u\n",
         msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, buf_name, index, (uint32_t)value);
}

TL_DEVICE void device_assert(bool condition) { assert(condition); }

TL_DEVICE void device_assert_with_msg(bool condition, const char *msg) {
  if (!condition) {
    printf("Device assert failed: %s\n", msg);
    assert(0);
  }
}

__device__ void debug_print_msg(const char *msg) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d)\n", msg,
         blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z);
}
