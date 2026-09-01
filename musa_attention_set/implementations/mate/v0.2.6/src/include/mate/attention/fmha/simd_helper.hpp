#pragma once

#include <mute/numeric/numeric_types.hpp>  // sizeof_bits_v

#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310))
#define MUSA_SIMD_MATH_ENABLED
#endif

namespace mate::attention::fmha {

// vector type definations

// TODO: check it for sub-byte usage!
template <typename T, int N>
struct vector_type_helper {
  using U    = std::conditional_t < sizeof_bits_v<T><8, uint8_t, T>;
  using type = U __attribute__((vector_size(N * sizeof_bits_v<U> / 8)));
};

template <typename T, int N>
using vector_type = typename vector_type_helper<T, N>::type;

using float32x2_t = float __attribute__((vector_size(8)));
using float32x4_t = float __attribute__((vector_size(16)));

using float16x2_t = _Float16 __attribute__((vector_size(4)));
using float16x4_t = _Float16 __attribute__((vector_size(8)));
using float16x8_t = _Float16 __attribute__((vector_size(16)));

using bfloat16x2_t = __bf16 __attribute__((vector_size(4)));
using bfloat16x4_t = __bf16 __attribute__((vector_size(8)));
using bfloat16x8_t = __bf16 __attribute__((vector_size(16)));

// SIMD overload

// Unary
MUTE_DEVICE
void vexp2(float32x2_t& d, float32x2_t const& s) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  d = __musa_exp2_f_bst2_v(s);
#else
  MUTE_UNROLL
  for (int i = 0; i < 2; ++i) {
    d[i] = exp2f(s[i]);
  }
#endif
}

MUTE_DEVICE
void vexp2(float32x4_t& d, float32x4_t const& s) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  d = __musa_exp2_f_bst4_v(s);
#else
  MUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    d[i] = exp2f(s[i]);
  }
#endif
}

// Binary

// vector + vector
MUTE_DEVICE
void vmax(float32x2_t& c, float32x2_t const& a, float32x2_t const& b) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  c = __musa_max_f_bst2_vv(a, b);
#else
  MUTE_UNROLL
  for (int i = 0; i < 2; ++i) {
    c[i] = std::max(a[i], b[i]);
  }
#endif
}
MUTE_DEVICE
void vmax(float32x4_t& c, float32x4_t const& a, float32x4_t const& b) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  c = __musa_max_f_bst4_vv(a, b);
#else
  MUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    c[i] = std::max(a[i], b[i]);
  }
#endif
}

MUTE_DEVICE
void vadd(float32x2_t& c, float32x2_t const& a, float32x2_t const& b) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  c = __musa_add_f_bst2_vv(a, b);
#else
  MUTE_UNROLL
  for (int i = 0; i < 2; ++i) {
    c[i] = a[i] + b[i];
  }
#endif
}

MUTE_DEVICE
void vadd(float32x4_t& c, float32x4_t const& a, float32x4_t const& b) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  c = __musa_add_f_bst4_vv(a, b);
#else
  MUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    c[i] = a[i] + b[i];
  }
#endif
}

// vector + scalar
MUTE_DEVICE
void vmax(float32x2_t& c, float32x2_t const& a, float const& b) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  c = __musa_max_f_bst2_sv(b, a);
#else
  MUTE_UNROLL
  for (int i = 0; i < 2; ++i) {
    c[i] = std::max(a[i], b);
  }
#endif
}

MUTE_DEVICE
void vmax(float32x4_t& c, float32x4_t const& a, float const& b) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  c = __musa_max_f_bst2_sv(b, a);
#else
  MUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    c[i] = std::max(a[i], b);
  }
#endif
}

MUTE_DEVICE
void vmax(float32x2_t& c, float const& a, float32x2_t const& b) {
  return vmax(c, b, a);
}

MUTE_DEVICE
void vmax(float32x4_t& c, float const& a, float32x4_t const& b) {
  return vmax(c, b, a);
}

MUTE_DEVICE
void vadd(float32x2_t& c, float32x2_t const& a, float const& b) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  c = __musa_add_f_bst2_sv(b, a);
#else
  MUTE_UNROLL
  for (int i = 0; i < 2; ++i) {
    c[i] = a[i] + b;
  }
#endif
}

MUTE_DEVICE
void vadd(float32x4_t& c, float32x4_t const& a, float const& b) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  c = __musa_add_f_bst4_sv(b, a);
#else
  MUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    c[i] = a[i] + b;
  }
#endif
}

MUTE_DEVICE
void vadd(float32x2_t& c, float const& a, float32x2_t const& b) {
  return vadd(c, b, a);
}

MUTE_DEVICE
void vadd(float32x4_t& c, float const& a, float32x4_t const& b) {
  return vadd(c, b, a);
}

// Ternary

// vector + vector + vector
MUTE_DEVICE
void vfma(float32x2_t& d, float32x2_t const& a, float32x2_t const& b, float32x2_t const& c) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  d = __musa_fma_f_bst2_vvv(a, b, c);
#else
  MUTE_UNROLL
  for (int i = 0; i < 2; ++i) {
    d[i] = a[i] * b[i] + c[i];
  }
#endif
}

MUTE_DEVICE
void vfma(float32x4_t& d, float32x4_t const& a, float32x4_t const& b, float32x4_t const& c) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  d = __musa_fma_f_bst4_vvv(a, b, c);
#else
  MUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    d[i] = a[i] * b[i] + c[i];
  }
#endif
}

// vector + scalar + vector
MUTE_DEVICE
void vfma(float32x4_t& d, float32x4_t const& a, float const& b, float32x4_t const& c) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  d = __musa_fma_f_bst4_svv(b, a, c);
#else
  MUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    d[i] = a[i] * b + c[i];
  }
#endif
}

MUTE_DEVICE
void vfma(float32x4_t& d, float const& a, float32x4_t const& b, float32x4_t const& c) {
  return vfma(d, b, a, c);
}

MUTE_DEVICE
void vfma(float32x4_t& d, float32x4_t const& a, float32x4_t const& b, float const& c) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  d = __musa_fma_f_bst4_vvs(a, b, c);
#else
  MUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    d[i] = a[i] * b[i] + c;
  }
#endif
}

MUTE_DEVICE
void vfma(float32x4_t& d, float const& a, float32x4_t const& b, float const& c) {
#if defined(MUTE_SIMD_MATH_ENABLED)
  d = __musa_fma_f_bst4_svs(a, b, c);
#else
  MUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    d[i] = a * b[i] + c;
  }
#endif
}

MUTE_DEVICE
void vfma(float32x4_t& d, float32x4_t const& a, float const& b, float const& c) {
  return vfma(d, b, a, c);
}

}  // namespace mate::attention::fmha
