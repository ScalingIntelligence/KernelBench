#pragma once

#include <musa_runtime.h>
#include <mutlass/bfloat16.h>
#include <mutlass/half.h>
#include <mutlass/mutlass.h>

#include <mute/arch/simd_mp31.hpp>

#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310))
#define MATE_FLAT_SIMD_MATH_ENABLED
#endif

namespace mate::flat::detail {

MUTLASS_DEVICE float fast_exp2(float x) {
  return __musa_exp2_f(x);
}
}  // namespace mate::flat::detail

namespace mate::flat::simd {

MUTLASS_DEVICE float4 fast_exp2(float4 x) {
  float4 y;
  mute::fast_exp2(y, x);
  return y;
}

MUTLASS_DEVICE float4 splat4(float x) {
  return make_float4(x, x, x, x);
}

MUTLASS_DEVICE void vadd(float4& c, float4 const& a, float b) {
#if defined(MATE_FLAT_SIMD_MATH_ENABLED)
  c = ::add(b, a);
#else
  c = make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
#endif
}

MUTLASS_DEVICE float4 vadd(float4 const& a, float b) {
  float4 c;
  vadd(c, a, b);
  return c;
}

MUTLASS_DEVICE void vadd(float4& c, float4 const& a, float4 const& b) {
#if defined(MATE_FLAT_SIMD_MATH_ENABLED)
  c = ::add(a, b);
#else
  c = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
#endif
}

MUTLASS_DEVICE float4 vadd(float4 const& a, float4 const& b) {
  float4 c;
  vadd(c, a, b);
  return c;
}

MUTLASS_DEVICE void vmul(float4& c, float4 const& a, float b) {
#if defined(MATE_FLAT_SIMD_MATH_ENABLED)
  c = ::mul(b, a);
#else
  c = make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
#endif
}

MUTLASS_DEVICE float4 vmul(float4 const& a, float b) {
  float4 c;
  vmul(c, a, b);
  return c;
}

MUTLASS_DEVICE void vmul(float4& c, float4 const& a, float4 const& b) {
#if defined(MATE_FLAT_SIMD_MATH_ENABLED)
  c = ::mul(a, b);
#else
  c = make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
#endif
}

MUTLASS_DEVICE float4 vmul(float4 const& a, float4 const& b) {
  float4 c;
  vmul(c, a, b);
  return c;
}

MUTLASS_DEVICE void vsub(float4& c, float4 const& a, float b) {
  vadd(c, a, -b);
}

MUTLASS_DEVICE float4 vsub(float4 const& a, float b) {
  float4 c;
  vsub(c, a, b);
  return c;
}

MUTLASS_DEVICE void vsub(float4& c, float4 const& a, float4 const& b) {
  vadd(c, a, vmul(b, -1.0f));
}

MUTLASS_DEVICE float4 vsub(float4 const& a, float4 const& b) {
  float4 c;
  vsub(c, a, b);
  return c;
}

MUTLASS_DEVICE void dot4(float4 const& a, float4 const& b, float& c) {
  c = fmaf(a.w, b.w, fmaf(a.z, b.z, fmaf(a.y, b.y, fmaf(a.x, b.x, c))));
}

MUTLASS_DEVICE void vfma(float4& d, float4 const& a, float b, float c) {
#if defined(MATE_FLAT_SIMD_MATH_ENABLED)
  d = ::fma(b, a, c);
#else
  d = make_float4(a.x * b + c, a.y * b + c, a.z * b + c, a.w * b + c);
#endif
}

MUTLASS_DEVICE float4 vfma(float4 const& a, float b, float c) {
  float4 d;
  vfma(d, a, b, c);
  return d;
}

}  // namespace mate::flat::simd

namespace mate::flat::simd {

// Packed4 helpers widen/narrow four contiguous 16-bit elements through float4.
MUTLASS_DEVICE float4 load_packed4_as_float4(mutlass::bfloat16_t const* ptr) {
  auto packed = *reinterpret_cast<__mt_bfloat164 const*>(ptr);
  return __bfloat1642float4(packed);
}

MUTLASS_DEVICE void store_float4_to_packed4_rn(mutlass::bfloat16_t* ptr, float4 value) {
  auto packed                             = __float42bfloat164_rn(value);
  *reinterpret_cast<__mt_bfloat164*>(ptr) = packed;
}

MUTLASS_DEVICE float4 load_packed4_as_float4(mutlass::half_t const* ptr) {
  auto packed = *reinterpret_cast<__half4 const*>(ptr);
  return __half42float4(packed);
}

MUTLASS_DEVICE void store_float4_to_packed4_rn(mutlass::half_t* ptr, float4 value) {
  auto packed                      = __float42half4_rn(value);
  *reinterpret_cast<__half4*>(ptr) = packed;
}

MUTLASS_DEVICE float4 load_float4(float const* ptr) {
  return *reinterpret_cast<float4 const*>(ptr);
}

MUTLASS_DEVICE void store_float4(float* ptr, float4 value) {
  *reinterpret_cast<float4*>(ptr) = value;
}

}  // namespace mate::flat::simd
