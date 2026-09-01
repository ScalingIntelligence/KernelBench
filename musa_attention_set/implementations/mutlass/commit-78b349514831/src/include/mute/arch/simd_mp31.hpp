#pragma once

#include <mute/config.hpp>
#include <mute/numeric/real.hpp>
#include <mute/container/array.hpp>

#include <musa_burst.h>

#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310))
#define MUTE_ARCH_SIMD_MATH_ENABLED
#endif

namespace mute {

MUTE_HOST_DEVICE
void
add(float2      & c,
    float2 const& a,
    float2 const& b)
{
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  c = add(a, b);
#else
  add(c.x, a.x, b.x);
  add(c.y, a.y, b.y);
#endif
}

MUTE_HOST_DEVICE
void
add(float4      & c,
    float4 const& a,
    float4 const& b)
{
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  c = add(a, b);
#else
  add(c.x, a.x, b.x);
  add(c.y, a.y, b.y);
  add(c.z, a.z, b.z);
  add(c.w, a.w, b.w);
#endif
}

MUTE_HOST_DEVICE
void
max(float2      & c,
    float2 const& a,
    float2 const& b)
{
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  c = max(a, b);
#else
  max(c.x, a.x, b.x);
  max(c.y, a.y, b.y);
#endif
}


MUTE_HOST_DEVICE
void
max(float4      & c,
    float4 const& a,
    float4 const& b)
{
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  c = max(a, b);
#else
  max(c.x, a.x, b.x);
  max(c.y, a.y, b.y);
  max(c.z, a.z, b.z);
  max(c.w, a.w, b.w);
#endif
}

MUTE_HOST_DEVICE
void
fast_exp2(float2      & y,
          float2 const& x)
{
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  y = __exp2f(x);
#else
  y.x = exp2f(x.x);
  y.y = exp2f(x.y);
#endif
}

MUTE_HOST_DEVICE
void
fast_exp2(float4      & y,
          float4 const& x)
{
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  y = __exp2f(x);
#else
  y.x = exp2f(x.x);
  y.y = exp2f(x.y);
  y.z = exp2f(x.z);
  y.w = exp2f(x.w);
#endif
}

} // namespace mute
