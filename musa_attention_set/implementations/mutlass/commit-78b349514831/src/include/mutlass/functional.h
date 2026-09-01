  /***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Define basic numeric operators

    This is inspired by the Standard Library's <functional> header.
*/
#pragma once

#include "mutlass/mutlass.h"
#include "mutlass/numeric_types.h"

#include <musa_runtime.h>

#ifdef _MSC_VER
// Provides support for alternate operators such as 'and', 'or', ...
#include <iso646.h>
#endif // _MSC_VER

namespace mutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct absolute_value_op {
  MUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return abs(lhs);
  }
};

template <>
struct absolute_value_op<float> {
  MUTLASS_HOST_DEVICE
  float operator()(float lhs) const { return fabs(lhs); }
};

template <typename T>
struct plus {
  MUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs += rhs;
    return lhs;
  }
};

template <typename T>
struct minus {
  MUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs -= rhs;
    return lhs;
  }
};

template <typename T>
struct multiplies {
  MUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs *= rhs;
    return lhs;
  }
};

template <typename T>
struct scale {
  T const scaling_factor_;

  MUTLASS_HOST_DEVICE
  scale(float scaling_factor) : scaling_factor_(scaling_factor) {
  }

  T operator()(T const &rhs) const {
    T result = rhs * scaling_factor_;
    return result;
  }
};

#if defined(__MUSA_ARCH__)
/// Partial specializations needed when __MUSA_NO_HALF2_OPERATORS__ is set
template<>
struct plus<__half2> {
  MUTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hadd2(lhs, rhs);
  }
};

template<>
struct minus<__half2> {
  MUTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hsub2(lhs, rhs);
  }
};

template<>
struct multiplies<__half2> {
  MUTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hmul2(lhs, rhs);
  }
};

/// Partial specializations needed when __MUSA_NO_HALF_OPERATORS__ is set
template<>
struct plus<__half> {
  MUTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hadd(lhs, rhs);
  }
};

template<>
struct minus<__half> {
  MUTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hsub(lhs, rhs);
  }
};

template<>
struct multiplies<__half> {
  MUTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hmul(lhs, rhs);
  }
};
#endif // defined(__MUSA_ARCH__)


/// Squares with optional conversion
template <typename T, typename Output = T>
struct square {
  MUTLASS_HOST_DEVICE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Returns the magnitude squared of an element.
template <typename T, typename Output = T>
struct magnitude_squared {
  MUTLASS_HOST_DEVICE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T>
struct square_difference {
  MUTLASS_HOST_DEVICE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T>
struct magnitude_squared_difference {
  MUTLASS_HOST_DEVICE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

// Computes the reciprocal square root
template <typename T>
struct inverse_square_root;

template <>
struct inverse_square_root<float> {
  MUTLASS_HOST_DEVICE
  float operator()(float const &lhs) const {
#if defined(__MUSA_ARCH__)
    return rsqrtf(lhs);
#else
    return 1.f / std::sqrt(lhs);
#endif
  }
};

template <>
struct inverse_square_root<half_t> {
  MUTLASS_HOST_DEVICE
  half_t operator()(half_t const &lhs) const {
#if defined(__MUSA_ARCH__)
    auto result = hrsqrt(reinterpret_cast<__half const &>(lhs));
    return reinterpret_cast<half_t const &>(result);
#else
    return half_t(1.f / std::sqrt(half_t::convert(lhs)));
#endif
  }
};

/// Divides
template <typename T>
struct divides {
  MUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs /= rhs;
    return lhs;
  }
};

/// reciprocal_approximate
template <typename T>
struct reciprocal_approximate {
  MUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return divides<T>{}(T(1), lhs);
  }
};

template <>
struct reciprocal_approximate <float> {
  MUTLASS_HOST_DEVICE
  float operator()(float lhs) const {
    float ret;
      ret = 1.0f / lhs;
    return ret;
  }
};

/// Negate
template <typename T>
struct negate {
  MUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return -lhs;
  }
};

/// Greater equal
template <typename T>
struct greater_equal {
  MUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs >= rhs);
  }
};

/// Greater
template <typename T>
struct greater {
  MUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs > rhs);
  }
};

/// Less equal
template <typename T>
struct less_equal {
  MUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs <= rhs);
  }
};

/// Less
template <typename T>
struct less {
  MUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs < rhs);
  }
};

template <typename T, bool PropagateNaN = false>
struct maximum {
  MUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    if constexpr (PropagateNaN && mutlass::platform::is_floating_point<T>::value) {
      using MUTLASS_CMATH_NAMESPACE :: isnan;

      // Call isnan unqualified, so argument-dependent lookup (ADL)
      // will find overloads such as mutlass::isnan(half_t).
      // Calling ::isnan or std::isnan directly would force
      // implicit conversions to float of custom number types
      // in the mutlass namespace (e.g., mutlass::half_t).
      return lhs > rhs || isnan(lhs) ? lhs : rhs;
    }
    else {
      return (lhs < rhs ? rhs : lhs);
    }
  }
};

// This is a subclass and not an alias
// in order to work around a known Clang issue,
// where a template template parameter with one template parameter
// does not match classes that take multiple template parameters
// but have defaults for all but the first.
template<typename T>
struct maximum_with_default_nan_propagation : public maximum<T>
{};

// Maximum with nan propagation
// To propagate NANs, the "max" of a two element that contains NaNs should also return a NaN
template <typename T>
struct maximum<T, true> {
  MUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
#if defined(__MUSA_ARCH__)
    return lhs > rhs or isnan(lhs) ? lhs : rhs;
#else
    return lhs > rhs or std::isnan(lhs) ? lhs : rhs;
#endif
  }
};

template <>
struct maximum<float, false> {
  MUTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const {
    return fmaxf(lhs, rhs);
  }
};

template <>
struct maximum<float, true> {
  MUTLASS_HOST_DEVICE
  float operator()(float const lhs, float const rhs) const {
    float res;
#if defined(__MUSA_ARCH__)
    res = lhs > rhs or isnan(lhs) ? lhs : rhs;
#else
    res = lhs > rhs or std::isnan(lhs) ? lhs : rhs;
#endif
    return res;
  }
};

// This is a subclass and not an alias
// in order to work around a known Clang issue,
// where a template template parameter with one template parameter
// does not match classes that take multiple template parameters
// but have defaults for all but the first.
template <typename T>
struct maximum_with_nan_propagation : maximum<T, true>
{};

// This alias exists for backwards compatibility only.
// Please use the correctly spelled class template above.
template <typename T>
using maximum_with_nan_propogation = maximum_with_nan_propagation<T>;

template <typename T, bool PropagateNaN = false>
struct minimum{
  MUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    if constexpr (PropagateNaN && mutlass::platform::is_floating_point<T>::value) {
      using MUTLASS_CMATH_NAMESPACE :: isnan;

      return lhs < rhs || isnan(lhs) ? lhs : rhs;
    }
    else {
      return (rhs < lhs ? rhs : lhs);
    }
  }
};

template <>
struct minimum<float, false> {
  MUTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const {
    return fminf(lhs, rhs);
  }
};

template <typename T>
struct minimum_with_nan_propagation : minimum<T, true> 
{};

template <typename T, bool PropagateNaN = false>
struct maximum_absolute_value {
  MUTLASS_HOST_DEVICE
  float operator()(T const &lhs, T const &rhs) const {
    absolute_value_op<T> abs_op;
    maximum<T, PropagateNaN> max_op;

    return max_op(abs_op(lhs), abs_op(rhs));
  }
};

// assumes the left operand is already an absolute value
template <typename T, bool PropagateNaN = false>
struct maximum_absolute_value_reduction {
  MUTLASS_HOST_DEVICE
  float operator()(T const &lhs, T const &rhs) const {
    absolute_value_op<T> abs_op;
    maximum<T, PropagateNaN> max_op;

    return max_op(lhs, abs_op(rhs));
  }
};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add {
  MUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    return C(a) * C(b) + c;
  }
};

template <typename T>
struct square_and_plus {
  MUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    multiply_add<T> multiply_add_op;
    return multiply_add_op(rhs, rhs, lhs);
  }
};

// Fused multiply-add that takes exactly one template parameter.
// This is useful for working around a known Clang issue,
// where a template template parameter with one template parameter
// does not match classes that take multiple template parameters
// but have defaults for all but the first.
template <typename A>
struct homogeneous_multiply_add : public multiply_add<A, A, A>
{};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add_relu0 {
  MUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    maximum<C> mx;
    return mx(C(a) * C(b) + c, C(0));
  }
};

/// Fused multiply-add
template <typename T>
struct and_add {
  MUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a & b) + c);
  }
};


/// Fused multiply-add
template <typename T>
struct xor_add {
  MUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a ^ b) + c);
  }
};

template <typename T>
struct conjugate {
  MUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return a;
  }
};

template <typename T>
struct first {
  MUTLASS_HOST_DEVICE
  T operator()(T const & first, T const &...) const {
    return first;
  }
  MUTLASS_HOST_DEVICE
  T operator()(T const & first) const {
    return first;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct logical_and {
  MUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return ((static_cast<bool>(a) && static_cast<bool>(b)) ? T(1) : T());
  }
};

template <typename T>
struct logical_or {
  MUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return ((static_cast<bool>(a) || static_cast<bool>(b)) ? T(1) : T());
  }
};

template <typename T>
struct logical_not {
  MUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return T(!(a));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct bit_and {
  MUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a & b;
  }
};

template <typename T>
struct bit_or {
  MUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a | b;
  }
};

template <typename T>
struct bit_not {
  MUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return ~a;
  }
};

template <typename T>
struct bit_xor {
  MUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a ^ b;
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Atomic reductions

template <typename T>
struct atomic_add
{
  MUTLASS_DEVICE
  void operator()(T *ptr, const T &data)
  {
#if defined(__MUSA_ARCH__)
    atomicAdd(ptr, data);
#else
    MUTLASS_UNUSED(ptr);
    MUTLASS_UNUSED(data);
    MUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

template <typename T>
struct atomic_maximum {
  MUTLASS_DEVICE
  T operator()(T *ptr, T value) const {
#if defined(__MUSA_ARCH__)
    return atomicMax(ptr, value);
#else
    MUTLASS_UNUSED(ptr);
    MUTLASS_UNUSED(value);
    MUTLASS_NOT_IMPLEMENTED();
    return 0;
#endif
  }
};

template <>
struct atomic_maximum<float> {
  MUTLASS_DEVICE
  float operator()(float *ptr, float value) const {
#if defined(__MUSA_ARCH__)
    return !signbit(value) ?
      __int_as_float(atomicMax((int*)ptr, __float_as_int(value))) :
      __uint_as_float(atomicMin((unsigned int*)ptr, __float_as_uint(value)));
#else
    MUTLASS_UNUSED(ptr);
    MUTLASS_UNUSED(value);
    MUTLASS_NOT_IMPLEMENTED();
    return 0;
#endif
  }
};

// is_atomic
template <class Fn>
struct is_atomic : platform::false_type {};
template <class T>
struct is_atomic<atomic_add<T>> : platform::true_type {};
template <class T>
struct is_atomic<atomic_maximum<T>> : platform::true_type {};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
