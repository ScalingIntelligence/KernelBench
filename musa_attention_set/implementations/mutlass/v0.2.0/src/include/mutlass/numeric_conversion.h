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
/*!
    \file
    \brief Boost-like numeric conversion operator for MUTLASS numeric types
*/

#pragma once

#if !defined(__MUSACC_RTC__)
#include <cfenv>
#endif

#include "mutlass/mutlass.h"
#include "mutlass/numeric_types.h"
#include "mutlass/transform/thread/unary_op.h"

#include "mutlass/array.h"
#include "mutlass/half.h"

namespace mutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Floating-point rounding style similare to Standard Library's formats but supporting
/// additional rounding options.
enum class FloatRoundStyle {
  round_indeterminate,          ///< rounding mode unknown
  round_toward_zero,            ///< round toward zero
  round_to_nearest,             ///< round to nearest even
  round_to_nearest_satfinite,   ///< round to nearest even, capping value to min and max of destination type
  round_toward_infinity,        ///< round toward infinity
  round_toward_neg_infinity,    ///< round toward negative infinity
  round_half_ulp_truncate,      ///< add 0.5ulp to integer representation then round toward zero
  round_half_ulp_trunc_dntz     ///< like round_half_ulp_truncate, except denorms are rounded *toward* zero
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
struct NumericConverter {

  using result_type = T;
  using source_type = S;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<result_type>(s);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> int32_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__MUSA_ARCH__)
template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    return __float2int_rn(s);
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    return __float2int_rz(s);
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<float, int32_t, FloatRoundStyle::round_to_nearest> {

  using result_type = float;
  using source_type = int32_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    return __int2float_rn(s);
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<float, int32_t, FloatRoundStyle::round_toward_zero> {

  using result_type = float;
  using source_type = int32_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    return __int2float_rz(s);
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};



#elif !defined(__MUSACC_RTC__)

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TONEAREST);
    return (result_type)std::nearbyint(s);
  }

  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TOWARDZERO);
    return (result_type)std::nearbyint(s);
  }

  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> int8_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__MUSA_ARCH__)
template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    return __float2char_rn_sat(s);

  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    return __float2char_rz_sat(s);

  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<float, int8_t, FloatRoundStyle::round_to_nearest> {

  using result_type = float;
  using source_type = int8_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    int32_t result;
    result = static_cast<int32_t>(s);
    return __int2float_rn(result);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<float, int8_t, FloatRoundStyle::round_toward_zero> {

  using result_type = float;
  using source_type = int8_t;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    int32_t result;
    result = static_cast<int32_t>(s);
    return __int2float_rz(result);
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> uint8_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct NumericConverter<uint8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = uint8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    return __musa_f2uc_rn_sat(s);

  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<uint8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = uint8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    return __musa_f2uc_rz_sat(s);

  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<float, uint8_t, FloatRoundStyle::round_to_nearest> {

  using result_type = float;
  using source_type = uint8_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    int32_t result;
    result = static_cast<uint32_t>(s);
    return __uint2float_rn(result);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<float, uint8_t, FloatRoundStyle::round_toward_zero> {

  using result_type = float;
  using source_type = uint8_t;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    int32_t result;
    result = static_cast<uint32_t>(s);
    return __uint2float_rz(result);
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#elif !defined(__MUSACC_RTC__)

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TONEAREST);
    int32_t intermediate = (int32_t)std::nearbyint(s);

    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<int8_t>::lowest());

    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<int8_t>::max());

    return static_cast<result_type>(intermediate);
  }

  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TOWARDZERO);
    int32_t intermediate = (int32_t)std::nearbyint(s);

    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<int8_t>::lowest());

    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<int8_t>::max());

    return static_cast<result_type>(intermediate);
  }

  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= mutlass::half_t
template <typename T, FloatRoundStyle Round>
struct NumericConverter<T, T, Round> {

  using result_type = T;
  using source_type = T;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return s;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> mutlass::half_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= mutlass::half_t
template <FloatRoundStyle Round>
struct NumericConverter<float, mutlass::half_t, Round> {

  using result_type = float;
  using source_type = mutlass::half_t;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<float>(s);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<mutlass::half_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = mutlass::half_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    
    result_type result = static_cast<mutlass::half_t>(s);

    return result;

  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<mutlass::half_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = mutlass::half_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & flt) {
    return mutlass::half_t(__float2half_rz(flt));
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for int32_t <=> mutlass::half_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////
#if defined(__MUSA_ARCH__)
/// Partial specialization for int32_t <= mutlass::half_t
template <FloatRoundStyle Round>
struct NumericConverter<int32_t, mutlass::half_t, Round> {

  using result_type = int32_t;
  using source_type = mutlass::half_t;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<int32_t>(s);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<int32_t, mutlass::half_t, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = mutlass::half_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    return __half2int_rn(s.to_half());
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<int32_t, mutlass::half_t, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = mutlass::half_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    return __half2int_rz(s.to_half());
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for int32_t => mutlass::half_t
template <FloatRoundStyle Round>
struct NumericConverter<mutlass::half_t, int32_t, Round> {

  using result_type = mutlass::half_t;
  using source_type = int32_t;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<mutlass::half_t>(s);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<mutlass::half_t, int32_t, FloatRoundStyle::round_to_nearest> {

  using result_type = mutlass::half_t;
  using source_type = int32_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return mutlass::half_t(__int2half_rn(s));
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<mutlass::half_t, int32_t, FloatRoundStyle::round_toward_zero> {

  using result_type = mutlass::half_t;
  using source_type = int32_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    return mutlass::half_t(__int2half_rz(s));
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for uint32_t <=> mutlass::half_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for uint32_t <= mutlass::half_t
template <FloatRoundStyle Round>
struct NumericConverter<uint32_t, mutlass::half_t, Round> {

  using result_type = uint32_t;
  using source_type = mutlass::half_t;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<uint32_t>(s);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<uint32_t, mutlass::half_t, FloatRoundStyle::round_to_nearest> {

  using result_type = uint32_t;
  using source_type = mutlass::half_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    return __half2uint_rn(s.to_half());
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<uint32_t, mutlass::half_t, FloatRoundStyle::round_toward_zero> {

  using result_type = uint32_t;
  using source_type = mutlass::half_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return __half2uint_rz(s.to_half());
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for uint32_t => mutlass::half_t
template <FloatRoundStyle Round>
struct NumericConverter<mutlass::half_t, uint32_t, Round> {

  using result_type = mutlass::half_t;
  using source_type = uint32_t;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<mutlass::half_t>(s);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<mutlass::half_t, uint32_t , FloatRoundStyle::round_to_nearest> {

  using result_type = mutlass::half_t;
  using source_type = uint32_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return mutlass::half_t(__uint2half_rn(s));
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<mutlass::half_t, uint32_t, FloatRoundStyle::round_toward_zero> {

  using result_type = mutlass::half_t;
  using source_type = uint32_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    return mutlass::half_t(__uint2half_rz(s));
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> mutlass::bfloat16_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 220)
/// Partial specialization for float <= mutlass::bfloat16_t
template <FloatRoundStyle Round>
struct NumericConverter<float, mutlass::bfloat16_t, Round> {

  using result_type = float;
  using source_type = mutlass::bfloat16_t;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<float>(s);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<mutlass::bfloat16_t, float, FloatRoundStyle::round_to_nearest> {
  using result_type = mutlass::bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return static_cast<mutlass::bfloat16_t>(s);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<mutlass::bfloat16_t, float, FloatRoundStyle::round_half_ulp_truncate> {
  using result_type = mutlass::bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_truncate;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    uint32_t x32 = reinterpret_cast<uint32_t const &>(s);

    #if defined(__MUSA_ARCH__)
    if (::isfinite(s)) {
      x32 += 0x8000;
    }
    #else
    if (std::isfinite(s)) {
      x32 += 0x8000;
    }
    #endif

    uint16_t x16 = uint16_t((x32 >> 16) & 0xffff);
    return mutlass::bfloat16_t::bitcast(x16);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<mutlass::bfloat16_t, float, FloatRoundStyle::round_toward_zero> {
  using result_type = mutlass::bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return mutlass::bfloat16_t(__float2bfloat16_rz(s));

  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for int32_t <=> mutlass::bfloat16_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for int32_t <= mutlass::bfloat16_t
template <FloatRoundStyle Round>
struct NumericConverter<int32_t, mutlass::bfloat16_t, Round> {

  using result_type = int32_t;
  using source_type = mutlass::bfloat16_t;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<int32_t>(s);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<int32_t, mutlass::bfloat16_t, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = mutlass::bfloat16_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    return __bfloat162int_rn(__mt_bfloat16(s));
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<int32_t, mutlass::bfloat16_t, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = mutlass::bfloat16_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return __bfloat162int_rz(__mt_bfloat16(s));
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for uint32_t <=> mutlass::bfloat16_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for uint32_t <= mutlass::bfloat16_t
template <FloatRoundStyle Round>
struct NumericConverter<uint32_t, mutlass::bfloat16_t, Round> {

  using result_type = uint32_t;
  using source_type = mutlass::bfloat16_t;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<uint32_t>(s);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<uint32_t, mutlass::bfloat16_t, FloatRoundStyle::round_to_nearest> {

  using result_type = uint32_t;
  using source_type = mutlass::bfloat16_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_DEVICE
  static result_type convert(source_type const & s) {
    return __bfloat162uint_rn(__mt_bfloat16(s));
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<uint32_t, mutlass::bfloat16_t, FloatRoundStyle::round_toward_zero> {

  using result_type = uint32_t;
  using source_type = mutlass::bfloat16_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return __bfloat162uint_rz(__mt_bfloat16(s));
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for uint32_t => mutlass::bfloat16_t
template <FloatRoundStyle Round>
struct NumericConverter<mutlass::bfloat16_t, uint32_t, Round> {

  using result_type = mutlass::bfloat16_t;
  using source_type = uint32_t;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<mutlass::bfloat16_t>(s);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for bfloat16_t <=> int8_t
template <typename S, FloatRoundStyle Round>
struct NumericConverter<mutlass::bfloat16_t, S, Round> {

  using result_type = mutlass::bfloat16_t;
  using source_type = S;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<mutlass::bfloat16_t>(s);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <typename S>
struct NumericConverter< mutlass::bfloat16_t, S, FloatRoundStyle::round_to_nearest> {

  using result_type = mutlass::bfloat16_t;
  using source_type = S;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    NumericConverter<float, S, FloatRoundStyle::round_to_nearest> convert_op1;
    NumericConverter<mutlass::bfloat16_t, float, FloatRoundStyle::round_to_nearest> convert_op2;

    return convert_op2(convert_op1(s));
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <typename S>
struct NumericConverter<mutlass::bfloat16_t, S, FloatRoundStyle::round_toward_zero> {

  using result_type = mutlass::bfloat16_t;
  using source_type = S;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    NumericConverter<float, S, FloatRoundStyle::round_toward_zero> convert_op1;
    NumericConverter<mutlass::bfloat16_t, float, FloatRoundStyle::round_toward_zero> convert_op2;

    return convert_op2(convert_op1(s));
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> mutlass::tfloat32_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= mutlass::tfloat32_t
template <FloatRoundStyle Round>
struct NumericConverter<float, mutlass::tfloat32_t, Round> {

  using result_type = float;
  using source_type = mutlass::tfloat32_t;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<float>(s);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<mutlass::tfloat32_t, float, FloatRoundStyle::round_to_nearest> {
  using result_type = mutlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    unsigned storage = reinterpret_cast<unsigned const &>(s);

#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 220
    return __musa_float_to_tf32(s);
#else
    if ((storage & 0x7f800000) != 0x7f800000) {

      bool mantissa_bit = ((storage & (1 << 13)) != 0);
      bool round_bit = ((storage & (1 << 12)) != 0);
      bool sticky_bit = ((storage & ((1 << 12) - 1)) != 0);

      if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
        storage += uint32_t(1 << 13);
      }

      // Note, the following is intentionally commented out. TF32
      // does not define the low order bits, so they may be left in
      // an undefined state.
      //
      // By not truncating these bit explicitly, we avoid an extra logical
      // operation.
      //
      // TF32 may be implicitly converted to float by performing this
      // operation as needed.
      //
      // storage = (storage & ~0x1fff);
    }
    else if (storage & ~0xff800000) {
      storage = 0x7fffffff;
    }
#endif

    return mutlass::tfloat32_t::bitcast(storage);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<mutlass::tfloat32_t, float, FloatRoundStyle::round_half_ulp_truncate> {
  using result_type = mutlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_truncate;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return mutlass::tfloat32_t::round_half_ulp_truncate(s);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// This rounding operation is similar to half_ulp_truncate except it rounds denorms toward zero.
/// It avoids predicated code, though it requires a temporary register.
template <>
struct NumericConverter<mutlass::tfloat32_t, float, FloatRoundStyle::round_half_ulp_trunc_dntz> {
  using result_type = mutlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_trunc_dntz;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    unsigned y = reinterpret_cast<unsigned const &>(s);
    y = y & 0xff800000;
    float d = reinterpret_cast<float const &>(y);
    float z = d / float(1 << 11) + s;

    return reinterpret_cast<result_type const &>(z);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<mutlass::tfloat32_t, float, FloatRoundStyle::round_toward_zero> {
  using result_type = mutlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    uint32_t x = reinterpret_cast<uint32_t const &>(s);
    return mutlass::tfloat32_t::bitcast(x & 0xffffe000);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion operator for float to mutlass::tfloat32_t big and small values
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  FloatRoundStyle RoundBig = FloatRoundStyle::round_toward_zero,
  FloatRoundStyle RoundSmall = FloatRoundStyle::round_half_ulp_truncate
>
struct NumericConverterFastF32 {

  // result_type holds big mutlass::tfloat32_t at idx(0) and small mutlass::tfloat32_t at idx(1)
  using result_type = Array<mutlass::tfloat32_t, 2>;

  // source data type
  using source_type = float;

  // rounding styles for big and small part
  static FloatRoundStyle const kRoundBig = RoundBig;
  static FloatRoundStyle const kRoundSmall = RoundSmall;

  MUTLASS_HOST_DEVICE
    static result_type convert(source_type const & source) {

    result_type result;
    NumericConverter<mutlass::tfloat32_t, float, kRoundBig> convert_big_;
    NumericConverter<mutlass::tfloat32_t, float, kRoundSmall> convert_small_;

    // convert and fill mutlass::tfloat32_t big at idx 0
    result[0] = convert_big_(source);

    // convert and fill mutlass::tfloat32_t small at idx 1
    result[1] = convert_small_(source - static_cast<float>(result[0]));

    return result;
  }

  MUTLASS_HOST_DEVICE
    result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#endif //defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 220)
/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion and Clamp operator for Integers
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S
>
struct NumericConverterClamp {

  using result_type = T;
  using source_type = S;

  MUTLASS_HOST_DEVICE
    static result_type convert(source_type const & s) {
    NumericConverter<result_type, source_type> convert_op;
    result_type const kClamp_max = platform::numeric_limits<result_type>::max();
    result_type const kClamp_min = platform::numeric_limits<result_type>::lowest();
    if (s < (source_type)kClamp_min)
      return kClamp_min;
    if (s > (source_type)kClamp_max)
      return kClamp_max;
    return convert_op(s);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

// This converter is needed to enable mutlass::half_t output types when using int32_t accumulators.
// Since floating-point types do not require a clamp, this converter simply casts from
// the source type to mutlass::half_t.
template <
  typename S
>
struct NumericConverterClamp<mutlass::half_t, S> {

  using result_type = mutlass::half_t;
  using source_type = S;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const &source) {
    return static_cast<mutlass::half_t>(source);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion operator for Array
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conversion operator for Array
template <
  typename T,
  typename S,
  int N,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename Transform = mutlass::transform::thread::UnaryTransform::Identity
>
struct NumericArrayConverter {

  using result_type = Array<T, N>;
  using source_type = Array<S, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, mutlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, mutlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result;
    NumericConverter<T, S, Round> convert_;

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      if (platform::is_same<Transform, mutlass::transform::thread::UnaryTransform::Identity>::value) {
        result[i] = convert_(s[i]);
      } else { // conjugate
        result[i] = conj(convert_(s[i]));
      }
    }

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  typename T,
  int N,
  FloatRoundStyle Round,
  typename Transform
>
struct NumericArrayConverter<T, T, N, Round, Transform> {

  using result_type = Array<T, N>;
  using source_type = Array<T, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, mutlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, mutlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const &source) {
    if (platform::is_same<Transform, mutlass::transform::thread::UnaryTransform::Identity>::value) {
      return source;
    } else {
      result_type result;
      for (int i = 0; i < N; ++i) {
          result[i] = conj(source[i]);
      }
      return result;
    }
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<half, 2> <= Array<float, 2>, round to nearest
template <>
struct NumericArrayConverter<mutlass::half_t, float, 2, FloatRoundStyle::round_to_nearest> {

  using result_type = Array<mutlass::half_t, 2>;
  using source_type = Array<float, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    Array<mutlass::half_t, 2> result;
    reinterpret_cast<__half2 &>(result) = __float22half2_rn(reinterpret_cast<float2 const &>(source));
    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, 2> <= Array<mutlass::half_t, 2>, round to nearest
template <FloatRoundStyle Round>
struct NumericArrayConverter<float, mutlass::half_t, 2, Round> {

  using result_type = Array<float, 2>;
  using source_type = Array<mutlass::half_t, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    float2 result2 = __half22float2(reinterpret_cast<__half2 const &>(source));
    return {
      float{result2.x},
      float{result2.y}
    };
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<half> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<mutlass::half_t, float, N, Round> {

  using result_type = Array<mutlass::half_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<mutlass::half_t, float, 2, Round> convert_vector_;
    NumericConverter<mutlass::half_t, float, Round> convert_element_;

    result_type result;

    Array<mutlass::half_t, 2> *result_ptr = reinterpret_cast<Array<mutlass::half_t, 2> *>(&result);
    Array<float, 2> const *source_ptr = reinterpret_cast<Array<float, 2> const *>(&source);

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/// Partial specialization for Array<half> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, mutlass::half_t, N, Round> {

  using result_type = Array<float, N>;
  using source_type = Array<mutlass::half_t, N>;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<float, mutlass::half_t, 2, Round> convert_vector_;
    NumericConverter<float, mutlass::half_t, Round> convert_element_;

    result_type result;

    Array<float, 2> *result_ptr = reinterpret_cast<Array<float, 2> *>(&result);
    Array<mutlass::half_t, 2> const *source_ptr = reinterpret_cast<Array<mutlass::half_t, 2> const *>(&source);

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 220)
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<mutlass::bfloat16_t, 2> <= Array<float, 2>, round to nearest
template <>
struct NumericArrayConverter<mutlass::bfloat16_t, float, 2, FloatRoundStyle::round_to_nearest> {

  using result_type = Array<mutlass::bfloat16_t, 2>;
  using source_type = Array<float, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    Array<mutlass::bfloat16_t, 2> result;

    reinterpret_cast<__mt_bfloat162 &>(result) = __float22bfloat162_rn(reinterpret_cast<float2 const &>(source));

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<mutlass::bfloat16_t> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<mutlass::bfloat16_t, float, N, Round> {

  using result_type = Array<mutlass::bfloat16_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<mutlass::bfloat16_t, float, 2, Round> convert_vector_;
    NumericConverter<mutlass::bfloat16_t, float, Round> convert_element_;

    result_type result;

    Array<mutlass::bfloat16_t, 2> *result_ptr = reinterpret_cast<Array<mutlass::bfloat16_t, 2> *>(&result);
    Array<float, 2> const *source_ptr = reinterpret_cast<Array<float, 2> const *>(&source);

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};
#endif // if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 220)

/// Partial specialization for Array<uint8_t, 1> <= Array<int, 1>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, 1, Round> {

  using result_type = Array<uint8_t, 1>;
  using source_type = Array<int, 1>;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericConverter<uint8_t, int, Round> convert_element_;

    result_type result;

    result[0] = convert_element_(source[0]);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t> <= Array<float>
/// Conversion is performed with saturation regardless of setting of
/// the `Round` template parameter.
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, float, 1, Round> {

  using result_type = Array<int8_t, 1>;
  using source_type = Array<float, 1>;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    // Convert to int to int8_t
    NumericConverter<int8_t, float, Round> destination_converter;
    result_type result;
    result[0] = destination_converter(source[0]);
    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

// To convert a FP32 to Int that has less than 32 bits, we need to convert it to int32 first.
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayFP32ToIntConverter {

  using result_type = Array<T, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::numeric_limits<T>::is_integer, "the dest type has to be int.");

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    // Convert float to int
    Array<int32_t, N> temporary;

    NumericArrayConverter<int32_t, float, N, Round> compute_converter;
    temporary = compute_converter(source);

    // Convert to int to int8_t
    NumericArrayConverter<T, int32_t, N, Round> destination_converter;
    return destination_converter(temporary);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, float, N, Round> {

  using result_type = Array<int8_t, N>;
  using source_type = Array<float, N>;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<int8_t, N, Round> converter;
    return converter(source);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, float, N, Round> {

  using result_type = Array<uint8_t, N>;
  using source_type = Array<float, N>;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<uint8_t, N, Round> converter;
    return converter(source);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int4b_t, float, N, Round> {

  using result_type = Array<int4b_t, N>;
  using source_type = Array<float, N>;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<int4b_t, N, Round> converter;
    return converter(source);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint4b_t, float, N, Round> {

  using result_type = Array<uint4b_t, N>;
  using source_type = Array<float, N>;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<uint4b_t, N, Round> converter;
    return converter(source);
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// FastNumericArrayConverter only works when the source is within center range.
/// Conversion operator for Array.  See the comments before
/// FastLinearCombinationClamp.
template <typename T, typename S, int N,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          typename Enable = void>
struct FastNumericArrayConverter {
  using result_type = Array<T, N>;
  using source_type = Array<S, N>;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_DEVICE
  static result_type convert(source_type const &s) {
    NumericArrayConverter<T, S, N, Round> convert_;

    return convert_(s);
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/// Partial specialization for Array<float> <= Array<int>
template <int N, FloatRoundStyle Round>
struct FastNumericArrayConverter<float, int, N, Round> {
  using result_type = Array<float, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int tmp = source[i] + 1262485504 /*0x4B400000*/;
      result[i] = reinterpret_cast<float const &>(tmp) - 12582912.0f;
    }

    return result;
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/// Partial specialization for Array<int8_t, 4> <= Array<float, 4>
// template <FloatRoundStyle Round>
// struct FastNumericArrayConverter<int8_t, float, 4, Round> {
//   using result_type = Array<int8_t, 4>;
//   using source_type = Array<float, 4>;
//   static FloatRoundStyle const round_style = Round;

//   MUTLASS_DEVICE
//   static result_type convert(source_type const &source) {
//     Array<int32_t, 4> result;

//     MUTLASS_PRAGMA_UNROLL
//     for (int i = 0; i < 4; ++i) {
//       float tmp = source[i] + 12582912.0f;
//       result[i] = reinterpret_cast<int32_t const &>(tmp);
//     }

//     result[0] = __byte_perm(result[0], result[1], 0x40);
//     result[2] = __byte_perm(result[2], result[3], 0x40);
//     result[0] = __byte_perm(result[0], result[2], 0x5410);

//     return reinterpret_cast<result_type const &>(result[0]);
//   }

//   MUTLASS_DEVICE
//   result_type operator()(source_type const &s) const { return convert(s); }
// };

/// Partial specialization for Array<int8_t> <= Array<float>
template <int N, FloatRoundStyle Round>
struct FastNumericArrayConverter<int8_t, float, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = Array<int8_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  MUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    FastNumericArrayConverter<int8_t, float, 4, Round> convert_vector_;

    result_type result;

    Array<int8_t, 4> *result_ptr =
        reinterpret_cast<Array<int8_t, 4> *>(&result);
    Array<float, 4> const *source_ptr =
        reinterpret_cast<Array<float, 4> const *>(&source);

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines preferred rounding mode for a pair of types
template <typename T, typename S>
struct PreferredRoundingMode {
  static FloatRoundStyle const kRound = FloatRoundStyle::round_to_nearest;
};

#if defined(__MUSA_ARCH__)
/// Defines preferred rounding mode for a pair of types
template <>
struct PreferredRoundingMode<mutlass::tfloat32_t, float> {
  static FloatRoundStyle const kRound = FloatRoundStyle::round_half_ulp_truncate;
};
#endif // defined(__MUSA_ARCH__)

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Packs predicates into an array.
template <int N>
struct PackPredicates {
  using result_type = Array<uint1b_t, N>;

  static_assert(!(N % 4), "Must pack predicates in a count that is a multiple of 4");

  MUTLASS_HOST_DEVICE
  result_type operator()(bool const predicates[]) {

    result_type packed;
    packed.clear();

    int const kWordSize = 8;
    uint8_t *bytes = reinterpret_cast<uint8_t *>(packed.data());

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int word_idx = (i / kWordSize);
      int bit_idx = (i % kWordSize);

      uint8_t mask = static_cast<uint8_t>((predicates[i] ? 1u : 0u) << bit_idx);
      bytes[word_idx] = (bytes[word_idx] | mask);
    }
    return packed;
  }
};

/// Packs predicates into an array
template <int N>
struct UnpackPredicates {
  using result_type = Array<uint1b_t, N>;

  static_assert(!(N % 4), "Must unpack predicates in a count that is a multiple of 4");

  MUTLASS_HOST_DEVICE
  void operator()(bool predicates[], result_type const &packed) {

    int const kWordSize = 8;
    uint8_t const *bytes = reinterpret_cast<uint8_t const *>(packed.data());

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int word_idx = (i / kWordSize);
      int bit_idx = (i % kWordSize);

      predicates[i] = bool((bytes[word_idx] >> bit_idx) & 0x1);
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
