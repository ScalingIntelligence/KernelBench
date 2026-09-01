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
    \brief Defines a proxy class for storing non-standard 16-bit floating point values with
          8 bits of exponent and 7 bit of mantissa.
*/

#pragma once

#if defined(__MUSACC_RTC__)
#include "mutlass/floating_point_mtrtc.h"
#else
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
#endif

#include <musa_bf16.h>
#include "mutlass/mutlass.h"
#include "mutlass/platform/platform.h"

namespace mutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Floating-point type with 8 bits of exponent and 7 bits of mantissa.
struct alignas(2) bfloat16_t {

  //
  // Data members
  //

  /// Storage type
  uint16_t storage;

  //
  // Methods
  //

  /// Constructs from an unsigned short
  MUTLASS_HOST_DEVICE
  static bfloat16_t bitcast(uint16_t x) {
    bfloat16_t h;
    h.storage = x;
    return h;
  }

private:
  struct from_32_bit_integer_t {};
  static constexpr from_32_bit_integer_t from_32_bit_integer{};

  template<class T>
  MUTLASS_HOST_DEVICE
  explicit bfloat16_t(from_32_bit_integer_t, T x) {
    static_assert(mutlass::platform::is_integral<T>::value && sizeof(T) == 4, "Requires 32-bit integer");

    float flt = static_cast<float>(x);
    uint32_t bits;

    #if defined(__MUSA_ARCH__)
    bits = reinterpret_cast<uint32_t &>(flt);
    #else
    std::memcpy(&bits, &flt, sizeof(bits));
    #endif

    storage = uint16_t(bits >> 16);
  }

public:
  /// Default constructor
  bfloat16_t() = default;

  /// Reinterpret cast from MUSA's bfloat16 type
  MUTLASS_HOST_DEVICE
  explicit bfloat16_t(__mt_bfloat16 const & x) {
    #if defined(__MUSA_ARCH__)
    storage = reinterpret_cast<uint16_t const &>(x);
    #else
    __mt_bfloat16_raw raw(x);
    std::memcpy(&storage, &raw.x, sizeof(storage));
    #endif
  }


  /// Floating-point conversion - round toward nearest
  MUTLASS_HOST_DEVICE
  explicit bfloat16_t(float x) {

    #if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 220)

    auto value = __float2bfloat16_rn(x);
    storage = *reinterpret_cast<uint16_t*>(&value);

    #else
    uint32_t bits;

    #if defined(__MUSA_ARCH__)
    bits = reinterpret_cast<uint32_t &>(x);
    #else
    std::memcpy(&bits, &x, sizeof(bits));
    #endif

    if ((bits & 0x7f800000) != 0x7f800000) {

      bool mantissa_bit = ((bits & (1 << 16)) != 0);
      bool round_bit = ((bits & (1 << 15)) != 0);
      bool sticky_bit = ((bits & ((1 << 15) - 1)) != 0);

      if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
        bits += uint32_t(1 << 16);
      }
    }
    else if (bits & ~0xff800000) {
      bits = 0x7fffffff;
    }

    storage = uint16_t((bits >> 16) & 0xffff);
    #endif
  }

  /// Floating-point conversion - round toward nearest
  MUTLASS_HOST_DEVICE
  explicit bfloat16_t(double x): bfloat16_t(float(x)) {

  }

  /// Integer conversion - round toward nearest
  MUTLASS_HOST_DEVICE
  explicit bfloat16_t(int x) : bfloat16_t(from_32_bit_integer, x) {}

  MUTLASS_HOST_DEVICE
  explicit bfloat16_t(uint32_t x) : bfloat16_t(from_32_bit_integer, x) {}

  /// Converts to float
  MUTLASS_HOST_DEVICE
  operator float() const {
    unsigned bits = (unsigned(storage) << 16);
    #if defined(__MUSA_ARCH__)
    return reinterpret_cast<float const &>(bits);
    #else
    float flt;
    std::memcpy(&flt, &bits, sizeof(flt));
    return flt;
    #endif
  }

  /// Converts to float
  MUTLASS_HOST_DEVICE
  explicit operator double() const {
    return double(float(*this));
  }

  /// Converts to int
  MUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(float(*this));
  }

  /// Casts to bool
  MUTLASS_HOST_DEVICE
  explicit operator bool() const {
    return (float(*this) != 0.0f);
  }

  /// Obtains raw bits
  MUTLASS_HOST_DEVICE
  uint16_t raw() const {
    return storage;
  }
    /// Returns the sign bit
  MUTLASS_HOST_DEVICE
  bool signbit() const {
    return ((raw() & 0x8000) != 0);
  }

  /// Returns the biased exponent
  MUTLASS_HOST_DEVICE
  int exponent_biased() const {
    return int((raw() >> 7) & 0x0ff);
  }

  /// Returns the unbiased exponent
  MUTLASS_HOST_DEVICE
  int exponent() const {
    return exponent_biased() - 127;
  }

  /// Returns the mantissa
  MUTLASS_HOST_DEVICE
  int mantissa() const {
    return int(raw() & 0x7f);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

MUTLASS_HOST_DEVICE
bool signbit(mutlass::bfloat16_t const& h) {
  return h.signbit();
}

MUTLASS_HOST_DEVICE
mutlass::bfloat16_t abs(mutlass::bfloat16_t const& h) {
  return mutlass::bfloat16_t::bitcast(h.raw() & 0x7fff);
}

MUTLASS_HOST_DEVICE
bool isnan(mutlass::bfloat16_t const& h) {
  return (h.exponent_biased() == 0x0ff) && h.mantissa();
}

MUTLASS_HOST_DEVICE
bool isfinite(mutlass::bfloat16_t const& h) {
  return (h.exponent_biased() != 0x0ff);
}

MUTLASS_HOST_DEVICE
mutlass::bfloat16_t nan_bf16(const char*) {
  return mutlass::bfloat16_t::bitcast(0x7fff);
}

MUTLASS_HOST_DEVICE
bool isinf(mutlass::bfloat16_t const& h) {
  return (h.exponent_biased() == 0x0ff) && !h.mantissa();
}

MUTLASS_HOST_DEVICE
bool isnormal(mutlass::bfloat16_t const& h) {
  return h.exponent_biased() && h.exponent_biased() != 0x0ff;
}

MUTLASS_HOST_DEVICE
int fpclassify(mutlass::bfloat16_t const& h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0x0ff) {
    if (mantissa) {
      return FP_NAN;
    }
    else {
      return FP_INFINITE;
    }
  }
  else if (!exp) {
    if (mantissa) {
      return FP_SUBNORMAL;
    }
    else {
      return FP_ZERO;
    }
  }
  return FP_NORMAL;
}

MUTLASS_HOST_DEVICE
mutlass::bfloat16_t sqrt(mutlass::bfloat16_t const& h) {
#if defined(__MUSACC_RTC__)
  return mutlass::bfloat16_t(sqrtf(float(h)));
#else
  return mutlass::bfloat16_t(std::sqrt(float(h)));
#endif
}

MUTLASS_HOST_DEVICE
bfloat16_t copysign(bfloat16_t const& a, bfloat16_t const& b) {

  uint16_t a_bits;
  uint16_t b_bits;

  #if defined(__MUSA_ARCH__)
  a_bits = reinterpret_cast<uint16_t const &>(a);
  b_bits = reinterpret_cast<uint16_t const &>(b);
  #else
  std::memcpy(&a_bits, &a, sizeof(a_bits));
  std::memcpy(&b_bits, &b, sizeof(b_bits));
  #endif

  uint16_t a_mag = (a_bits & 0x7fff);
  uint16_t b_sign = (b_bits & 0x8000);
  uint16_t result = (a_mag | b_sign);

  return bfloat16_t::bitcast(result);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace std {

#if !defined(__MUSACC_RTC__)
/// Numeric limits
template <>
struct numeric_limits<mutlass::bfloat16_t> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 7;

  /// Least positive value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t min() { return mutlass::bfloat16_t::bitcast(0x01); }

  /// Minimum finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t lowest() { return mutlass::bfloat16_t::bitcast(0xff7f); }

  /// Maximum finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t max() { return mutlass::bfloat16_t::bitcast(0x7f7f); }

  /// Returns smallest finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t epsilon() { return mutlass::bfloat16_t::bitcast(0x1000); }

  /// Returns smallest finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t round_error() { return mutlass::bfloat16_t(0.5f); }

  /// Returns smallest finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t infinity() { return mutlass::bfloat16_t::bitcast(0x7f80); }

  /// Returns smallest finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t quiet_NaN() { return mutlass::bfloat16_t::bitcast(0x7fff); }

  /// Returns smallest finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t signaling_NaN() { return mutlass::bfloat16_t::bitcast(0x7fff); }

  /// Returns smallest finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t denorm_min() { return mutlass::bfloat16_t::bitcast(0x1); }
};
#endif

} // namespace std

namespace mutlass {
namespace platform {

/// std::numeric_limits
template <class T>
struct numeric_limits;

/// Numeric limits
template <>
struct numeric_limits<mutlass::bfloat16_t> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
#if !defined(__MUSACC_RTC__)
  static std::float_denorm_style const has_denorm = std::denorm_present;
#endif
  static bool const has_denorm_loss = true;
#if !defined(__MUSACC_RTC__)
  static std::float_round_style const round_style = std::round_to_nearest;
#endif
  static bool const is_iec559 = true;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 10;

  /// Least positive value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t min() { return mutlass::bfloat16_t::bitcast(0x0001); }

  /// Minimum finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t lowest() { return mutlass::bfloat16_t::bitcast(0xfbff); }

  /// Maximum finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t max() { return mutlass::bfloat16_t::bitcast(0x7bff); }

  /// Returns smallest finite value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t epsilon() { return mutlass::bfloat16_t::bitcast(0x1800); }

  /// Returns maximum rounding error
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t round_error() { return mutlass::bfloat16_t(0.5f); }

  /// Returns positive infinity value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t infinity() { return mutlass::bfloat16_t::bitcast(0x7c00); }

  /// Returns quiet NaN value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t quiet_NaN() { return mutlass::bfloat16_t::bitcast(0x7fff); }

  /// Returns signaling NaN value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t signaling_NaN() { return mutlass::bfloat16_t::bitcast(0x7fff); }

  /// Returns smallest positive subnormal value
  MUTLASS_HOST_DEVICE
  static mutlass::bfloat16_t denorm_min() { return mutlass::bfloat16_t::bitcast(0x0001); }
};
}  // namespace platform
}  // namespace mutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

MUTLASS_HOST_DEVICE
bool operator==(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) == float(rhs);
}

MUTLASS_HOST_DEVICE
bool operator!=(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) != float(rhs);
}

MUTLASS_HOST_DEVICE
bool operator<(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) < float(rhs);
}

MUTLASS_HOST_DEVICE
bool operator<=(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) <= float(rhs);
}

MUTLASS_HOST_DEVICE
bool operator>(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) > float(rhs);
}

MUTLASS_HOST_DEVICE
bool operator>=(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return float(lhs) >= float(rhs);
}

MUTLASS_HOST_DEVICE
bfloat16_t operator+(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return bfloat16_t(float(lhs) + float(rhs));
}

MUTLASS_HOST_DEVICE
bfloat16_t operator-(bfloat16_t const& lhs) {
  return bfloat16_t(-float(lhs));
}

MUTLASS_HOST_DEVICE
bfloat16_t operator-(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return bfloat16_t(float(lhs) - float(rhs));
}

MUTLASS_HOST_DEVICE
bfloat16_t operator*(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return bfloat16_t(float(lhs) * float(rhs));
}

MUTLASS_HOST_DEVICE
bfloat16_t operator/(bfloat16_t const& lhs, bfloat16_t const& rhs) {
  return bfloat16_t(float(lhs) / float(rhs));
}

MUTLASS_HOST_DEVICE
bfloat16_t& operator+=(bfloat16_t & lhs, bfloat16_t const& rhs) {
  lhs = bfloat16_t(float(lhs) + float(rhs));
  return lhs;
}

MUTLASS_HOST_DEVICE
bfloat16_t& operator-=(bfloat16_t & lhs, bfloat16_t const& rhs) {
  lhs = bfloat16_t(float(lhs) - float(rhs));
  return lhs;
}

MUTLASS_HOST_DEVICE
bfloat16_t& operator*=(bfloat16_t & lhs, bfloat16_t const& rhs) {
  lhs = bfloat16_t(float(lhs) * float(rhs));
  return lhs;
}

MUTLASS_HOST_DEVICE
bfloat16_t& operator/=(bfloat16_t & lhs, bfloat16_t const& rhs) {
  lhs = bfloat16_t(float(lhs) / float(rhs));
  return lhs;
}

MUTLASS_HOST_DEVICE
bfloat16_t& operator++(bfloat16_t & lhs) {
  float tmp(lhs);
  ++tmp;
  lhs = bfloat16_t(tmp);
  return lhs;
}

MUTLASS_HOST_DEVICE
bfloat16_t& operator--(bfloat16_t & lhs) {
  float tmp(lhs);
  --tmp;
  lhs = bfloat16_t(tmp);
  return lhs;
}

MUTLASS_HOST_DEVICE
bfloat16_t operator++(bfloat16_t & lhs, int) {
  bfloat16_t ret(lhs);
  float tmp(lhs);
  tmp++;
  lhs = bfloat16_t(tmp);
  return ret;
}

MUTLASS_HOST_DEVICE
bfloat16_t operator--(bfloat16_t & lhs, int) {
  bfloat16_t ret(lhs);
  float tmp(lhs);
  tmp--;
  lhs = bfloat16_t(tmp);
  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

MUTLASS_HOST_DEVICE
mutlass::bfloat16_t operator "" _bf16(long double x) {
  return mutlass::bfloat16_t(float(x));
}

MUTLASS_HOST_DEVICE
mutlass::bfloat16_t operator "" _bf16(unsigned long long int x) {
  return mutlass::bfloat16_t(int(x));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
