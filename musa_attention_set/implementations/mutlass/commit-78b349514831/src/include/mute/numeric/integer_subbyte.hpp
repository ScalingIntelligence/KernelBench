/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#if defined(__MUSACC_RTC__)
#include <musa/std/cstdint>
#else
#include <cstdint>
#endif

#include <mutlass/integer_subbyte.h>

#include <mute/config.hpp>
#include <mute/util/type_traits.hpp>

namespace mute {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, bool Signed = true>
struct integer_subbyte
{
  /// Storage type
  using Storage = uint8_t;

  /// Number of bits
  static_assert(Bits <= 8*sizeof(Storage), "Require a subbyte of bits in integer_subbyte");

  /// External type
  using xint_t = typename conditional<Signed, int, unsigned>::type;

  /// Bitmask for truncation from larger integers
  static constexpr Storage bits_mask_ = Storage((1 << Bits) - 1);
  /// Bitmask for the sign bit
  static constexpr Storage sign_mask_ = Storage((Signed ? 1 : 0) << (Bits - 1));

  //
  // Data members
  //

  Storage storage;

  //
  // Methods
  //

  /// No operation
  MUTE_HOST_DEVICE constexpr
  integer_subbyte() {}

  /// Conversion from integer type
  MUTE_HOST_DEVICE constexpr
  integer_subbyte(int value)   // NOTE: Sign extension?
      : storage(reinterpret_cast<Storage const&>(value) & bits_mask_) {}

  MUTE_HOST_DEVICE constexpr
  integer_subbyte(unsigned value)
      : storage(reinterpret_cast<Storage const&>(value) & bits_mask_) {}

  /// Convert to int or unsigned
  MUTE_HOST_DEVICE constexpr
  operator xint_t() const {
    if (sign_mask_ & storage) {  // Sign extend
      return xint_t(storage) | ~xint_t(bits_mask_);
    } else {
      return xint_t(storage);
    }
  }

  /// Equality
  MUTE_HOST_DEVICE constexpr
  bool operator==(integer_subbyte const& rhs) const {
    return storage == rhs.storage;
  }

  /// Inequality
  MUTE_HOST_DEVICE constexpr
  bool operator!=(integer_subbyte const& rhs) const {
    return storage != rhs.storage;
  }

  /// Less than or equal
  MUTE_HOST_DEVICE constexpr
  bool operator<=(integer_subbyte const& rhs) const {
    if (sign_mask_ & storage) {
      return !(rhs.storage < storage);
    } else {
      return storage <= rhs.storage;
    }
  }

  /// Less than
  MUTE_HOST_DEVICE constexpr
  bool operator<(integer_subbyte const& rhs) const {
    if (sign_mask_ & storage) {
      return !(rhs.storage <= storage);
    } else {
      return storage < rhs.storage;
    }
  }

  /// Greater than or equal
  MUTE_HOST_DEVICE constexpr
  bool operator>=(integer_subbyte const& rhs) const {
    return !(*this < rhs);
  }

  /// Greater than
  MUTE_HOST_DEVICE constexpr
  bool operator>(integer_subbyte const& rhs) const {
    return !(*this <= rhs);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 1-bit unsigned integer type
using uint1b_t = integer_subbyte<1, false>;

/// 2-bit integer type
using int2b_t = integer_subbyte<2, true>;

/// 2-bit unsigned integer type
using uint2b_t = integer_subbyte<2, false>;

/// 4-bit integer type
using int4b_t = integer_subbyte<4, true>;

/// 4-bit unsigned integer type
using uint4b_t = integer_subbyte<4, false>;

/// 1-bit binary type
using bin1_t = bool;

} // namespace mute

///////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__MUSACC_RTC__)

#include <limits>

namespace MUTE_STL_NAMESPACE {

template <>
struct numeric_limits<mute::uint1b_t> {
  MUTE_HOST_DEVICE static constexpr
  mute::uint1b_t const lowest() noexcept { return 0; }
  MUTE_HOST_DEVICE static constexpr
  mute::uint1b_t const min() noexcept { return 0; }
  MUTE_HOST_DEVICE static constexpr
  mute::uint1b_t const max() noexcept { return 1; }
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = false;
};

template <>
struct numeric_limits<mute::int2b_t> {
  MUTE_HOST_DEVICE static constexpr
  mute::int2b_t lowest() noexcept { return -2; }
  MUTE_HOST_DEVICE static constexpr
  mute::int2b_t min() noexcept { return -2; }
  MUTE_HOST_DEVICE static constexpr
  mute::int2b_t max() noexcept { return 1; }
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = true;
};

template <>
struct numeric_limits<mute::uint2b_t> {
  MUTE_HOST_DEVICE static constexpr
  mute::uint2b_t const lowest() noexcept { return 0; }
  MUTE_HOST_DEVICE static constexpr
  mute::uint2b_t const min() noexcept { return 0; }
  MUTE_HOST_DEVICE static constexpr
  mute::uint2b_t const max() noexcept { return 3; }
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = false;
};

template <>
struct numeric_limits<mute::int4b_t> {
  MUTE_HOST_DEVICE static constexpr
  mute::int4b_t lowest() noexcept { return -8; }
  MUTE_HOST_DEVICE static constexpr
  mute::int4b_t min() noexcept { return -8; }
  MUTE_HOST_DEVICE static constexpr
  mute::int4b_t max() noexcept { return 7; }
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = true;
};

template <>
struct numeric_limits<mute::uint4b_t> {
  MUTE_HOST_DEVICE static constexpr
  mute::uint4b_t const lowest() noexcept { return 0; }
  MUTE_HOST_DEVICE static constexpr
  mute::uint4b_t const min() noexcept { return 0; }
  MUTE_HOST_DEVICE static constexpr
  mute::uint4b_t const max() noexcept { return 15; }
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = false;
};

}  // namespace std

#endif // !defined(__MUSACC_RTC__)
