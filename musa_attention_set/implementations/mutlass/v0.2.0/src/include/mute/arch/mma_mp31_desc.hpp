/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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

#include <mute/config.hpp>

#include <mute/arch/mma.hpp>


namespace mute {

////////////////////////////////////////////////////////////////////////////////////////////////////
// SQMMA Descriptor and utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA enums and utilities
namespace MP31::SQMMA
{

enum class LeadingStrideType : uint8_t {
  B8   = 1,
  B16  = 2,
  B32  = 3,
  B64  = 4,
  B128 = 5,
  B256 = 6,
};

enum class SwizzleGranularityType : uint8_t {
  SWIZZLE_GRANULARITY_NONE = 0,
  SWIZZLE_GRANULARITY_B16  = 1,
  SWIZZLE_GRANULARITY_B32  = 2,
  SWIZZLE_GRANULARITY_B64  = 3,
};


MUTE_HOST_DEVICE char const* to_string(LeadingStrideType const& t) {
  switch (t) {
    case LeadingStrideType::B8:    return "8B";
    case LeadingStrideType::B16:   return "16B";
    case LeadingStrideType::B32:   return "32B";
    case LeadingStrideType::B64:   return "64B";
    case LeadingStrideType::B128:  return "128B";
    case LeadingStrideType::B256:  return "256B";
  }
  return nullptr;
}

MUTE_HOST_DEVICE char const* to_string(SwizzleGranularityType const& t) {
  switch (t) {
    case SwizzleGranularityType::SWIZZLE_GRANULARITY_NONE: return "NONE";
    case SwizzleGranularityType::SWIZZLE_GRANULARITY_B16:  return "SG_16B";
    case SwizzleGranularityType::SWIZZLE_GRANULARITY_B32:  return "SG_32B";
    case SwizzleGranularityType::SWIZZLE_GRANULARITY_B64:  return "SG_64B";
  }
  return nullptr;
}

#if !defined(__MUSACC_RTC__)
MUTE_HOST std::ostream& operator<<(std::ostream& os, LeadingStrideType const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}

MUTE_HOST std::ostream& operator<<(std::ostream& os, SwizzleGranularityType const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}
#endif // !defined(__MUSACC_RTC__)


union SqmmaDescriptor
{

  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor() noexcept : desc_(0) {}
  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor(uint32_t desc) noexcept : desc_(desc) {}
  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor(SqmmaDescriptor const& t) noexcept : desc_(t.desc_) {}
  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor(SqmmaDescriptor && t) noexcept : desc_(t.desc_) {}

  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor& operator=(SqmmaDescriptor const& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor& operator=(SqmmaDescriptor && t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  int32_t desc_;

  // Bitfield implementation avoids the need for shifts in assignment
  struct BitField {
    // start_address, bit [0, 18)
    uint32_t start_address_ : 18;
    // leading dimension byte offset type, bit [18, 21)
    // 8B:1,16B:2,32B:3,64B:4,128B:5,256B:6
    uint8_t leading_stride_type_ : 3;
    // swizzle granularity type, bit [21, 23)
    // SG_NONE=0,SG_16B=1,SG_32B=2,SG_64B=3
    uint8_t swizzle_granularity_type_ : 2;
  } bitfield;

  static_assert(sizeof_bits_v<BitField> == 32, "The underlying bits of BitField must be 32.");

  // Decay to a int32_t
  MUTE_HOST_DEVICE constexpr
  operator int32_t() const noexcept { return desc_; }

  // Printer
  MUTE_HOST_DEVICE friend void print(SqmmaDescriptor const& t)
  {
    #if !defined(__MUSACC_RTC__)
    printf("SqmmaDescriptor: 0x%08x\n", t.desc_);
    printf("     start_addr: 0x%05x\n", t.bitfield.start_address_);
    printf(" leading_stride: 0x%01x(%s)\n", t.bitfield.leading_stride_type_,
           to_string(static_cast<SQMMA::LeadingStrideType>(t.bitfield.leading_stride_type_)));
    printf(" sw_granularity: 0x%01x(%s)\n", t.bitfield.swizzle_granularity_type_,
           to_string(static_cast<SQMMA::SwizzleGranularityType>(t.bitfield.swizzle_granularity_type_)));
    #endif
  }
};

} // namespace MP31::SQMMA

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mute

////////////////////////////////////////////////////////////////////////////////////////////////////
