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

#include <mute/arch/copy_mp31_desc.hpp>
#include <mute/swizzle_layout.hpp>


namespace mute::detail {

template <int B_, int M_, int S_, class T = uint8_t>
MUTE_HOST_DEVICE constexpr
auto
get_tme_swizzle_granularity(Swizzle<B_,M_,S_> _)
{
  // Recast to byte unit
  using TmeSwizzle = decltype(recast_layout<T, uint8_t>(_));
  constexpr int M = TmeSwizzle::num_base;

  static_assert(4 <= M && M <= 6, "Unknown Swizzle Granularity.");
  switch (M) {
    case  4: return TME::SmemSwizzleGranularity::B16;
    case  5: return TME::SmemSwizzleGranularity::B32;
    case  6: return TME::SmemSwizzleGranularity::B64;
  }
}

template <TME::SmemSwizzleGranularity sg>
MUTE_HOST_DEVICE constexpr
auto
get_tme_swizzle_base()
{
  static_assert(0 <= static_cast<uint8_t>(sg) && static_cast<uint8_t>(sg) <= 3, "Unknown Swizzle Granularity.");
  switch (sg) {
    case  TME::SmemSwizzleGranularity::B16:  return 4;
    case  TME::SmemSwizzleGranularity::B32:  return 5;
    case  TME::SmemSwizzleGranularity::B64:  return 6;
  }
}

template <int B_, int M_, int S_, class T = uint8_t>
MUTE_HOST_DEVICE constexpr
auto
get_tme_swizzle_line(Swizzle<B_,M_,S_> _)
{
  // Recast to byte unit
  using TmeSwizzle = decltype(recast_layout<T, uint8_t>(_));
  constexpr int M = TmeSwizzle::num_base;
  constexpr int S = TmeSwizzle::num_shft;

  static_assert((M+S) == 8 || (M+S) == 7, "Unknown Swizzle Line.");
  return (M + S) == 8 ? TME::SmemSwizzleLine::B256
                      : TME::SmemSwizzleLine::B128;
}

template <TME::SmemSwizzleLine sl, int M>
MUTE_HOST_DEVICE constexpr
auto
get_tme_swizzle_shft()
{
  static_assert(4 <= M && M <= 6, "Unknown Swizzle Granularity.");
  static_assert(0 <= static_cast<uint8_t>(sl) && static_cast<uint8_t>(sl) <= 1, "Unknown Swizzle Line.");
  switch (sl) {
    case  TME::SmemSwizzleLine::B256: return  8 - M;
    case  TME::SmemSwizzleLine::B128: return  7 - M;
  }                  
}

template <int B_, int M_, int S_, class T = uint8_t>
MUTE_HOST_DEVICE constexpr
auto
get_tme_swizzle_stride(Swizzle<B_,M_,S_> _)
{
  // Recast to byte unit
  using TmeSwizzle = decltype(recast_layout<T, uint8_t>(_));
  constexpr int B = TmeSwizzle::num_bits;
  constexpr int M = TmeSwizzle::num_base;

  static_assert(5 <= (M+B) && (M+B) <= 8, "Unknown Swizzle Stride.");
  switch ((M+B)) {
    case 5:  return TME::SmemSwizzleStride::B32;
    case 6:  return TME::SmemSwizzleStride::B64;
    case 7:  return TME::SmemSwizzleStride::B128;
    case 8:  return TME::SmemSwizzleStride::B256;
  }
}

template <TME::SmemSwizzleStride ss, int M>
MUTE_HOST_DEVICE constexpr
auto
get_tme_swizzle_bits()
{
  static_assert(4 <= M && M <= 6, "Unknown Swizzle Granularity.");
  static_assert(0 <= static_cast<uint8_t>(ss) && static_cast<uint8_t>(ss) <= 3, "Unknown Swizzle Stride.");
  switch (ss) {
    case  TME::SmemSwizzleStride::B32:  return 5 - M;
    case  TME::SmemSwizzleStride::B64:  return 6 - M;
    case  TME::SmemSwizzleStride::B128: return 7 - M;
    case  TME::SmemSwizzleStride::B256: return 8 - M;
  }           
}


template <int B_, int M_, int S_, class T = uint8_t>
MUTE_HOST_DEVICE constexpr
auto
get_tme_swizzle_attributes(Swizzle<B_,M_,S_> _)
{
  // Recast to byte unit
  using TmeSwizzle = decltype(recast_layout<T, uint8_t>(_));
  constexpr int B = TmeSwizzle::num_bits;
  constexpr int M = TmeSwizzle::num_base;
  constexpr int S = TmeSwizzle::num_shft;

  // If B==0 then we don't do swizzle
  if constexpr (B == 0) {
    return mute::make_tuple(TME::SmemSwizzleGranularity::NONE,
                            TME::SmemSwizzleStride::B256,
                            TME::SmemSwizzleLine::B256);
  }
  else {
    constexpr TME::SmemSwizzleGranularity sg = get_tme_swizzle_granularity(TmeSwizzle{});
    constexpr TME::SmemSwizzleStride      ss = get_tme_swizzle_stride(TmeSwizzle{});
    constexpr TME::SmemSwizzleLine        sl = get_tme_swizzle_line(TmeSwizzle{});

    // Check validation SG < SS <= SL
    static_assert(sg < ss && ss <= sl, "Invalid Smem Swizzle Pattern.");
    return mute::make_tuple(sg, ss, sl);
  }
}

template <class Layout, class T = uint8_t>
MUTE_HOST_DEVICE constexpr
auto
get_tme_swizzle_attributes(Layout const& layout) {
  return get_tme_swizzle_attributes<T>(get_swizzle_portion(layout));
}

template <TME::SmemSwizzleLine sl, TME::SmemSwizzleStride ss, TME::SmemSwizzleGranularity sg>
MUTE_HOST_DEVICE constexpr
auto
get_tme_swizzle() {
  static_assert(sg < ss && ss <= sl, "Invalid Smem Swizzle Pattern.");
  if constexpr (sg == TME::SmemSwizzleGranularity::NONE) {
    return Swizzle<0, 4, 3>{};
  }
  else {
    constexpr int M = get_tme_swizzle_base<sg>();
    constexpr int B = get_tme_swizzle_bits<ss, M>();
    constexpr int S = get_tme_swizzle_shft<sl, M>();
    return Swizzle<B, M, S>{};
  }
}

} // namespace mute::detail
