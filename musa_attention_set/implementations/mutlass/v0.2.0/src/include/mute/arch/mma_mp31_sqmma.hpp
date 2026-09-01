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
#include <mute/arch/mma_mp31.hpp>

#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310))
#define MUTE_ARCH_SQMMA_MP31_ENABLED
#endif

namespace mute {

// Wait all previous in-flight SQMMA to complete
template <int N = 0>
MUTE_HOST_DEVICE
void
warpsquad_wait()
{
  static_assert(N == 0, "SQMMA wait: N must be 0 now");
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
  __musa_sqmma_wait();
#else
  MUTE_INVALID_CONTROL_PATH("Attempting to use sqmma wait without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
}


namespace MP31::SQMMA {

enum class ScaleOut {
  One  = 0,
  Zero = 1,
};

enum class ScaleIn {
  One = 0,
  Neg = 1,
};

} // namespace SQMMA

////////////////////////////////////////////////////////////////////////////////////////////////////
// SQMMA Intrinsic definitions:  C = (scaleA * A) * (scaleB * B) + (scaleD * C)
////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x64x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x64x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x64x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x32x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x32x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x32x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x64x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x64x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x64x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x128x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x128x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x128x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x16x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x16x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x16x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x32x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x32x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x32x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x64x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x64x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x64x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x128x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x128x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x128x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x32x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x32x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x32x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x64x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x64x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x64x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x128x32_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x128x64_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x128 U32+=U8*U8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x128x128_U32U8U8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x128_U32U8U8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x64x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x64x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x64x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x32x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x32x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x32x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x64x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x64x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x64x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x128x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x128x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_32x128x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x16x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x16x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x16x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x32x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x32x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x32x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x64x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x64x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x64x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x128x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x128x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_64x128x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x32x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x32x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x32x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x64x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x64x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x64x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x128x32_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x128x64_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x128 S32+=S8*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_128x128x128_S32S8S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x128_S32S8S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x16 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x16_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x16_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=F16*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32F16F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32F16F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x16 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x16_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x16_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=BF16*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32BF16BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32BF16BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x8 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x8_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 8, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x8_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x16 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x16_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x16_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x8 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x8_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 8, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x8_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x16 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x16_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x16_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x8 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x8_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 8, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x8_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x16 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x16_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x16_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x8 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x8_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 8, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x8_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x16 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x16_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x16_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x8 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x8_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 8, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x8_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x16 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x16_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x16_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x8 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x8_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 8, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x8_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x16 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x16_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x16_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x8 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x8_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  static_assert(tnspA == TCE::Major::K,
      "TF32 SQMMA 128x64 operand A must have K major layout.");
  static_assert(tnspB == TCE::Major::K,
      "TF32 SQMMA 128x64 operand B must have K major layout.");

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, 0, 1,
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 8, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x8_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x16 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x16_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  static_assert(tnspA == TCE::Major::K,
      "TF32 SQMMA 128x64 operand A must have K major layout.");
  static_assert(tnspB == TCE::Major::K,
      "TF32 SQMMA 128x64 operand B must have K major layout.");

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, 0, 1,
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x16_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  static_assert(tnspA == TCE::Major::K,
      "TF32 SQMMA 128x64 operand A must have K major layout.");
  static_assert(tnspB == TCE::Major::K,
      "TF32 SQMMA 128x64 operand B must have K major layout.");

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, 0, 1,
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x8 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x8_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  static_assert(tnspA == TCE::Major::K,
      "TF32 SQMMA 128x128 operand A must have K major layout.");
  static_assert(tnspB == TCE::Major::K,
      "TF32 SQMMA 128x128 operand B must have K major layout.");

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, 0, 1,
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 8, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x8_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x16 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x16_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  static_assert(tnspA == TCE::Major::K,
      "TF32 SQMMA 128x128 operand A must have K major layout.");
  static_assert(tnspB == TCE::Major::K,
      "TF32 SQMMA 128x128 operand B must have K major layout.");

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, 0, 1,
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 16, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x16_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=TF32*TF32
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32TF32TF32_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  static_assert(tnspA == TCE::Major::K,
      "TF32 SQMMA 128x128 operand A must have K major layout.");
  static_assert(tnspB == TCE::Major::K,
      "TF32 SQMMA 128x128 operand B must have K major layout.");

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, 0, 1,
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32TF32TF32_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x128 F32+=E4M3*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x128_F32E4M3E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x128_F32E4M3E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x128 F32+=E5M2*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x128_F32E5M2E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x128_F32E5M2E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x64_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=F16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32F16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32F16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x64_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=F16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32F16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32F16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x64_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=S4*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32S4F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32S4F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x64_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=S8*F16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32S8F16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32S8F16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x128 F32+=E4M3*E5M2
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x128_F32E4M3E5M2_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x128_F32E4M3E5M2_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x128 F32+=E5M2*E4M3
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x128_F32E5M2E4M3_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 128, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x128_F32E5M2E4M3_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x64_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=BF16*S4
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32BF16S4_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32BF16S4_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x32 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x32_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x32_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 16x64x64 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_16x64x64_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m16n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x64x64_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=BF16*S8
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32BF16S8_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), 0, static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32BF16S8_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x64_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=S4*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32S4BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32S4BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x32 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x32_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x32_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x32x64 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x32x64_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x32x64_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x32 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x32_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x32_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x64x64 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x64x64_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x64x64_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x32 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x32_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x32_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 32x128x64 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_32x128x64_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_32x128x64_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x32 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x32_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x32_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x16x64 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x16x64_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n16_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x16x64_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x32 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x32_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x32_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x32x64 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x32x64_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x32x64_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x32 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x32_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x32_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x64x64 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x64x64_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x64x64_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x32 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x32_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x32_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 64x128x64 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_64x128x64_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_64x128x64_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x32 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x32_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x32_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x32x64 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x32x64_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x32x64_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x32 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x32_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x32_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x64x64 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x64x64_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x64x64_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x32 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x32_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 32, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x32_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA 128x128x64 F32+=S8*BF16
template <
  TCE::Major tnspA,
  TCE::Major tnspB,
  MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One
>
struct MP31_128x128x64_F32S8BF16_SS
{
  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      0, static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ 64, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_128x128x64_F32S8BF16_SS without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace mute
