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
#include <mute/arch/tce_desc.hpp>

#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310))
#define MUTE_ARCH_MMA_MP31_ENABLED
#endif

namespace mute {

namespace MP31 {

// TCE Src Operand enum
enum class TceABDtype {
  U8U8     = 0x00,
  S8S8     = 0x01,
  F16F16   = 0x02,
  BF16BF16 = 0x03,
  TF32TF32 = 0x04,
  E4M3E4M3 = 0x05,
  E5M2E5M2 = 0x06,
  F16S4    = 0x07,
  F16S8    = 0x08,
  S4F16    = 0x09,
  S8F16    = 0x0A,
  E4M3E5M2 = 0x0B,
  E5M2E4M3 = 0x0C,
  BF16S4   = 0x0D,
  BF16S8   = 0x0E,
  S4BF16   = 0x0F,
  S8BF16   = 0x10,
};

} // namespace MP31

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x8_F32F16F16F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k8_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x8_F32F16F16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x8_F32BF16BF16F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k8_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x8_F32BF16BF16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_F32F16F16F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_F32F16F16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_F32BF16BF16F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_F32BF16BF16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_F32F16F16F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_F32F16F16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_F32BF16BF16F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_F32BF16BF16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32F16F16F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32F16F16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32BF16BF16F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32BF16BF16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32F16F16F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32F16F16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32BF16BF16F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32BF16BF16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_S32S8S8S32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 1, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_S32S8S8S32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_U32U8U8U32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_U32U8U8U32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_F32E5M2E5M2F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_F32E5M2E5M2F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_F32E4M3E4M3F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_F32E4M3E4M3F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_F32E4M3E5M2F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_F32E4M3E5M2F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_F32E5M2E4M3F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_F32E5M2E4M3F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_S32S8S8S32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 1, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_S32S8S8S32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_U32U8U8U32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_U32U8U8U32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_F32E5M2E5M2F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_F32E5M2E5M2F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_F32E4M3E4M3F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_F32E4M3E4M3F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_F32E4M3E5M2F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_F32E4M3E5M2F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_F32E5M2E4M3F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_F32E5M2E4M3F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_S32S8S8S32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 1, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_S32S8S8S32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_U32U8U8U32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_U32U8U8U32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32E5M2E5M2F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32E5M2E5M2F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32E4M3E4M3F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32E4M3E4M3F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32E4M3E5M2F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32E4M3E5M2F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32E5M2E4M3F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32E5M2E4M3F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_S32S8S8S32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 1, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_S32S8S8S32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_U32U8U8U32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_U32U8U8U32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32E5M2E5M2F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32E5M2E5M2F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32E4M3E4M3F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32E4M3E4M3F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32E4M3E5M2F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32E4M3E5M2F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32E5M2E4M3F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32E5M2E4M3F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x64_S32S8S8S32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k64_mma(d, a, b, c, 0, 0, 0, 0, 1, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x64_S32S8S8S32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x64_U32U8U8U32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k64_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::U8U8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x64_U32U8U8U32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x64_F32E5M2E5M2F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k64_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x64_F32E5M2E5M2F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x64_F32E4M3E4M3F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k64_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x64_F32E4M3E4M3F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x64_F32E4M3E5M2F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k64_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E4M3E5M2));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x64_F32E4M3E5M2F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x64_F32E5M2E4M3F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k64_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::E5M2E4M3));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x64_F32E5M2E4M3F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x4_F32TF32TF32F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k4_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x4_F32TF32TF32F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x8_F32TF32TF32F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k8_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x8_F32TF32TF32F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32TF32TF32F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::TF32TF32));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32TF32TF32F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32F16S4F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32F16S4F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32S4F16F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32S4F16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32BF16S4F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S4));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32BF16S4F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32S4BF16F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S4BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32S4BF16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_F32F16S8F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_F32F16S8F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_F32S8F16F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_F32S8F16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_F32BF16S8F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_F32BF16S8F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x8x16_F32S8BF16F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n8k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x8x16_F32S8BF16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_F32F16S8F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_F32F16S8F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_F32S8F16F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_F32S8F16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_F32BF16S8F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_F32BF16S8F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_8x16x16_F32S8BF16F32
{
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[4];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m8n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_8x16x16_F32S8BF16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32F16S8F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32F16S8F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32S8F16F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32S8F16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32BF16S8F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32BF16S8F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x16_F32S8BF16F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x16_F32S8BF16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32F16S8F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::F16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32F16S8F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32S8F16F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8F16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32S8F16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32BF16S8F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[8];
  using BRegisters = int32_t[4];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::BF16S8));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32BF16S8F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TCE::Major tnspA,
  TCE::Major tnspB
>
struct MP31_16x16x32_F32S8BF16F32
{
  using DRegisters = int32_t[8];
  using ARegisters = int32_t[4];
  using BRegisters = int32_t[8];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t      * d,
      int32_t const* a,
      int32_t const* b,
      int32_t const* c)
  {
#if defined(MUTE_ARCH_MMA_MP31_ENABLED)
    constexpr int layout_num = ((static_cast<int>(tnspA) << 1) | static_cast<int>(tnspB)) ^ 0b01;
    __musa_wmma_m16n16k32_mma(d, a, b, c, 0, 0, 0, 0, 0, layout_num, /* mma_type = */ static_cast<int>(MP31::TceABDtype::S8BF16));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP31_16x16x32_F32S8BF16F32 without MUTE_ARCH_MMA_MP31_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mute

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <mute/arch/mma_mp31_desc.hpp>
#include <mute/arch/mma_mp31_sqmma.hpp>

namespace mute {
namespace MP31::SQMMA {

template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  TCE::Major MajorA = TCE::Major::K,
  TCE::Major MajorB = TCE::Major::K,
  class MaxInstructionM = _128,
  class MaxInstructionN = _128,
  auto... Args                         // e.g. SQMMA::ScaleOut::One, [SQMMA::ScaleIn::One, SQMMA::ScaleIn::One]
                                       // But most commonly leave empty for defaults
>
MUTE_HOST_DEVICE constexpr
auto
ss_op_selector()
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  auto Tile_M = size<0>(TileShape_MNK{});
  auto Tile_N = size<1>(TileShape_MNK{});
  auto Tile_K = size<2>(TileShape_MNK{});
  auto MaxInstM = MaxInstructionM{};
  auto MaxInstN = MaxInstructionN{};
  static_assert(MaxInstM >= 16 && MaxInstM <= 128 && (MaxInstM & (MaxInstM-1)) == 0);
  static_assert(MaxInstN >= 16 && MaxInstN <= 128 && (MaxInstN & (MaxInstN-1)) == 0);

  // Input A: uint8_t, Input B: uint8_t
  if constexpr (is_same_v<ElementA, uint8_t> && is_same_v<ElementB, uint8_t>) {
    static_assert(is_same_v<ElementC, uint32_t>, "ElementC should be uint32_t");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x128x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x64x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x32x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x128x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x64x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x32x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x16x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x128x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x64x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x32x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_16x64x128_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_U32U8U8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_U32U8U8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: int8_t, Input B: int8_t
  else if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, int8_t>) {
    static_assert(is_same_v<ElementC, int32_t>, "ElementC should be int32_t");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x128x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x64x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x32x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x128x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x64x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x32x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x16x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x128x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x64x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x32x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_16x64x128_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_S32S8S8_SS<MajorA, MajorB>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_S32S8S8_SS<MajorA, MajorB>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: half_t, Input B: half_t
  else if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, half_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_128x128x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_128x64x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_128x32x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x128x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x64x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x32x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x16x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_32x128x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_32x64x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_32x32x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_16x64x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: bfloat16_t, Input B: bfloat16_t
  else if constexpr (is_same_v<ElementA, bfloat16_t> && is_same_v<ElementB, bfloat16_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_128x128x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_128x64x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_128x32x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x128x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x64x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x32x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x16x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_32x128x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_32x64x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_32x32x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_16x64x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 16 == 0, "Tile_K must be a multiple of 16.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: tfloat32_t, Input B: tfloat32_t
  else if constexpr (is_same_v<ElementA, tfloat32_t> && is_same_v<ElementB, tfloat32_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr ((Tile_M % 128 == 0)     &&
                  MajorA == TCE::Major::K &&
                  MajorB == TCE::Major::K &&
                  MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_128x128x16_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 8 == 0) {
          return MP31_128x128x8_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 8 == 0, "Tile_K must be a multiple of 8.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_128x64x16_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 8 == 0) {
          return MP31_128x64x8_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 8 == 0, "Tile_K must be a multiple of 8.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x64x16_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 8 == 0) {
          return MP31_64x64x8_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 8 == 0, "Tile_K must be a multiple of 8.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x32x16_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 8 == 0) {
          return MP31_64x32x8_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 8 == 0, "Tile_K must be a multiple of 8.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_64x16x16_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 8 == 0) {
          return MP31_64x16x8_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 8 == 0, "Tile_K must be a multiple of 8.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_32x64x16_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 8 == 0) {
          return MP31_32x64x8_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 8 == 0, "Tile_K must be a multiple of 8.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_32x32x16_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 8 == 0) {
          return MP31_32x32x8_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 8 == 0, "Tile_K must be a multiple of 8.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 16 == 0) {
          return MP31_16x64x16_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 8 == 0) {
          return MP31_16x64x8_F32TF32TF32_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 8 == 0, "Tile_K must be a multiple of 8.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: float_e4m3_t, Input B: float_e4m3_t
  else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e4m3_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x128x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x64x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x32x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x128x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x64x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x32x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x16x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x128x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x64x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x32x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_16x64x128_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32E4M3E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: float_e5m2_t, Input B: float_e5m2_t
  else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e5m2_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x128x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x64x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x32x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x128x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x64x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x32x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x16x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x128x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x64x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x32x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_16x64x128_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32E5M2E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: half_t, Input B: int4b_t
  else if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, int4b_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32F16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: half_t, Input B: int8_t
  else if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, int8_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32F16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: int4b_t, Input B: half_t
  else if constexpr (is_same_v<ElementA, int4b_t> && is_same_v<ElementB, half_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32S4F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else {
      static_assert(Tile_M % 32 == 0, "Tile_M must be a multiple of 32.");
    }
  }
  // Input A: int8_t, Input B: half_t
  else if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, half_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32S8F16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else {
      static_assert(Tile_M % 32 == 0, "Tile_M must be a multiple of 32.");
    }
  }
  // Input A: float_e4m3_t, Input B: float_e5m2_t
  else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e5m2_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x128x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x64x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x32x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x128x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x64x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x32x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x16x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x128x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x64x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x32x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_16x64x128_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32E4M3E5M2_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: float_e5m2_t, Input B: float_e4m3_t
  else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e4m3_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x128x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x64x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_128x32x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x128x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x64x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x32x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_64x16x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x128x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x64x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_32x32x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 128 == 0) {
          return MP31_16x64x128_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32E5M2E4M3_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: bfloat16_t, Input B: int4b_t
  else if constexpr (is_same_v<ElementA, bfloat16_t> && is_same_v<ElementB, int4b_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32BF16S4_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: bfloat16_t, Input B: int8_t
  else if constexpr (is_same_v<ElementA, bfloat16_t> && is_same_v<ElementB, int8_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 16 == 0 && MaxInstM >= 16) {
      if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_16x64x64_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_16x64x32_F32BF16S8_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 64 == 0, "Tile_N must be a multiple of 64.");
      }
    }
    else {
      static_assert(Tile_M % 16 == 0, "Tile_M must be a multiple of 16.");
    }
  }
  // Input A: int4b_t, Input B: bfloat16_t
  else if constexpr (is_same_v<ElementA, int4b_t> && is_same_v<ElementB, bfloat16_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32S4BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else {
      static_assert(Tile_M % 32 == 0, "Tile_M must be a multiple of 32.");
    }
  }
  // Input A: int8_t, Input B: bfloat16_t
  else if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, bfloat16_t>) {
    static_assert(is_same_v<ElementC, float>, "ElementC should be float");
    if constexpr (Tile_M % 128 == 0 && MaxInstM >= 128) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x128x64_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x128x32_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x64x64_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x64x32_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_128x32x64_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_128x32x32_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else if constexpr (Tile_M % 64 == 0 && MaxInstM >= 64) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x128x64_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x128x32_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x64x64_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x64x32_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x32x64_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x32x32_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 16 == 0 && MaxInstN >= 16) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_64x16x64_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_64x16x32_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 16 == 0, "Tile_N must be a multiple of 16.");
      }
    }
    else if constexpr (Tile_M % 32 == 0 && MaxInstM >= 32) {
      if constexpr (Tile_N % 128 == 0 && MaxInstN >= 128) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x128x64_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x128x32_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 64 == 0 && MaxInstN >= 64) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x64x64_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x64x32_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else if constexpr (Tile_N % 32 == 0 && MaxInstN >= 32) {
        if constexpr (Tile_K % 64 == 0) {
          return MP31_32x32x64_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else if constexpr (Tile_K % 32 == 0) {
          return MP31_32x32x32_F32S8BF16_SS<MajorA, MajorB, Args...>{};
        }
        else {
          static_assert(Tile_K % 32 == 0, "Tile_K must be a multiple of 32.");
        }
      }
      else {
        static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 32.");
      }
    }
    else {
      static_assert(Tile_M % 32 == 0, "Tile_M must be a multiple of 32.");
    }
  }
  else {
    static_assert(sizeof(ElementA) == 0, "Unknown TCE AB Dtype");
  }
}

} // namespace MP31::SQMMA
} // namespace mute
