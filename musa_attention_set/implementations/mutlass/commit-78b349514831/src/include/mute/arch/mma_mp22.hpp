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


#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 220))
#define MUTE_ARCH_MMA_MP22_ENABLED
#endif


namespace mute {

//
// MP22 HMMA 32x32x16
//

struct MP22_32x32x16_F32F16F16F32_TT
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_fmma_m32n32k16_mma(d, a, b, c, 0, 0, 0, 0, 1, 0); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x16_F32F16F16F32_TT without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x16_F32F16F16F32_TN
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_fmma_m32n32k16_mma(d, a, b, c, 0, 0, 0, 0, 1, 1); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x16_F32F16F16F32_TN without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x16_F32F16F16F32_NT
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_fmma_m32n32k16_mma(d, a, b, c, 0, 0, 0, 0, 1, 2); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x16_F32F16F16F32_NT without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
  
struct MP22_32x32x16_F32F16F16F32_NN
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_fmma_m32n32k16_mma(d, a, b, c, 0, 0, 0, 0, 1, 3); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x16_F32F16F16F32_NN without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// MP22 BFMMA 32x32x16
//

struct MP22_32x32x16_F32BF16BF16F32_TT
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_bfmma_m32n32k16_mma(d, a, b, c, 0, 0, 0, 0, 1, 0); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x16_F32BF16BF16F32_TT without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x16_F32BF16BF16F32_TN
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_bfmma_m32n32k16_mma(d, a, b, c, 0, 0, 0, 0, 1, 1); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x16_F32BF16BF16F32_TN without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x16_F32BF16BF16F32_NT
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_bfmma_m32n32k16_mma(d, a, b, c, 0, 0, 0, 0, 1, 2); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x16_F32BF16BF16F32_NT without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x16_F32BF16BF16F32_NN
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_bfmma_m32n32k16_mma(d, a, b, c, 0, 0, 0, 0, 1, 3); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x16_F32BF16BF16F32_NN without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// MP22 TFMMA 32x32x8
//

struct MP22_32x32x8_F32TF32TF32F32_TT
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_tfmma_m32n32k8_mma(d, a, b, c, 0, 0, 0, 0, 1, 0); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x8_F32TF32TF32F32_TT without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x8_F32TF32TF32F32_TN
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_tfmma_m32n32k8_mma(d, a, b, c, 0, 0, 0, 0, 1, 1); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x8_F32TF32TF32F32_TN without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x8_F32TF32TF32F32_NT
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_tfmma_m32n32k8_mma(d, a, b, c, 0, 0, 0, 0, 1, 2); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x8_F32TF32TF32F32_NT without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x8_F32TF32TF32F32_NN
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_tfmma_m32n32k8_mma(d, a, b, c, 0, 0, 0, 0, 1, 3); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x8_F32TF32TF32F32_NN without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// MP22 IMMA 32x32x32
//

struct MP22_32x32x32_S32S8S8S32_TT
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_imma_m32n32k32_mma(d, a, b, c, 0, 0, 0, 0, 1, 0); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x32_S32S8S8S32_TT without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x32_S32S8S8S32_TN
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_imma_m32n32k32_mma(d, a, b, c, 0, 0, 0, 0, 1, 1); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x32_S32S8S8S32_TN without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x32_S32S8S8S32_NT
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_imma_m32n32k32_mma(d, a, b, c, 0, 0, 0, 0, 1, 2); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x32_S32S8S8S32_NT without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP22_32x32x32_S32S8S8S32_NN
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
#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
    __musa_imma_m32n32k32_mma(d, a, b, c, 0, 0, 0, 0, 1, 3); 
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP22_32x32x32_S32S8S8S32_NN without MUTE_ARCH_MMA_MP22_ENABLED");
#endif
  }
};

} // namespace mute
