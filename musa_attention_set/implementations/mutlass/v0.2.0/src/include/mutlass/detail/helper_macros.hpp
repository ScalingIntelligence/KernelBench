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

/*! \file
    \brief Helper macros for the MUTLASS library
*/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////


#ifdef MUTLASS_NAMESPACE
#define concat_tok(a, b) a ## b
#define mkmutlassnamespace(pre, ns) concat_tok(pre, ns)
#define mutlass mkmutlassnamespace(mutlass_, MUTLASS_NAMESPACE)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__MUSACC__)
#define MUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define MUTLASS_DEVICE __forceinline__ __device__
#elif defined(__MUSACC_RTC__)
#define MUTLASS_HOST_DEVICE __forceinline__ __device__
#define MUTLASS_DEVICE __forceinline__ __device__
#else
#define MUTLASS_HOST_DEVICE inline
#define MUTLASS_DEVICE inline
#endif

#if ! defined(_MSC_VER)
#define MUTLASS_LAMBDA_FUNC_INLINE __attribute__((always_inline))
#else
#define MUTLASS_LAMBDA_FUNC_INLINE [[msvc::forceinline]]
#endif

#define MUTLASS_HOST __host__
#define MUTLASS_GLOBAL __global__ static

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
MUTLASS_HOST_DEVICE void __MUTLASS_UNUSED(T const &)
{ }

#if defined(__GNUC__)
  #define MUTLASS_UNUSED(expr) __MUTLASS_UNUSED(expr)
#else
  #define MUTLASS_UNUSED(expr) do { ; } while (&expr != &expr)
#endif

#ifdef _MSC_VER
// Provides support for alternative operators 'and', 'or', and 'not'
#include <ciso646>
#endif // _MSC_VER

#if !defined(__MUSACC_RTC__)
#include <cassert>
#endif

#if defined(__MUSA_ARCH__)
  #if defined(_MSC_VER)
    #define MUTLASS_NOT_IMPLEMENTED() { printf("%s not implemented\n", __FUNCSIG__); __musa_exit(); }
  #else
    #define MUTLASS_NOT_IMPLEMENTED() { printf("%s not implemented\n", __PRETTY_FUNCTION__); __musa_exit(); }
  #endif
#else
  #if defined(_MSC_VER)
    #define MUTLASS_NOT_IMPLEMENTED() assert(0 && __FUNCSIG__)
  #else
    #define MUTLASS_NOT_IMPLEMENTED() assert(0 && __PRETTY_FUNCTION__)
  #endif
#endif

#if defined(__MUSA_ARCH__)
#  define MUTLASS_CMATH_NAMESPACE
#else
#  define MUTLASS_CMATH_NAMESPACE std
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {


#ifndef MUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED
#define MUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED 0
#endif


// MUSA 10.1 introduces the mma instruction
#if !defined(MUTLASS_ENABLE_TENSOR_CORE_MMA)
#define MUTLASS_ENABLE_TENSOR_CORE_MMA 0
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define MUTLASS_ASSERT(x) assert(x)

////////////////////////////////////////////////////////////////////////////////////////////////////

// MUTLASS_PRAGMA_(UNROLL|NO_UNROLL) optimization directives for the MUSA compiler.
#if defined(__MUSA_ARCH__) && !defined(__INTELLISENSE__)
  #if defined(__MUSACC_RTC__) || (defined(__clang__) && defined(__MUSA__))
    #define MUTLASS_PRAGMA_UNROLL _Pragma("unroll")
    #define MUTLASS_PRAGMA_NO_UNROLL _Pragma("unroll 1")
  #else
    #define MUTLASS_PRAGMA_UNROLL #pragma unroll
    #define MUTLASS_PRAGMA_NO_UNROLL #pragma unroll 1
  #endif

  #define MUTLASS_GEMM_LOOP MUTLASS_PRAGMA_NO_UNROLL

#else

    #define MUTLASS_PRAGMA_UNROLL
    #define MUTLASS_PRAGMA_NO_UNROLL
    #define MUTLASS_GEMM_LOOP

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__MUSACC_RTC__)
#define MUTLASS_THREAD_LOCAL thread_local
#else
#define MUTLASS_THREAD_LOCAL
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(_MSVC_LANG)
#  define MUTLASS_CPLUSPLUS _MSVC_LANG
#else
#  define MUTLASS_CPLUSPLUS __cplusplus
#endif

#if (201700L <= MUTLASS_CPLUSPLUS)
#define MUTLASS_CONSTEXPR_IF_CXX17 constexpr
#define MUTLASS_CXX17_OR_LATER 1
#else
#define MUTLASS_CONSTEXPR_IF_CXX17
#define MUTLASS_CXX17_OR_LATER 0
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

}; // namespace mutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
