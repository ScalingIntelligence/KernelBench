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

#if defined(__MUSACC__)
#  define MUTE_HOST_DEVICE __forceinline__ __host__ __device__
#  define MUTE_DEVICE      __forceinline__          __device__
#  define MUTE_HOST        __forceinline__ __host__
#else
#  define MUTE_HOST_DEVICE inline
#  define MUTE_DEVICE      inline
#  define MUTE_HOST        inline
#endif // MUTE_HOST_DEVICE, MUTE_DEVICE

#if defined(__MUSACC_RTC__)
#  define MUTE_HOST_RTC MUTE_HOST_DEVICE
#else
#  define MUTE_HOST_RTC MUTE_HOST
#endif

#if defined(__MUSACC_RTC__) || defined(__clang__)
#  define MUTE_UNROLL    _Pragma("unroll")
#  define MUTE_NO_UNROLL _Pragma("unroll 1")
#else
#  define MUTE_UNROLL
#  define MUTE_NO_UNROLL
#endif // MUTE_UNROLL

#if defined(__MUSA_ARCH__)
#  define MUTE_INLINE_CONSTANT                 static const __device__
#else
#  define MUTE_INLINE_CONSTANT                 static constexpr
#endif

#define MUTE_GRID_CONSTANT

// Some versions of GCC < 11 have trouble deducing that a
// function with "auto" return type and all of its returns in an "if
// constexpr ... else" statement must actually return.  Thus, GCC
// emits spurious "missing return statement" build warnings.
// Developers can suppress these warnings by using the
// MUTE_GCC_UNREACHABLE macro, which must be followed by a semicolon.
// It's harmless to use the macro for other GCC versions or other
// compilers, but it has no effect.
#if ! defined(MUTE_GCC_UNREACHABLE)
#  if defined(__GNUC__)
#    define MUTE_GCC_UNREACHABLE __builtin_unreachable()
#  else
#    define MUTE_GCC_UNREACHABLE
#  endif
#endif

#if defined(_MSC_VER)
// Provides support for alternative operators 'and', 'or', and 'not'
#  include <iso646.h>
#endif // _MSC_VER

#if defined(__MUSACC_RTC__)
#  define MUTE_STL_NAMESPACE musa::std
#  define MUTE_STL_NAMESPACE_IS_MUSA_STD
#else
#  define MUTE_STL_NAMESPACE std
#endif

//
// Assertion helpers
//

#if defined(__MUSACC_RTC__)
#  include <musa/std/cassert>
#else
#  include <cassert>
#endif

#define MUTE_STATIC_V(x)            decltype(x)::value

#define MUTE_STATIC_ASSERT          static_assert
#define MUTE_STATIC_ASSERT_V(x,...) static_assert(decltype(x)::value, ##__VA_ARGS__)

// Fail and print a message. Typically used for notification of a compiler misconfiguration.
#if defined(__MUSA_ARCH__)
#  define MUTE_INVALID_CONTROL_PATH(x) assert(0 && x); printf(x);
#else
#  define MUTE_INVALID_CONTROL_PATH(x) assert(0 && x); printf(x)
#endif

//
// IO
//

#if !defined(__MUSACC_RTC__)
#  include <cstdio>
#  include <iostream>
#  include <iomanip>
#endif

//
// AddressSpace
//
#include <mute/address_space.hpp>

//
// Support
//

#include <mute/util/type_traits.hpp>

//
// Basic types
//

#include <mute/numeric/numeric_types.hpp>

//
// Debugging utilities
//

#include <mute/util/print.hpp>
#include <mute/util/debug.hpp>
