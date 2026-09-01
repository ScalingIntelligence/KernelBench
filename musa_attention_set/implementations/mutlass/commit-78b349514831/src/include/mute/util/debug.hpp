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

/**
 * \file
 * \brief Debugging and logging functionality
 */

#include <musa_runtime_api.h>

#include <mute/config.hpp>

namespace mute
{

/******************************************************************************
 * Debug and logging macros
 ******************************************************************************/

/**
 * Formats and prints the given message to stdout
 */
#if !defined(MUTE_LOG)
#  if !defined(__MUSA_ARCH__)
#    define MUTE_LOG(format, ...) printf(format, __VA_ARGS__)
#  else
#    define MUTE_LOG(format, ...)                                \
        printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, \
               blockIdx.x,  blockIdx.y,  blockIdx.z,             \
               threadIdx.x, threadIdx.y, threadIdx.z,            \
               __VA_ARGS__);
#  endif
#endif

/**
 * Formats and prints the given message to stdout only if DEBUG is defined
 */
#if !defined(MUTE_LOG_DEBUG)
#  ifdef DEBUG
#    define MUTE_LOG_DEBUG(format, ...) MUTE_LOG(format, __VA_ARGS__)
#  else
#    define MUTE_LOG_DEBUG(format, ...)
#  endif
#endif

/**
 * \brief Perror macro with exit
 */
#if !defined(MUTE_ERROR_EXIT)
#  define MUTE_ERROR_EXIT(e)                                         \
      do {                                                           \
        musaError_t code = (e);                                      \
        if (code != musaSuccess) {                                   \
          fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",               \
                  __FILE__, __LINE__, #e,                            \
                  musaGetErrorName(code), musaGetErrorString(code)); \
          fflush(stderr);                                            \
          exit(1);                                                   \
        }                                                            \
      } while (0)
#endif

#if !defined(MUTE_CHECK_LAST)
#  define MUTE_CHECK_LAST() MUTE_ERROR_EXIT(musaPeekAtLastError()); MUTE_ERROR_EXIT(musaDeviceSynchronize())
#endif

#if !defined(MUTE_CHECK_ERROR)
#  define MUTE_CHECK_ERROR(e) MUTE_ERROR_EXIT(e)
#endif

// A dummy function that uses compilation failure to print a type
template <class... T>
MUTE_HOST_DEVICE void
print_type() {
  static_assert(sizeof...(T) < 0, "Printing type T.");
}

template <class... T>
MUTE_HOST_DEVICE void
print_type(T&&...) {
  static_assert(sizeof...(T) < 0, "Printing type T.");
}

//
// Device-specific helpers
//
// e.g.
// if (thread0()) print(...);
// if (block0()) print(...);
// if (thread(42)) print(...);

MUTE_HOST_DEVICE
bool
block(int bid)
{
#if defined(__MUSA_ARCH__)
  return blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y == static_cast<unsigned int>(bid);;
#else
  return true;
#endif
}

MUTE_HOST_DEVICE
bool
thread(int tid, int bid)
{
#if defined(__MUSA_ARCH__)
  return (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y == static_cast<unsigned int>(tid)) && block(bid);
#else
  return true;
#endif
}

MUTE_HOST_DEVICE
bool
thread(int tid)
{
  return thread(tid,0);
}

MUTE_HOST_DEVICE
bool
thread0()
{
  return thread(0,0);
}

MUTE_HOST_DEVICE
bool
block0()
{
  return block(0);
}

}  // end namespace mute
