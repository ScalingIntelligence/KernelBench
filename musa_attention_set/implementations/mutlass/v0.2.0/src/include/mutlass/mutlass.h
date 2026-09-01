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

/*! \file
    \brief Basic include for MUTLASS.
*/

/*
  Note:  MUTLASS 3x increases the host compiler requirements to C++17. However, certain
         existing integrations of MUTLASS require C++11 host compilers.

         Until this requirement can be lifted, certain headers with this annotation are required
         to be remain consistent with C++11 syntax.

         C++11 compatibility is enforced by `mutlass_test_unit_core_cpp11`.
*/

#pragma once

#include "mutlass/detail/helper_macros.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {

/// Status code returned by MUTLASS operations
enum class Status {
  kSuccess,                    ///< Operation was successful.
  kErrorMisalignedOperand,     ///< operands fail alignment requirements.
  kErrorInvalidDataType,       ///< DataType fails requirement.
  kErrorInvalidLayout,         ///< Layout fails alignment requirement.
  kErrorInvalidProblem,        ///< Specified problem size is not supported by operator.
  kErrorNotSupported,          ///< Operation is not supported on current device.
  kErrorWorkspaceNull,         ///< The given workspace is null when it is required to be non-null.
  kErrorInternal,              ///< An error within MUTLASS occurred.
  kErrorArchMismatch,          ///< MUTLASS runs on a device that it was not compiled for.
  kErrorInsufficientDriver,    ///< MUTLASS runs with a driver that is too old.
  kErrorMemoryAllocation,      ///< Kernel launch failed due to insufficient device memory.
  kInvalid                     ///< Status is unspecified.
};

/// Convert mutlass status to status strings
MUTLASS_HOST_DEVICE
static char const* mutlassGetStatusString(mutlass::Status status) {
  switch (status) {
    case mutlass::Status::kSuccess:
      return "Success";
    case mutlass::Status::kErrorMisalignedOperand:
      return "Error Misaligned Operand";
    case mutlass::Status::kErrorInvalidDataType:
      return "Error Invalid Data Type";
    case mutlass::Status::kErrorInvalidLayout:
      return "Error Invalid Layout";
    case mutlass::Status::kErrorInvalidProblem:
      return "Error Invalid Problem";
    case mutlass::Status::kErrorNotSupported:
      return "Error Not Supported";
    case mutlass::Status::kErrorWorkspaceNull:
      return "Error Workspace Null";
    case mutlass::Status::kErrorInternal:
      return "Error Internal";
    case mutlass::Status::kErrorInsufficientDriver:
      return "Error Insufficient Driver";
    case mutlass::Status::kErrorArchMismatch:
      return "Error Architecture Mismatch";
    case mutlass::Status::kErrorMemoryAllocation:
      return "Error Memory Allocation failed";
    case mutlass::Status::kInvalid: break;
  }

  return "Invalid status";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static const int NumThreadsPerWarpBeforeMP31 = 128;
static const int NumThreadsPerWarp = 32;
static const int NumThreadsPerWarpSquad = 128;
static const int NumWarpsPerWarpSquad = NumThreadsPerWarpSquad / NumThreadsPerWarp;
static const int NumThreadsPerHalfWarp = NumThreadsPerWarp / 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper function to return true when called by thread 0 of threadblock 0.
MUTLASS_HOST_DEVICE bool thread0() {
  #if defined(__MUSA_ARCH__)
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
  #else
    return false;
  #endif
}

/// Returns a lane index in the warp. The threads in warp may not be convergent
MUTLASS_DEVICE
int canonical_lane_idx() {
  #if defined(__MUSA_ARCH__)
  #if (__MUSA_ARCH__ >= 310)
    return threadIdx.x % NumThreadsPerWarp;
  #else
    return threadIdx.x % NumThreadsPerWarpBeforeMP31;
  #endif // #if (__MUSA_ARCH__ >= 310)
  #else
    return 0;
  #endif // #if defined(__MUSA_ARCH__)
}

/// Returns a warp index in the CTA. The threads in warp may not be convergent
/// As it doesn't sync the warp, it faster and allows forward progress
MUTLASS_DEVICE
int canonical_warp_idx() {
  #if defined(__MUSA_ARCH__)
  #if (__MUSA_ARCH__ >= 310)
    return threadIdx.x / NumThreadsPerWarp;
  #else
    return threadIdx.x / NumThreadsPerWarpBeforeMP31;
  #endif // #if (__MUSA_ARCH__ >= 310)
  #else
    return 0;
  #endif // #if defined(__MUSA_ARCH__)
}

MUTLASS_DEVICE
int canonical_warp_squad_idx() {
  #if defined(__MUSA_ARCH__)
    return threadIdx.x / NumThreadsPerWarpSquad;
  #else
    return 0;
  #endif // #if defined(__MUSA_ARCH__)
}


////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace mutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
