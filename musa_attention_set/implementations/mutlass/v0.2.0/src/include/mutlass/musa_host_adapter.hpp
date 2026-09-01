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
    \brief Interface betweeen a MUTLASS device-wide operator and MUSA.
*/

#pragma once

#include <musa_runtime_api.h>
#include "mutlass/mutlass.h"
#include "mutlass/trace.h"

#include "mutlass/platform/platform.h"
#if ! defined(__MUSACC_RTC__)
#include <cstdio>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Macro-level guard for MUSA Host Adapter
//
#if !defined(MUTLASS_ENABLE_MUSA_HOST_ADAPTER)
#define MUTLASS_ENABLE_MUSA_HOST_ADAPTER false
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This class defines an object which abstracts interactions between the MUTLASS device-wide GEMM and
/// MUSA. The intention is to enable MUTLASS to be used with both the MUSA Runtime API and MUSA Driver API.
struct MusaHostAdapter {

  /// Limit the number of kernels
  static constexpr int32_t kMaximumKernelCount = 4;

  /// Maximum cluster size
  static constexpr int MaxClusterSize = 32;

  //
  // Data members
  //

  /// Handles
  void        *kernel_handles[kMaximumKernelCount];
  int32_t      kernel_count = 0;

  //
  // Methods
  //

  /// Ctor
  MusaHostAdapter() = default;

  /// Dtor
  virtual ~MusaHostAdapter() {}

  /// Copy Ctor
  inline MusaHostAdapter(const MusaHostAdapter & rhs):
    kernel_count(rhs.kernel_count)
  {
    MUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
  }

  /// Copy Assignment
  inline MusaHostAdapter& operator=(const MusaHostAdapter & rhs) {

    MUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
    kernel_count = rhs.kernel_count;
    return *this;
  }

  /// Move ctor
  inline MusaHostAdapter(MusaHostAdapter && rhs):
    kernel_count(rhs.kernel_count)
  {
    MUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
  }

  /// Move assignment
  inline MusaHostAdapter& operator=(MusaHostAdapter && rhs) {

    MUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }

    kernel_count = rhs.kernel_count;

    return *this;
  }

  /// Ctor
  inline MusaHostAdapter(
    void **kernel_handles_, 
    int32_t kernel_count_
  ): 
    kernel_count(kernel_count_)
  {
    MUTLASS_ASSERT(kernel_count >= 0);
    for (int32_t i = 0; i < kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = kernel_handles_[i];
    }
  }

  /// Returns true if the MusaHostAdapter is empty (kernel_count == 0)
  inline bool empty() const { return !kernel_count; }

  /// Returns kernel_count
  inline size_t size() const { return static_cast<size_t>(kernel_count); }

  /// Queries the occupancy of a kernel
  virtual Status query_occupancy(
    int32_t *device_sms, 
    int32_t *sm_occupancy,
    int32_t kernel_index,
    int32_t thread_count,
    int32_t smem_size) const = 0;
 
  /// Launches a kernel without using Threadblock Clusters. 
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    size_t const smem_size,
    musaStream_t musa_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;

  /// Launches a kernel using the MUSA Extensible Launch API and Threadblock Clusters.
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const cluster_dims,
    dim3 const block_dims,
    size_t const smem_size,
    musaStream_t musa_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;

protected:

  /**
   * Fills a buffer in Global Memory with a byte sequence copied from host memory.
   * This function can be overriden to dispatch to the appropriate muMemsetD*Async API
  */
  virtual Status memsetDeviceImpl(
    void* destination, ///< Device memory pointer to be filled
    void const* fill_value, ///< Value to be filled in the buffer
    size_t fill_size, ///< Size of the data type to be used for filling the buffer
    size_t count, ///< Number of elements of size fill_size
    musaStream_t stream) const = 0;

public:

  /// Fills a buffer in Global Memory with a byte sequence copied from host memory
  template<class FillValueType>
  Status memsetDevice(
    void* destination,
    FillValueType fill_value, 
    size_t count,
    musaStream_t stream) const
  {
    return this->memsetDeviceImpl(
      destination,
      &fill_value,
      sizeof(FillValueType),
      count,
      stream);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
