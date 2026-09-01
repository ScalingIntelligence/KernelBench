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
    \brief Utilities for initializing workspaces
*/

#pragma once

#if !defined(__MUSACC_RTC__)
#include "musa.h"
#include "musa_runtime.h"

#include "mutlass/trace.h"
#endif

#include "mutlass.h"
#include "mutlass/musa_host_adapter.hpp"
namespace mutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int MinWorkspaceAlignment = 16;

#if !defined(__MUSACC_RTC__)
static Status
zero_workspace(void* workspace, size_t workspace_size, musaStream_t stream = nullptr) {
  if (workspace_size > 0) {
    if (workspace == nullptr) {
      MUTLASS_TRACE_HOST("  error: device workspace must not be null");
      return Status::kErrorWorkspaceNull;
    }

    MUTLASS_TRACE_HOST("  clearing workspace");
    musaError_t result = musaMemsetAsync(workspace, 0, workspace_size, stream);
    if (musaSuccess != result) {
      result = musaGetLastError(); // to clear the error bit
      MUTLASS_TRACE_HOST("  musaMemsetAsync() returned error " << musaGetErrorString(result));
      return Status::kErrorInternal;
    }
  }

  return Status::kSuccess;
}
#endif

#if !defined(__MUSACC_RTC__)
template <typename T>
Status
fill_workspace(void* workspace, T fill_value, size_t fill_count, musaStream_t stream = nullptr, MusaHostAdapter *musa_adapter = nullptr) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "Unsupported fill type");
  if (fill_count > 0) {
    if (workspace == nullptr) {
      MUTLASS_TRACE_HOST("  error: device workspace must not be null");
      return Status::kErrorWorkspaceNull;
    }

    MUTLASS_TRACE_HOST("  filling workspace");
    MUdeviceptr d_workspace = reinterpret_cast<MUdeviceptr>(workspace);

#if defined(MUTLASS_ENABLE_MUSA_HOST_ADAPTER) && MUTLASS_ENABLE_MUSA_HOST_ADAPTER

    //
    // Use the musa host adapter
    //
    MUTLASS_ASSERT(musa_adapter);
    if (musa_adapter) {
      Status status = Status::kErrorInternal;

      status = musa_adapter->memsetDevice(workspace, fill_value, fill_count, stream);

      if (status!=Status::kSuccess) {
        return Status::kErrorInternal;
      }
    }
    else {
      return Status::kErrorInternal;
    }
#else
    MUresult result = MUSA_SUCCESS;
    if (sizeof(T) == 4) {
      result = muMemsetD32Async(d_workspace, reinterpret_cast<uint32_t&>(fill_value), fill_count, stream);
    }
    else if (sizeof(T) == 2) {
      result = muMemsetD16Async(d_workspace, reinterpret_cast<uint16_t&>(fill_value), fill_count, stream);
    }
    else if (sizeof(T) == 1) {
      result = muMemsetD8Async(d_workspace, reinterpret_cast<uint8_t&>(fill_value), fill_count, stream);
    }

    if (MUSA_SUCCESS != result) {
      const char** error_string_ptr = nullptr;
      (void) muGetErrorString(result, error_string_ptr);
      if (error_string_ptr != nullptr) {
        MUTLASS_TRACE_HOST("  muMemsetD" << sizeof(T) * 8 << "Async() returned error " << *error_string_ptr);
      }
      else {
        MUTLASS_TRACE_HOST("  muMemsetD" << sizeof(T) * 8 << "Async() returned unrecognized error");
      }
      return Status::kErrorInternal;
    }
#endif
  }

  return Status::kSuccess;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass
