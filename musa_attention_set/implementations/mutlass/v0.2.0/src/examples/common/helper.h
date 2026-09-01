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
#pragma once

#include "musa_runtime.h"
#include <iostream>

/**
 * Panic wrapper for unwinding MUTLASS errors
 */
#define MUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    mutlass::Status error = status;                                                              \
    if (error != mutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got mutlass error: " << mutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


/**
 * Panic wrapper for unwinding MUSA runtime errors
 */
#define MUSA_CHECK(status)                                              \
  {                                                                     \
    musaError_t error = status;                                         \
    if (error != musaSuccess) {                                         \
      std::cerr << "Got bad musa status: " << musaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU stream
 */
struct GpuTimer
{
    musaStream_t _stream_id;
    musaEvent_t _start;
    musaEvent_t _stop;

    /// Constructor
    GpuTimer() : _stream_id(0)
    {
        MUSA_CHECK(musaEventCreate(&_start));
        MUSA_CHECK(musaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        MUSA_CHECK(musaEventDestroy(_start));
        MUSA_CHECK(musaEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(musaStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        MUSA_CHECK(musaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop()
    {
        MUSA_CHECK(musaEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        MUSA_CHECK(musaEventSynchronize(_stop));
        MUSA_CHECK(musaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};
