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
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>

#include <musa_runtime_api.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gets a MUSA device
musaDeviceProp GetMusaDevice();

/// Prints device properties
std::ostream &operator<<(std::ostream &out, musaDeviceProp const &device);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Sets flags for Unit test
void FilterArchitecture();

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Reads environment variable `MUTLASS_UNIT_TEST_PROBLEM_COUNT` to control the number and order
//  of problem sizes run by MUTLASS unit tests
int MutlassUnitTestProblemCount();

/////////////////////////////////////////////////////////////////////////////////////////////////

// active test macro
#define MUTLASS_TEST_LEVEL_ACTIVE(LEVEL,NAME_STATIC,NAME_DYNAMIC,...) \
    TEST(NAME_STATIC,L##LEVEL##_##NAME_DYNAMIC) __VA_ARGS__

// disabled test macro
#define MUTLASS_TEST_LEVEL_DISABLED(LEVEL,NAME_STATIC,NAME_DYNAMIC,...) \
    TEST(NAME_STATIC,DISABLED_L##LEVEL##_##NAME_DYNAMIC) {}

#if MUTLASS_TEST_LEVEL == 0
#define MUTLASS_TEST_L0(NAME_STATIC,NAME_DYNAMIC,...)   MUTLASS_TEST_LEVEL_ACTIVE(0,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define MUTLASS_TEST_L1(NAME_STATIC,NAME_DYNAMIC,...) MUTLASS_TEST_LEVEL_DISABLED(1,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define MUTLASS_TEST_L2(NAME_STATIC,NAME_DYNAMIC,...) MUTLASS_TEST_LEVEL_DISABLED(2,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#elif MUTLASS_TEST_LEVEL == 1
#define MUTLASS_TEST_L0(NAME_STATIC,NAME_DYNAMIC,...)   MUTLASS_TEST_LEVEL_ACTIVE(0,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define MUTLASS_TEST_L1(NAME_STATIC,NAME_DYNAMIC,...)   MUTLASS_TEST_LEVEL_ACTIVE(1,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define MUTLASS_TEST_L2(NAME_STATIC,NAME_DYNAMIC,...) MUTLASS_TEST_LEVEL_DISABLED(2,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#else
#define MUTLASS_TEST_L0(NAME_STATIC,NAME_DYNAMIC,...)   MUTLASS_TEST_LEVEL_ACTIVE(0,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define MUTLASS_TEST_L1(NAME_STATIC,NAME_DYNAMIC,...)   MUTLASS_TEST_LEVEL_ACTIVE(1,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#define MUTLASS_TEST_L2(NAME_STATIC,NAME_DYNAMIC,...)   MUTLASS_TEST_LEVEL_ACTIVE(2,NAME_STATIC,NAME_DYNAMIC,__VA_ARGS__)
#endif

#if !defined(MUTLASS_TEST_UNIT_ENABLE_WARNINGS)
#define MUTLASS_TEST_UNIT_ENABLE_WARNINGS false
#endif


#include <mutlass/mutlass.h>
#include <mutlass/numeric_types.h>
#include <mutlass/trace.h>

/////////////////////////////////////////////////////////////////////////////////////////////////
