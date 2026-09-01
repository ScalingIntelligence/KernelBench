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
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "mutlass/mutlass.h"
#include "mute/tensor.hpp"
#include "mute/atom/mma_atom.hpp"

#include "mutlass/gemm/device/gemm_universal_adapter.h"
#include "default_gemm_configuration.hpp"

#include "../../common/mutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

using namespace mute;

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_Device_Gemm_f32n_f32n_f32n_simt_f32, 128x128x64_64x64x64) {
  constexpr int ThreadCount = 256;
  constexpr int AlignmentA = 1;
  constexpr int AlignmentB = 1;
  using TiledMma = TiledMMA<MMA_Atom<UniversalFMA<float, float, float, float>>,
                            Layout<Shape<_16, _16, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassSimt, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _128, _4>,
    float, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB>;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_Device_Gemm_f32n_f32t_f32n_simt_f32, 128x128x64_64x64x64) {
  constexpr int ThreadCount = 256;
  constexpr int AlignmentA = 1;
  constexpr int AlignmentB = 1;
  using TiledMma = TiledMMA<MMA_Atom<UniversalFMA<float, float, float, float>>,
                            Layout<Shape<_16, _16, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassSimt, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _128, _4>,
    float, mutlass::layout::ColumnMajor,
    float, mutlass::layout::RowMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB>;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_Device_Gemm_f32t_f32n_f32n_simt_f32, 128x128x64_64x64x64) {
  constexpr int ThreadCount = 256;
  constexpr int AlignmentA = 1;
  constexpr int AlignmentB = 1;
  using TiledMma = TiledMMA<MMA_Atom<UniversalFMA<float, float, float, float>>, 
                            Layout<Shape<_16, _16, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassSimt, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _128, _4>,
    float, mutlass::layout::RowMajor,
    float, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB>;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_Device_Gemm_f32t_f32t_f32n_simt_f32, 128x128x64_64x64x64) {
  constexpr int ThreadCount = 256;
  constexpr int AlignmentA = 1;
  constexpr int AlignmentB = 1;
  using TiledMma = TiledMMA<MMA_Atom<UniversalFMA<float, float, float, float>>, 
                            Layout<Shape<_16, _16, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassSimt, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _128, _4>,
    float, mutlass::layout::RowMajor,
    float, mutlass::layout::RowMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB>;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
