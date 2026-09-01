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

TEST(MP22_gemm_tensorop_F32TF32TF32F32_NN, 128_128x32x16) {
  constexpr int ThreadCount = 128;
  constexpr int AlignmentA = 4;
  constexpr int AlignmentB = 4;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x8_F32TF32TF32F32_NN>,
                            Layout<Shape<_1, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _32, _16>,
    tfloat32_t, mutlass::layout::ColumnMajor,
    tfloat32_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32F16F16F32_NN, 128_128x32x32) {
  constexpr int ThreadCount = 128;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32F16F16F32_NN>,
                            Layout<Shape<_1, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _32, _32>,
    half_t, mutlass::layout::ColumnMajor,
    half_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32BF16BF16F32_NN, 128_128x32x32) {
  constexpr int ThreadCount = 128;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32BF16BF16F32_NN>,
                            Layout<Shape<_1, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _32, _32>,
    bfloat16_t, mutlass::layout::ColumnMajor,
    bfloat16_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_S32S8S8S32_NN, 128_128x32x64) {
  constexpr int ThreadCount = 128;
  constexpr int AlignmentA = 16;
  constexpr int AlignmentB = 16;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x32_S32S8S8S32_NN>,
                            Layout<Shape<_1, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _32, _64>,
    int8_t, mutlass::layout::ColumnMajor,
    int8_t, mutlass::layout::ColumnMajor,
    int32_t, mutlass::layout::ColumnMajor,
    int32_t,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32TF32TF32F32_NN, 256_256x128x16) {
  constexpr int ThreadCount = 256;
  constexpr int AlignmentA = 4;
  constexpr int AlignmentB = 4;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x8_F32TF32TF32F32_NN>,
                            Layout<Shape<_2, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _128, _16>,
    tfloat32_t, mutlass::layout::ColumnMajor,
    tfloat32_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32F16F16F32_NN, 256_256x128x32) {
  constexpr int ThreadCount = 256;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32F16F16F32_NN>,
                            Layout<Shape<_2, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _128, _32>,
    half_t, mutlass::layout::ColumnMajor,
    half_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32BF16BF16F32_NN, 256_256x128x32) {
  constexpr int ThreadCount = 256;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32BF16BF16F32_NN>,
                            Layout<Shape<_2, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _128, _32>,
    bfloat16_t, mutlass::layout::ColumnMajor,
    bfloat16_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// TEST(MP22_gemm_tensorop_S32S8S8S32_NN, 256_256x128x64) {
//   constexpr int ThreadCount = 256;
//   constexpr int AlignmentA = 16;
//   constexpr int AlignmentB = 16;
//   using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x32_S32S8S8S32_NN>,
//                             Layout<Shape<_2, _1, _1>>>;
//   using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
//     mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
//     TiledMma,
//     Shape<_256, _128, _64>,
//     int8_t, mutlass::layout::ColumnMajor,
//     int8_t, mutlass::layout::ColumnMajor,
//     int32_t, mutlass::layout::ColumnMajor,
//     int32_t,
//     ThreadCount,
//     AlignmentA, AlignmentB
//     >;

//   using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
//       Shape<int,int,int,int>,
//       Config::CollectiveMainloop,
//       Config::DefaultCollectiveEpilogue
//   >;

//   using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
//   EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
// }

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32TF32TF32F32_NN, 512_256x256x16) {
  constexpr int ThreadCount = 512;
  constexpr int AlignmentA = 4;
  constexpr int AlignmentB = 4;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x8_F32TF32TF32F32_NN>,
                            Layout<Shape<_2, _2, _1>,Stride<_1, _2, _0>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _256, _16>,
    tfloat32_t, mutlass::layout::ColumnMajor,
    tfloat32_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}
///////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32F16F16F32_NN, 512_256x256x32) {
  constexpr int ThreadCount = 512;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32F16F16F32_NN>,
                            Layout<Shape<_2, _2, _1>,Stride<_2, _1, _0>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _256, _32>,
    half_t, mutlass::layout::ColumnMajor,
    half_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32BF16BF16F32_NN, 512_256x256x32) {
  constexpr int ThreadCount = 512;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32BF16BF16F32_NN>,
                            Layout<Shape<_2, _2, _1>,Stride<_2, _1, _0>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _256, _32>,
    bfloat16_t, mutlass::layout::ColumnMajor,
    bfloat16_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// TEST(MP22_gemm_tensorop_S32S8S8S32_NN, 512_256x256x64) {
//   constexpr int ThreadCount = 512;
//   constexpr int AlignmentA = 16;
//   constexpr int AlignmentB = 16;
//   using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x32_S32S8S8S32_NN>,
//                             Layout<Shape<_2, _2, _1>,Stride<_1, _2, _0>>>;
//   using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
//     mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
//     TiledMma,
//     Shape<_256, _256, _64>,
//     int8_t, mutlass::layout::ColumnMajor,
//     int8_t, mutlass::layout::ColumnMajor,
//     int32_t, mutlass::layout::ColumnMajor,
//     int32_t,
//     ThreadCount,
//     AlignmentA, AlignmentB
//     >;

//   using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
//       Shape<int,int,int,int>,
//       Config::CollectiveMainloop,
//       Config::DefaultCollectiveEpilogue
//   >;

//   using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
//   EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
// }


// TN
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32TF32TF32F32_TN, 128_128x32x16) {
  constexpr int ThreadCount = 128;
  constexpr int AlignmentA = 4;
  constexpr int AlignmentB = 4;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x8_F32TF32TF32F32_TN>,
                            Layout<Shape<_1, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _32, _16>,
    tfloat32_t, mutlass::layout::RowMajor,
    tfloat32_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32F16F16F32_TN, 128_128x32x32) {
  constexpr int ThreadCount = 128;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32F16F16F32_TN>,
                            Layout<Shape<_1, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _32, _32>,
    half_t, mutlass::layout::RowMajor,
    half_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32BF16BF16F32_TN, 128_128x32x32) {
  constexpr int ThreadCount = 128;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32BF16BF16F32_TN>,
                            Layout<Shape<_1, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _32, _32>,
    bfloat16_t, mutlass::layout::RowMajor,
    bfloat16_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_S32S8S8S32_TN, 128_128x32x64) {
  constexpr int ThreadCount = 128;
  constexpr int AlignmentA = 16;
  constexpr int AlignmentB = 16;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x32_S32S8S8S32_TN>,
                            Layout<Shape<_1, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_128, _32, _64>,
    int8_t, mutlass::layout::RowMajor,
    int8_t, mutlass::layout::ColumnMajor,
    int32_t, mutlass::layout::ColumnMajor,
    int32_t,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32TF32TF32F32_TN, 256_256x128x16) {
  constexpr int ThreadCount = 256;
  constexpr int AlignmentA = 4;
  constexpr int AlignmentB = 4;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x8_F32TF32TF32F32_TN>,
                            Layout<Shape<_2, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _128, _16>,
    tfloat32_t, mutlass::layout::RowMajor,
    tfloat32_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32F16F16F32_TN, 256_256x128x32) {
  constexpr int ThreadCount = 256;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32F16F16F32_TN>,
                            Layout<Shape<_2, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _128, _32>,
    half_t, mutlass::layout::RowMajor,
    half_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32BF16BF16F32_TN, 256_256x128x32) {
  constexpr int ThreadCount = 256;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32BF16BF16F32_TN>,
                            Layout<Shape<_2, _1, _1>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _128, _32>,
    bfloat16_t, mutlass::layout::RowMajor,
    bfloat16_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// TEST(MP22_gemm_tensorop_S32S8S8S32_TN, 256_256x128x64) {
//   constexpr int ThreadCount = 256;
//   constexpr int AlignmentA = 16;
//   constexpr int AlignmentB = 16;
//   using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x32_S32S8S8S32_TN>,
//                             Layout<Shape<_2, _1, _1>>>;
//   using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
//     mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
//     TiledMma,
//     Shape<_256, _128, _64>,
//     int8_t, mutlass::layout::RowMajor,
//     int8_t, mutlass::layout::ColumnMajor,
//     int32_t, mutlass::layout::ColumnMajor,
//     int32_t,
//     ThreadCount,
//     AlignmentA, AlignmentB
//     >;

//   using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
//       Shape<int,int,int,int>,
//       Config::CollectiveMainloop,
//       Config::DefaultCollectiveEpilogue
//   >;

//   using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
//   EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
// }

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32TF32TF32F32_TN, 512_256x256x16) {
  constexpr int ThreadCount = 512;
  constexpr int AlignmentA = 4;
  constexpr int AlignmentB = 4;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x8_F32TF32TF32F32_TN>,
                            Layout<Shape<_2, _2, _1>,Stride<_1, _2, _0>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _256, _16>,
    tfloat32_t, mutlass::layout::RowMajor,
    tfloat32_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}
///////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32F16F16F32_TN, 512_256x256x32) {
  constexpr int ThreadCount = 512;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32F16F16F32_TN>,
                            Layout<Shape<_2, _2, _1>,Stride<_2, _1, _0>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _256, _32>,
    half_t, mutlass::layout::RowMajor,
    half_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP22_gemm_tensorop_F32BF16BF16F32_TN, 512_256x256x32) {
  constexpr int ThreadCount = 512;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x16_F32BF16BF16F32_TN>,
                            Layout<Shape<_2, _2, _1>,Stride<_2, _1, _0>>>;
  using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
    mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
    TiledMma,
    Shape<_256, _256, _32>,
    bfloat16_t, mutlass::layout::RowMajor,
    bfloat16_t, mutlass::layout::ColumnMajor,
    float, mutlass::layout::ColumnMajor,
    float,
    ThreadCount,
    AlignmentA, AlignmentB
    >;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::DefaultCollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// TEST(MP22_gemm_tensorop_S32S8S8S32_TN, 512_256x256x64) {
//   constexpr int ThreadCount = 512;
//   constexpr int AlignmentA =16;
//   constexpr int AlignmentB =16;
//   using TiledMma = TiledMMA<MMA_Atom<MP22_32x32x32_S32S8S8S32_TN>,
//                             Layout<Shape<_2, _2, _1>,Stride<_1, _2, _0>>>;
//   using Config = mutlass::gemm::device::DefaultGemmConfigurationToMutlass3Types<
//     mutlass::arch::OpClassTensorOp, mutlass::arch::Mp22,
//     TiledMma,
//     Shape<_256, _256, _64>,
//     int8_t, mutlass::layout::RowMajor,
//     int8_t, mutlass::layout::ColumnMajor,
//     int32_t, mutlass::layout::ColumnMajor,
//     int32_t,
//     ThreadCount,
//     AlignmentA, AlignmentB
//     >;

//   using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
//       Shape<int,int,int,int>,
//       Config::CollectiveMainloop,
//       Config::DefaultCollectiveEpilogue
//   >;

//   using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
//   EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
// }

/////////////////////////////////////////////////////////////////////////////////////////////////
