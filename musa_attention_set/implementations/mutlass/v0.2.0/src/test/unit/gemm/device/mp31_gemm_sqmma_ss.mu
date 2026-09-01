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
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/epilogue/collective/collective_builder.hpp"
#include "mutlass/gemm/device/gemm_universal_adapter.h"
#include "default_gemm_configuration.hpp"
#include "../../common/mutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

using namespace mute;

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(MP31_gemm_sqmma_ss_F32F16F16F32_TNT, 384_256_32) {
  // A matrix configuration
  using         ElementA    = mutlass::half_t;                                // Element type for A matrix operand
  using         LayoutA     = mutlass::layout::RowMajor;                     // Layout type for A matrix operand
  static constexpr int AlignmentA  = 128 / mutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = mutlass::half_t;                                // Element type for B matrix operand
  using         LayoutB     = mutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
  static constexpr int AlignmentB  = 128 / mutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementC    = float;                                           // Element type for C and D matrix operands
  using         LayoutC     = mutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
  static constexpr int AlignmentC  = 128 / mutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementD    = float;                                            // Element type for C and D matrix operands
  using         LayoutD     = mutlass::layout::RowMajor;                     // Layout type for C and D matrix operands
  static constexpr int AlignmentD  = 128 / mutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // Core kernel configurations
  using ElementAccumulator  = float;                                           // Element type for internal accumulation
  using ElementCompute      = float;                                           // Element type for epilogue computation
  using ArchTag             = mutlass::arch::Mp31;                             // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = mutlass::arch::OpClassTensorOp;                  // Operator class tag
  using TileShape           = Shape<_384, _256, _32>;                          // Threadblock-level tile size
  using ClusterShape        = Shape<_1,_1,_1>;                                 // Shape of the threadblocks in a cluster
  using StageCountType      = mutlass::gemm::collective::StageCountAuto;       // Stage count maximized based on the tile size
  using KernelSchedule      = mutlass::gemm::KernelTme;                        // Kernel to launch
  using EpilogueSchedule    = mutlass::epilogue::WithTme;                      // Epilogue to launch
  using ThreadEpilogueOp    = mutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>;

  using CollectiveMainloop = typename mutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      StageCountType,
      KernelSchedule
    >::CollectiveOp;
  using CollectiveEpilogue = typename mutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        TileShape,
        TileShape,
        ElementD,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementD,
        LayoutD,
        AlignmentD,
        EpilogueSchedule,
        ThreadEpilogueOp,
        CollectiveMainloop
        >::CollectiveOp;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1, 1));
}

TEST(MP31_gemm_sqmma_ss_F32F16F16F32_TNN, 256_256_64) {
  // A matrix configuration
  using         ElementA    = mutlass::half_t;                                // Element type for A matrix operand
  using         LayoutA     = mutlass::layout::RowMajor;                      // Layout type for A matrix operand
  static constexpr int AlignmentA  = 128 / mutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = mutlass::half_t;                                // Element type for B matrix operand
  using         LayoutB     = mutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
  static constexpr int AlignmentB  = 128 / mutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementC    = float;                                // Element type for C and D matrix operands
  using         LayoutC     = mutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
  static constexpr int AlignmentC  = 128 / mutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementD    = float;                                // Element type for C and D matrix operands
  using         LayoutD     = mutlass::layout::ColumnMajor;                      // Layout type for C and D matrix operands
  static constexpr int AlignmentD  = 128 / mutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // Core kernel configurations
  using ElementAccumulator  = float;                                           // Element type for internal accumulation
  using ElementCompute      = float;                                           // Element type for epilogue computation
  using ArchTag             = mutlass::arch::Mp31;                             // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = mutlass::arch::OpClassTensorOp;                  // Operator class tag
  using TileShape           = Shape<_256, _256, _64>;                          // Threadblock-level tile size
  using ClusterShape        = Shape<_1,_1,_1>;                                 // Shape of the threadblocks in a cluster
  using StageCountType      = mutlass::gemm::collective::StageCountAuto;       // Stage count maximized based on the tile size
  using KernelSchedule      = mutlass::gemm::KernelTme;                        // Kernel to launch
  using EpilogueSchedule    = mutlass::epilogue::WithTme;                      // Epilogue to launch
  using ThreadEpilogueOp    = mutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>;

  using CollectiveMainloop = typename mutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      StageCountType,
      KernelSchedule
    >::CollectiveOp;
  using CollectiveEpilogue = typename mutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        TileShape,
        TileShape,
        ElementD,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementD,
        LayoutD,
        AlignmentD,
        EpilogueSchedule,
        ThreadEpilogueOp,
        CollectiveMainloop
        >::CollectiveOp;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1, 1));
}

TEST(MP31_gemm_sqmma_ss_F16F16F16F16_TTN, 256_64_64) {
  // A matrix configuration
  using         ElementA    = mutlass::half_t;                                // Element type for A matrix operand
  using         LayoutA     = mutlass::layout::RowMajor;                      // Layout type for A matrix operand
  static constexpr int AlignmentA  = 128 / mutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = mutlass::half_t;                                // Element type for B matrix operand
  using         LayoutB     = mutlass::layout::RowMajor;                   // Layout type for B matrix operand
  static constexpr int AlignmentB  = 128 / mutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementC    = mutlass::half_t;                                // Element type for C and D matrix operands
  using         LayoutC     = mutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
  static constexpr int AlignmentC  = 128 / mutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementD    = half_t;                                // Element type for C and D matrix operands
  using         LayoutD     = mutlass::layout::ColumnMajor;                      // Layout type for C and D matrix operands
  static constexpr int AlignmentD  = 128 / mutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // Core kernel configurations
  using ElementAccumulator  = float;                                           // Element type for internal accumulation
  using ElementCompute      = float;                                           // Element type for epilogue computation
  using ArchTag             = mutlass::arch::Mp31;                             // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = mutlass::arch::OpClassTensorOp;                  // Operator class tag
  using TileShape           = Shape<_256, _64, _64>;                          // Threadblock-level tile size
  using ClusterShape        = Shape<_1,_1,_1>;                                 // Shape of the threadblocks in a cluster
  using StageCountType      = mutlass::gemm::collective::StageCountAuto;       // Stage count maximized based on the tile size
  using KernelSchedule      = mutlass::gemm::KernelTme;                        // Kernel to launch
  using EpilogueSchedule    = mutlass::epilogue::WithTme;                      // Epilogue to launch
  using ThreadEpilogueOp    = mutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>;

  using CollectiveMainloop = typename mutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      StageCountType,
      KernelSchedule
    >::CollectiveOp;
  using CollectiveEpilogue = typename mutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        TileShape,
        TileShape,
        ElementD,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementD,
        LayoutD,
        AlignmentD,
        EpilogueSchedule,
        ThreadEpilogueOp,
        CollectiveMainloop
        >::CollectiveOp;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1, 1));
}