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

#include "mutlass/arch/arch.h"
#include "mutlass/gemm/gemm.h"

#include "mute/layout.hpp"
#include "mute/numeric/integral_constant.hpp"
//////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm {
using namespace mute;

//////////////////////////////////////////////////////////////////////////////

//
// Policies for categorical dispatch of mainloop against kernel grid schedules
//
struct KernelMultistage { };
struct KernelTme { };
struct KernelTmeWarpSpecialized { };

template <
  int ScaleGranularityM = 0,
  int ScaleGranularityN = 0,
  int ScaleGranularityK = 0
>
struct KernelTmeGroupScaledAccum : KernelTme { };

template <
  int ScaleGranularityM = 0,
  int ScaleGranularityN = 0,
  int ScaleGranularityK = 0
>
struct KernelTmeWarpSpecializedScaledAccum : KernelTmeWarpSpecialized { };

// Policies for dispatch of epilogue
struct EpilogueDefault { };
struct EpilogueTransposed { };

//
// Collective Mainloop Policies
//

// 2 stage pipeline through 1 stage in smem, 1 in rmem, WITHOUT predicated gmem loads
struct MainloopMp22TwoStageUnpredicated {
  constexpr static int Stages = 2;
  using ArchTag = arch::Mp22;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// 2 stage pipeline through 1 stage in smem, 1 in rmem, with predicated gmem loads
struct MainloopMp22TwoStage {
  constexpr static int Stages = 2;
  using ArchTag = arch::Mp22;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// n-buffer in smem(TME), pipelined with SQMMA and TME, static schedule between TME and SQMMA
template <
  int Stages_,
  class KernelSchedule = KernelTme>
struct MainloopMp31TmeSqmma {
  constexpr static int Stages = Stages_;
  using ArchTag = arch::Mp31;
  using Schedule = KernelSchedule;
  using ClusterShape = Shape<_1,_1,_1>;
};

template <
  int Stages_,
  class KernelSchedule = KernelTmeWarpSpecialized>
struct MainloopMp31TmeSqmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ArchTag = arch::Mp31;
  using Schedule = KernelSchedule;
  using ClusterShape = Shape<_1,_1,_1>;
};

template <
  int Stages_,
  class KernelSchedule = KernelTme,
  int ScaleGranularityM = 0, // `ScaleGranularityM` specifies scaling granularity along M, while zero-value `ScaleGranularityM` indicates that scaling granularity is `size<0>(TileShape_MNK{})` along M
  int ScaleGranularityN = 0, // `ScaleGranularityN` specifies scaling granularity along N, while zero-value `ScaleGranularityN` indicates that scaling granularity is `size<1>(TileShape_MNK{})` along N
  int ScaleGranularityK = 0
>
struct MainloopMp31TmeSqmmaBlockScalingFP8
  : MainloopMp31TmeSqmma<Stages_, KernelSchedule> {
  static_assert(
    mute::is_same_v<KernelSchedule, KernelTmeGroupScaledAccum<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>>,
    "Kernel Schedule must be the scaled accum policy.");
};

template <
  int Stages_,
  class KernelSchedule = KernelTmeWarpSpecialized,
  int ScaleGranularityM = 0, // `ScaleGranularityM` specifies scaling granularity along M, while zero-value `ScaleGranularityM` indicates that scaling granularity is `size<0>(TileShape_MNK{})` along M
  int ScaleGranularityN = 0, // `ScaleGranularityN` specifies scaling granularity along N, while zero-value `ScaleGranularityN` indicates that scaling granularity is `size<1>(TileShape_MNK{})` along N
  int ScaleGranularityK = 0
>
struct MainloopMp31TmeSqmmaBlockWarpSpecializedScalingFP8
  : MainloopMp31TmeSqmmaWarpSpecialized<Stages_, KernelSchedule> {
  static_assert(
    mute::is_same_v<KernelSchedule, KernelTmeWarpSpecializedScaledAccum<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>>,
    "Kernel Schedule must be the scaled accum policy.");
};
//////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm
