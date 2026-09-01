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
#pragma once

#include "mute/atom/mma_atom.hpp"
#include "mute/atom/copy_atom.hpp"

#include "mutlass/mutlass.h"
#include "mutlass/gemm/gemm.h"
#include "mutlass/arch/arch.h"
#include "mutlass/arch/mma.h"
#include "mutlass/layout/layout.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/epilogue/collective/collective_epilogue.hpp"
#include "mutlass/epilogue/collective/default_epilogue.hpp"
#include "mutlass/epilogue/thread/linear_combination.h"

namespace mutlass::epilogue::collective {

template <
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class EpilogueScheduleType,
  class MainloopCollective,
  FloatRoundStyle RoundStyle
>
struct CollectiveBuilder<
  mutlass::arch::Mp22,
  OpClass,
  TileShape_MNK,
  ClusterShape_MNK,
  EpilogueTileType,
  ElementAccumulator,
  ElementCompute,
  ElementC_,
  GmemLayoutTagC_,
  AlignmentC,
  ElementD,
  GmemLayoutTagD,
  AlignmentD,
  EpilogueScheduleType,
  fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute,RoundStyle>,
  MainloopCollective,
  void
> {
  // Passing void C disables source load
  using ElementC = mute::conditional_t<mute::is_void_v<ElementC_>,
    ElementD, ElementC_>; // prevents mute breakages
  using GmemLayoutTagC = mute::conditional_t<mute::is_void_v<ElementC_>,
    GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = mute::is_void_v<ElementC_> ?
    thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

  static constexpr int FragmentSize = 1;
  using ThreadOp = thread::LinearCombination<
    ElementD, FragmentSize, ElementAccumulator, ElementCompute,
    ScaleType, FloatRoundStyle::round_to_nearest, ElementC>;

  using CollectiveOp = mutlass::epilogue::collective::DefaultEpilogue<
                        mutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
                        mutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
                        ThreadOp,
                        mutlass::gemm::EpilogueDefault
                      >;
};

} // namespace mutlass::epilogue::collective
