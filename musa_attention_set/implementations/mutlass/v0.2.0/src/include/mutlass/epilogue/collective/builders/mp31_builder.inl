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
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "mute/atom/mma_traits_mp31.hpp"
#include "mute/atom/mma_traits_mp31_sqmma.hpp"
#include "mute/atom/copy_traits_mp31.hpp"

#include "mutlass/detail/dependent_false.hpp"
#include "mutlass/detail/layout.hpp"
#include "mutlass/epilogue/dispatch_policy.hpp"
#include "mutlass/epilogue/collective/collective_epilogue.hpp"
#include "mutlass/epilogue/thread/linear_combination.h"
#include "mutlass/epilogue/fusion/callbacks.hpp"
#include "mutlass/epilogue/collective/builders/mp31_builder_common.inl"

#if defined(__MUSACC_RTC__)
#include <musa/std/type_traits>
#else
#include <type_traits>
#endif
using namespace mute;
///////////////////////////////////////////////////////////////////////////////

namespace mutlass::epilogue::collective {

///////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the parameterized dispatch policy for the TME epilogue
template<class TileShapeMNK, class EpilogueTileMN,  class ElementC, class ElementD>
constexpr auto
mp31_get_tme_dispatch_policy() {
  using AccAtomM = _16;
  using AccAtomN =  _8;
  auto EpilogueShapeMN = make_shape(size<0>(EpilogueTileMN{}), size<1>(EpilogueTileMN{}));
  constexpr int EpiTiles = size(shape_div(take<0,2>(TileShapeMNK{}), EpilogueShapeMN));
  constexpr int AtomM = get<0>(shape<0>(EpilogueTileMN{}));
  constexpr int AtomN = get<0>(shape<1>(EpilogueTileMN{}));
  constexpr int FragmentSize = (AtomN / AccAtomN{}) * (AtomM / AccAtomM{});
  constexpr bool ReuseSmem = (sizeof_bits_v<ElementC> == sizeof_bits_v<ElementD>) && (sizeof_bits_v<ElementD> > 8);
  constexpr int StagesD = mute::min(EpiTiles, 3);
  constexpr int StagesC = ReuseSmem ? mute::max(mute::min(EpiTiles, 3), StagesD)
                                    : mute::min(EpiTiles, 2);
  constexpr bool DelayTmeStore = (mute::is_void_v<ElementC> || not ReuseSmem) && StagesD > 1;
  return Mp31CollectiveEpilogue<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmeStore>{};
}

// Returns the smem layout atom to be used for C or D matrix
template<class GmemStrideType, class Element, class EpilogueTile_MN>
constexpr auto
mp31_get_epilogue_smem_swizzle_layout_atom() {
  // ColMajor C/D (M-major)
  if constexpr (mutlass::gemm::detail::is_major<0>(GmemStrideType{})) {
    return mutlass::epilogue::collective::detail::ss_smem_selector_C<
      mute::TCE::Major::MN, Element, EpilogueTile_MN
    >();
  }
  // RowMajor C/D (N-major)
  else if constexpr (mutlass::gemm::detail::is_major<1>(GmemStrideType{})) {
    return mutlass::epilogue::collective::detail::ss_smem_selector_C<
      mute::TCE::Major::K , Element, EpilogueTile_MN
    >();
  }
  else {
    static_assert(mutlass::detail::dependent_false<GmemStrideType>, "Unsupported gmem layout.");
  }
}

// Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.
template <class TileShape_MNK, class AtomShape_MNK>
constexpr auto
mp31_compute_tile_shape() {
  constexpr int tile_m = size<0>(TileShape_MNK{});
  constexpr int tile_n = size<1>(TileShape_MNK{});
  constexpr int atom_tile_m = size<0>(AtomShape_MNK{});
  constexpr int atom_tile_n = size<1>(AtomShape_MNK{});
  constexpr int atom_m = tile_m / atom_tile_m;
  constexpr int atom_n = tile_n / atom_tile_n;

  if constexpr (atom_m > 1 && atom_n == 1) {
    using tiler_m = Layout<Shape <Int<atom_tile_m>, Int<atom_m>>,
                           Stride<              _1, Int<atom_tile_m>>>;
    using tiler_n = Layout<Shape <Int<8>, Int<atom_n>>,
                           Stride<    _1, Int<atom_tile_n>>>;
    return make_tile(tiler_m{}, tiler_n{});
  }
  else {
    using tiler_m = Layout<Shape <_16, Int<atom_m>>,
                           Stride< _1, Int<atom_tile_m>>>;
    using tiler_n = Layout<Shape <Int<atom_tile_n>, Int<atom_n>>,
                           Stride<              _1, Int<atom_tile_n>>>;
    return make_tile(tiler_m{}, tiler_n{});
  }

}

// Helper for building TME collective epilogues, specialized by
// the fusion operation performed and the dispatch policy to use.
template <
  class TileShape_MNK,
  class EpilogueTile_MN,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD_,
  class GmemLayoutTagD,
  int AlignmentD,
  class ThreadOp,
  class DispatchPolicy
>
struct Mp31TmeBuilderImpl {

  using ElementD = ElementD_;
  // Passing void C disables source load + smem allocation
  using ElementC = mute::conditional_t<mute::is_void_v<ElementC_>,ElementD,ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = mute::conditional_t<mute::is_void_v<ElementC_>,GmemLayoutTagD,GmemLayoutTagC_>;

  using GmemStrideTypeC = mutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = mutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  using UnderlyingGmemStrideTypeC = mute::remove_pointer_t<GmemStrideTypeC>;
  using UnderlyingGmemStrideTypeD = mute::remove_pointer_t<GmemStrideTypeD>;

  using CopyOpS2G = MP31_TME_STORE;
  using CopyOpG2S = MP31_TME_LOAD;

  using CopyOpS2R = Copy_Atom<DefaultCopy,ElementC>;
  using CopyOpR2S = Copy_Atom<UniversalCopy<ElementD>, ElementD>;

  using CollectiveOp = mutlass::epilogue::collective::CollectiveEpilogue<
      DispatchPolicy,
      ThreadOp,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      CopyOpG2S,
      CopyOpS2G,
      decltype(detail::mp31_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      decltype(detail::mp31_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      CopyOpS2R,
      CopyOpR2S
    >;
};
} // namespace detail

// No-smem builder
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
  class Schedule,
  class MainloopCollective,
  FloatRoundStyle RoundStyle
>
struct CollectiveBuilder<
    arch::Mp31,
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
    Schedule,
    fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute,RoundStyle>,
    MainloopCollective,
    mute::enable_if_t<mute::is_same_v<Schedule, NoSmem>>> {

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
    ScaleType, RoundStyle, ElementC>;

  using CollectiveOp = mutlass::epilogue::collective::DefaultEpilogue<
        mutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
        mutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
        ThreadOp,
        mutlass::gemm::EpilogueDefault>;
};

// Tme builder
template <
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class MainloopCollective
>
struct CollectiveBuilder<
    arch::Mp31,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>,
    MainloopCollective,
    mute::enable_if_t<mute::is_same_v<Schedule, WithTme> &&
    mute::is_void_v<mute::void_t<typename MainloopCollective::TiledMma>>>
    > {
private:

  using TiledMma = typename MainloopCollective::TiledMma;
  using AtomShape_MNK = typename TiledMma::AtomShape_MNK;
  using ThreadOp = thread::LinearCombination<ElementD, 1, ElementAccumulator, ElementCompute>;
  using EpilogueTile_MN = decltype(detail::mp31_compute_tile_shape<TileShape_MNK, AtomShape_MNK>());
  using DispatchPolicy =
    decltype(detail::mp31_get_tme_dispatch_policy<TileShape_MNK, EpilogueTile_MN, ElementC, ElementD>());

public:
  using CollectiveOp =
    typename detail::Mp31TmeBuilderImpl<
      TileShape_MNK,
      EpilogueTile_MN,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      GmemLayoutTagC,
      AlignmentC,
      ElementD,
      GmemLayoutTagD,
      AlignmentD,
      ThreadOp,
      DispatchPolicy
    >::CollectiveOp;
};

// Auto builder
template <
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOperation,
  class MainloopCollective
>
struct CollectiveBuilder<
    arch::Mp31,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleAuto,
    FusionOperation,
    MainloopCollective,
    void> {
private:
  static_assert(mute::is_same_v<FusionOperation, fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>>,
                "Auto schedule doesn't support fusion. Use one of the TME schedules instead.");

  // Pick No-Smem epilogue as the Auto Epilogue Schedule (Auto schedules do not guarantee best performance)
  // since TME epilogues are not compatible with non-TME non-WS mainloops
  using EpilogueSchedule = NoSmem;
  using _CollectiveBuilder = CollectiveBuilder<
    arch::Mp31,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueSchedule,
    FusionOperation,
    MainloopCollective
  >;

public:
  using CollectiveOp = typename _CollectiveBuilder::CollectiveOp;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::epilogue::collective
