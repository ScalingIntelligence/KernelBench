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
#include "mutlass/gemm/collective/collective_mma.hpp"
#include "mutlass/gemm/collective/builders/common.inl"

namespace mutlass::gemm::collective {

namespace detail {

template <class Element, class StrideAB>
constexpr auto make_mp22_smem_atom_layout() {
  constexpr int size = sizeof(Element);

  using half_inst_mn = Int<16>;
  using half_inst_k = Int<16/size>;
  using inst_k = Int<32/size>;

  if constexpr (gemm::detail::is_mn_major<StrideAB>()) {
    using shape = Shape<half_inst_mn, inst_k>;
    return make_ordered_layout(shape{}, Step<_0, _1>{});
  } else {
    using shape = Shape<half_inst_mn, Shape<half_inst_k, _2>>;
    return make_ordered_layout(shape{}, Step<_1, Step<_0, _2>>{});
  }
};

template <
  class ElementA, class StrideA,
  class ElementB, class StrideB>
constexpr auto mp22_mma_operation_select()
{
  static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be same");
  static_assert((gemm::detail::is_mn_major<StrideA>() || gemm::detail::is_k_major<StrideA>()) &&
                (gemm::detail::is_mn_major<StrideB>() || gemm::detail::is_k_major<StrideB>()),
                 "StrideA/B must be MN or K major");
  // tfloat32
  if constexpr (is_same_v<ElementA, tfloat32_t>) {
    if constexpr (gemm::detail::is_mn_major<StrideA>()) {
      if constexpr (gemm::detail::is_k_major<StrideB>()) {
        return MP22_32x32x8_F32TF32TF32F32_NN{};
      }
      else {
        return MP22_32x32x8_F32TF32TF32F32_NT{};
      }
    } else {
      if constexpr (gemm::detail::is_k_major<StrideB>()) {
        return MP22_32x32x8_F32TF32TF32F32_TN{};
      }
      else {
        return MP22_32x32x8_F32TF32TF32F32_TT{};
      }
    }
  }
  // half
  else if constexpr (is_same_v<ElementA, half_t>) {
    if constexpr (gemm::detail::is_mn_major<StrideA>()) {
      if constexpr (gemm::detail::is_k_major<StrideB>()) {
        return MP22_32x32x16_F32F16F16F32_NN{};
      }
      else {
        return MP22_32x32x16_F32F16F16F32_NT{};
      }
    } else {
      if constexpr (gemm::detail::is_k_major<StrideB>()) {
        return MP22_32x32x16_F32F16F16F32_TN{};
      }
      else {
        return MP22_32x32x16_F32F16F16F32_TT{};
      }
    }
  }
  // bfloat16
  else if constexpr (is_same_v<ElementA, bfloat16_t>) {
    if constexpr (gemm::detail::is_mn_major<StrideA>()) {
      if constexpr (gemm::detail::is_k_major<StrideB>()) {
        return MP22_32x32x16_F32BF16BF16F32_NN{};
      }
      else {
        return MP22_32x32x16_F32BF16BF16F32_NT{};
      }
    } else {
      if constexpr (gemm::detail::is_k_major<StrideB>()) {
        return MP22_32x32x16_F32BF16BF16F32_TN{};
      }
      else {
        return MP22_32x32x16_F32BF16BF16F32_TT{};
      }
    }
  }
  // int8
  else if constexpr (is_same_v<ElementA, int8_t>) {
    if constexpr (gemm::detail::is_mn_major<StrideA>()) {
      if constexpr (gemm::detail::is_k_major<StrideB>()) {
        return MP22_32x32x32_S32S8S8S32_NN{};
      }
      else {
        return MP22_32x32x32_S32S8S8S32_NT{};
      }
    } else {
      if constexpr (gemm::detail::is_k_major<StrideB>()) {
        return MP22_32x32x32_S32S8S8S32_TN{};
      }
      else {
        return MP22_32x32x32_S32S8S8S32_TT{};
      }
    }
  } else {
    static_assert(sizeof(ElementA) == 0, "No eligible MMA operator for request configuration.");
  }
}

template <
  class TileShape_MNK,
  class ElementA,
  class ElementB,
  class ElementAccumulator>
constexpr auto
mp22_make_simt_tiled_mma() {
  static_assert(sizeof(ElementA) == 4 && sizeof(ElementB) == 4);
  static_assert(sizeof(ElementAccumulator) == 4);

  constexpr int TileM = size<0>(TileShape_MNK{});
  constexpr int TileN = size<1>(TileShape_MNK{});
  constexpr int TileK = size<2>(TileShape_MNK{});

  constexpr int MaxAccumPerThread = 64;
  constexpr int MaxFmaInstInHotloop = 256;
  constexpr int PreferedLdsWidth = 4;

  constexpr int ThreadCountForOccupancy = mute::max(TileM * TileN / MaxAccumPerThread, NumThreadsPerWarpBeforeMP31);
  constexpr int ThreadCount =
    ThreadCountForOccupancy * mute::max(TileM * TileN * TileK / (ThreadCountForOccupancy * MaxFmaInstInHotloop), 1);

  constexpr int MaxLdsTile = ThreadCount == 512 ? 128 : 64;
  static_assert(ThreadCount * PreferedLdsWidth <= (TileM * TileN), "Too small tile shape");
  if constexpr (TileM > TileN) {
    constexpr int ThreadAtomM = TileM > MaxLdsTile ? (MaxLdsTile / PreferedLdsWidth) : (TileM / PreferedLdsWidth);
    constexpr int ThreadAtomN = ThreadCount / ThreadAtomM;
    constexpr int ValueN = (TileN / ThreadAtomN) > PreferedLdsWidth ? PreferedLdsWidth : (TileN / ThreadAtomN);

    using AtomLayout = Layout<Shape<Int<ThreadAtomM>, Int<ThreadAtomN>, _1>>;
    using PermuteLayout = mute::Tile<
            Layout<Shape<Int<ThreadAtomM>, Int<PreferedLdsWidth>>, Stride<          Int<ValueN>, _1>>,
            Layout<Shape<Int<ThreadAtomN>,           Int<ValueN>>, Stride<Int<PreferedLdsWidth>, _1>>,
            X>;

    return TiledMMA<MMA_Atom<UniversalFMA<
                                ElementAccumulator,
                                ElementA,
                                ElementB,
                                ElementAccumulator>>,
                    AtomLayout, PermuteLayout>{};
  } else {
    constexpr int ThreadAtomN = TileN > MaxLdsTile ? (MaxLdsTile / PreferedLdsWidth) : (TileN / PreferedLdsWidth);
    constexpr int ThreadAtomM = ThreadCount / ThreadAtomN;
    constexpr int ValueM = (TileM / ThreadAtomM) > PreferedLdsWidth ? PreferedLdsWidth : (TileM / ThreadAtomM);
    using AtomLayout = Layout<Shape<Int<ThreadAtomM>, Int<ThreadAtomN>, _1>>;

    using PermuteLayout = mute::Tile<
            Layout<Shape<Int<ThreadAtomM>,           Int<ValueM>>, Stride<          Int<ValueM>, _1>>,
            Layout<Shape<Int<ThreadAtomN>, Int<PreferedLdsWidth>>, Stride<Int<PreferedLdsWidth>, _1>>,
            X>;

    return TiledMMA<MMA_Atom<UniversalFMA<
                                ElementAccumulator,
                                ElementA,
                                ElementB,
                                ElementAccumulator>>,
                    AtomLayout, PermuteLayout>{};
  }
}

template <class TileShape_MNK, class MmaOp>
constexpr auto
mp22_make_tensorop_tiled_mma() {
  constexpr int TileM = size<0>(TileShape_MNK{});
  constexpr int TileN = size<1>(TileShape_MNK{});

  constexpr int MmaOpShapeM = size<0>(typename MMA_Traits<MmaOp>::Shape_MNK{});
  constexpr int MmaOpShapeN = size<1>(typename MMA_Traits<MmaOp>::Shape_MNK{});

  static_assert(TileM % MmaOpShapeM == 0 && TileN % MmaOpShapeN == 0, "Invalid TileShape");

  constexpr int MaxWarpTile = 128;

  constexpr int ValueLayoutM = mute::min(MaxWarpTile, TileM) / MmaOpShapeM;
  constexpr int ValueLayoutN = mute::min(MaxWarpTile, TileN) / MmaOpShapeN;

  constexpr int AtomTileM = ValueLayoutM * MmaOpShapeM;
  constexpr int AtomTileN = ValueLayoutN * MmaOpShapeN;

  constexpr int AtomExtendM = TileM / AtomTileM;
  constexpr int AtomExtendN = TileN / AtomTileN;

  using AtomLayout = decltype(make_layout(Shape<Int<AtomExtendM>, Int<AtomExtendN>, _1>{}, LayoutLeft{}));

  using PermuteLayoutK = X;

  using ShapeM0 = Int<MmaOpShapeM>;
  using ShapeM1 = Int<size<0>(AtomLayout{})>;
  using ShapeM2 = Int<ValueLayoutM>;

  using StrideM0 = _1;
  using StrideM1 = Int<MmaOpShapeM * ValueLayoutM>;
  using StrideM2 = Int<MmaOpShapeM>;

  using PermuteLayoutM = Layout<Shape < ShapeM0,  ShapeM1,  ShapeM2>,
                                Stride<StrideM0, StrideM1, StrideM2>>;

  using ShapeN0 = Int<MmaOpShapeN>;
  using ShapeN1 = Int<size<1>(AtomLayout{})>;
  using ShapeN2 = Int<ValueLayoutN>;

  using StrideN0 = _1;
  using StrideN1 = Int<MmaOpShapeN * ValueLayoutN>;
  using StrideN2 = Int<MmaOpShapeN>;

  using PermuteLayoutN = Layout<Shape < ShapeN0,  ShapeN1,  ShapeN2>,
                                Stride<StrideN0, StrideN1, StrideN2>>;

  using PermuteLayout = mute::Tile<PermuteLayoutM, PermuteLayoutN, PermuteLayoutK>;

  return TiledMMA<MMA_Atom<MmaOp>, AtomLayout, PermuteLayout>{};
}

} // namespace detail


template <
  class ArchTag,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
  ArchTag,
  arch::OpClassSimt,
  ElementA,
  GmemLayoutA,
  AlignmentA,
  ElementB,
  GmemLayoutB,
  AlignmentB,
  ElementAccumulator,
  TileShape_MNK,
  ClusterShape_MNK,
  StageCountType,
  KernelScheduleType
> {
  static_assert(is_static<TileShape_MNK>::value, "TileShape must be static");
  static_assert(is_static<ClusterShape_MNK>::value, "ClusterShape must be static");

  static constexpr int BlockM = shape<0>(TileShape_MNK{});
  static constexpr int BlockN = shape<1>(TileShape_MNK{});
  static constexpr int BlockK = shape<2>(TileShape_MNK{});

  using TiledMma = decltype(detail::mp22_make_simt_tiled_mma<TileShape_MNK, ElementA, ElementB, ElementAccumulator>());

  static constexpr int ThreadCount = size(TiledMma{});
  using GmemTiledCopyA = decltype(detail::make_simt_tiled_copy<
                                    ThreadCount, ElementA, AlignmentA, TagToStrideA_t<GmemLayoutA>,
                                    BlockM, BlockK,
                                    UniversalCopy<uint_bit_t<AlignmentA*sizeof_bits_v<ElementA>>>>());
  using GmemTiledCopyB = decltype(detail::make_simt_tiled_copy<
                                    ThreadCount, ElementB, AlignmentB, TagToStrideB_t<GmemLayoutB>,
                                    BlockN, BlockK,
                                    UniversalCopy<uint_bit_t<AlignmentB*sizeof_bits_v<ElementB>>>>());

  using SmemLayoutAtomA = Layout<Shape <Int<BlockM>, Int<BlockK>>,
                                 Stride<         _1, Int<BlockM>>>;
  using SmemLayoutAtomB = Layout<Shape <Int<BlockN>, Int<BlockK>>,
                                 Stride<         _1, Int<BlockM>>>;

  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  using DispatchPolicy = MainloopMp22TwoStage;

  using CollectiveOp = collective::CollectiveMma<
    DispatchPolicy, TileShape_MNK,
    ElementA, TagToStrideA_t<GmemLayoutA>,
    ElementB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, mute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, mute::identity   // B
  >;
};

template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
  arch::Mp22,
  arch::OpClassTensorOp,
  ElementA,
  GmemLayoutA,
  AlignmentA,
  ElementB,
  GmemLayoutB,
  AlignmentB,
  ElementAccumulator,
  TileShape_MNK,
  ClusterShape_MNK,
  StageCountType,
  KernelScheduleType
> {
  static_assert(is_static<TileShape_MNK>::value, "TileShape must be static");
  static_assert(is_static<ClusterShape_MNK>::value, "ClusterShape must be static");

  static constexpr int BlockM = shape<0>(TileShape_MNK{});
  static constexpr int BlockN = shape<1>(TileShape_MNK{});
  static constexpr int BlockK = shape<2>(TileShape_MNK{});

  using StrideA = TagToStrideA_t<GmemLayoutA>;
  using StrideB = TagToStrideB_t<GmemLayoutB>;

  // For fp32 types, map to tf32 MMA value type
  using MmaElementA = mute::conditional_t<mute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using MmaElementB = mute::conditional_t<mute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using MmaOp = decltype(detail::mp22_mma_operation_select<MmaElementA, StrideA, MmaElementB, StrideB>());

  using TiledMma = decltype(detail::mp22_make_tensorop_tiled_mma<TileShape_MNK, MmaOp>());

  static constexpr int ThreadCount = size(TiledMma{});

  // A
  using SmemLayoutAtomA = decltype(detail::make_mp22_smem_atom_layout<MmaElementA, StrideA>());
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, MmaElementA>;
  using GmemTiledCopyA = decltype(detail::make_simt_tiled_copy<
                                    ThreadCount, MmaElementA, AlignmentA, StrideA,
                                    BlockM, BlockK,
                                    UniversalCopy<uint_bit_t<AlignmentA*sizeof_bits_v<MmaElementA>>>>());
  // B
  using SmemLayoutAtomB = decltype(detail::make_mp22_smem_atom_layout<MmaElementB, StrideB>());
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, MmaElementB>;
  using GmemTiledCopyB = decltype(detail::make_simt_tiled_copy<
                                    ThreadCount, MmaElementB, AlignmentB, StrideB,
                                    BlockN, BlockK,
                                    UniversalCopy<uint_bit_t<AlignmentB*sizeof_bits_v<MmaElementB>>>>());
  using DispatchPolicy = MainloopMp22TwoStage;

  using CollectiveOp = collective::CollectiveMma<
    DispatchPolicy, TileShape_MNK,
    MmaElementA, StrideA,
    MmaElementB, StrideB,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, mute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, mute::identity   // B
  >;
};

} // namespace mutlass::gemm::collective
