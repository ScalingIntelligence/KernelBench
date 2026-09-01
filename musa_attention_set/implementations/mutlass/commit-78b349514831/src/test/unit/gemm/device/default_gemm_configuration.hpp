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

#include "mute/atom/mma_atom.hpp"
#include "mute/atom/copy_atom.hpp"

#include "mutlass/mutlass.h"
#include "mutlass/gemm/gemm.h"
#include "mutlass/arch/arch.h"
#include "mutlass/arch/mma.h"
#include "mutlass/layout/layout.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/gemm/collective/collective_mma.hpp"
#include "mutlass/epilogue/collective/collective_epilogue.hpp"

#include "mutlass/epilogue/collective/default_epilogue.hpp"
#include "mutlass/epilogue/collective/mp22_epilogue_vectorized.hpp"
#include "mutlass/epilogue/thread/linear_combination.h"
#include "mutlass/epilogue/fusion/operations.hpp"

namespace mutlass {
namespace gemm {
namespace device {
using namespace mute;

// This type is only intended to demonstrate porting 2.x kernels to 3.0
template<
  class OperatorClass, class ArchTag,
  class TiledMma,
  class TileShape,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator,
  int ThreadCount,
  int AlignmentA, int AlignmentB>
struct DefaultGemmConfigurationToMutlass3Types {
  static_assert(sizeof(ElementA) == 0, "No valid DefaultGemmConfigurationToMutlass3Types configuration exists.");
};


template<int ThreadCount, class Element, int Alignment, class StrideType, int TileSizeMN, int TileSizeK, class CopyOperation>
constexpr auto
make_gmem_tiled_copy() {
  // Maximize the number of threads along the gmem major mode to promote coalesced reads
  // While making sure our thread layout tiles the threadblock tile evenly

  if constexpr (mutlass::gemm::detail::is_k_major<StrideType>()) {
    // K major thread layout for K major gmem
    constexpr int threads_major = TileSizeK   / Alignment;
    constexpr int threads_minor = ThreadCount / threads_major;
    static_assert(threads_major > 0);
    static_assert(ThreadCount % threads_major == 0);
    static_assert(threads_minor == 0 || (TileSizeMN % threads_minor == 0));
    return make_tiled_copy(
      Copy_Atom<CopyOperation, Element>{},
      Layout<Shape <Int<threads_minor>, Int<threads_major>>,
             Stride<Int<threads_major>,                _1>>{},
      Layout<Shape<_1, Int<Alignment>>>{});
  }
  else if constexpr (mutlass::gemm::detail::is_mn_major<StrideType>()) {
    // MN major thread layout for MN major gmem
    constexpr int threads_major = TileSizeMN  / Alignment;
    constexpr int threads_minor = ThreadCount / threads_major;
    static_assert(threads_major > 0);
    static_assert(ThreadCount % threads_major == 0);
    static_assert(threads_minor == 0 || (TileSizeK % threads_minor == 0));
    return make_tiled_copy(
      Copy_Atom<CopyOperation, Element>{},
      Layout<Shape <Int<threads_major>, Int<threads_minor>>,
             Stride<                _1, Int<threads_major>>>{},
      Layout<Shape<Int<Alignment>, _1>>{});
  }
  else {
    static_assert(mute::is_void_v<Element>, "Unsupported gmem layout for automatic gmem tiled copy builder.");
  }
}


template<int ThreadCount, class Element, int Alignment, class StrideType, int TileSizeM, int TileSizeN, class CopyOperation>
constexpr auto
make_smem_tiled_copy() {
  return make_gmem_tiled_copy<ThreadCount, Element, Alignment, StrideType, TileSizeM, TileSizeN, CopyOperation>();
}


///////////////////////////////////////////////////////////////////////////////
//////////////////////////// SIMT TWO STAGE ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, typename Layout, int ThreadCount, int ShapeM, int ShapeK, int Alignment = 1>
struct DefaultGemm_Simt_OperandA;

template <typename Element, typename Layout, int ThreadCount, int ShapeN, int ShapeK, int Alignment = 1>
struct DefaultGemm_Simt_OperandB;
///////////////////////////////////////////////////////////////////////////////

template <typename Element, int ThreadCount, int ShapeM, int ShapeK, int Alignment>
struct DefaultGemm_Simt_OperandA<Element, layout::ColumnMajor, ThreadCount, ShapeM, ShapeK, Alignment>
{
  using SmemLayoutAtom = Layout<Shape <Int<ShapeM>, Int<ShapeK>>,
                                Stride<         _1, Int<ShapeM>>>;

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(make_gmem_tiled_copy<
    ThreadCount, Element, Alignment, TagToStrideA_t<layout::ColumnMajor>,
    ShapeM, ShapeK, 
    UniversalCopy<uint_bit_t<Alignment*sizeof_bits_v<Element>>>>());
};

template <typename Element, int ThreadCount, int ShapeM, int ShapeK, int Alignment>
struct DefaultGemm_Simt_OperandA<Element, layout::RowMajor, ThreadCount, ShapeM, ShapeK, Alignment>
{
  using SmemLayoutAtom = Layout<Shape <Int<ShapeM>, Int<ShapeK>>,
                                Stride<         _1, Int<ShapeM>>>;

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(make_gmem_tiled_copy<
    ThreadCount, Element, Alignment, TagToStrideA_t<layout::RowMajor>,
    ShapeM, ShapeK, 
    UniversalCopy<uint_bit_t<Alignment*sizeof_bits_v<Element>>>>());
};

template <typename Element, int ThreadCount, int ShapeN, int ShapeK, int Alignment>
struct DefaultGemm_Simt_OperandB<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK, Alignment>
     : DefaultGemm_Simt_OperandA<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK, Alignment> {};

template <typename Element, int ThreadCount, int ShapeN, int ShapeK, int Alignment>
struct DefaultGemm_Simt_OperandB<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK, Alignment>
     : DefaultGemm_Simt_OperandA<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK, Alignment> {};


// tensorop

template <typename Element, typename Layout, int ThreadCount, int ShapeM, int ShapeK, int Alignment>
struct Gemm_TensorOpMp22_OperandA;

template <typename Element, typename Layout, int ThreadCount, int ShapeM, int ShapeK, int Alignment>
struct Gemm_TensorOpMp22_OperandB;

// epilogue

template <typename Element, class TiledMma, class TileShape, typename Layout, int AlignmentBytes = 16>
struct Gemm_Epilogue;

template <typename Element, int ThreadCount, int ShapeM, int ShapeK, int Alignment>
struct Gemm_TensorOpMp22_OperandA<Element, layout::ColumnMajor, ThreadCount, ShapeM, ShapeK, Alignment>
{
  using SmemLayoutAtom = Layout<Shape <Shape <_16,   _2>, _16>,
                                Stride<Stride< _1, _256>, _16>>;

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(make_gmem_tiled_copy<
    ThreadCount, Element, Alignment, TagToStrideA_t<layout::ColumnMajor>,
    ShapeM, ShapeK, 
    UniversalCopy<uint_bit_t<Alignment*sizeof_bits_v<Element>>>>());
};

template <typename Element, int ThreadCount, int ShapeM, int ShapeK, int Alignment>
struct Gemm_TensorOpMp22_OperandA<Element, layout::RowMajor, ThreadCount, ShapeM, ShapeK, Alignment>
{
  using SmemLayoutAtom = Layout<Shape <Shape <_16,   _2>, Shape<_8,   _2>>,
                                Stride<Stride< _8, _256>, Shape<_1, _128>>>;

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(make_gmem_tiled_copy<
    ThreadCount, Element, Alignment, TagToStrideA_t<layout::RowMajor>,
    ShapeM, ShapeK, 
    UniversalCopy<uint_bit_t<Alignment*sizeof_bits_v<Element>>>>());
};



template <typename Element, int ThreadCount, int ShapeN, int ShapeK, int Alignment>
struct Gemm_TensorOpMp22_OperandB<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK, Alignment>
     : Gemm_TensorOpMp22_OperandA<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK, Alignment> {};

template <typename Element, int ThreadCount, int ShapeN, int ShapeK, int Alignment>
struct Gemm_TensorOpMp22_OperandB<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK, Alignment>
     : Gemm_TensorOpMp22_OperandA<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK, Alignment> {};


template <typename Element, class TiledMma, class TileShape, int AlignmentBytes>
struct Gemm_Epilogue<Element, TiledMma, TileShape, layout::RowMajor, AlignmentBytes>
{
  static constexpr int BlockM = size<0>(TileShape{});
  static constexpr int BlockN = size<1>(TileShape{});
  
  static constexpr int TiledShapeM = mute::tile_size<0>(TiledMma{});
  
  static constexpr int ShapeM = TiledShapeM;
  static constexpr int ShapeN = BlockN < 256 ? BlockN: 128;

  static constexpr int ThreadCount = size(TiledMma{});

  static constexpr int Alignment = ShapeM * ShapeN / ThreadCount * sizeof(Element) <= AlignmentBytes ? 
                                   ShapeM * ShapeN / ThreadCount : AlignmentBytes / sizeof(Element);

  using CopyOperation = UniversalCopy<uint_bit_t<Alignment*sizeof_bits_v<Element>>>;

  using SmemLayout = Layout<Shape <Int<ShapeM>, Int<ShapeN>>, 
                            Stride<Int<ShapeN>,         _1>>;

  using CopyAtomR2G = Copy_Atom<CopyOperation, Element>;

  using TiledCopyS2R = decltype(make_smem_tiled_copy<
    ThreadCount, Element, Alignment, TagToStrideC_t<layout::RowMajor>,
    ShapeM, ShapeN, 
    DefaultCopy>());
};

template <typename Element, class TiledMma, class TileShape, int AlignmentBytes>
struct Gemm_Epilogue<Element, TiledMma, TileShape, layout::ColumnMajor, AlignmentBytes>
{
  static constexpr int BlockM = size<0>(TileShape{});
  static constexpr int BlockN = size<1>(TileShape{});
  
  static constexpr int TiledShapeN = mute::tile_size<1>(TiledMma{});
  
  static constexpr int ShapeM = BlockM < 256 ? BlockM: 128;
  static constexpr int ShapeN = TiledShapeN;

  static constexpr int ThreadCount = size(TiledMma{});

  static constexpr int Alignment = ShapeM * ShapeN / ThreadCount * sizeof(Element) <= AlignmentBytes ? 
                                   ShapeM * ShapeN / ThreadCount : AlignmentBytes / sizeof(Element);

  using CopyOperation = UniversalCopy<uint_bit_t<Alignment*sizeof_bits_v<Element>>>;

  using SmemLayout = Layout<Shape <Int<ShapeM>, Int<ShapeN>>, 
                            Stride<         _1, Int<ShapeM>>>;

  using CopyAtomR2G = Copy_Atom<CopyOperation, Element>;

  using TiledCopyS2R = decltype(make_smem_tiled_copy<
    ThreadCount, Element, Alignment, TagToStrideC_t<layout::ColumnMajor>,
    ShapeM, ShapeN, 
    DefaultCopy>());
};

} // end namespace detail

// SIMT Two Stage
template <
  class ArchTag,
  class TiledMma,
  class TileShape,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator,
  int ThreadCount,
  int AlignmentA, int AlignmentB>
struct DefaultGemmConfigurationToMutlass3Types<
    arch::OpClassSimt, ArchTag,
    TiledMma,
    TileShape,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    ThreadCount,
    AlignmentA, AlignmentB>
{
  static constexpr int BlockM = size<0>(TileShape{});
  static constexpr int BlockN = size<1>(TileShape{});
  static constexpr int BlockK = size<2>(TileShape{});
  using DispatchPolicy = MainloopMp22TwoStage;

  // A
  using DefaultOperandA = detail::DefaultGemm_Simt_OperandA<ElementA, LayoutA, ThreadCount, BlockM, BlockK, AlignmentA>;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
  using SmemCopyAtomA   = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA  = typename DefaultOperandA::GmemTiledCopy;

  // B
  using DefaultOperandB = detail::DefaultGemm_Simt_OperandB<ElementB, LayoutB, ThreadCount, BlockN, BlockK, AlignmentB>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
  using SmemCopyAtomB   = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB  = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<LayoutA>,
    ElementB, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, mute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, mute::identity   // B
  >;

  // default epilogue
  using DefaultCollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    mutlass::gemm::EpilogueDefault>;

  // Epilogue
  using Epilogue     = detail::Gemm_Epilogue<ElementAccumulator, TiledMma, TileShape, LayoutC>;
  using SmemLayout   = typename Epilogue::SmemLayout;
  using CopyAtomR2G  = typename Epilogue::CopyAtomR2G;
  using TiledCopyS2R = typename Epilogue::TiledCopyS2R;
  using CollectiveEpilogue = epilogue::collective::Epilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    SmemLayout,
    Copy_Atom<DefaultCopy, ElementAccumulator>,
    TiledCopyS2R,
    CopyAtomR2G
    >;
};



template <
  class ArchTag,
  class TiledMma,
  class TileShape,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator,
  int ThreadCount,
  int AlignmentA, int AlignmentB>
struct DefaultGemmConfigurationToMutlass3Types<
  arch::OpClassTensorOp, ArchTag,
  TiledMma,
  TileShape,
  ElementA, LayoutA,
  ElementB, LayoutB,
  ElementC, LayoutC,
  ElementAccumulator,
  ThreadCount,
  AlignmentA, AlignmentB>
{
  static constexpr int BlockM = size<0>(TileShape{});
  static constexpr int BlockN = size<1>(TileShape{});
  static constexpr int BlockK = size<2>(TileShape{});
  using DispatchPolicy = MainloopMp22TwoStage;

  // A
  using DefaultOperandA = detail::Gemm_TensorOpMp22_OperandA<ElementA, LayoutA, ThreadCount, BlockM, BlockK, AlignmentA>;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
  using SmemCopyAtomA   = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA  = typename DefaultOperandA::GmemTiledCopy;

  // B
  using DefaultOperandB = detail::Gemm_TensorOpMp22_OperandB<ElementB, LayoutB, ThreadCount, BlockN, BlockK, AlignmentB>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
  using SmemCopyAtomB   = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB  = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<LayoutA>,
    ElementB, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, mute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, mute::identity   // B
  >;

  // default epilogue
  using DefaultCollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    mutlass::gemm::EpilogueDefault>;

  // Epilogue
  using Epilogue     = detail::Gemm_Epilogue<ElementAccumulator, TiledMma, TileShape, LayoutC>;
  using SmemLayout   = typename Epilogue::SmemLayout;
  using CopyAtomR2G  = typename Epilogue::CopyAtomR2G;
  using TiledCopyS2R = typename Epilogue::TiledCopyS2R;
  using CollectiveEpilogue = epilogue::collective::Epilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    SmemLayout,
    Copy_Atom<DefaultCopy, ElementAccumulator>,
    TiledCopyS2R,
    CopyAtomR2G
    >;
};



///////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace mutlass
