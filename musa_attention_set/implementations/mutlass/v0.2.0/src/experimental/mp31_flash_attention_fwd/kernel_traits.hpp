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

#include <mute/tensor.hpp>
#include <mute/atom/mma_atom.hpp>
#include <mute/atom/copy_atom.hpp>

#include <mutlass/mutlass.h>
#include <mutlass/gemm/gemm.h>
#include <mutlass/gemm/collective/collective_builder.hpp>

using namespace mute;

template <class TileN, class PermuteTileShape>
constexpr auto
get_permute_tile_for_operand_B() {
  using SqmmaAtomN = _8;
  static_assert(TileN{} % PermuteTileShape{} == 0);
  static_assert(PermuteTileShape{} % SqmmaAtomN{} == 0);
  using RepeatInsidePermuteTile = decltype(PermuteTileShape{} / SqmmaAtomN{});
  using RepeatAcrossPermuteTile = decltype(TileN{} / PermuteTileShape{});

  using PermuteTileN = Layout<Shape <             SqmmaAtomN, RepeatInsidePermuteTile, RepeatAcrossPermuteTile>,
                              Stride<RepeatInsidePermuteTile,                      _1,        PermuteTileShape>>;
  return PermuteTileN{};
}

template <
  class Gemm1Element,
  class Gemm2Element,
  class ElementAccum,
  class SmemLayoutQ,
  class SmemLayoutK,
  class SmemLayoutV,
  class SmemLayoutVt,
  class SmemLayoutS,
  class SmemLayoutLse,
  class SmemLayoutAlpha
>
struct SharedStorage {
  static_assert(sizeof(Gemm1Element) == sizeof(Gemm2Element));
  mute::array_aligned<Gemm1Element, mute::cosize_v<SmemLayoutQ>, 256> smem_q;
  union {
    mute::array_aligned<Gemm1Element, mute::cosize_v<SmemLayoutK>, 256> smem_k;
    mute::array_aligned<ElementAccum, mute::cosize_v<SmemLayoutLse>, 256> smem_lse;
  };

  union {
    mute::array_aligned<Gemm2Element, mute::cosize_v<SmemLayoutV>, 256> smem_v;
    mute::array_aligned<Gemm2Element, mute::cosize_v<SmemLayoutVt>,256> smem_vt;
  };
  mute::array_aligned<Gemm2Element, mute::cosize_v<SmemLayoutS>, 256> smem_s;
  mute::array_aligned<ElementAccum, mute::cosize_v<SmemLayoutAlpha>, 256> smem_alpha;
};

template <
  int BlockM_,
  int BlockN_,
  int HeadDimQK_,
  int HeadDimV_,
  class Element_ = mutlass::half_t,
  int KStages_ = 2,
  int VStages_ = 1
>
struct FlashAttentionFwdKernelTraits {
  using Element = Element_;
  using ElementAccum = float;
  using ElementOutput = Element;

  static constexpr int BlockM  = BlockM_;
  static constexpr int BlockN  = BlockN_;
  static constexpr int HeadDimQK = HeadDimQK_;
  static constexpr int HeadDimV = HeadDimV_;
  static constexpr int KStages = KStages_;
  static constexpr int VStages = VStages_;

  using PipeLineQ = mutlass::arch::AsyncTransactionBarrier;
  using PipeLineAlpha = mutlass::arch::AsyncTransactionBarrier;
  using PipeLineK = mutlass::Mp31PipelineAsync<KStages>;
  using PipeLineV = mutlass::Mp31PipelineTmeAsync<VStages>;
  using PipeLineGemm = mutlass::Mp31PipelineAsync<1>;

  #pragma pack(push, 1)
  struct BarrierStorage{
    uint8_t pipeline_Q[1];
    uint8_t pipeline_Alpha[1];
    uint8_t pipeline_K[PipeLineK::NumBarriers];
    uint8_t pipeline_V[PipeLineV::NumBarriers];
    uint8_t pipeline_Gemm[PipeLineGemm::NumBarriers];
  };
  #pragma pack(pop)

  using TileShapeQK = Shape<Int<BlockM>, Int<BlockN>, Int<HeadDimQK>>;
  using TileShapePV = Shape<Int<BlockM>, Int<HeadDimV>, Int<BlockN>>;

  // TiledMma
  static constexpr int MaxInstructionM = BlockM / 2;
  using AtomLayoutMNK = Layout<Shape<Int<2>, _1, _1>>;

  using TiledMmaQK = decltype(mute::make_tiled_mma(
      mute::MP31::SQMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK, TCE::Major::K, TCE::Major::K, Int<MaxInstructionM>>(),
      AtomLayoutMNK{}));
  using MmaQKAtomShapeMNK = typename TiledMmaQK::AtomShape_MNK;

  using TiledMmaPV = decltype(mute::make_tiled_mma(
      mute::MP31::SQMMA::ss_op_selector<Element, Element, ElementAccum, TileShapePV,
                                        TCE::Major::K, TCE::Major::MN, Int<MaxInstructionM>>(),
      AtomLayoutMNK{}));
  using MmaPVAtomShapeMNK = typename TiledMmaPV::AtomShape_MNK;

  static_assert(size<2>(MmaQKAtomShapeMNK{}) == 64);
  static_assert(size<2>(MmaPVAtomShapeMNK{}) == 64);
  static constexpr int PNumThreads = size(TiledMmaQK{});
  static constexpr int CNumThreads = size(TiledMmaQK{});
  static constexpr int NumThreads = PNumThreads + CNumThreads;

  // Q
  using SmemLayoutAtomQ = decltype(mute::MP31::SQMMA::Layout_SL256_SS256_SG16_Atom<
                            get<0>(typename TiledMmaQK::AtomShape_MNK{}), get<2>(typename TiledMmaQK::AtomShape_MNK{}), Element, TCE::Major::K>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShapeQK{})));

  // K
  using SmemLayoutAtomK = decltype(mute::MP31::SQMMA::Layout_SL256_SS256_SG16_Atom<
                            get<1>(typename TiledMmaQK::AtomShape_MNK{}), get<2>(typename TiledMmaQK::AtomShape_MNK{}), Element, TCE::Major::K>());
  using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
                                             make_shape(get<1>(TileShapeQK{}), get<2>(TileShapeQK{}), Int<KStages>{})));

  static constexpr int AtomShape_M = size<0>(typename TiledMmaQK::AtomShape_MNK{});
  static constexpr int InstAtomShape_N = size<0, 0>(typename TiledMmaQK::LayoutC_TV{});

  static constexpr int EaDivE = sizeof_bits_v<ElementAccum> / sizeof_bits_v<Element>;

  using SmemLayoutLse = Layout<
    Shape<Shape<Int<AtomShape_M>, Int<BlockM/AtomShape_M>>, Int<InstAtomShape_N>, Int<1>>, 
    Stride<Stride<Int<1>, Int<AtomShape_M>>, Int<0>, Int<0>>>;
  using SmemLayoutAlpha = Layout<
    Shape<Shape<Int<AtomShape_M>, Int<BlockM/AtomShape_M>>, Int<InstAtomShape_N>, Int<KStages>>, 
    Stride<Stride<Int<1>, Int<AtomShape_M>>, Int<0>, Int<BlockM>>>;

  // V
  using SmemLayoutAtomV = mute::conditional_t<mute::is_same_v<Element, mutlass::float_e4m3_t>,
                                              decltype(mute::MP31::SQMMA::Layout_SL256_SS256_SG16_Atom<
                                                       get<2>(typename TiledMmaPV::AtomShape_MNK{}), get<1>(typename TiledMmaPV::AtomShape_MNK{}), Element, TCE::Major::K>()),
                                              decltype(mute::MP31::SQMMA::Layout_SL256_SS256_SG32_Atom<
                                                       get<2>(typename TiledMmaPV::AtomShape_MNK{}), get<1>(typename TiledMmaPV::AtomShape_MNK{}), Element, TCE::Major::K>())>;
  using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{},
                                             make_shape(get<2>(TileShapePV{}), get<1>(TileShapePV{}), Int<VStages>{})));

  using SmemLayoutVt = decltype(composition(SmemLayoutV{},
                                  make_ordered_layout(
                                    make_shape(get<1>(TileShapePV{}), get<2>(TileShapePV{}), Int<VStages>{}),
                                    Step<_2, _1, _3>{})));

  // S
  using SmemLayoutAtomS = decltype(mute::MP31::SQMMA::Layout_SL256_SS256_SG16_Atom<
                            get<0>(typename TiledMmaPV::AtomShape_MNK{}), get<2>(typename TiledMmaPV::AtomShape_MNK{}), Element, TCE::Major::K>());
  using SmemLayoutS = decltype(tile_to_shape(SmemLayoutAtomS{}, select<0, 1>(TileShapeQK{})));

  // TiledCopy
  using GmemTiledCopyQ = MP31_TME_LOAD;

  static constexpr int FragmentSize = mute::min(size<1>(SmemLayoutAtomS{}), 128) / sizeof_bits_v<Element>;
  using GmemTiledCopyOpK = MP31_ROBUST_LDGSTS<mute::uint_bit_t<sizeof_bits_v<Element> * FragmentSize>>;
  using GmemTiledCopyK =
      decltype(mutlass::gemm::collective::detail::make_simt_tiled_copy<
                  PNumThreads, Element, FragmentSize, Stride<int, _1> /* HeadDim-major */,
                  shape<0>(SmemLayoutAtomK{}), shape<1>(SmemLayoutAtomK{}), GmemTiledCopyOpK>());
  using KeyPermuteTile =
      decltype(make_tile(get_permute_tile_for_operand_B<
                            decltype(get<1>(TileShapeQK{})), decltype(get<2>(typename TiledMmaQK::AtomShape_MNK{}))>(),
                         get<2>(TileShapeQK{})));

  using TiledMmaQKPermute = decltype(mute::make_tiled_mma(
      mute::MP31::SQMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK, TCE::Major::K, TCE::Major::K, Int<MaxInstructionM>>(),
      AtomLayoutMNK{},
      make_tile(Underscore{},
                get_permute_tile_for_operand_B<
                    decltype(get<1>(TileShapeQK{})), decltype(get<2>(typename TiledMmaPV::AtomShape_MNK{}))>(),
                Underscore{})));

  using GmemTiledCopyV = MP31_TME_LOAD;

  using SharedStorage = SharedStorage<Element, Element, ElementAccum, SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutVt, SmemLayoutS, SmemLayoutLse, SmemLayoutAlpha>;

  static constexpr uint32_t TmeTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmeTransactionBytesV = static_cast<uint32_t>(size(take<0,2>(SmemLayoutV{})) * sizeof_bits_v<Element> / 8);

};
