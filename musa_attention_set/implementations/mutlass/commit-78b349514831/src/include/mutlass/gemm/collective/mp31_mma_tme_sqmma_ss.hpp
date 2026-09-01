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

#include "mutlass/mutlass.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/pipeline/pipeline.hpp"

#include "mute/atom/mma_atom.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::collective {

using namespace mute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopMp31TmeSqmma<Stages>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopMp31TmeSqmma<Stages>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = mutlass::Mp31PipelineTmeAsync<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename mutlass::PipelineState<DispatchPolicy::Stages>;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(DispatchPolicy::Stages >= 2, "Stages should set to value 2 or more.");

  static_assert(mute::is_base_of<mute::MP31::SQMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                mute::is_base_of<mute::MP31::SQMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");

  static_assert(mute::is_same_v<GmemTiledCopyA, MP31_TME_LOAD>, "GmemTiledCopy - invalid MP31 TME copy atom specified.");
  static_assert(mute::is_same_v<GmemTiledCopyB, MP31_TME_LOAD>, "GmemTiledCopy - invalid MP31 TME copy atom specified.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  static constexpr int NumWarps = size(TiledMma{}) / NumThreadsPerWarp;

  static constexpr int NumBarriers = MainloopPipeline::NumBarriers;

  static constexpr int SmemAlignmentBytes = 256;

  struct SharedStorage
  {
    mute::array_aligned<typename TiledMma::ValTypeA, mute::cosize_v<SmemLayoutA>, SmemAlignmentBytes> smem_A;
    mute::array_aligned<typename TiledMma::ValTypeB, mute::cosize_v<SmemLayoutB>, SmemAlignmentBytes> smem_B;
  };

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  // Device side kernel params
  struct Params {
    using TME_A = decltype(make_tme_copy(
        GmemTiledCopyA{},
        make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        take<0,2>(SmemLayoutA{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));

    using TME_B = decltype(make_tme_copy(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        take<0,2>(SmemLayoutB{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));
    TME_A tme_load_a;
    TME_B tme_load_b;
  };


  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor tensor_a = make_tensor(args.ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(N,K,L), args.dB));

    typename Params::TME_A tme_load_a = make_tme_copy(
        GmemTiledCopyA{},
        tensor_a,
        take<0,2>(SmemLayoutA{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})));

    typename Params::TME_B tme_load_b = make_tme_copy(
        GmemTiledCopyB{},
        tensor_b,
        take<0,2>(SmemLayoutB{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})));
    return {
      tme_load_a,
      tme_load_b
    };
  }

  template <class ProblemShape>
  MUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tme_alignment_bits = 32;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int min_tme_aligned_elements_A = tme_alignment_bits / mutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && mutlass::detail::check_alignment<min_tme_aligned_elements_A>(mute::make_shape(M,K,L), StrideA{});
    constexpr int min_tme_aligned_elements_B = tme_alignment_bits / mutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && mutlass::detail::check_alignment<min_tme_aligned_elements_B>(mute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      MUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TME.\n");
    }
    return implementable;
  }

  /// Issue Tme Descriptor Prefetch
  MUTLASS_DEVICE
  static void prefetch_tme_descriptors(Params const& mainloop_params)
  {
    mute::prefetch_tme_descriptor(mainloop_params.tme_load_a.get_tme_descriptor());
    mute::prefetch_tme_descriptor(mainloop_params.tme_load_b.get_tme_descriptor());
  }

  template <class ProblemShape_MNKL>
  MUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // TME requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TME
    Tensor mA_mkl = mainloop_params.tme_load_a.get_tme_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tme_load_b.get_tme_tensor(make_shape(N,K,L));                            // (n,k,l)

    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    auto blk_coord = make_coord(_,_,_);                                                   // (m,n,k) -- defer the slice

    // Makke tiled views
    Tensor gA_mkl = local_tile(mA_mkl, blk_shape, blk_coord, Step<_1, X,_1>{});                  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, blk_shape, blk_coord, Step< X,_1,_1>{});                  // (BLK_N,BLK_K,n,k,l)

    return mute::make_tuple(gA_mkl, gB_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class TensorA, class TensorB,
    class BlockCoord,
    class FrgTensorC,
    class KTileIterator
  >
  MUTLASS_DEVICE void
  operator() (
      mute::tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      FrgTensorC& accum,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      char* shared_memory,
      Params const& mainloop_params)
  {
    using namespace mute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2.");
    static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2.");
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(mute::is_void_v<SmemCopyAtomA>,
      "MP31 SQMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(mute::is_void_v<SmemCopyAtomB>,
      "MP31 SQMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    auto cta_tme_a = mainloop_params.tme_load_a.get_slice(0);
    auto cta_tme_b = mainloop_params.tme_load_b.get_slice(0);

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);
    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);


    // Applies the mapping frome tme
    Tensor tAgA = cta_tme_a.partition_S(gA);
    Tensor tAsA = cta_tme_a.partition_D(sA);

    Tensor tBgB = cta_tme_b.partition_S(gB);
    Tensor tBsB = cta_tme_b.partition_D(sB);

    // Set the bytes transferred in this TME transaction (may involve multiple issues)
    constexpr uint32_t TmeTransactionBytes = static_cast<uint32_t>(
        (size<0>(sA) * size<1>(sA) * sizeof(ElementA)) +
        (size<0>(sB) * size<1>(sB) * sizeof(ElementB)));

    // Obtain warp index
    int warp_idx = canonical_warp_idx();

    PipelineParams params;
    params.transaction_bytes = TmeTransactionBytes;
    params.num_consumers = NumWarps;

    // Init pipeline
    MainloopPipeline pipeline(params);

    // State variables used for iterating the circular buffer
    // smem_pipe_read / release is used by the consumer of SMEM data - i.e MMA
    // smem_pipe_write is used by the producer of SMEM data - i.e. TME
    PipelineState smem_pipe_read;
    PipelineState smem_pipe_release;
    PipelineState smem_pipe_write = mutlass::make_producer_start_state<MainloopPipeline>();

    // We need this to gurantee that the Pipeline init is visible
    // to all warps in the CTA
    __syncthreads();


    //
    // Prologue TME
    //

    // Keep a copy to know when to stop issuing loads
    int k_tile_count_tme = k_tile_count;

    // Start async tme loads for all pipes but the last
    MUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      if (warp_idx == 0 && k_tile_count_tme > 0) {
        pipeline.producer_acquire(smem_pipe_write);
        uint32_t bar_id = pipeline.producer_get_barrier_id(smem_pipe_write);
        copy(mainloop_params.tme_load_a.with(bar_id), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe));
        copy(mainloop_params.tme_load_b.with(bar_id), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe));

      }
      // advance producer pipeline
      ++smem_pipe_write;
      ++k_tile_iter;
      --k_tile_count_tme;
    }

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thread_mma.partition_A(sA);                                  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                  // (MMA,MMA_N,MMA_K,PIPE)
    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                            // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                            // (MMA,MMA_N,MMA_K,PIPE)

    MUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                     // M
    MUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                     // N
    MUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                      // K
    MUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                      // PIPE
    MUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tAsA));                      // PIPE
    MUTE_STATIC_ASSERT_V(size<3>(tCsB) == size<3>(tBsB));                      // PIPE
    MUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    MUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE
    // Currently we prefer one instruction per squad atom
    MUTE_STATIC_ASSERT_V(size<1>(accum) == Int<1>{});
    MUTE_STATIC_ASSERT_V(size<2>(accum) == Int<1>{});
    MUTE_STATIC_ASSERT_V(size<2>(tCsA)  == Int<1>{});

    // Prologue MMA
    {
      pipeline.consumer_wait(smem_pipe_read);

      MUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        mute::gemm(tiled_mma, tCrA(_,_,k_block,smem_pipe_read.index()), tCrB(_,_,k_block,smem_pipe_read.index()), accum);
      }

      // Last stage tme
      if (warp_idx == 0 && k_tile_count_tme > 0) {
        pipeline.producer_acquire(smem_pipe_write);
        uint32_t bar_id = pipeline.producer_get_barrier_id(smem_pipe_write);
        copy(mainloop_params.tme_load_a.with(bar_id), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,DispatchPolicy::Stages-1));
        copy(mainloop_params.tme_load_b.with(bar_id), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,DispatchPolicy::Stages-1));
      }
      // advance producer pipeline
      ++smem_pipe_write;
      ++k_tile_iter;
      --k_tile_count_tme;

      warpsquad_wait();
      pipeline.consumer_release(smem_pipe_release);

      // advance consumer pipeline
      ++smem_pipe_read;
      ++smem_pipe_release;
      --k_tile_count;
    }

    if (k_tile_count == 0) return;

    // PIPELINED MAIN LOOP

    MUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count)
    {
      pipeline.consumer_wait(smem_pipe_read);

      MUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        mute::gemm(tiled_mma, tCrA(_,_,k_block,smem_pipe_read.index()), tCrB(_,_,k_block,smem_pipe_read.index()), accum);
      }

      if (warp_idx == 0 && k_tile_count_tme > 0) {
        pipeline.producer_acquire(smem_pipe_write);
        uint32_t bar_id = pipeline.producer_get_barrier_id(smem_pipe_write);
        copy(mainloop_params.tme_load_a.with(bar_id), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write.index()));
        copy(mainloop_params.tme_load_b.with(bar_id), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write.index()));
      }
      // advance producer pipeline
      ++smem_pipe_write;
      ++k_tile_iter;
      --k_tile_count_tme;

      warpsquad_wait();
      pipeline.consumer_release(smem_pipe_release);

      // advance consumer pipeline
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    {
      pipeline.consumer_wait(smem_pipe_read);

      MUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        mute::gemm(tiled_mma, tCrA(_,_,k_block,smem_pipe_read.index()), tCrB(_,_,k_block,smem_pipe_read.index()), accum);
      }

      warpsquad_wait();
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
