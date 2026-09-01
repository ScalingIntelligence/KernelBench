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

#include "mutlass/mutlass.h"
#include "mute/tensor.hpp"
#include "mutlass/epilogue/collective/builders/mp31_builder_common.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmeStore_,
  class ThreadEpilogueOp_,
  class CtaTileMNK_,
  class EpilogueTile_,
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class CopyOpG2S,
  class CopyOpS2G,
  class SmemLayoutAtomC_,
  class SmemLayoutAtomD_,
  class CopyOpS2R_,
  class CopyOpR2S_
>
class CollectiveEpilogue<
    Mp31CollectiveEpilogue<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmeStore_>,
    ThreadEpilogueOp_,
    CtaTileMNK_,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    CopyOpG2S,
    CopyOpS2G,
    SmemLayoutAtomC_,
    SmemLayoutAtomD_,
    CopyOpS2R_,
    CopyOpR2S_
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = Mp31CollectiveEpilogue<StagesC_, StagesD_, FragmentSize_,ReuseSmemC_, DelayTmeStore_>;
  using CtaTileMNK = CtaTileMNK_;
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using EpilogueTile = EpilogueTile_;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using CopyOpR2S = CopyOpR2S_;
  using CopyOpS2R = CopyOpS2R_;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using GmemTiledCopyC = CopyOpG2S;
  using GmemTiledCopyD = CopyOpS2G;

  static_assert(!is_layout<EpilogueTile>::value && is_tuple<EpilogueTile>::value, "EpilogueTile must be a mute::Tile or mute::Shape");
  static_assert(mute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(mute::rank(EpilogueTile{}) == 2, "EpilogueTile must be rank-2: [EPI_TILE_M, EPI_TILE_N]");
  static_assert(size<0>(CtaTileMNK{}) % size<0>(shape(EpilogueTile{})) == 0, "EPI_TILE_M must divide CTA_M");
  static_assert(size<1>(CtaTileMNK{}) % size<1>(shape(EpilogueTile{})) == 0, "EPI_TILE_N must divide CTA_N");
  static_assert(mute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(mute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

private:  
  constexpr static bool is_source_supported = not mute::is_void_v<ElementC>;
  constexpr static bool is_destination_supported = not mute::is_void_v<ElementD>;
  using NonVoidElementD = ElementD;
  static_assert(not mute::is_void_v<NonVoidElementD>, "SmemElementD is void");
  using NonVoidElementC = mute::conditional_t<not is_source_supported,NonVoidElementD,ElementC>; // prevents void ref breakages

  using SmemElementC = typename mutlass::detail::get_unpacked_element_type<NonVoidElementC>::type;
  using SmemElementD = typename mutlass::detail::get_unpacked_element_type<NonVoidElementD>::type;
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static bool is_m_major_C = detail::is_m_major<StrideC>();
  constexpr static bool is_m_major_D = detail::is_m_major<StrideD>();
  constexpr static bool ReuseSmemC = ReuseSmemC_ and is_destination_supported;
  constexpr static bool DelayTmeStore = DelayTmeStore_;

  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}), Int<StagesC>{}),
      mute::conditional_t<is_m_major_C, Step<_2,_1,_3>, Step<_1,_2,_3>>{} ));
  using SmemLayoutD = decltype(tile_to_shape(
      SmemLayoutAtomD{},
      make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}), Int<ReuseSmemC ? StagesC : StagesD>{}),
      mute::conditional_t<is_m_major_D, Step<_2,_1,_3>, Step<_1,_2,_3>>{} ));
  constexpr static bool support_smem_reuse = is_source_supported && is_destination_supported && StagesD <= StagesC
                                            && cosize(take<0,2>(SmemLayoutC{})) == cosize(take<0,2>(SmemLayoutD{}));
  static_assert(not (ReuseSmemC && not support_smem_reuse), "Smem reuse requirements not met");
  constexpr static size_t SmemAlignmentD = mutlass::detail::alignment_for_swizzle(SmemLayoutD{});
  constexpr static size_t SmemAlignmentC = mutlass::detail::alignment_for_swizzle(SmemLayoutC{});
  constexpr static size_t MaxSmemAlignment = mute::max(SmemAlignmentC, SmemAlignmentD);

  struct CollectiveStorageWithC {
    mute::array_aligned<SmemElementC, mute::cosize_v<SmemLayoutC>, SmemAlignmentC> smem_C;
    mute::array_aligned<SmemElementD, mute::cosize_v<SmemLayoutD>, SmemAlignmentD> smem_D;
  };

  union CollectiveStorageWithoutC {
    mute::array<SmemElementC, 0> smem_C;
    mute::array_aligned<SmemElementD, mute::cosize_v<SmemLayoutD>, SmemAlignmentD> smem_D;
  };

  union CollectiveStorageReuseC {
    mute::array_aligned<SmemElementC, mute::cosize_v<SmemLayoutC>, MaxSmemAlignment> smem_C;
    mute::array_aligned<SmemElementD, mute::cosize_v<SmemLayoutD>, MaxSmemAlignment> smem_D;
  };

public:
  // TME pipeline for loading C
  using LoadPipeline       = mutlass::Mp31PipelineTmeAsync<DispatchPolicy::StagesC>;
  using LoadPipelineParams = typename LoadPipeline::Params;
  using LoadPipelineState  = mutlass::PipelineState<DispatchPolicy::StagesC>;
  using StorePipelineState = mute::conditional_t<ReuseSmemC, 
                             mutlass::PipelineState<DispatchPolicy::StagesC>,
                             mutlass::PipelineState<DispatchPolicy::StagesD>>;
  static constexpr int NumBarriers = is_source_supported ? LoadPipeline::NumBarriers: 0;
  struct SharedStorage
  {
    using CollectiveStorage = mute::conditional_t<not is_source_supported, CollectiveStorageWithoutC,
                              mute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
    CollectiveStorage collective;
  };


  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C;
    StrideC dC;
    ElementD const* ptr_D;
    StrideD dD;
  };

  // Device side epilogue params
  struct Params {
    using TME_C = decltype(make_tme_copy(
        CopyOpG2S{},
        make_tensor(make_gmem_ptr(static_cast<NonVoidElementC const*>(nullptr)), 
            repeat_like(StrideC{}, int32_t(0)), StrideC{}),
        take<0, 2>(SmemLayoutC{}),
        make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}))));
    using TME_D = decltype(make_tme_copy(
        CopyOpS2G{},
        make_tensor(make_gmem_ptr(static_cast<NonVoidElementD const*>(nullptr)), 
            repeat_like(StrideD{}, int32_t(0)), StrideD{}),
        take<0, 2>(SmemLayoutD{}),
        make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}))));
    typename ThreadEpilogueOp::Params thread{};
    TME_C tme_load_c;
    TME_D tme_store_d;
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

    typename Params::TME_C tme_load_c = {};
    if constexpr (is_source_supported) {
      Tensor tensor_c = make_tensor(args.ptr_C, make_layout(make_shape(M,N,L), args.dC));
      tme_load_c = make_tme_copy(
        MP31_TME_LOAD{},
        tensor_c,
        take<0, 2>(SmemLayoutC{}),
        make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{})));
    }
    
    typename Params::TME_D tme_store_d = {};
    if constexpr (is_destination_supported) {
      Tensor tensor_d = make_tensor(args.ptr_D, make_layout(make_shape(M,N,L), args.dD));
      tme_store_d = make_tme_copy(
          MP31_TME_STORE{},
          tensor_d,
          take<0, 2>(SmemLayoutD{}),
          make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{})));
    }

    typename ThreadEpilogueOp::Params thread{args.thread};

    return {thread, tme_load_c, tme_store_d};
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
    if constexpr (not mute::is_void_v<ElementC>) {
      constexpr int min_tme_aligned_elements_C = tme_alignment_bits / mutlass::sizeof_bits<ElementC>::value;
      implementable = implementable && mutlass::detail::check_alignment<min_tme_aligned_elements_C>(mute::make_shape(M,N), StrideC{});
    }

    if constexpr (is_destination_supported) {
      constexpr int min_tme_aligned_elements_D = tme_alignment_bits / mutlass::sizeof_bits<ElementD>::value;
      implementable = implementable && mutlass::detail::check_alignment<min_tme_aligned_elements_D>(mute::make_shape(M,N), StrideD{});
    }

    if (!implementable) {
      MUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TME.\n");
    }
    return implementable;
  }

  MUTLASS_HOST_DEVICE
  CollectiveEpilogue(Params const& params_)
      : params(params_), epilogue_op(params_.thread) {}

  MUTLASS_DEVICE
  bool
  is_source_needed() {
    return is_source_supported && epilogue_op.is_source_needed();
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledMma,
    class ResidueMNK
  >
  MUTLASS_DEVICE auto
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      mute::Tensor<AccEngine,AccLayout> accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf) {
    using namespace mute;
    using ElementAccumulator = typename AccEngine::value_type;

    static_assert(is_rmem<AccEngine>::value, "Accumulator must be RF resident.");
    static_assert(rank(AccLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA,MMA_M,MMA_N)");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<TileShapeMNK>::value, "TileShapeMNK must be static");
    static_assert(rank(TileShapeMNK{}) == 3, "TileShapeMNK must be rank 3");
    static_assert(rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

    Tensor mD_mn = params.tme_store_d.get_tme_tensor(make_shape(M,N,L));                               // (M,N,L)
    Tensor mD = coalesce(mD_mn, take<0,2>(CtaTileMNK{}));
    Tensor gD = local_tile(mD, take<0,2>(CtaTileMNK{}), make_coord(m_coord, n_coord, l_coord));     // (CTA_M,CTA_N)

    Tensor mC_mn = params.tme_load_c.get_tme_tensor(make_shape(M,N,L));                             //  (M,N,L)
    Tensor mC = coalesce(mC_mn, take<0,2>(CtaTileMNK{}));
    Tensor gC = local_tile(mC, take<0,2>(CtaTileMNK{}), make_coord(m_coord, n_coord, l_coord));

    Tensor gC_epi = flat_divide(gC, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor gD_epi = flat_divide(gD, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    // Apply epilogue subtiling
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sC_epi = make_tensor(make_smem_ptr(storage.collective.smem_C.begin()), SmemLayoutC{});             // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    Tensor sD_epi = make_tensor(make_smem_ptr(storage.collective.smem_D.begin()), SmemLayoutD{});             // (EPI_TILE_M,EPI_TILE_N,PIPE_D)
    
    constexpr int FragmentSize = DispatchPolicy::FragmentSize;
    using CopyV = Int<FragmentSize>;
    TiledCopy r2s_tiled_copy = detail::make_tiled_copy_C_sqmma<CopyOpR2S, CopyV, EpilogueTile>(tiled_mma);
    ThrCopy thread_r2s = r2s_tiled_copy.get_thread_slice(thread_idx);
    Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);                                   // ((R2S,R2S_V),MMA_M,MMA_N)
    Tensor tRS_sD   = thread_r2s.partition_D(sD_epi);                                      // (R2S,R2S_M,R2S_N,PIPE_D)

    auto mma_tile_m = size<0>(TileShapeMNK{}) / size<1>(tRS_rAcc);
    auto mma_tile_n = size<1>(TileShapeMNK{}) / size<2>(tRS_rAcc);
    auto epi_tile_m = size<0>(EpilogueTile{});
    auto epi_tile_n = size<1>(EpilogueTile{});
    // Allocate D registers
    Layout tRS_rD_layout = make_layout(take<0,3>(shape(thread_r2s.partition_D(sD_epi))));
    Tensor tRS_rD = make_tensor<SmemElementD>(tRS_rD_layout);                                   // (R2S,R2S_M,R2S_N)

    TiledCopy tiled_s2r  = detail::make_tiled_copy_C_sqmma<CopyOpS2R, CopyV, EpilogueTile>(tiled_mma);
    ThrCopy thread_s2r   = tiled_s2r.get_slice(thread_idx);
    Tensor tSR_sC        = thread_s2r.partition_S(sC_epi);                                  // (S2R,S2R_M,S2R_N,PIPE_C)
    Layout tSR_rC_layout = thread_s2r.retile_D(tRS_rD).layout();                            // (S2R,S2R_M,S2R_N)

    Tensor tRS_rC = make_tensor<SmemElementC>(tRS_rD_layout);                             // (R2S,R2S_M,R2S_N)
    Tensor tSR_rC = thread_s2r.retile_D(tRS_rC);                                            // (S2R,S2R_M,S2R_N)

    ThrCopy thrblk_g2s = params.tme_load_c.get_slice(Int<0>{});
    Tensor bGS_gC      = thrblk_g2s.partition_S(gC_epi);                                    // (G2S,G2S_M,G2S_N,EPI_M,EPI_N)
    Tensor bGS_sC      = thrblk_g2s.partition_D(sC_epi);                                    // (G2S,G2S_M,G2S_N,PIPE_C)
    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
    ThrCopy thrblk_s2g = params.tme_store_d.get_slice(Int<0>{});
    Tensor bSG_sD      = thrblk_s2g.partition_S(sD_epi);                                    // (S2G,S2G_M,S2G_N,PIPE_D)
    Tensor bSG_gD      = thrblk_s2g.partition_D(gD_epi);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

    MUTE_STATIC_ASSERT(mma_tile_n % epi_tile_n == 0, "MMA_TILE_M must divide EPI_TILE_M");

    MUTE_STATIC_ASSERT(mma_tile_m % epi_tile_m == 0, "EPI_TILE_N must divide MMA_TILE_N");

    // Predication for TMA store (one warp issues TMA store)
    bool issue_tme_store = canonical_warp_idx() == 0;
    bool issue_tme_load  = canonical_warp_idx() == 0;
    constexpr uint32_t start_store = DelayTmeStore ? StagesD - 1: 0;
    constexpr uint32_t TmeTransactionBytes = static_cast<uint32_t>(
        size<0>(EpilogueTile{}) * size<1>(EpilogueTile{}) * sizeof(SmemElementC));
    LoadPipelineParams pipe_params;
    pipe_params.transaction_bytes = TmeTransactionBytes;
    constexpr int NumWarps = size(TiledMma{}) / NumThreadsPerWarp;
    pipe_params.num_consumers = NumWarps;
    LoadPipeline load_pipeline(pipe_params);

    LoadPipelineState smem_pipe_load_read;
    LoadPipelineState smem_pipe_load_write = mutlass::make_producer_start_state<LoadPipelineState>();
    LoadPipelineState smem_pipe_release;
    StorePipelineState smem_pipe_store_read;
    StorePipelineState smem_pipe_store_write;
    

    uint32_t tme_load_tile = 0;
    uint32_t tme_store_tile = 0;
    __syncthreads();

    MUTLASS_PRAGMA_UNROLL
    for (int pipe_c = 0; pipe_c < StagesC-1; ++pipe_c) {
      if (issue_tme_load &&
          is_source_needed() &&
          tme_load_tile < size<2>(gD_epi) * size<3>(gD_epi)) {
        load_pipeline.producer_acquire(smem_pipe_load_write);
        uint32_t bar_id = load_pipeline.producer_get_barrier_id(smem_pipe_load_write);
        uint32_t cur_epi_m = tme_load_tile / size<3>(gD_epi);
        uint32_t cur_epi_n = tme_load_tile % size<3>(gD_epi);
        copy(params.tme_load_c.with(bar_id),bGS_gC(_,_,_,cur_epi_m,cur_epi_n), bGS_sC(_,_,_,pipe_c));
      }
      ++smem_pipe_load_write;
      ++tme_load_tile;
    }
    MUTLASS_PRAGMA_UNROLL
    for (int epi_m = 0; epi_m < size<2>(gD_epi); ++epi_m) {
      MUTLASS_PRAGMA_UNROLL
      for (int epi_n = 0; epi_n < size<3>(gD_epi); ++epi_n) {
        if (is_source_needed()) {
          load_pipeline.consumer_wait(smem_pipe_load_read);
          copy(tiled_s2r, tSR_sC(_,_,_,smem_pipe_load_read.index()), tSR_rC);
        }
        int mma_m = (epi_m * size<0>(EpilogueTile{})) / mma_tile_m;
        int mma_n = (epi_n * size<1>(EpilogueTile{})) / mma_tile_n;
        Tensor tRS_rAcc_mn = tRS_rAcc(_,mma_m,mma_n);
        if constexpr (size<3>(gD_epi) > 1 && size<2>(gD_epi) == 1) {
          MUTLASS_PRAGMA_UNROLL
          for (int epi_v = 0; epi_v < size(tRS_rD); ++epi_v) {
            if (is_source_needed()) {
              tRS_rD(epi_v) = epilogue_op(tRS_rAcc_mn(epi_n + epi_v * size<3>(gD_epi)), tSR_rC(epi_v));
            } else {
              tRS_rD(epi_v) = ElementD(tRS_rAcc_mn(epi_n + epi_v * size<3>(gD_epi)));
            }
          }
        } else {
          int epi_m_in_mma = epi_m % (mma_tile_m / epi_tile_m);
          int epi_n_in_mma = epi_n % (mma_tile_n / epi_tile_n);
          int r2s_v = epi_m_in_mma * size<3>(gD_epi) * size(tRS_rD) + epi_n_in_mma * size(tRS_rD);
          MUTLASS_PRAGMA_UNROLL
          for (int epi_v = 0; epi_v < size(tRS_rD); ++epi_v) {
            if (is_source_needed()) {
              tRS_rD(epi_v) = epilogue_op(tRS_rAcc_mn(r2s_v + epi_v), tSR_rC(epi_v));
            } else {
              tRS_rD(epi_v) = ElementD(tRS_rAcc_mn(r2s_v + epi_v));
            }
          }
        }
        
        // copy tile from register to smem
        if (tme_store_tile > start_store) {
          if (issue_tme_store) {
            mute::tme_store_wait();
          }
          if (not is_source_needed()) {
            __syncthreads();
          }
        }
        copy(r2s_tiled_copy, tRS_rD, tRS_sD(_,_,_,smem_pipe_store_write.index()));
        // if reuse smemC or StagesD == 1, supposed to do s2g immediately
        if constexpr (not DelayTmeStore) {
           __syncthreads();
           if (issue_tme_store) {
            copy(params.tme_store_d, bSG_sD(_,_,_,smem_pipe_store_read.index()), bSG_gD(_,_,_,epi_m, epi_n));
            mute::tme_store_arrive();
           }
          ++smem_pipe_store_read;
        } else if (tme_store_tile >= start_store){
            uint32_t cur_epi_m = (tme_store_tile - StagesD + 1) / size<3>(gD_epi);
            uint32_t cur_epi_n = (tme_store_tile - StagesD + 1) % size<3>(gD_epi);
            if (issue_tme_store) {
              copy(params.tme_store_d, bSG_sD(_,_,_,smem_pipe_store_read.index()), bSG_gD(_,_,_,cur_epi_m, cur_epi_n));
              mute::tme_store_arrive();
            }
            ++smem_pipe_store_read;
        }

        if (issue_tme_load &&
            is_source_needed() &&
            tme_load_tile < size<2>(gD_epi) * size<3>(gD_epi)) {
          load_pipeline.producer_acquire(smem_pipe_load_write);
          uint32_t bar_id = load_pipeline.producer_get_barrier_id(smem_pipe_load_write);
          uint32_t cur_epi_m = tme_load_tile / size<3>(gD_epi);
          uint32_t cur_epi_n = tme_load_tile % size<3>(gD_epi);
          copy(params.tme_load_c.with(bar_id), bGS_gC(_,_,_,cur_epi_m, cur_epi_n), bGS_sC(_,_,_,smem_pipe_load_write.index()));
        }

        if (is_source_needed()) {
          load_pipeline.consumer_release(smem_pipe_release);
        }

        ++tme_load_tile;
        ++tme_store_tile;
        ++smem_pipe_load_write;
        ++smem_pipe_load_read;
        ++smem_pipe_store_write;
        ++smem_pipe_release;

      } // for epi_n
    } // for epi_m
    // store residual smem_D
    if constexpr (DelayTmeStore) {
      MUTLASS_PRAGMA_UNROLL
      for (int pip_d = 0; pip_d < StagesD -1; ++pip_d) {
        if (issue_tme_store) {
          mute::tme_store_wait();
          uint32_t cur_epi_m = (tme_store_tile - StagesD + 1) / size<3>(gD_epi);
          uint32_t cur_epi_n = (tme_store_tile - StagesD + 1) % size<3>(gD_epi);
          copy(params.tme_store_d, bSG_sD(_,_,_,smem_pipe_store_read.index()), bSG_gD(_,_,_,cur_epi_m, cur_epi_n));
          ++smem_pipe_store_read;
          ++tme_store_tile;
          mute::tme_store_arrive();
        }
      }
    }
  }

private:
  Params const& params;
  ThreadEpilogueOp epilogue_op;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
