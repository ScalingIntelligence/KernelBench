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
#include "mutlass/epilogue/dispatch_policy.hpp"
#include "mutlass/epilogue/collective/detail.hpp"
#include "mutlass/epilogue/fusion/callbacks.hpp"
#include "mutlass/epilogue/fusion/mp22_callbacks.hpp"
#include "mutlass/detail/layout.hpp"

#include "mute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::epilogue::collective {
using namespace mute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages_,
  int FragmentSize_,
  class SmemLayout_,
  class TiledCopyS2R_,
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class ElementAccumulator_,
  class FusionCallbacks_
>
class CollectiveEpilogue <
  Mp22CollectiveEpilogue<Stages_, FragmentSize_>,
  SmemLayout_,
  TiledCopyS2R_,
  ElementC_,
  StrideC_,
  ElementD_,
  StrideD_,
  ElementAccumulator_,
  FusionCallbacks_
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = Mp22CollectiveEpilogue<Stages_, FragmentSize_>;
  using SmemLayout = SmemLayout_;
  using TiledCopyS2R = TiledCopyS2R_;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using ElementAccumulator = ElementAccumulator_;
  using FusionCallbacks = FusionCallbacks_;

  using ThreadEpilogueOp = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;

  static_assert(mute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(mute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

private:
  static constexpr int Stages = Stages_;
  static constexpr int FragmentSize = FragmentSize_;

  static constexpr bool is_source_supported = not mute::is_void_v<ElementC>;
  static constexpr bool is_destination_supported = not mute::is_void_v<ElementD>;
  static constexpr int NumBarriers = 0;

public:
  struct SharedStorage {
    mute::array_aligned<ElementAccumulator, mute::cosize_v<SmemLayout>> smem_epi;

    using FusionStorage = typename FusionCallbacks::SharedStorage;
    FusionStorage thread;
  };

  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const* ptr_C;
    StrideC dC;
    ElementD* ptr_D;
    StrideD dD;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return FusionCallbacks::get_workspace_size(problem_shape, args.thread);
  }

  template <class ProblemShape>
  static mutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, musaStream_t stream) {
    return FusionCallbacks::initialize_workspace(problem_shape, args.thread, workspace, stream);
  }

  MUTLASS_HOST_DEVICE
  CollectiveEpilogue(Params const& params_, SharedStorage& shared_storage)
    : params(params_), fusion_callbacks(params_.thread, shared_storage.thread) {}


  template <
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  MUTLASS_HOST_DEVICE void
  operator() (
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      mute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf)
  {
    using namespace mute;
    using X = Underscore;
    using ElementCompute_ = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::ElementCompute;
    using ElementCompute  = mute::conditional_t<mute::is_void_v<ElementCompute_>, ElementAccumulator, ElementCompute_>;

    static_assert(mute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(mute::rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(mute::rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 4");
    static_assert(rank(FrgLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA,MMA_M,MMA_N)");
    static_assert(is_rmem<FrgEngine>::value, "Accumulator must be RF resident.");
    static_assert(is_same_v<typename FrgEngine::value_type, ElementAccumulator>, "Accumulator type doesn't match");


    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    auto residue_mn = make_tuple(get<0>(residue_mnk), get<1>(residue_mnk));

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M,N,L), params.dC);      //             (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), params.dD);      //             (m,n,l)
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gC = gC_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)


    auto mma_tile_m = tile_size<0>(tiled_mma);
    auto mma_tile_n = tile_size<1>(tiled_mma);
    auto epi_tile_m = size<0>(SmemLayout{});
    auto epi_tile_n = size<1>(SmemLayout{});

    static_assert(epi_tile_m % mma_tile_m == 0, "MMA_TILE_M must divide EPI_TILE_M");
    static_assert(epi_tile_n % mma_tile_n == 0, "MMA_TILE_N must divide EPI_TILE_N");
    static_assert(typename TiledCopyS2R::TiledNumThr{} == size(tiled_mma));

    // Construct a tensor in SMEM that we can partition for rearranging data
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sC = make_tensor(make_smem_ptr(storage.smem_epi.data()), SmemLayout{});

    using CopyAtomR2S = Copy_Atom<DefaultCopy, ElementAccumulator>;
    auto tiled_copy_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto thread_r2s = tiled_copy_r2s.get_thread_slice(thread_idx);
    Tensor tCaC = thread_r2s.retile_S(accumulators);                                          // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor tCsC = thread_r2s.partition_D(sC);                                                 // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Tile gD and gC by the shape of SmemLayout first
    auto tile  = make_shape(size<0>(sC), size<1>(sC));
    Tensor gCt = flat_divide(gC, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor gDt = flat_divide(gD, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)

    auto tiled_copy_s2r = TiledCopyS2R{};
    auto thread_s2r = tiled_copy_s2r.get_thread_slice(thread_idx);
    Tensor tDsC = thread_s2r.partition_S(sC);                                   //               ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDgC = thread_s2r.partition_D(gCt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    Tensor tDgD = thread_s2r.partition_D(gDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    // Allocate intermediate registers on the dst tensors
    Tensor tDrC = make_tensor<ElementAccumulator>(take<0,3>(shape(tDgC)));       // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDrD = make_tensor<ElementD>(shape(tDrC));                            // ((Atom,AtomNum),ATOM_M,ATOM_N)

    // Allocate intermediate registers on the src tensors
    Tensor tCrC = make_tensor<ElementC>(shape(tDrC));                            // ((Atom,AtomNum),ATOM_M,ATOM_N)

    // Construct proper tiled copy
    constexpr int vec_elem = decltype(max_common_vector(tDgC, tDgD))::value;
    constexpr int vec_bits = vec_elem * sizeof_bits_v<ElementC>;
    using CopyAtomG2R = Copy_Atom<
            UniversalCopy<uint_bit_t<mute::min(1024, vec_bits)>>, ElementC>;

    using TiledCopyG2R = TiledCopy<CopyAtomG2R,
                              typename TiledCopyS2R::TiledLayout_TV,
                              typename TiledCopyS2R::Tiler_MN>;
    TiledCopyG2R tiled_copy_g2r{};
    auto thr_copy_g2r = tiled_copy_g2r.get_slice(thread_idx);
    Tensor tCgC       = thr_copy_g2r.partition_S(gCt);
    Tensor tCrC_view  = thr_copy_g2r.retile_D(tCrC);

    // Repeat the D-partitioning for coordinates and predication
    Tensor cD   = make_identity_tensor(make_shape(size<0>(gD),size<1>(gD)));          // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cDt  = flat_divide(cD, tile);                                //                (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor tDcD = thread_s2r.partition_D(cDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    bool is_C_load_needed = is_source_supported and fusion_callbacks.is_C_load_needed();

    auto args = mutlass::epilogue::fusion::detail::VisitorArgs {
                  problem_shape_mnkl,
                  blk_shape_MNK,
                  blk_coord_mnkl,
                  residue_mn,
                  tile,
                  tiled_copy_s2r,
                  thread_idx,
                  cD,
                  tDcD,
                  tCrC,
                  tDgD
                };

    auto callbacks = fusion_callbacks.get_callbacks(args);

    callbacks.begin_epilogue();

    MUTLASS_PRAGMA_UNROLL
    for (int epi_m = 0; epi_m < size<2>(cDt); ++epi_m) {
      MUTLASS_PRAGMA_UNROLL
      for (int epi_n = 0; epi_n < size<3>(cDt); ++epi_n) {
        int step_idx = epi_m * size<3>(cDt) + epi_n;
        callbacks.begin_step(epi_m, epi_n, step_idx);

        MUTLASS_PRAGMA_UNROLL
        for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {
          MUTLASS_PRAGMA_UNROLL
          for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
            int mma_m = epi_m * size<1>(tCsC) + pipe_m;
            int mma_n = epi_n * size<2>(tCsC) + pipe_n;
            copy(tiled_copy_r2s, tCaC(_, mma_m, mma_n), tCsC(_, pipe_m, pipe_n));
          } // for pipe_n
        } // for pipe_m

        __syncthreads();

        copy(tiled_copy_s2r, tDsC, tDrC);

        __syncthreads();

        Tensor tDgDmn = tDgD(_,_,_,epi_m,epi_n);
        Tensor tDcDmn = tDcD(_,_,_,epi_m,epi_n);

        if (is_C_load_needed) {
          tCrC_view = tCgC(_, _, _, epi_m, epi_n);
        }

        Tensor tSR_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(tDrC);
        Tensor tSR_rD_frg   = recast<Array<ElementD          , FragmentSize>>(tDrD);

        MUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size(tSR_rAcc_frg); ++epi_v) {
          tSR_rD_frg(epi_v) = callbacks.visit(tSR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
        }

        callbacks.end_step(epi_m, epi_n, step_idx);

        if constexpr (is_destination_supported) {
          MUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tDgDmn); ++m) {
            MUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tDgDmn); ++n) {
              if (elem_less(tDcDmn(0, m, n), residue_mn)) {
                copy(tDrD(_, m, n), tDgDmn(_, m, n));
              }
            } // for n
          } // for m
        }
      } // for epi_n
    } // for epi_m
    callbacks.end_epilogue();
  }
private:
  Params const& params;
  FusionCallbacks fusion_callbacks;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::epilogue::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
