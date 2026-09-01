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
#include "mutlass/kernel_hardware_info.hpp"
#include "mutlass/epilogue/collective/detail.hpp"
#include "mutlass/gemm/kernel/tile_scheduler.hpp"
#include "mutlass/gemm/gemm.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/trace.h"
#include "mutlass/arch/barrier.hpp"

#include "mute/tensor.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  mute::enable_if_t<mute::is_base_of_v<KernelTme, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //

  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  static_assert(ArchTag::kMinComputeCapability == 31);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  //static_assert(mute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
  //  "Mainloop and epilogue do not agree on accumulator value type.");
  static_assert(mute::is_void_v<TileScheduler_> || mute::is_same_v<TileScheduler_, DefaultScheduler>,
    "TME kernel only support default tile scheduler");

  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
        TileScheduler_, ArchTag, TileShape, ClusterShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr int SharedStorageSize = static_cast<int>(mute::max(
        sizeof(typename CollectiveMainloop::SharedStorage),
        sizeof(typename CollectiveEpilogue::SharedStorage)));

  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr int SmemAlignmentBytes = CollectiveMainloop::SmemAlignmentBytes;

  static constexpr int NumBarriers = mute::max(CollectiveEpilogue::NumBarriers, CollectiveMainloop::NumBarriers);
  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };


  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    KernelHardwareInfo hw_info{args.hw_info.device_id, args.hw_info.sm_count};
    auto problem_shape_MNKL = append<4>(args.problem_shape, Int<1>{});
    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
      hw_info,
      TileScheduler::to_underlying_arguments(problem_shape_MNKL, TileShape{}, hw_info, args.scheduler)
    };
  }

  MUTLASS_HOST_DEVICE static
  bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    if (!implementable) {
      MUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static int
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static mutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, musaStream_t stream = nullptr,
      MusaHostAdapter* musa_adapter = nullptr) {
    return Status::kSuccess;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    TileSchedulerArguments args{};
    args.swizzle_size = params.scheduler.swizzle_size_;
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  MUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace mute;
    using X = Underscore;

    // Preconditions
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    int thread_idx = int(threadIdx.x);
    int warp_idx   = canonical_warp_idx();

    arch::allocate_async_barriers(NumBarriers);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    //// TME requires special handling of strides to deal with coord codomain mapping
    //// Represent the full tensors -- get these from TMA
    //Tensor mA_mkl = params.mainloop.tme_load_a.get_tme_tensor(make_shape(M,K,L));                            // (m,k,l)
    //Tensor mB_nkl = params.mainloop.tme_load_b.get_tme_tensor(make_shape(N,K,L));                            // (n,k,l)

    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    //auto blk_coord = make_coord(_,_,_);                                                   // (m,n,k) -- defer the slice

    //// Make tiled views
    //Tensor gA_mkl = local_tile(mA_mkl, blk_shape, blk_coord, Step<_1, X,_1>{});                  // (BLK_M,BLK_K,m,k,l)
    //Tensor gB_nkl = local_tile(mB_nkl, blk_shape, blk_coord, Step< X,_1,_1>{});                  // (BLK_N,BLK_K,n,k,l)

    // Perform the collective scoped MMA
    CollectiveMainloop collective_mma;

    auto load_inputs = collective_mma.load_init(problem_shape_MNKL, params.mainloop);

    static_assert(mute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    TileScheduler scheduler{params.scheduler};
    auto work_tile_info = scheduler.initial_work_tile_info();
    auto m_coord = idx2crd(int(work_tile_info.M_idx), shape<2>(gA_mkl));
    auto n_coord = idx2crd(int(work_tile_info.N_idx), shape<2>(gB_nkl));
    auto l_coord = idx2crd(int(work_tile_info.L_idx), shape<4>(gB_nkl));
    auto output_tile_coord = make_coord(m_coord, n_coord, _, l_coord);

    // Slice with m_coord and n_coord
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                       // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                       // (BLK_N,BLK_K,k)

    // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));                   // (MMA,MMA_M,MMA_N)

    auto k_tile_iter  = mute::make_coord_iterator(shape<2>(gA));
    auto k_tile_count = size<2>(gA);

    collective_mma(
      load_inputs,
      output_tile_coord,
      accumulators,
      k_tile_iter, k_tile_count,
      thread_idx,
      smem_buf,
      params.mainloop
    );

    constexpr int BLK_M_RANK = rank<0>(blk_shape);
    bool m_oob = int(work_tile_info.M_idx) >= size<2>(gA_mkl);
    auto m_max_coord = unwrap(mute::transform(make_seq<BLK_M_RANK>{}, [&](auto i) {
        return  m_oob ? 0 : get<i>(M) - get<0,i>(blk_shape) * get<i>(m_coord);
      }));

    constexpr int BLK_N_RANK = rank<1>(blk_shape);
    bool n_oob = int(work_tile_info.N_idx) >= size<2>(gB_nkl);
    auto n_max_coord = unwrap(mute::transform(make_seq<BLK_N_RANK>{}, [&](auto i) {
        return  n_oob ? 0 : get<i>(N) - get<1,i>(blk_shape) * get<i>(n_coord);
      }));
    auto residue_mnk = mute::make_tuple(m_max_coord, n_max_coord, Int<0>{});

    // Epilogue and write to gD
    CollectiveEpilogue epilogue{params.epilogue};
    epilogue(
      problem_shape_MNKL,
      blk_shape,
      output_tile_coord,
      accumulators,
      tiled_mma,
      residue_mnk,
      thread_idx,
      smem_buf
    );
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::kernel

///////////////////////////////////////////////////////////////////////////////
