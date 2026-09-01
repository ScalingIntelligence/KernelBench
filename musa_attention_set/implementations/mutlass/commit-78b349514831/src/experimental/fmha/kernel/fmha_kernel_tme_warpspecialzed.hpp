#pragma once

#include <mutlass/mutlass.h>
#include <mutlass/pipeline/pipeline.hpp>

#include <mute/tensor.hpp>

#include "fmha_options.hpp"
#include "collective/fmha_common.hpp"


namespace mutlass::fmha::kernel {

using namespace mute;

template <
  class CollectiveMainloop,
  class CollectiveEpilogue,
  class TileScheduler,
  class... Options
>
struct FmhaKernelTmeWarpSpecialized {

  static constexpr int NumLoadWarpSquads = CollectiveMainloop::NumLoadWarpSquads;
  static constexpr int NumMmaWarpSquads  = CollectiveMainloop::NumMmaWarpSquads;

  using TileShape = typename CollectiveMainloop::TileShape;

  using SharedStorage = typename CollectiveMainloop::SharedStorage;

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static constexpr bool IsVarlen = CollectiveMainloop::IsVarlen;

  // 0:Q, 1:K, 2:D_QK, 3:D_VO, 4:H_Q, 5:H_K, 6:B
  using ProblemShapeFixed  = Shape<int, int, int, int, int, int, int>;
  using ProblemShapeVarlen = Shape<collective::VariableLength, collective::VariableLength, int, int, int, int, int>;

  using ProblemShape = conditional_t<IsVarlen,
                                     ProblemShapeVarlen,
                                     ProblemShapeFixed>;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock = (NumLoadWarpSquads + NumMmaWarpSquads) * NumThreadsPerWarpSquad;
  static constexpr uint32_t NumMmaThreads = NumMmaWarpSquads * NumThreadsPerWarpSquad;
  static constexpr uint32_t NumMmaWarps   = NumMmaThreads / NumThreadsPerWarp;

  static constexpr int SmemAlignmentBytes = 256;

  static constexpr bool UseTmeLoadQ = find_option_t<Tag::TmeLoadQ, true_type, Options...>::value;
  static constexpr bool SIMT_KEY = CollectiveMainloop::SIMT_KEY;
  using BarrierQ = conditional_t<UseTmeLoadQ, mutlass::arch::AsyncTransactionBarrier, mutlass::arch::AsyncBarrier>;
  using NamedBarrier = mutlass::arch::AsyncBarrier;

  using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
  using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
  using MainloopPipelineKParams = typename CollectiveMainloop::PipelineKParams;
  using MainloopPipelineKState = typename CollectiveMainloop::PipelineKState;
  using MainloopPipelineVParams = typename CollectiveMainloop::PipelineVParams;
  using MainloopPipelineVState = typename CollectiveMainloop::PipelineVState;

  static constexpr uint32_t StagesPerMathSquad = 2;
  using MathOrderBarrier = mutlass::OrderedSequenceBarrier<StagesPerMathSquad, NumMmaWarpSquads>;
  using MathOrderBarrierParams = typename MathOrderBarrier::Params;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t BarrierQ;
    uint8_t BarrierTail;
    uint8_t PipelineK[MainloopPipelineK::NumBarriers];
    uint8_t PipelineV[MainloopPipelineV::NumBarriers];
    uint8_t PipelineMath[MathOrderBarrier::NumBarriers];
  };

  struct Arguments {
    ProblemShape problem_size;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
  };

  struct Params {
    ProblemShape problem_size;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params scheduler;
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace = nullptr) {
    return Params {
      args.problem_size,
      CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_size, args.epilogue, workspace),
      TileScheduler::to_underlying_arguments(args.problem_size, TileShape{}),
    };
  }

  template <class BatchCoord>
  MUTLASS_DEVICE
  auto
  apply_batch_offset(ProblemShape const& problem_shape, BatchCoord const& batch_coord) {
    auto [problem_size, blk_offset] = collective::apply_variable_length_offset(problem_shape, batch_coord);
    auto k_offset_remainder = 0;

    // For varlen and not SIMT Key, we will adjust seqlen k
    if constexpr (IsVarlen && !SIMT_KEY) {
      k_offset_remainder = get<1>(blk_offset) % CollectiveMainloop::TmeLoadKeyBuilder::Granularity;
      get<1>(blk_offset) = get<1>(blk_offset) - k_offset_remainder;
      get<1>(problem_size) = get<1>(problem_size) + k_offset_remainder;
    }

    auto blk_offset_qk = make_tuple(get<0>(blk_offset), get<1>(blk_offset));

    return mute::make_tuple(problem_size, blk_offset_qk, k_offset_remainder);
  }

  MUTLASS_DEVICE
  void operator()(const Params& params, char* smem) {
    enum class WarpSquadRole {
      Producer  = 0,
      Consumer0 = 1,
      Consumer1 = 2,
      Consumer2 = 3,
      Consumer3 = 4,
    };

    auto& storage = *reinterpret_cast<SharedStorage*>(smem);

    int thread_idx = threadIdx.x;
    int warp_idx = mutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % NumThreadsPerWarp;
    int warp_squad_idx = mutlass::canonical_warp_squad_idx();
    int warp_idx_in_warp_squad = warp_idx % NumWarpsPerWarpSquad;
    int consumer_warp_squad_idx = warp_squad_idx - static_cast<int>(WarpSquadRole::Consumer0);
    int thread_idx_in_warp_squad = thread_idx % NumThreadsPerWarpSquad;

    auto warp_squad_role = WarpSquadRole(warp_squad_idx);

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    BarrierQ barrier_q(reinterpret_cast<uint64_t>(&barrier_storage->BarrierQ));
    NamedBarrier barrier_tail(reinterpret_cast<uint64_t>(&barrier_storage->BarrierTail));

    if (warp_idx == 0) {
      barrier_q.init(UseTmeLoadQ ? 1 : 4);
      barrier_tail.init(4);
    }

    MainloopPipelineKParams pipeline_params_k;
    if constexpr (SIMT_KEY) {
      pipeline_params_k.producer_arv_count = NumLoadWarpSquads * NumWarpsPerWarpSquad;
      pipeline_params_k.consumer_arv_count = NumMmaWarps;
    } else {
      pipeline_params_k.transaction_bytes = CollectiveMainloop::TmeTransactionBytesK;
      pipeline_params_k.num_consumers = NumMmaWarps;
      pipeline_params_k.num_producers = 1;
    }
    MainloopPipelineK pipeline_k(pipeline_params_k, reinterpret_cast<uint64_t>(&barrier_storage->PipelineK));
    MainloopPipelineKState mainloop_pipe_k_producer_state = mutlass::make_producer_start_state<MainloopPipelineK>();
    MainloopPipelineKState mainloop_pipe_k_consumer_state;

    MainloopPipelineVParams pipeline_params_v;
    pipeline_params_v.transaction_bytes = CollectiveMainloop::TmeTransactionBytesV;
    pipeline_params_v.num_consumers = NumMmaWarps;
    pipeline_params_v.num_producers = 1;
    MainloopPipelineV pipeline_v(pipeline_params_v, reinterpret_cast<uint64_t>(&barrier_storage->PipelineV));
    MainloopPipelineVState mainloop_pipe_v_producer_state = mutlass::make_producer_start_state<MainloopPipelineV>();
    MainloopPipelineVState mainloop_pipe_v_consumer_state;

    MathOrderBarrierParams math_order_barrier_params;
    math_order_barrier_params.group_id = consumer_warp_squad_idx;
    math_order_barrier_params.group_size = NumWarpsPerWarpSquad;
    MathOrderBarrier math_order_barrier(math_order_barrier_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineMath));

    __syncthreads();

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    TileScheduler tile_scheduler(params.scheduler);

    if (warp_squad_role == WarpSquadRole::Producer) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_work_tile_coord();

        auto [problem_size, blk_offset, prev_mask_boundary] = apply_batch_offset(params.problem_size, get<3>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(problem_size)) {
          continue;
        }
        // Or we can make batch_stride=0
        if constexpr (IsVarlen) {
          get<3>(blk_coord) = 0;
        }

        auto barrier_tuple = make_tuple(barrier_q, barrier_tail);
        mainloop.load(params.mainloop, shared_storage, barrier_tuple,
                      pipeline_k, mainloop_pipe_k_producer_state,
                      pipeline_v, mainloop_pipe_v_producer_state,
                      blk_coord, problem_size, blk_offset, prev_mask_boundary, warp_idx);
      }
    }
    else if (warp_squad_role == WarpSquadRole::Consumer0 ||
             warp_squad_role == WarpSquadRole::Consumer1 ||
             warp_squad_role == WarpSquadRole::Consumer2 ||
             warp_squad_role == WarpSquadRole::Consumer3) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_work_tile_coord();
        auto [problem_size, blk_offset, prev_mask_boundary] = apply_batch_offset(params.problem_size, get<3>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(problem_size)) {
          continue;
        }

        // Or we can make batch_stride=0
        if constexpr (IsVarlen) {
          get<3>(blk_coord) = 0;
        }

        auto results =
          mainloop.compute(blk_coord, params.mainloop, problem_size, prev_mask_boundary,
                                         pipeline_k, mainloop_pipe_k_consumer_state,
                                         pipeline_v, mainloop_pipe_v_consumer_state,
                                         barrier_q, math_order_barrier, shared_storage,
                                         thread_idx_in_warp_squad, consumer_warp_squad_idx);

        typename CollectiveMainloop::TiledMmaPV tiled_mma_pv;

        auto consumer_qo_coord = get<0>(blk_coord) * NumMmaWarpSquads + consumer_warp_squad_idx;
        epilogue(blk_coord, results, tiled_mma_pv, select<0,3,4,6>(problem_size), blk_offset, params.epilogue, thread_idx_in_warp_squad, consumer_qo_coord);
      }
    }
  }
};

} // namespace mutlass::fmha::kernel
