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
struct PagedFmhaKernelTmeWarpSpecialized {

  static constexpr int NumLoadWarpSquads = CollectiveMainloop::NumLoadWarpSquads;
  static constexpr int NumMmaWarpSquads  = CollectiveMainloop::NumMmaWarpSquads;

  using TileShape = typename CollectiveMainloop::TileShape;

  using SharedStorage = typename CollectiveMainloop::SharedStorage;

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // 0:Q, 1:K, 2:D_QK, 3:D_VO, 4:H_Q, 5:H_K, 6:B
  using ProblemShape = Shape<int, int, int, int, int, int, int>;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock = (NumLoadWarpSquads + NumMmaWarpSquads) * NumThreadsPerWarpSquad;
  static constexpr uint32_t NumMmaThreads = NumMmaWarpSquads * NumThreadsPerWarpSquad;
  static constexpr uint32_t NumMmaWarps   = NumMmaThreads / NumThreadsPerWarp;

  static constexpr int SmemAlignmentBytes = 256;

  static constexpr bool PackGQA = false;

  using NamedBarrier = mutlass::arch::AsyncBarrier;

  using MainloopPipelineQ = typename CollectiveMainloop::MainloopPipelineQ;
  using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
  using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
  using MainloopPipelineQParams = typename CollectiveMainloop::PipelineQParams;
  using MainloopPipelineQState = typename CollectiveMainloop::PipelineQState;
  using MainloopPipelineKParams = typename CollectiveMainloop::PipelineKParams;
  using MainloopPipelineKState = typename CollectiveMainloop::PipelineKState;
  using MainloopPipelineVParams = typename CollectiveMainloop::PipelineVParams;
  using MainloopPipelineVState = typename CollectiveMainloop::PipelineVState;

  static constexpr uint32_t StagesPerMathSquad = 2;
  using MathOrderBarrier = mutlass::OrderedSequenceBarrier<StagesPerMathSquad, NumMmaWarpSquads>;
  using MathOrderBarrierParams = typename MathOrderBarrier::Params;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineQ[MainloopPipelineQ::NumBarriers];
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


  MUTLASS_DEVICE
  void operator()(const Params& params, char* smem) {
    enum class WarpSquadRole {
      Producer  = 0,
      Consumer0 = 1,
      Consumer1 = 2,
      Consumer2 = 3,
      Consumer3 = 4,
    };

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

    MainloopPipelineQParams pipeline_params_q;
    pipeline_params_q.transaction_bytes = CollectiveMainloop::TmeTransactionBytesQ;
    pipeline_params_q.num_consumers = NumMmaWarps;
    MainloopPipelineQ pipeline_q(pipeline_params_q, reinterpret_cast<uint64_t>(&barrier_storage->PipelineQ));
    MainloopPipelineQState mainloop_pipe_q_producer_state = mutlass::make_producer_start_state<MainloopPipelineQ>();
    MainloopPipelineQState mainloop_pipe_q_consumer_state;

    MainloopPipelineKParams pipeline_params_k;
    pipeline_params_k.transaction_bytes = CollectiveMainloop::TmeTransactionBytesK;
    pipeline_params_k.num_consumers = NumMmaWarps;
    MainloopPipelineK pipeline_k(pipeline_params_k, reinterpret_cast<uint64_t>(&barrier_storage->PipelineK));
    MainloopPipelineKState mainloop_pipe_k_producer_state = mutlass::make_producer_start_state<MainloopPipelineK>();
    MainloopPipelineKState mainloop_pipe_k_consumer_state;

    MainloopPipelineVParams pipeline_params_v;
    pipeline_params_v.transaction_bytes = CollectiveMainloop::TmeTransactionBytesV;
    pipeline_params_v.num_consumers = NumMmaWarps;
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

        mainloop.load(params.mainloop, shared_storage,
                      pipeline_q, mainloop_pipe_q_producer_state,
                      pipeline_k, mainloop_pipe_k_producer_state,
                      pipeline_v, mainloop_pipe_v_producer_state,
                      blk_coord, params.problem_size, warp_idx);
      }
    }
    else if (warp_squad_role == WarpSquadRole::Consumer0 ||
             warp_squad_role == WarpSquadRole::Consumer1 ||
             warp_squad_role == WarpSquadRole::Consumer2 ||
             warp_squad_role == WarpSquadRole::Consumer3) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_work_tile_coord();

        auto [Q, K_, D_QK, D_VO, H, H_K, B] = params.problem_size;
        int cur_seqlen = params.mainloop.ptr_seqlen[get<3>(blk_coord)];
        auto problem_size = make_shape(Q, cur_seqlen, D_QK, D_VO, H, H_K, B);

        auto results =
          mainloop.compute(blk_coord, params.mainloop, problem_size,
                           pipeline_q, mainloop_pipe_q_consumer_state,
                           pipeline_k, mainloop_pipe_k_consumer_state,
                           pipeline_v, mainloop_pipe_v_consumer_state,
                           math_order_barrier,
                           shared_storage, thread_idx_in_warp_squad, consumer_warp_squad_idx);

        typename CollectiveMainloop::TiledMmaPV tiled_mma_pv;
        auto consumer_qo_coord = get<0>(blk_coord) * NumMmaWarpSquads + consumer_warp_squad_idx;
        epilogue(blk_coord, results, tiled_mma_pv, select<0,3,4,6>(params.problem_size), 0, params.epilogue, thread_idx_in_warp_squad, consumer_qo_coord);
      }
    }
  }
};

} // namespace mutlass::fmha::kernel
