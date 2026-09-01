#pragma once

#include <mutlass/mutlass.h>
#include <mutlass/pipeline/pipeline.hpp>

#include <mute/tensor.hpp>

#include "fmha_options.hpp"
#include "collective/fmha_common.hpp"

using namespace mute;

namespace mutlass::fmha::kernel {

template <
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class... Options
>
struct MlaKernelTmeWarpSpecialized {

  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;

  static constexpr int NumLoadWarpSquads = CollectiveMainloop::NumLoadWarpSquads;
  static constexpr int NumMmaWarpSquads  = CollectiveMainloop::NumMmaWarpSquads;
  static constexpr int NumTransposeWarpSquads = CollectiveMainloop::NumTransposeWarpSquads;

  static constexpr bool InKernelTranspose = NumTransposeWarpSquads > 0;

  using TileShape = typename CollectiveMainloop::TileShape;
  using SharedStorage = typename CollectiveMainloop::SharedStorage;

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // 0:Q, 1:PageSize, 2:(L, R), 3:H, 4:PageCount 5:B
  using ProblemShape = Shape<int, int, Shape<int, int>, int, int, int>;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock = (NumLoadWarpSquads + NumMmaWarpSquads + NumTransposeWarpSquads) * NumThreadsPerWarpSquad;
  static constexpr uint32_t NumMmaThreads = NumMmaWarpSquads * NumThreadsPerWarpSquad;
  static constexpr uint32_t NumMmaWarps   = NumMmaThreads / NumThreadsPerWarp;
  static constexpr uint32_t NumTransWarps = NumTransposeWarpSquads * NumWarpsPerWarpSquad;

  static constexpr int SmemAlignmentBytes = 256;

  using MainloopPipelineQKLatent = typename CollectiveMainloop::MainloopPipelineQKLatent;
  using PipelineQKLatentParams = typename CollectiveMainloop::PipelineQKLatentParams;
  using PipelineQKLatentState = typename CollectiveMainloop::PipelineQKLatentState;

  using MainloopPipelineQKRope = typename CollectiveMainloop::MainloopPipelineQKRope;
  using PipelineQKRopeParams = typename CollectiveMainloop::PipelineQKRopeParams;
  using PipelineQKRopeState = typename CollectiveMainloop::PipelineQKRopeState;

  using MainloopPipelinePV = typename CollectiveMainloop::MainloopPipelinePV;
  using PipelinePVParams = typename CollectiveMainloop::PipelinePVParams;
  using PipelinePVState = typename CollectiveMainloop::PipelinePVState;

  using NamedBarrier = mutlass::arch::AsyncBarrier;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineQKLatent[MainloopPipelineQKLatent::NumBarriers];
    uint8_t PipelineQKRope[MainloopPipelineQKRope::NumBarriers];
    uint8_t PipelinePV[MainloopPipelinePV::NumBarriers];
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
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace = nullptr) {
    return Params {
      args.problem_size,
      CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_size, args.epilogue, workspace),
    };
  }

  MUTLASS_DEVICE
  void operator()(const Params& params, char* smem) {

    enum class WarpSquadRole {
      Producer   = 0,
      Consumer   = 1,
      Transposer = 2,
    };

    auto& storage = *reinterpret_cast<SharedStorage*>(smem);

    int thread_idx = threadIdx.x;
    int warp_idx = mutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % NumThreadsPerWarp;
    int warp_squad_idx = mutlass::canonical_warp_squad_idx();
    int warp_idx_in_warp_squad = warp_idx % NumWarpsPerWarpSquad;
    int consumer_warp_squad_idx = warp_squad_idx - static_cast<int>(WarpSquadRole::Consumer);
    int thread_idx_in_warp_squad = thread_idx % NumThreadsPerWarpSquad;

    auto warp_squad_role = WarpSquadRole(warp_squad_idx);

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineQKLatentParams pipeline_params_qk_latent;
    pipeline_params_qk_latent.transaction_bytes = CollectiveMainloop::TmeTransactionBytesC;
    pipeline_params_qk_latent.num_consumers = InKernelTranspose ? NumMmaWarps + NumTransWarps : NumMmaWarps;
    pipeline_params_qk_latent.num_producers = 1;

    MainloopPipelineQKLatent pipeline_qk_latent(pipeline_params_qk_latent, reinterpret_cast<uint64_t>(&barrier_storage->PipelineQKLatent));
    PipelineQKLatentState mainloop_pipe_qk_latent_producer_state = mutlass::make_producer_start_state<MainloopPipelineQKLatent>();
    PipelineQKLatentState mainloop_pipe_qk_latent_consumer_state;

    PipelineQKRopeParams pipeline_params_qk_rope;
    pipeline_params_qk_rope.transaction_bytes = CollectiveMainloop::TmeTransactionBytesKRope;
    pipeline_params_qk_rope.num_consumers = NumMmaWarps;
    pipeline_params_qk_rope.num_producers = 1;

    MainloopPipelineQKRope pipeline_qk_rope(pipeline_params_qk_rope, reinterpret_cast<uint64_t>(&barrier_storage->PipelineQKRope));
    PipelineQKRopeState mainloop_pipe_qk_rope_producer_state = mutlass::make_producer_start_state<MainloopPipelineQKRope>();
    PipelineQKRopeState mainloop_pipe_qk_rope_consumer_state;

    PipelinePVParams pipeline_params_pv;
    if constexpr (InKernelTranspose) {
      pipeline_params_pv.consumer_arv_count = NumMmaWarps;
      pipeline_params_pv.producer_arv_count = NumTransWarps;
    } else {
      pipeline_params_pv.transaction_bytes = CollectiveMainloop::TmeTransactionBytesC;
      pipeline_params_pv.num_consumers = NumMmaWarps;
      pipeline_params_pv.num_producers = 1;
    }

    MainloopPipelinePV pipeline_pv(pipeline_params_pv, reinterpret_cast<uint64_t>(&barrier_storage->PipelinePV));
    PipelinePVState mainloop_pipe_pv_producer_state = mutlass::make_producer_start_state<MainloopPipelinePV>();
    PipelinePVState mainloop_pipe_pv_consumer_state;

    __syncthreads();

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    auto [qo_coord, split_coord, batch_coord] = static_cast<uint3>(blockIdx);

    auto blk_coord = make_coord(qo_coord, batch_coord, split_coord);

    if (warp_squad_role == WarpSquadRole::Producer) {
      mainloop.load(params.mainloop, shared_storage,
                    pipeline_qk_latent, mainloop_pipe_qk_latent_producer_state,
                    pipeline_qk_rope, mainloop_pipe_qk_rope_producer_state,
                    pipeline_pv, mainloop_pipe_pv_producer_state,
                    blk_coord, params.problem_size, warp_idx);
    }
    else if (warp_squad_role == WarpSquadRole::Consumer) {
      auto [Q, PageSize, D, H, PageCount, B] = params.problem_size;
      auto [D_latent, D_rope] = D;
      int cur_seqlen = params.mainloop.ptr_seqlen[batch_coord];

      auto problem_size = make_shape(Q, cur_seqlen, D_latent, D_latent, H, 1, B);

      auto results =
        mainloop.compute(blk_coord, params.mainloop, problem_size,
                         pipeline_qk_latent, mainloop_pipe_qk_latent_consumer_state,
                         pipeline_qk_rope, mainloop_pipe_qk_rope_consumer_state,
                         pipeline_pv, mainloop_pipe_pv_consumer_state,
                         shared_storage, thread_idx_in_warp_squad);

      typename CollectiveMainloop::TiledMmaPV tiled_mma_pv;

      auto blk_coord_epilogue = make_coord(qo_coord, _0{}, _0{}, batch_coord);

      auto problem_size_epi = make_shape(Q * H, cur_seqlen, 1, B);
      epilogue(blk_coord_epilogue, results, tiled_mma_pv, problem_size_epi, 0, params.epilogue, thread_idx_in_warp_squad, get<0>(blk_coord));
    }
    else if (warp_squad_role == WarpSquadRole::Transposer) {
      if constexpr (InKernelTranspose) {
        auto [Q, PageSize, D, H, PageCount, B] = params.problem_size;
        auto [D_latent, D_rope] = D;
        int cur_seqlen = params.mainloop.ptr_seqlen[batch_coord];

        auto problem_size = make_shape(Q, cur_seqlen, D_latent, D_latent, H, 1, B);

        mainloop.transpose(blk_coord, params.mainloop, problem_size,
                           pipeline_qk_latent, mainloop_pipe_qk_latent_consumer_state,
                           pipeline_pv, mainloop_pipe_pv_producer_state,
                           shared_storage, thread_idx_in_warp_squad);
      }

    }
  }
};

} // namespace mutlass::fmha::kernel
