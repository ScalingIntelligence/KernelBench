#pragma once

#include <mutlass/mutlass.h>

#include <mute/tensor.hpp>
#include <mutlass/pipeline/pipeline.hpp>

#include "collective/fmha_common.hpp"
#include "fmha_options.hpp"

using namespace mute;
using namespace mutlass;
using namespace mutlass::fmha::collective;

namespace mate::flash_mla {

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_, class... Options>
struct MlaKernelTmeWarpSpecialized {
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler      = TileScheduler_;

  static constexpr int  NumLoadWarpSquads   = CollectiveMainloop::NumLoadWarpSquads;
  static constexpr int  NumMmaWarpSquads    = CollectiveMainloop::NumMmaWarpSquads;
  static constexpr bool UseTrivialScheduler = TileScheduler::UseTrivialScheduler;
  static constexpr bool VarlenQ             = CollectiveMainloop::VarlenQ;

  using TileShape     = typename CollectiveMainloop::TileShape;
  using SharedStorage = typename CollectiveMainloop::SharedStorage;

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // 0:Q, 1:PageSize, 2:(L, R), 3:H, 4:PageCount 5:B, 6:TotalQ, 7:MaxQ (varlen Q)
  using ProblemShapeFixed   = Shape<int, int, Shape<int, int>, int, int, int, int, int>;
  using ProblemShapeVerlenQ = Shape<VariableLength, int, Shape<int, int>, int, int, int, int, int>;
  using ProblemShape        = conditional_t<VarlenQ, ProblemShapeVerlenQ, ProblemShapeFixed>;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock = (NumLoadWarpSquads + NumMmaWarpSquads) * NumThreadsPerWarpSquad;
  static constexpr uint32_t NumMmaThreads      = NumMmaWarpSquads * NumThreadsPerWarpSquad;
  static constexpr uint32_t NumMmaWarps        = NumMmaThreads / NumThreadsPerWarp;

  static constexpr int SmemAlignmentBytes = 256;

  using MainloopPipelineQKLatent = typename CollectiveMainloop::MainloopPipelineQKLatent;
  using PipelineQKLatentParams   = typename CollectiveMainloop::PipelineQKLatentParams;
  using PipelineQKLatentState    = typename CollectiveMainloop::PipelineQKLatentState;

  using MainloopPipelineQKRope = typename CollectiveMainloop::MainloopPipelineQKRope;
  using PipelineQKRopeParams   = typename CollectiveMainloop::PipelineQKRopeParams;
  using PipelineQKRopeState    = typename CollectiveMainloop::PipelineQKRopeState;

  using MainloopPipelinePV = typename CollectiveMainloop::MainloopPipelinePV;
  using PipelinePVParams   = typename CollectiveMainloop::PipelinePVParams;
  using PipelinePVState    = typename CollectiveMainloop::PipelinePVState;

  using NamedBarrier = mutlass::arch::AsyncBarrier;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineQKLatent[MainloopPipelineQKLatent::NumBarriers];
    uint8_t PipelineQKRope[MainloopPipelineQKRope::NumBarriers];
    uint8_t ReuseP[1];
    uint8_t FlipSync[1];
  };

  struct Arguments {
    ProblemShape                           problem_size;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    typename TileScheduler::Arguments      tile_scheduler;
  };

  struct Params {
    ProblemShape                        problem_size;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params      tile_scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace = nullptr) {
    return Params{
        args.problem_size,
        CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.problem_size, args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(args.problem_size, TileShape{}, args.tile_scheduler),
    };
  }

  template <class BatchCoord>
  MUTLASS_DEVICE auto apply_batch_offset(ProblemShape const& problem_shape, BatchCoord const& batch_coord) {
    auto [problem_size, blk_offset] = apply_variable_length_offset(problem_shape, batch_coord);

    return mute::make_tuple(problem_size, blk_offset);
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

    int thread_idx               = threadIdx.x;
    int warp_idx                 = mutlass::canonical_warp_idx_sync();
    int lane_idx                 = thread_idx % NumThreadsPerWarp;
    int warp_squad_idx           = mutlass::canonical_warp_squad_idx();
    int warp_idx_in_warp_squad   = warp_idx % NumWarpsPerWarpSquad;
    int consumer_warp_squad_idx  = warp_squad_idx - static_cast<int>(WarpSquadRole::Consumer0);
    int thread_idx_in_warp_squad = thread_idx % NumThreadsPerWarpSquad;

    auto warp_squad_role = WarpSquadRole(warp_squad_idx);

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineQKLatentParams pipeline_params_qk_latent;
    pipeline_params_qk_latent.transaction_bytes = CollectiveMainloop::TmeTransactionBytesC;
    pipeline_params_qk_latent.num_consumers     = NumMmaWarps;
    pipeline_params_qk_latent.num_producers     = 1;

    MainloopPipelineQKLatent pipeline_qk_latent(pipeline_params_qk_latent,
                                                reinterpret_cast<uint64_t>(&barrier_storage->PipelineQKLatent));
    PipelineQKLatentState    mainloop_pipe_qk_latent_producer_state =
        mutlass::make_producer_start_state<MainloopPipelineQKLatent>();
    PipelineQKLatentState mainloop_pipe_qk_latent_consumer_state;

    PipelineQKRopeParams pipeline_params_qk_rope;
    pipeline_params_qk_rope.transaction_bytes = CollectiveMainloop::TmeTransactionBytesKRope;
    pipeline_params_qk_rope.num_consumers     = NumMmaWarps;
    pipeline_params_qk_rope.num_producers     = 1;

    MainloopPipelineQKRope pipeline_qk_rope(pipeline_params_qk_rope,
                                            reinterpret_cast<uint64_t>(&barrier_storage->PipelineQKRope));
    PipelineQKRopeState    mainloop_pipe_qk_rope_producer_state =
        mutlass::make_producer_start_state<MainloopPipelineQKRope>();
    PipelineQKRopeState mainloop_pipe_qk_rope_consumer_state;

    NamedBarrier reuse_p(reinterpret_cast<uint64_t>(&barrier_storage->ReuseP));
    if (warp_idx == 0) {
      reuse_p.init(NumMmaWarps);
    }

    __syncthreads();

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    if constexpr (UseTrivialScheduler) {
      auto [qo_coord, split_coord, batch_coord] = static_cast<uint3>(blockIdx);
      auto blk_coord                            = make_coord(qo_coord, batch_coord, split_coord);

      struct WorkloadInfo {
        int start_block_idx;
        int end_block_idx;
        int seqlen_k;
      };

      int cur_seqlen = params.mainloop.ptr_seqlen[batch_coord];

      WorkloadInfo workload;
      workload.start_block_idx = 0;
      workload.end_block_idx   = ceil_div(cur_seqlen, get<1>(TileShape{}));
      workload.seqlen_k        = cur_seqlen;

      if (warp_squad_role == WarpSquadRole::Producer) {
        mainloop.load(params.mainloop,
                      shared_storage,
                      pipeline_qk_latent,
                      mainloop_pipe_qk_latent_producer_state,
                      pipeline_qk_rope,
                      mainloop_pipe_qk_rope_producer_state,
                      pipeline_qk_latent,
                      mainloop_pipe_qk_latent_producer_state,
                      blk_coord,
                      params.problem_size,
                      warp_idx,
                      workload,
                      0);
      } else if (warp_squad_role == WarpSquadRole::Consumer0 || warp_squad_role == WarpSquadRole::Consumer1 ||
                 warp_squad_role == WarpSquadRole::Consumer2 || warp_squad_role == WarpSquadRole::Consumer3) {
        auto [Q, PageSize, D, H, PageCount, B, TotalQ, MaxQ] = params.problem_size;
        auto [D_latent, D_rope]                              = D;
        auto problem_size                                    = make_shape(Q, cur_seqlen, D_latent, D_latent, H, 1, B);
        auto results                                         = mainloop.compute(blk_coord,
                                        params.mainloop,
                                        problem_size,
                                        pipeline_qk_latent,
                                        mainloop_pipe_qk_latent_consumer_state,
                                        pipeline_qk_rope,
                                        mainloop_pipe_qk_rope_consumer_state,
                                        pipeline_qk_latent,
                                        mainloop_pipe_qk_latent_consumer_state,
                                        reuse_p,
                                        shared_storage,
                                        thread_idx_in_warp_squad,
                                        consumer_warp_squad_idx,
                                        workload);

        typename CollectiveMainloop::TiledMmaPV tiled_mma_pv;
        auto consumer_qo_coord  = get<0>(blk_coord) * NumMmaWarpSquads + consumer_warp_squad_idx;
        auto problem_size_epi   = make_shape(Q * H, cur_seqlen, H, 1, B, TotalQ, MaxQ);
        auto blk_coord_epilogue = make_coord(qo_coord, _0{}, _0{}, batch_coord);

        int               split_idx = 0;
        epilogue.template operator()<true>(blk_coord_epilogue,
                                           results,
                                           tiled_mma_pv,
                                           problem_size_epi,
                                           0,
                                           params.epilogue,
                                           thread_idx_in_warp_squad,
                                           consumer_qo_coord,
                                           split_idx,
                                           0);
      }
    }
    // FlashMla-style scheduler
    else {
      auto [qo_coord, heads_coord, partition_idx] = static_cast<uint3>(blockIdx);

      int* tile_scheduler_metadata_ptr =
          params.tile_scheduler.metadata_ptr + partition_idx * TileScheduler::TileSchedulerMetaDataSize;
      mute::v4i32_t tile_scheduler_metadata = *reinterpret_cast<mute::v4i32_t*>(tile_scheduler_metadata_ptr);
      int           begin_idx               = tile_scheduler_metadata[0];
      int           sched_begin_block_idx   = tile_scheduler_metadata[1];
      int           end_idx                 = tile_scheduler_metadata[2];
      int           sched_end_block_idx     = tile_scheduler_metadata[3];

      if (begin_idx >= get<5>(params.problem_size)) return;

      int begin_n_split_idx = *(tile_scheduler_metadata_ptr + 4);

      struct WorkloadInfo {
        int start_block_idx;
        int end_block_idx;
        int seqlen_k;
      };

      for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        auto [problem_size, blk_offsets]                     = apply_batch_offset(params.problem_size, batch_idx);
        auto [Q, PageSize, D, H, PageCount, B, TotalQ, MaxQ] = problem_size;
        int seq_q_begin                                      = get<0>(blk_offsets);

        // Compute only if q_idx in range.
        if (qo_coord < ceil_div(Q * H, get<0>(TileShape{}))) {
          int       seqlen_k        = params.mainloop.ptr_seqlen[batch_idx];
          int const start_block_idx = batch_idx == begin_idx ? sched_begin_block_idx : 0;
          int const end_block_idx =
              batch_idx == end_idx ? sched_end_block_idx : ceil_div(seqlen_k, get<1>(TileShape{}));

          // for epilogue
          int const  n_split_idx = batch_idx == begin_idx ? begin_n_split_idx : 0;
          bool const is_no_split = (params.tile_scheduler.num_splits_ptr[batch_idx + 1] -
                                    params.tile_scheduler.num_splits_ptr[batch_idx]) == 1;

          WorkloadInfo workload{start_block_idx, end_block_idx, seqlen_k};

          auto blk_coord = make_coord(qo_coord, batch_idx, 0);

          if (warp_squad_role == WarpSquadRole::Producer) {
            mainloop.load(params.mainloop,
                          shared_storage,
                          pipeline_qk_latent,
                          mainloop_pipe_qk_latent_producer_state,
                          pipeline_qk_rope,
                          mainloop_pipe_qk_rope_producer_state,
                          pipeline_qk_latent,
                          mainloop_pipe_qk_latent_producer_state,
                          blk_coord,
                          problem_size,
                          warp_idx,
                          workload,
                          seq_q_begin);
          } else if (warp_squad_role == WarpSquadRole::Consumer0 || warp_squad_role == WarpSquadRole::Consumer1 ||
                     warp_squad_role == WarpSquadRole::Consumer2 || warp_squad_role == WarpSquadRole::Consumer3) {
            auto [D_latent, D_rope] = D;
            // int cur_seqlen = params.mainloop.ptr_seqlen[batch_coord];
            int cur_seqlen = workload.seqlen_k;

            auto problem_size = make_shape(Q, cur_seqlen, D_latent, D_latent, H, 1, B);

            auto results = mainloop.compute(blk_coord,
                                            params.mainloop,
                                            problem_size,
                                            pipeline_qk_latent,
                                            mainloop_pipe_qk_latent_consumer_state,
                                            pipeline_qk_rope,
                                            mainloop_pipe_qk_rope_consumer_state,
                                            pipeline_qk_latent,
                                            mainloop_pipe_qk_latent_consumer_state,
                                            reuse_p,
                                            shared_storage,
                                            thread_idx_in_warp_squad,
                                            consumer_warp_squad_idx,
                                            workload);

            typename CollectiveMainloop::TiledMmaPV tiled_mma_pv;
            auto consumer_qo_coord  = get<0>(blk_coord) * NumMmaWarpSquads + consumer_warp_squad_idx;
            auto problem_size_epi   = make_shape(Q * H, cur_seqlen, H, 1, B, TotalQ, MaxQ);
            auto blk_coord_epilogue = make_coord(qo_coord, _0{}, _0{}, batch_idx);

            int split_idx = params.tile_scheduler.num_splits_ptr[batch_idx] + n_split_idx;

            if (is_no_split) {
              epilogue.template operator()<true>(blk_coord_epilogue,
                                                 results,
                                                 tiled_mma_pv,
                                                 problem_size_epi,
                                                 0,
                                                 params.epilogue,
                                                 thread_idx_in_warp_squad,
                                                 consumer_qo_coord,
                                                 split_idx,
                                                 seq_q_begin);
            } else {
              epilogue.template operator()<false>(blk_coord_epilogue,
                                                  results,
                                                  tiled_mma_pv,
                                                  problem_size_epi,
                                                  0,
                                                  params.epilogue,
                                                  thread_idx_in_warp_squad,
                                                  consumer_qo_coord,
                                                  split_idx,
                                                  seq_q_begin);
            }
          }
        }

        __syncthreads();
      }
    }
  }
};

}  // namespace mate::flash_mla
