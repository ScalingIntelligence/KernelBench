#pragma once

#include <mute/tensor.hpp>
#include <mutlass/kernel_hardware_info.hpp>

namespace mate::attention::msa {

template <class CollectiveMainloop_, class TileScheduler_>
struct MsaMaxScoreKernelTmeWarpSpecialized {
  using CollectiveMainloop = CollectiveMainloop_;
  using TileScheduler      = TileScheduler_;

  struct ProblemSize {
    int batch_size;
    int total_q;
    int total_k;
    int num_qo_heads;
    int num_kv_heads;
    int max_seqlen_q;
    int max_seqlen_k;
    int max_k_tiles;
  };

  using SharedStorage                                  = typename CollectiveMainloop::SharedStorage;
  using BarrierStorage                                 = typename CollectiveMainloop::BarrierStorage;
  static constexpr int      SmemAlignmentBytes         = CollectiveMainloop::SmemAlignmentBytes;
  static constexpr int      SharedStorageSize          = sizeof(SharedStorage);
  static constexpr int      MaxThreadsPerBlock         = CollectiveMainloop::NumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  struct Arguments {
    ProblemSize                            problem_size;
    typename CollectiveMainloop::Arguments mainloop;
    typename TileScheduler::Arguments      scheduler;
    mutlass::KernelHardwareInfo            hw_info{};
  };

  struct Params {
    ProblemSize                         problem_size;
    typename CollectiveMainloop::Params mainloop;
    typename TileScheduler::Params      scheduler;
    mutlass::KernelHardwareInfo         hw_info{};
  };

  static Params to_underlying_arguments(Arguments const& args) {
    int mp_count = args.hw_info.sm_count;
    if (mp_count <= 0) {
      mp_count = mutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    return {
        args.problem_size,
        CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop),
        TileScheduler::to_underlying_arguments(args.problem_size, args.scheduler, mp_count),
        mutlass::KernelHardwareInfo{args.hw_info.device_id, mp_count},
    };
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    int thread_idx             = int(threadIdx.x);
    int warp_idx               = mutlass::canonical_warp_idx_sync();
    int warp_squad_idx         = mutlass::canonical_warp_squad_idx();
    int warp_idx_in_warp_squad = warp_idx % mutlass::NumWarpsPerWarpSquad;
    int consumer_thread_idx    = thread_idx - CollectiveMainloop::NumProducerThreads;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    CollectiveMainloop                    mainloop;
    TileScheduler                         scheduler;
    typename CollectiveMainloop::Pipeline pipeline(barrier_storage);
    __syncthreads();

    if (warp_squad_idx == 0) {
      int lane_idx = thread_idx % mutlass::NumThreadsPerWarp;
      for (auto q_work_tile = scheduler.get_initial_work(params.scheduler); q_work_tile.is_valid(params.scheduler);
           q_work_tile      = scheduler.get_next_work(params.scheduler, q_work_tile)) {
        __musa_loop_transparent_outermost();
        if (!TileScheduler::is_valid_q_tile(q_work_tile)) {
          continue;
        }
        int k_tile_count = mainloop.k_tile_count(params.mainloop, q_work_tile);
        if (TileScheduler::ParallelKTiles && q_work_tile.k_partition_idx >= k_tile_count) {
          continue;
        }
        if (lane_idx == 0 && warp_idx_in_warp_squad == CollectiveMainloop::QLoadWarpInProducer) {
          mainloop.load_q(params.mainloop, pipeline, shared_storage, q_work_tile);
        }
        if constexpr (TileScheduler::ParallelKTiles) {
          MUTLASS_PRAGMA_NO_UNROLL
          for (int k_tile_idx = q_work_tile.k_partition_idx; k_tile_idx < k_tile_count;
               k_tile_idx += params.scheduler.k_partitions) {
            auto k_work_tile = TileScheduler::make_k_work_tile(q_work_tile, k_tile_idx);
            if (lane_idx == 0 && warp_idx_in_warp_squad == CollectiveMainloop::KLoadWarpInProducer) {
              mainloop.load_k(params.mainloop, pipeline, shared_storage, k_work_tile);
              if constexpr (CollectiveMainloop::EnableKPrefetch) {
                int next_k_tile_idx = k_tile_idx + params.scheduler.k_partitions;
                if (next_k_tile_idx < k_tile_count) {
                  auto next_k_work_tile = TileScheduler::make_k_work_tile(q_work_tile, next_k_tile_idx);
                  mainloop.prefetch_k(params.mainloop, next_k_work_tile);
                }
              }
            }
          }
        } else {
          MUTLASS_PRAGMA_NO_UNROLL
          for (int k_tile_idx = 0; k_tile_idx < k_tile_count; ++k_tile_idx) {
            auto k_work_tile = TileScheduler::make_k_work_tile(q_work_tile, k_tile_idx);
            if (lane_idx == 0 && warp_idx_in_warp_squad == CollectiveMainloop::KLoadWarpInProducer) {
              mainloop.load_k(params.mainloop, pipeline, shared_storage, k_work_tile);
              if constexpr (CollectiveMainloop::EnableKPrefetch) {
                int next_k_tile_idx = k_tile_idx + 1;
                if (next_k_tile_idx < k_tile_count) {
                  auto next_k_work_tile = TileScheduler::make_k_work_tile(q_work_tile, next_k_tile_idx);
                  mainloop.prefetch_k(params.mainloop, next_k_work_tile);
                }
              }
            }
          }
        }
      }
    } else if (warp_squad_idx <= CollectiveMainloop::NumMmaWarpSquads) {
      for (auto q_work_tile = scheduler.get_initial_work(params.scheduler); q_work_tile.is_valid(params.scheduler);
           q_work_tile      = scheduler.get_next_work(params.scheduler, q_work_tile)) {
        __musa_loop_transparent_outermost();
        if (!TileScheduler::is_valid_q_tile(q_work_tile)) {
          continue;
        }
        int k_tile_count = mainloop.k_tile_count(params.mainloop, q_work_tile);
        if (TileScheduler::ParallelKTiles && q_work_tile.k_partition_idx >= k_tile_count) {
          continue;
        }
        mainloop.wait_q(pipeline, consumer_thread_idx);
        if constexpr (TileScheduler::ParallelKTiles) {
          MUTLASS_PRAGMA_NO_UNROLL
          for (int k_tile_idx = q_work_tile.k_partition_idx; k_tile_idx < k_tile_count;
               k_tile_idx += params.scheduler.k_partitions) {
            auto k_work_tile = TileScheduler::make_k_work_tile(q_work_tile, k_tile_idx);
            mainloop.compute_k_tile(params.mainloop, pipeline, shared_storage, k_work_tile, consumer_thread_idx);
          }
        } else {
          MUTLASS_PRAGMA_NO_UNROLL
          for (int k_tile_idx = 0; k_tile_idx < k_tile_count; ++k_tile_idx) {
            auto k_work_tile = TileScheduler::make_k_work_tile(q_work_tile, k_tile_idx);
            mainloop.compute_k_tile(params.mainloop, pipeline, shared_storage, k_work_tile, consumer_thread_idx);
          }
        }
        mainloop.release_q(pipeline, consumer_thread_idx);
      }
    }
  }
};

}  // namespace mate::attention::msa
