#pragma once

#include <mute/tensor.hpp>
#include <mutlass/kernel_hardware_info.hpp>

namespace mate::attention::msa {

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
struct MsaFwdKernelTmeWarpSpecialized {
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler      = TileScheduler_;

  // Binding ABI. total_k is the physical paged-KV token capacity
  // (num_pages * 128), while seqused_k carries each sequence's logical KV
  // length and qo_offset aligns its query positions for causal masking.
  struct ProblemSize {
    int batch_size;
    int total_q;
    int total_k;
    int num_qo_heads;
    int num_kv_heads;
    int max_seqlen_q;
    int max_seqlen_k;
  };

  using SharedStorage                                  = typename CollectiveMainloop::SharedStorage;
  using BarrierStorage                                 = typename CollectiveMainloop::BarrierStorage;
  static constexpr int      SmemAlignmentBytes         = CollectiveMainloop::SmemAlignmentBytes;
  static constexpr int      SharedStorageSize          = sizeof(SharedStorage);
  static constexpr int      MaxThreadsPerBlock         = CollectiveMainloop::NumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static_assert(sizeof(BarrierStorage) + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount <=
                    mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "MSA forward async barrier id exceeds the MP31 hardware limit.");

  struct Arguments {
    ProblemSize                            problem_size;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    typename TileScheduler::Arguments      scheduler;
    mutlass::KernelHardwareInfo            hw_info{};
  };

  struct Params {
    ProblemSize                         problem_size;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
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
        CollectiveEpilogue::to_underlying_arguments(args.problem_size, args.epilogue),
        TileScheduler::to_underlying_arguments(args.problem_size, args.scheduler),
        mutlass::KernelHardwareInfo{args.hw_info.device_id, mp_count},
    };
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.problem_size, args.mainloop) &&
           CollectiveEpilogue::can_implement(args.problem_size, args.epilogue) &&
           TileScheduler::can_implement(args.problem_size, args.scheduler);
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
    int lane_idx               = thread_idx % mutlass::NumThreadsPerWarp;
    int consumer_thread_idx    = thread_idx - CollectiveMainloop::NumProducerThreads;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage) +
                                           mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount);
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    CollectiveMainloop                    mainloop;
    CollectiveEpilogue                    epilogue;
    TileScheduler                         scheduler;
    typename CollectiveMainloop::Pipeline pipeline(barrier_storage);
    __syncthreads();

    if (warp_squad_idx < CollectiveMainloop::NumProducerWarpSquads) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (auto work_tile = scheduler.get_initial_work(params.scheduler); work_tile.is_valid(params.scheduler);
           work_tile      = scheduler.get_next_work(params.scheduler, work_tile)) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        if (!TileScheduler::is_valid_query(work_tile)) {
          continue;
        }

        if (lane_idx == 0 && warp_idx_in_warp_squad == CollectiveMainloop::QLoadWarpInProducer) {
          mainloop.load_q(params.mainloop, pipeline, shared_storage, work_tile);
        }

        if (lane_idx == 0 && warp_idx_in_warp_squad == CollectiveMainloop::KVLoadWarpInProducer) {
          mainloop.load_kv(params.mainloop, pipeline, shared_storage, work_tile);
        }
      }
    } else if (warp_squad_idx < CollectiveMainloop::NumProducerWarpSquads + CollectiveMainloop::NumMmaWarpSquads) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (auto work_tile = scheduler.get_initial_work(params.scheduler); work_tile.is_valid(params.scheduler);
           work_tile      = scheduler.get_next_work(params.scheduler, work_tile)) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        if (!TileScheduler::is_valid_query(work_tile)) {
          continue;
        }

        auto  result = mainloop.compute(params.mainloop, pipeline, shared_storage, work_tile, consumer_thread_idx);
        auto& acc_pv = mute::get<0>(result);
        auto& lse    = mute::get<1>(result);
        if (consumer_thread_idx < int(mute::size(typename CollectiveMainloop::TiledMmaPV{}))) {
          epilogue.store(
              params.epilogue, acc_pv, lse, typename CollectiveMainloop::TiledMmaPV{}, work_tile, consumer_thread_idx);
        }
      }
    }
  }
};

}  // namespace mate::attention::msa
