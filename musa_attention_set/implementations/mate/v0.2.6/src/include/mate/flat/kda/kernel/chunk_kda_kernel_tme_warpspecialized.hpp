#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/pipeline/pipeline.hpp>
#include <type_traits>

#include "mate/flat/flat_options.hpp"

namespace mate::flat::kda {

template <class CollectiveMainloop_, class TileScheduler_>
struct ChunkKdaKernel {
  using CollectiveMainloop = CollectiveMainloop_;
  using TileScheduler      = TileScheduler_;
  using CuSeqlensElement   = typename TileScheduler::CuSeqlensElement;
  using SharedStorage      = typename CollectiveMainloop::SharedStorage;
  using BarrierStorage     = typename CollectiveMainloop::BarrierStorage;
  using Pipeline           = typename CollectiveMainloop::Pipeline;

  struct ProblemSize {
    int                     B;
    int                     T;
    int                     H;
    int                     Hqk;
    int                     N;
    CuSeqlensElement const* cu_seqlens = nullptr;
  };

  static constexpr int      SharedStorageSize          = CollectiveMainloop::SharedStorageSize;
  static constexpr int      SmemAlignmentBytes         = CollectiveMainloop::SmemAlignmentBytes;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int      MaxThreadsPerBlock         = CollectiveMainloop::NumProducerThreads +
                                            CollectiveMainloop::NumStateThreads + CollectiveMainloop::NumOutputThreads;

  struct Arguments {
    ProblemSize                            problem_size;
    typename CollectiveMainloop::Arguments mainloop;
  };

  struct Params {
    ProblemSize                         problem_size;
    typename CollectiveMainloop::Params mainloop;
    typename TileScheduler::Params      scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return Params{
        args.problem_size,
        CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop),
        TileScheduler::to_underlying_arguments(args.problem_size),
    };
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    enum class WarpSpecRole {
      Producer = 0,
      State0   = 1,
      State1   = 2,
      Output   = 3,
    };

    int  thread_idx    = int(threadIdx.x);
    int  warp_spec_idx = mutlass::canonical_warp_squad_idx();
    auto warp_role     = WarpSpecRole(warp_spec_idx);

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    TileScheduler tile_scheduler(params.scheduler);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);
    CollectiveMainloop::init_stage_barriers(barrier_storage, int(threadIdx.x));
    Pipeline pipeline(barrier_storage);

    __syncthreads();

    for (auto work_desc = tile_scheduler.get_next_work(params.problem_size); work_desc.is_valid(tile_scheduler);
         work_desc      = tile_scheduler.get_next_work(params.problem_size)) {
      if (warp_role == WarpSpecRole::Producer) {
        CollectiveMainloop::load(shared_storage, params.mainloop, params.problem_size, work_desc, pipeline, thread_idx);
      } else if (warp_role == WarpSpecRole::State0 || warp_role == WarpSpecRole::State1) {
        CollectiveMainloop::compute_state(shared_storage,
                                          params.mainloop,
                                          params.problem_size,
                                          work_desc,
                                          pipeline,
                                          thread_idx - CollectiveMainloop::NumProducerThreads);
      } else if (warp_role == WarpSpecRole::Output) {
        CollectiveMainloop::compute_output(
            shared_storage,
            params.mainloop,
            params.problem_size,
            work_desc,
            pipeline,
            thread_idx - CollectiveMainloop::NumProducerThreads - CollectiveMainloop::NumStateThreads);
      }
    }
  }
};

}  // namespace mate::flat::kda
