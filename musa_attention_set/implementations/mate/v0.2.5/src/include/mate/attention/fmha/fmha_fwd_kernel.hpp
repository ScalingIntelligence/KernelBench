#pragma once

#include <mute/tensor.hpp>
#include <mutlass/kernel_hardware_info.hpp>
#include <mutlass/pipeline/pipeline.hpp>

#include "named_barrier.hpp"
#include "pipeline_ws.hpp"

namespace mate::attention::fmha {

using namespace mute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
struct FmhaFwdKernelWarpSpecialized {
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler      = TileScheduler_;

  static constexpr bool IsAppendKV = CollectiveMainloop::IsAppendKV;
  static constexpr bool HasQv      = CollectiveMainloop::HasQv;
  static constexpr bool Split      = CollectiveMainloop::Split;

  static constexpr int NumLoadWarpSquads = CollectiveMainloop::NumLoadWarpSquads;
  static constexpr int NumMmaWarpSquads  = CollectiveMainloop::NumMmaWarpSquads;

  static constexpr int NumQKMmaWarps = CollectiveMainloop::NumQKConsumers * mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumPVMmaWarps = CollectiveMainloop::NumPVConsumers * mutlass::NumWarpsPerWarpSquad;

  static constexpr bool     InKernelTranspose  = CollectiveMainloop::InKernelTranspose;
  static constexpr uint32_t NumTransWarpSquads = InKernelTranspose ? 1 : 0;
  static constexpr uint32_t NumTransWarps      = NumTransWarpSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr uint32_t TransWarpSquadIdx  = NumLoadWarpSquads + NumMmaWarpSquads;

  static constexpr int      SmemAlignmentBytes         = 256;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock =
      (NumLoadWarpSquads + NumMmaWarpSquads + NumTransWarpSquads) * mutlass::NumThreadsPerWarpSquad;

  using SharedStorage                    = typename CollectiveMainloop::SharedStorage;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static constexpr bool UseLSULoadQ        = CollectiveMainloop::UseLSULoadQ;
  static constexpr bool SingleProducerWarp = CollectiveMainloop::NumProducerThreads == mutlass::NumThreadsPerWarp;

  static constexpr int NumLoadWarps = !SingleProducerWarp ? NumLoadWarpSquads * mutlass::NumWarpsPerWarpSquad : 1;

  static constexpr bool UseLSULoadK = CollectiveMainloop::UseLSULoadK;
  static constexpr bool UseLSULoadV = CollectiveMainloop::UseLSULoadV;

  using MainloopPipelineQ           = typename CollectiveMainloop::MainloopPipelineQ;
  using MainloopPipelineQv          = typename CollectiveMainloop::MainloopPipelineQv;
  using MainloopPipelineK           = typename CollectiveMainloop::MainloopPipelineK;
  using MainloopPipelineV           = typename CollectiveMainloop::MainloopPipelineV;
  using MainloopPipelineVt          = typename CollectiveMainloop::MainloopPipelineVt;
  using MainloopPipelineKVNew       = typename CollectiveMainloop::MainloopPipelineKVNew;
  using MainloopPipelineQState      = typename CollectiveMainloop::PipelineQState;
  using MainloopPipelineQvState     = typename CollectiveMainloop::PipelineQvState;
  using MainloopPipelineKState      = typename CollectiveMainloop::PipelineKState;
  using MainloopPipelineVState      = typename CollectiveMainloop::PipelineVState;
  using MainloopPipelineVtState     = typename CollectiveMainloop::PipelineVtState;
  using MainloopPipelineKVNewState  = typename CollectiveMainloop::PipelineKVNewState;
  using MainloopPipelineQParams     = typename MainloopPipelineQ::Params;
  using MainloopPipelineQvParams    = typename MainloopPipelineQv::Params;
  using MainloopPipelineKParams     = typename MainloopPipelineK::Params;
  using MainloopPipelineVParams     = typename MainloopPipelineV::Params;
  using MainloopPipelineVtParams    = typename MainloopPipelineVt::Params;
  using MainloopPipelineKVNewParams = typename MainloopPipelineKVNew::Params;

  static constexpr int StagesQ  = CollectiveMainloop::StagesQ;
  static constexpr int StagesK  = CollectiveMainloop::StagesK;
  static constexpr int StagesV  = CollectiveMainloop::StagesV;
  static constexpr int StagesVt = CollectiveMainloop::StagesVt;

  using SeqlenInfo = typename CollectiveMainloop::SeqlenInfo;

  using NamedBarrier = mutlass::arch::AsyncBarrier;
  using Element      = typename CollectiveMainloop::Element;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t NamedBarriers[static_cast<int32_t>(FwdNamedBarriers::NumFwdNamedBarriers)];
    uint8_t PipelineQ[MainloopPipelineQ::NumBarriers];
    uint8_t PipelineQv[MainloopPipelineQv::NumBarriers];
    uint8_t PipelineK[MainloopPipelineK::NumBarriers];
    uint8_t PipelineV[MainloopPipelineV::NumBarriers];
    uint8_t PipelineVt[MainloopPipelineVt::NumBarriers];
    uint8_t PipelineKNew[MainloopPipelineKVNew::NumBarriers];
    uint8_t PipelineVNew[MainloopPipelineKVNew::NumBarriers];
  };
  static_assert(sizeof(BarrierStorage) + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount <=
                    mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "FMHA async barrier storage exceeds hardware limit");

  struct Arguments {
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    typename TileScheduler::Arguments      scheduler;
    mutlass::KernelHardwareInfo            hw_info{};
  };

  struct Params {
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params      scheduler;
    mutlass::KernelHardwareInfo         hw_info{};
  };

  static Params to_underlying_arguments(const Arguments& args) {
    MUTLASS_TRACE_HOST("to_underlying_arguments():");

    int mp_count = args.hw_info.sm_count;
    if (mp_count <= 0) {
      MUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid MP count.\n"
          " For optimal performance, popluate the arguments KernelHardwareInfo struct with the MP count.");
      mp_count = mutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    mutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, mp_count};

    return {CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            TileScheduler::to_underlying_arguments(args.scheduler),
            hw_info};
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.mp_count);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  MUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    int            thread_idx               = threadIdx.x;
    int            warp_idx                 = mutlass::canonical_warp_idx();
    int            lane_idx                 = thread_idx % mutlass::NumThreadsPerWarp;
    int            warp_squad_idx           = mutlass::canonical_warp_squad_idx();
    int            warp_idx_in_warp_squad   = warp_idx % mutlass::NumWarpsPerWarpSquad;
    int            consumer_warp_squad_idx  = warp_squad_idx - 1;
    int            thread_idx_in_warp_squad = thread_idx % mutlass::NumThreadsPerWarpSquad;
    SharedStorage& shared_storage           = *reinterpret_cast<SharedStorage*>(smem_buf);

    static constexpr int MmaThreadOffset = NumLoadWarpSquads * mutlass::NumThreadsPerWarpSquad;

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    MainloopPipelineQParams pipeline_params_q;
    if constexpr (UseLSULoadQ) {
      pipeline_params_q.producer_arv_count = CollectiveMainloop::NumProducerThreads / mutlass::NumThreadsPerWarp;
      pipeline_params_q.consumer_arv_count = NumQKMmaWarps;
    } else {
      pipeline_params_q.transaction_bytes = CollectiveMainloop::TmeTransactionBytesQ;
      pipeline_params_q.num_consumers     = NumQKMmaWarps;
    }
    MainloopPipelineQ      pipeline_q(pipeline_params_q, reinterpret_cast<uint64_t>(&barrier_storage->PipelineQ));
    MainloopPipelineQState mainloop_pipe_q_producer_state =
        mutlass::make_producer_start_state_warpspecialized<MainloopPipelineQ>();
    MainloopPipelineQState mainloop_pipe_q_consumer_state;

    MainloopPipelineQvParams pipeline_params_qv;
    if constexpr (UseLSULoadQ) {
      pipeline_params_qv.producer_arv_count = CollectiveMainloop::NumProducerThreads / mutlass::NumThreadsPerWarp;
      pipeline_params_qv.consumer_arv_count = NumQKMmaWarps;
    } else {
      pipeline_params_qv.transaction_bytes = CollectiveMainloop::TmeTransactionBytesQv;
      pipeline_params_qv.num_consumers     = NumQKMmaWarps;
    }
    auto pipeline_qv = conditional_return<HasQv>(
        MainloopPipelineQv(pipeline_params_qv, reinterpret_cast<uint64_t>(&barrier_storage->PipelineQv)), nullptr);
    MainloopPipelineQvState mainloop_pipe_qv_producer_state =
        mutlass::make_producer_start_state_warpspecialized<MainloopPipelineQv>();
    MainloopPipelineQvState mainloop_pipe_qv_consumer_state;

    MainloopPipelineKParams pipeline_params_k;
    if constexpr (UseLSULoadK) {
      pipeline_params_k.producer_arv_count = CollectiveMainloop::NumProducerThreads / mutlass::NumThreadsPerWarp;
      pipeline_params_k.consumer_arv_count = NumQKMmaWarps;
    } else {
      pipeline_params_k.transaction_bytes = CollectiveMainloop::TmeTransactionBytesK;
      pipeline_params_k.num_consumers     = NumQKMmaWarps;
    }
    MainloopPipelineK      pipeline_k(pipeline_params_k, reinterpret_cast<uint64_t>(&barrier_storage->PipelineK));
    MainloopPipelineKState mainloop_pipe_k_producer_state =
        mutlass::make_producer_start_state_warpspecialized<MainloopPipelineK>();
    MainloopPipelineKState mainloop_pipe_k_consumer_state;

    MainloopPipelineVParams pipeline_params_v;
    if constexpr (UseLSULoadV) {
      pipeline_params_v.producer_arv_count = CollectiveMainloop::NumProducerThreads / mutlass::NumThreadsPerWarp;
      pipeline_params_v.consumer_arv_count = InKernelTranspose ? NumQKMmaWarps + NumTransWarps : NumPVMmaWarps;
    } else {
      pipeline_params_v.transaction_bytes = CollectiveMainloop::TmeTransactionBytesV;
      pipeline_params_v.num_consumers     = InKernelTranspose ? NumQKMmaWarps + NumTransWarps : NumPVMmaWarps;
    }
    MainloopPipelineV      pipeline_v(pipeline_params_v, reinterpret_cast<uint64_t>(&barrier_storage->PipelineV));
    MainloopPipelineVState mainloop_pipe_v_producer_state =
        mutlass::make_producer_start_state_warpspecialized<MainloopPipelineV>();
    MainloopPipelineVState mainloop_pipe_v_consumer_state;

    MainloopPipelineVtParams pipeline_params_vt;
    pipeline_params_vt.producer_arv_count = NumTransWarps;
    pipeline_params_vt.consumer_arv_count = NumPVMmaWarps;
    auto pipeline_vt                      = conditional_return<InKernelTranspose>(
        MainloopPipelineVt(pipeline_params_vt, reinterpret_cast<uint64_t>(&barrier_storage->PipelineVt)), nullptr);
    MainloopPipelineVtState mainloop_pipe_vt_producer_state =
        mutlass::make_producer_start_state_warpspecialized<MainloopPipelineVt>();
    MainloopPipelineVtState mainloop_pipe_vt_consumer_state;

    MainloopPipelineKVNewParams pipeline_params_k_new;
    pipeline_params_k_new.transaction_bytes = CollectiveMainloop::TmeTransactionBytesK;
    pipeline_params_k_new.num_consumers     = NumQKMmaWarps;
    uint64_t pipeline_k_new_storage         = reinterpret_cast<uint64_t>(&barrier_storage->PipelineKNew);
    auto     pipeline_k_new =
        conditional_return<IsAppendKV>(MainloopPipelineKVNew(pipeline_params_k_new, pipeline_k_new_storage), nullptr);
    MainloopPipelineKVNewState mainloop_pipe_kv_new_producer_state =
        mutlass::make_producer_start_state_warpspecialized<MainloopPipelineKVNew>();
    MainloopPipelineKVNewState mainloop_pipe_kv_new_consumer_state;

    MainloopPipelineKVNewParams pipeline_params_v_new;
    pipeline_params_v_new.transaction_bytes = CollectiveMainloop::TmeTransactionBytesV;
    pipeline_params_v_new.num_consumers     = NumPVMmaWarps;
    uint64_t pipeline_v_new_storage         = reinterpret_cast<uint64_t>(&barrier_storage->PipelineVNew);
    auto     pipeline_v_new =
        conditional_return<IsAppendKV>(MainloopPipelineKVNew(pipeline_params_v_new, pipeline_v_new_storage), nullptr);

    NamedBarrier barrier_kv(
        reinterpret_cast<uint64_t>(&barrier_storage->NamedBarriers[static_cast<int32_t>(FwdNamedBarriers::BarrierKV)]));
    NamedBarrier appendkv(
        reinterpret_cast<uint64_t>(&barrier_storage->NamedBarriers[static_cast<int32_t>(FwdNamedBarriers::AppendKV)]));
    NamedBarrier rotary_q(
        reinterpret_cast<uint64_t>(&barrier_storage->NamedBarriers[static_cast<int32_t>(FwdNamedBarriers::RotaryQ)]));
    if (warp_idx == 0) {
      if constexpr (IsAppendKV) {
        barrier_kv.init(NumLoadWarps + NumQKMmaWarps);
      } else {
        barrier_kv.init(NumQKMmaWarps);
      }
      appendkv.init(NumLoadWarps + NumQKMmaWarps);  // QK Warps == PV Warps
      rotary_q.init(NumQKMmaWarps);                 // QK Warps == PV Warps
    }

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;
    TileScheduler      scheduler;

    __syncthreads();

    int work_idx = 0;
    if (warp_squad_idx == 0) {
      // Producer
      if constexpr (SingleProducerWarp) {
        if (warp_idx_in_warp_squad != 0) {
          return;
        }
      }

      MUTLASS_PRAGMA_NO_UNROLL
      for (auto work_tile_info = scheduler.get_initial_work(params.scheduler);
           work_tile_info.is_valid(params.scheduler);
           work_tile_info = scheduler.get_next_work(params.scheduler, work_tile_info)) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        auto [block_idx, head_idx, batch_idx, split_idx] = work_tile_info.get_block_coord(params.scheduler);

        int num_splits = 1;  // Number of splits for the batch of current work_tile.

        if constexpr (scheduler.HasMetadata) {
          num_splits       = params.scheduler.num_splits_dynamic_ptr[batch_idx];
          int num_m_blocks = params.scheduler.num_m_blocks_ptr[batch_idx];
          if (split_idx >= num_splits || block_idx >= num_m_blocks) {
            continue;
          }
          batch_idx = params.scheduler.batch_table_ptr[batch_idx];
        }

        auto block_coord = make_shape(block_idx, head_idx, batch_idx, split_idx);

        // if (threadIdx.x == 0) {
        //   printf("[MP %02d], tile %2d, bidb %d bidm %d bidh %d bids %d\n",
        //          blockIdx.x,
        //          work_tile_info.tile_idx,
        //          batch_idx,
        //          block_idx,
        //          head_idx,
        //          split_idx);
        // }

        SeqlenInfo seqlen_info{
            static_cast<uint32_t>(get<2>(block_coord)) /* batch_idx */,
            static_cast<uint32_t>(get<0>(params.mainloop.shape_Q)) /* seqlen_q_static */,
            static_cast<uint32_t>(!CollectiveMainloop::IsPagedKV /* seqlen_k_static */
                                      ? size<0>(params.mainloop.shape_K)
                                      : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable)),
            static_cast<uint32_t>(get<0>(params.mainloop.shape_K_new)) /* shape_K_new_0 */,
            params.mainloop.cu_seqlens_q /* cu_seqlens_q */,
            params.mainloop.cu_seqlens_k /* cu_seqlens_k */,
            params.mainloop.cu_seqlens_k_new /* cu_seqlens_k_new */,
            params.mainloop.seqused_q /* seqused_q */,
            params.mainloop.seqused_k /* seqused_k */,
            params.mainloop.leftpad_k /* ptr_leftpad_k */,
            params.mainloop.seqlens_rotary /* seqlens_rotary */,
            params.mainloop.cp_world_size /* cp_world_size */,
            params.mainloop.cp_rank /* cp_rank */,
            params.mainloop.cp_tot_seqused_k /* cp_tot_seqused_k */};

        if constexpr (IsAppendKV) {
          barrier_kv.sync();
          bool is_valid_new = mainloop.load_kv_new(params.mainloop,
                                                   pipeline_k_new,
                                                   pipeline_v_new,
                                                   mainloop_pipe_kv_new_producer_state,
                                                   shared_storage,
                                                   seqlen_info,
                                                   block_coord,
                                                   warp_idx_in_warp_squad,
                                                   work_idx,
                                                   num_splits);
          if (is_valid_new) {
            named_barrier_sync(static_cast<uint32_t>(FwdNamedBarriers::AppendKV));
          }
        }

        mainloop.load(params.mainloop,
                      pipeline_q,
                      pipeline_qv,
                      pipeline_k,
                      pipeline_v,
                      mainloop_pipe_q_producer_state,
                      mainloop_pipe_qv_producer_state,
                      mainloop_pipe_k_producer_state,
                      mainloop_pipe_v_producer_state,
                      shared_storage,
                      barrier_storage,
                      seqlen_info,
                      block_coord,
                      work_idx,
                      num_splits);
      }
    } else if (warp_squad_idx == TransWarpSquadIdx) {
      if constexpr (InKernelTranspose) {
        MUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work(params.scheduler);
             work_tile_info.is_valid(params.scheduler);
             work_tile_info = scheduler.get_next_work(params.scheduler, work_tile_info)) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
          __musa_loop_transparent_outermost();
#endif
          auto [block_idx, head_idx, batch_idx, split_idx] = work_tile_info.get_block_coord(params.scheduler);

          int num_splits = 1;

          if constexpr (scheduler.HasMetadata) {
            num_splits       = params.scheduler.num_splits_dynamic_ptr[batch_idx];
            int num_m_blocks = params.scheduler.num_m_blocks_ptr[batch_idx];
            if (split_idx >= num_splits || block_idx >= num_m_blocks) {
              continue;
            }
            batch_idx = params.scheduler.batch_table_ptr[batch_idx];
          }

          auto block_coord = make_shape(block_idx, head_idx, batch_idx, split_idx);

          SeqlenInfo seqlen_info{
              static_cast<uint32_t>(get<2>(block_coord)) /* batch_idx */,
              static_cast<uint32_t>(get<0>(params.mainloop.shape_Q)) /* seqlen_q_static */,
              static_cast<uint32_t>(!CollectiveMainloop::IsPagedKV /* seqlen_k_static */
                                        ? size<0>(params.mainloop.shape_K)
                                        : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable)),
              static_cast<uint32_t>(get<0>(params.mainloop.shape_K_new)) /* shape_K_new_0 */,
              params.mainloop.cu_seqlens_q /* cu_seqlens_q */,
              params.mainloop.cu_seqlens_k /* cu_seqlens_k */,
              params.mainloop.cu_seqlens_k_new /* cu_seqlens_k_new */,
              params.mainloop.seqused_q /* seqused_q */,
              params.mainloop.seqused_k /* seqused_k */,
              params.mainloop.leftpad_k /* ptr_leftpad_k */,
              params.mainloop.seqlens_rotary /* seqlens_rotary */,
              params.mainloop.cp_world_size /* cp_world_size */,
              params.mainloop.cp_rank /* cp_rank */,
              params.mainloop.cp_tot_seqused_k /* cp_tot_seqused_k */};

          mainloop.transpose(block_coord,
                             params.mainloop,
                             pipeline_v,
                             mainloop_pipe_v_consumer_state,
                             pipeline_vt,
                             mainloop_pipe_vt_producer_state,
                             shared_storage,
                             seqlen_info,
                             thread_idx_in_warp_squad,
                             num_splits);
        }
      }
    } else {
      // Consumer
      MUTLASS_PRAGMA_NO_UNROLL
      for (auto work_tile_info = scheduler.get_initial_work(params.scheduler);
           work_tile_info.is_valid(params.scheduler);
           work_tile_info = scheduler.get_next_work(params.scheduler, work_tile_info)) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        auto [block_idx, head_idx, batch_idx, split_idx] = work_tile_info.get_block_coord(params.scheduler);

        int num_splits = 1;  // Number of splits for the batch of current work_tile.

        if constexpr (scheduler.HasMetadata) {
          num_splits       = params.scheduler.num_splits_dynamic_ptr[batch_idx];
          int num_m_blocks = params.scheduler.num_m_blocks_ptr[batch_idx];
          if (split_idx >= num_splits || block_idx >= num_m_blocks) {
            continue;
          }
          batch_idx = params.scheduler.batch_table_ptr[batch_idx];
        }
        auto block_coord = make_shape(block_idx, head_idx, batch_idx, split_idx);

        // if (threadIdx.x == 128) {
        //   printf("MP[%2d], tile %d, bidm=%d, bidb=%d, bidh=%d, bids=%d, num_splits=%d\n",
        //          blockIdx.x,
        //          work_tile_info.tile_idx,
        //          block_idx,
        //          batch_idx,
        //          head_idx,
        //          split_idx,
        //          num_splits);
        // }

        SeqlenInfo seqlen_info{
            static_cast<uint32_t>(get<2>(block_coord)) /* batch_idx */,
            static_cast<uint32_t>(get<0>(params.mainloop.shape_Q)) /* seqlen_q_static */,
            static_cast<uint32_t>(!CollectiveMainloop::IsPagedKV /* seqlen_k_static */
                                      ? size<0>(params.mainloop.shape_K)
                                      : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable)),
            static_cast<uint32_t>(get<0>(params.mainloop.shape_K_new)) /* shape_K_new_0 */,
            params.mainloop.cu_seqlens_q /* cu_seqlens_q */,
            params.mainloop.cu_seqlens_k /* cu_seqlens_k */,
            params.mainloop.cu_seqlens_k_new /* cu_seqlens_k_new */,
            params.mainloop.seqused_q /* seqused_q */,
            params.mainloop.seqused_k /* seqused_k */,
            params.mainloop.leftpad_k /* ptr_leftpad_k */,
            params.mainloop.seqlens_rotary /* seqlens_rotary */,
            params.mainloop.cp_world_size /* cp_world_size */,
            params.mainloop.cp_rank /* cp_rank */,
            params.mainloop.cp_tot_seqused_k /* cp_tot_seqused_k */};

        if constexpr (IsAppendKV) {
          barrier_kv.sync();
          bool is_valid_new = mainloop.store_kv_new(params.mainloop,
                                                    pipeline_k_new,
                                                    pipeline_v_new,
                                                    mainloop_pipe_kv_new_consumer_state,
                                                    threadIdx.x - MmaThreadOffset,
                                                    shared_storage,
                                                    seqlen_info,
                                                    block_coord,
                                                    num_splits);
          if (is_valid_new) {
            __threadfence();
            named_barrier_arrive(static_cast<uint32_t>(FwdNamedBarriers::AppendKV));
          }
        }

        auto results = mainloop.mma(params.mainloop,
                                    pipeline_q,
                                    pipeline_qv,
                                    pipeline_k,
                                    pipeline_v,
                                    pipeline_vt,
                                    mainloop_pipe_q_consumer_state,
                                    mainloop_pipe_qv_consumer_state,
                                    mainloop_pipe_k_consumer_state,
                                    mainloop_pipe_v_consumer_state,
                                    mainloop_pipe_vt_consumer_state,
                                    shared_storage,
                                    barrier_storage,
                                    seqlen_info,
                                    block_coord,
                                    threadIdx.x - MmaThreadOffset,
                                    work_idx,
                                    num_splits);

        // In situations where input tiles are valid but there is no valid input data (e.g., due to causal)
        // we directly write 0 to output and -inf to lse, as they are still needed in combine phase.
        auto  is_valid = get<0>(results);
        auto  values   = get<1>(results);
        auto& acc      = get<0>(values);
        auto& lse      = get<1>(values);
        auto  do_store = [&](auto store_zero, auto split_kv) {
          static constexpr bool StoreZero = decltype(store_zero)::value;
          static constexpr bool SplitKV   = decltype(split_kv)::value;
          epilogue.template store<StoreZero, SplitKV>(params.epilogue,
                                                      acc,
                                                      lse,
                                                      typename CollectiveMainloop::TiledMmaPV{},
                                                      threadIdx.x - MmaThreadOffset,
                                                      seqlen_info,
                                                      block_coord);
        };

        if (!is_valid) {
          if (num_splits > 1) {
            do_store(/* StoreZero */ mute::true_type{}, /* SplitKV */ mute::true_type{});
          } else {
            do_store(/* StoreZero */ mute::true_type{}, /* SplitKV */ mute::false_type{});
          }
        } else {
          if (num_splits > 1) {
            do_store(/* StoreZero */ mute::false_type{}, /* SplitKV */ mute::true_type{});
          } else {
            do_store(/* StoreZero */ mute::false_type{}, /* SplitKV */ mute::false_type{});
          }
        }
        // barrier_kv.sync();
      }
    }
  }
};

}  // namespace mate::attention::fmha
