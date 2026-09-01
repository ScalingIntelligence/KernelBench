#pragma once

#include <mutlass/mutlass.h>
#include <mutlass/numeric_types.h>

#include <cstddef>
#include <mute/algorithm/clear.hpp>
#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/arch/mma_mp31.hpp>
#include <mute/atom/copy_atom.hpp>
#include <mute/atom/copy_traits_mp31.hpp>
#include <mute/int_tuple.hpp>
#include <mute/tensor.hpp>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/pipeline/mp31_pipeline.hpp>

#include "mate/gemm/deep_gemm/scaling_accumulation.hpp"
#include "mate/mega_moe/mega_barrier.hpp"
#include "mate/mega_moe/mega_moe_layout.hpp"
#include "mate/mega_moe/mega_moe_sym_buffer.hpp"
#include "mate/mega_moe/stage1_scheduler.hpp"
#include "mate/mega_moe/utils.hpp"

namespace mega_moe {

template <uint32_t kNumMaxTokensPerRank,
          uint32_t kHidden,
          uint32_t kIntermediateHidden,
          uint32_t kNumExperts,
          uint32_t kNumTopk,
          uint32_t kNumExpertsPerWave,
          uint32_t BLOCK_M,
          uint32_t BLOCK_N,
          uint32_t BLOCK_K,
          uint32_t kNumMaxPoolTokens,
          uint32_t kNumPaddedSFPoolTokens,
          uint32_t kNumStages,
          uint32_t kNumDispatchThreads,
          uint32_t kNumComputeThreads,
          uint32_t kNumMPs,
          uint32_t kNumRanks,
          uint32_t L1_SHAPE_N         = kIntermediateHidden * 2,
          uint32_t L1_SHAPE_K         = kHidden,
          uint32_t kNumDispatchWarps  = kNumDispatchThreads / mutlass::NumThreadsPerWarp,
          uint32_t kNumThreads        = kNumDispatchThreads + kNumComputeThreads,
          uint32_t kNumTokensPerWarp  = mutlass::NumThreadsPerWarp / kNumTopk,
          uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks>
__global__ __launch_bounds__(kNumThreads,
                             1) void fp8_fp8_mega_moe_stage1_impl(const uint32_t                     num_tokens,
                                                                  const layout::SymBuffer<kNumRanks> sym_buffer,
                                                                  const float* __restrict__ l1_weights_sf,
                                                                  const float* __restrict__ l1_acts_sf,
                                                                  mutlass::float_e4m3_t*   l2_acts,
                                                                  float*                   l2_acts_sf,
                                                                  const uint64_t           l1_weights_sf_expert_stride,
                                                                  const uint64_t           l1_weights_sf_n_stride,
                                                                  const uint64_t           l1_weights_sf_k_stride,
                                                                  const uint64_t           l1_acts_sf_row_stride,
                                                                  const uint64_t           l1_acts_sf_k_stride,
                                                                  const uint64_t           l2_acts_sf_row_stride,
                                                                  const uint64_t           l2_acts_sf_k_stride,
                                                                  const MUtensorDescriptor tensor_map_l1_acts,
                                                                  const MUtensorDescriptor tensor_map_l1_weights) {
  const uint32_t mp_idx     = blockIdx.x;
  const uint32_t thread_idx = threadIdx.x;
  const uint32_t warp_idx   = mutlass::canonical_warp_idx_sync();
  const uint32_t lane_idx   = mutlass::canonical_lane_idx();

  // Workspaces
  const auto workspace =
      layout::Workspace(sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk, BLOCK_M);

  // Token and buffer layouts
  constexpr auto fp8_token_layout              = layout::Data(kHidden);
  constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
  constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32, false);
  constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 32);
  constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
  constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
  constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

  const auto input_token_buffer = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
  const auto input_sf_buffer = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
  const auto input_topk_idx_buffer =
      layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
  const auto input_topk_weights_buffer =
      layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

  // L1 inputs
  const auto l1_token_buffer =
      layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
  const auto l1_sf_buffer = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
  const auto l1_topk_weights_buffer =
      layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());
  const auto l1_topk_weights_base = l1_topk_weights_buffer.get_base_ptr<float>();

  const auto l2_token_buffer =
      layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
  const auto l2_sf_buffer =
      layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());
  // Shared memory
  constexpr uint32_t kSharedMemoryAlignment = 4096;
  extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];
  using namespace stage1_sched;
  static_assert(kNumDispatchThreads > 0, "MegaMoE fused kernel requires dispatch threads");
  constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
      math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
  static_assert(kNumExperts * sizeof(uint32_t) + L1TileDesc::kNumWords * sizeof(uint32_t) <= SMEM_EXPERT_COUNT_SIZE,
                "Fused compute L1 tile descriptor must fit in expert-count shared-memory padding");

  constexpr uint32_t SMEM_TOKEN_PULL_SIZE =
      math::constexpr_align<uint32_t>(kHidden, kSharedMemoryAlignment) * kNumDispatchWarps;

  const auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
  auto       smem_l1_tile_desc = smem_expert_count + kNumExperts;
  auto smem_token_pull_base = reinterpret_cast<uint8_t*>(math::advance_ptr(smem_expert_count, SMEM_EXPERT_COUNT_SIZE));
  auto smem_mate_compute_base = math::advance_ptr<uint8_t>(smem_token_pull_base, SMEM_TOKEN_PULL_SIZE);

  for (int i = thread_idx; i < kNumExperts; i += kNumThreads) {
    smem_expert_count[i] = 0;
  }

  static_assert(BLOCK_M == 32, "Fused masked mate compute expects BLOCK_M=32");
  static_assert(BLOCK_N == 256, "Fused masked mate compute expects BLOCK_N=256");
  static_assert(BLOCK_K == 128, "Fused masked mate compute expects BLOCK_K=128");
  static_assert(kNumDispatchThreads == 128, "Fused masked mate compute expects one dispatch squad");
  static_assert(kNumComputeThreads == 384, "Fused masked mate compute expects three compute squads");
  static_assert(kNumStages == 3 || kNumStages == 4, "Fused masked mate compute expects three or four stages");
  static_assert(kIntermediateHidden % 128 == 0, "Fused masked mate output scales expect 128-column alignment");

  using namespace mute;
  using ElementA                                = mutlass::float_e4m3_t;
  using ElementB                                = mutlass::float_e4m3_t;
  using ElementD                                = mutlass::bfloat16_t;
  using ElementAccumulator                      = float;
  using TileShape                               = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
  constexpr uint32_t kSwigluGran                = 64;
  constexpr uint32_t kSwigluPairColumns         = 2 * kSwigluGran;
  constexpr uint32_t kSwigluColumnGroupsPerTile = BLOCK_N / kSwigluPairColumns;
  constexpr uint32_t kOutputColumnsPerTile      = BLOCK_N / 2;
  constexpr uint32_t kOutputScaleGroups         = kIntermediateHidden / kOutputColumnsPerTile;
  constexpr int      AlignmentA                 = 32 / sizeof_bits_v<ElementA>;
  constexpr int      AlignmentB                 = 32 / sizeof_bits_v<ElementB>;
  static_assert(BLOCK_N % kSwigluPairColumns == 0, "Fused masked mate expects full SwiGLU column pairs");
  static_assert(kSwigluColumnGroupsPerTile == 2, "Fused masked mate output expects two SwiGLU column groups");
  static_assert(kOutputColumnsPerTile == 128, "Fused masked mate output scales expect 128 output columns per tile");

  using StrideA       = decltype(make_stride(Int<L1_SHAPE_K>{}, _1{}, Int<kNumMaxPoolTokens * L1_SHAPE_K>{}));
  using StrideB       = decltype(make_stride(Int<L1_SHAPE_K>{}, _1{}, Int<L1_SHAPE_N * L1_SHAPE_K>{}));
  using CollectiveMma = typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                                              mutlass::arch::OpClassTensorOp,
                                                                              ElementA,
                                                                              StrideA,
                                                                              AlignmentA,
                                                                              ElementB,
                                                                              StrideB,
                                                                              AlignmentB,
                                                                              ElementAccumulator,
                                                                              TileShape,
                                                                              Shape<_1, _1, _1>,
                                                                              Int<kNumStages>,
                                                                              mutlass::gemm::KernelTme>::CollectiveOp;

  using TiledMma    = typename CollectiveMma::TiledMma;
  using SmemLayoutA = typename CollectiveMma::SmemLayoutA;
  using SmemLayoutB = typename CollectiveMma::SmemLayoutB;

  constexpr int NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  constexpr int NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;
  constexpr int NumLoadWarpSquads      = 1;
  constexpr int NumMmaWarpSquads       = 2;
  constexpr int NumMmaThreads          = NumMmaWarpSquads * NumThreadsPerWarpSquad;
  constexpr int NumMmaWarps            = NumMmaThreads / NumThreadsPerWarp;
  constexpr int WarpsPerWarpSquad      = NumThreadsPerWarpSquad / NumThreadsPerWarp;
  constexpr int TotalWarpSquads        = NumLoadWarpSquads + NumMmaWarpSquads;

  using PipelineAB          = mutlass::Mp31PipelineTmeAsync<kNumStages>;
  using PipelineABParams    = typename PipelineAB::Params;
  using PipelineABState     = typename PipelineAB::PipelineState;
  using PipelineSeq         = mutlass::OrderedSequenceBarrier<1, NumMmaWarpSquads>;
  using PipelineSeqParams   = typename PipelineSeq::Params;
  using ConsumerSyncBarrier = mutlass::arch::AsyncBarrier;

  constexpr int TmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  constexpr int TmeTransactionBytesB =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);
  constexpr int      TmeTransactionBytesAB        = TmeTransactionBytesA + TmeTransactionBytesB;
  constexpr uint32_t kComputeTileDescBarrierWarps = 1 + NumMmaWarps;
  constexpr uint32_t kNumDispatchBarriers         = 2;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineAB[PipelineAB::NumBarriers];
    uint8_t PipelineSeq[PipelineSeq::NumBarriers];
    uint8_t ConsumerSync[1];
    uint8_t ComputeTileDesc[1];
    uint8_t TokenPull[kNumDispatchWarps];
    uint8_t Dispatch[kNumDispatchBarriers];
  };
  static_assert(sizeof(BarrierStorage) + ConsumerSyncBarrier::ReservedAsyncBarrierCount <=
                    ConsumerSyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "Async barrier storage exceeds hardware barrier range");

  struct MateSharedStorage {
    mute::array_aligned<ElementA, cosize_v<SmemLayoutA>, 256>             smem_a;
    mute::array_aligned<ElementB, cosize_v<SmemLayoutB>, 256>             smem_b;
    mute::array_aligned<float, kSwigluColumnGroupsPerTile * BLOCK_M, 256> smem_output_amax;
  };

  auto& shared          = *reinterpret_cast<MateSharedStorage*>(smem_mate_compute_base);
  auto* barrier_storage = reinterpret_cast<BarrierStorage*>(0);
  mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage) + ConsumerSyncBarrier::ReservedAsyncBarrierCount);

  PipelineABParams pipe_params;
  pipe_params.transaction_bytes = TmeTransactionBytesAB;
  pipe_params.num_consumers     = NumMmaWarps;
  pipe_params.num_producers     = 1;
  PipelineAB pipeline(pipe_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineAB));

  const int  compute_thread_idx = int(threadIdx.x) - int(kNumDispatchThreads);
  const int  compute_warp_idx   = compute_thread_idx >= 0 ? compute_thread_idx / NumThreadsPerWarp : 0;
  const int  squad              = compute_thread_idx >= 0 ? compute_thread_idx / NumThreadsPerWarpSquad : 0;
  const int  warp_idx_in_squad  = compute_warp_idx & (WarpsPerWarpSquad - 1);
  const bool is_producer_squad  = (threadIdx.x >= kNumDispatchThreads) && (squad == 0);
  const bool is_consumer_squad =
      (threadIdx.x >= kNumDispatchThreads) && (squad >= NumLoadWarpSquads) && (squad < TotalWarpSquads);
  const bool is_tme_issue_warp = is_producer_squad && (warp_idx_in_squad == 0);

  Tensor sA = make_tensor(make_smem_ptr(shared.smem_a.data()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(shared.smem_b.data()), SmemLayoutB{});

  PipelineABState pipe_write = mutlass::make_producer_start_state<PipelineAB>();
  PipelineABState pipe_read;

  PipelineSeqParams seq_params{};
  seq_params.group_size = WarpsPerWarpSquad;
  seq_params.group_id   = is_consumer_squad ? squad - NumLoadWarpSquads : 0;
  PipelineSeq         pipeline_seq(seq_params,
                           static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->PipelineSeq)));
  ConsumerSyncBarrier consumer_sync(static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->ConsumerSync)));
  const uint32_t      compute_tile_desc_barrier_idx =
      static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->ComputeTileDesc)) +
      ConsumerSyncBarrier::ReservedAsyncBarrierCount;
  const uint32_t token_pull_barrier_base_idx =
      static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->TokenPull)) +
      ConsumerSyncBarrier::ReservedAsyncBarrierCount;
  const uint32_t dispatch_barrier_base_idx =
      static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->Dispatch)) +
      ConsumerSyncBarrier::ReservedAsyncBarrierCount;
  if (thread_idx == 0) {
    consumer_sync.init(NumMmaWarps, 0);
    ConsumerSyncBarrier::init(compute_tile_desc_barrier_idx, kComputeTileDescBarrierWarps, 0);
#pragma unroll
    for (uint32_t i = 0; i < kNumDispatchWarps; ++i) {
      ConsumerSyncBarrier::init(token_pull_barrier_base_idx + i, 1, 0);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumDispatchBarriers; ++i) {
      ConsumerSyncBarrier::init(dispatch_barrier_base_idx + i, kNumDispatchWarps, 0);
    }
  }

  // Publish shared-memory zeroing and async barrier initialization.
  __syncthreads();

  uint32_t dispatch_barrier_call_idx = 0;
  auto     dispatch_warp_barrier     = [&]() {
    const uint32_t barrier_idx = dispatch_barrier_base_idx + (dispatch_barrier_call_idx & 1u);
    const unsigned phase_id    = ConsumerSyncBarrier::arrive<true>(barrier_idx);
    ConsumerSyncBarrier::wait(barrier_idx, phase_id);
    ++dispatch_barrier_call_idx;
  };
  auto compute_thread_sync = [&]() {
    const unsigned phase_id = ConsumerSyncBarrier::arrive<true>(compute_tile_desc_barrier_idx);
    ConsumerSyncBarrier::wait(compute_tile_desc_barrier_idx, phase_id);
  };

  // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
  constexpr uint32_t kDispatchGridSyncIndex = 0;

  // MTLink barrier tags
  constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;

  using DispatchScheduler = stage1_sched::MegaMoEL1Scheduler<BLOCK_M,
                                                             BLOCK_N,
                                                             BLOCK_K,
                                                             L1_SHAPE_N,
                                                             L1_SHAPE_K,
                                                             kNumExpertsPerRank,
                                                             kNumExpertsPerWave,
                                                             kNumMPs,
                                                             kNumRanks>;

  if (warp_idx < kNumDispatchWarps) {
    // Adjust registers FIXME not Supported
    // cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

    constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
    const auto         read_topk_idx     = [&](const auto& process) {
// TODO: figure out better unrolling
// Now, `unroll` is better than `unroll 8`
#pragma unroll
      for (uint32_t i = (mp_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp; i < num_tokens;
           i += kNumMPs * kNumDispatchWarps * kNumTokensPerWarp) {
        // Allocate slots for each token-topk
        int expert_idx = -1;
        if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
          expert_idx = static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
          if (expert_idx >= 0) process(i * kNumTopk + lane_idx, expert_idx);
        }
        __syncwarp();
      }
    };

    // Count experts' tokens
    read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
      atomic_add_shared_u32(smem_expert_count + expert_idx, 1);
    });

    dispatch_warp_barrier();

    // Get MP offset (~6.5 us)
#pragma unroll
    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
      const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
      smem_expert_count[i]      = static_cast<uint32_t>(
          atomic_add_global_u64((unsigned long long*)workspace.get_expert_send_count_ptr(i), send_value));
    }

    dispatch_warp_barrier();

    // Write source indices (~2 us with 512 tokens)
    read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
      const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
      const auto dst_slot_idx = atomic_add_shared_u32(smem_expert_count + expert_idx, 1);
      const auto dst_ptr =
          workspace.get_src_token_topk_idx_ptr(expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
      *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
    });

    // Grid sync
    grid_sync<kNumMPs, kDispatchGridSyncIndex>(workspace, mp_idx, thread_idx, [&]() { dispatch_warp_barrier(); });

    // Write expert count
    if (mp_idx == 0) {
#pragma unroll
      for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
        const auto dst_rank_idx         = i / kNumExpertsPerRank;
        const auto dst_local_expert_idx = i % kNumExpertsPerRank;
        const auto expert_status        = ld_volatile_global(workspace.get_expert_send_count_ptr(i));
        st_relaxed_sys_global(
            sym_buffer.map(workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                           dst_rank_idx),
            expert_status & 0xffffffffull);
        atomic_add_global_u64(reinterpret_cast<unsigned long long*>(sym_buffer.map(
                                  workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx)),
                              static_cast<unsigned long long>(expert_status));
      }
    }

    dispatch_warp_barrier();

    // Barrier before pulling
    mtlink_barrier<kNumRanks,
                   kNumMPs,
                   kNumDispatchThreads,
                   kDispatchGridSyncIndex,
                   kBeforeDispatchPullBarrierTag,
                   true>(
        workspace,
        sym_buffer,
        mp_idx,
        thread_idx,
        [&]() { dispatch_warp_barrier(); },
        /* After the grid sync above, there is no more writes by other MPs (except 0) */ false,
        /* After the MTLink barrier, there is a grid sync */ true);

    DispatchScheduler scheduler(workspace);
    if (warp_idx == 0) {
      scheduler.fetch_expert_recv_count();
      scheduler.store_expert_recv_count_to_shared(smem_expert_count);
    }
    dispatch_warp_barrier();
    scheduler.load_expert_recv_count_from_shared(smem_expert_count);

    // Materialize remote dispatched tokens into the local L1 pool; compute waits on l1_arrival_count.
    if constexpr (true) {
      // Per-rank counts for current expert (re-loaded when expert changes)
      constexpr uint32_t kNumRanksPerLane   = math::constexpr_ceil_div(kNumRanks, uint32_t(mutlass::NumThreadsPerWarp));
      int                current_expert_idx = -1;
      uint32_t           stored_rank_count[kNumRanksPerLane] = {};
      uint32_t           expert_start_idx = 0, expert_end_idx = 0;
      uint32_t           expert_pool_block_offset = 0;

      constexpr uint32_t kNumGlobalWarps = kNumMPs * kNumDispatchWarps;

      for (uint32_t token_idx = mp_idx * kNumDispatchWarps + warp_idx;; token_idx += kNumGlobalWarps) {
        // Advance expert until within the range
        int old_expert_idx = current_expert_idx;
        while (token_idx >= expert_end_idx) {
          if (++current_expert_idx >= kNumExpertsPerRank) break;

          // Update pool block offset for the new expert
          expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

          // Move start and end to the next expert
          expert_start_idx = expert_end_idx;
          expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
        }

        // Finish all tokens
        if (current_expert_idx >= kNumExpertsPerRank) break;

        // Load per-rank counts when expert changes
        if (old_expert_idx != current_expert_idx) {
          old_expert_idx = current_expert_idx;
#pragma unroll
          for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
            const uint32_t j = i * mutlass::NumThreadsPerWarp + lane_idx;
            // TODO: this is not coalesced
            stored_rank_count[i] = j < kNumRanks ? static_cast<uint32_t>(ld_volatile_global(
                                                       workspace.get_expert_recv_count_ptr(j, current_expert_idx)))
                                                 : 0;
          }
        }

        // Round-robin rank selection via iterative min-peeling
        uint32_t current_rank_in_expert_idx;
        uint32_t remaining[kNumRanksPerLane];
#pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++i) remaining[i] = stored_rank_count[i];
        uint32_t offset              = 0;
        uint32_t token_idx_in_expert = token_idx - expert_start_idx;
        uint32_t slot_idx            = token_idx_in_expert;
        uint32_t token_idx_in_rank;
        if constexpr (kNumRanksPerLane == 1) {
          while (true) {
            const uint32_t active_mask      = __ballot_sync(0xffffffffu, remaining[0] > 0);
            const uint32_t num_active_ranks = __popc(active_mask);
            const uint32_t length           = warp_reduce_min_u32(remaining[0] > 0 ? remaining[0] : 0xffffffff);

            // Hit in the current round
            const uint32_t num_round_tokens = length * num_active_ranks;
            if (slot_idx < num_round_tokens) {
              const uint32_t slot_idx_in_round  = slot_idx % num_active_ranks;
              const uint32_t lane_lt_mask       = (1u << lane_idx) - 1u;
              const uint32_t rank_order         = __popc(active_mask & lane_lt_mask);
              const bool     is_selected_rank   = remaining[0] > 0 and rank_order == slot_idx_in_round;
              const uint32_t selected_rank_mask = __ballot_sync(0xffffffffu, is_selected_rank);
              current_rank_in_expert_idx        = static_cast<uint32_t>(__ffs(selected_rank_mask) - 1);
              token_idx_in_rank                 = offset + (slot_idx / num_active_ranks);
              break;
            }

            // Move into the next round
            slot_idx -= num_round_tokens;
            offset += length;
            remaining[0] -= mute::min(remaining[0], length);
          }
        } else {
          while (true) {
            // Compute active count and min across all ranks
            // NOTES: reduce within each lane first, then warp-reduce once
            uint32_t num_actives_in_lane = 0;
            uint32_t min_in_lane         = 0xffffffff;
#pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
              num_actives_in_lane += remaining[i] > 0;
              if (remaining[i] > 0) min_in_lane = mute::min(min_in_lane, remaining[i]);
            }
            const uint32_t num_active_ranks = warp_reduce_add_u32(num_actives_in_lane);
            const uint32_t length           = warp_reduce_min_u32(min_in_lane);

            // Hit in the current round
            const uint32_t num_round_tokens = length * num_active_ranks;
            if (slot_idx < num_round_tokens) {
              const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
              uint32_t       num_seen_ranks    = 0;
              current_rank_in_expert_idx       = 0;
#pragma unroll
              for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                const uint32_t mask             = __ballot_sync(0xffffffffu, remaining[i] > 0);
                const uint32_t num_active_lanes = __popc(mask);
                if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                  current_rank_in_expert_idx =
                      i * mutlass::NumThreadsPerWarp + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                num_seen_ranks += num_active_lanes;
              }
              token_idx_in_rank = offset + (slot_idx / num_active_ranks);
              break;
            }

            // Move into the next round
            slot_idx -= num_round_tokens;
            offset += length;
#pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++i) remaining[i] -= mute::min(remaining[i], length);
          }
        }

        // Read source token-topk index (written by remote dispatch via MTLink)
        uint32_t src_token_topk_idx = 0;
        if (lane_idx == 0) {
          src_token_topk_idx = ld_volatile_global(
              workspace.get_src_token_topk_idx_ptr(current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank));
        }
        src_token_topk_idx           = __shfl_sync(0xffffffffu, src_token_topk_idx, 0);
        const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
        const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

        // Store weights and token data
        const uint32_t     pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
        constexpr uint32_t hidden_int4    = kHidden / sizeof(int4);
        const auto         src_data = sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                             current_rank_in_expert_idx);
        auto               smem_token_pull =
            smem_token_pull_base + warp_idx * math::constexpr_align<uint32_t>(kHidden, kSharedMemoryAlignment);
        const uint32_t token_pull_barrier_idx = token_pull_barrier_base_idx + warp_idx;
        const auto     dst_token_data         = l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr();
        uint32_t       token_pull_phase_id    = 0;
        if (lane_idx == 0) {
          mutlass::arch::AsyncTransactionBarrier::expect_transaction(token_pull_barrier_idx, kHidden);
          mute::MP31_BLK_COPY_G2S::copy(src_data, token_pull_barrier_idx, smem_token_pull, kHidden);
          token_pull_phase_id      = ConsumerSyncBarrier::arrive<true>(token_pull_barrier_idx);
          const float route_weight = *sym_buffer.map(
              input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx, current_rank_in_expert_idx);

          *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = route_weight;

          // Write source metadata for combine write-back.
          *workspace.get_token_src_metadata_ptr(pool_token_idx) = {
              current_rank_in_expert_idx, src_token_idx, src_topk_idx};
        }

        // Load and store SF while the token TME load is in flight.
        // SF buffer uses compute-only row-major layout:
        // sf_base[token_idx * kNumSFUint32 + k_block]
        constexpr uint32_t kNumSFUint32 = kHidden / 128;
        static_assert(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
        const auto remote_sf_ptr = sym_buffer.map(input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                                                  current_rank_in_expert_idx);
        const auto local_sf_base = l1_sf_buffer.get_base_ptr<float>();
        for (int i = lane_idx; i < kNumSFUint32; i += mutlass::NumThreadsPerWarp)
          local_sf_base[pool_token_idx * kNumSFUint32 + i] = remote_sf_ptr[i];

        if (lane_idx == 0) {
          ConsumerSyncBarrier::wait(token_pull_barrier_idx, token_pull_phase_id);
        }
        __syncwarp();
        const auto src_shared_data = (int4*)smem_token_pull;
        const auto dst_data        = (int4*)dst_token_data;
        UNROLLED_WARP_COPY(8, lane_idx, hidden_int4, dst_data, src_shared_data);

        __syncwarp();

        if (lane_idx == 0) {
          /* Publish local pooled token data and metadata before signaling GEMM. */
          __threadfence_system_noflush();

          atomic_add_global_u32(
              workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
        }
      }
    }
  }
  if (thread_idx >= kNumDispatchThreads) {
    constexpr uint32_t kNumKBlocks       = L1_SHAPE_K / BLOCK_K;
    constexpr int      ScaleGranularityM = 1;
    constexpr int      ScaleGranularityN = kSwigluGran;
    constexpr int      ScaleMsPerTile    = BLOCK_M / ScaleGranularityM;
    constexpr int      ScaleNsPerTile    = BLOCK_N / ScaleGranularityN;
    static_assert(ScaleMsPerTile == 32, "Fused masked mate scale-A tile expects 32 rows");
    static_assert(ScaleNsPerTile == 4, "Fused masked mate scale-B tile expects four 64-col chunks");

    using TmeLoadA = mute::MP31_TME_LOAD_3D<mute::TME::SmemSwizzleGranularity::B16,
                                            mute::TME::SmemSwizzleStride::B256,
                                            mute::TME::SmemSwizzleLine::B256,
                                            mute::TME::CacheHint::CACHE_NORMAL,
                                            mute::TME::CacheHint::CACHE_NONE,
                                            mute::PrefetchSize::B128>;
    using TmeLoadB = mute::MP31_TME_LOAD_3D<mute::TME::SmemSwizzleGranularity::B16,
                                            mute::TME::SmemSwizzleStride::B256,
                                            mute::TME::SmemSwizzleLine::B256,
                                            mute::TME::CacheHint::CACHE_NORMAL,
                                            mute::TME::CacheHint::CACHE_NONE,
                                            mute::PrefetchSize::B128>;

    if (is_tme_issue_warp) {
      DispatchScheduler scheduler(workspace);
      scheduler.init_l1_dispatch_count_scheduler();
      MUTLASS_PRAGMA_NO_UNROLL
      while (true) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        scheduler.schedule_next_arrived_l1_tile(static_cast<uint32_t>(compute_thread_idx), smem_l1_tile_desc);
        compute_thread_sync();

        const L1TileDesc desc = L1TileDesc::load_from_smem(smem_l1_tile_desc);
        if (!desc.is_valid()) break;

        const uint32_t expert_idx  = desc.local_expert_idx;
        const uint32_t n_block_idx = desc.n_block_idx;
        const uint32_t m_base      = desc.pool_block_idx * BLOCK_M;

        MUTLASS_PRAGMA_NO_UNROLL
        for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
          const int stage = int(pipe_write.index());
          pipeline.producer_acquire(pipe_write);
          const uint32_t bar_id = pipeline.producer_get_barrier_id(pipe_write);

          auto tAsA = sA(_, _, stage);
          TmeLoadA::copy(&tensor_map_l1_acts,
                         bar_id,
                         raw_pointer_cast(tAsA.data()),
                         static_cast<int32_t>(kb * BLOCK_K),
                         static_cast<int32_t>(m_base),
                         0,
                         static_cast<int32_t>(BLOCK_K),
                         static_cast<int32_t>(BLOCK_M),
                         1);

          auto tBsB = sB(_, _, stage);
          TmeLoadB::copy(&tensor_map_l1_weights,
                         bar_id,
                         raw_pointer_cast(tBsB.data()),
                         static_cast<int32_t>(kb * BLOCK_K),
                         static_cast<int32_t>(n_block_idx * BLOCK_N),
                         static_cast<int32_t>(expert_idx),
                         static_cast<int32_t>(BLOCK_K),
                         static_cast<int32_t>(BLOCK_N),
                         1);
          ++pipe_write;
        }
      }
    } else if (is_consumer_squad) {
      MUTLASS_PRAGMA_NO_UNROLL
      while (true) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        compute_thread_sync();

        const L1TileDesc desc = L1TileDesc::load_from_smem(smem_l1_tile_desc);
        if (!desc.is_valid()) break;

        const uint32_t expert_idx           = desc.local_expert_idx;
        const uint32_t n_block_idx          = desc.n_block_idx;
        const uint32_t pool_block_idx       = desc.pool_block_idx;
        const uint32_t valid_m              = desc.valid_m;
        auto           consumer_thread_sync = [&]() {
          const unsigned phase_id = consumer_sync.arrive</* return_phase = */ true>();
          consumer_sync.wait(phase_id);
        };

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(compute_thread_idx - int(NumLoadWarpSquads * NumThreadsPerWarpSquad));

        auto rAcc = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
        fill(rAcc, ElementAccumulator(0));

        Tensor tCsA = thr_mma.partition_A(sA);
        Tensor tCsB = thr_mma.partition_B(sB);
        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);

        Tensor mScaleA = make_tensor(make_gmem_ptr(l1_acts_sf),
                                     make_shape(Int<kNumPaddedSFPoolTokens>{}, Int<kNumKBlocks>{}),
                                     make_stride(l1_acts_sf_row_stride, l1_acts_sf_k_stride));
        Tensor mScaleB = make_tensor(
            make_gmem_ptr(l1_weights_sf),
            make_shape(Int<L1_SHAPE_N / ScaleGranularityN>{}, Int<kNumKBlocks>{}, Int<kNumExpertsPerRank>{}),
            make_stride(l1_weights_sf_n_stride, l1_weights_sf_k_stride, l1_weights_sf_expert_stride));

        Tensor gScaleA    = local_tile(mScaleA, make_tile(Int<ScaleMsPerTile>{}), make_coord(pool_block_idx, _));
        Tensor gScaleB_nk = mScaleB(_, _, expert_idx);
        Tensor gScaleB    = local_tile(gScaleB_nk, make_tile(Int<ScaleNsPerTile>{}), make_coord(n_block_idx, _));

        using ScaleAViewAsCLayout =
            Layout<Shape<Shape<Int<ScaleGranularityM>, Int<ScaleMsPerTile>>, Int<BLOCK_N>>, Stride<Stride<_0, _1>, _0>>;
        using ScaleBViewAsCLayout =
            Layout<Shape<Int<BLOCK_M>, Shape<Int<ScaleGranularityN>, Int<ScaleNsPerTile>>>, Stride<_0, Stride<_0, _1>>>;

        Tensor tCgScaleA0 = thr_mma.partition_C(gScaleA(_, 0).compose(ScaleAViewAsCLayout{}));
        Tensor tCgScaleB0 = thr_mma.partition_C(gScaleB(_, 0).compose(ScaleBViewAsCLayout{}));
        Tensor tCrScaleA  = make_tensor_like<float>(tCgScaleA0);
        Tensor tCrScaleB  = make_tensor_like<float>(tCgScaleB0);

        using ElementBlockScale = float;
        using ScaleGmemCopyAtom = Copy_Atom<MP31_ROBUST_LOAD<ElementBlockScale>, ElementBlockScale>;
        auto scale_copy_a =
            ScaleGmemCopyAtom{}.with(make_robust_desc(l1_acts_sf, static_cast<size_t>(cosize(mScaleA.layout()))));
        auto scale_copy_b =
            ScaleGmemCopyAtom{}.with(make_robust_desc(l1_weights_sf, static_cast<size_t>(cosize(mScaleB.layout()))));

        using AccumTensor     = decltype(rAcc);
        using DualBufferAccum = ::mate::deep_gemm::ScalingAccumulation<typename AccumTensor::engine_type,
                                                                       typename AccumTensor::layout_type>;

        auto cD   = make_identity_tensor(make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{}));
        auto tCcD = thr_mma.partition_C(cD);

        const uint32_t  mma_per_iter = uint32_t(size<2>(tCrA));
        DualBufferAccum accumulation(rAcc, mma_per_iter, mma_per_iter);
        tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;

        MUTLASS_PRAGMA_NO_UNROLL
        for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
          if (accumulation.prepare_if_needed()) {
            tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
          }

          Tensor tCgScaleA     = thr_mma.partition_C(gScaleA(_, kb).compose(ScaleAViewAsCLayout{}));
          Tensor tCgScaleB     = thr_mma.partition_C(gScaleB(_, kb).compose(ScaleBViewAsCLayout{}));
          auto   tCgScaleAFlat = mute::filter_zeros(tCgScaleA);
          auto   tCgScaleBFlat = mute::filter_zeros(tCgScaleB);
          auto   tCrScaleAFlat = mute::filter_zeros(tCrScaleA);
          auto   tCrScaleBFlat = mute::filter_zeros(tCrScaleB);
          copy(scale_copy_a, tCgScaleAFlat, tCrScaleAFlat);
          copy(scale_copy_b, tCgScaleBFlat, tCrScaleBFlat);

          pipeline.consumer_wait(pipe_read);
          pipeline_seq.wait();
          const int stage = int(pipe_read.index());

          MUTE_UNROLL
          for (int mma_k = 0; mma_k < size<2>(tCrA); ++mma_k) {
            gemm(tiled_mma, tCrA(_, _, mma_k, stage), tCrB(_, _, mma_k, stage), accumulation());
            tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
          }
          ::mate::warpsquad_commit_batch();
          pipeline_seq.arrive();

          ::mate::warpsquad_wait();
          pipeline.consumer_release(pipe_read);
          ++pipe_read;

          accumulation.scale_if_needed(tCrScaleA, tCrScaleB);
        }
        accumulation.scale_residue_if_needed(tCrScaleA, tCrScaleB);

        static_assert(size(take<0, 2>(TileShape{})) == size(make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{})),
                      "Unexpected fused masked mate C tile shape");
        constexpr int kAccumPerThread      = 32;
        constexpr int kAccumColsPerRow     = 16;
        constexpr int kGateAccumColsPerRow = kAccumColsPerRow / 2;
        constexpr int kPairOffset          = kGateAccumColsPerRow;
        static_assert(kAccumPerThread == 32, "Fused masked mate output expects 32 C registers per thread");

        const int      consumer_thread_idx = compute_thread_idx - int(NumLoadWarpSquads * NumThreadsPerWarpSquad);
        const uint32_t lane_idx            = uint32_t(consumer_thread_idx) % uint32_t(NumThreadsPerWarp);
        const uint32_t lane_col            = uint32_t(consumer_thread_idx) & 7u;
        if (consumer_thread_idx < int(kSwigluColumnGroupsPerTile * BLOCK_M)) {
          shared.smem_output_amax[consumer_thread_idx] = 0.0f;
        }
        consumer_thread_sync();

        MUTE_UNROLL
        for (int row_group = 0; row_group < int(kSwigluColumnGroupsPerTile); ++row_group) {
          float          local_amax       = 0.0f;
          const int      first_gate_i     = row_group * kAccumColsPerRow;
          const auto     first_coord      = tCcD(first_gate_i);
          const uint32_t row_for_group    = uint32_t(get<0>(first_coord));
          const uint32_t swiglu_group_idx = uint32_t(get<1>(first_coord)) / kSwigluPairColumns;
          const bool     group_valid      = row_for_group < valid_m;
          MUTE_UNROLL
          for (int col_group = 0; col_group < kGateAccumColsPerRow; ++col_group) {
            const int gate_i = row_group * kAccumColsPerRow + col_group;
            const int up_i   = gate_i + kPairOffset;
            if (!group_valid) {
              continue;
            }

            const auto  gate_bf16     = ElementD(rAcc(gate_i));
            const auto  up_bf16       = ElementD(rAcc(up_i));
            const float gate          = static_cast<float>(gate_bf16);
            const float up            = static_cast<float>(up_bf16);
            const float gate_act      = gate / (1.0f + expf(-gate));
            const float gate_act_bf16 = static_cast<float>(ElementD(gate_act));
            const float y             = static_cast<float>(ElementD(up * gate_act_bf16));
            rAcc(gate_i)              = y;
            local_amax                = fmaxf(local_amax, fabsf(y));
          }

          const float reduced_amax = octet_reduce_max(local_amax, lane_idx);
          if (group_valid && lane_col == 0) {
            shared.smem_output_amax[swiglu_group_idx * BLOCK_M + row_for_group] = reduced_amax;
          }
        }
        consumer_thread_sync();

        if (consumer_thread_idx < int(valid_m)) {
          constexpr uint32_t kFirstSwigluColumnGroup  = 0;
          constexpr uint32_t kSecondSwigluColumnGroup = 1;
          const uint32_t     row                      = uint32_t(consumer_thread_idx);
          const float        amax      = fmaxf(fmaxf(shared.smem_output_amax[kFirstSwigluColumnGroup * BLOCK_M + row],
                                         shared.smem_output_amax[kSecondSwigluColumnGroup * BLOCK_M + row]),
                                   kSGLangSwiGLUFP8AmaxFloor);
          const float        scale     = amax / kFinfoAmaxE4M3;
          shared.smem_output_amax[row] = scale;
          const uint64_t row_global    = uint64_t(pool_block_idx) * uint64_t(BLOCK_M) + row;
          l2_acts_sf[row_global * l2_acts_sf_row_stride + uint64_t(n_block_idx) * l2_acts_sf_k_stride] = scale;
        }
        consumer_thread_sync();

        MUTE_UNROLL
        for (int row_group = 0; row_group < int(kSwigluColumnGroupsPerTile); ++row_group) {
          const int      first_gate_i = row_group * kAccumColsPerRow;
          const auto     first_coord  = tCcD(first_gate_i);
          const uint32_t row          = uint32_t(get<0>(first_coord));
          if (row >= valid_m) {
            continue;
          }

          const float    scale      = shared.smem_output_amax[row];
          const uint64_t row_global = uint64_t(pool_block_idx) * uint64_t(BLOCK_M) + row;
          MUTE_UNROLL
          for (int col_group = 0; col_group < kGateAccumColsPerRow; ++col_group) {
            const int      gate_i = row_group * kAccumColsPerRow + col_group;
            const auto     coord  = tCcD(gate_i);
            const uint32_t col    = uint32_t(get<1>(coord));

            const float    y = rAcc(gate_i);
            const uint32_t output_col =
                n_block_idx * kOutputColumnsPerTile + (col / kSwigluPairColumns) * kSwigluGran + (col % kSwigluGran);
            l2_acts[row_global * uint64_t(kIntermediateHidden) + output_col] =
                static_cast<mutlass::float_e4m3_t>(clamp_fp8_e4m3(y / scale));
          }
        }
      }
    }
  }
}

}  // namespace mega_moe
