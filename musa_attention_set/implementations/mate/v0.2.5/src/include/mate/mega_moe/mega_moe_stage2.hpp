#pragma once

#include <mutlass/mutlass.h>

#include <cstddef>
#include <cstdint>
#include <mute/algorithm/clear.hpp>
#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/arch/mma_mp31.hpp>
#include <mute/atom/copy_atom.hpp>
#include <mute/atom/copy_traits_mp31.hpp>
#include <mute/tensor.hpp>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/pipeline/mp31_pipeline.hpp>

#include "mate/gemm/deep_gemm/scaling_accumulation.hpp"
#include "mate/mega_moe/mega_barrier.hpp"
#include "mate/mega_moe/mega_moe_layout.hpp"
#include "mate/mega_moe/mega_moe_sym_buffer.hpp"
#include "mate/mega_moe/stage2_scheduler.hpp"
#include "mate/mega_moe/utils.hpp"

namespace mega_moe {
namespace mate_stage2 {

template <uint32_t kNumExperts,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumRanks,
          uint32_t BLOCK_M,
          uint32_t kNumMPs,
          uint32_t kNumThreads,
          typename Scheduler>
__device__ __forceinline__ void clear_workspace_for_next_use(const layout::Workspace& workspace,
                                                             const Scheduler&         scheduler) {
  static_assert(kNumMPs > 1, "Workspace clean expects at least two CTAs");

  const uint32_t mp_idx     = blockIdx.x;
  const uint32_t thread_idx = threadIdx.x;

  if (mp_idx == 0) {
    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads) {
      st_relaxed_sys_global(workspace.get_expert_send_count_ptr(i), 0ull);
    }
  } else {
    for (uint32_t expert_idx = mp_idx - 1; expert_idx < kNumExpertsPerRank; expert_idx += kNumMPs - 1) {
      const uint32_t num_recv_tokens          = scheduler.get_num_tokens(expert_idx);
      const uint32_t num_recv_m_blocks        = (num_recv_tokens + BLOCK_M - 1) / BLOCK_M;
      const uint32_t expert_pool_block_offset = scheduler.get_pool_block_offset(expert_idx);

      for (uint32_t rank_idx = thread_idx; rank_idx < kNumRanks; rank_idx += kNumThreads) {
        st_relaxed_sys_global(workspace.get_expert_recv_count_ptr(rank_idx, expert_idx), 0ull);
      }

      for (uint32_t block_idx = thread_idx; block_idx < num_recv_m_blocks; block_idx += kNumThreads) {
        st_relaxed_sys_global(workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + block_idx), 0u);
      }

      if (thread_idx == 0) {
        st_relaxed_sys_global(workspace.get_expert_recv_count_sum_ptr(expert_idx), 0ull);
      }
    }
  }
}

}  // namespace mate_stage2

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
          uint32_t kNumMPs,
          uint32_t kNumRanks,
          uint32_t kNumThreads        = 512,
          uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks>
__global__ __launch_bounds__(kNumThreads,
                             1) void fp8_fp8_mega_moe_stage2_impl(const layout::SymBuffer<kNumRanks> sym_buffer,
                                                                  mutlass::bfloat16_t* __restrict__ y,
                                                                  const uint32_t num_tokens,
                                                                  const float* __restrict__ l2_acts_sf,
                                                                  const float* __restrict__ l2_weights_sf,
                                                                  const uint64_t           l2_acts_sf_row_stride,
                                                                  const uint64_t           l2_acts_sf_k_stride,
                                                                  const uint64_t           l2_weights_sf_expert_stride,
                                                                  const uint64_t           l2_weights_sf_n_stride,
                                                                  const uint64_t           l2_weights_sf_k_stride,
                                                                  const MUtensorDescriptor tensor_map_l2_acts,
                                                                  const MUtensorDescriptor tensor_map_l2_weights) {
  (void)kNumExpertsPerWave;

  using namespace mute;
  using ElementA                  = mutlass::float_e4m3_t;
  using ElementB                  = mutlass::float_e4m3_t;
  using ElementD                  = mutlass::bfloat16_t;
  using ElementAccumulator        = float;
  using TileShape                 = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
  constexpr int ScaleGranularityM = 1;
  constexpr int ScaleGranularityN = 128;

  const auto workspace =
      layout::Workspace(sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk, BLOCK_M);

  static_assert(BLOCK_M == 32, "Stage2 mate-style kernel expects BLOCK_M=32");
  static_assert(BLOCK_N == 256, "Stage2 mate-style kernel expects BLOCK_N=256");
  static_assert(BLOCK_K == 128, "Stage2 mate-style kernel expects BLOCK_K=128");
  static_assert(kNumThreads == 512, "Stage2 mate-style kernel expects four 128-thread squads");
  static_assert(kHidden % BLOCK_N == 0, "Stage2 hidden must be BLOCK_N-aligned");
  static_assert(kIntermediateHidden % BLOCK_K == 0, "Stage2 intermediate must be BLOCK_K-aligned");
  static_assert(kHidden % ScaleGranularityN == 0, "Stage2 L2 weight scales expect 128-column alignment");

  constexpr int kNumStages = 4;
  constexpr int AlignmentA = 32 / sizeof_bits_v<ElementA>;
  constexpr int AlignmentB = 32 / sizeof_bits_v<ElementB>;

  using StrideA =
      decltype(make_stride(Int<kIntermediateHidden>{}, _1{}, Int<kNumMaxPoolTokens * kIntermediateHidden>{}));
  using StrideB = decltype(make_stride(Int<kIntermediateHidden>{}, _1{}, Int<kHidden * kIntermediateHidden>{}));

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
  constexpr int NumStoreWarpSquads     = 1;
  constexpr int NumMmaThreads          = NumMmaWarpSquads * NumThreadsPerWarpSquad;
  constexpr int NumMmaWarps            = NumMmaThreads / NumThreadsPerWarp;
  constexpr int NumStoreThreads        = NumStoreWarpSquads * NumThreadsPerWarpSquad;
  constexpr int NumStoreWarps          = NumStoreThreads / NumThreadsPerWarp;
  constexpr int WarpsPerWarpSquad      = NumThreadsPerWarpSquad / NumThreadsPerWarp;
  constexpr int TotalWarpSquads        = NumLoadWarpSquads + NumMmaWarpSquads + NumStoreWarpSquads;

  using PipelineAB        = mutlass::Mp31PipelineTmeAsync<kNumStages>;
  using PipelineABParams  = typename PipelineAB::Params;
  using PipelineABState   = typename PipelineAB::PipelineState;
  using PipelineSeq       = mutlass::OrderedSequenceBarrier<1, NumMmaWarpSquads>;
  using PipelineSeqParams = typename PipelineSeq::Params;

  constexpr int TmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  constexpr int TmeTransactionBytesB =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);
  constexpr int TmeTransactionBytesAB = TmeTransactionBytesA + TmeTransactionBytesB;

  constexpr uint32_t kNumCDStages         = 1;
  constexpr uint32_t kStoreBlockM         = BLOCK_M;
  constexpr uint32_t kCDDescWords         = 4;
  constexpr uint32_t kCDDescFlagValidTile = 1u << 0;
  constexpr uint32_t kCDDescFlagLastTile  = 1u << 1;
  constexpr uint32_t kCDSlotBytes         = kStoreBlockM * BLOCK_N * sizeof(ElementD);
  static_assert(kStoreBlockM == BLOCK_M, "Stage2 CD path stores one full M tile per slot");

  struct SharedStorage {
    mute::array_aligned<ElementA, cosize_v<SmemLayoutA>, 256>       smem_a;
    mute::array_aligned<ElementB, cosize_v<SmemLayoutB>, 256>       smem_b;
    mute::array_aligned<uint8_t, kNumCDStages * kCDSlotBytes, 256>  smem_cd;
    mute::array_aligned<uint32_t, kNumCDStages * kCDDescWords, 256> smem_cd_desc;
  };

  extern __shared__ __align__(256) uint8_t smem_buffer[];
  auto&                                    shared = *reinterpret_cast<SharedStorage*>(smem_buffer);

  using PipelineCD       = mutlass::Mp31PipelineAsync<kNumCDStages>;
  using PipelineCDParams = typename PipelineCD::Params;
  using PipelineCDState  = typename PipelineCD::PipelineState;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineAB[PipelineAB::NumBarriers];
    uint8_t PipelineSeq[PipelineSeq::NumBarriers];
    uint8_t PipelineCD[PipelineCD::NumBarriers];
  };
  static_assert(sizeof(BarrierStorage) + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount <=
                    mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "Stage2 async barrier id exceeds hardware barrier range");

  auto* barrier_storage = reinterpret_cast<BarrierStorage*>(0);
  mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage) +
                                         mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount);

  PipelineABParams pipe_params;
  pipe_params.transaction_bytes = TmeTransactionBytesAB;
  pipe_params.num_consumers     = NumMmaWarps;
  pipe_params.num_producers     = 1;
  PipelineAB pipeline(pipe_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineAB));

  const int  squad             = mutlass::canonical_warp_squad_idx();
  const int  warp_idx_in_squad = mutlass::canonical_warp_idx_sync() & (WarpsPerWarpSquad - 1);
  const bool is_producer_squad = (squad == 0);
  const bool is_consumer_squad = (squad >= NumLoadWarpSquads) && (squad < NumLoadWarpSquads + NumMmaWarpSquads);
  const bool is_store_squad    = (squad == NumLoadWarpSquads + NumMmaWarpSquads);
  const bool is_tme_issue_warp = is_producer_squad && (warp_idx_in_squad == 0);

  Tensor sA = make_tensor(make_smem_ptr(shared.smem_a.data()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(shared.smem_b.data()), SmemLayoutB{});

  PipelineABState pipe_write = mutlass::make_producer_start_state<PipelineAB>();
  PipelineABState pipe_read;

  PipelineSeqParams seq_params{};
  seq_params.group_size = WarpsPerWarpSquad;
  seq_params.group_id   = is_consumer_squad ? squad - NumLoadWarpSquads : 0;
  PipelineSeq pipeline_seq(seq_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineSeq));

  PipelineCDParams cd_pipe_params;
  cd_pipe_params.producer_arv_count = NumMmaWarps;
  cd_pipe_params.consumer_arv_count = NumStoreWarps;
  PipelineCD      cd_pipeline(cd_pipe_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineCD));
  PipelineCDState cd_pipe_write = mutlass::make_producer_start_state<PipelineCD>();
  PipelineCDState cd_pipe_read;

  __syncthreads();

  auto cd_slot_ptr = [&](const uint32_t slot) { return shared.smem_cd.data() + slot * kCDSlotBytes; };
  auto cd_desc_ptr = [&]() { return shared.smem_cd_desc.data(); };

  constexpr uint32_t kNumBlockNs    = kHidden / BLOCK_N;
  constexpr uint32_t kNumKBlocks    = kIntermediateHidden / BLOCK_K;
  constexpr int      ScaleMsPerTile = BLOCK_M / ScaleGranularityM;
  constexpr int      ScaleNsPerTile = BLOCK_N / ScaleGranularityN;
  static_assert(ScaleMsPerTile == 32, "Stage2 scale-A tile expects 32 rows");
  static_assert(ScaleNsPerTile == 2, "Stage2 scale-B tile expects two 128-col chunks");

  using Scheduler = mate_stage2::WorkspaceTileScheduler<kNumExpertsPerRank, kNumBlockNs, kNumMPs, kNumRanks, BLOCK_M>;

  using TmeLoadA = mute::MP31_TME_LOAD_3D<mute::TME::SmemSwizzleGranularity::B16,
                                          mute::TME::SmemSwizzleStride::B256,
                                          mute::TME::SmemSwizzleLine::B256,
                                          mute::TME::CacheHint::CACHE_NORMAL,
                                          mute::TME::CacheHint::CACHE_NORMAL,
                                          mute::PrefetchSize::B128>;
  using TmeLoadB = mute::MP31_TME_LOAD_3D<mute::TME::SmemSwizzleGranularity::B16,
                                          mute::TME::SmemSwizzleStride::B256,
                                          mute::TME::SmemSwizzleLine::B256,
                                          mute::TME::CacheHint::CACHE_NORMAL,
                                          mute::TME::CacheHint::CACHE_NONE,
                                          mute::PrefetchSize::B128>;

  constexpr auto fp8_token_layout              = layout::Data(kHidden);
  constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(mutlass::bfloat16_t));
  constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
  constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32, false);
  constexpr auto fp8_intermediate_sf_layout    = layout::Data((kIntermediateHidden / BLOCK_K) * sizeof(float));
  constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
  constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
  constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

  const auto input_token_buffer = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
  const auto input_sf_buffer = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
  const auto input_topk_idx_buffer =
      layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
  const auto input_topk_weights_buffer =
      layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());
  const auto l1_token_buffer =
      layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
  const auto l1_sf_buffer = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
  const auto l1_topk_weights_buffer =
      layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());
  const auto l2_token_buffer =
      layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
  const auto l2_sf_buffer =
      layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());
  const auto combine_token_buffer =
      layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

  const uint32_t lane_idx = mutlass::canonical_lane_idx();
  const uint32_t warp_idx = mutlass::canonical_warp_idx_sync();

  if (is_producer_squad) {
    Scheduler scheduler;
    auto      work = scheduler.initial_work_tile_info(workspace);
    MUTLASS_PRAGMA_NO_UNROLL
    for (; work.valid;) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
      __musa_loop_transparent_outermost();
#endif
      const uint32_t expert_idx  = work.expert_idx;
      const uint32_t n_block_idx = work.n_block_idx;
      const uint32_t m_base      = work.pool_block_idx * BLOCK_M;

      MUTLASS_PRAGMA_NO_UNROLL
      for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
        const int stage = int(pipe_write.index());
        if (is_tme_issue_warp) {
          pipeline.producer_acquire(pipe_write);
          const uint32_t bar_id = pipeline.producer_get_barrier_id(pipe_write);

          auto tAsA = sA(_, _, stage);
          TmeLoadA::copy(&tensor_map_l2_acts,
                         bar_id,
                         raw_pointer_cast(tAsA.data()),
                         static_cast<int32_t>(kb * BLOCK_K),
                         static_cast<int32_t>(m_base),
                         0,
                         static_cast<int32_t>(BLOCK_K),
                         static_cast<int32_t>(BLOCK_M),
                         1);

          auto tBsB = sB(_, _, stage);
          TmeLoadB::copy(&tensor_map_l2_weights,
                         bar_id,
                         raw_pointer_cast(tBsB.data()),
                         static_cast<int32_t>(kb * BLOCK_K),
                         static_cast<int32_t>(n_block_idx * BLOCK_N),
                         static_cast<int32_t>(expert_idx),
                         static_cast<int32_t>(BLOCK_K),
                         static_cast<int32_t>(BLOCK_N),
                         1);
        }
        ++pipe_write;
      }

      scheduler.advance_to_next_work();
      work = scheduler.get_work_tile_info();
    }
  } else if (is_consumer_squad) {
    Scheduler scheduler;
    auto      work                = scheduler.initial_work_tile_info(workspace);
    const int consumer_thread_idx = int(threadIdx.x) - int(NumLoadWarpSquads * NumThreadsPerWarpSquad);

    MUTLASS_PRAGMA_NO_UNROLL
    for (; work.valid;) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
      __musa_loop_transparent_outermost();
#endif
      const uint32_t expert_idx     = work.expert_idx;
      const uint32_t n_block_idx    = work.n_block_idx;
      const uint32_t pool_block_idx = work.pool_block_idx;
      const uint32_t valid_m        = work.valid_m;

      TiledMma tiled_mma;
      auto     thr_mma = tiled_mma.get_thread_slice(int(threadIdx.x) - int(NumLoadWarpSquads * NumThreadsPerWarpSquad));

      auto rAcc = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
      fill(rAcc, ElementAccumulator(0));

      Tensor tCsA = thr_mma.partition_A(sA);
      Tensor tCsB = thr_mma.partition_B(sB);
      Tensor tCrA = thr_mma.make_fragment_A(tCsA);
      Tensor tCrB = thr_mma.make_fragment_B(tCsB);

      Tensor mScaleA = make_tensor(make_gmem_ptr(l2_acts_sf),
                                   make_shape(Int<kNumPaddedSFPoolTokens>{}, Int<kNumKBlocks>{}),
                                   make_stride(l2_acts_sf_row_stride, l2_acts_sf_k_stride));
      Tensor mScaleB =
          make_tensor(make_gmem_ptr(l2_weights_sf),
                      make_shape(Int<kHidden / ScaleGranularityN>{}, Int<kNumKBlocks>{}, Int<kNumExpertsPerRank>{}),
                      make_stride(l2_weights_sf_n_stride, l2_weights_sf_k_stride, l2_weights_sf_expert_stride));

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
          ScaleGmemCopyAtom{}.with(make_robust_desc(l2_acts_sf, static_cast<size_t>(cosize(mScaleA.layout()))));
      auto scale_copy_b =
          ScaleGmemCopyAtom{}.with(make_robust_desc(l2_weights_sf, static_cast<size_t>(cosize(mScaleB.layout()))));

      using AccumTensor = decltype(rAcc);
      using DualBufferAccum =
          ::mate::deep_gemm::ScalingAccumulation<typename AccumTensor::engine_type, typename AccumTensor::layout_type>;

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

      scheduler.advance_to_next_work();
      auto next_work = scheduler.get_work_tile_info();

      cd_pipeline.producer_acquire(cd_pipe_write);
      const uint32_t slot = uint32_t(cd_pipe_write.index());
      if (consumer_thread_idx == 0) {
        auto* desc = cd_desc_ptr() + slot * kCDDescWords;
        desc[0]    = kCDDescFlagValidTile | (next_work.valid ? 0u : kCDDescFlagLastTile);
        desc[1]    = pool_block_idx * BLOCK_M;
        desc[2]    = n_block_idx * BLOCK_N;
        desc[3]    = valid_m;
      }

      MUTE_UNROLL
      for (int i = 0; i < size(rAcc); ++i) {
        const auto     coord = tCcD(i);
        const uint32_t row   = uint32_t(get<0>(coord));
        const uint32_t col   = uint32_t(get<1>(coord));
        if (row >= valid_m) {
          continue;
        }

        auto* smem_ptr = reinterpret_cast<ElementD*>(cd_slot_ptr(slot) + (row * BLOCK_N + col) * sizeof(ElementD));
        *smem_ptr      = ElementD(rAcc(i));
      }

      __threadfence_block();
      __syncwarp();
      if (lane_idx == 0) {
        cd_pipeline.producer_commit(cd_pipe_write);
      }
      ++cd_pipe_write;

      work = next_work;
    }

    if (cd_pipe_write.count() == 0) {
      cd_pipeline.producer_acquire(cd_pipe_write);
      const uint32_t slot = uint32_t(cd_pipe_write.index());
      if (consumer_thread_idx == 0) {
        auto* desc = cd_desc_ptr() + slot * kCDDescWords;
        desc[0]    = kCDDescFlagLastTile;
        desc[1]    = 0;
        desc[2]    = 0;
        desc[3]    = 0;
      }
      __threadfence_block();
      __syncwarp();
      if (lane_idx == 0) {
        cd_pipeline.producer_commit(cd_pipe_write);
      }
    }
  } else if (is_store_squad) {
    const uint32_t store_warp_idx =
        uint32_t(warp_idx) - uint32_t(NumLoadWarpSquads + NumMmaWarpSquads) * uint32_t(WarpsPerWarpSquad);

    MUTLASS_PRAGMA_NO_UNROLL
    while (true) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
      __musa_loop_transparent_outermost();
#endif
      const uint32_t slot = uint32_t(cd_pipe_read.index());
      cd_pipeline.consumer_wait(cd_pipe_read);
      __threadfence_block();

      auto*          desc            = cd_desc_ptr() + slot * kCDDescWords;
      const uint32_t desc_flags      = desc[0];
      const bool     has_tile        = (desc_flags & kCDDescFlagValidTile) != 0;
      const bool     is_last_tile    = (desc_flags & kCDDescFlagLastTile) != 0;
      const uint32_t m_base          = desc[1];
      const uint32_t output_col_base = desc[2];
      const uint32_t valid_m         = desc[3];

#pragma unroll
      for (uint32_t j = 0; j < kStoreBlockM / NumStoreWarps; ++j) {
        const uint32_t row_in_store = j * NumStoreWarps + store_warp_idx;
        const uint32_t row_in_tile  = row_in_store;
        if (has_tile && row_in_tile < valid_m) {
          const uint32_t row_global    = m_base + row_in_tile;
          const auto     src_metadata  = *workspace.get_token_src_metadata_ptr(row_global);
          const uint32_t dst_rank_idx  = src_metadata.rank_idx;
          const uint32_t dst_token_idx = src_metadata.token_idx;
          const uint32_t dst_topk_idx  = src_metadata.topk_idx;

          auto* smem_ptr = reinterpret_cast<const uint4*>(
              cd_slot_ptr(slot) + row_in_store * BLOCK_N * static_cast<uint32_t>(sizeof(ElementD)) +
              lane_idx * static_cast<uint32_t>(sizeof(uint4)));
          const uint4 packed = *smem_ptr;

          const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token_idx);
          auto*      dst_ptr   = math::advance_ptr<uint4>(dst_token.get_base_ptr(),
                                                   output_col_base * static_cast<uint32_t>(sizeof(ElementD)) +
                                                       lane_idx * static_cast<uint32_t>(sizeof(uint4)));
          *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
        }
      }

      __threadfence_block();
      __syncwarp();
      if (lane_idx == 0) {
        cd_pipeline.consumer_release(cd_pipe_read);
      }
      ++cd_pipe_read;

      if (is_last_tile) {
        break;
      }
    }
  }

  if (is_store_squad) {
    __threadfence_system_noflush();
  }
  Scheduler cleanup_scheduler;
  cleanup_scheduler.fetch_counts(workspace);
  grid_sync<kNumMPs, 0>(workspace, blockIdx.x, threadIdx.x, [&]() { __syncthreads_lm(); });
  mate_stage2::clear_workspace_for_next_use<kNumExperts, kNumExpertsPerRank, kNumRanks, BLOCK_M, kNumMPs, kNumThreads>(
      workspace, cleanup_scheduler);
  __syncthreads_lm();

  mtlink_barrier<kNumRanks, kNumMPs, kNumThreads, 0, 2, true>(
      workspace, sym_buffer, blockIdx.x, threadIdx.x, [&]() { __syncthreads_lm(); }, true, true);

  constexpr uint32_t kNumWarps            = kNumThreads / NumThreadsPerWarp;
  constexpr uint32_t kNumHiddenBytes      = kHidden * sizeof(mutlass::bfloat16_t);
  constexpr uint32_t kNumElemsPerUint4    = sizeof(uint4) / sizeof(mutlass::bfloat16_t);
  constexpr uint32_t kNumUint4PerToken    = kNumHiddenBytes / sizeof(uint4);
  constexpr uint32_t kNumUint4PerWarpTile = NumThreadsPerWarp;
  constexpr uint32_t kNumHiddenWarpTiles  = kNumUint4PerToken / kNumUint4PerWarpTile;
  static_assert(kNumTopk <= NumThreadsPerWarp, "Top-k must fit in one warp");
  static_assert(kNumHiddenBytes % sizeof(uint4) == 0, "Hidden bytes must be uint4 aligned");
  static_assert(kNumUint4PerToken % kNumUint4PerWarpTile == 0,
                "Stage2 combine expects full warp-aligned hidden chunks");

  const uint32_t num_combine_tasks = num_tokens * kNumHiddenWarpTiles;
  for (uint32_t task_idx = blockIdx.x * kNumWarps + warp_idx; task_idx < num_combine_tasks;
       task_idx += kNumMPs * kNumWarps) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
    __musa_loop_transparent_outermost();
#endif
    const uint32_t token_idx          = task_idx / kNumHiddenWarpTiles;
    const uint32_t hidden_tile_idx    = task_idx - token_idx * kNumHiddenWarpTiles;
    auto*          dst_token          = y + static_cast<uint64_t>(token_idx) * kHidden;
    const auto     topk_idx_token     = input_topk_idx_buffer.get_data_buffer(token_idx);
    const auto     topk_weights_token = input_topk_weights_buffer.get_data_buffer(token_idx);
    const auto*    topk_idx_ptr       = topk_idx_token.get_base_ptr<int64_t>();
    const auto*    topk_weights_ptr   = topk_weights_token.get_base_ptr<float>();

    const uint32_t uint4_idx                  = hidden_tile_idx * kNumUint4PerWarpTile + lane_idx;
    float          reduced[kNumElemsPerUint4] = {};
#pragma unroll
    for (uint32_t slot_idx = 0; slot_idx < kNumTopk; ++slot_idx) {
      const int64_t expert_idx = __ldg(topk_idx_ptr + slot_idx);
      if (expert_idx < 0) {
        continue;
      }
      const float route_weight = __ldg(topk_weights_ptr + slot_idx);
      const auto  src_token    = combine_token_buffer.get_rank_buffer(slot_idx).get_data_buffer(token_idx);
      const auto* src_ptr      = reinterpret_cast<const int4*>(src_token.get_base_ptr()) + uint4_idx;
      const int4  packed       = __lsu_ld_cache_hint(src_ptr, 3, 3, 0, 0);
      const auto* bf16_values  = reinterpret_cast<const mutlass::bfloat16_t*>(&packed);

#pragma unroll
      for (uint32_t i = 0; i < kNumElemsPerUint4; ++i) reduced[i] += static_cast<float>(bf16_values[i]) * route_weight;
    }

    int4  casted;
    auto* casted_bf16 = reinterpret_cast<mutlass::bfloat16_t*>(&casted);
#pragma unroll
    for (uint32_t i = 0; i < kNumElemsPerUint4; ++i) casted_bf16[i] = static_cast<mutlass::bfloat16_t>(reduced[i]);

    auto* out_ptr = reinterpret_cast<int4*>(dst_token) + uint4_idx;
    __lsu_st_cache_hint(out_ptr, casted, 4, 2, 1, 1);
  }
}

}  // namespace mega_moe
