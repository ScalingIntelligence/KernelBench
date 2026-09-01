#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "mate/attention/fmha/pipeline_ws.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/gemm/deep_gemm/gemm_type.hpp"
#include "mate/gemm/deep_gemm/mp31_scheduler.hpp"

namespace mate::deep_gemm {

using namespace mute;

template <GemmType kType,
          typename ElementA,
          typename ElementB,
          typename ElementD,
          typename StrideA,
          typename StrideB,
          typename StrideD,
          typename TileShape,
          uint32_t kStages,
          uint32_t kNumMmaWarpSquads>
struct Mp31Bf16Gemm {
  static constexpr int BlockM = size<0>(TileShape{});
  static constexpr int BlockN = size<1>(TileShape{});
  static constexpr int BlockK = size<2>(TileShape{});

  static constexpr int AlignmentA = 32 / sizeof_bits_v<ElementA>;
  static constexpr int AlignmentB = 32 / sizeof_bits_v<ElementB>;

  using ElementAccumulator = float;

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
                                                                              Int<kStages>,
                                                                              mutlass::gemm::KernelTme>::CollectiveOp;

  using TiledMma    = typename CollectiveMma::TiledMma;
  using SmemLayoutA = typename CollectiveMma::SmemLayoutA;
  using SmemLayoutB = typename CollectiveMma::SmemLayoutB;

  static constexpr int NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  static constexpr int NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;
  static constexpr int WarpsPerWarpSquad      = NumThreadsPerWarpSquad / NumThreadsPerWarp;

  static constexpr int NumLoadWarpSquads = 1;
  static constexpr int NumMmaWarpSquads  = int(kNumMmaWarpSquads);

  static_assert(int(kNumMmaWarpSquads) * NumThreadsPerWarpSquad == int(size(TiledMma{})));

  static constexpr int NumMmaThreads = NumMmaWarpSquads * NumThreadsPerWarpSquad;
  static constexpr int NumMmaWarps   = NumMmaThreads / NumThreadsPerWarp;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock =
      (NumLoadWarpSquads + NumMmaWarpSquads) * mutlass::NumThreadsPerWarpSquad;

  static constexpr int SmemAlignmentBytes = 256;

  static constexpr TME::CacheHint TmeAInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeAOuterHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeBInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeBOuterHint = TME::CacheHint::CACHE_NONE;

  using PipelineAB       = mutlass::Mp31PipelineTmeAsync<kStages>;
  using PipelineABParams = typename PipelineAB::Params;
  using PipelineABState  = typename PipelineAB::PipelineState;

  using PipelineSeq       = mutlass::OrderedSequenceBarrier<1, NumMmaWarpSquads>;
  using PipelineSeqParams = typename PipelineSeq::Params;

  static constexpr int TmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  static constexpr int TmeTransactionBytesB =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);
  static constexpr int TmeTransactionBytesAB = TmeTransactionBytesA + TmeTransactionBytesB;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineAB[PipelineAB::NumBarriers];
    uint8_t PipelineSeq[PipelineSeq::NumBarriers];
  };

  struct SharedStorage {
    mute::array_aligned<ElementA, cosize_v<SmemLayoutA>, 256> smem_a;
    mute::array_aligned<ElementB, cosize_v<SmemLayoutB>, 256> smem_b;
  };

  using TME_A = decltype(make_tme_copy<TmeAInnerHint, TmeAOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
      take<0, 2>(SmemLayoutA{})));

  using TME_B = decltype(make_tme_copy<TmeBInnerHint, TmeBOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
      take<0, 2>(SmemLayoutB{})));

  using TensorGroup = detail::GemmTensorGroupTraits<kType>;

  struct Arguments {
    ElementA const* ptr_a = nullptr;
    ElementB const* ptr_b = nullptr;
    ElementD*       ptr_d = nullptr;

    int32_t const* ptr_grouped_layout = nullptr;

    StrideA stride_a;
    StrideB stride_b;
    StrideD stride_d;

    int m          = 0;
    int n          = 0;
    int k          = 0;
    int num_groups = 1;
    int expected_m = 0;
    int num_mps    = 0;
  };

  struct Params {
    TME_A tme_a;
    TME_B tme_b;

    ElementD*      ptr_d;
    int32_t const* ptr_grouped_layout;

    StrideD stride_d;

    int m, n, k;
    int num_groups;
    int expected_m;
    int num_mps;

    int k_blocks;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    const int k_blocks = ceil_div(args.k, BlockK);

    auto  gA    = make_tensor(make_gmem_ptr(args.ptr_a), make_shape(args.m, args.k, args.num_groups), args.stride_a);
    auto  gB    = make_tensor(make_gmem_ptr(args.ptr_b), make_shape(args.n, args.k, args.num_groups), args.stride_b);
    TME_A tme_a = make_tme_copy<TmeAInnerHint, TmeAOuterHint>(MP31_TME_LOAD{}, gA, take<0, 2>(SmemLayoutA{}));
    TME_B tme_b = make_tme_copy<TmeBInnerHint, TmeBOuterHint>(MP31_TME_LOAD{}, gB, take<0, 2>(SmemLayoutB{}));

    return Params{tme_a,
                  tme_b,
                  args.ptr_d,
                  args.ptr_grouped_layout,
                  args.stride_d,
                  args.m,
                  args.n,
                  args.k,
                  args.num_groups,
                  args.expected_m,
                  args.num_mps,
                  k_blocks};
  }

  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));

  static dim3 get_grid_shape_persistent(Params const& p) {
    const int mps = (p.num_mps > 0 ? p.num_mps : 1);
    return dim3(uint32_t(mps), 1u, 1u);
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineABParams pipe_params;
    pipe_params.transaction_bytes = TmeTransactionBytesAB;
    pipe_params.num_consumers     = NumMmaWarps;
    pipe_params.num_producers     = 1;

    PipelineAB pipeline(pipe_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineAB));

    auto cta_tme_a = params.tme_a.get_slice(0);
    auto cta_tme_b = params.tme_b.get_slice(0);

    static constexpr int TotalWarpSquads = NumLoadWarpSquads + NumMmaWarpSquads;

    const int squad             = mutlass::canonical_warp_squad_idx();
    const int warp_idx_in_squad = mutlass::canonical_warp_idx_sync() & (WarpsPerWarpSquad - 1);

    const bool is_producer_squad = (squad == 0);
    const bool is_consumer_squad = (squad >= NumLoadWarpSquads) && (squad < TotalWarpSquads);
    const bool is_tme_issue_warp = is_producer_squad && (warp_idx_in_squad == 0);

    Tensor sA = make_tensor(make_smem_ptr(shared.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared.smem_b.data()), SmemLayoutB{});

    auto mA      = params.tme_a.get_tme_tensor(make_shape(params.m, params.k, params.num_groups));
    auto gA_full = local_tile(mA, make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<1>{}), make_coord(_, _, _));

    auto mB      = params.tme_b.get_tme_tensor(make_shape(params.n, params.k, params.num_groups));
    auto gB_full = local_tile(mB, make_shape(Int<BlockN>{}, Int<BlockK>{}, Int<1>{}), make_coord(_, _, _));

    using Scheduler = mate::deep_gemm::detail::Mp31PersistentTileScheduler<kType, BlockM, BlockN, 4>;
    Scheduler scheduler(static_cast<uint32_t>(params.m),
                        static_cast<uint32_t>(params.n),
                        static_cast<uint32_t>(params.num_groups),
                        params.ptr_grouped_layout,
                        static_cast<uint32_t>(params.expected_m));

    const int k_blocks = params.k_blocks;

    PipelineABState pipe_write = mutlass::make_producer_start_state<PipelineAB>();
    PipelineABState pipe_read;

    PipelineSeqParams seq_params{};
    seq_params.group_size = WarpsPerWarpSquad;
    seq_params.group_id   = is_consumer_squad ? squad - NumLoadWarpSquads : 0;

    constexpr uint32_t kSeqBarrierBase = PipelineAB::NumBarriers;
    PipelineSeq        pipeline_seq(seq_params, kSeqBarrierBase);

    __syncthreads();

    // ----------------------------------------------------------------
    // Producer
    // ----------------------------------------------------------------
    if (is_producer_squad) {
      auto work = scheduler.initial_work_tile_info();

      MUTLASS_PRAGMA_NO_UNROLL
      for (; work.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        const uint32_t m_block_idx = uint32_t(work.M_idx);
        const uint32_t n_block_idx = uint32_t(work.N_idx);
        const uint32_t gid         = uint32_t(work.G_idx);
        const uint32_t a_gid       = TensorGroup::kA ? gid : 0u;
        const uint32_t b_gid       = TensorGroup::kB ? gid : 0u;
        const uint32_t d_gid       = TensorGroup::kD ? gid : 0u;
        const uint32_t m_abs_block = work.m_row_offset / BlockM;

        const auto issue_ab = [&](int k_block) {
          const int stage = int(pipe_write.index());

          if (is_tme_issue_warp) {
            pipeline.producer_acquire(pipe_write);
            const uint32_t bar_id = pipeline.producer_get_barrier_id(pipe_write);

            Tensor tAgA =
                cta_tme_a.partition_S(gA_full(_, _, _0{}, m_abs_block, static_cast<uint32_t>(k_block), a_gid));
            Tensor tAsA = cta_tme_a.partition_D(sA(_, _, stage));
            copy(params.tme_a.with(bar_id), tAgA, tAsA);

            Tensor tBgB =
                cta_tme_b.partition_S(gB_full(_, _, _0{}, n_block_idx, static_cast<uint32_t>(k_block), b_gid));
            Tensor tBsB = cta_tme_b.partition_D(sB(_, _, stage));
            copy(params.tme_b.with(bar_id), tBgB, tBsB);
          }

          ++pipe_write;
        };

        MUTLASS_PRAGMA_NO_UNROLL
        for (int kb = 0; kb < k_blocks; ++kb) {
          issue_ab(kb);
        }

        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    }

    // ----------------------------------------------------------------
    // Consumer
    // ----------------------------------------------------------------
    else if (is_consumer_squad) {
      auto work = scheduler.initial_work_tile_info();

      MUTLASS_PRAGMA_NO_UNROLL
      for (; work.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        const uint32_t m_block_idx = uint32_t(work.M_idx);
        const uint32_t n_block_idx = uint32_t(work.N_idx);
        const uint32_t gid         = uint32_t(work.G_idx);
        const uint32_t a_gid       = TensorGroup::kA ? gid : 0u;
        const uint32_t b_gid       = TensorGroup::kB ? gid : 0u;
        const uint32_t d_gid       = TensorGroup::kD ? gid : 0u;
        const uint32_t m_abs_block = work.m_row_offset / BlockM;

        if (!work.is_compute_valid()) {
          MUTLASS_PRAGMA_NO_UNROLL
          for (int iter = 0; iter < k_blocks; ++iter) {
            pipeline.consumer_wait(pipe_read);
            pipeline_seq.wait();
            pipeline_seq.arrive();
            pipeline.consumer_release(pipe_read);
            ++pipe_read;
          }
          scheduler.advance_to_next_work();
          work = scheduler.get_work_tile_info();
          continue;
        }

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(int(threadIdx.x) - int(NumLoadWarpSquads * NumThreadsPerWarpSquad));

        auto rAcc = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
        fill(rAcc, ElementAccumulator(0));

        Tensor tCsA = thr_mma.partition_A(sA);
        Tensor tCsB = thr_mma.partition_B(sB);
        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);

        MUTLASS_PRAGMA_NO_UNROLL
        for (int iter = 0; iter < k_blocks; ++iter) {
          pipeline.consumer_wait(pipe_read);
          pipeline_seq.wait();
          const int stage = int(pipe_read.index());

          MUTE_UNROLL
          for (int kb = 0; kb < size<2>(tCrA); ++kb) {
            mute::gemm(tiled_mma, tCrA(_, _, kb, stage), tCrB(_, _, kb, stage), rAcc);
          }
          mate::warpsquad_commit_batch();
          pipeline_seq.arrive();

          mate::warpsquad_wait();
          pipeline.consumer_release(pipe_read);
          ++pipe_read;
        }

        Tensor mD = make_tensor(
            make_gmem_ptr(params.ptr_d), make_shape(params.m, params.n, params.num_groups), params.stride_d);
        Tensor gD = local_tile(mD, make_shape(Int<BlockM>{}, Int<BlockN>{}, Int<1>{}), make_coord(_, _, _));
        Tensor tD = gD(_, _, _0{}, int(m_abs_block), int(n_block_idx), int(d_gid));

        auto     tCgD = thr_mma.partition_C(tD);
        uint32_t m0   = m_block_idx * uint32_t(BlockM);
        uint32_t n0   = n_block_idx * uint32_t(BlockN);
        if (m0 + uint32_t(BlockM) <= uint32_t(params.m) && n0 + uint32_t(BlockN) <= uint32_t(params.n)) {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(rAcc); ++i) {
            tCgD(i) = ElementD(rAcc(i));
          }
        } else {
          auto cD         = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
          auto tCcD       = thr_mma.partition_C(cD);
          auto residue_mn = make_coord(min(uint32_t(BlockM), uint32_t(params.m) > m0 ? uint32_t(params.m) - m0 : 0u),
                                       min(uint32_t(BlockN), uint32_t(params.n) > n0 ? uint32_t(params.n) - n0 : 0u));

          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(rAcc); ++i) {
            if (elem_less(tCcD(i), residue_mn)) {
              tCgD(i) = ElementD(rAcc(i));
            }
          }
        }

        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    }
  }
};

}  // namespace mate::deep_gemm
