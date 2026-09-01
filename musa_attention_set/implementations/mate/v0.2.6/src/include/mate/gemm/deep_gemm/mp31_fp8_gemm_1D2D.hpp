#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/atom/copy_atom.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <type_traits>

#include "mate/attention/fmha/pipeline_ws.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/gemm/deep_gemm/gemm_type.hpp"
#include "mate/gemm/deep_gemm/mp31_scheduler.hpp"
#include "mate/gemm/deep_gemm/scaling_accumulation.hpp"

namespace mate::deep_gemm {

using namespace mute;

// Mp31 FP8 GEMM 1D2D
template <GemmType  kType,
          ScaleMode kScaleMode,
          typename ElementA,
          typename ElementB,
          typename ElementD,
          typename StrideA,
          typename StrideB,
          typename StrideD,
          typename StrideSFA,
          typename StrideSFB,
          typename TileShape,
          uint32_t kStages,
          uint32_t kQuantTile,
          uint32_t kNumMmaWarpSquads,
          typename Epilogue = EpilogueGemm>
struct Mp31Fp8Gemm1D2D {
  static constexpr int BlockM = size<0>(TileShape{});
  static constexpr int BlockN = size<1>(TileShape{});
  static constexpr int BlockK = size<2>(TileShape{});

  static_assert(BlockK == int(kQuantTile), "FP8 scale accumulation here assumes BlockK == kQuantTile");

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

  static constexpr int ScaleGranularityM = 1;
  static constexpr int ScaleGranularityN = (BlockN >= int(kQuantTile)) ? int(kQuantTile) : BlockN;
  static constexpr int ScaleMsPerTile    = BlockM / ScaleGranularityM;
  static constexpr int ScaleNsPerTile    = BlockN / ScaleGranularityN;
  static constexpr int NBlocksPerScaleN  = (BlockN >= int(kQuantTile)) ? 1 : (int(kQuantTile) / BlockN);

  static constexpr int NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  static constexpr int NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;

  static constexpr int TilesM = BlockM / int(kQuantTile);
  static constexpr int TilesN = BlockN / int(kQuantTile);

  static constexpr int NumLoadWarpSquads = 1;
  static constexpr int NumMmaWarpSquads  = int(kNumMmaWarpSquads);

  static constexpr int NumMmaThreads = NumMmaWarpSquads * NumThreadsPerWarpSquad;
  static constexpr int NumMmaWarps   = NumMmaThreads / NumThreadsPerWarp;

  static constexpr int WarpsPerWarpSquad = NumThreadsPerWarpSquad / NumThreadsPerWarp;

  static constexpr int SmemAlignmentBytes = 256;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock =
      (NumLoadWarpSquads + NumMmaWarpSquads) * mutlass::NumThreadsPerWarpSquad;

  static constexpr TME::CacheHint TmeAInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeAOuterHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeBInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeBOuterHint = TME::CacheHint::CACHE_NONE;

  static constexpr int BarPerStageRatio = 2;
  using PipelineAB                      = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<kStages, BarPerStageRatio>;
  using PipelineABParams                = typename PipelineAB::Params;
  using PipelineABState                 = typename PipelineAB::PipelineState;

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

  // Arguments / Params
  struct Arguments {
    ElementA const* ptr_a   = nullptr;
    ElementB const* ptr_b   = nullptr;
    float const*    ptr_sfa = nullptr;
    float const*    ptr_sfb = nullptr;
    ElementD*       ptr_d   = nullptr;

    int32_t const* ptr_grouped_layout = nullptr;

    StrideA   stride_a;
    StrideB   stride_b;
    StrideD   stride_d;
    StrideSFA stride_sfa;
    StrideSFB stride_sfb;

    int m = 0, n = 0, d_n = 0, k = 0;
    int num_groups = 1;
    int expected_m = 0;
    int num_mps    = 0;
    int quant_tile = int(kQuantTile);
  };

  struct Params {
    TME_A tme_a;
    TME_B tme_b;

    ElementD*      ptr_d;
    int32_t const* ptr_grouped_layout;

    float const* ptr_sfa;
    float const* ptr_sfb;

    StrideSFA stride_sfa;
    StrideSFB stride_sfb;
    StrideD   stride_d;

    RobustDescriptor desc_sfa;
    RobustDescriptor desc_sfb;

    int m, n, d_n, k;
    int num_groups;
    int expected_m;
    int num_mps;
    int quant_tile;

    int k_tiles_qt;
    int n_tiles_qt;
    int k_blocks;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    const int k_tiles_qt = ceil_div(args.k, args.quant_tile);
    const int n_tiles_qt = ceil_div(args.n, args.quant_tile);
    const int k_blocks   = ceil_div(args.k, BlockK);

    auto  gA    = make_tensor(make_gmem_ptr(args.ptr_a), make_shape(args.m, args.k, args.num_groups), args.stride_a);
    auto  gB    = make_tensor(make_gmem_ptr(args.ptr_b), make_shape(args.n, args.k, args.num_groups), args.stride_b);
    TME_A tme_a = make_tme_copy<TmeAInnerHint, TmeAOuterHint>(MP31_TME_LOAD{}, gA, take<0, 2>(SmemLayoutA{}));
    TME_B tme_b = make_tme_copy<TmeBInnerHint, TmeBOuterHint>(MP31_TME_LOAD{}, gB, take<0, 2>(SmemLayoutB{}));

    auto sfa_layout = make_layout(make_shape(args.m, k_tiles_qt, args.num_groups), args.stride_sfa);
    auto sfb_layout = make_layout(make_shape(n_tiles_qt, k_tiles_qt, args.num_groups), args.stride_sfb);
    auto desc_sfa   = make_robust_desc(args.ptr_sfa, static_cast<size_t>(cosize(sfa_layout)));
    auto desc_sfb   = make_robust_desc(args.ptr_sfb, static_cast<size_t>(cosize(sfb_layout)));

    return Params{tme_a,           tme_b,
                  args.ptr_d,      args.ptr_grouped_layout,
                  args.ptr_sfa,    args.ptr_sfb,
                  args.stride_sfa, args.stride_sfb,
                  args.stride_d,   desc_sfa,
                  desc_sfb,        args.m,
                  args.n,          args.d_n > 0 ? args.d_n : args.n,
                  args.k,          args.num_groups,
                  args.expected_m, args.num_mps,
                  args.quant_tile, k_tiles_qt,
                  n_tiles_qt,      k_blocks};
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

    const uint32_t m          = uint32_t(params.m);
    const uint32_t n          = uint32_t(params.n);
    const uint32_t d_n        = uint32_t(params.d_n);
    const uint32_t num_groups = uint32_t(params.num_groups);

    auto mA      = params.tme_a.get_tme_tensor(make_shape(params.m, params.k, params.num_groups));
    auto gA_full = local_tile(mA, make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<1>{}), make_coord(_, _, _));

    auto mB      = params.tme_b.get_tme_tensor(make_shape(params.n, params.k, params.num_groups));
    auto gB_full = local_tile(mB, make_shape(Int<BlockN>{}, Int<BlockK>{}, Int<1>{}), make_coord(_, _, _));

    using Scheduler = detail::Mp31PersistentTileScheduler<kType, BlockM, BlockN, 4>;
    Scheduler scheduler(m, n, num_groups, params.ptr_grouped_layout);

    const uint32_t k_blocks = uint32_t(params.k_blocks);

    PipelineABState pipe_write = mutlass::make_producer_start_state_warpspecialized<PipelineAB>();
    PipelineABState pipe_read;

    PipelineSeqParams seq_params{};
    seq_params.group_size = WarpsPerWarpSquad;
    seq_params.group_id   = is_consumer_squad ? squad - NumLoadWarpSquads : 0;

    constexpr uint32_t kSeqBarrierBase = PipelineAB::NumBarriers;
    PipelineSeq        pipeline_seq(seq_params, kSeqBarrierBase);

    __syncthreads();

    // Producer
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
        uint32_t       m_abs_block = uint32_t(work.M_idx);
        if constexpr (kType == GemmType::MGroupedContiguousWithPsumLayout) {
          m_abs_block = work.m_row_offset / uint32_t(BlockM);
        }

        const auto issue_ab = [&](uint32_t k_block) {
          const int stage = int(pipe_write.index());

          if (is_tme_issue_warp) {
            pipeline.producer_acquire(pipe_write);
            const uint32_t bar_id = pipeline.producer_get_barrier_id(pipe_write);

            Tensor tAgA = cta_tme_a.partition_S(gA_full(_, _, _0{}, m_abs_block, k_block, a_gid));
            Tensor tAsA = cta_tme_a.partition_D(sA(_, _, stage));
            copy(params.tme_a.with(bar_id), tAgA, tAsA);

            Tensor tBgB = cta_tme_b.partition_S(gB_full(_, _, _0{}, n_block_idx, k_block, b_gid));
            Tensor tBsB = cta_tme_b.partition_D(sB(_, _, stage));
            copy(params.tme_b.with(bar_id), tBgB, tBsB);
          }

          ++pipe_write;
        };

        MUTLASS_PRAGMA_NO_UNROLL
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
          issue_ab(kb);
        }

        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    }

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
        uint32_t       m_abs_block = uint32_t(work.M_idx);
        if constexpr (kType == GemmType::MGroupedContiguousWithPsumLayout) {
          m_abs_block = work.m_row_offset / uint32_t(BlockM);
        }

        if (!work.is_compute_valid()) {
          MUTLASS_PRAGMA_NO_UNROLL
          for (uint32_t iter = 0; iter < k_blocks; ++iter) {
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

        using ElementBlockScale = float;
        using ScaleGmemCopyAtom = Copy_Atom<MP31_ROBUST_LOAD<ElementBlockScale>, ElementBlockScale>;

        Tensor mScaleA = make_tensor(make_gmem_ptr(params.ptr_sfa),
                                     make_shape(params.m, params.k_tiles_qt, params.num_groups),
                                     params.stride_sfa);
        Tensor mScaleB = make_tensor(make_gmem_ptr(params.ptr_sfb),
                                     make_shape(params.n_tiles_qt, params.k_tiles_qt, params.num_groups),
                                     params.stride_sfb);

        Tensor         gScaleA_mk = mScaleA(_, _, a_gid);
        Tensor         gScaleB_nk = mScaleB(_, _, b_gid);
        Tensor         gScaleA = local_tile(gScaleA_mk, make_tile(Int<ScaleMsPerTile>{}), make_coord(m_abs_block, _));
        const uint32_t n_scale_idx = n_block_idx / uint32_t(NBlocksPerScaleN);
        Tensor         gScaleB = local_tile(gScaleB_nk, make_tile(Int<ScaleNsPerTile>{}), make_coord(n_scale_idx, _));

        using ScaleAViewAsCLayout =
            Layout<Shape<Shape<Int<ScaleGranularityM>, Int<ScaleMsPerTile>>, Int<BlockN>>, Stride<Stride<_0, _1>, _0>>;
        using ScaleBViewAsCLayout =
            Layout<Shape<Int<BlockM>, Shape<Int<ScaleGranularityN>, Int<ScaleNsPerTile>>>, Stride<_0, Stride<_0, _1>>>;

        Tensor tCgScaleA0 = thr_mma.partition_C(gScaleA(_, 0).compose(ScaleAViewAsCLayout{}));
        Tensor tCgScaleB0 = thr_mma.partition_C(gScaleB(_, 0).compose(ScaleBViewAsCLayout{}));
        Tensor tCrScaleA  = make_tensor_like<ElementBlockScale>(tCgScaleA0);
        Tensor tCrScaleB  = make_tensor_like<ElementBlockScale>(tCgScaleB0);

        auto scale_copy_a = ScaleGmemCopyAtom{}.with(params.desc_sfa);
        auto scale_copy_b = ScaleGmemCopyAtom{}.with(params.desc_sfb);

        using AccumTensor  = decltype(rAcc);
        using ScaleATensor = decltype(tCrScaleA);
        using ScaleBTensor = decltype(tCrScaleB);

        if constexpr (kScaleMode == ScaleMode::Iterative) {
          using IterAccum = ScalingAccumulationIterative<typename AccumTensor::engine_type,
                                                         typename AccumTensor::layout_type,
                                                         typename ScaleATensor::engine_type,
                                                         typename ScaleATensor::layout_type,
                                                         typename ScaleBTensor::engine_type,
                                                         typename ScaleBTensor::layout_type>;

          const uint32_t mma_per_iter = (uint32_t)size<2>(tCrA);
          IterAccum      accumulation(rAcc, mma_per_iter, mma_per_iter, tCrScaleA, tCrScaleB);
          accumulation.initializeA(scale_copy_a, tCgScaleA0, tCrScaleA);
          accumulation.initializeB(scale_copy_b, tCgScaleB0, tCrScaleB);

          auto consumer_iter = [&](uint32_t iter) {
            const uint32_t kqt = iter;

            {
              Tensor tCgScaleA = thr_mma.partition_C(gScaleA(_, kqt).compose(ScaleAViewAsCLayout{}));
              Tensor tCgScaleB = thr_mma.partition_C(gScaleB(_, kqt).compose(ScaleBViewAsCLayout{}));
              accumulation.copyA(scale_copy_a, tCgScaleA, tCrScaleA);
              accumulation.copyB(scale_copy_b, tCgScaleB, tCrScaleB);
            }

            if (accumulation.prepare_if_needed()) {
              accumulation.update_iterationAB(tCrScaleA, tCrScaleB);
              accumulation.div_if_needAB();
            }

            pipeline.consumer_wait(pipe_read);
            pipeline_seq.wait();
            const int stage = int(pipe_read.index());

            MUTE_UNROLL
            for (int kb = 0; kb < size<2>(tCrA); ++kb) {
              mute::gemm(tiled_mma, tCrA(_, _, kb, stage), tCrB(_, _, kb, stage), accumulation());
            }
            mate::warpsquad_commit_batch();
            pipeline_seq.arrive();

            accumulation.advance();

            mate::warpsquad_wait();

            pipeline.consumer_release(pipe_read);
            ++pipe_read;
          };

          MUTLASS_PRAGMA_NO_UNROLL
          for (uint32_t iter = 0; iter < k_blocks; ++iter) {
            consumer_iter(iter);
          }

          accumulation.scale_residue_if_needed(tCrScaleA, tCrScaleB);
        } else {
          using DualBufferAccum =
              ScalingAccumulation<typename AccumTensor::engine_type, typename AccumTensor::layout_type>;

          const uint32_t  mma_per_iter = (uint32_t)size<2>(tCrA);
          DualBufferAccum accumulation(rAcc, mma_per_iter, mma_per_iter);
          tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;

          auto consumer_iter = [&](uint32_t iter) {
            const uint32_t kqt = iter;

            if (accumulation.prepare_if_needed()) {
              tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
            }

            Tensor tCgScaleA     = thr_mma.partition_C(gScaleA(_, kqt).compose(ScaleAViewAsCLayout{}));
            Tensor tCgScaleB     = thr_mma.partition_C(gScaleB(_, kqt).compose(ScaleBViewAsCLayout{}));
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
            for (int kb = 0; kb < size<2>(tCrA); ++kb) {
              mute::gemm(tiled_mma, tCrA(_, _, kb, stage), tCrB(_, _, kb, stage), accumulation());
              tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
            }
            mate::warpsquad_commit_batch();
            pipeline_seq.arrive();

            mate::warpsquad_wait();
            pipeline.consumer_release(pipe_read);
            ++pipe_read;

            accumulation.scale_if_needed(tCrScaleA, tCrScaleB);
          };

          MUTLASS_PRAGMA_NO_UNROLL
          for (uint32_t iter = 0; iter < k_blocks; ++iter) {
            consumer_iter(iter);
          }

          accumulation.scale_residue_if_needed(tCrScaleA, tCrScaleB);
        }

        Tensor mD = make_tensor(
            make_gmem_ptr(params.ptr_d), make_shape(params.m, params.d_n, params.num_groups), params.stride_d);
        uint32_t m0 = m_abs_block * uint32_t(BlockM);
        uint32_t n0 = n_block_idx * uint32_t(BlockN);

        // normal write back
        if constexpr (std::is_same_v<Epilogue, EpilogueGemm>) {
          Tensor gD = local_tile(mD, make_shape(Int<BlockM>{}, Int<BlockN>{}, Int<1>{}), make_coord(_, _, _));
          Tensor tD = gD(_, _, _0{}, m_abs_block, n_block_idx, d_gid);

          auto tCgD = thr_mma.partition_C(tD);
          if (m0 + uint32_t(BlockM) <= m && n0 + uint32_t(BlockN) <= n) {
            MUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rAcc); ++i) {
              tCgD(i) = ElementD(rAcc(i));
            }
          } else {
            auto cD   = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
            auto tCcD = thr_mma.partition_C(cD);
            auto residue_mn =
                make_coord(min(uint32_t(BlockM), m > m0 ? m - m0 : 0u), min(uint32_t(BlockN), n > n0 ? n - n0 : 0u));

            MUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rAcc); ++i) {
              if (elem_less(tCcD(i), residue_mn)) {
                tCgD(i) = ElementD(rAcc(i));
              }
            }
          }
        } else if constexpr (detail::is_epilogue_head_splits_v<Epilogue>) {
          if constexpr (Epilogue::Left == Epilogue::Right && Epilogue::CompactHead % uint32_t(BlockN) == 0) {
            constexpr uint32_t HeadStride  = Epilogue::Left + Epilogue::Mid + Epilogue::Right;
            constexpr uint32_t RightOffset = Epilogue::Left + Epilogue::Mid;

            Tensor mDHead = make_tensor(make_gmem_ptr(params.ptr_d),
                                        make_shape(params.m,
                                                   params.n / int(Epilogue::CompactHead),
                                                   make_shape(Int<Epilogue::Left>{}, Int<2>{}),
                                                   params.num_groups),
                                        make_stride(get<0>(params.stride_d),
                                                    Int<HeadStride>{},
                                                    make_stride(_1{}, Int<RightOffset>{}),
                                                    get<2>(params.stride_d)));
            Tensor gD     = mDHead(_, n_block_idx, _, d_gid);
            Tensor tD     = local_tile(gD,
                                   make_shape(Int<BlockM>{}, make_shape(Int<Epilogue::Left>{}, Int<2>{})),
                                   make_coord(m_abs_block, _0{}));
            auto   tCgD   = thr_mma.partition_C(tD);

            // Use compile-time store chunks to expose accumulator and destination indices
            // as constants during lowering, reducing temporary register pressure and spills
            constexpr int StgChunkSize    = 4;
            constexpr int NumAccumulators = size(rAcc);
            static_assert(NumAccumulators % StgChunkSize == 0,
                          "The head-split epilogue requires complete accumulator store batches");
            constexpr int NumStoreChunks = NumAccumulators / StgChunkSize;

            if (m0 + uint32_t(BlockM) <= m) {
              for_each(make_seq<NumStoreChunks>{}, [&](auto batch) {
                for_each(make_seq<StgChunkSize>{}, [&](auto item) {
                  constexpr int i = int(batch) * StgChunkSize + int(item);
                  tCgD(i)         = ElementD(rAcc(i));
                });
              });
            } else {
              auto cD   = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
              auto tCcD = thr_mma.partition_C(cD);

              for_each(make_seq<NumStoreChunks>{}, [&](auto batch) {
                for_each(make_seq<StgChunkSize>{}, [&](auto item) {
                  constexpr int  i   = int(batch) * StgChunkSize + int(item);
                  const uint32_t row = m0 + uint32_t(get<0>(tCcD(i)));
                  if (row < m) {
                    tCgD(i) = ElementD(rAcc(i));
                  }
                });
              });
            }
          } else {
            auto cD   = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
            auto tCcD = thr_mma.partition_C(cD);

            if (m0 + uint32_t(BlockM) <= m && n0 + uint32_t(BlockN) <= n) {
              MUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(rAcc); ++i) {
                const uint32_t row         = m0 + uint32_t(get<0>(tCcD(i)));
                const uint32_t compact_col = n0 + uint32_t(get<1>(tCcD(i)));
                const uint32_t full_col    = Epilogue::apply_index_n(compact_col);
                mD(row, full_col, d_gid)   = ElementD(rAcc(i));
              }
            } else {
              auto residue_mn =
                  make_coord(min(uint32_t(BlockM), m > m0 ? m - m0 : 0u), min(uint32_t(BlockN), n > n0 ? n - n0 : 0u));

              MUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(rAcc); ++i) {
                if (elem_less(tCcD(i), residue_mn)) {
                  const uint32_t row         = m0 + uint32_t(get<0>(tCcD(i)));
                  const uint32_t compact_col = n0 + uint32_t(get<1>(tCcD(i)));
                  const uint32_t full_col    = Epilogue::apply_index_n(compact_col);
                  mD(row, full_col, d_gid)   = ElementD(rAcc(i));
                }
              }
            }
          }
        } else {
          static_assert(std::is_same_v<Epilogue, EpilogueGemm>, "Unsupported epilogue type");
        }

        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    }
  }
};

}  // namespace mate::deep_gemm
