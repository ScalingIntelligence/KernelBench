#pragma once

#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/pipeline/mp31_pipeline.hpp>

#include "../../attention/fmha/pipeline_ws.hpp"
#include "../../common/mma_mp31_sqmma.hpp"

namespace mutlass {
template <>
struct NumericArrayConverter<float, bfloat16_t, 4, FloatRoundStyle::round_to_nearest> {
  using result_type = Array<float, 4>;
  using source_type = Array<bfloat16_t, 4>;

  MUTLASS_DEVICE
  static result_type convert(source_type const& src) {
    __mt_bfloat164 b0123 = reinterpret_cast<__mt_bfloat164 const&>(src);
    float4         f0123 = __bfloat1642float4(b0123);

    result_type dst;
    dst[0] = f0123.x;
    dst[1] = f0123.y;
    dst[2] = f0123.z;
    dst[3] = f0123.w;
    return dst;
  }

  MUTLASS_DEVICE
  result_type operator()(source_type const& s) const {
    return convert(s);
  }
};
}  // namespace mutlass

namespace mate::deep_gemm {

using namespace mute;

namespace detail {

template <int FragmentSize, class EngineSrc, typename Layout, class EngineDst>
MUTE_DEVICE void convert_op(Tensor<EngineSrc, Layout> const& src, Tensor<EngineDst, Layout>& dst) {
  using SrcType = typename EngineSrc::value_type;
  using DstType = typename EngineDst::value_type;

  static_assert(MUTE_STATIC_V(size(src)) % FragmentSize == 0);

  Tensor src_frg = recast<mutlass::Array<SrcType, FragmentSize> const>(src);
  Tensor dst_frg = recast<mutlass::Array<DstType, FragmentSize>>(dst);
  static_assert(size(src_frg) == size(dst_frg));

  mutlass::NumericArrayConverter<DstType, SrcType, FragmentSize, mutlass::FloatRoundStyle::round_to_nearest>
      convert_impl;

  MUTE_UNROLL
  for (int i = 0; i < size(src_frg); ++i) {
    dst_frg(i) = convert_impl(src_frg(i));
  }
}

}  // namespace detail

template <typename ElementA_,
          typename ElementB_,
          typename ElementD_,
          typename StrideA_,
          typename StrideB_,
          typename StrideD_,
          typename StrideSqrSum_,
          typename TileShape_,
          uint32_t kStages_,
          uint32_t kNumMmaWarpSquads_>
struct Mp31Tf32HcPrenormGemm {
  using ElementA     = ElementA_;
  using ElementB     = ElementB_;
  using ElementD     = ElementD_;
  using StrideA      = StrideA_;
  using StrideB      = StrideB_;
  using StrideD      = StrideD_;
  using StrideSqrSum = StrideSqrSum_;
  using TileShape    = TileShape_;

  static constexpr uint32_t kStages           = kStages_;
  static constexpr uint32_t kNumMmaWarpSquads = kNumMmaWarpSquads_;

  static constexpr int BlockM = get<0>(TileShape{});
  static constexpr int BlockN = get<1>(TileShape{});
  static constexpr int BlockK = get<2>(TileShape{});

  static constexpr int AlignmentA  = 32 / sizeof_bits_v<ElementA>;
  static constexpr int AlignmentB  = 32 / sizeof_bits_v<ElementB>;
  static constexpr int AlignmentTF = 32 / sizeof_bits_v<mutlass::tfloat32_t>;

  using ElementAccumulator = float;

  using TileShapeAA     = Shape<Int<BlockM>, Int<BlockM>, Int<BlockK>>;
  using CollectiveMmaAA = typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                                                mutlass::arch::OpClassTensorOp,
                                                                                ElementA,
                                                                                StrideA,
                                                                                AlignmentA,
                                                                                ElementA,
                                                                                StrideA,
                                                                                AlignmentA,
                                                                                ElementAccumulator,
                                                                                TileShapeAA,
                                                                                Shape<_1, _1, _1>,
                                                                                Int<kStages>,
                                                                                mutlass::gemm::KernelTme>::CollectiveOp;

  using CollectiveMmaAB = typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                                                mutlass::arch::OpClassTensorOp,
                                                                                mutlass::tfloat32_t,
                                                                                StrideA,
                                                                                AlignmentTF,
                                                                                mutlass::tfloat32_t,
                                                                                StrideB,
                                                                                AlignmentB,
                                                                                ElementAccumulator,
                                                                                TileShape,
                                                                                Shape<_1, _1, _1>,
                                                                                Int<kStages>,
                                                                                mutlass::gemm::KernelTme>::CollectiveOp;

  using TiledMmaAA     = typename CollectiveMmaAA::TiledMma;
  using SmemLayoutAA_A = typename CollectiveMmaAA::SmemLayoutA;
  using SmemLayoutAA_B = typename CollectiveMmaAA::SmemLayoutB;

  using TiledMmaAB     = typename CollectiveMmaAB::TiledMma;
  using SmemLayoutAB_A = typename CollectiveMmaAB::SmemLayoutA;
  using SmemLayoutB    = typename CollectiveMmaAB::SmemLayoutB;

  using TME_A = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
      take<0, 2>(SmemLayoutAA_A{})));

  using TME_B = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
      take<0, 2>(SmemLayoutB{})));

  static constexpr int TmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutAA_A{})) * sizeof_bits_v<ElementA>);
  static constexpr int TmeTransactionBytesB =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);

  static constexpr uint32_t NumLoadWarpSquads = 1;
  static constexpr uint32_t NumSqrsumSquads   = 1;
  static constexpr uint32_t NumGemmSquads     = kNumMmaWarpSquads - NumSqrsumSquads;
  static_assert(NumGemmSquads >= 1, "kNumMmaWarpSquads must be >= 2");

  static constexpr uint32_t NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  static constexpr uint32_t NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;
  static constexpr uint32_t WarpsPerWarpSquad      = NumThreadsPerWarpSquad / NumThreadsPerWarp;

  static constexpr uint32_t NumCastWarps         = NumSqrsumSquads * WarpsPerWarpSquad;
  static constexpr uint32_t NumConsumerWarpsA    = NumSqrsumSquads * WarpsPerWarpSquad;
  static constexpr uint32_t NumConsumerWarpsB    = NumGemmSquads * WarpsPerWarpSquad;
  static constexpr uint32_t NumConsumerWarpsMain = NumConsumerWarpsA + NumConsumerWarpsB;
  static constexpr uint32_t MaxThreadsPerBlock =
      (NumLoadWarpSquads + NumSqrsumSquads + NumGemmSquads) * NumThreadsPerWarpSquad;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int      SmemAlignmentBytes         = 256;

  static constexpr uint32_t BarRatio = 4;
  using PipelineMain                 = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<kStages, BarRatio>;
  using PipelineCast                 = mutlass::Mp31PipelineAsyncWarpSpecialized<kStages, BarRatio>;
  using PipelineState                = typename PipelineMain::PipelineState;
  using PipelineCastState            = typename PipelineCast::PipelineState;

  static constexpr int kCastFragmentSize  = 4;
  static constexpr int kCastThreadsPerRow = BlockK / kCastFragmentSize;
  static constexpr int kS2RFragmentBits   = sizeof_bits_v<ElementA> * kCastFragmentSize;
  using CastThreadLayout = Layout<Shape<Int<NumThreadsPerWarpSquad / kCastThreadsPerRow>, Int<kCastThreadsPerRow>>,
                                  Stride<Int<kCastThreadsPerRow>, _1>>;
  using R2SFragmentType  = mute::uint_bit_t<sizeof_bits_v<float> * kCastFragmentSize>;
  // Keep S2R lowered as 64-bit LMA loads; mute::uint_bit_t<64> is scalar uint64_t here.
  using S2RFragmentType = mute::uint32_t __attribute__((vector_size(kS2RFragmentBits / 8)));
  using R2SCopyAtom     = Copy_Atom<UniversalCopy<R2SFragmentType>, float>;
  using R2STiledCopy =
      decltype(make_tiled_copy(R2SCopyAtom{}, CastThreadLayout{}, Layout<Shape<_1, Int<kCastFragmentSize>>>{}));
  using S2RCopyAtom = Copy_Atom<UniversalCopy<S2RFragmentType>, ElementA>;
  using S2RTiledCopy =
      decltype(make_tiled_copy(S2RCopyAtom{}, CastThreadLayout{}, Layout<Shape<_1, Int<kCastFragmentSize>>>{}));

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineMain[PipelineMain::NumBarriers];
    uint8_t PipelineCast[PipelineCast::NumBarriers];
  };

  struct SharedStorage {
    mute::array_aligned<ElementA, cosize_v<SmemLayoutAA_A>, 256> smem_a_bf16;
    mute::array_aligned<float, cosize_v<SmemLayoutAB_A>, 256>    smem_a_fp32;
    mute::array_aligned<ElementB, cosize_v<SmemLayoutB>, 256>    smem_b;
  };

  struct Arguments {
    ElementA const* ptr_a;
    ElementB const* ptr_b;
    ElementD*       ptr_d;
    float*          ptr_sqr_sum;
    StrideA         stride_a;
    StrideB         stride_b;
    StrideD         stride_d;
    StrideSqrSum    stride_sqr_sum;
    int             m, n, k;
    int             num_splits;
  };

  struct Params {
    TME_A        tme_a;
    TME_B        tme_b;
    ElementD*    ptr_d;
    float*       ptr_sqr_sum;
    StrideD      stride_d;
    StrideSqrSum stride_sqr_sum;
    int          m, n, k;
    int          num_splits;
    int          num_m_blocks;
    int          num_n_blocks;
    int          num_k_blocks;
    int          num_k_blocks_per_split;
    int          remain_k_blocks;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    auto  gA    = make_tensor(make_gmem_ptr(args.ptr_a), make_shape(args.m, args.k), args.stride_a);
    auto  gB    = make_tensor(make_gmem_ptr(args.ptr_b), make_shape(args.n, args.k), args.stride_b);
    TME_A tme_a = make_tme_copy(MP31_TME_LOAD{}, gA, take<0, 2>(SmemLayoutAA_A{}));
    TME_B tme_b = make_tme_copy(MP31_TME_LOAD{}, gB, take<0, 2>(SmemLayoutB{}));

    const int num_splits             = max(args.num_splits, 1);
    const int num_m_blocks           = (args.m + BlockM - 1) / BlockM;
    const int num_n_blocks           = (args.n + BlockN - 1) / BlockN;
    const int num_k_blocks           = (args.k + BlockK - 1) / BlockK;
    const int num_k_blocks_per_split = num_k_blocks / num_splits;
    const int remain_k_blocks        = num_k_blocks % num_splits;

    return Params{
        tme_a,
        tme_b,
        args.ptr_d,
        args.ptr_sqr_sum,
        args.stride_d,
        args.stride_sqr_sum,
        args.m,
        args.n,
        args.k,
        num_splits,
        num_m_blocks,
        num_n_blocks,
        num_k_blocks,
        num_k_blocks_per_split,
        remain_k_blocks,
    };
  }

  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));

  static dim3 get_grid_shape(Params const& p) {
    return dim3(uint32_t(p.num_splits), uint32_t(p.num_n_blocks), uint32_t(p.num_m_blocks));
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* bs = reinterpret_cast<BarrierStorage*>(0);

    typename PipelineMain::Params main_params;
    main_params.transaction_bytes = TmeTransactionBytesA + TmeTransactionBytesB;
    main_params.num_consumers     = NumConsumerWarpsMain;
    main_params.num_producers     = 1;
    PipelineMain pipeline_main(main_params, reinterpret_cast<uint64_t>(&bs->PipelineMain));

    typename PipelineCast::Params pipeline_cast_params;
    pipeline_cast_params.producer_arv_count = NumCastWarps;
    pipeline_cast_params.consumer_arv_count = NumConsumerWarpsB;
    PipelineCast pipeline_cast(pipeline_cast_params, reinterpret_cast<uint64_t>(&bs->PipelineCast));

    const int squad         = mutlass::canonical_warp_squad_idx();
    const int warp_idx      = mutlass::canonical_warp_idx_sync();
    const int warp_in_squad = warp_idx % int(WarpsPerWarpSquad);

    const bool is_producer_squad = (squad == 0);
    const bool is_sqrsum_squad   = (squad == int(NumLoadWarpSquads));
    const bool is_gemm_squad     = (squad > int(NumLoadWarpSquads));
    const bool is_tme_warp       = is_producer_squad && (warp_in_squad == 0);

    const int k_split_idx = int(blockIdx.x);
    const int n_block_idx = int(blockIdx.y);
    const int m_block_idx = int(blockIdx.z);
    const int m_offset    = m_block_idx * BlockM;
    const int n_offset    = n_block_idx * BlockN;

    const int k_blocks_this_split = params.num_k_blocks_per_split + (k_split_idx < params.remain_k_blocks ? 1 : 0);
    const int k_block_start = k_split_idx * params.num_k_blocks_per_split + min(k_split_idx, params.remain_k_blocks);

    Tensor sA_bf16_A = make_tensor(make_smem_ptr(shared.smem_a_bf16.data()), SmemLayoutAA_A{});
    Tensor sA_bf16_B = make_tensor(make_smem_ptr(shared.smem_a_bf16.data()), SmemLayoutAA_B{});
    Tensor sA_fp32 =
        make_tensor(make_smem_ptr(reinterpret_cast<mutlass::tfloat32_t*>(shared.smem_a_fp32.data())), SmemLayoutAB_A{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<mutlass::tfloat32_t*>(shared.smem_b.data())), SmemLayoutB{});

    auto   cta_tme_a = params.tme_a.get_slice(0);
    auto   cta_tme_b = params.tme_b.get_slice(0);
    Tensor mA        = params.tme_a.get_tme_tensor(make_shape(params.m, params.k));
    Tensor gA_full   = local_tile(mA, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(_, _));
    Tensor mB        = params.tme_b.get_tme_tensor(make_shape(params.n, params.k));
    Tensor gB_full   = local_tile(mB, make_shape(Int<BlockN>{}, Int<BlockK>{}), make_coord(_, _));

    PipelineState     pipe_write = mutlass::make_producer_start_state_warpspecialized<PipelineMain>();
    PipelineState     pipe_read;
    PipelineCastState pipe_cast_write = mutlass::make_producer_start_state_warpspecialized<PipelineCast>();
    PipelineCastState pipe_cast_read;

    __syncthreads();

    if (is_producer_squad) {
      if (is_tme_warp) {
        for (int kb = 0; kb < k_blocks_this_split; ++kb) {
          const int gkb = k_block_start + kb;

          pipeline_main.producer_acquire(pipe_write);

          const uint32_t bar_id = pipeline_main.producer_get_barrier_id(pipe_write);
          const int      stage  = int(pipe_write.index());

          auto tAgA = cta_tme_a.partition_S(gA_full(_, _, m_block_idx, gkb));
          auto tAsA = cta_tme_a.partition_D(sA_bf16_A(_, _, stage));
          copy(params.tme_a.with(bar_id), tAgA, tAsA);

          auto tBgB = cta_tme_b.partition_S(gB_full(_, _, n_block_idx, gkb));
          auto tBsB = cta_tme_b.partition_D(sB(_, _, stage));
          copy(params.tme_b.with(bar_id), tBgB, tBsB);

          ++pipe_write;
        }
      }
      return;
    }

    if (is_sqrsum_squad) {
      const int tid_sq = int(threadIdx.x) - int(NumLoadWarpSquads * NumThreadsPerWarpSquad);

      TiledMmaAA tiled_mma_aa;
      auto       thr_mma_aa = tiled_mma_aa.get_thread_slice(tid_sq);
      Tensor     tCsAA_A    = thr_mma_aa.partition_A(sA_bf16_A);
      Tensor     tCsAA_B    = thr_mma_aa.partition_B(sA_bf16_B);
      Tensor     tCrAA_A    = thr_mma_aa.make_fragment_A(tCsAA_A);
      Tensor     tCrAA_B    = thr_mma_aa.make_fragment_B(tCsAA_B);

      S2RTiledCopy tiled_copy_s2r;
      R2STiledCopy tiled_copy_r2s;
      auto         thr_copy_s2r = tiled_copy_s2r.get_thread_slice(tid_sq);
      auto         thr_copy_r2s = tiled_copy_r2s.get_thread_slice(tid_sq);

      auto rA_bf16      = make_fragment_like(thr_copy_s2r.partition_S(sA_bf16_A(_, _, _0{})));
      auto rA_bf16_view = thr_copy_s2r.retile_D(rA_bf16);
      auto rA_fp32      = make_fragment_like<float>(rA_bf16);
      auto rA_fp32_view = thr_copy_r2s.retile_S(rA_fp32);

      auto rAcc_AA = partition_fragment_C(tiled_mma_aa, make_shape(Int<BlockM>{}, Int<BlockM>{}));
      clear(rAcc_AA);

      const int valid_m = min(BlockM, params.m - m_offset);

      for (int kb = 0; kb < k_blocks_this_split; ++kb) {
        pipeline_main.consumer_wait(pipe_read);
        const int stage_main = int(pipe_read.index());

        MUTE_UNROLL
        for (int mma_k = 0; mma_k < size<2>(tCrAA_A); ++mma_k) {
          mute::gemm(tiled_mma_aa, tCrAA_A(_, _, mma_k, stage_main), tCrAA_B(_, _, mma_k, stage_main), rAcc_AA);
        }
        mate::warpsquad_commit_batch();

        pipeline_cast.producer_acquire(pipe_cast_write);
        const int stage_cast = int(pipe_cast_write.index());

        auto sA_bf16       = sA_bf16_A(_, _, stage_main);
        auto sA_fp32_stage = sA_fp32(_, _, stage_cast);
        auto tCsA_bf16     = thr_copy_s2r.partition_S(sA_bf16);
        auto tCsA_fp32     = thr_copy_r2s.partition_D(sA_fp32_stage);
        copy(tiled_copy_s2r, tCsA_bf16, rA_bf16_view);

        detail::convert_op<kCastFragmentSize>(rA_bf16, rA_fp32);
        mate::warpsquad_wait();
        copy(tiled_copy_r2s, rA_fp32_view, tCsA_fp32);

        __syncwarp();
        pipeline_cast.producer_commit(pipe_cast_write);
        pipeline_main.consumer_release(pipe_read);
        ++pipe_cast_write;
        ++pipe_read;
      }

      if (n_block_idx == 0) {
        auto       cId_AA    = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockM>{}));
        auto       tCc_AA    = thr_mma_aa.partition_C(cId_AA);
        auto       mS        = make_tensor(make_gmem_ptr(params.ptr_sqr_sum),
                              make_shape(params.num_splits, params.m),
                              params.stride_sqr_sum)(k_split_idx, _);
        Tensor     gS        = local_tile(mS, Shape<Int<BlockM>>{}, make_coord(m_block_idx));
        const auto residue_m = make_coord(min(BlockM, params.m - m_offset));

        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(rAcc_AA); ++i) {
          const int mi = int(get<0>(tCc_AA(i)));
          const int ni = int(get<1>(tCc_AA(i)));
          if (mi == ni && elem_less(make_coord(mi), residue_m)) {
            gS(mi) = rAcc_AA(i);
          }
        }
      }
      return;
    }

    if (is_gemm_squad) {
      const int  mma_thread_offset = int((NumLoadWarpSquads + NumSqrsumSquads) * NumThreadsPerWarpSquad);
      TiledMmaAB tiled_mma;
      auto       thr_mma = tiled_mma.get_thread_slice(int(threadIdx.x) - mma_thread_offset);

      auto rAcc = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
      clear(rAcc);

      Tensor tCsA = thr_mma.partition_A(sA_fp32);
      Tensor tCsB = thr_mma.partition_B(sB);
      Tensor tCrA = thr_mma.make_fragment_A(tCsA);
      Tensor tCrB = thr_mma.make_fragment_B(tCsB);

      for (int kb = 0; kb < k_blocks_this_split; ++kb) {
        pipeline_main.consumer_wait(pipe_read);
        pipeline_cast.consumer_wait(pipe_cast_read);

        const int stage_a = int(pipe_cast_read.index());
        const int stage_b = int(pipe_read.index());

        MUTE_UNROLL
        for (int mma_k = 0; mma_k < size<2>(tCrA); ++mma_k) {
          mute::gemm(tiled_mma, tCrA(_, _, mma_k, stage_a), tCrB(_, _, mma_k, stage_b), rAcc);
        }
        mate::warpsquad_commit_batch();
        mate::warpsquad_wait();

        pipeline_cast.consumer_release(pipe_cast_read);
        pipeline_main.consumer_release(pipe_read);
        ++pipe_cast_read;
        ++pipe_read;
      }

      auto mD =
          make_tensor(make_gmem_ptr(params.ptr_d), make_shape(params.num_splits, params.m, params.n), params.stride_d)(
              k_split_idx, _, _);
      Tensor gD   = local_tile(mD, take<0, 2>(TileShape{}), make_coord(m_block_idx, n_block_idx));
      auto   tCgD = thr_mma.partition_C(gD);
      auto   cD   = make_identity_tensor(take<0, 2>(TileShape{}));
      auto   tCcD = thr_mma.partition_C(cD);

      const auto residue_mn = make_coord(min(BlockM, params.m - m_offset), min(BlockN, params.n - n_offset));

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rAcc); ++i) {
        if (elem_less(tCcD(i), residue_mn)) {
          tCgD(i) = ElementD(rAcc(i));
        }
      }
    }
  }
};

}  // namespace mate::deep_gemm
