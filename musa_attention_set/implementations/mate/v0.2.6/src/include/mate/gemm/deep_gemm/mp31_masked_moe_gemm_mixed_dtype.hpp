#pragma once

#include <musa_burst.h>
#include <mutlass/bfloat16.h>
#include <mutlass/float8.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/algorithm/clear.hpp>
#include <mute/atom/copy_atom.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "mate/attention/fmha/pipeline_ws.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/common/numeric_conversion.hpp"
#include "mate/gemm/deep_gemm/gemm_type.hpp"
#include "mate/gemm/deep_gemm/mp31_scheduler.hpp"

namespace mate::gemm::masked_moe_gemm_mixed_dtype {

using namespace mute;

template <class ElementA_,
          class ElementB_,
          class ElementScaleA_,
          class ElementScaleB_,
          class ElementD_,
          class TileShape_,
          uint32_t Stages_,
          int      ScaleABlockK_ = -1,
          bool     ScaleAMMajor_ = false>
struct Mp31MixedDtypeMaskedMoeGEMMS4FP8 {
  using ElementA      = ElementA_;
  using ElementB      = ElementB_;
  using ElementScaleA = ElementScaleA_;
  using ElementScaleB = ElementScaleB_;
  using ElementMmaA   = mutlass::half_t;
  using ElementD      = ElementD_;
  using TileShape     = TileShape_;

  using StrideA      = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideB      = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideScaleA = conditional_t<ScaleAMMajor_,
                                     decltype(make_stride(_1{}, int64_t{}, int64_t{})),
                                     decltype(make_stride(int64_t{}, _1{}, int64_t{}))>;
  using StrideScaleB = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideD      = decltype(make_stride(int64_t{}, _1{}, int64_t{}));

  static constexpr int  BlockM             = size<0>(TileShape{});
  static constexpr int  BlockN             = size<1>(TileShape{});
  static constexpr int  BlockK             = size<2>(TileShape{});
  static constexpr int  ScaleBlockK        = 128;
  static constexpr int  ScaleBlocksPerTile = BlockK / ScaleBlockK;
  static constexpr int  TmeScaleKBlocks    = 8;
  static constexpr int  TmeScaleAKBlocks   = 4;
  static constexpr int  SqmmasPerScale     = 2;
  static constexpr int  ScaleABlockK       = ScaleABlockK_;
  static constexpr bool IsPerTokenScaleA   = ScaleABlockK == -1;
  static constexpr bool IsGroupwiseScaleA  = ScaleABlockK == ScaleBlockK;
  static constexpr bool IsScaleAMMajor     = ScaleAMMajor_;
  static constexpr int  Stages             = int(Stages_);

  static_assert(BlockK == 256, "This kernel requires a K256 tile");
  static_assert(ScaleBlocksPerTile == 2, "Each K256 mainloop tile must contain two K128 weight-scale blocks");
  static_assert(TmeScaleKBlocks * int(sizeof(ElementScaleB)) >= 16,
                "Scale-B TME loads require at least 16 contiguous bytes");
  static_assert(IsPerTokenScaleA || IsGroupwiseScaleA,
                "Scale-A K block must be per-token or match the K128 scale block");
  static_assert(!IsScaleAMMajor || IsGroupwiseScaleA, "M-major Scale-A is only supported for grouped quantization");
  static_assert(Stages >= 2 && Stages <= 3, "The mixed S4/FP16 K256 pipeline supports 2 or 3 stages");
  static_assert(is_same_v<ElementA, mutlass::float_e4m3_t>, "The initial caster supports E4M3 activations only");
  static_assert(is_same_v<ElementB, int4_t>, "The B operand must use signed INT4 elements");
  static_assert(sizeof_bits_v<ElementMmaA> == 16, "The widened A operand must use FP16 elements");
  static_assert(sizeof_bits_v<ElementScaleA> == 32, "A scales must use FP32 elements");
  static_assert(is_same_v<ElementScaleB, mutlass::bfloat16_t>, "B scales must use BF16 elements");

  static constexpr TCE::Major SqmmaMajorA = TCE::Major::K;
  static constexpr TCE::Major SqmmaMajorB = TCE::Major::K;

  using SqmmaOp =
      decltype(MP31::SQMMA::
                   ss_op_selector<ElementMmaA, ElementB, float, TileShape, SqmmaMajorA, SqmmaMajorB, _32, _128>());
  using AtomLayout = decltype(make_layout(Shape<_1, _2, _1>{}, LayoutRight{}));
  using TiledMma   = decltype(make_tiled_mma(SqmmaOp{}, AtomLayout{}));

  using SmemLayoutAtomMmaA =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_A<SqmmaMajorA, ElementMmaA, SqmmaOp, TileShape>());
  using SmemLayoutAtomB =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_B<SqmmaMajorB, ElementB, SqmmaOp, TileShape>());
  using SmemLayoutMmaA =
      decltype(tile_to_shape(SmemLayoutAtomMmaA{}, make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<Stages>{})));
  using SmemLayoutB =
      decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<BlockN>{}, Int<BlockK>{}, Int<Stages>{})));

  using SmemLayoutAtomA = Layout<Shape<Int<BlockM>, Int<BlockK>, _2>, Stride<Int<BlockK>, _1, Int<BlockM * BlockK>>>;
  using SmemLayoutAStorage =
      decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<BlockM>{}, Int<BlockK>{}, _2{}, Int<Stages>{})));
  using SmemLayoutA = decltype(SmemLayoutAStorage{}(_, _, _0{}, _));

  static constexpr TME::CacheHint TmeAInnerHint      = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeAOuterHint      = TME::CacheHint::CACHE_PERSIST;
  static constexpr TME::CacheHint TmeBInnerHint      = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeBOuterHint      = TME::CacheHint::CACHE_NONE;
  static constexpr TME::CacheHint TmeScaleAInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeScaleAOuterHint = TME::CacheHint::CACHE_PERSIST;
  static constexpr TME::CacheHint TmeScaleBInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeScaleBOuterHint = TME::CacheHint::CACHE_PERSIST;

  using TME_A = decltype(make_tme_copy<TmeAInnerHint, TmeAOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), repeat_like(StrideA{}, int64_t(0)), StrideA{}),
      take<0, 2>(SmemLayoutA{})));

  using TME_B = decltype(make_tme_copy<TmeBInnerHint, TmeBOuterHint, uint8_t>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr<ElementB>(nullptr), repeat_like(StrideB{}, int64_t(0)), StrideB{}),
      take<0, 2>(SmemLayoutB{}),
      make_shape(Int<BlockN>{}, Int<BlockK>{})));

  static constexpr int      SmemAlignmentBytes           = 256;
  static constexpr uint32_t MinBlocksPerMultiprocessor   = 1;
  static constexpr uint32_t Num1DBlocksPerSchedulerGroup = 4;

  static constexpr uint32_t NumProducerWarpSquads  = 1;
  static constexpr uint32_t NumCasterWarpSquads    = 1;
  static constexpr uint32_t NumConsumerWarpSquads  = 2;
  static constexpr uint32_t NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  static constexpr uint32_t NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;
  static constexpr uint32_t WarpsPerWarpSquad      = NumThreadsPerWarpSquad / NumThreadsPerWarp;
  static constexpr uint32_t NumCasterThreads       = NumCasterWarpSquads * NumThreadsPerWarpSquad;
  static constexpr uint32_t MaxThreadsPerBlock =
      (NumProducerWarpSquads + NumCasterWarpSquads + NumConsumerWarpSquads) * NumThreadsPerWarpSquad;

  static_assert(int(NumConsumerWarpSquads * NumThreadsPerWarpSquad) == int(size(TiledMma{})),
                "The MMA thread layout must cover both consumer warp squads");

  static constexpr int MaxBarriersPerStage = 2;
  using PipelineBarrierRatio =
      mutlass::Mp31PipelineWarpSpecializedBarrierRatio<MaxBarriersPerStage, 1, Stages, Stages, Stages>;
  static constexpr int BarriersPerStage = PipelineBarrierRatio::value;
  static_assert(BarriersPerStage > 0, "A, B, cast, and Scale-B pipelines exceed the MP31 async-barrier budget");

  using PipelineLoadA       = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<Stages, BarriersPerStage>;
  using PipelineLoadB       = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<Stages, BarriersPerStage>;
  using PipelineCastA       = mutlass::Mp31PipelineAsyncWarpSpecialized<Stages, BarriersPerStage>;
  using PipelineLoadAParams = typename PipelineLoadA::Params;
  using PipelineLoadBParams = typename PipelineLoadB::Params;
  using PipelineCastAParams = typename PipelineCastA::Params;
  using PipelineLoadAState  = typename PipelineLoadA::PipelineState;
  using PipelineLoadBState  = typename PipelineLoadB::PipelineState;
  using PipelineCastAState  = typename PipelineCastA::PipelineState;
  using CasterSyncBarrier   = mutlass::arch::AsyncBarrier;

  static constexpr int TmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  static constexpr int TmeTransactionBytesB =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);
  static constexpr int MaxSharedStorageBytes = 192 * 1024;
  using SmemLayoutScaleA                     = conditional_t<IsScaleAMMajor,
                                                             Layout<Shape<Int<BlockM>, Int<TmeScaleAKBlocks>, Int<Stages>>,
                                                                    Stride<_1, Int<BlockM>, Int<BlockM * TmeScaleAKBlocks>>>,
                                                             Layout<Shape<Int<BlockM>, Int<TmeScaleAKBlocks>, Int<Stages>>,
                                                                    Stride<Int<TmeScaleAKBlocks>, _1, Int<BlockM * TmeScaleAKBlocks>>>>;
  using TME_ScaleA                           = decltype(make_tme_copy<TmeScaleAInnerHint, TmeScaleAOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementScaleA const*>(nullptr)),
                  make_shape(int64_t{}, int64_t{}, int64_t{}),
                  StrideScaleA{}),
      take<0, 2>(SmemLayoutScaleA{}),
      make_shape(Int<BlockM>{}, Int<TmeScaleAKBlocks>{})));
  using SmemLayoutScaleB                     = Layout<Shape<Int<BlockN>, Int<TmeScaleKBlocks>, Int<Stages>>,
                                                      Stride<Int<TmeScaleKBlocks>, _1, Int<BlockN * TmeScaleKBlocks>>>;
  using TME_ScaleB                           = decltype(make_tme_copy<TmeScaleBInnerHint, TmeScaleBOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementScaleB const*>(nullptr)),
                  make_shape(int64_t{}, int64_t{}, int64_t{}),
                  make_stride(int64_t{}, _1{}, int64_t{})),
      take<0, 2>(SmemLayoutScaleB{}),
      make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{})));
  static constexpr int TmeTransactionBytesScaleB =
      mutlass::bits_to_bytes(BlockN * TmeScaleKBlocks * sizeof_bits_v<ElementScaleB>);
  static constexpr int TmeTransactionBytesScaleA =
      IsGroupwiseScaleA ? mutlass::bits_to_bytes(BlockM * TmeScaleAKBlocks * sizeof_bits_v<ElementScaleA>) : 0;

  using Scheduler    = mate::deep_gemm::detail::Mp31PersistentTileScheduler<mate::deep_gemm::GemmType::MGroupedMasked,
                                                                            BlockM,
                                                                            BlockN,
                                                                            Num1DBlocksPerSchedulerGroup>;
  using WorkTileInfo = typename Scheduler::WorkTileInfo;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineLoadA[PipelineLoadA::NumBarriers];
    uint8_t PipelineLoadB[PipelineLoadB::NumBarriers];
    uint8_t PipelineCastA[PipelineCastA::NumBarriers];
    uint8_t CasterSync[1];
  };
  static_assert(sizeof(BarrierStorage) + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount <=
                    mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "Async barrier storage exceeds the MP31 hardware limit");

  struct SharedStorage {
    union {
      MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementA, cosize_v<SmemLayoutA>> smem_a;
      MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementMmaA, cosize_v<SmemLayoutMmaA>> smem_mma_a;
    };
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementB, cosize_v<SmemLayoutB>> smem_b;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementScaleA, cosize_v<SmemLayoutScaleA>> smem_scale_a;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementScaleB, cosize_v<SmemLayoutScaleB>> smem_scale_b;
  };

  struct Arguments {
    ElementA const*      ptr_a        = nullptr;
    ElementScaleA const* ptr_scale_a  = nullptr;
    ElementB const*      ptr_b        = nullptr;
    ElementScaleB const* ptr_scale_b  = nullptr;
    int32_t const*       ptr_masked_m = nullptr;
    ElementD*            ptr_d        = nullptr;

    StrideA      stride_a;
    StrideB      stride_b;
    StrideScaleA stride_scale_a;
    StrideScaleB stride_scale_b;
    StrideD      stride_d;

    int m          = 0;
    int n          = 0;
    int k          = 0;
    int num_groups = 0;
    int expected_m = 0;
    int num_mps    = 0;
  };

  struct Params {
    TME_A                tme_a;
    TME_B                tme_b;
    TME_ScaleA           tme_scale_a;
    TME_ScaleB           tme_scale_b;
    ElementScaleA const* ptr_scale_a;
    RobustDescriptor     desc_scale_a;
    int32_t const*       ptr_masked_m;
    ElementD*            ptr_d;

    StrideScaleA stride_scale_a;
    StrideD      stride_d;

    int m;
    int n;
    int k;
    int num_groups;
    int expected_m;
    int num_mps;
    int k_blocks;
    int scale_a_k_blocks;
    int scale_k_blocks;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    const int k_blocks         = ceil_div(args.k, BlockK);
    const int scale_a_k_blocks = IsGroupwiseScaleA ? ceil_div(args.k, ScaleABlockK) : 1;
    const int scale_k_blocks   = k_blocks * ScaleBlocksPerTile;

    auto  gA    = make_tensor(make_gmem_ptr(args.ptr_a), make_shape(args.m, args.k, args.num_groups), args.stride_a);
    TME_A tme_a = make_tme_copy<TmeAInnerHint, TmeAOuterHint>(MP31_TME_LOAD{}, gA, take<0, 2>(SmemLayoutA{}));

    auto  gB    = make_tensor(make_gmem_ptr<ElementB>(static_cast<void const*>(args.ptr_b)),
                          make_shape(args.n, args.k, args.num_groups),
                          args.stride_b);
    TME_B tme_b = make_tme_copy<TmeBInnerHint, TmeBOuterHint, uint8_t>(
        MP31_TME_LOAD{}, gB, take<0, 2>(SmemLayoutB{}), make_shape(Int<BlockN>{}, Int<BlockK>{}));

    auto scale_a_layout = make_layout(make_shape(args.m, scale_a_k_blocks, args.num_groups), args.stride_scale_a);
    RobustDescriptor desc_scale_a = make_robust_desc(args.ptr_scale_a, static_cast<size_t>(cosize(scale_a_layout)));
    auto             gScaleA      = make_tensor(make_gmem_ptr(args.ptr_scale_a), scale_a_layout);
    TME_ScaleA       tme_scale_a  = make_tme_copy<TmeScaleAInnerHint, TmeScaleAOuterHint>(
        MP31_TME_LOAD{}, gScaleA, take<0, 2>(SmemLayoutScaleA{}), make_shape(Int<BlockM>{}, Int<TmeScaleAKBlocks>{}));

    auto       scale_b_layout = make_layout(make_shape(args.n, scale_k_blocks, args.num_groups), args.stride_scale_b);
    auto       gScaleB        = make_tensor(make_gmem_ptr(args.ptr_scale_b), scale_b_layout);
    TME_ScaleB tme_scale_b    = make_tme_copy<TmeScaleBInnerHint, TmeScaleBOuterHint>(
        MP31_TME_LOAD{}, gScaleB, take<0, 2>(SmemLayoutScaleB{}), make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{}));

    return Params{
        tme_a,
        tme_b,
        tme_scale_a,
        tme_scale_b,
        args.ptr_scale_a,
        desc_scale_a,
        args.ptr_masked_m,
        args.ptr_d,
        args.stride_scale_a,
        args.stride_d,
        args.m,
        args.n,
        args.k,
        args.num_groups,
        args.expected_m,
        args.num_mps,
        k_blocks,
        scale_a_k_blocks,
        scale_k_blocks,
    };
  }

  static dim3 get_grid_shape(Params const& params) {
    const int num_mps = params.num_mps > 0 ? params.num_mps : 1;
    return dim3(static_cast<uint32_t>(num_mps), 1u, 1u);
  }

  MUTLASS_DEVICE
  static void load(Params const&       params,
                   SharedStorage&      shared_storage,
                   PipelineLoadA&      pipeline_load_a,
                   PipelineLoadB&      pipeline_load_b,
                   PipelineLoadAState& pipe_a_write,
                   PipelineLoadBState& pipe_b_write,
                   WorkTileInfo const& work_tile,
                   int                 warp_idx_in_squad) {
    const bool is_tme_issue_warp = warp_idx_in_squad == 0;

    Tensor sA      = make_tensor(make_smem_ptr(shared_storage.smem_a.begin()), SmemLayoutA{});
    Tensor sB      = make_tensor(make_smem_ptr(shared_storage.smem_b.begin()), SmemLayoutB{});
    Tensor sScaleB = make_tensor(make_smem_ptr(shared_storage.smem_scale_b.begin()), SmemLayoutScaleB{});

    auto cta_tme_a       = params.tme_a.get_slice(0);
    auto cta_tme_b       = params.tme_b.get_slice(0);
    auto cta_tme_scale_b = params.tme_scale_b.get_slice(0);

    const uint32_t m_block_idx = static_cast<uint32_t>(work_tile.M_idx);
    const uint32_t n_block_idx = static_cast<uint32_t>(work_tile.N_idx);
    const uint32_t group_idx   = static_cast<uint32_t>(work_tile.G_idx);

    auto mA  = params.tme_a.get_tme_tensor(make_shape(params.m, params.k, params.num_groups))(_, _, group_idx);
    auto gA_ = local_tile(mA, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(_, _));

    auto mB  = params.tme_b.get_tme_tensor(make_shape(params.n, params.k, params.num_groups))(_, _, group_idx);
    auto gB_ = local_tile(mB, make_shape(Int<BlockN>{}, Int<BlockK>{}), make_coord(_, _));

    Tensor gA = gA_(_, _, m_block_idx, _);
    Tensor gB = gB_(_, _, n_block_idx, _);

    Tensor tAgA = cta_tme_a.partition_S(gA);
    Tensor tAsA = cta_tme_a.partition_D(sA);
    Tensor tBgB = cta_tme_b.partition_S(gB);
    Tensor tBsB = cta_tme_b.partition_D(sB);

    auto mScaleB = params.tme_scale_b.get_tme_tensor(make_shape(params.n, params.scale_k_blocks, params.num_groups))(
        _, _, group_idx);
    Tensor tSsScaleB = cta_tme_scale_b.partition_D(sScaleB);

    MUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = 0; k_block < params.k_blocks; ++k_block) {
      if (is_tme_issue_warp) {
        pipeline_load_a.producer_acquire(pipe_a_write);
        const uint32_t barrier_id_a = pipeline_load_a.producer_get_barrier_id(pipe_a_write);
        copy(params.tme_a.with(barrier_id_a), tAgA(_, _, _, k_block), tAsA(_, _, _, int(pipe_a_write.index())));

        pipeline_load_b.producer_acquire(pipe_b_write);
        const uint32_t barrier_id_b = pipeline_load_b.producer_get_barrier_id(pipe_b_write);
        auto           mScaleBAtK   = domain_offset(make_coord(_0{}, k_block * ScaleBlocksPerTile), mScaleB);
        Tensor         gScaleB =
            local_tile(mScaleBAtK, make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{}), make_coord(n_block_idx, _0{}));
        Tensor tSgScaleB = cta_tme_scale_b.partition_S(gScaleB);
        copy(params.tme_scale_b.with(barrier_id_b), tSgScaleB, tSsScaleB(_, _, _, int(pipe_b_write.index())));

        if constexpr (IsGroupwiseScaleA) {
          auto cta_tme_scale_a = params.tme_scale_a.get_slice(0);
          auto mScaleA         = params.tme_scale_a.get_tme_tensor(
              make_shape(params.m, params.scale_a_k_blocks, params.num_groups))(_, _, group_idx);
          auto   mScaleAAtK = domain_offset(make_coord(_0{}, k_block * ScaleBlocksPerTile), mScaleA);
          Tensor gScaleA =
              local_tile(mScaleAAtK, make_shape(Int<BlockM>{}, Int<TmeScaleAKBlocks>{}), make_coord(m_block_idx, _0{}));
          Tensor sScaleA   = make_tensor(make_smem_ptr(shared_storage.smem_scale_a.begin()), SmemLayoutScaleA{});
          Tensor tSgScaleA = cta_tme_scale_a.partition_S(gScaleA);
          Tensor tSsScaleA = cta_tme_scale_a.partition_D(sScaleA);
          copy(params.tme_scale_a.with(barrier_id_b), tSgScaleA, tSsScaleA(_, _, _, int(pipe_b_write.index())));
        }

        copy(params.tme_b.with(barrier_id_b), tBgB(_, _, _, k_block), tBsB(_, _, _, int(pipe_b_write.index())));

        ++pipe_a_write;
        ++pipe_b_write;
      }
    }
  }

  MUTLASS_DEVICE
  static void cast(Params const&            params,
                   SharedStorage&           shared_storage,
                   PipelineLoadA&           pipeline_load_a,
                   PipelineCastA&           pipeline_cast_a,
                   PipelineLoadAState&      pipe_a_read,
                   PipelineCastAState&      pipe_cast_a_write,
                   CasterSyncBarrier const& caster_sync,
                   int                      caster_thread_idx) {
    Tensor sA    = make_tensor(make_smem_ptr(shared_storage.smem_a.begin()), SmemLayoutA{});
    Tensor sMmaA = make_tensor(make_smem_ptr(shared_storage.smem_mma_a.begin()), SmemLayoutMmaA{});

    constexpr int Fp8ElementsPerVector  = sizeof(uint32_t) / sizeof(ElementA);
    constexpr int Fp16ElementsPerVector = sizeof(uint64_t) / sizeof(ElementMmaA);
    static_assert(Fp8ElementsPerVector == 4 && Fp16ElementsPerVector == 4);
    static_assert(BlockK % Fp8ElementsPerVector == 0);
    static_assert(int(NumCasterThreads) % BlockM == 0);

    using CastConverter = mutlass::
        NumericArrayConverter<ElementMmaA, ElementA, Fp8ElementsPerVector, mutlass::FloatRoundStyle::round_to_nearest>;

    constexpr int VectorsPerRow    = BlockK / Fp8ElementsPerVector;
    constexpr int ThreadsPerRow    = int(NumCasterThreads) / BlockM;
    constexpr int VectorsPerThread = VectorsPerRow / ThreadsPerRow;
    static_assert(VectorsPerRow % ThreadsPerRow == 0);

    using StoreVector                      = uint128_t;
    constexpr int ConvertedVectorsPerStore = sizeof(StoreVector) / sizeof(uint64_t);
    constexpr int Fp16ElementsPerStore     = sizeof(StoreVector) / sizeof(ElementMmaA);
    constexpr int StoresPerThread          = VectorsPerThread / ConvertedVectorsPerStore;
    constexpr int Fp16ElementsPerThread    = VectorsPerThread * Fp16ElementsPerVector;
    static_assert(ConvertedVectorsPerStore == 2 && Fp16ElementsPerStore == 8);
    static_assert(VectorsPerThread % ConvertedVectorsPerStore == 0);

    using StoreCopyAtom     = Copy_Atom<UniversalCopy<StoreVector>, ElementMmaA>;
    using StoreThreadLayout = Layout<Shape<Int<BlockM>, Int<ThreadsPerRow>>, Stride<Int<ThreadsPerRow>, _1>>;
    using StoreValueLayout  = Layout<Shape<_1, Int<Fp16ElementsPerThread>>>;
    using StoreTiledCopy    = decltype(make_tiled_copy(StoreCopyAtom{}, StoreThreadLayout{}, StoreValueLayout{}));

    const int m_idx               = caster_thread_idx / ThreadsPerRow;
    const int thread_idx_in_row   = caster_thread_idx % ThreadsPerRow;
    const int first_vector_in_row = thread_idx_in_row * VectorsPerThread;

    StoreTiledCopy tiled_copy_store;
    auto           thr_copy_store = tiled_copy_store.get_thread_slice(caster_thread_idx);

    Tensor        rA                  = make_tensor<uint32_t>(Shape<Int<VectorsPerThread>>{});
    Tensor        rAArray             = recast<typename CastConverter::source_type>(rA);
    auto          tCsMmaAStorePattern = thr_copy_store.partition_D(sMmaA(_, _, _0{}));
    auto          tCrMmaAStore        = make_fragment_like<ElementMmaA>(tCsMmaAStorePattern);
    Tensor        rMmaAArray          = recast<typename CastConverter::result_type>(tCrMmaAStore);
    CastConverter converter;

    MUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = 0; k_block < params.k_blocks; ++k_block) {
      pipeline_load_a.consumer_wait(pipe_a_read);

      Tensor sAVec = recast<uint32_t>(sA(_, _, int(pipe_a_read.index())));
      MUTE_UNROLL
      for (int vector_idx = 0; vector_idx < VectorsPerThread; ++vector_idx) {
        UniversalCopy<uint32_t>::copy(sAVec(m_idx, first_vector_in_row + vector_idx), rA(vector_idx));
      }

      ++pipe_a_read;

      MUTE_UNROLL
      for (int vector_idx = 0; vector_idx < VectorsPerThread; ++vector_idx) {
        rMmaAArray(vector_idx) = converter(rAArray(vector_idx));
      }

      const uint32_t caster_phase = caster_sync.arrive</* return_phase = */ true>();
      caster_sync.wait(caster_phase);

      pipeline_cast_a.producer_acquire(pipe_cast_a_write);

      const int stage        = int(pipe_cast_a_write.index());
      auto      tCsMmaAStore = thr_copy_store.partition_D(sMmaA(_, _, stage));
      copy(tiled_copy_store, tCrMmaAStore, tCsMmaAStore);

      __syncwarp();
      pipeline_cast_a.producer_commit(pipe_cast_a_write);
      ++pipe_cast_a_write;
    }
  }

  template <class AccumTensor>
  MUTLASS_DEVICE static void epilogue(Params const&       params,
                                      AccumTensor const&  rAcc,
                                      WorkTileInfo const& work_tile,
                                      int                 consumer_thread_idx) {
    TiledMma tiled_mma;
    auto     thr_mma = tiled_mma.get_thread_slice(consumer_thread_idx);

    auto mD =
        make_tensor(make_gmem_ptr(params.ptr_d), make_shape(params.m, params.n, params.num_groups), params.stride_d);
    auto gD_ = local_tile(mD, make_shape(Int<BlockM>{}, Int<BlockN>{}, _1{}), make_coord(_, _, _));
    auto gD  = gD_(_, _, _0{}, int(work_tile.M_idx), int(work_tile.N_idx), int(work_tile.G_idx));

    auto tCgD = thr_mma.partition_C(gD);
    MUTE_STATIC_ASSERT_V(size(rAcc) == size(tCgD));

    const uint32_t m0 = uint32_t(work_tile.M_idx) * uint32_t(BlockM);
    const uint32_t n0 = uint32_t(work_tile.N_idx) * uint32_t(BlockN);
    if (m0 + uint32_t(BlockM) <= uint32_t(params.m) && n0 + uint32_t(BlockN) <= uint32_t(params.n)) {
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rAcc); ++i) {
        tCgD(i) = ElementD(rAcc(i));
      }
    } else {
      auto cD   = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
      auto tCcD = thr_mma.partition_C(cD);

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rAcc); ++i) {
        const auto     coord   = tCcD(i);
        const uint32_t coord_m = m0 + uint32_t(get<0>(coord));
        const uint32_t coord_n = n0 + uint32_t(get<1>(coord));
        if (coord_m < uint32_t(params.m) && coord_n < uint32_t(params.n)) {
          tCgD(i) = ElementD(rAcc(i));
        }
      }
    }
  }

  template <int ScaleIndex, class ScaleTensor, class ScaleATensor, class PackedScaleBTensor>
  MUTLASS_DEVICE static void calculate_scale_bst4(ScaleTensor&              scale,
                                                  ScaleATensor const&       scale_a,
                                                  PackedScaleBTensor const& scale_b_packed) {
    static_assert(ScaleIndex == 0 || ScaleIndex == 1);
    static_assert(size(ScaleTensor{}) % 4 == 0, "BST4 scaling requires groups of four accumulators");

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(scale); i += 4) {
      constexpr int Shift       = ScaleIndex * 16;
      float4        scale_b_vec = make_float4(float(ElementScaleB::bitcast(uint16_t(scale_b_packed(i + 0) >> Shift))),
                                       float(ElementScaleB::bitcast(uint16_t(scale_b_packed(i + 1) >> Shift))),
                                       float(ElementScaleB::bitcast(uint16_t(scale_b_packed(i + 2) >> Shift))),
                                       float(ElementScaleB::bitcast(uint16_t(scale_b_packed(i + 3) >> Shift))));
      float4        scale_vec   = ::mul(float(scale_a(i)), scale_b_vec);
      scale(i + 0)              = scale_vec.x;
      scale(i + 1)              = scale_vec.y;
      scale(i + 2)              = scale_vec.z;
      scale(i + 3)              = scale_vec.w;
    }
  }

  template <class AccumTensor, class ScaleTensor>
  MUTLASS_DEVICE static void scale_accumulate_bst4(AccumTensor&       accum,
                                                   AccumTensor const& accum_temp,
                                                   ScaleTensor const& scale) {
    static_assert(size(AccumTensor{}) % 4 == 0, "BST4 scaling requires groups of four accumulators");

    auto accum_bst4      = recast<f4>(accum);
    auto accum_temp_bst4 = recast<f4>(accum_temp);
    auto scale_bst4      = recast<f4>(scale);

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_bst4); ++i) {
      accum_bst4(i) += accum_temp_bst4(i) * scale_bst4(i);
    }
  }

  MUTLASS_DEVICE
  static auto compute(Params const&       params,
                      SharedStorage&      shared_storage,
                      PipelineLoadA&      pipeline_load_a,
                      PipelineLoadB&      pipeline_load_b,
                      PipelineCastA&      pipeline_cast_a,
                      PipelineLoadAState& pipe_a_release,
                      PipelineLoadBState& pipe_b_read,
                      PipelineLoadBState& pipe_b_prefetch,
                      PipelineCastAState& pipe_cast_a_read,
                      WorkTileInfo const& work_tile,
                      int                 consumer_thread_idx) {
    TiledMma tiled_mma;
    auto     thr_mma = tiled_mma.get_thread_slice(consumer_thread_idx);

    Tensor sMmaA   = make_tensor(make_smem_ptr(shared_storage.smem_mma_a.begin()), SmemLayoutMmaA{});
    Tensor sB      = make_tensor(make_smem_ptr(shared_storage.smem_b.begin()), SmemLayoutB{});
    Tensor sScaleA = make_tensor(make_smem_ptr(shared_storage.smem_scale_a.begin()), SmemLayoutScaleA{});
    Tensor sScaleB = make_tensor(make_smem_ptr(shared_storage.smem_scale_b.begin()), SmemLayoutScaleB{});

    auto rAcc           = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
    auto rAccTempFirst  = make_fragment_like(rAcc);
    auto rAccTempSecond = make_fragment_like(rAcc);
    auto rScaleFirst    = make_fragment_like(rAcc);
    auto rScaleSecond   = make_fragment_like(rAcc);
    fill(rAcc, float(0));

    Tensor tCsA = thr_mma.partition_A(sMmaA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    MUTE_STATIC_ASSERT_V(size<2>(tCrA) == _4{});
    MUTE_STATIC_ASSERT_V(size<2>(tCrB) == _4{});

    const uint32_t m_block_idx = static_cast<uint32_t>(work_tile.M_idx);
    const uint32_t group_idx   = static_cast<uint32_t>(work_tile.G_idx);

    auto mScaleA = make_tensor(make_gmem_ptr(params.ptr_scale_a),
                               make_shape(params.m, params.scale_a_k_blocks, params.num_groups),
                               params.stride_scale_a);
    auto gScaleA = local_tile(mScaleA(_, _, group_idx), make_tile(Int<BlockM>{}), make_coord(m_block_idx, _));

    using ScaleAViewAsCLayout = Layout<Shape<Shape<_1, Int<BlockM>>, Int<BlockN>>, Stride<Stride<_0, _1>, _0>>;
    using ScaleBViewAsCLayout = Layout<Shape<Int<BlockM>, Shape<_1, Int<BlockN>>>, Stride<_0, Stride<_0, _1>>>;

    Tensor tCgScaleA               = thr_mma.partition_C(gScaleA(_, _0{}).compose(ScaleAViewAsCLayout{}));
    Tensor tCrScaleAFirst          = make_tensor_like<ElementScaleA>(tCgScaleA);
    Tensor tCrScaleASecond         = make_tensor_like<ElementScaleA>(tCgScaleA);
    Tensor tCrScaleANextFirst      = make_tensor_like<ElementScaleA>(tCgScaleA);
    Tensor tCrScaleANextSecond     = make_tensor_like<ElementScaleA>(tCgScaleA);
    auto   tCrScaleAFirstFlat      = filter_zeros(tCrScaleAFirst);
    auto   tCrScaleASecondFlat     = filter_zeros(tCrScaleASecond);
    auto   tCrScaleANextFirstFlat  = filter_zeros(tCrScaleANextFirst);
    auto   tCrScaleANextSecondFlat = filter_zeros(tCrScaleANextSecond);

    if constexpr (!IsGroupwiseScaleA) {
      using ScaleAGmemCopyAtom = Copy_Atom<MP31_ROBUST_LOAD<ElementScaleA>, ElementScaleA>;
      auto scale_a_copy        = ScaleAGmemCopyAtom{}.with(params.desc_scale_a);
      copy(scale_a_copy, filter_zeros(tCgScaleA), tCrScaleAFirstFlat);
    }

    Tensor tCsScaleBTemplate       = thr_mma.partition_C(sScaleB(_, _0{}, _0{}).compose(ScaleBViewAsCLayout{}));
    Tensor tCrScaleBPacked         = make_tensor_like<uint32_t>(tCsScaleBTemplate);
    Tensor tCrScaleBNextPacked     = make_tensor_like<uint32_t>(tCsScaleBTemplate);
    auto   tCrScaleBPackedFlat     = filter_zeros(tCrScaleBPacked);
    auto   tCrScaleBNextPackedFlat = filter_zeros(tCrScaleBNextPacked);

    MUTE_STATIC_ASSERT_V(size(rAcc) == size(tCrScaleAFirst));
    MUTE_STATIC_ASSERT_V(size(rAcc) == size(tCrScaleBPacked));

    pipeline_load_b.consumer_wait(pipe_b_prefetch);
    const int initial_scale_stage = int(pipe_b_prefetch.index());
    Tensor tCsScaleBFirst = thr_mma.partition_C(sScaleB(_, _0{}, initial_scale_stage).compose(ScaleBViewAsCLayout{}));
    auto   tCsScaleBFirstFlat = filter_zeros(tCsScaleBFirst);
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCrScaleBPackedFlat); ++i) {
      UniversalCopy<uint32_t>::copy(reinterpret_cast<uint32_t const&>(tCsScaleBFirstFlat(i)), tCrScaleBPackedFlat(i));
    }
    if constexpr (IsGroupwiseScaleA) {
      Tensor tCsScaleAFirst = thr_mma.partition_C(sScaleA(_, _0{}, initial_scale_stage).compose(ScaleAViewAsCLayout{}));
      Tensor tCsScaleASecond =
          thr_mma.partition_C(sScaleA(_, _1{}, initial_scale_stage).compose(ScaleAViewAsCLayout{}));
      auto tCsScaleAFirstFlat  = filter_zeros(tCsScaleAFirst);
      auto tCsScaleASecondFlat = filter_zeros(tCsScaleASecond);
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrScaleAFirstFlat); ++i) {
        UniversalCopy<ElementScaleA>::copy(tCsScaleAFirstFlat(i), tCrScaleAFirstFlat(i));
        UniversalCopy<ElementScaleA>::copy(tCsScaleASecondFlat(i), tCrScaleASecondFlat(i));
      }
    }
    ++pipe_b_prefetch;

    MUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = 0; k_block < params.k_blocks; ++k_block) {
      if (k_block < params.k_blocks - 1) {
        pipeline_load_b.consumer_wait(pipe_b_prefetch);
        const int next_scale_stage = int(pipe_b_prefetch.index());
        Tensor    tCsScaleBNextFirst =
            thr_mma.partition_C(sScaleB(_, _0{}, next_scale_stage).compose(ScaleBViewAsCLayout{}));
        auto tCsScaleBNextFirstFlat = filter_zeros(tCsScaleBNextFirst);
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrScaleBNextPackedFlat); ++i) {
          UniversalCopy<uint32_t>::copy(reinterpret_cast<uint32_t const&>(tCsScaleBNextFirstFlat(i)),
                                        tCrScaleBNextPackedFlat(i));
        }
        if constexpr (IsGroupwiseScaleA) {
          Tensor tCsScaleANextFirst =
              thr_mma.partition_C(sScaleA(_, _0{}, next_scale_stage).compose(ScaleAViewAsCLayout{}));
          Tensor tCsScaleANextSecond =
              thr_mma.partition_C(sScaleA(_, _1{}, next_scale_stage).compose(ScaleAViewAsCLayout{}));
          auto tCsScaleANextFirstFlat  = filter_zeros(tCsScaleANextFirst);
          auto tCsScaleANextSecondFlat = filter_zeros(tCsScaleANextSecond);
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tCrScaleANextFirstFlat); ++i) {
            UniversalCopy<ElementScaleA>::copy(tCsScaleANextFirstFlat(i), tCrScaleANextFirstFlat(i));
            UniversalCopy<ElementScaleA>::copy(tCsScaleANextSecondFlat(i), tCrScaleANextSecondFlat(i));
          }
        }
        ++pipe_b_prefetch;
      }

      pipeline_cast_a.consumer_wait(pipe_cast_a_read);
      pipeline_load_b.consumer_wait(pipe_b_read);

      const int stage_a = int(pipe_cast_a_read.index());
      const int stage_b = int(pipe_b_read.index());

      tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
      MUTLASS_PRAGMA_UNROLL
      for (int mma_k = 0; mma_k < SqmmasPerScale; ++mma_k) {
        mute::gemm(tiled_mma, tCrA(_, _, mma_k, stage_a), tCrB(_, _, mma_k, stage_b), rAccTempFirst);
        tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
      }
      ::mate::warpsquad_commit_batch();

      tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
      MUTLASS_PRAGMA_UNROLL
      for (int mma_k = SqmmasPerScale; mma_k < ScaleBlocksPerTile * SqmmasPerScale; ++mma_k) {
        mute::gemm(tiled_mma, tCrA(_, _, mma_k, stage_a), tCrB(_, _, mma_k, stage_b), rAccTempSecond);
        tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
      }
      ::mate::warpsquad_commit_batch();

      if constexpr (!IsGroupwiseScaleA) {
        calculate_scale_bst4<0>(rScaleFirst, tCrScaleAFirst, tCrScaleBPacked);
        calculate_scale_bst4<1>(rScaleSecond, tCrScaleAFirst, tCrScaleBPacked);
      } else {
        calculate_scale_bst4<0>(rScaleFirst, tCrScaleAFirst, tCrScaleBPacked);
        calculate_scale_bst4<1>(rScaleSecond, tCrScaleASecond, tCrScaleBPacked);
      }
      if (k_block < params.k_blocks - 1) {
        copy(tCrScaleBNextPackedFlat, tCrScaleBPackedFlat);
        if constexpr (IsGroupwiseScaleA) {
          copy(tCrScaleANextFirstFlat, tCrScaleAFirstFlat);
          copy(tCrScaleANextSecondFlat, tCrScaleASecondFlat);
        }
      }
      ::mate::warpsquad_wait<1>();
      scale_accumulate_bst4(rAcc, rAccTempFirst, rScaleFirst);
      ::mate::warpsquad_wait<0>();
      scale_accumulate_bst4(rAcc, rAccTempSecond, rScaleSecond);

      pipeline_load_a.consumer_release(pipe_a_release);
      pipeline_cast_a.consumer_release(pipe_cast_a_read);
      pipeline_load_b.consumer_release(pipe_b_read);
      ++pipe_a_release;
      ++pipe_cast_a_read;
      ++pipe_b_read;
    }

    return rAcc;
  }

  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static_assert(SharedStorageSize <= MaxSharedStorageBytes, "Shared storage exceeds the MP31 per-block limit");

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineLoadAParams pipeline_a_params;
    pipeline_a_params.transaction_bytes = TmeTransactionBytesA;
    pipeline_a_params.num_consumers     = NumConsumerWarpSquads * WarpsPerWarpSquad;
    pipeline_a_params.num_producers     = 1;
    PipelineLoadA pipeline_load_a(pipeline_a_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineLoadA));

    PipelineLoadBParams pipeline_b_params;
    pipeline_b_params.transaction_bytes = TmeTransactionBytesB + TmeTransactionBytesScaleB + TmeTransactionBytesScaleA;
    pipeline_b_params.num_consumers     = NumConsumerWarpSquads * WarpsPerWarpSquad;
    pipeline_b_params.num_producers     = 1;
    PipelineLoadB pipeline_load_b(pipeline_b_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineLoadB));

    PipelineCastAParams pipeline_cast_a_params;
    pipeline_cast_a_params.producer_arv_count = NumCasterWarpSquads * WarpsPerWarpSquad;
    pipeline_cast_a_params.consumer_arv_count = NumConsumerWarpSquads * WarpsPerWarpSquad;
    PipelineCastA pipeline_cast_a(pipeline_cast_a_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineCastA));

    CasterSyncBarrier caster_sync(reinterpret_cast<uint64_t>(&barrier_storage->CasterSync));
    if (threadIdx.x == 0) {
      caster_sync.init(WarpsPerWarpSquad, 0);
    }

    const int squad_idx           = mutlass::canonical_warp_squad_idx();
    const int warp_idx_in_squad   = mutlass::canonical_warp_idx_sync() % int(WarpsPerWarpSquad);
    const int thread_idx_in_squad = int(threadIdx.x) % int(NumThreadsPerWarpSquad);

    constexpr int FirstCasterSquad   = int(NumProducerWarpSquads);
    constexpr int FirstConsumerSquad = int(NumProducerWarpSquads + NumCasterWarpSquads);
    constexpr int TotalWarpSquads    = int(NumProducerWarpSquads + NumCasterWarpSquads + NumConsumerWarpSquads);

    const bool is_producer_squad = squad_idx == 0;
    const bool is_caster_squad   = squad_idx >= FirstCasterSquad && squad_idx < FirstConsumerSquad;
    const bool is_consumer_squad = squad_idx >= FirstConsumerSquad && squad_idx < TotalWarpSquads;
    const int  caster_thread_idx = (squad_idx - FirstCasterSquad) * int(NumThreadsPerWarpSquad) + thread_idx_in_squad;
    const int  consumer_thread_idx =
        (squad_idx - FirstConsumerSquad) * int(NumThreadsPerWarpSquad) + thread_idx_in_squad;

    PipelineLoadAState pipe_a_write      = mutlass::make_producer_start_state_warpspecialized<PipelineLoadA>();
    PipelineLoadBState pipe_b_write      = mutlass::make_producer_start_state_warpspecialized<PipelineLoadB>();
    PipelineCastAState pipe_cast_a_write = mutlass::make_producer_start_state_warpspecialized<PipelineCastA>();

    Scheduler scheduler(static_cast<uint32_t>(params.m),
                        static_cast<uint32_t>(params.n),
                        static_cast<uint32_t>(params.num_groups),
                        params.ptr_masked_m,
                        static_cast<uint32_t>(params.expected_m));

    __syncthreads();

    auto work_tile = scheduler.initial_work_tile_info();

    if (is_producer_squad) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        load(params,
             shared_storage,
             pipeline_load_a,
             pipeline_load_b,
             pipe_a_write,
             pipe_b_write,
             work_tile,
             warp_idx_in_squad);
        scheduler.advance_to_next_work();
        work_tile = scheduler.get_work_tile_info();
      }
      return;
    }

    PipelineLoadAState pipe_a_read;
    PipelineLoadAState pipe_a_release;
    PipelineLoadBState pipe_b_read;
    PipelineLoadBState pipe_b_prefetch;
    PipelineCastAState pipe_cast_a_read;

    if (is_caster_squad) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        cast(params,
             shared_storage,
             pipeline_load_a,
             pipeline_cast_a,
             pipe_a_read,
             pipe_cast_a_write,
             caster_sync,
             caster_thread_idx);
        scheduler.advance_to_next_work();
        work_tile = scheduler.get_work_tile_info();
      }
      return;
    }

    if (is_consumer_squad) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        auto rAcc = compute(params,
                            shared_storage,
                            pipeline_load_a,
                            pipeline_load_b,
                            pipeline_cast_a,
                            pipe_a_release,
                            pipe_b_read,
                            pipe_b_prefetch,
                            pipe_cast_a_read,
                            work_tile,
                            consumer_thread_idx);
        epilogue(params, rAcc, work_tile, consumer_thread_idx);

        scheduler.advance_to_next_work();
        work_tile = scheduler.get_work_tile_info();
      }
    }
  }
};

}  // namespace mate::gemm::masked_moe_gemm_mixed_dtype
