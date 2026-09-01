#pragma once

#include <mutlass/fast_math.h>
#include <mutlass/mutlass.h>

#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "../../common/mma_mp31_sqmma.hpp"
#include "../../common/numeric_conversion.hpp"
#include "block_info.hpp"
#include "load_primitive_builder.hpp"
#include "mask.hpp"
#include "pack_gqa.hpp"
#include "paged_kv.hpp"
#include "pipeline_ws.hpp"
#include "rotary.hpp"
#include "seqlen.hpp"
#include "softmax.hpp"
#include "utils.hpp"

#define SHOW(x)   \
  print(#x ": "); \
  print(x);       \
  print("\n")

namespace mate::attention::fmha {

using namespace mute;

template <class Element_,
          class ElementAccumulator_,
          class TileShape_,
          int  StagesK_,
          int  StagesV_,
          int  HeadDimV_,
          int  NumQKConsumers_,
          bool HasCuseqlensQ_,
          bool HasCuseqlensK_,
          bool HasCuseqlensKNew_,
          bool HasKvBatchIdx_,
          bool HasSequsedQ_,
          bool HasSequsedK_,
          bool HasLeftpadK_,
          bool HasQDscale_,
          bool HasKDscale_,
          bool HasVDscale_,
          bool IsPagedKV_,
          bool UseLSULoadK_,
          bool UseLSULoadV_,
          bool IsCausal_,
          bool IsLocal_,
          bool HasLearnableSink_,
          bool HasSoftcap_,
          bool IsAppendKV_,
          bool HasQv_,
          bool OnlyQv_,
          int  HeadRatio_,
          bool IsPackGQA_,
          bool Split_,
          bool IsRotary_,
          bool IsRotaryInterleaved_,
          bool HasSeqlensRotary_,
          bool EnableCP_,
          bool HasAttentionChunk_,
          int  NumPVConsumers_ = NumQKConsumers_>
struct Mp31FmhaFwdTmeWarpSpecialized {
  using Element            = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementSink        = mutlass::bfloat16_t;

  using TileShape                    = TileShape_;
  static constexpr int TileM         = get<0>(TileShape{});
  static constexpr int TileN         = get<1>(TileShape{});
  static constexpr int TileK         = get<2>(TileShape{});  // Entry for future headdim tiling
  static constexpr int HeadDimQK     = get<2>(TileShape{});
  static constexpr int HeadDimVO     = HeadDimV_;
  static constexpr int HeadRatio     = HeadRatio_;
  static constexpr int TileHeadRatio = std::min(HeadRatio, TileM);

  static constexpr int NumLoadWarpSquads = 1;
  static constexpr int NumQKConsumers    = NumQKConsumers_;
  static constexpr int NumPVConsumers    = NumPVConsumers_;
  // For Pinghu, the consumer granularity is WarpSquad
  static constexpr int NumMmaWarpSquads = std::max(NumQKConsumers_, NumPVConsumers_);
  static constexpr int NumMmaThreads    = NumMmaWarpSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumQKMmaThreads  = NumQKConsumers * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumPVMmaThreads  = NumPVConsumers * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumTransWarps    = mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumTransThreads  = NumTransWarps * mutlass::NumThreadsPerWarp;

  static_assert(TileM % NumQKConsumers == 0);
  static_assert(TileM % NumPVConsumers == 0);

  static constexpr bool HasCuseqlensQ     = HasCuseqlensQ_;
  static constexpr bool HasCuseqlensK     = HasCuseqlensK_;
  static constexpr bool HasCuseqlensKNew  = HasCuseqlensKNew_;
  static constexpr bool HasKvBatchIdx     = HasKvBatchIdx_;
  static constexpr bool HasSequsedQ       = HasSequsedQ_;
  static constexpr bool HasSequsedK       = HasSequsedK_;
  static constexpr bool HasLeftpadK       = HasLeftpadK_;
  static constexpr bool HasQv             = HasQv_;
  static constexpr bool OnlyQv            = OnlyQv_;
  static constexpr bool InKernelTranspose = HasQv;
  static_assert(!OnlyQv || HasQv, "OnlyQv requires HasQv");

  static constexpr bool HasQDescale = HasQDscale_;
  static constexpr bool HasKDescale = HasKDscale_;
  static constexpr bool HasVDescale = HasVDscale_;

  static constexpr bool IsPackGQA = IsPackGQA_;
  static constexpr bool IsPagedKV = IsPagedKV_;
  static constexpr bool IsCausal  = IsCausal_;
  static constexpr bool IsLocal   = IsLocal_;
  static constexpr bool Split     = Split_;

  static constexpr bool HasLearnableSink = HasLearnableSink_;
  static constexpr bool HasSoftcap       = HasSoftcap_;

  static constexpr bool EnableCP          = EnableCP_;
  static constexpr bool HasAttentionChunk = HasAttentionChunk_;

  static constexpr bool IsAppendKV          = IsAppendKV_;
  static constexpr bool IsRotary            = IsRotary_;
  static constexpr bool IsRotaryInterleaved = IsRotaryInterleaved_;
  static constexpr bool HasSeqlensRotary    = HasSeqlensRotary_;

  static constexpr bool SameHeadDim = HeadDimQK == HeadDimVO;

  static constexpr bool IsFP8 =
      mute::is_same_v<Element, mutlass::float_e4m3_t> || mute::is_same_v<Element, mutlass::float_e5m2_t>;
  static constexpr int MaxOffset = IsFP8 ? 8 : 0;

  static constexpr int UsePackGQATMELoad =
      IsPackGQA_ && mutlass::is_pow2<TileHeadRatio>::value && TileM % TileHeadRatio == 0;
  static constexpr bool UseTMELoadQ = !IsPackGQA_ || UsePackGQATMELoad;
  static constexpr bool UseLSULoadQ = !UseTMELoadQ;

  static constexpr bool UseLSULoadK = UseLSULoadK_;
  static constexpr bool UseLSULoadV = UseLSULoadV_;

  static_assert(!IsPagedKV || (UseLSULoadK && UseLSULoadV) || (!UseLSULoadK && !UseLSULoadV),
                "KV Load methods must be same if paged KV is enabled!");
  static_assert(IsPagedKV || ((HasCuseqlensK || HasLeftpadK) && UseLSULoadK) || !HasCuseqlensK,
                "Load K support LSU Only if ragged KV or leftpad_k is enabled!");

  static constexpr int NumProducerThreads =
      UseLSULoadQ || UseLSULoadK || UseLSULoadV ? mutlass::NumThreadsPerWarpSquad : mutlass::NumThreadsPerWarp;
  static constexpr bool SingleProducerWarp = NumProducerThreads == mutlass::NumThreadsPerWarp;

  static constexpr bool IntraWarpSquadOverlap = !HasQv || (HeadDimQK == 64 && HeadDimVO == 256);

  static constexpr bool IsMmaPvRS = false;

  static constexpr int StagesQ  = 1;
  static constexpr int StagesK  = StagesK_;
  static constexpr int StagesV  = StagesV_;
  static constexpr int StagesVt = StagesV;
  static constexpr int StagesQv = 1;

  static constexpr int MaxBarPerStageRatio = 4;

  static constexpr int AdditionalBarrier = static_cast<int>(FwdNamedBarriers::NumFwdNamedBarriers) +
                                           static_cast<int>(mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount);

  using BarPerStageRatioHelper = mutlass::Mp31PipelineWarpSpecializedBarrierRatio<MaxBarPerStageRatio,
                                                                                  AdditionalBarrier,
                                                                                  StagesQ,
                                                                                  HasQv ? StagesQv : 0,
                                                                                  StagesK,
                                                                                  StagesV,
                                                                                  InKernelTranspose ? StagesVt : 0,
                                                                                  2 * (IsAppendKV ? StagesK : 0)>;

  static constexpr int BarPerStageRatio = BarPerStageRatioHelper::value;
  static_assert(BarPerStageRatio > 0, "FMHA async barrier storage exceeds hardware limit even with BarPerStageRatio=1");

  static constexpr int Alignment = 32 / sizeof_bits_v<Element>;

  using ShapeQKV  = Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
  using StrideQKV = Stride<int64_t, _1, int64_t, int64_t>;

  // ((head_ratio, seqlen_q), d, nheads_kv, batch)
  using ShapeQPacked_  = Shape<Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>;
  using StrideQPacked_ = Stride<Stride<int64_t, int64_t>, _1, int64_t, int64_t>;

  using ShapeQPacked  = std::conditional_t<!IsPackGQA, ShapeQKV, ShapeQPacked_>;
  using StrideQPacked = std::conditional_t<!IsPackGQA, StrideQKV, StrideQPacked_>;

  using ShapePageTable  = Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
  using StridePageTable = Stride<int64_t, _1>;

  using ShapeRotary  = Shape<int32_t, int32_t>;
  using StrideRotary = Stride<int64_t, _1>;

  using StrideDescale = Stride<int64_t, int64_t>;

  using NamedBarrier = mutlass::arch::AsyncBarrier;

  using SeqlenInfo = SeqlenInfoQK<HasCuseqlensQ,
                                  HasSequsedQ,
                                  HasCuseqlensK,
                                  HasSequsedK,
                                  HasLeftpadK,
                                  IsAppendKV,
                                  HasCuseqlensKNew,
                                  HasSeqlensRotary,
                                  EnableCP>;
  using BlockInfo =
      BlockInfo<SeqlenInfo, TileM, TileN, HeadRatio, IsCausal, IsLocal, IsPackGQA, HasAttentionChunk, Split, EnableCP>;

  // NOTE: RoPE not implemented yet.
  // using Rotary = Rotary<TileN, TileK, NumMmaThreads, Element>;

  using PackGQAManager   = PackGQAManager<Element, HeadRatio, TileM, HeadDimQK, NumProducerThreads>;
  using TileMPack        = Shape<Int<TileHeadRatio>, Int<TileM / TileHeadRatio>>;
  using LayoutMPack      = decltype(make_layout(TileMPack{}));
  using PackGQATileShape = Shape<TileMPack, Int<HeadDimQK>>;
  using PackGQvManager =
      ::mate::attention::fmha::PackGQAManager<Element, HeadRatio, TileM, HeadDimVO, NumProducerThreads>;
  using PackGQvTileShape = Shape<TileMPack, Int<HeadDimVO>>;

  // Tile View
  using TileShapeQKD = Shape<Int<TileM>, Int<TileN>, Int<HeadDimQK>>;
  using TileShapeQvD = Shape<Int<TileM>, Int<TileN>, Int<HeadDimVO>>;
  using TileShapePDV = Shape<Int<TileM>, Int<HeadDimVO>, Int<TileN>>;

  using AtomLayoutQK = Layout<Shape<Int<NumQKConsumers>, _1, _1>>;

  using TiledMmaQK = decltype(mute::make_tiled_mma(mute::MP31::SQMMA::ss_op_selector<Element,
                                                                                     Element,
                                                                                     ElementAccumulator,
                                                                                     TileShapeQKD,
                                                                                     TCE::Major::K,
                                                                                     TCE::Major::K,
                                                                                     Int<TileM / NumQKConsumers>>(),
                                                   AtomLayoutQK{}));
  using TiledMmaQv = decltype(mute::make_tiled_mma(mute::MP31::SQMMA::ss_op_selector<Element,
                                                                                     Element,
                                                                                     ElementAccumulator,
                                                                                     TileShapeQvD,
                                                                                     TCE::Major::K,
                                                                                     TCE::Major::K,
                                                                                     Int<TileM / NumQKConsumers>>(),
                                                   AtomLayoutQK{}));

  using AtomLayoutPV = Layout<Shape<Int<NumPVConsumers>, _1, _1>>;

  using TiledMmaPV = decltype(mute::make_tiled_mma(mute::MP31::SQMMA::ss_op_selector<Element,
                                                                                     Element,
                                                                                     ElementAccumulator,
                                                                                     TileShapePDV,
                                                                                     TCE::Major::K,
                                                                                     TCE::Major::MN,
                                                                                     Int<TileM / NumPVConsumers>>(),
                                                   AtomLayoutPV{}));

  // TODO: refine ss_smem_selector
  using SmemAtomLayoutQ =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaQK::Atom::MMA_Op, TileShapeQKD>());
  using SmemAtomLayoutK =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::K, Element, typename TiledMmaQK::Atom::MMA_Op, TileShapeQKD>());

  using SmemAtomLayoutQv =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaQv::Atom::MMA_Op, TileShapeQvD>());
  // K-major for dot(Qv, V)
  using SmemAtomLayoutVMmaQV =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::K, Element, typename TiledMmaQv::Atom::MMA_Op, TileShapeQvD>());

  using SmemAtomLayoutP =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaPV::Atom::MMA_Op, TileShapePDV>());

  // MN-major for dot(P, V)
  using SmemAtomLayoutVMmaPV =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::MN, Element, typename TiledMmaPV::Atom::MMA_Op, TileShapePDV>());

  using SmemLayoutQ         = decltype(tile_to_shape(SmemAtomLayoutQ{}, select<0, 2>(TileShapeQKD{})));
  using SmemLayoutQPack_    = decltype(composition(SmemLayoutQ{}, make_tile(LayoutMPack{}, Underscore{})));
  using SmemLayoutTMELoadQ  = std::conditional_t<!IsPackGQA, SmemLayoutQ, SmemLayoutQPack_>;
  using SmemLayoutQv        = decltype(tile_to_shape(SmemAtomLayoutQv{}, select<0, 2>(TileShapeQvD{})));
  using SmemLayoutQvPack_   = decltype(composition(SmemLayoutQv{}, make_tile(LayoutMPack{}, Underscore{})));
  using SmemLayoutTMELoadQv = std::conditional_t<!IsPackGQA, SmemLayoutQv, SmemLayoutQvPack_>;
  using SmemLayoutK         = decltype(tile_to_shape(
      SmemAtomLayoutK{}, make_shape(shape<1>(TileShapeQKD{}), shape<2>(TileShapeQKD{}), Int<StagesK>{})));
  using SmemLayoutP         = decltype(tile_to_shape(SmemAtomLayoutP{}, select<0, 2>(TileShapePDV{})));

  /* For dot(Qv, V), we need K-Major SmemLayout */
  using SmemLayoutVMmaQV = decltype(tile_to_shape(
      SmemAtomLayoutVMmaQV{}, make_shape(shape<1>(TileShapeQvD{}), shape<2>(TileShapeQvD{}), Int<StagesV>{})));

  /* For dot(P, V), we need MN-Major SmemLayout */
  using SmemLayoutVMmaPV = decltype(tile_to_shape(
      SmemAtomLayoutVMmaPV{}, make_shape(shape<1>(TileShapePDV{}), shape<2>(TileShapePDV{}), Int<StagesV>{})));

  // SmemLayoutV is used for TME Load
  using SmemLayoutV = std::conditional_t<HasQv, SmemLayoutVMmaQV, SmemLayoutVMmaPV>;

  // Transpose for dot(P, V)
  using SmemLayoutVStore = decltype(composition(
      SmemLayoutVMmaPV{},
      make_ordered_layout(
          make_shape(size<1>(SmemLayoutVMmaPV{}), size<0>(SmemLayoutVMmaPV{}), size<2>(SmemLayoutVMmaPV{})),
          Step<_2, _1, _3>{})));

  // Dealing with LSU load V
  using SmemAtomLayoutVLsu = std::conditional_t<
      !IsFP8,
      decltype(mute::MP31::SQMMA::Layout_SL256_SS256_SG32_Atom<get<2>(typename TiledMmaPV::AtomShape_MNK{}),
                                                               get<1>(typename TiledMmaPV::AtomShape_MNK{}),
                                                               Element,
                                                               TCE::Major::K>{}),
      decltype(mute::MP31::SQMMA::Layout_SL256_SS256_SG16_Atom<get<2>(typename TiledMmaPV::AtomShape_MNK{}),
                                                               get<1>(typename TiledMmaPV::AtomShape_MNK{}),
                                                               Element,
                                                               TCE::Major::K>{})>;

  // Match the stride but without PermuteLoad
  using SmemLayoutVLsu = decltype(tile_to_shape(
      SmemAtomLayoutVLsu{}, make_shape(shape<2>(TileShapePDV{}), shape<1>(TileShapePDV{}), Int<StagesV>{})));

  static constexpr TME::CacheHint TmeQInnerHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeQOuterHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeQvInnerHint   = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeQvOuterHint   = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKInnerHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKOuterHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVInnerHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVOuterHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKNewInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKNewOuterHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVNewInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVNewOuterHint = TME::CacheHint::CACHE_NORMAL;

  static constexpr int KLoadVectorBits = IsFP8 && TileN == 64 ? 64 : 128;
  static constexpr int MmaPvAtomM      = get<0>(typename TiledMmaPV::AtomShape_MNK{});
  using TmeLoadKeyBuilder =
      Mp31FmhaTmeLoadKeyBuilder<Element, SmemLayoutK, StrideQKV, TmeKInnerHint, TmeKOuterHint, KLoadVectorBits>;

  // Only used for dot(Qv, V)
  using TmeLoadVLayout  = std::conditional_t<HasQv, SmemLayoutVMmaQV, /* We don't need it actually */ SmemLayoutK>;
  using TmeLoadVBuilder = Mp31FmhaTmeLoadKeyBuilder<Element, TmeLoadVLayout, StrideQKV, TmeVInnerHint, TmeVOuterHint>;

  static constexpr int FragmentSize    = TmeLoadKeyBuilder::Fragment;
  static constexpr int CvtFragmentSize = FragmentSize;

  using FragmentTypeR2S = typename TmeLoadKeyBuilder::FragmentType;
  using PermuteTileR2S  = Tile<Underscore, typename TmeLoadKeyBuilder::PermuteTileN, Underscore>;

  using PermutedShapeK = decltype(TmeLoadKeyBuilder::get_permuted_shape(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{})));
  using PermutedShapeV = decltype(TmeLoadVBuilder::get_permuted_shape(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{})));

  using TmeKTileShape = typename TmeLoadKeyBuilder::TmeKTileShape;
  using TmeVTileShape = typename TmeLoadVBuilder::TmeKTileShape;

  using BarrierQ = mutlass::arch::AsyncTransactionBarrier;

  using TileShapeQ  = std::conditional_t<!IsPackGQA, Shape<Int<TileM>, Int<HeadDimQK>>, PackGQATileShape>;
  using TileShapeQv = std::conditional_t<!IsPackGQA, Shape<Int<TileM>, Int<HeadDimVO>>, PackGQvTileShape>;
  using TME_Q       = decltype(make_tme_copy<TmeQInnerHint, TmeQOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideQPacked{}, int32_t(0)),
                  StrideQPacked{}),
      SmemLayoutTMELoadQ{},
      TileShapeQ{}));
  using TME_Qv      = decltype(make_tme_copy<TmeQvInnerHint, TmeQvOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideQPacked{}, int32_t(0)),
                  StrideQPacked{}),
      SmemLayoutTMELoadQv{},
      TileShapeQv{}));
  using TME_K       = typename TmeLoadKeyBuilder::TME_K;
  // V TME TiledCopy for MmaPV
  using TME_V_ = decltype(make_tme_copy<TmeVInnerHint, TmeVOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideQKV{}, int32_t(0)),
                  select<1, 0, 2, 3>(StrideQKV{})),
      take<0, 2>(SmemLayoutVMmaPV{})));
  // If HasQv, using TmeBuilder to get a permuted tme load, otherwise using trivial version
  using TME_V = std::conditional_t<HasQv, typename TmeLoadVBuilder::TME_K, TME_V_>;
  // Append V_new is stored back to the KV cache, so keep it in the store-friendly layout even in Qv mode.
  using TME_VNew = TME_V_;
  using TME_KNew = decltype(make_tme_copy<TmeKNewInnerHint, TmeKNewOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
      take<0, 2>(SmemLayoutK{})));

  using PermuteTileForQK  = Tile<Underscore, typename TmeLoadKeyBuilder::MmaPermuteTile, Underscore>;
  using PermuteTiledMmaQK = decltype(convert_to_permuted_mma(TiledMmaQK{}, PermuteTileForQK{}));
  using R2SCopyAtom       = Copy_Atom<UniversalCopy<FragmentTypeR2S>, Element>;
  using R2STiledCopy      = decltype(make_tiled_copy_C(R2SCopyAtom{}, PermuteTiledMmaQK{}));

  // LSU Load Key (Always permuted load)
  using LsuLoadKeyBuilder =
      Mp31FmhaLsuLoadKeyBuilder<Element, TileShapeQKD, SmemAtomLayoutK, NumProducerThreads, StrideQKV, KLoadVectorBits>;
  using GmemTiledCopyK = typename LsuLoadKeyBuilder::GmemTiledCopy;
  using LsuKTile       = typename LsuLoadKeyBuilder::LsuKTile;
  using PermuteTileK   = Tile<Underscore, typename LsuLoadKeyBuilder::PermuteTileN, Underscore>;
  using FragmentTypeK  = typename LsuLoadKeyBuilder::FragmentType;

  // LSU Value for QV(Always permuted load)
  using LsuLoadValueBuilder = Mp31FmhaLsuLoadKeyBuilder<Element,
                                                        TileShapeQvD,
                                                        SmemAtomLayoutVMmaQV,
                                                        NumProducerThreads,
                                                        StrideQKV,
                                                        KLoadVectorBits>;
  using GmemTiledCopyV      = typename LsuLoadValueBuilder::GmemTiledCopy;
  using LsuVTile            = typename LsuLoadValueBuilder::LsuKTile;

  // InKernelTranspose
  using TransTiledCopy =
      decltype(mutlass::gemm::collective::detail::make_simt_tiled_copy<NumTransThreads,
                                                                       Element,
                                                                       FragmentSize,
                                                                       StrideQKV,
                                                                       size<0>(SmemAtomLayoutVMmaPV{}),
                                                                       size<1>(SmemAtomLayoutVMmaPV{}),
                                                                       UniversalCopy<FragmentTypeR2S>>());

  static constexpr int GmemElemsPerLoad = 128 / sizeof_bits_v<Element>;
  static constexpr int HeadDimGCD       = mute::gcd(HeadDimQK, HeadDimVO);
  static constexpr int BytePerHalfRow   = HeadDimGCD / 2 * sizeof(Element);
  static constexpr int BlockKGmem =
      (BytePerHalfRow % 128 == 0 ? 128 : (BytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int GmemThreadsPerRow = BlockKGmem / GmemElemsPerLoad;
  using GmemCopyAtomAppendKV             = mute::Copy_Atom<MP31_ROBUST_STORE<mute::uint128_t>, Element>;
  using GmemLayoutAtomAppendKV =
      Layout<Shape<Int<NumMmaThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>, Stride<Int<GmemThreadsPerRow>, _1>>;
  using GmemTiledCopyAppendKV = decltype(make_tiled_copy(
      GmemCopyAtomAppendKV{}, GmemLayoutAtomAppendKV{}, Layout<Shape<_1, Int<GmemElemsPerLoad>>>{}));

  using MainloopPipelineQ = std::conditional_t<!UseTMELoadQ,
                                               mutlass::Mp31PipelineAsyncWarpSpecialized<StagesQ, BarPerStageRatio>,
                                               mutlass::Mp31PipelineTmeAsyncWarpSpecialized<StagesQ, BarPerStageRatio>>;
  using MainloopPipelineQv =
      std::conditional_t<!UseTMELoadQ,
                         mutlass::Mp31PipelineAsyncWarpSpecialized<HasQv ? StagesQv : 0, BarPerStageRatio>,
                         mutlass::Mp31PipelineTmeAsyncWarpSpecialized<HasQv ? StagesQv : 0, BarPerStageRatio>>;
  using MainloopPipelineK = std::conditional_t<UseLSULoadK,
                                               mutlass::Mp31PipelineAsyncWarpSpecialized<StagesK, BarPerStageRatio>,
                                               mutlass::Mp31PipelineTmeAsyncWarpSpecialized<StagesK, BarPerStageRatio>>;
  using MainloopPipelineV = std::conditional_t<UseLSULoadV,
                                               mutlass::Mp31PipelineAsyncWarpSpecialized<StagesV, BarPerStageRatio>,
                                               mutlass::Mp31PipelineTmeAsyncWarpSpecialized<StagesV, BarPerStageRatio>>;
  using MainloopPipelineVt =
      mutlass::Mp31PipelineAsyncWarpSpecialized<InKernelTranspose ? StagesVt : 0, BarPerStageRatio>;
  using MainloopPipelineKVNew =
      mutlass::Mp31PipelineTmeAsyncWarpSpecialized<IsAppendKV ? StagesK : 0,
                                                   BarPerStageRatio>;  // Always use TME for new KV

  using PipelineQState     = typename MainloopPipelineQ::PipelineState;
  using PipelineQvState    = typename MainloopPipelineQv::PipelineState;
  using PipelineKState     = typename MainloopPipelineK::PipelineState;
  using PipelineVState     = typename MainloopPipelineV::PipelineState;
  using PipelineVtState    = typename MainloopPipelineVt::PipelineState;
  using PipelineKVNewState = typename MainloopPipelineKVNew::PipelineState;

  static_assert(cosize_v<SmemLayoutVMmaQV> == cosize_v<SmemLayoutVMmaPV>,
                "QV and PV V smem layouts must use the same storage size.");
  static_assert(size(take<0, 2>(SmemLayoutVMmaQV{})) == size(take<0, 2>(SmemLayoutVMmaPV{})),
                "QV and PV V smem layouts must use the same TME transaction size.");

  struct SharedStorage {
    mute::array_aligned<Element, cosize_v<SmemLayoutQ>>                  smem_q;
    mute::array_aligned<Element, cosize_v<SmemLayoutK>>                  smem_k;
    mute::array_aligned<Element, cosize_v<SmemLayoutP>>                  smem_p;
    mute::array_aligned<Element, cosize_v<SmemLayoutV>>                  smem_v;
    mute::array_aligned<Element, HasQv ? cosize_v<SmemLayoutQv> : 0>     smem_qv;
    mute::array_aligned<Element, HasQv ? cosize_v<SmemLayoutVStore> : 0> smem_vt;
  };

  static constexpr int TmeTransactionBytesQ = mutlass::bits_to_bytes(size(SmemLayoutQ{}) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesK =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesQv = mutlass::bits_to_bytes(size(SmemLayoutQv{}) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesV =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutV{})) * sizeof_bits_v<Element>);

  struct Arguments {
    Element const* const ptr_Q;
    ShapeQKV const       shape_Q;
    StrideQKV const      stride_Q;
    Element* const       ptr_K;
    ShapeQKV const       shape_K;
    StrideQKV const      stride_K;
    Element* const       ptr_V;
    int32_t const        headdim_V;
    StrideQKV const      stride_V;

    // AppendKV
    Element const* const ptr_K_new;
    ShapeQKV const       shape_K_new;
    StrideQKV const      stride_K_new;
    Element const* const ptr_V_new;
    StrideQKV const      stride_V_new;

    // Qv
    Element const* const ptr_Qv;
    ShapeQKV const       shape_Qv;
    StrideQKV const      stride_Qv;

    // Rotary
    Element const* const ptr_rotary_cos;
    ShapeRotary const    shape_rotary;
    StrideRotary const   stride_rotary_cos;
    Element const* const ptr_rotary_sin;
    StrideRotary const   stride_rotary_sin;
    // bool const is_rotary_interleaved;  // Use IsRotaryInterleaved instead

    // PageTable
    int const* const      ptr_pagetable;
    ShapePageTable const  shape_pagetable;
    StridePageTable const stride_pagetable;

    // Scale
    float const         softmax_scale;
    float const*        ptr_q_descale;
    StrideDescale const stride_q_descale;
    float const*        ptr_k_descale;
    StrideDescale const stride_k_descale;
    float const*        ptr_v_descale;
    StrideDescale const stride_v_descale;

    // Local
    int const window_size_left  = -1;
    int const window_size_right = -1;

    // Chunk
    int const attention_chunk = 0;

    // Learnable Sink
    ElementSink const* ptr_learnable_sink;

    // Softcap
    float const softcap_val;

    int const num_splits;

    // Aux tensors
    uint32_t const* const kv_batch_idx     = nullptr;
    uint32_t const* const cu_seqlens_q     = nullptr;
    uint32_t const* const cu_seqlens_k     = nullptr;
    uint32_t const* const cu_seqlens_k_new = nullptr;
    uint32_t const* const seqused_q        = nullptr;
    uint32_t const* const seqused_k        = nullptr;
    uint32_t const* const leftpad_k        = nullptr;
    uint32_t const* const seqlens_rotary   = nullptr;

    // CP
    int             cp_world_size    = 1;
    int             cp_rank          = 0;
    uint32_t const* cp_tot_seqused_k = nullptr;
  };

  struct Params {
    Element const* const ptr_Q;
    ShapeQKV const       shape_Q;
    StrideQKV const      stride_Q;
    ShapeQPacked const   shape_Q_packed;
    StrideQPacked const  stride_Q_packed;
    Element* const       ptr_K;
    ShapeQKV const       shape_K;
    PermutedShapeK const permuted_shape_K;
    StrideQKV const      stride_K;
    Element* const       ptr_V;
    ShapeQKV const       shape_V;
    PermutedShapeV const permuted_shape_V;
    int32_t const        headdim_V;
    StrideQKV const      stride_V;

    // AppendKV
    Element const* const ptr_K_new;
    ShapeQKV const       shape_K_new;
    StrideQKV const      stride_K_new;
    Element const* const ptr_V_new;
    StrideQKV const      stride_V_new;

    // Qv
    Element const* const ptr_Qv;
    StrideQKV const      stride_Qv;
    ShapeQPacked const   shape_Qv_packed;
    StrideQPacked const  stride_Qv_packed;

    // Rotary
    Element const* const ptr_rotary_cos;
    ShapeRotary const    shape_rotary;
    StrideRotary const   stride_rotary_cos;
    Element const* const ptr_rotary_sin;
    StrideRotary const   stride_rotary_sin;
    // bool const is_rotary_interleaved;  // Use IsRotaryInterleaved instead

    // PageTable
    int const* const      ptr_pagetable;
    ShapePageTable const  shape_pagetable;
    StridePageTable const stride_pagetable;

    mutlass::FastDivmod const page_size_divmod;
    // mutlass::FastDivmod const blockN_per_page_size_divmod;
    // mutlass::FastDivmod const qhead_per_khead_divmod;

    // TiledCopy
    TME_Q    tme_load_Q;
    TME_Qv   tme_load_Qv;
    TME_K    tme_load_K;
    TME_V    tme_load_V;
    TME_KNew tme_load_K_new;
    TME_VNew tme_load_V_new;

    // Robust Desc
    RobustDescriptor const desc_Q;
    RobustDescriptor const desc_Qv;
    RobustDescriptor const desc_K;
    RobustDescriptor const desc_V;
    RobustDescriptor const desc_page_table;
    RobustDescriptor const desc_K_new;
    RobustDescriptor const desc_V_new;
    RobustDescriptor const desc_Cos;
    RobustDescriptor const desc_Sin;

    // Scale
    float const         softmax_scale;
    float const         softmax_scale_log2;
    float const*        ptr_q_descale;
    StrideDescale const stride_q_descale;
    float const*        ptr_k_descale;
    StrideDescale const stride_k_descale;
    float const*        ptr_v_descale;
    StrideDescale const stride_v_descale;

    // Local
    int const window_size_left  = -1;
    int const window_size_right = -1;

    // Sink
    // Unused
    int const sink_token_length = 0;

    // Learnable Sink
    ElementSink const* ptr_learnable_sink = nullptr;

    float const softcap_val = 0.f;

    // Chunk
    mutlass::FastDivmod attention_chunk_divmod;

    // Aux tensors
    uint32_t const* const kv_batch_idx     = nullptr;
    uint32_t const* const cu_seqlens_q     = nullptr;
    uint32_t const* const cu_seqlens_k     = nullptr;
    uint32_t const* const cu_seqlens_k_new = nullptr;
    uint32_t const* const seqused_q        = nullptr;
    uint32_t const* const seqused_k        = nullptr;
    uint32_t const* const leftpad_k        = nullptr;
    uint32_t const* const seqlens_rotary   = nullptr;

    // CP
    int             cp_world_size    = 1;
    int             cp_rank          = 0;
    uint32_t const* cp_tot_seqused_k = nullptr;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    // If IsPackGQA, reshape Q to be ((head_ratio, seqlen_q), head_size, num_head_kv, batch_size)

    int const  qhead_per_k_head = !IsPackGQA ? 1 : HeadRatio;
    auto const shape_Q_packed_  = make_shape(make_shape(qhead_per_k_head, get<0>(args.shape_Q)),
                                            get<1>(args.shape_Q),
                                            get<2>(args.shape_K),
                                            get<3>(args.shape_Q));
    auto const shape_Q_packed   = mute::conditional_return<!IsPackGQA>(args.shape_Q, shape_Q_packed_);

    auto const stride_Q_packed_  = make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)),
                                              get<1>(args.stride_Q),
                                              get<2>(args.stride_Q) * qhead_per_k_head,
                                              get<3>(args.stride_Q));
    auto const stride_Q_packed   = mute::conditional_return<!IsPackGQA>(args.stride_Q, stride_Q_packed_);
    auto const shape_Qv_packed_  = make_shape(make_shape(qhead_per_k_head, get<0>(args.shape_Qv)),
                                             get<1>(args.shape_Qv),
                                             get<2>(args.shape_K),
                                             get<3>(args.shape_Qv));
    auto const shape_Qv_packed   = mute::conditional_return<!IsPackGQA>(args.shape_Qv, shape_Qv_packed_);
    auto const stride_Qv_packed_ = make_stride(make_stride(get<2>(args.stride_Qv), get<0>(args.stride_Qv)),
                                               get<1>(args.stride_Qv),
                                               get<2>(args.stride_Qv) * qhead_per_k_head,
                                               get<3>(args.stride_Qv));
    auto const stride_Qv_packed  = mute::conditional_return<!IsPackGQA>(args.stride_Qv, stride_Qv_packed_);

    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), shape_Q_packed, stride_Q_packed);
    TME_Q  tme_load_Q =
        make_tme_copy<TmeQInnerHint, TmeQOuterHint>(MP31_TME_LOAD{}, mQ, SmemLayoutTMELoadQ{}, TileShapeQ{});
    Tensor mQv = make_tensor(make_gmem_ptr(args.ptr_Qv), shape_Qv_packed, stride_Qv_packed);
    TME_Qv tme_load_Qv =
        make_tme_copy<TmeQvInnerHint, TmeQvOuterHint>(MP31_TME_LOAD{}, mQv, SmemLayoutTMELoadQv{}, TileShapeQv{});

    int const cosize_q  = get<0>(args.shape_Q) == 0 ? 0 : cosize(make_layout(shape_Q_packed, stride_Q_packed));
    auto      desc_Q    = make_robust_desc(args.ptr_Q, cosize_q);
    int const cosize_qv = get<0>(args.shape_Qv) == 0 ? 0 : cosize(make_layout(shape_Qv_packed, stride_Qv_packed));
    auto      desc_Qv   = make_robust_desc(args.ptr_Qv, cosize_qv);

    // print("TME_Q:");
    // print(tme_load_Q);
    // print("\n");

    Tensor         mK               = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TME_K          tme_load_K       = TmeLoadKeyBuilder::make_tme_copy(mK);
    PermutedShapeK permuted_shape_K = TmeLoadKeyBuilder::get_permuted_shape(mK);
    int const      cosize_k         = get<0>(args.shape_K) == 0 ? 0 : cosize(make_layout(args.shape_K, args.stride_K));

    RobustDescriptor desc_K = make_robust_desc(args.ptr_K, cosize_k);

    // print("TME_K:");
    // print(tme_load_K);
    // print("\n");

    auto shape_V_gmem = make_shape(get<0>(args.shape_K), args.headdim_V, get<2>(args.shape_K), get<3>(args.shape_K));
    auto shape_V      = conditional_return<HasQv>(shape_V_gmem, select<1, 0, 2, 3>(shape_V_gmem));
    auto stride_V     = conditional_return<HasQv>(args.stride_V, select<1, 0, 2, 3>(args.stride_V));
    auto mV           = make_tensor(make_gmem_ptr(args.ptr_V), shape_V, stride_V);

    TME_V tme_load_V = [&]() {
      if constexpr (HasQv) {
        return TmeLoadVBuilder::make_tme_copy(mV);
      } else {
        return make_tme_copy<TmeVInnerHint, TmeVOuterHint>(MP31_TME_LOAD{}, mV, take<0, 2>(SmemLayoutVMmaPV{}));
      }
    }();

    // Used for permute tme load
    PermutedShapeV permuted_shape_V = TmeLoadVBuilder::get_permuted_shape(mV);
    int const      cosize_v         = get<0>(shape_V_gmem) == 0 ? 0 : cosize(make_layout(shape_V_gmem, args.stride_V));

    RobustDescriptor desc_V = make_robust_desc(args.ptr_V, cosize_v);

    // print("TME_V:");
    // print(tme_load_V);
    // print("\n");

    // AppendKV
    Tensor   mKnew          = make_tensor(make_gmem_ptr(args.ptr_K_new), args.shape_K_new, args.stride_K_new);
    TME_KNew tme_load_K_new = make_tme_copy<TmeKNewInnerHint, TmeKNewOuterHint>(
        MP31_TME_LOAD{}, conditional_return<IsAppendKV>(mKnew, mK), take<0, 2>(SmemLayoutK{}));
    int const cosize_k_new =
        get<0>(args.shape_K_new) == 0 ? 0 : cosize(make_layout(args.shape_K_new, args.stride_K_new));

    RobustDescriptor desc_K_new = make_robust_desc(args.ptr_K_new, cosize_k_new);

    Tensor mV_store =
        make_tensor(make_gmem_ptr(args.ptr_V), select<1, 0, 2, 3>(shape_V_gmem), select<1, 0, 2, 3>(args.stride_V));
    Tensor mVnew = make_tensor(
        make_gmem_ptr(args.ptr_V_new),
        make_shape(args.headdim_V, get<0>(args.shape_K_new), get<2>(args.shape_K_new), get<3>(args.shape_K_new)),
        select<1, 0, 2, 3>(args.stride_V_new));
    TME_VNew tme_load_V_new = make_tme_copy<TmeVInnerHint, TmeVOuterHint>(
        MP31_TME_LOAD{}, conditional_return<IsAppendKV>(mVnew, mV_store), take<0, 2>(SmemLayoutVMmaPV{}));
    int const cosize_v_new = get<0>(args.shape_K_new) == 0 ? 0 : cosize(mVnew.layout());

    RobustDescriptor desc_V_new = make_robust_desc(args.ptr_V_new, cosize_v_new);

    RobustDescriptor desc_Cos = make_robust_desc(
        args.ptr_rotary_cos,
        get<0>(args.shape_rotary) == 0 ? 0 : cosize(make_layout(args.shape_rotary, args.stride_rotary_cos)));
    RobustDescriptor desc_Sin = make_robust_desc(
        args.ptr_rotary_sin,
        get<0>(args.shape_rotary) == 0 ? 0 : cosize(make_layout(args.shape_rotary, args.stride_rotary_sin)));

    // Qv

    float const log2e = std::log2(std::exp(1.0f));
    // float const effective_softmax_scale = HasSoftcap ? args.softcap_val : args.softmax_scale;

    int const           page_size = IsPagedKV ? get<0>(args.shape_K) : 1;
    mutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
    attention_chunk_divmod.divisor = args.attention_chunk;

    RobustDescriptor desc_page_table = make_robust_desc(args.ptr_pagetable, cosize(make_layout(args.shape_pagetable)));

    // SHOW(shape_Q_packed);
    // SHOW(stride_Q_packed);

    // SHOW(shape_Q_packed);
    // SHOW(stride_Q_packed);

    // SHOW(TiledMmaQK{});
    // SHOW(TiledMmaPV{});

    // SHOW(SmemLayoutQ{});
    // SHOW(SmemLayoutTMELoadQ{});
    // SHOW(SmemLayoutK{});
    // SHOW(SmemLayoutP{});
    // SHOW(SmemLayoutV{});
    // SHOW(R2STiledCopy{});

    // SHOW(UseLSULoadQ);
    // SHOW(UseLSULoadK);
    // SHOW(UseLSULoadV);

    return Params{
        .ptr_Q            = args.ptr_Q,
        .shape_Q          = args.shape_Q,
        .stride_Q         = args.stride_Q,
        .shape_Q_packed   = shape_Q_packed,
        .stride_Q_packed  = stride_Q_packed,
        .ptr_K            = args.ptr_K,
        .shape_K          = args.shape_K,
        .permuted_shape_K = permuted_shape_K,
        .stride_K         = args.stride_K,
        .ptr_V            = args.ptr_V,
        .shape_V          = shape_V,
        .permuted_shape_V = permuted_shape_V,
        .headdim_V        = args.headdim_V,
        .stride_V         = args.stride_V,

        // AppendKV
        .ptr_K_new    = args.ptr_K_new,
        .shape_K_new  = args.shape_K_new,
        .stride_K_new = args.stride_K_new,
        .ptr_V_new    = args.ptr_V_new,
        .stride_V_new = args.stride_V_new,

        // Qv
        .ptr_Qv           = args.ptr_Qv,
        .stride_Qv        = args.stride_Qv,
        .shape_Qv_packed  = shape_Qv_packed,
        .stride_Qv_packed = stride_Qv_packed,

        // Rotary
        .ptr_rotary_cos    = args.ptr_rotary_cos,
        .shape_rotary      = args.shape_rotary,
        .stride_rotary_cos = args.stride_rotary_cos,
        .ptr_rotary_sin    = args.ptr_rotary_sin,
        .stride_rotary_sin = args.stride_rotary_sin,
        // .is_rotary_interleaved;  // Use IsRotaryInterleaved instead

        // PageTable
        .ptr_pagetable    = args.ptr_pagetable,
        .shape_pagetable  = args.shape_pagetable,
        .stride_pagetable = args.stride_pagetable,
        .page_size_divmod = mutlass::FastDivmod(page_size),

        // TiledCopy
        .tme_load_Q     = tme_load_Q,
        .tme_load_Qv    = tme_load_Qv,
        .tme_load_K     = tme_load_K,
        .tme_load_V     = tme_load_V,
        .tme_load_K_new = tme_load_K_new,
        .tme_load_V_new = tme_load_V_new,

        .desc_Q          = desc_Q,
        .desc_Qv         = desc_Qv,
        .desc_K          = desc_K,
        .desc_V          = desc_V,
        .desc_page_table = desc_page_table,
        .desc_K_new      = desc_K_new,
        .desc_V_new      = desc_V_new,
        .desc_Cos        = desc_Cos,
        .desc_Sin        = desc_Sin,

        .softmax_scale      = args.softmax_scale,
        .softmax_scale_log2 = args.softmax_scale * log2e,
        .ptr_q_descale      = args.ptr_q_descale,
        .stride_q_descale   = args.stride_q_descale,
        .ptr_k_descale      = args.ptr_k_descale,
        .stride_k_descale   = args.stride_k_descale,
        .ptr_v_descale      = args.ptr_v_descale,
        .stride_v_descale   = args.stride_v_descale,

        .window_size_left  = args.window_size_left,
        .window_size_right = args.window_size_right,

        .ptr_learnable_sink = args.ptr_learnable_sink,

        .softcap_val = args.softcap_val,

        // Chunk
        .attention_chunk_divmod = attention_chunk_divmod,

        // Aux tensors
        .kv_batch_idx     = args.kv_batch_idx,
        .cu_seqlens_q     = args.cu_seqlens_q,
        .cu_seqlens_k     = args.cu_seqlens_k,
        .cu_seqlens_k_new = args.cu_seqlens_k_new,
        .seqused_q        = args.seqused_q,
        .seqused_k        = args.seqused_k,
        .leftpad_k        = args.leftpad_k,
        .seqlens_rotary   = args.seqlens_rotary,

        // CP
        .cp_world_size    = args.cp_world_size,
        .cp_rank          = args.cp_rank,
        .cp_tot_seqused_k = args.cp_tot_seqused_k,
    };
  }

  template <class BarrierStorage, class BlockCoord, class PipelineQv>
  MUTE_DEVICE void load(Params const&      params,
                        MainloopPipelineQ& pipeline_q,
                        PipelineQv&        pipeline_qv,
                        MainloopPipelineK& pipeline_k,
                        MainloopPipelineV& pipeline_v,
                        PipelineQState&    smem_pipe_write_q,
                        PipelineQvState&   smem_pipe_write_qv,
                        PipelineKState&    smem_pipe_write_k,
                        PipelineVState&    smem_pipe_write_v,
                        SharedStorage&     shared_storage,
                        BarrierStorage*    barrier_storage,
                        SeqlenInfo const&  seqlen_info,
                        BlockCoord         blk_coord,
                        int&               work_idx,
                        int const          num_splits) {
    int const m_block   = get<0>(blk_coord);
    int const bidh      = get<1>(blk_coord);
    int const bidb      = get<2>(blk_coord);
    int const split_idx = get<3>(blk_coord);

    auto [n_block_min, n_block_max] = BlockInfo::get_n_block_min_max(seqlen_info,
                                                                     m_block,
                                                                     split_idx,
                                                                     /*splits*/ num_splits,
                                                                     params.window_size_left,
                                                                     params.window_size_right,
                                                                     params.attention_chunk_divmod);

    // If no valid block, no need to load.
    if (n_block_min >= n_block_max) {
      return;
    }

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutTMELoadQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    // Used for LSU
    Tensor sV_lsu = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVLsu{});

    int const thread_idx             = threadIdx.x % NumProducerThreads;
    int const bidh_kv                = !IsPackGQA ? bidh / HeadRatio : bidh;
    int const bidb_kv                = !HasKvBatchIdx ? bidb : params.kv_batch_idx[bidb];
    int const warp_idx               = mutlass::canonical_warp_idx();
    int const warp_idx_in_warp_squad = warp_idx % mutlass::NumWarpsPerWarpSquad;
    int const seqlen_k               = seqlen_info.seqlen_k;

    auto offset_coord_q = [&]() {
      if constexpr (!IsPackGQA) {
        return make_coord(seqlen_info.offset_q, _0{});
      } else {
        return make_coord(make_coord(_0{}, seqlen_info.offset_q), _0{});
      }
    }();
    Tensor mQ = params.tme_load_Q.get_tme_tensor(params.shape_Q_packed)(_, _, bidh, HasCuseqlensQ ? 0 : bidb);
    Tensor mK = params.tme_load_K.get_tme_tensor(params.permuted_shape_K)(_, _, bidh_kv, _);
    auto   mV = [&]() {
      if constexpr (HasQv) {
        return params.tme_load_V.get_tme_tensor(params.permuted_shape_V)(_, _, bidh_kv, _);
      } else {
        return params.tme_load_V.get_tme_tensor(params.shape_V)(_, _, bidh_kv, _);
      }
    }();

    Tensor gQ = local_tile(domain_offset(offset_coord_q, mQ), TileShapeQ{}, make_coord(m_block, _0{}));
    Tensor gK = local_tile(
        domain_offset(make_coord(seqlen_info.offset_k, _0{}, _0{}), mK), TmeKTileShape{}, make_coord(_, _0{}, _));

    auto gV = [&]() {
      if constexpr (HasQv) {
        return local_tile(
            domain_offset(make_coord(seqlen_info.offset_k, _0{}, _0{}), mV), TmeVTileShape{}, make_coord(_, _0{}, _));
      } else {
        return local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k, _0{}), mV),
                          select<1, 2>(TileShapePDV{}),
                          make_coord(_0{}, _, _));
      }
    }();

    auto   cta_tme_Q = params.tme_load_Q.get_slice(0);
    Tensor tQgQ      = group_modes<0, 3>(cta_tme_Q.partition_S(gQ));
    Tensor tQsQ      = group_modes<0, 3>(cta_tme_Q.partition_D(sQ));

    auto   cta_tme_K = params.tme_load_K.get_slice(0);
    Tensor tKgK      = group_modes<0, 3>(cta_tme_K.partition_S(gK));  // (TME, n, l)
    Tensor tKsK      = group_modes<0, 3>(cta_tme_K.partition_D(sK));  // (TME, pipe)

    auto   cta_tme_V = params.tme_load_V.get_slice(0);
    Tensor tVgV      = group_modes<0, 3>(cta_tme_V.partition_S(gV));  // (TME, n, b)
    Tensor tVsV      = group_modes<0, 3>(cta_tme_V.partition_D(sV));  // (TME, pipe)

    int const bidb_kv_idx = !HasCuseqlensK && !IsPagedKV ? bidb_kv : 0;

    using KVManager = PagedKVManager<IsPagedKV,
                                     Element,
                                     NumProducerThreads,
                                     TileN,
                                     HeadDimQK,
                                     HeadDimVO,
                                     !IntraWarpSquadOverlap,
                                     1,
                                     KLoadVectorBits>;
    GmemTiledCopyK tiled_copy_k;
    auto           thr_copy_k = tiled_copy_k.get_thread_slice(threadIdx.x);

    Tensor mK_lsu = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(
        _, _, bidh_kv, HasCuseqlensK ? 0 : bidb_kv);
    Tensor gK_lsu =
        local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK_lsu), LsuKTile{}, make_coord(_, _0{}));

    GmemTiledCopyV tiled_copy_v;
    auto           thr_copy_v = tiled_copy_v.get_thread_slice(threadIdx.x);
    Tensor         mV_lsu     = make_tensor(make_gmem_ptr(params.ptr_V), params.shape_V, params.stride_V)(
        _, _, bidh_kv, HasCuseqlensK ? 0 : bidb_kv);
    Tensor gV_lsu =
        local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV_lsu), LsuVTile{}, make_coord(_, _0{}));

    KVManager paged_kv_manager{params.ptr_pagetable,
                               params.shape_pagetable,
                               params.stride_pagetable,
                               params.desc_page_table,
                               params.ptr_K,
                               params.shape_K,
                               params.stride_K,
                               params.desc_K,
                               params.ptr_V,
                               params.headdim_V,
                               params.stride_V,
                               params.desc_V,
                               params.page_size_divmod,
                               seqlen_k,
                               static_cast<int>(seqlen_info.leftpad_k),
                               thread_idx,
                               bidb_kv,
                               bidh_kv,
                               bidb_kv_idx};

    auto load_K = [&](int const n_block, auto& smem_pipe_write) {
      pipeline_k.producer_acquire(smem_pipe_write);
      if constexpr (IsPagedKV) {
        if constexpr (UseLSULoadK) {
          paged_kv_manager.load_K(n_block, sK(_, _, smem_pipe_write.index()));
          mute::ldgsts_wait();
          pipeline_k.producer_commit(smem_pipe_write);
        } else {
          uint32_t bar_id                 = pipeline_k.producer_get_barrier_id(smem_pipe_write);
          auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_tme_k();
          copy(params.tme_load_K.with(bar_id), tKgK(_, n_block_idx, bidb_kv_idx), tKsK(_, smem_pipe_write.index()));
        }
        ++smem_pipe_write;
      } else {
        // NOT Paged KV
        // Got BSHD or Ragged KV
        if constexpr (UseLSULoadK) {
          Tensor tKgK_lsu = group_modes<0, 3>(thr_copy_k.partition_S(gK_lsu(_, _, n_block)));
          Tensor tKsK_lsu = group_modes<0, 3>(thr_copy_k.partition_D(sK));
          copy(tiled_copy_k.with(params.desc_K), tKgK_lsu, tKsK_lsu(_, smem_pipe_write.index()));

          mute::ldgsts_wait();
          pipeline_k.producer_commit(smem_pipe_write);
        } else {
          uint32_t bar_id = pipeline_k.producer_get_barrier_id(smem_pipe_write);
          copy(params.tme_load_K.with(bar_id), tKgK(_, n_block, bidb_kv_idx), tKsK(_, smem_pipe_write.index()));
        }
        ++smem_pipe_write;
      }
    };

    auto load_V = [&](int const n_block, auto& smem_pipe_write) {
      pipeline_v.producer_acquire(smem_pipe_write);
      if constexpr (IsPagedKV) {
        if constexpr (UseLSULoadV) {
          if constexpr (HasQv) {
            paged_kv_manager.template load_V(n_block, sV(_, _, smem_pipe_write.index()));
          } else {
            paged_kv_manager.template load_V(n_block, sV_lsu(_, _, smem_pipe_write.index()));
          }
          mute::ldgsts_wait();
          pipeline_v.producer_commit(smem_pipe_write);
        } else {
          uint32_t bar_id                 = pipeline_v.producer_get_barrier_id(smem_pipe_write);
          auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_tme_v();
          copy(params.tme_load_V.with(bar_id), tVgV(_, n_block_idx, bidb_kv_idx), tVsV(_, smem_pipe_write.index()));
        }
        ++smem_pipe_write;
      } else {
        // NOT Paged KV
        // Got BSHD or Ragged KV

        // For non-paged case, we only use lsu load v when HasQv
        if constexpr (UseLSULoadV && HasQv) {
          Tensor tVgV_lsu = group_modes<0, 3>(thr_copy_v.partition_S(gV_lsu(_, _, n_block)));
          Tensor tVsV_lsu = group_modes<0, 3>(thr_copy_v.partition_D(sV));
          copy(tiled_copy_v.with(params.desc_V), tVgV_lsu, tVsV_lsu(_, smem_pipe_write.index()));

          mute::ldgsts_wait();
          pipeline_v.producer_commit(smem_pipe_write);
        } else {
          uint32_t bar_id = pipeline_v.producer_get_barrier_id(smem_pipe_write);
          copy(params.tme_load_V.with(bar_id), tVgV(_, n_block, bidb_kv_idx), tVsV(_, smem_pipe_write.index()));
        }
        ++smem_pipe_write;
      }
    };

    bool should_load_K = UseLSULoadK || SingleProducerWarp || warp_idx_in_warp_squad == 0;
    bool should_load_V = UseLSULoadV || SingleProducerWarp || warp_idx_in_warp_squad == 0;

    int n_block = n_block_max - 1;

    if constexpr (UseTMELoadQ) {
      // (Non-)PackGQA TME load Q
      if (SingleProducerWarp || warp_idx_in_warp_squad == 0) {
        pipeline_q.producer_acquire(smem_pipe_write_q);
        uint32_t bar_id = pipeline_q.producer_get_barrier_id(smem_pipe_write_q);
        copy(params.tme_load_Q.with(bar_id), tQgQ, tQsQ);
        ++smem_pipe_write_q;

        if constexpr (HasQv) {
          Tensor sQv = make_tensor(make_smem_ptr(shared_storage.smem_qv.data()), SmemLayoutTMELoadQv{});
          Tensor mQv = params.tme_load_Qv.get_tme_tensor(params.shape_Qv_packed)(_, _, bidh, HasCuseqlensQ ? 0 : bidb);
          Tensor gQv = local_tile(domain_offset(offset_coord_q, mQv), TileShapeQv{}, make_coord(m_block, _0{}));
          auto   cta_tme_Qv = params.tme_load_Qv.get_slice(0);
          Tensor tQvgQv     = group_modes<0, 3>(cta_tme_Qv.partition_S(gQv));
          Tensor tQvsQv     = group_modes<0, 3>(cta_tme_Qv.partition_D(sQv));

          pipeline_qv.producer_acquire(smem_pipe_write_qv);
          uint32_t qv_bar_id = pipeline_qv.producer_get_barrier_id(smem_pipe_write_qv);
          copy(params.tme_load_Qv.with(qv_bar_id), tQvgQv, tQvsQv);
          ++smem_pipe_write_qv;
        }
      }
    } else {
      // PackGQA LSU Load Q
      pipeline_q.producer_acquire(smem_pipe_write_q);
      Tensor mQPack =
          make_tensor(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q),
                      make_layout(params.shape_Q_packed, params.stride_Q_packed))(_, _, bidh, HasCuseqlensQ ? 0 : bidb);
      Tensor sQPack = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
      PackGQAManager::load_Q(params, mQPack, sQPack, thread_idx, m_block);

      if constexpr (HasQv) {
        pipeline_qv.producer_acquire(smem_pipe_write_qv);
        Tensor mQvPack = make_tensor(params.ptr_Qv + seqlen_info.offset_q * get<0>(params.stride_Qv),
                                     make_layout(params.shape_Qv_packed, params.stride_Qv_packed))(
            _, _, bidh, HasCuseqlensQ ? 0 : bidb);
        Tensor sQvPack = make_tensor(make_smem_ptr(shared_storage.smem_qv.data()), SmemLayoutQv{});
        PackGQvManager::template load_Q</*IsQv*/ true>(params, mQvPack, sQvPack, thread_idx, m_block);
      }
    }

    // Prologue load from page_table + load K
    if (should_load_K) {
      if constexpr (IsPagedKV) {
        // NOTE: force use same load method (LSU or TME) for both K and V if IsPagedKV
        if constexpr (UseLSULoadK || UseLSULoadV) {
          paged_kv_manager.template load_page_table_for_lsu</* FirstIter */ true,
                                                            /* PermuteK */ true,
                                                            /* PermuteV */ HasQv>(n_block);
        } else {
          paged_kv_manager.template load_page_table_for_tme</* FirstIter */ true>(n_block);
        }
      }
      load_K(n_block, smem_pipe_write_k);
    }

    if constexpr (!UseTMELoadQ) {
      mute::ldgsts_wait();
      pipeline_q.producer_commit(smem_pipe_write_q);
      ++smem_pipe_write_q;
      if constexpr (HasQv) {
        pipeline_qv.producer_commit(smem_pipe_write_qv);
        ++smem_pipe_write_qv;
      }
    }

    if constexpr (!IntraWarpSquadOverlap) {
      if (should_load_V) {
        load_V(n_block, smem_pipe_write_v);
      }
    }

    int n_block_prev = n_block;
    --n_block;
    for (; n_block >= n_block_min; --n_block) {
      if constexpr (IsPagedKV) {
        // NOTE: using same load method (LSU or TME) for both K and V if IsPagedKV
        if (should_load_K) {
          if constexpr (UseLSULoadK || UseLSULoadV) {
            paged_kv_manager.template load_page_table_for_lsu</* FirstIter */ false,
                                                              /* PermuteK */ true,
                                                              /* PermuteV */ HasQv>(n_block);
          } else {
            paged_kv_manager.template load_page_table_for_tme</* FirstIter */ false>(n_block);
          }
        }
      }
      if (should_load_K) {
        load_K(n_block, smem_pipe_write_k);
      }

      if (should_load_V) {
        if constexpr (IntraWarpSquadOverlap) {
          load_V(n_block_prev, smem_pipe_write_v);
        } else {
          load_V(n_block, smem_pipe_write_v);
        }
      }
      n_block_prev = n_block;
    }

    if constexpr (IntraWarpSquadOverlap) {
      if (should_load_V) {
        load_V(n_block_prev, smem_pipe_write_v);
      }
    }
  }

  template <class BlockCoord, class PipelineVt>
  MUTLASS_DEVICE void transpose(BlockCoord const&  blk_coord,
                                Params const&      params,
                                MainloopPipelineV& pipeline_v,
                                PipelineVState&    smem_pipe_v_read,
                                PipelineVt&        pipeline_vt,
                                PipelineVtState&   smem_pipe_vt_write,
                                SharedStorage&     shared_storage,
                                SeqlenInfo const&  seqlen_info,
                                int const          thread_idx,
                                int const          num_splits) {
    static_assert(InKernelTranspose);

    TransTiledCopy tiled_copy_trans;
    ThrCopy        thr_copy_trans = tiled_copy_trans.get_thread_slice(thread_idx);

    Tensor smem_src = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), typename TmeLoadVBuilder::SmemLayoutK{});
    Tensor smem_dst = make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), SmemLayoutVStore{});

    Tensor tVsV  = thr_copy_trans.partition_S(smem_src);
    Tensor tVsVt = thr_copy_trans.partition_D(smem_dst);
    Tensor tVrV  = make_fragment_like(tVsV(_, _, _, 0));

    int const m_block   = get<0>(blk_coord);
    int const split_idx = get<3>(blk_coord);

    auto [n_block_min, n_block_max] = BlockInfo::get_n_block_min_max(seqlen_info,
                                                                     m_block,
                                                                     split_idx,
                                                                     /*splits*/ num_splits,
                                                                     params.window_size_left,
                                                                     params.window_size_right,
                                                                     params.attention_chunk_divmod);

    for (int n_block = n_block_min; n_block < n_block_max; ++n_block) {
      pipeline_vt.producer_acquire(smem_pipe_vt_write);
      pipeline_v.consumer_wait(smem_pipe_v_read);

      copy(tiled_copy_trans, tVsV(_, _, _, smem_pipe_v_read.index()), tVrV);
      __syncwarp();

      pipeline_v.consumer_release(smem_pipe_v_read);
      copy(tiled_copy_trans, tVrV, tVsVt(_, _, _, smem_pipe_vt_write.index()));
      __syncwarp();

      pipeline_vt.producer_commit(smem_pipe_vt_write);
      ++smem_pipe_v_read;
      ++smem_pipe_vt_write;
    }
  }

  template <class BarrierStorage, class BlockCoord, class PipelineQv, class PipelineVt>
  MUTE_DEVICE auto mma(Params const&      params,
                       MainloopPipelineQ& pipeline_q,
                       PipelineQv&        pipeline_qv,
                       MainloopPipelineK& pipeline_k,
                       MainloopPipelineV& pipeline_v,
                       PipelineVt&        pipeline_vt,
                       PipelineQState&    smem_pipe_read_q,
                       PipelineQvState&   smem_pipe_read_qv,
                       PipelineKState&    smem_pipe_read_k,
                       PipelineVState&    smem_pipe_read_v,
                       PipelineVtState&   smem_pipe_read_vt,
                       SharedStorage&     shared_storage,
                       BarrierStorage*    barrier_storage,
                       SeqlenInfo const&  seqlen_info,
                       BlockCoord         blk_coord,
                       int const          thread_idx,
                       int&               work_idx,
                       int const          num_splits) {
    int const m_block   = get<0>(blk_coord);
    int const bidh      = get<1>(blk_coord);
    int const bidb      = get<2>(blk_coord);
    int const split_idx = get<3>(blk_coord);
    int const bidh_kv   = !IsPackGQA ? bidh / HeadRatio : bidh;

    auto [n_block_min, n_block_max] = BlockInfo::get_n_block_min_max(seqlen_info,
                                                                     m_block,
                                                                     split_idx,
                                                                     /*splits*/ num_splits,
                                                                     params.window_size_left,
                                                                     params.window_size_right,
                                                                     params.attention_chunk_divmod);

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    TiledMmaQv tiled_mma_qv;

    Tensor acc_pv    = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePDV{}));
    auto   acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    constexpr int Rows = size<0>(layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    // If invalid, return empty result.
    if (n_block_min >= n_block_max) {
      auto lse = make_tensor<float>(Shape<Int<Rows>>{});
      return mute::make_tuple(false, mute::make_tuple(acc_pv, lse));
    }

    int const thread_idx_in_warpgroup = thread_idx % mutlass::NumThreadsPerWarpSquad;
    int const warpgroup_idx           = thread_idx / mutlass::NumThreadsPerWarpSquad;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{});
    Tensor sV =
        make_tensor(make_smem_ptr(InKernelTranspose ? shared_storage.smem_vt.data() : shared_storage.smem_v.data()),
                    SmemLayoutVMmaPV{});

    Tensor sQv     = make_tensor(make_smem_ptr(shared_storage.smem_qv.data()), SmemLayoutQv{});
    Tensor sVMmaQV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVMmaQV{});

    ThrMMA thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    ThrMMA thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
    ThrMMA thr_mma_qv = tiled_mma_qv.get_thread_slice(thread_idx);

    Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
    Tensor tOrP = thr_mma_pv.partition_fragment_A(sP);
    Tensor tOrV = thr_mma_pv.partition_fragment_B(sV);

    Tensor tSrQv = thr_mma_qv.partition_fragment_A(sQv);
    Tensor tSrV  = thr_mma_qv.partition_fragment_B(sVMmaQV);

    auto wait_pv = [&]() {
      if constexpr (InKernelTranspose) {
        pipeline_vt.consumer_wait(smem_pipe_read_vt);
      } else {
        pipeline_v.consumer_wait(smem_pipe_read_v);
      }
    };

    auto gemm_pv = [&]() {
      if constexpr (InKernelTranspose) {
        mute::gemm(tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_vt.index()), acc_pv);
      } else {
        mute::gemm(tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), acc_pv);
      }
    };

    auto release_pv = [&]() {
      if constexpr (InKernelTranspose) {
        pipeline_vt.consumer_release(smem_pipe_read_vt);
        ++smem_pipe_read_vt;
      } else {
        pipeline_v.consumer_release(smem_pipe_read_v);
        ++smem_pipe_read_v;
      }
    };

    R2STiledCopy tiled_copy_r2s;
    ThrCopy      thr_copy_r2s = tiled_copy_r2s.get_thread_slice(thread_idx);
    Tensor       tPsP         = thr_copy_r2s.partition_D(sP);

    float effective_scale      = HasSoftcap ? 1.0f : params.softmax_scale;
    float effective_scale_log2 = HasSoftcap ? static_cast<float>(M_LOG2E) : params.softmax_scale_log2;
    float softcap_val          = params.softcap_val;
    float softcap_scale        = params.softmax_scale / params.softcap_val;
    float qk_descale           = 1.f;
    float qv_descale           = 1.f;

    if constexpr (IsFP8) {
      float const q_descale = [&] {
        if constexpr (!HasQDescale) {
          return 1.f;
        } else {
          auto q_index = bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale);
          return params.ptr_q_descale[q_index];
        }
      }();
      float const k_descale = [&] {
        if constexpr (!HasKDescale) {
          return 1.f;
        } else {
          auto k_index = bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale);
          return params.ptr_k_descale[k_index];
        }
      }();
      float const v_descale = [&] {
        if constexpr (!HasVDescale) {
          return 1.f;
        } else {
          auto v_index = bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale);
          return params.ptr_v_descale[v_index];
        }
      }();

      qk_descale = q_descale * k_descale;
      qv_descale = q_descale * v_descale;

      if constexpr (!HasQv) {
        if constexpr (HasSoftcap) {
          softcap_scale *= qk_descale;
        } else {
          effective_scale *= qk_descale;
          effective_scale_log2 *= qk_descale;
        }
      }
    }

    Softmax<Rows, HasLearnableSink, MaxOffset> softmax{effective_scale, effective_scale_log2};

    auto write_P_to_smem = [&](auto& accum_cvt) {
      Tensor tPrP = thr_copy_r2s.retile_S(accum_cvt);
      copy(tiled_copy_r2s, tPrP, tPsP);
      // TODO: remote sync
    };

    auto arrive_on_P_write_barrier = [&] {
      __syncwarp();
      // TODO: remote sync
    };

    auto release_qv_v = [&]() {
      if constexpr (HasQv) {
        pipeline_v.consumer_release(smem_pipe_read_v);
        ++smem_pipe_read_v;
      }
    };

    auto gemm_qv = [&](auto& acc_qk) {
      if constexpr (HasQv) {
        pipeline_v.consumer_wait(smem_pipe_read_v);
        if constexpr (IsFP8) {
          Tensor acc_qv = make_fragment_like(acc_qk);
          clear(acc_qv);
          mute::gemm(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read_v.index()), acc_qv);

          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(acc_qk); ++i) {
            acc_qk(i) = acc_qk(i) * qk_descale + acc_qv(i) * qv_descale;
          }
        } else {
          mute::gemm(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read_v.index()), acc_qk);
        }
      }
    };

    auto apply_softcap = [&](auto& acc_qk) {
      if constexpr (HasSoftcap) {
        // float const scale_times_inv_softcap = params.softmax_scale / softcap_val;
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_qk); ++i) {
          acc_qk(i) = softcap_val * mutlass::fast_tanh(acc_qk(i) * softcap_scale);
        }
      }
    };

    int const seqlen_q = seqlen_info.seqlen_q;
    int const seqlen_k = seqlen_info.seqlen_k;

    // Q is ready
    if constexpr (!IsAppendKV || !IsRotary) {
      pipeline_q.consumer_wait(smem_pipe_read_q);
    } else {  // Rotary
      using Rotary_t = Rotary<TileM,
                              HeadDimQK,
                              NumQKMmaThreads,
                              Element,
                              FragmentSize,
                              !(IsCausal || IsLocal) /*FixedPosition*/,
                              HeadRatio>;
      Rotary_t rotary{params.ptr_rotary_cos,
                      params.shape_rotary,
                      params.stride_rotary_cos,
                      params.ptr_rotary_sin,
                      params.stride_rotary_sin,
                      thread_idx,
                      seqlen_q,
                      seqlen_info.seqlen_rotary};

      Tensor sQ_pi = mute::as_position_independent_swizzle_tensor(sQ);

      auto [tRrCos, tRrSin] =
          conditional_return<!IsPackGQA>(rotary.template load_cos_sin<IsRotaryInterleaved /*IsInterleaved*/>(
                                             m_block, params.desc_Cos, params.desc_Sin),
                                         rotary.template load_cos_sin_packgqa<IsRotaryInterleaved /*IsInterleaved*/>(
                                             m_block, params.desc_Cos, params.desc_Sin));
      pipeline_q.consumer_wait(smem_pipe_read_q);
      if constexpr (IsRotaryInterleaved) {
        rotary.apply_Q_interleaved(sQ_pi, tRrCos, tRrSin, m_block);
      } else {
        rotary.apply_Q_contiguous(sQ_pi, tRrCos, tRrSin, m_block);
      }
      __syncwarp();
      named_barrier_sync(
          static_cast<uint32_t>(FwdNamedBarriers::RotaryQ));  // Ensure rotated Q is visible to all consumers
    }
    if constexpr (HasQv) {
      pipeline_qv.consumer_wait(smem_pipe_read_qv);
    }

    // Tensor acc_pv = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePDV{}));

    clear(acc_pv);

    // if (thread_idx == 0) {
    //   SHOW(tPsP);
    // }

    Mask<PermuteTiledMmaQK,
         SeqlenInfo,
         TileM,
         TileN,
         HeadRatio,
         IsPackGQA,
         IsCausal,
         IsLocal,
         HasAttentionChunk,
         EnableCP>
        mask(thread_idx,
             seqlen_info,
             params.window_size_left,
             params.window_size_right,
             params.sink_token_length,
             params.attention_chunk_divmod);

    int n_block = n_block_max - 1;

    if constexpr (IntraWarpSquadOverlap) {
      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
      // trigger zero init sqmma
      clear(acc_qk);

      pipeline_k.consumer_wait(smem_pipe_read_k);

      // MMA QK
      if constexpr (!OnlyQv) {
        mute::gemm(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), acc_qk);
      }

      // MMA QV
      gemm_qv(acc_qk);

      mate::warpsquad_commit_batch();
      mate::warpsquad_wait();

      pipeline_k.consumer_release(smem_pipe_read_k);
      ++smem_pipe_read_k;
      release_qv_v();

      apply_softcap(acc_qk);

      // Mask Mode
      mask.template apply</*SeqlenMask*/ true>(acc_qk, m_block, n_block);

      // Softmax
      Tensor correction_scales = softmax.template online_softmax<true, true>(acc_qk, tiled_mma_qk);

      Tensor accum_cvt = make_fragment_like<Element>(acc_qk);
      convert_type<CvtFragmentSize>(acc_qk, accum_cvt);

      if constexpr (!IsMmaPvRS) {
        write_P_to_smem(accum_cvt);
      }
      if constexpr (!IsMmaPvRS) {
        arrive_on_P_write_barrier();
      }

      --n_block;

      // Each step does gemm0 for iter n_block, gemm1 for iter n_block+1, and softmax for iter n_block
      auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
        static constexpr bool CheckInf = decltype(check_inf_type)::value;

        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
        // trigger zero init sqmma
        clear(acc_qk);

        pipeline_k.consumer_wait(smem_pipe_read_k);
        if constexpr (!OnlyQv) {
          mute::gemm(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), acc_qk);
        }

        gemm_qv(acc_qk);

        mate::warpsquad_commit_batch();

        wait_pv();
        gemm_pv();

        mate::warpsquad_commit_batch();

        // wait QK done
        mate::warpsquad_wait<1>();

        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;
        release_qv_v();

        apply_softcap(acc_qk);

        // Mask Mode
        mask_fn(acc_qk, n_block);

        // Softmax
        mute::copy(softmax.template online_softmax<false, CheckInf>(acc_qk, tiled_mma_qk), correction_scales);

        // wait PV done
        mate::warpsquad_wait<0>();

        release_pv();

        Tensor accum_cvt = make_fragment_like<Element>(acc_qk);
        convert_type<CvtFragmentSize>(acc_qk, accum_cvt);

        if constexpr (!IsMmaPvRS) {
          write_P_to_smem(accum_cvt);
        }
        softmax.rescale_o(acc_pv, tiled_mma_pv, correction_scales);
        if constexpr (!IsMmaPvRS) {
          arrive_on_P_write_barrier();
        }
      };

      // Causal/Local Masking
      if constexpr (IsCausal || IsLocal) {
        auto mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*seqlenk mask*/ false>(tSrS, m_block, n_block);
        };
        int const n_block_min_causal_local_mask = BlockInfo::get_n_block_min_causal_local_mask(
            seqlen_info, m_block, n_block_min, params.window_size_right, params.attention_chunk_divmod);

        for (; n_block >= n_block_min_causal_local_mask; --n_block) {
          fwd_step(n_block, mask_fn, /* CheckInf */ mute::true_type{});
        }
      }
      // No mask iterations
      int const n_block_min_before_local_mask = BlockInfo::get_n_block_min_before_local_mask(
          seqlen_info, m_block, n_block_min, params.window_size_left, params.attention_chunk_divmod);
      auto no_mask_fn = [](auto& tSrS, int n_block) {};
      for (; n_block >= n_block_min_before_local_mask; --n_block) {
        fwd_step(n_block, no_mask_fn, /* CheckInf */ mute::false_type{});
      }

      // Local mask iterations
      if constexpr (IsLocal) {
        auto local_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*SeqlenKMask*/ false>(tSrS, m_block, n_block);
        };
        for (; n_block >= n_block_min; --n_block) {
          fwd_step(n_block, local_mask_fn, /* CheckInf */ mute::true_type{});
        }
      }

      pipeline_q.consumer_release(smem_pipe_read_q);
      ++smem_pipe_read_q;

      // Last PV MMA
      wait_pv();
      gemm_pv();

      mate::warpsquad_commit_batch();
      mate::warpsquad_wait();
      release_pv();
    } else {
      auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
        static constexpr bool IsFirstIter = decltype(is_first_iter_type)::value;
        static constexpr bool CheckInf    = decltype(check_inf_type)::value;

        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
        // trigger zero init sqmma
        clear(acc_qk);

        pipeline_k.consumer_wait(smem_pipe_read_k);
        // MMA QK
        if constexpr (!OnlyQv) {
          mute::gemm(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), acc_qk);
        }

        gemm_qv(acc_qk);

        mate::warpsquad_commit_batch();
        mate::warpsquad_wait();
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;
        release_qv_v();

        apply_softcap(acc_qk);

        // Mask Mode
        mask_fn(acc_qk, n_block);

        // Softmax
        Tensor correction_scales = softmax.template online_softmax<IsFirstIter, CheckInf>(acc_qk, tiled_mma_qk);

        Tensor accum_cvt = make_fragment_like<Element>(acc_qk);
        convert_type<CvtFragmentSize>(acc_qk, accum_cvt);
        if constexpr (!IsMmaPvRS) {
          write_P_to_smem(accum_cvt);
        }
        if constexpr (!IsFirstIter) {
          softmax.rescale_o(acc_pv, tiled_mma_pv, correction_scales);
        }
        if constexpr (!IsMmaPvRS) {
          arrive_on_P_write_barrier();
        }

        wait_pv();

        // MMA PV
        gemm_pv();

        mate::warpsquad_commit_batch();
        mate::warpsquad_wait();
        release_pv();
      };

      auto first_iter_mask_fn = [&](auto& tSrS, int n_block) {
        mask.template apply</*seqlenk mask*/ true>(tSrS, m_block, n_block);
      };
      fwd_step(n_block, first_iter_mask_fn, /*IsFirstIter*/ mute::true_type{}, /*CheckInf*/ mute::true_type{});
      --n_block;

      // Causal/Local Masking
      if constexpr (IsCausal || IsLocal) {
        auto mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*seqlenk mask*/ false>(tSrS, m_block, n_block);
        };

        int const n_block_min_causal_local_mask = BlockInfo::get_n_block_min_causal_local_mask(
            seqlen_info, m_block, n_block_min, params.window_size_right, params.attention_chunk_divmod);

        for (; n_block >= n_block_min_causal_local_mask; --n_block) {
          fwd_step(n_block, mask_fn, /* IsFirstIter */ mute::false_type{}, /* CheckInf */ mute::true_type{});
        }
      }

      // No mask iterations
      int const n_block_min_before_local_mask = BlockInfo::get_n_block_min_before_local_mask(
          seqlen_info, m_block, n_block_min, params.window_size_left, params.attention_chunk_divmod);
      auto no_mask_fn = [](auto& tSrS, int n_block) {};
      for (; n_block >= n_block_min_before_local_mask; --n_block) {
        fwd_step(n_block, no_mask_fn, /* IsFirstIter */ mute::false_type{}, /* CheckInf */ mute::false_type{});
      }

      // Local mask iterations
      if constexpr (IsLocal) {
        auto local_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*SeqlenKMask*/ false>(tSrS, m_block, n_block);
        };
        for (; n_block >= n_block_min; --n_block) {
          fwd_step(n_block, local_mask_fn, /*IsFirstIter*/ mute::false_type{}, /* CheckInf */ mute::true_type{});
        }
      }

      // release q
      pipeline_q.consumer_release(smem_pipe_read_q);
      ++smem_pipe_read_q;
    }
    if constexpr (HasQv) {
      pipeline_qv.consumer_release(smem_pipe_read_qv);
      ++smem_pipe_read_qv;
    }
    ++work_idx;

    auto sink_vals = [&]() {
      if constexpr (IsPackGQA) {
        return make_tensor<ElementAccumulator>(Shape<Int<Rows>>{});
      } else {
        return make_tensor<ElementAccumulator>(Layout<Shape<Int<Rows>>, Stride<Int<0>>>{});
      }
    }();
    if constexpr (HasLearnableSink) {
      if constexpr (IsPackGQA) {
        int const  head_idx_qo_base = bidh * HeadRatio;
        auto const tScS_m           = mask.tScS_mn(_, _0{});
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Rows; ++i) {
          int const head_idx_qo = head_idx_qo_base + (m_block * TileM + get<0>(tScS_m(i))) % HeadRatio;
          sink_vals(i)          = static_cast<ElementAccumulator>(params.ptr_learnable_sink[head_idx_qo]);
        }
      } else {
        int const head_idx_qo = bidh;
        sink_vals(0)          = static_cast<ElementAccumulator>(params.ptr_learnable_sink[head_idx_qo]);
      }
    }

    float const v_descale =
        !IsFP8 || !HasVDescale
            ? 1.0f
            : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
    Tensor lse = softmax.tail(acc_pv, tiled_mma_pv, sink_vals, v_descale);

    return mute::make_tuple(true, mute::make_tuple(acc_pv, lse));
  }

  template <class BlockCoord>
  MUTE_DEVICE bool load_kv_new(Params const&          params,
                               MainloopPipelineKVNew& pipeline_k_new,
                               MainloopPipelineKVNew& pipeline_v_new,
                               PipelineKVNewState&    smem_pipe_write_kv_new,
                               SharedStorage&         shared_storage,
                               SeqlenInfo const&      seqlen_info,
                               BlockCoord             blk_coord,
                               int const              warp_idx_in_warp_squad,
                               int&                   work_idx,
                               int const              num_splits) {
    int const m_block   = get<0>(blk_coord);
    int const bidh      = get<1>(blk_coord);
    int const bidb      = get<2>(blk_coord);
    int const split_idx = get<3>(blk_coord);

    auto [n_block_new_min, n_block_new_max] = BlockInfo::get_n_block_k_new_min_max(seqlen_info,
                                                                                   m_block,
                                                                                   bidb,
                                                                                   split_idx,
                                                                                   num_splits,
                                                                                   params.window_size_left,
                                                                                   params.window_size_right,
                                                                                   params.attention_chunk_divmod);
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //   printf("MP=%d, bidm=%d, bidb=%d, bidh=%d, bids=%d, num_splits=%d, n_block_new_min=%d, n_block_new_max=%d\n",
    //          blockIdx.x,
    //          m_block,
    //          bidb,
    //          bidh,
    //          split_idx,
    //          num_splits,
    //          n_block_new_min,
    //          n_block_new_max);
    // }
    if (n_block_new_max <= n_block_new_min) {
      return false;
    }

    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVMmaPV{});

    int const bidh_kv = !IsPackGQA ? bidh / HeadRatio : bidh;

    Tensor mKnew =
        params.tme_load_K_new.get_tme_tensor(params.shape_K_new)(_, _, bidh_kv, !HasCuseqlensKNew ? bidb : 0);
    auto shape_Vnew = make_shape(
        params.headdim_V, get<0>(params.shape_K_new), get<2>(params.shape_K_new), get<3>(params.shape_K_new));
    Tensor mVnew = params.tme_load_V_new.get_tme_tensor(shape_Vnew)(_, _, bidh_kv, !HasCuseqlensKNew ? bidb : 0);

    Tensor gKnew = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mKnew),
                              select<1, 2>(TileShapeQKD{}),
                              make_coord(_, _0{}));  // (N, K, _)
    Tensor gVnew = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k_new), mVnew),
                              select<1, 2>(TileShapePDV{}),
                              make_coord(_0{}, _));  // (K_v, N, _)

    auto   cta_tme_K_new = params.tme_load_K_new.get_slice(0);
    Tensor tKgKnew       = group_modes<0, 3>(cta_tme_K_new.partition_S(gKnew));  // (TME, k)
    Tensor tKsKnew       = group_modes<0, 3>(cta_tme_K_new.partition_D(sK));     // (TME, pipe)

    auto   cta_tme_V_new = params.tme_load_V_new.get_slice(0);
    Tensor tVgVnew       = group_modes<0, 3>(cta_tme_V_new.partition_S(gVnew));  // (TME, k)
    Tensor tVsVnew       = group_modes<0, 3>(cta_tme_V_new.partition_D(sV));     // (TME, pipe)

    auto load_K_new = [&](int const n_block, auto const& smem_pipe_write) {
      pipeline_k_new.producer_acquire(smem_pipe_write);
      auto bar_id = pipeline_k_new.producer_get_barrier_id(smem_pipe_write);
      copy(params.tme_load_K_new.with(bar_id), tKgKnew(_, n_block), tKsKnew(_, smem_pipe_write.index()));
    };

    auto load_V_new = [&](int const n_block, auto const& smem_pipe_write) {
      pipeline_v_new.producer_acquire(smem_pipe_write);
      auto bar_id = pipeline_v_new.producer_get_barrier_id(smem_pipe_write);
      copy(params.tme_load_V_new.with(bar_id), tVgVnew(_, n_block), tVsVnew(_, smem_pipe_write.index()));
    };

    bool should_load_kv = SingleProducerWarp || warp_idx_in_warp_squad == 0;

    // pipeline_kv_guard.producer_acquire(smem_pipe_write_kv_guard);

    int n_block = n_block_new_max - 1;
    // Unlike the Hopper kernel, we don't need barrier_O here.
    // This kernel doesn't have the async O-side epilogue / cluster handoff that keeps
    // shared memory alive across stages, so load_kv_new doesn't need an extra recycle
    // barrier before reusing smem_k and smem_v.
    // Note: TME copies are issued by a producer warp, not a single elected thread,
    // so we intentionally don't use elect_one_sync() here.
    if (should_load_kv) {
      load_K_new(n_block, smem_pipe_write_kv_new);
      load_V_new(n_block, smem_pipe_write_kv_new);
    }
    ++smem_pipe_write_kv_new;
    --n_block;
    for (; n_block >= n_block_new_min; --n_block) {
      if (should_load_kv) {
        load_K_new(n_block, smem_pipe_write_kv_new);
        load_V_new(n_block, smem_pipe_write_kv_new);
      }
      ++smem_pipe_write_kv_new;
    }

    return true;
  }

  template <class BlockCoord>
  MUTLASS_DEVICE bool store_kv_new(Params const&          params,
                                   MainloopPipelineKVNew& pipeline_k_new,
                                   MainloopPipelineKVNew& pipeline_v_new,
                                   PipelineKVNewState&    smem_pipe_read_kv_new,
                                   int const              thread_idx,
                                   SharedStorage&         shared_storage,
                                   SeqlenInfo const&      seqlen_info,
                                   BlockCoord             blk_coord,
                                   int const              num_splits) {
    int const m_block                       = get<0>(blk_coord);
    int const bidh                          = get<1>(blk_coord);
    int const bidb                          = get<2>(blk_coord);
    int const split_idx                     = get<3>(blk_coord);
    auto [n_block_new_min, n_block_new_max] = BlockInfo::get_n_block_k_new_min_max(seqlen_info,
                                                                                   m_block,
                                                                                   bidb,
                                                                                   split_idx,
                                                                                   num_splits,
                                                                                   params.window_size_left,
                                                                                   params.window_size_right,
                                                                                   params.attention_chunk_divmod);

    if (n_block_new_max <= n_block_new_min) {
      return false;
    }

    Tensor sK = mute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{}));
    Tensor sV = mute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVLsu{}));

    int const bidh_kv = !IsPackGQA ? bidh / HeadRatio : bidh;
    int const bidb_kv = !HasKvBatchIdx ? bidb : params.kv_batch_idx[bidb];

    Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(
        _, _, bidh_kv, !HasCuseqlensK ? bidb_kv : 0);
    auto shape_V = make_shape(get<0>(params.shape_K), params.headdim_V, get<2>(params.shape_K), get<3>(params.shape_K));
    Tensor mV =
        make_tensor(make_gmem_ptr(params.ptr_V), shape_V, params.stride_V)(_, _, bidh_kv, !HasCuseqlensK ? bidb_kv : 0);

    int const offset_k = seqlen_info.offset_k + seqlen_info.seqlen_k_og;

    Tensor gK = local_tile(
        domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShapeQKD{}), make_coord(_, _0{}));  // (N, K, _)
    Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV),
                           select<2, 1>(TileShapePDV{}),
                           make_coord(_, _0{}));  // (N, K_v, _)

    int const seqlen_k_new = seqlen_info.seqlen_k_new;

    using Rotary_t = Rotary<TileN, HeadDimQK, NumMmaThreads, Element, FragmentSize>;

    Rotary_t rotary{params.ptr_rotary_cos,
                    params.shape_rotary,
                    params.stride_rotary_cos,
                    params.ptr_rotary_sin,
                    params.stride_rotary_sin,
                    thread_idx,
                    seqlen_k_new,
                    seqlen_info.seqlen_rotary};

    // This is used to index into the batch dimension of mK and mV
    int const bidb_kv_idx = !HasCuseqlensKNew && !IsPagedKV ? bidb_kv : 0;

    using KVManager = PagedKVManager<IsPagedKV,
                                     Element,
                                     NumMmaThreads,
                                     TileN,
                                     HeadDimQK,
                                     HeadDimVO,
                                     true /* IsKVSameIter */,
                                     2 /* LoadsPerRow_LB */>;

    // passing offset_k instead of leftpad_k will move the PageTable pointer to the right position
    KVManager paged_kv_manager{params.ptr_pagetable,
                               params.shape_pagetable,
                               params.stride_pagetable,
                               params.desc_page_table,
                               params.ptr_K,
                               params.shape_K,
                               params.stride_K,
                               params.desc_K,
                               params.ptr_V,
                               params.headdim_V,
                               params.stride_V,
                               params.desc_V,
                               params.page_size_divmod,
                               seqlen_k_new,
                               offset_k,  // seqlen_info.offset_k + seqlen_info.seqlen_k_og
                               thread_idx,
                               bidb_kv,
                               bidh_kv,
                               bidb_kv_idx};

    GmemTiledCopyAppendKV gmem_tiled_copy_kv;

    auto gmem_thr_copy_kv = gmem_tiled_copy_kv.get_thread_slice(thread_idx);

    Tensor tKgK = gmem_thr_copy_kv.partition_D(gK);
    Tensor tKsK = gmem_thr_copy_kv.partition_S(sK);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tVgV = gmem_thr_copy_kv.partition_D(gV);
    Tensor tVsV = gmem_thr_copy_kv.partition_S(sV);  // ((Atom,AtomNum),ATOM_M,ATOM_N)

    Tensor cK   = make_identity_tensor(select<1, 2>(TileShapeQKD{}));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_kv.partition_D(cK);
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKgK)));
    MUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(tKpK); ++k) {
      tKpK(k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(params.shape_K);
    }

    Tensor cV    = make_identity_tensor(select<2, 1>(TileShapePDV{}));  // (BLK_N,BLK_K_V) -> (blk_n,blk_k_v)
    Tensor tVcV  = conditional_return<SameHeadDim>(tKcK, gmem_thr_copy_kv.partition_D(cV));
    Tensor tVpV_ = make_tensor<bool>(make_shape(size<2>(tVsV)));
    MUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(tVpV_); ++k) {
      tVpV_(k) = get<1>(tVcV(_0{}, _0{}, k)) < params.headdim_V;
    }
    Tensor tVpV = conditional_return<SameHeadDim>(tKpK, tVpV_);

    static_assert(std::is_same_v<GmemLayoutAtomAppendKV, typename Rotary_t::LayoutAtom>);
    static_assert(!IsPagedKV || std::is_same_v<GmemLayoutAtomAppendKV, typename KVManager::GmemLayoutAtom>);

    auto store_K = [&](int const n_block, auto const& smem_pipe_read) {
      int const n_limit = std::min(seqlen_k_new - n_block * TileN, TileN);

      if constexpr (!IsRotary) {  // No rotary, smem -> rmem -> gmem directly
        pipeline_k_new.consumer_wait(smem_pipe_read);
        Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_read.index());
        Tensor tKrK     = make_fragment_like(tKsK_cur);  // ((_8,_1),_4,_2):((_1,_0),_8,_32)
        Tensor tKrK_src = gmem_thr_copy_kv.retile_S(tKrK);
        copy(tKsK_cur, tKrK);
        if constexpr (!IsPagedKV) {
          Tensor tKgK_cur = tKgK(_, _, _, n_block);
          MUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tKgK_cur); ++m) {
            bool row_valid = get<0>(tKcK(_0{}, m, _0{})) < n_limit;
            MUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < size<2>(tKgK_cur); ++k) {
              bool pred = row_valid && tKpK(k);
              copy(gmem_tiled_copy_kv.with(params.desc_K).with(pred), tKrK_src(_, m, k), tKgK_cur(_, m, k));
            }
          }
        } else {
          paged_kv_manager.store_K(n_block, tKrK_src);
        }
      } else {
        Tensor gK_cur  = gK(_, _, n_block);
        auto   tPrKPtr = conditional_return<IsPagedKV>(paged_kv_manager.compute_K_ptr(), nullptr);

        auto [tRrCos, tRrSin] = rotary.template load_cos_sin<IsRotaryInterleaved /*kInterleaved*/>(
            n_block, params.desc_Cos, params.desc_Sin);
        pipeline_k_new.consumer_wait(smem_pipe_read);
        if constexpr (IsRotaryInterleaved) {
          rotary.template apply_K_interleaved<IsPagedKV>(sK(_, _, smem_pipe_read.index()),
                                                         gK_cur,
                                                         tKpK,
                                                         tRrCos,
                                                         tRrSin,
                                                         tPrKPtr,
                                                         n_block,
                                                         get<1>(params.shape_K),
                                                         params.desc_K);
        } else {
          rotary.template apply_K_contiguous<IsPagedKV>(sK(_, _, smem_pipe_read.index()),
                                                        gK_cur,
                                                        tKpK,
                                                        tRrCos,
                                                        tRrSin,
                                                        tPrKPtr,
                                                        n_block,
                                                        get<1>(params.shape_K),
                                                        params.desc_K);
        }
      }

      pipeline_k_new.consumer_release(smem_pipe_read);
    };

    auto store_V = [&](int const n_block, auto const& smem_pipe_read) {
      int const n_limit = std::min(seqlen_k_new - n_block * TileN, TileN);

      pipeline_v_new.consumer_wait(smem_pipe_read);
      Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_read.index());
      Tensor tVrV     = make_fragment_like(tVsV_cur);
      Tensor tVrV_src = gmem_thr_copy_kv.retile_S(tVrV);
      copy(tVsV_cur, tVrV);
      if constexpr (!IsPagedKV) {
        Tensor tVgV_cur = tVgV(_, _, _, n_block);
        MUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<1>(tVgV_cur); ++m) {
          bool row_valid = get<0>(tVcV(_0{}, m, _0{})) < n_limit;
          MUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < size<2>(tVgV_cur); ++k) {
            bool pred = row_valid && tVpV(k);
            copy(gmem_tiled_copy_kv.with(params.desc_V).with(pred), tVrV_src(_, m, k), tVgV_cur(_, m, k));
          }
        }
      } else {
        paged_kv_manager.store_V(n_block, tVrV_src);
      }
      pipeline_v_new.consumer_release(smem_pipe_read);
    };

    // int n_block = 0;  // DEBUG ONLY
    int n_block = n_block_new_max - 1;
    if constexpr (IsPagedKV) {
      if constexpr (IsRotary) {
        paged_kv_manager.template load_page_table_for_lsu<true /* FirstIter */,
                                                          false /* PermuteK */,
                                                          false /* PermuteV */,
                                                          Rotary_t::GmemThreadsPerRow>(n_block);
      } else {
        paged_kv_manager
            .template load_page_table_for_lsu<true /* FirstIter */, false /* PermuteK */, false /* PermuteV */>(
                n_block);
      }
    }
    store_K(n_block, smem_pipe_read_kv_new);
    store_V(n_block, smem_pipe_read_kv_new);
    ++smem_pipe_read_kv_new;
    --n_block;

    for (; n_block >= n_block_new_min; --n_block) {
      if constexpr (IsPagedKV) {
        if constexpr (IsRotary) {
          paged_kv_manager.template load_page_table_for_lsu<false /* FirstIter */,
                                                            false /* PermuteK */,
                                                            false /* PermuteV */,
                                                            Rotary_t::GmemThreadsPerRow>(n_block);
        } else {
          paged_kv_manager
              .template load_page_table_for_lsu<false /* FirstIter */, false /* PermuteK */, false /* PermuteV */>(
                  n_block);
        }
      }
      store_K(n_block, smem_pipe_read_kv_new);
      store_V(n_block, smem_pipe_read_kv_new);
      ++smem_pipe_read_kv_new;
    }

    return true;
  }
};

}  // namespace mate::attention::fmha
