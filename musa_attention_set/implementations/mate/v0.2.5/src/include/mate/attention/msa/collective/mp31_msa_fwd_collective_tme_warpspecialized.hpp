#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <type_traits>

#include "mate/attention/fmha/load_primitive_builder.hpp"
#include "mate/attention/fmha/pipeline_ws.hpp"
#include "mate/attention/fmha/utils.hpp"
#include "mate/attention/msa/collective/msa_fwd_softmax.hpp"
#include "mate/attention/msa/msa_options.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/common/numeric_conversion.hpp"

namespace mate::attention::msa::collective {

using namespace mute;

// One CTA computes all 16 query heads associated with one (query, KV-head)
// pair. The sparse block list is read directly from q2k[query, KV-head, 16],
// so no K-to-Q CSR schedule or partial-output combine is involved.
template <class Element_, int QStages_ = 1, int KStages_ = 2, int VStages_ = 2, class... Options_>
struct Mp31MsaFwdCollectiveTmeWarpSpecialized {
  using Element            = Element_;
  using ElementAccumulator = float;

  static constexpr bool IsCausal      = find_option_t<Tag::IsCausal, std::false_type, Options_...>::value;
  static constexpr bool PackQueryPair = find_option_t<Tag::PackQueryPair, std::false_type, Options_...>::value;

  // SQMMA is physically M=16. The logical GQA ratio is runtime 8 or 16; for
  // ratio 8 the Q TME descriptor has extent 8 and hardware OOB constant-fill
  // supplies the unused physical rows.
  static constexpr int HeadRatio         = 16;
  static constexpr int TileM             = HeadRatio;
  static constexpr int TileN             = 128;
  static constexpr int HeadDim           = 128;
  static constexpr int SparseBlockSize   = 128;
  static constexpr int PageSize          = SparseBlockSize;
  static constexpr int TopK              = 16;
  static constexpr int PairUnionCapacity = 2 * TopK;
  static constexpr int QStages           = QStages_;
  static constexpr int KStages           = KStages_;
  static constexpr int VStages           = VStages_;

  static constexpr int NumThreadsPerWarp     = mutlass::NumThreadsPerWarp;
  static constexpr int WarpsPerWarpSquad     = mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumProducerWarpSquads = 1;
  static constexpr int NumMmaWarpSquads      = 1;
  static constexpr int NumProducerWarps      = NumProducerWarpSquads * WarpsPerWarpSquad;
  static constexpr int NumConsumerWarps      = NumMmaWarpSquads * WarpsPerWarpSquad;
  static constexpr int NumProducerThreads    = NumProducerWarps * NumThreadsPerWarp;
  static constexpr int NumConsumerThreads    = NumConsumerWarps * NumThreadsPerWarp;
  static constexpr int NumThreads            = NumProducerThreads + NumConsumerThreads;
  static constexpr int QLoadWarpInProducer   = 0;
  static constexpr int KVLoadWarpInProducer  = 1;
  static constexpr int SmemAlignmentBytes    = 256;
  // The key-load permutation must cover the full 128-row sparse block in one
  // TME block. FP16/BF16 therefore need a 256-bit fragment where FP8 needs
  // 128 bits; both choices contain 16 elements.
  static constexpr int KLoadVectorBits = 16 * sizeof_bits_v<Element>;

  static constexpr bool IsSupportedElement = std::is_same_v<Element, mutlass::float_e4m3_t> ||
                                             std::is_same_v<Element, mutlass::half_t> ||
                                             std::is_same_v<Element, mutlass::bfloat16_t>;
  static_assert(IsSupportedElement, "MSA forward supports FP8 E4M3, FP16, and BF16.");
  static_assert(QStages == 1, "Q is resident for one MSA forward work item and uses one stage.");
  static_assert(KStages >= 1 && VStages >= 1, "K/V TME pipelines require at least one stage.");
  static_assert(NumMmaWarpSquads == 1, "TileM=16 requires exactly one SQMMA consumer warp-squad.");

  using TileShapeQK = Shape<Int<TileM>, Int<TileN>, Int<HeadDim>>;
  using TileShapePV = Shape<Int<TileM>, Int<HeadDim>, Int<TileN>>;

  using AtomLayoutQK = Layout<Shape<_1, _1, _1>>;
  using TiledMmaQK   = decltype(mute::make_tiled_mma(
      mute::MP31::SQMMA::
          ss_op_selector<Element, Element, ElementAccumulator, TileShapeQK, TCE::Major::K, TCE::Major::K, Int<TileM>>(),
      AtomLayoutQK{}));
  static_assert(decltype(size(TiledMmaQK{}))::value == NumConsumerThreads);

  using AtomLayoutPV = Layout<Shape<_1, _1, _1>>;
  using TiledMmaPV   = decltype(mute::make_tiled_mma(mute::MP31::SQMMA::ss_op_selector<Element,
                                                                                       Element,
                                                                                       ElementAccumulator,
                                                                                       TileShapePV,
                                                                                       TCE::Major::K,
                                                                                       TCE::Major::MN,
                                                                                       Int<TileM>>(),
                                                   AtomLayoutPV{}));
  static_assert(decltype(size(TiledMmaPV{}))::value == NumConsumerThreads);

  using SmemAtomLayoutQ =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaQK::Atom::MMA_Op, TileShapeQK>());
  using SmemAtomLayoutK =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::K, Element, typename TiledMmaQK::Atom::MMA_Op, TileShapeQK>());
  using SmemAtomLayoutP =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaPV::Atom::MMA_Op, TileShapePV>());
  using SmemAtomLayoutV =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::MN, Element, typename TiledMmaPV::Atom::MMA_Op, TileShapePV>());

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemAtomLayoutQ{}, make_shape(shape<0>(TileShapeQK{}), shape<2>(TileShapeQK{}), Int<QStages>{})));
  using SmemLayoutK = decltype(tile_to_shape(
      SmemAtomLayoutK{}, make_shape(shape<1>(TileShapeQK{}), shape<2>(TileShapeQK{}), Int<KStages>{})));
  using SmemLayoutP =
      decltype(tile_to_shape(SmemAtomLayoutP{}, make_shape(shape<0>(TileShapePV{}), shape<2>(TileShapePV{}))));
  using SmemLayoutV = decltype(tile_to_shape(
      SmemAtomLayoutV{}, make_shape(shape<1>(TileShapePV{}), shape<2>(TileShapePV{}), Int<VStages>{})));

  // Q is physically [total_q, Hq, D]. The ordinary path maps one query's 16
  // heads to M16. The pair path maps (8 heads, 2 adjacent queries) to those
  // same 16 physical rows.
  using QRowTile       = std::conditional_t<PackQueryPair,
                                            Layout<Shape<_8, _2>, Stride<_1, _8>>,
                                            Layout<Shape<Int<HeadRatio>, _1>, Stride<_1, Int<HeadRatio>>>>;
  using SmemLayoutQTme = decltype(flatten(composition(SmemLayoutQ{}, make_tile(QRowTile{}, _, _))));

  using ShapeQTme  = Shape<int32_t, int32_t, int32_t, int32_t>;  // (head_local, token_q, d, head_kv)
  using StrideQTme = Stride<int64_t, int64_t, _1, int64_t>;
  using ShapeKTme  = Shape<int32_t, int32_t, int32_t, int32_t>;  // (token_in_page, d, head_kv, page)
  using StrideKTme = Stride<int64_t, _1, int64_t, int64_t>;
  using ShapeVTme  = Shape<int32_t, int32_t, int32_t, int32_t>;  // (d, token_in_page, head_kv, page)
  using StrideVTme = Stride<_1, int64_t, int64_t, int64_t>;

  using TmeTileShapeQ =
      std::conditional_t<PackQueryPair, Shape<_8, _2, Int<HeadDim>>, Shape<Int<HeadRatio>, _1, Int<HeadDim>>>;
  using TmeTileShapeV = Shape<Int<HeadDim>, Int<TileN>>;

  using TME_Q = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQTme{}, int32_t(0)), StrideQTme{}),
      take<0, 3>(SmemLayoutQTme{}),
      TmeTileShapeQ{}));

  using TmeLoadKBuilder = ::mate::attention::fmha::Mp31FmhaTmeLoadKeyBuilder<Element,
                                                                             SmemLayoutK,
                                                                             StrideKTme,
                                                                             TME::CacheHint::CACHE_NORMAL,
                                                                             TME::CacheHint::CACHE_NORMAL,
                                                                             KLoadVectorBits>;
  using TME_K           = typename TmeLoadKBuilder::TME_K;
  using TmeTileShapeK   = typename TmeLoadKBuilder::TmeKTileShape;
  using PermutedShapeK  = decltype(TmeLoadKBuilder::get_permuted_shape(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideKTme{}, int32_t(0)), StrideKTme{})));

  using TME_V = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideVTme{}, int32_t(0)), StrideVTme{}),
      take<0, 2>(SmemLayoutV{})));

  // P is converted and stored through 128-bit register/shared-memory vectors.
  // This is independent of the wider 16-bit K TME permutation above.
  static constexpr int CvtFragmentSize = 128 / sizeof_bits_v<Element>;
  using FragmentTypeR2S                = mute::uint_bit_t<128>;
  static_assert(TmeLoadKBuilder::Fragment == 16, "K TME permutation must span 16 elements.");
  static_assert(CvtFragmentSize * sizeof_bits_v<Element> == 128, "P conversion must use 128-bit fragments.");
  static_assert(TmeLoadKBuilder::Fragment % CvtFragmentSize == 0,
                "P conversion fragments must evenly partition the K permutation.");
  using PermuteTileForQK = Tile<Underscore, typename TmeLoadKBuilder::MmaPermuteTile, Underscore>;
  using PermuteTiledMmaQK =
      decltype(::mate::attention::fmha::convert_to_permuted_mma(TiledMmaQK{}, PermuteTileForQK{}));
  using R2SCopyAtom  = Copy_Atom<UniversalCopy<FragmentTypeR2S>, Element>;
  using R2STiledCopy = decltype(make_tiled_copy_C(R2SCopyAtom{}, PermuteTiledMmaQK{}));

  using MainloopPipelineQ = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<QStages>;
  using MainloopPipelineK = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<KStages>;
  using MainloopPipelineV = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<VStages>;
  using PipelineQParams   = typename MainloopPipelineQ::Params;
  using PipelineKParams   = typename MainloopPipelineK::Params;
  using PipelineVParams   = typename MainloopPipelineV::Params;
  using PipelineQState    = typename MainloopPipelineQ::PipelineState;
  using PipelineKState    = typename MainloopPipelineK::PipelineState;
  using PipelineVState    = typename MainloopPipelineV::PipelineState;

  static constexpr int TmeTransactionBytesQ =
      mutlass::bits_to_bytes(size(take<0, 3>(SmemLayoutQTme{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesK =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesV =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutV{})) * sizeof_bits_v<Element>);

  static_assert(TmeTransactionBytesQ == TileM * HeadDim * int(sizeof(Element)));
  static_assert(TmeTransactionBytesK == TileN * HeadDim * int(sizeof(Element)));
  static_assert(TmeTransactionBytesV == TileN * HeadDim * int(sizeof(Element)));

  struct PairUnionStorage {
    alignas(16) int blocks[PairUnionCapacity];
    alignas(16) uint32_t masks[PairUnionCapacity];
    int count;
  };
  struct EmptyPairUnionStorage {};
  using PairUnionStorageT = std::conditional_t<PackQueryPair, PairUnionStorage, EmptyPairUnionStorage>;

  struct SharedStorage {
    mute::array_aligned<Element, cosize_v<SmemLayoutQ>, SmemAlignmentBytes> smem_q;
    mute::array_aligned<Element, cosize_v<SmemLayoutK>, SmemAlignmentBytes> smem_k;
    mute::array_aligned<Element, cosize_v<SmemLayoutP>, SmemAlignmentBytes> smem_p;
    mute::array_aligned<Element, cosize_v<SmemLayoutV>, SmemAlignmentBytes> smem_v;
    PairUnionStorageT                                                       pair_union;
  };
  static_assert(sizeof(SharedStorage) <= 196608, "MSA forward exceeds MP31 shared-memory capacity.");

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t pipeline_q[MainloopPipelineQ::NumBarriers];
    uint8_t pipeline_k[MainloopPipelineK::NumBarriers];
    uint8_t pipeline_v[MainloopPipelineV::NumBarriers];
  };

  // ABI expected by the generated binding:
  //   Q:   [total_q, Hq, 128], contiguous
  //   K/V: [num_pages, 128, Hkv, 128], contiguous
  //   q2k: [total_q, Hkv, 16], contiguous int32
  // page_table is either [batch, max_pages] or a flat array accompanied by
  // kv_page_indptr[batch + 1].
  // FP8 cache scales follow the SGLang per-tensor contract: K is restored by
  // folding k_scale into the logits scale, while v_scale is folded into the
  // existing final softmax normalization before O is cast to Element.
  struct Arguments {
    Element const* ptr_q              = nullptr;
    Element const* ptr_k              = nullptr;
    Element const* ptr_v              = nullptr;
    int32_t const* ptr_q2k            = nullptr;
    int32_t const* ptr_page_table     = nullptr;
    int32_t const* ptr_kv_page_indptr = nullptr;
    float          softmax_scale      = 1.0f;
    float          k_scale            = 1.0f;
    float          v_scale            = 1.0f;
  };

  struct TensorParams {
    Element const* ptr_q;
    Element const* ptr_k;
    Element const* ptr_v;
    int32_t const* ptr_q2k;
    int32_t const* ptr_page_table;
    int32_t const* ptr_kv_page_indptr;
    int            total_q;
    int            total_k;
    int            num_qo_heads;
    int            num_kv_heads;
    int            num_pages;
    int            max_pages_per_batch;
    float          softmax_scale;
    float          softmax_scale_log2;
    float          v_scale;
  };

  struct TmeLoadQParams {
    TME_Q     tme_load;
    ShapeQTme shape;
  };

  struct TmeLoadKParams {
    TME_K          tme_load;
    PermutedShapeK shape;
  };

  struct TmeLoadVParams {
    TME_V     tme_load;
    ShapeVTme shape;
  };

  struct Params {
    TensorParams   args;
    TmeLoadQParams load_q;
    TmeLoadKParams load_k;
    TmeLoadVParams load_v;
  };

  struct Pipeline {
    MainloopPipelineQ q;
    MainloopPipelineK k;
    MainloopPipelineV v;
    PipelineQState    q_read;
    PipelineQState    q_write;
    PipelineKState    k_read;
    PipelineKState    k_write;
    PipelineVState    v_read;
    PipelineVState    v_write;

    static MUTLASS_DEVICE PipelineQParams make_q_params() {
      PipelineQParams params{};
      params.transaction_bytes = TmeTransactionBytesQ;
      params.num_consumers     = NumConsumerWarps;
      return params;
    }

    static MUTLASS_DEVICE PipelineKParams make_k_params() {
      PipelineKParams params{};
      params.transaction_bytes = TmeTransactionBytesK;
      params.num_consumers     = NumConsumerWarps;
      return params;
    }

    static MUTLASS_DEVICE PipelineVParams make_v_params() {
      PipelineVParams params{};
      params.transaction_bytes = TmeTransactionBytesV;
      params.num_consumers     = NumConsumerWarps;
      return params;
    }

    MUTLASS_DEVICE explicit Pipeline(BarrierStorage* barrier_storage)
        : q(make_q_params(), static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->pipeline_q))),
          k(make_k_params(), static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->pipeline_k))),
          v(make_v_params(), static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->pipeline_v))),
          q_read{},
          q_write(mutlass::make_producer_start_state_warpspecialized<MainloopPipelineQ>()),
          k_read{},
          k_write(mutlass::make_producer_start_state_warpspecialized<MainloopPipelineK>()),
          v_read{},
          v_write(mutlass::make_producer_start_state_warpspecialized<MainloopPipelineV>()) {
    }
  };

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    int const num_pages           = mutlass::ceil_div(problem_size.total_k, PageSize);
    int const max_pages_per_batch = mutlass::ceil_div(problem_size.max_seqlen_k, PageSize);

    int const  logical_head_ratio = problem_size.num_qo_heads / problem_size.num_kv_heads;
    ShapeQTme  shape_q  = make_shape(logical_head_ratio, problem_size.total_q, HeadDim, problem_size.num_kv_heads);
    StrideQTme stride_q = make_stride(
        int64_t(HeadDim), int64_t(problem_size.num_qo_heads) * HeadDim, _1{}, int64_t(logical_head_ratio) * HeadDim);

    ShapeKTme  shape_k  = make_shape(PageSize, HeadDim, problem_size.num_kv_heads, num_pages);
    StrideKTme stride_k = make_stride(int64_t(problem_size.num_kv_heads) * HeadDim,
                                      _1{},
                                      int64_t(HeadDim),
                                      int64_t(PageSize) * problem_size.num_kv_heads * HeadDim);

    ShapeVTme  shape_v  = make_shape(HeadDim, PageSize, problem_size.num_kv_heads, num_pages);
    StrideVTme stride_v = make_stride(_1{},
                                      int64_t(problem_size.num_kv_heads) * HeadDim,
                                      int64_t(HeadDim),
                                      int64_t(PageSize) * problem_size.num_kv_heads * HeadDim);

    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_q), shape_q, stride_q);
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_k), shape_k, stride_k);
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_v), shape_v, stride_v);

    constexpr float Log2e                   = 1.4426950408889634f;
    float const     effective_softmax_scale = args.softmax_scale * args.k_scale;
    TensorParams    tensor_args{
        args.ptr_q,
        args.ptr_k,
        args.ptr_v,
        args.ptr_q2k,
        args.ptr_page_table,
        args.ptr_kv_page_indptr,
        problem_size.total_q,
        problem_size.total_k,
        problem_size.num_qo_heads,
        problem_size.num_kv_heads,
        num_pages,
        max_pages_per_batch,
        effective_softmax_scale,
        effective_softmax_scale * Log2e,
        args.v_scale,
    };

    TmeLoadQParams load_q{
        make_tme_copy(MP31_TME_LOAD{}, mQ, take<0, 3>(SmemLayoutQTme{}), TmeTileShapeQ{}),
        shape_q,
    };
    TmeLoadKParams load_k{
        TmeLoadKBuilder::make_tme_copy(mK),
        TmeLoadKBuilder::get_permuted_shape(mK),
    };
    TmeLoadVParams load_v{
        make_tme_copy(MP31_TME_LOAD{}, mV, take<0, 2>(SmemLayoutV{})),
        shape_v,
    };
    return {tensor_args, load_q, load_k, load_v};
  }

  template <class ProblemSize>
  static bool can_implement(ProblemSize const& problem_size, Arguments const& args) {
    if (problem_size.total_q == 0) {
      return true;
    }
    if (problem_size.num_kv_heads <= 0) {
      return false;
    }
    int const ratio = problem_size.num_qo_heads / problem_size.num_kv_heads;
    if constexpr (PackQueryPair) {
      if (ratio != 8) {
        return false;
      }
    }
    return problem_size.batch_size > 0 && problem_size.total_k > 0 && problem_size.num_kv_heads > 0 &&
           problem_size.num_qo_heads % problem_size.num_kv_heads == 0 && (ratio == 8 || ratio == HeadRatio) &&
           problem_size.max_seqlen_q > 0 && problem_size.max_seqlen_k > 0 && args.ptr_q != nullptr &&
           args.ptr_k != nullptr && args.ptr_v != nullptr && args.ptr_q2k != nullptr && args.ptr_page_table != nullptr;
  }

  MUTLASS_DEVICE static int sparse_block(Params const& params, int q_abs, int head_kv, int topk_slot) {
    int64_t offset = (int64_t(q_abs) * params.args.num_kv_heads + int64_t(head_kv)) * TopK + int64_t(topk_slot);
    return params.args.ptr_q2k[offset];
  }

  template <class WorkTile>
  MUTLASS_DEVICE static bool valid_sparse_block(WorkTile const& work_tile, int logical_block) {
    return logical_block >= 0 && logical_block < mutlass::ceil_div(work_tile.kv_len, SparseBlockSize);
  }

  // Thread 0 builds a compact union for two adjacent TP8 queries.  The input
  // lists need not be sorted: an existing block simply gains the second
  // query's membership bit.  A masked dummy entry keeps the online-softmax
  // pipeline well-defined when both lists are empty.
  template <class WorkTile>
  MUTLASS_DEVICE static int build_pair_union(Params const&   params,
                                             WorkTile const& work_tile,
                                             int*            union_blocks,
                                             uint32_t*       union_masks) {
    static_assert(PackQueryPair, "pair-union metadata is only valid for the pair-query collective");
    int union_count = 0;
    if (work_tile.q_count == 2) {
      bool lists_equal = true;
      MUTLASS_PRAGMA_NO_UNROLL
      for (int topk_slot = 0; topk_slot < TopK; ++topk_slot) {
        lists_equal = lists_equal && sparse_block(params, work_tile.q_abs, work_tile.head_kv, topk_slot) ==
                                         sparse_block(params, work_tile.q_abs + 1, work_tile.head_kv, topk_slot);
      }
      if (lists_equal) {
        MUTLASS_PRAGMA_NO_UNROLL
        for (int topk_slot = 0; topk_slot < TopK; ++topk_slot) {
          int logical_block = sparse_block(params, work_tile.q_abs, work_tile.head_kv, topk_slot);
          if (!valid_sparse_block(work_tile, logical_block)) {
            continue;
          }
          uint32_t query_mask = 3u;
          if constexpr (IsCausal) {
            query_mask        = 0;
            int logical_begin = logical_block * SparseBlockSize;
            for (int query_in_tile = 0; query_in_tile < 2; ++query_in_tile) {
              int causal_limit = work_tile.q_local + query_in_tile + work_tile.qo_offset;
              if (logical_begin <= causal_limit) {
                query_mask |= uint32_t(1) << query_in_tile;
              }
            }
          }
          if (query_mask != 0) {
            union_blocks[union_count] = logical_block;
            union_masks[union_count]  = query_mask;
            ++union_count;
          }
        }
        if (union_count == 0) {
          union_blocks[0] = 0;
          union_masks[0]  = 0;
          union_count     = 1;
        }
        return union_count;
      }
    }

    // sparse_topk_select emits strictly ascending block ids followed by -1.
    // Merge that common form in O(TopK); retain the generic path below for
    // externally supplied lists that do not follow the ordering contract.
    bool lists_sorted = true;
    for (int query_in_tile = 0; query_in_tile < work_tile.q_count; ++query_in_tile) {
      int  previous_block = -1;
      bool saw_invalid    = false;
      int  q_abs          = work_tile.q_abs + query_in_tile;
      MUTLASS_PRAGMA_NO_UNROLL
      for (int topk_slot = 0; topk_slot < TopK; ++topk_slot) {
        int logical_block = sparse_block(params, q_abs, work_tile.head_kv, topk_slot);
        if (!valid_sparse_block(work_tile, logical_block)) {
          saw_invalid = true;
          continue;
        }
        if (saw_invalid || logical_block <= previous_block) {
          lists_sorted = false;
        }
        previous_block = logical_block;
      }
    }

    if (lists_sorted) {
      constexpr int Sentinel = 0x7fffffff;
      int           slot0    = 0;
      int           slot1    = 0;
      while (slot0 < TopK || (work_tile.q_count == 2 && slot1 < TopK)) {
        int block0 = slot0 < TopK ? sparse_block(params, work_tile.q_abs, work_tile.head_kv, slot0) : Sentinel;
        int block1 = work_tile.q_count == 2 && slot1 < TopK
                         ? sparse_block(params, work_tile.q_abs + 1, work_tile.head_kv, slot1)
                         : Sentinel;
        if (!valid_sparse_block(work_tile, block0)) {
          block0 = Sentinel;
        }
        if (!valid_sparse_block(work_tile, block1)) {
          block1 = Sentinel;
        }
        int logical_block = min(block0, block1);
        if (logical_block == Sentinel) {
          break;
        }

        uint32_t query_mask = 0;
        if (block0 == logical_block) {
          query_mask |= 1u;
          ++slot0;
        }
        if (block1 == logical_block) {
          query_mask |= 2u;
          ++slot1;
        }
        if constexpr (IsCausal) {
          int logical_begin = logical_block * SparseBlockSize;
          if (logical_begin > work_tile.q_local + work_tile.qo_offset) {
            query_mask &= ~1u;
          }
          if (logical_begin > work_tile.q_local + 1 + work_tile.qo_offset) {
            query_mask &= ~2u;
          }
        }
        if (query_mask != 0) {
          union_blocks[union_count] = logical_block;
          union_masks[union_count]  = query_mask;
          ++union_count;
        }
      }
      if (union_count == 0) {
        union_blocks[0] = 0;
        union_masks[0]  = 0;
        union_count     = 1;
      }
      return union_count;
    }

    for (int query_in_tile = 0; query_in_tile < work_tile.q_count; ++query_in_tile) {
      int q_abs = work_tile.q_abs + query_in_tile;
      for (int topk_slot = 0; topk_slot < TopK; ++topk_slot) {
        int logical_block = sparse_block(params, q_abs, work_tile.head_kv, topk_slot);
        if (!valid_sparse_block(work_tile, logical_block)) {
          continue;
        }
        if constexpr (IsCausal) {
          int causal_limit = work_tile.q_local + query_in_tile + work_tile.qo_offset;
          if (logical_block * SparseBlockSize > causal_limit) {
            continue;
          }
        }
        int union_slot = 0;
        for (; union_slot < union_count; ++union_slot) {
          if (union_blocks[union_slot] == logical_block) {
            break;
          }
        }
        uint32_t query_mask = uint32_t(1) << query_in_tile;
        if (union_slot < union_count) {
          union_masks[union_slot] |= query_mask;
        } else {
          union_blocks[union_count] = logical_block;
          union_masks[union_count]  = query_mask;
          ++union_count;
        }
      }
    }
    if (union_count == 0) {
      union_blocks[0] = 0;
      union_masks[0]  = 0;
      union_count     = 1;
    }
    return union_count;
  }

  template <class WorkTile>
  MUTLASS_DEVICE static int selected_block_count(WorkTile const& work_tile) {
    int visible_tokens = work_tile.kv_len;
    if constexpr (IsCausal) {
      visible_tokens = min(visible_tokens, max(work_tile.q_local + work_tile.qo_offset + 1, 0));
    }
    return max(1, min(TopK, mutlass::ceil_div(visible_tokens, SparseBlockSize)));
  }

  template <class WorkTile>
  MUTLASS_DEVICE static int64_t page_table_begin(Params const& params, WorkTile const& work_tile) {
    return params.args.ptr_kv_page_indptr != nullptr ? int64_t(params.args.ptr_kv_page_indptr[work_tile.batch_idx])
                                                     : int64_t(work_tile.batch_idx) * params.args.max_pages_per_batch;
  }

  template <class WorkTile>
  MUTLASS_DEVICE static int physical_page(Params const&   params,
                                          WorkTile const& work_tile,
                                          int64_t         page_begin,
                                          int             logical_block) {
    if (!valid_sparse_block(work_tile, logical_block)) {
      return 0;
    }
    int physical = params.args.ptr_page_table[page_begin + logical_block];
    return physical >= 0 ? physical : 0;
  }

  template <class WorkTile>
  static MUTLASS_DEVICE auto make_tme_q_gmem(TmeLoadQParams const& load_q, WorkTile const& work_tile) {
    Tensor mQHead = load_q.tme_load.get_tme_tensor(load_q.shape)(_, _, _, work_tile.head_kv);
    Tensor mQTile = domain_offset(make_coord(_0{}, work_tile.q_abs, _0{}), mQHead);
    return local_tile(mQTile, TmeTileShapeQ{}, make_coord(_0{}, _0{}, _0{}));
  }

  template <class WorkTile>
  static MUTLASS_DEVICE auto make_tme_k_gmem(TmeLoadKParams const& load_k,
                                             WorkTile const&       work_tile,
                                             int                   physical_page_idx) {
    Tensor mKPage = load_k.tme_load.get_tme_tensor(load_k.shape)(_, _, work_tile.head_kv, physical_page_idx);
    return local_tile(mKPage, TmeTileShapeK{}, make_coord(_0{}, _0{}));
  }

  template <class WorkTile>
  static MUTLASS_DEVICE auto make_tme_v_gmem(TmeLoadVParams const& load_v,
                                             WorkTile const&       work_tile,
                                             int                   physical_page_idx) {
    Tensor mVPage = load_v.tme_load.get_tme_tensor(load_v.shape)(_, _, work_tile.head_kv, physical_page_idx);
    return local_tile(mVPage, TmeTileShapeV{}, make_coord(_0{}, _0{}));
  }

  template <class WorkTile>
  MUTLASS_DEVICE void load_q(Params const&   params,
                             Pipeline&       pipeline,
                             SharedStorage&  shared_storage,
                             WorkTile const& work_tile) const {
    pipeline.q.producer_acquire(pipeline.q_write);
    uint32_t bar_id  = pipeline.q.producer_get_barrier_id(pipeline.q_write);
    auto     cta_tme = params.load_q.tme_load.get_slice(0);
    Tensor   gQ      = make_tme_q_gmem(params.load_q, work_tile);
    Tensor   sQ =
        make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQTme{})(_, _, _, pipeline.q_write.index());
    copy(params.load_q.tme_load.with(bar_id), cta_tme.partition_S(gQ), cta_tme.partition_D(sQ));
    ++pipeline.q_write;
  }

  template <class WorkTile>
  MUTLASS_DEVICE void issue_k(Params const&   params,
                              Pipeline&       pipeline,
                              SharedStorage&  shared_storage,
                              WorkTile const& work_tile,
                              int             physical_page_idx) const {
    pipeline.k.producer_acquire(pipeline.k_write);
    uint32_t bar_id  = pipeline.k.producer_get_barrier_id(pipeline.k_write);
    auto     cta_tme = params.load_k.tme_load.get_slice(0);
    Tensor   gK      = make_tme_k_gmem(params.load_k, work_tile, physical_page_idx);
    Tensor   sK      = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    copy(params.load_k.tme_load.with(bar_id),
         cta_tme.partition_S(gK),
         cta_tme.partition_D(sK(_, _, pipeline.k_write.index())));
    ++pipeline.k_write;
  }

  template <class WorkTile>
  MUTLASS_DEVICE void issue_v(Params const&   params,
                              Pipeline&       pipeline,
                              SharedStorage&  shared_storage,
                              WorkTile const& work_tile,
                              int             physical_page_idx) const {
    pipeline.v.producer_acquire(pipeline.v_write);
    uint32_t bar_id  = pipeline.v.producer_get_barrier_id(pipeline.v_write);
    auto     cta_tme = params.load_v.tme_load.get_slice(0);
    Tensor   gV      = make_tme_v_gmem(params.load_v, work_tile, physical_page_idx);
    Tensor   sV      = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
    copy(params.load_v.tme_load.with(bar_id),
         cta_tme.partition_S(gV),
         cta_tme.partition_D(sV(_, _, pipeline.v_write.index())));
    ++pipeline.v_write;
  }

  // Match the dense FMHA producer ordering: make K for the next QK step
  // available before issuing V for the previous PV step. K and V share the
  // same resolved sparse page, avoiding a second q2k/page-table lookup.
  template <class WorkTile>
  MUTLASS_DEVICE void load_kv(Params const&   params,
                              Pipeline&       pipeline,
                              SharedStorage&  shared_storage,
                              WorkTile const& work_tile) const {
    int64_t page_begin    = page_table_begin(params, work_tile);
    int     num_blocks    = selected_block_count(work_tile);
    int     logical_block = sparse_block(params, work_tile.q_abs, work_tile.head_kv, 0);
    int     previous_page = physical_page(params, work_tile, page_begin, logical_block);

    issue_k(params, pipeline, shared_storage, work_tile, previous_page);

    MUTLASS_PRAGMA_NO_UNROLL
    for (int topk_slot = 1; topk_slot < num_blocks; ++topk_slot) {
      logical_block    = sparse_block(params, work_tile.q_abs, work_tile.head_kv, topk_slot);
      int current_page = physical_page(params, work_tile, page_begin, logical_block);
      issue_k(params, pipeline, shared_storage, work_tile, current_page);
      issue_v(params, pipeline, shared_storage, work_tile, previous_page);
      previous_page = current_page;
    }

    issue_v(params, pipeline, shared_storage, work_tile, previous_page);
  }

  template <class WorkTile>
  MUTLASS_DEVICE void load_kv_union(Params const&   params,
                                    Pipeline&       pipeline,
                                    SharedStorage&  shared_storage,
                                    WorkTile const& work_tile,
                                    int const*      union_blocks,
                                    int             union_count) const {
    static_assert(PackQueryPair, "union loads are only valid for the pair-query collective");
    int64_t page_begin    = page_table_begin(params, work_tile);
    int     previous_page = physical_page(params, work_tile, page_begin, union_blocks[0]);

    issue_k(params, pipeline, shared_storage, work_tile, previous_page);

    MUTLASS_PRAGMA_NO_UNROLL
    for (int union_slot = 1; union_slot < union_count; ++union_slot) {
      int current_page = physical_page(params, work_tile, page_begin, union_blocks[union_slot]);
      issue_k(params, pipeline, shared_storage, work_tile, current_page);
      issue_v(params, pipeline, shared_storage, work_tile, previous_page);
      previous_page = current_page;
    }

    issue_v(params, pipeline, shared_storage, work_tile, previous_page);
  }

  MUTLASS_DEVICE void wait_q(Pipeline& pipeline) const {
    pipeline.q.consumer_wait(pipeline.q_read);
  }

  MUTLASS_DEVICE void release_q(Pipeline& pipeline) const {
    pipeline.q.consumer_release(pipeline.q_read);
    ++pipeline.q_read;
  }

  MUTLASS_DEVICE void wait_k(Pipeline& pipeline) const {
    pipeline.k.consumer_wait(pipeline.k_read);
  }

  MUTLASS_DEVICE void release_k(Pipeline& pipeline) const {
    pipeline.k.consumer_release(pipeline.k_read);
    ++pipeline.k_read;
  }

  MUTLASS_DEVICE void wait_v(Pipeline& pipeline) const {
    pipeline.v.consumer_wait(pipeline.v_read);
  }

  MUTLASS_DEVICE void release_v(Pipeline& pipeline) const {
    pipeline.v.consumer_release(pipeline.v_read);
    ++pipeline.v_read;
  }

  template <class WorkTile, class AccQK>
  MUTLASS_DEVICE void apply_qk_mask(WorkTile const& work_tile,
                                    int             logical_block,
                                    int             consumer_thread_idx,
                                    AccQK&          acc_qk) const {
    bool  block_valid   = valid_sparse_block(work_tile, logical_block);
    int   causal_limit  = work_tile.q_local + work_tile.qo_offset;
    int   logical_begin = logical_block * SparseBlockSize;
    float neg_inf       = -std::numeric_limits<float>::infinity();

    bool reject_block = !block_valid || logical_begin >= work_tile.kv_len;
    if constexpr (IsCausal) {
      reject_block = reject_block || logical_begin > causal_limit;
    }
    if (reject_block) {
      mute::fill(acc_qk, neg_inf);
      return;
    }

    bool whole_block_valid = logical_begin + TileN <= work_tile.kv_len;
    if constexpr (IsCausal) {
      whole_block_valid = whole_block_valid && logical_begin + TileN - 1 <= causal_limit;
    }
    if (whole_block_valid) {
      return;
    }

    TiledMmaQK tiled_mma_qk;
    auto       thr_mma_qk = tiled_mma_qk.get_thread_slice(consumer_thread_idx);
    Tensor     acc_qk_mn =
        make_tensor(acc_qk.data(), ::mate::attention::fmha::layout_acc_mn(tiled_mma_qk, acc_qk.layout()));
    Tensor cQK      = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));
    Tensor tCcQK    = thr_mma_qk.partition_C(cQK);
    Tensor tCcQK_mn = make_tensor(tCcQK.data(), ::mate::attention::fmha::layout_acc_mn(tiled_mma_qk, tCcQK.layout()));

    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(acc_qk_mn); ++m) {
      MUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < size<1>(acc_qk_mn); ++n) {
        int mma_col     = static_cast<int>(get<1>(tCcQK_mn(m, n)));
        int logical_col = (mma_col % (TileN / TmeLoadKBuilder::Fragment)) * TmeLoadKBuilder::Fragment +
                          mma_col / (TileN / TmeLoadKBuilder::Fragment);
        int  kv_local = logical_begin + logical_col;
        bool keep     = kv_local < work_tile.kv_len;
        if constexpr (IsCausal) {
          keep = keep && kv_local <= causal_limit;
        }
        if (!keep) {
          acc_qk_mn(m, n) = neg_inf;
        }
      }
    }
  }

  template <class WorkTile, class AccQK>
  MUTLASS_DEVICE void apply_pair_qk_mask(WorkTile const& work_tile,
                                         int             logical_block,
                                         uint32_t        membership_mask,
                                         int             consumer_thread_idx,
                                         AccQK&          acc_qk) const {
    static_assert(PackQueryPair, "pair masking is only valid for the pair-query collective");
    float neg_inf = -std::numeric_limits<float>::infinity();
    if (!valid_sparse_block(work_tile, logical_block)) {
      mute::fill(acc_qk, neg_inf);
      return;
    }

    int      logical_begin     = logical_block * SparseBlockSize;
    uint32_t active_query_mask = (uint32_t(1) << work_tile.q_count) - 1u;
    bool     all_rows_selected = (membership_mask & active_query_mask) == active_query_mask;
    bool     whole_block_valid = logical_begin + TileN <= work_tile.kv_len;
    if constexpr (IsCausal) {
      // Query 0 is the earliest row in the physical pair, so it gives the
      // strictest causal bound for a block that is valid for every active row.
      whole_block_valid = whole_block_valid && logical_begin + TileN - 1 <= work_tile.q_local + work_tile.qo_offset;
    }
    if (all_rows_selected && whole_block_valid) {
      return;
    }

    TiledMmaQK tiled_mma_qk;
    auto       thr_mma_qk = tiled_mma_qk.get_thread_slice(consumer_thread_idx);
    Tensor     acc_qk_mn =
        make_tensor(acc_qk.data(), ::mate::attention::fmha::layout_acc_mn(tiled_mma_qk, acc_qk.layout()));
    Tensor cQK      = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));
    Tensor tCcQK    = thr_mma_qk.partition_C(cQK);
    Tensor tCcQK_mn = make_tensor(tCcQK.data(), ::mate::attention::fmha::layout_acc_mn(tiled_mma_qk, tCcQK.layout()));

    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(acc_qk_mn); ++m) {
      int  physical_row          = static_cast<int>(get<0>(tCcQK_mn(m, 0)));
      int  query_in_tile         = physical_row / 8;
      bool row_selected          = query_in_tile < work_tile.q_count && ((membership_mask >> query_in_tile) & 1u) != 0;
      int  causal_limit          = work_tile.q_local + query_in_tile + work_tile.qo_offset;
      bool row_whole_block_valid = row_selected && logical_begin + TileN <= work_tile.kv_len;
      if constexpr (IsCausal) {
        row_whole_block_valid = row_whole_block_valid && logical_begin + TileN - 1 <= causal_limit;
      }
      if (row_whole_block_valid) {
        continue;
      }

      bool reject_row = !row_selected || logical_begin >= work_tile.kv_len;
      if constexpr (IsCausal) {
        reject_row = reject_row || logical_begin > causal_limit;
      }
      if (reject_row) {
        MUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size<1>(acc_qk_mn); ++n) {
          acc_qk_mn(m, n) = neg_inf;
        }
        continue;
      }

      MUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < size<1>(acc_qk_mn); ++n) {
        int mma_col     = static_cast<int>(get<1>(tCcQK_mn(m, n)));
        int logical_col = (mma_col % (TileN / TmeLoadKBuilder::Fragment)) * TmeLoadKBuilder::Fragment +
                          mma_col / (TileN / TmeLoadKBuilder::Fragment);
        int  kv_local = logical_begin + logical_col;
        bool keep     = kv_local < work_tile.kv_len;
        if constexpr (IsCausal) {
          keep = keep && kv_local <= causal_limit;
        }
        if (!keep) {
          acc_qk_mn(m, n) = neg_inf;
        }
      }
    }
  }

  template <bool UsePairUnion = false, class WorkTile>
  MUTLASS_DEVICE auto compute(Params const&   params,
                              Pipeline&       pipeline,
                              SharedStorage&  shared_storage,
                              WorkTile const& work_tile,
                              int             consumer_thread_idx,
                              int const*      union_blocks    = nullptr,
                              uint32_t const* union_masks     = nullptr,
                              int const*      union_count_ptr = nullptr) const {
    static_assert(!UsePairUnion || PackQueryPair,
                  "pair-union compute requires the pair-query collective specialization");
    wait_q(pipeline);

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    auto       thr_mma_qk = tiled_mma_qk.get_thread_slice(consumer_thread_idx);
    auto       thr_mma_pv = tiled_mma_pv.get_thread_slice(consumer_thread_idx);

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{})(_, _, pipeline.q_read.index());
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
    Tensor tSrP = thr_mma_pv.partition_fragment_A(sP);
    Tensor tSrV = thr_mma_pv.partition_fragment_B(sV);

    Tensor acc_pv = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePV{}));
    clear(acc_pv);
    auto acc_pv_mn = make_tensor(acc_pv.data(), ::mate::attention::fmha::layout_acc_mn(tiled_mma_pv, acc_pv.layout()));
    constexpr int              SoftmaxRows = decltype(size<0>(acc_pv_mn))::value;
    MsaFwdSoftmax<SoftmaxRows> softmax{params.args.softmax_scale, params.args.softmax_scale_log2};

    R2STiledCopy tiled_copy_r2s;
    auto         thr_copy_r2s = tiled_copy_r2s.get_thread_slice(consumer_thread_idx);
    Tensor       tPsP         = thr_copy_r2s.partition_D(mute::as_position_independent_swizzle_tensor(sP));

    auto get_logical_block = [&](int topk_slot) {
      int lane_idx      = consumer_thread_idx % NumThreadsPerWarp;
      int logical_block = 0;
      if (lane_idx == 0) {
        if constexpr (UsePairUnion) {
          logical_block = union_blocks[topk_slot];
        } else {
          logical_block = sparse_block(params, work_tile.q_abs, work_tile.head_kv, topk_slot);
        }
      }
      return __shfl_sync(uint32_t(-1), logical_block, 0);
    };
    auto get_membership_mask = [&](int topk_slot) {
      uint32_t membership_mask = 1;
      if constexpr (UsePairUnion) {
        int lane_idx = consumer_thread_idx % NumThreadsPerWarp;
        if (lane_idx == 0) {
          membership_mask = union_masks[topk_slot];
        }
        membership_mask = __shfl_sync(uint32_t(-1), membership_mask, 0);
      }
      return membership_mask;
    };
    int num_blocks = 0;
    if constexpr (UsePairUnion) {
      // The KV producer writes the metadata immediately before issuing the
      // first K TME. Waiting on that transaction also establishes visibility
      // without a CTA-wide barrier.
      wait_k(pipeline);
      num_blocks = *union_count_ptr;
    } else {
      num_blocks = selected_block_count(work_tile);
    }

    auto write_p = [&](auto const& acc_qk) {
      Tensor p_fragment = make_fragment_like<Element>(acc_qk);
      ::mate::attention::fmha::convert_type<CvtFragmentSize>(acc_qk, p_fragment);
      Tensor tPrP = thr_copy_r2s.retile_S(p_fragment);
      copy(tiled_copy_r2s, tPrP, tPsP);
      __syncwarp();
    };

    int logical_block = get_logical_block(0);
    if constexpr (!UsePairUnion) {
      wait_k(pipeline);
    }
    Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
    clear(acc_qk);
    mute::gemm(tiled_mma_qk, tSrQ, tSrK(_, _, _, pipeline.k_read.index()), acc_qk);
    mate::warpsquad_commit_batch();
    mate::warpsquad_wait();
    release_k(pipeline);

    Tensor correction_scales = make_tensor<ElementAccumulator>(Shape<Int<SoftmaxRows>>{});
    clear(correction_scales);
    if constexpr (UsePairUnion) {
      apply_pair_qk_mask(work_tile, logical_block, get_membership_mask(0), consumer_thread_idx, acc_qk);
    } else {
      apply_qk_mask(work_tile, logical_block, consumer_thread_idx, acc_qk);
    }
    if constexpr (UsePairUnion) {
      if (num_blocks == 1) {
        release_q(pipeline);
      }
    }
    mute::copy(softmax.template online_softmax<true, true>(acc_qk, tiled_mma_qk), correction_scales);
    write_p(acc_qk);
    // Q is only an operand of QK.  Releasing it here (for a one-block
    // selection) lets the producer start the next work item's Q TME while
    // this tile finishes its final PV operation.
    if constexpr (!UsePairUnion) {
      if (num_blocks == 1) {
        release_q(pipeline);
      }
    }

    MUTLASS_PRAGMA_NO_UNROLL
    for (int topk_slot = 1; topk_slot < num_blocks; ++topk_slot) {
      logical_block = get_logical_block(topk_slot);
      wait_k(pipeline);
      Tensor acc_qk_next = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
      clear(acc_qk_next);
      mute::gemm(tiled_mma_qk, tSrQ, tSrK(_, _, _, pipeline.k_read.index()), acc_qk_next);
      mate::warpsquad_commit_batch();

      wait_v(pipeline);
      mute::gemm(tiled_mma_pv, tSrP, tSrV(_, _, _, pipeline.v_read.index()), acc_pv);
      mate::warpsquad_commit_batch();

      mate::warpsquad_wait<1>();
      release_k(pipeline);

      // After the last QK MMA, no consumer instruction reads Q again.  Move
      // the release ahead of the row-wise softmax/PV tail so the producer can
      // overlap the next query's TME load with that work.
      if constexpr (!UsePairUnion) {
        if (topk_slot == num_blocks - 1) {
          release_q(pipeline);
        }
      }

      if constexpr (UsePairUnion) {
        apply_pair_qk_mask(work_tile, logical_block, get_membership_mask(topk_slot), consumer_thread_idx, acc_qk_next);
      } else {
        apply_qk_mask(work_tile, logical_block, consumer_thread_idx, acc_qk_next);
      }
      if constexpr (UsePairUnion) {
        if (topk_slot == num_blocks - 1) {
          release_q(pipeline);
        }
      }
      mute::copy(softmax.template online_softmax<false, true>(acc_qk_next, tiled_mma_qk), correction_scales);

      mate::warpsquad_wait<0>();
      release_v(pipeline);

      write_p(acc_qk_next);
      softmax.rescale_o(acc_pv, tiled_mma_pv, correction_scales);
    }

    wait_v(pipeline);
    mute::gemm(tiled_mma_pv, tSrP, tSrV(_, _, _, pipeline.v_read.index()), acc_pv);
    mate::warpsquad_commit_batch();
    mate::warpsquad_wait();
    release_v(pipeline);

    Tensor sink_values = make_tensor<ElementAccumulator>(Shape<Int<SoftmaxRows>>{});
    clear(sink_values);
    Tensor lse = softmax.tail(acc_pv, tiled_mma_pv, sink_values, params.args.v_scale);
    return mute::make_tuple(acc_pv, lse);
  }
};

}  // namespace mate::attention::msa::collective
