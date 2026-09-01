#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <limits>
#include <mute/algorithm/prefetch.hpp>
#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/arch/simd_mp31.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <type_traits>

#include "mate/attention/fmha/pipeline_ws.hpp"
#include "mate/attention/fmha/utils.hpp"
#include "mate/attention/msa/msa_options.hpp"
#include "mate/attention/msa/paged_kv.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"

namespace mate::attention::msa::collective {

using namespace mute;

template <class Element_, class TileShape_, int HeadRatio_, int QStages_, int KStages_, class... Options_>
struct Mp31MsaMaxScoreCollectiveTmeWarpSpecialized {
  using Element            = Element_;
  using ElementAccumulator = float;
  using TileShape          = TileShape_;

  static constexpr bool IsPagedKV = find_option_t<Tag::IsPagedKV, std::false_type, Options_...>::value;
  static constexpr bool IsCausal  = find_option_t<Tag::IsCausal, std::false_type, Options_...>::value;
  static constexpr int  HeadRatio = HeadRatio_;
  static constexpr int  QStages   = QStages_;
  static constexpr int  KStages   = KStages_;

  static constexpr int  TileQ              = get<0>(TileShape{});
  static constexpr int  TileKV             = get<1>(TileShape{});
  static constexpr int  HeadDim            = get<2>(TileShape{});
  static constexpr int  QTokensPerTile     = TileQ / HeadRatio;
  static constexpr int  SmemAlignmentBytes = 256;
  static constexpr bool EnableKPrefetch    = TileQ > 16 && KStages == 1;

  static constexpr int NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  static constexpr int NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;
  static constexpr int WarpsPerWarpSquad      = mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumProducerWarpSquads  = 1;
  static constexpr int NumMmaWarpSquads       = TileQ == 16 ? 1 : 4;
  static constexpr int NumProducerWarps       = NumProducerWarpSquads * WarpsPerWarpSquad;
  static constexpr int NumConsumerWarps       = NumMmaWarpSquads * WarpsPerWarpSquad;
  static constexpr int NumProducerThreads     = NumProducerWarps * NumThreadsPerWarp;
  static constexpr int NumConsumerThreads     = NumConsumerWarps * NumThreadsPerWarp;
  static constexpr int NumThreads             = NumProducerThreads + NumConsumerThreads;
  static constexpr int QLoadWarpInProducer    = 0;
  static constexpr int KLoadWarpInProducer    = 1;

  static_assert(HeadRatio > 0);
  static_assert(TileQ > 0 && TileKV == 128 && HeadDim > 0);
  static_assert(TileQ % HeadRatio == 0);
  static_assert(TileQ % NumMmaWarpSquads == 0);
  static_assert(QStages > 0 && KStages > 0);
  static_assert(HeadDim % 8 == 0);

  using AtomLayoutQK = Layout<Shape<Int<NumMmaWarpSquads>, _1, _1>>;
  using TiledMmaQK   = decltype(mute::make_tiled_mma(mute::MP31::SQMMA::ss_op_selector<Element,
                                                                                       Element,
                                                                                       ElementAccumulator,
                                                                                       TileShape,
                                                                                       TCE::Major::K,
                                                                                       TCE::Major::K,
                                                                                       Int<TileQ / NumMmaWarpSquads>>(),
                                                   AtomLayoutQK{}));
  static_assert(decltype(size(TiledMmaQK{}))::value == NumConsumerThreads);

  using SmemAtomLayoutQ =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaQK::Atom::MMA_Op, TileShape>());
  using SmemAtomLayoutK =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::K, Element, typename TiledMmaQK::Atom::MMA_Op, TileShape>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemAtomLayoutQ{},
                                             make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<QStages>{})));
  using SmemLayoutK = decltype(tile_to_shape(SmemAtomLayoutK{},
                                             make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<KStages>{})));

  using QRowTile       = Layout<Shape<Int<HeadRatio>, Int<QTokensPerTile>>, Stride<_1, Int<HeadRatio>>>;
  using SmemLayoutQTme = decltype(flatten(composition(SmemLayoutQ{}, make_tile(QRowTile{}, _, _))));

  using ShapeQTme       = Shape<int32_t, int32_t, int32_t, int32_t>;  // (head_local, token_q, d, head_kv)
  using StrideQTme      = Stride<int64_t, int64_t, _1, int64_t>;
  using ShapeKTme       = Shape<int32_t, int32_t, int32_t, int32_t>;  // (token_k, d, head_kv, page)
  using StrideKTme      = Stride<int64_t, _1, int64_t, int64_t>;
  using ShapePageTable  = Shape<int32_t, int32_t>;
  using StridePageTable = Stride<int64_t, _1>;
  using ShapeMaxScore   = Shape<int32_t, int32_t, int32_t>;  // (token_q, head_q, k_tile)
  using StrideMaxScore  = Stride<int64_t, int64_t, _1>;

  using TmeTileShapeQ = Shape<Int<HeadRatio>, Int<QTokensPerTile>, Int<HeadDim>>;
  using TmeTileShapeK = Shape<Int<TileKV>, Int<HeadDim>>;

  using TME_Q = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQTme{}, int32_t(0)), StrideQTme{}),
      take<0, 3>(SmemLayoutQTme{}),
      TmeTileShapeQ{}));

  using TME_K          = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideKTme{}, int32_t(0)), StrideKTme{}),
      take<0, 2>(SmemLayoutK{})));
  using PagedKvManager = mate::attention::msa::KVManager<IsPagedKV, TileKV>;

  using MainloopPipelineQ = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<QStages>;
  using MainloopPipelineK = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<KStages>;
  using PipelineQParams   = typename MainloopPipelineQ::Params;
  using PipelineKParams   = typename MainloopPipelineK::Params;
  using PipelineQState    = typename MainloopPipelineQ::PipelineState;
  using PipelineKState    = typename MainloopPipelineK::PipelineState;

  static constexpr int TmeTransactionBytesQ =
      mutlass::bits_to_bytes(size(take<0, 3>(SmemLayoutQTme{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesK =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * sizeof_bits_v<Element>);

  struct SharedStorage {
    mute::array_aligned<Element, cosize_v<SmemLayoutQ>, SmemAlignmentBytes> smem_q;
    mute::array_aligned<Element, cosize_v<SmemLayoutK>, SmemAlignmentBytes> smem_k;
  };

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t pipeline_q[MainloopPipelineQ::NumBarriers];
    uint8_t pipeline_k[MainloopPipelineK::NumBarriers];
  };

  struct Arguments {
    Element const* ptr_q;
    Element const* ptr_k;
    int32_t const* ptr_page_table;
    int32_t const* ptr_kv_page_indptr;
    int32_t const* ptr_cu_seqlens_q;
    int32_t const* ptr_cu_seqlens_k;
    int32_t const* ptr_qo_offset;
    float*         ptr_max_score;
  };

  struct TensorParams {
    Element const*  ptr_q;
    Element const*  ptr_k;
    int32_t const*  ptr_page_table;
    int32_t const*  ptr_kv_page_indptr;
    ShapePageTable  shape_page_table;
    StridePageTable stride_page_table;
    int32_t const*  ptr_cu_seqlens_q;
    int32_t const*  ptr_cu_seqlens_k;
    int32_t const*  ptr_qo_offset;
    float*          ptr_max_score;
    int             batch_size;
    int             total_q;
    int             total_k;
    int             num_qo_heads;
    int             num_kv_heads;
    int             max_seqlen_q;
    int             max_seqlen_k;
    int             max_k_tiles;
  };

  struct TmeLoadQParams {
    TME_Q     tme_load;
    ShapeQTme shape;
  };

  struct TmeLoadKParams {
    TME_K     tme_load;
    ShapeKTme shape;
  };

  struct Params {
    TensorParams   args;
    TmeLoadQParams load_q;
    TmeLoadKParams load_k;
  };

  struct Pipeline {
    MainloopPipelineQ q;
    MainloopPipelineK k;
    PipelineQState    q_read;
    PipelineQState    q_write;
    PipelineKState    k_read;
    PipelineKState    k_write;

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

    MUTLASS_DEVICE explicit Pipeline(BarrierStorage* barrier_storage)
        : q(make_q_params(), static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->pipeline_q))),
          k(make_k_params(), static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->pipeline_k))),
          q_read{},
          q_write(mutlass::make_producer_start_state_warpspecialized<MainloopPipelineQ>()),
          k_read{},
          k_write(mutlass::make_producer_start_state_warpspecialized<MainloopPipelineK>()) {
    }
  };

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    int max_pages_per_batch = mutlass::ceil_div(problem_size.max_seqlen_k, TileKV);
    int page_extent         = IsPagedKV ? mutlass::ceil_div(problem_size.total_k, TileKV) : 1;

    ShapeQTme  shape_q = make_shape(HeadRatio, problem_size.total_q, HeadDim, problem_size.num_kv_heads);
    StrideQTme stride_q =
        make_stride(int64_t(HeadDim), int64_t(problem_size.num_qo_heads) * HeadDim, _1{}, int64_t(HeadRatio) * HeadDim);

    ShapeKTme  shape_k{};
    StrideKTme stride_k{};
    if constexpr (IsPagedKV) {
      shape_k  = make_shape(TileKV, HeadDim, problem_size.num_kv_heads, page_extent);
      stride_k = make_stride(int64_t(problem_size.num_kv_heads) * HeadDim,
                             _1{},
                             int64_t(HeadDim),
                             int64_t(TileKV) * problem_size.num_kv_heads * HeadDim);
    } else {
      shape_k  = make_shape(problem_size.total_k, HeadDim, problem_size.num_kv_heads, 1);
      stride_k = make_stride(int64_t(problem_size.num_kv_heads) * HeadDim, _1{}, int64_t(HeadDim), int64_t(0));
    }

    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_q), shape_q, stride_q);
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_k), shape_k, stride_k);

    ShapePageTable  shape_page_table  = make_shape(problem_size.batch_size, max_pages_per_batch);
    StridePageTable stride_page_table = make_stride(int64_t(max_pages_per_batch), _1{});

    TensorParams tensor_args{
        args.ptr_q,
        args.ptr_k,
        IsPagedKV ? args.ptr_page_table : nullptr,
        IsPagedKV ? args.ptr_kv_page_indptr : nullptr,
        shape_page_table,
        stride_page_table,
        args.ptr_cu_seqlens_q,
        args.ptr_cu_seqlens_k,
        args.ptr_qo_offset,
        args.ptr_max_score,
        problem_size.batch_size,
        problem_size.total_q,
        problem_size.total_k,
        problem_size.num_qo_heads,
        problem_size.num_kv_heads,
        problem_size.max_seqlen_q,
        problem_size.max_seqlen_k,
        problem_size.max_k_tiles,
    };

    TmeLoadQParams load_q{
        make_tme_copy(MP31_TME_LOAD{}, mQ, take<0, 3>(SmemLayoutQTme{}), TmeTileShapeQ{}),
        shape_q,
    };
    TmeLoadKParams load_k{
        make_tme_copy(MP31_TME_LOAD{}, mK, take<0, 2>(SmemLayoutK{})),
        shape_k,
    };
    return {tensor_args, load_q, load_k};
  }

  template <class WorkTile>
  static MUTLASS_DEVICE int safe_page_idx(Params const& params, WorkTile const& work_tile) {
    if constexpr (IsPagedKV) {
      if (work_tile.k_tile_begin < work_tile.kv_len && work_tile.k_tile_idx < get<1>(params.args.shape_page_table)) {
        PagedKvManager manager = PagedKvManager::create(params.args.ptr_page_table,
                                                        params.args.ptr_kv_page_indptr,
                                                        get<0>(params.args.stride_page_table),
                                                        params.args.ptr_cu_seqlens_k,
                                                        nullptr);
        return manager.physical_block_index(work_tile.batch_idx, work_tile.k_tile_idx);
      }
    }
    return 0;
  }

  template <class WorkTile>
  static MUTLASS_DEVICE auto make_tme_q_gmem(TmeLoadQParams const& load_q, WorkTile const& work_tile) {
    Tensor mQ_head = load_q.tme_load.get_tme_tensor(load_q.shape)(_, _, _, work_tile.head_kv);
    Tensor mQ_tile = domain_offset(make_coord(_0{}, work_tile.q_abs_begin, _0{}), mQ_head);
    return local_tile(mQ_tile, TmeTileShapeQ{}, make_coord(_0{}, _0{}, _0{}));
  }

  template <class WorkTile>
  static MUTLASS_DEVICE auto make_tme_k_gmem(TmeLoadKParams const& load_k, WorkTile const& work_tile, int page_idx) {
    if constexpr (IsPagedKV) {
      Tensor mK_paged = load_k.tme_load.get_tme_tensor(load_k.shape)(_, _, work_tile.head_kv, page_idx);
      return local_tile(mK_paged, TmeTileShapeK{}, make_coord(_0{}, _0{}));
    } else {
      Tensor mK_contiguous = load_k.tme_load.get_tme_tensor(load_k.shape)(_, _, work_tile.head_kv, _0{});
      Tensor mK_tile = domain_offset(make_coord(work_tile.k_batch_begin + work_tile.k_tile_begin, _0{}), mK_contiguous);
      return local_tile(mK_tile, TmeTileShapeK{}, make_coord(_0{}, _0{}));
    }
  }

  template <class WorkTile>
  static MUTLASS_DEVICE int k_tile_count(Params const& params, WorkTile const& work_tile) {
    int k_tiles = work_tile.valid_k_tiles;
    if constexpr (IsCausal) {
      // A Q tile only needs keys visible to its last valid query.  The prior
      // implementation computed every K tile and masked roughly half of the
      // scores back to -inf during causal prefill.  The output is initialized
      // to -inf by the caller, so future K tiles can be skipped altogether.
      int last_q_local   = work_tile.q_local_begin + work_tile.q_count - 1;
      int visible_keys   = last_q_local + params.args.ptr_qo_offset[work_tile.batch_idx] + 1;
      int causal_k_tiles = visible_keys > 0 ? mutlass::ceil_div(visible_keys, TileKV) : 0;
      if (causal_k_tiles < k_tiles) {
        k_tiles = causal_k_tiles;
      }
    }
    return k_tiles;
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
  MUTLASS_DEVICE void load_k(Params const&   params,
                             Pipeline&       pipeline,
                             SharedStorage&  shared_storage,
                             WorkTile const& work_tile) const {
    int page_idx = safe_page_idx(params, work_tile);
    pipeline.k.producer_acquire(pipeline.k_write);
    uint32_t bar_id  = pipeline.k.producer_get_barrier_id(pipeline.k_write);
    auto     cta_tme = params.load_k.tme_load.get_slice(0);
    Tensor   gK      = make_tme_k_gmem(params.load_k, work_tile, page_idx);
    Tensor   sK      = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    copy(params.load_k.tme_load.with(bar_id),
         cta_tme.partition_S(gK),
         cta_tme.partition_D(sK(_, _, pipeline.k_write.index())));
    ++pipeline.k_write;
  }

  template <class WorkTile>
  MUTLASS_DEVICE void prefetch_k(Params const& params, WorkTile const& work_tile) const {
    int    page_idx = safe_page_idx(params, work_tile);
    Tensor gK       = make_tme_k_gmem(params.load_k, work_tile, page_idx);
    auto   cta_tme  = params.load_k.tme_load.get_slice(0);
    mute::prefetch(params.load_k.tme_load, cta_tme.partition_S(gK));
  }

  MUTLASS_DEVICE void wait_q(Pipeline& pipeline, int consumer_thread_idx) const {
    int lane_idx = consumer_thread_idx % NumThreadsPerWarp;
    if (lane_idx == 0) {
      pipeline.q.consumer_wait(pipeline.q_read);
    }
    __syncwarp();
  }

  MUTLASS_DEVICE void release_q(Pipeline& pipeline, int consumer_thread_idx) const {
    int lane_idx = consumer_thread_idx % NumThreadsPerWarp;
    __syncwarp();
    if (lane_idx == 0) {
      pipeline.q.consumer_release(pipeline.q_read);
    }
    ++pipeline.q_read;
  }

  MUTLASS_DEVICE void wait_k(Pipeline& pipeline, int consumer_thread_idx) const {
    int lane_idx = consumer_thread_idx % NumThreadsPerWarp;
    if (lane_idx == 0) {
      pipeline.k.consumer_wait(pipeline.k_read);
    }
    __syncwarp();
  }

  MUTLASS_DEVICE void release_k(Pipeline& pipeline, int consumer_thread_idx) const {
    int lane_idx = consumer_thread_idx % NumThreadsPerWarp;
    __syncwarp();
    if (lane_idx == 0) {
      pipeline.k.consumer_release(pipeline.k_read);
    }
    ++pipeline.k_read;
  }

  template <bool ApplyMask, class AccQK, class WorkTile>
  MUTLASS_DEVICE void store_rowmax_impl(Params const&   params,
                                        AccQK&          acc_qk,
                                        WorkTile const& work_tile,
                                        int             consumer_thread_idx) const {
    TiledMmaQK tiled_mma_qk;
    auto       thr_mma_qk = tiled_mma_qk.get_thread_slice(consumer_thread_idx);

    auto          reduction_target_qk = mate::attention::fmha::reduction_target_n(tiled_mma_qk);
    constexpr int red_rank            = decltype(rank(reduction_target_qk))::value;
    constexpr int reduction_size      = size(mate::attention::fmha::reduction_target_n(TiledMmaQK{}));

    Tensor acc_qk_mn = make_tensor(acc_qk.data(), mate::attention::fmha::layout_acc_mn(tiled_mma_qk, acc_qk.layout()));
    Tensor cQK       = make_identity_tensor(make_shape(Int<TileQ>{}, Int<TileKV>{}));
    Tensor tCcQK     = thr_mma_qk.partition_C(cQK);
    Tensor tCcQK_mn  = make_tensor(tCcQK.data(), mate::attention::fmha::layout_acc_mn(tiled_mma_qk, tCcQK.layout()));

    static_assert(size<1>(acc_qk_mn) % 4 == 0, "N must be a multiple of 4");

    int  q_token_begin  = work_tile.q_local_begin;
    int  k_begin        = work_tile.k_tile_begin;
    int  causal_offset  = IsCausal ? params.args.ptr_qo_offset[work_tile.batch_idx] : 0;
    int  lane_idx       = consumer_thread_idx % NumThreadsPerWarp;
    bool reduction_head = (lane_idx % reduction_size) == 0;

    ShapeMaxScore  shape_max_score = make_shape(params.args.total_q, params.args.num_qo_heads, params.args.max_k_tiles);
    StrideMaxScore stride_max_score = make_stride(
        int64_t(params.args.num_qo_heads) * params.args.max_k_tiles, int64_t(params.args.max_k_tiles), _1{});
    Tensor mMaxScore = make_tensor(make_gmem_ptr(params.args.ptr_max_score), shape_max_score, stride_max_score);
    Tensor gMaxScore = mMaxScore(_, _, work_tile.k_tile_idx);

    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(acc_qk_mn); ++m) {
      int    row       = int(get<0>(tCcQK_mn(m, 0)));
      int    token_row = row / HeadRatio;
      bool   row_valid = token_row < work_tile.q_count;
      float4 row_max_cur;

      MUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < size<1>(acc_qk_mn); n += 4) {
        auto masked_acc = [&](int ni) {
          if constexpr (!ApplyMask) {
            return float(acc_qk_mn(m, n + ni));
          } else {
            int  col     = int(get<1>(tCcQK_mn(m, n + ni)));
            int  k_local = k_begin + col;
            bool valid   = row_valid && k_local < work_tile.kv_len;
            if constexpr (IsCausal) {
              int q_local = q_token_begin + token_row;
              valid       = valid && k_local <= q_local + causal_offset;
            }
            return valid ? float(acc_qk_mn(m, n + ni)) : -std::numeric_limits<float>::infinity();
          }
        };

        float4 vec_acc = make_float4(masked_acc(0), masked_acc(1), masked_acc(2), masked_acc(3));
        if (n == 0) {
          row_max_cur = vec_acc;
        } else {
          mute::max(row_max_cur, vec_acc, row_max_cur);
        }
      }

      float2 row_max_pair;
      mute::max(row_max_pair, make_float2(row_max_cur.x, row_max_cur.y), make_float2(row_max_cur.z, row_max_cur.w));
      float row_max = max(row_max_pair.x, row_max_pair.y);

      for_each(make_seq<red_rank>{}, [&](auto r) {
        MUTLASS_PRAGMA_UNROLL
        for (int j = 1; j < shape<r>(reduction_target_qk); j *= 2) {
          row_max = max(row_max, __shfl_xor_sync(uint32_t(-1), row_max, stride<r>(reduction_target_qk) * j));
        }
      });

      if (row_valid && reduction_head) {
        int token_row_local      = row / HeadRatio;
        int head_local           = row - token_row_local * HeadRatio;
        int q_abs                = work_tile.q_abs_begin + token_row_local;
        int head_q               = work_tile.head_kv * HeadRatio + head_local;
        gMaxScore(q_abs, head_q) = row_max;
      }
    }
  }

  template <class AccQK, class WorkTile>
  MUTLASS_DEVICE void store_rowmax(Params const&   params,
                                   AccQK&          acc_qk,
                                   WorkTile const& work_tile,
                                   int             consumer_thread_idx) const {
    int  q_token_begin    = work_tile.q_local_begin;
    int  k_begin          = work_tile.k_tile_begin;
    int  causal_offset    = IsCausal ? params.args.ptr_qo_offset[work_tile.batch_idx] : 0;
    bool tile_fully_valid = k_begin + TileKV <= work_tile.kv_len;
    if constexpr (IsCausal) {
      tile_fully_valid = tile_fully_valid && k_begin + TileKV - 1 <= q_token_begin + causal_offset;
    }
    if (tile_fully_valid) {
      store_rowmax_impl<false>(params, acc_qk, work_tile, consumer_thread_idx);
    } else {
      store_rowmax_impl<true>(params, acc_qk, work_tile, consumer_thread_idx);
    }
  }

  template <class WorkTile>
  MUTLASS_DEVICE void compute_k_tile(Params const&   params,
                                     Pipeline&       pipeline,
                                     SharedStorage&  shared_storage,
                                     WorkTile const& work_tile,
                                     int             consumer_thread_idx) const {
    wait_k(pipeline, consumer_thread_idx);

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{})(_, _, pipeline.q_read.index());
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{})(_, _, pipeline.k_read.index());

    TiledMmaQK tiled_mma_qk;
    auto       thr_mma_qk = tiled_mma_qk.get_thread_slice(consumer_thread_idx);
    Tensor     tSrQ       = thr_mma_qk.partition_fragment_A(sQ);
    Tensor     tSrK       = thr_mma_qk.partition_fragment_B(sK);
    Tensor     acc_qk     = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShape{}));
    clear(acc_qk);

    mute::gemm(tiled_mma_qk, tSrQ, tSrK, acc_qk);
    mate::warpsquad_commit_batch();
    mate::warpsquad_wait();
    release_k(pipeline, consumer_thread_idx);

    store_rowmax(params, acc_qk, work_tile, consumer_thread_idx);
  }
};

}  // namespace mate::attention::msa::collective
