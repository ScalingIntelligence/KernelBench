#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <cstdint>
#include <limits>
#include <mute/swizzle_layout.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/pipeline/pipeline.hpp>
#include <type_traits>

#include "mate/common/layout.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/flat/collective/flat_collective_inverse_nxn.hpp"
#include "mate/flat/collective/flat_collective_load_tme.hpp"
#include "mate/flat/collective/flat_collective_mma_tme_sqmma.hpp"
#include "mate/flat/flat_options.hpp"
#include "mate/flat/kda/named_barrier.hpp"
#include "mate/flat/simd_helper.hpp"

#ifndef INLINE_LAMBDA
#define INLINE_LAMBDA __attribute__((always_inline))
#endif

namespace mate::flat::kda {

using namespace mute;

static constexpr float kLog2E = 1.4426950408889634074f;

template <class Element_,
          class StateElement_,
          class TileShape_,
          class StrideQ_,
          class StrideK_,
          class StrideV_,
          class StrideG_,
          class StrideO_,
          class... Options_>
struct ChunkKdaCollectiveTmeWarpSpecialized {
  using Element      = Element_;
  using StateElement = StateElement_;
  using TileShape    = TileShape_;
  using StrideQ      = StrideQ_;
  using StrideK      = StrideK_;
  using StrideV      = StrideV_;
  using StrideG      = StrideG_;
  using StrideO      = StrideO_;
  static constexpr bool HasStateIn =
      mate::flat::find_option_t<mate::flat::Tag::HasStateIn, std::false_type, Options_...>::value;
  static constexpr bool HasStateOut =
      mate::flat::find_option_t<mate::flat::Tag::HasStateOut, std::true_type, Options_...>::value;
  static constexpr bool HasGateParams =
      mate::flat::find_option_t<mate::flat::Tag::HasGateParams, std::true_type, Options_...>::value;
  static constexpr bool NormalizeQK =
      mate::flat::find_option_t<mate::flat::Tag::NormalizeQK, std::true_type, Options_...>::value;

  static constexpr int kChunk   = decltype(get<0>(TileShape{}))::value;
  static constexpr int kHeadDim = decltype(get<2>(TileShape{}))::value;
  static_assert(decltype(get<0>(TileShape{}))::value == decltype(get<1>(TileShape{}))::value);

  static constexpr int TmeStages          = 1;
  static constexpr int NumProducerSquads  = 1;
  static constexpr int NumStateSquads     = 2;
  static constexpr int NumOutputSquads    = 1;
  static constexpr int NumProducerThreads = NumProducerSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumStateThreads    = NumStateSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumOutputThreads   = NumOutputSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumProducerWarps   = NumProducerSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumStateWarps      = NumStateSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumOutputWarps     = NumOutputSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumStageBarriers   = static_cast<int>(KdaNamedBarrier::NumNamedBarriers);
  static constexpr int SmemAlignmentBytes = 256;
  static constexpr int VecSize            = 4;

  using MmaStageCount = mutlass::gemm::collective::StageCount<2>;
  using TileShapeQK   = TileShape;
  using TileShapeKU   = Shape<Int<kHeadDim>, Int<kHeadDim>, Int<kChunk>>;
  using TileShapeQS   = Shape<Int<kChunk>, Int<kHeadDim>, Int<kHeadDim>>;
  using TileShapePU   = Shape<Int<kChunk>, Int<kHeadDim>, Int<kChunk>>;
  using CollectiveMmaQK =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideQ,
                                                            2,
                                                            Element,
                                                            StrideK,
                                                            2,
                                                            float,
                                                            TileShapeQK,
                                                            Shape<_1, _1, _1>,
                                                            MmaStageCount,
                                                            mutlass::gemm::KernelTmeWarpSpecialized>::CollectiveOp;
  using StateAtomLayout = Layout<Shape<_1, Int<2>, _1>>;
  using RowAtomLayout   = Layout<Shape<_1, _1, _1>>;

  using CollectiveMmaKU = typename collective::Mp31TmeSqmmaCollective<Element,
                                                                      StrideV,
                                                                      StrideV,
                                                                      TileShapeKU,
                                                                      Int<kHeadDim>,
                                                                      Int<kHeadDim / 2>,
                                                                      StateAtomLayout>::CollectiveOp;
  using CollectiveMmaQS = typename collective::
      Mp31TmeSqmmaCollective<Element, StrideQ, StrideV, TileShapeQS, Int<kChunk>, Int<kHeadDim / 2>, RowAtomLayout>::
          CollectiveOp;
  using CollectiveMmaPU = typename collective::
      Mp31TmeSqmmaCollective<Element, StrideQ, StrideV, TileShapePU, Int<kChunk>, Int<kHeadDim / 2>, RowAtomLayout>::
          CollectiveOp;

  using TiledMmaKU = typename CollectiveMmaKU::TiledMma;
  using TiledMmaQS = typename CollectiveMmaQS::TiledMma;
  using TiledMmaPU = typename CollectiveMmaPU::TiledMma;
  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
  static_assert(decltype(size(TiledMmaKU{}))::value == NumStateThreads);
  static_assert(decltype(size(TiledMmaQS{}))::value == NumOutputThreads);
  static_assert(decltype(size(TiledMmaPU{}))::value == NumOutputThreads);

  using SmemLayoutQ = decltype(mate::unstage_smem_layout(typename CollectiveMmaQS::SmemLayoutA{}, Int<TmeStages>{}));
  static_assert(
      std::is_same_v<SmemLayoutQ,
                     decltype(mate::unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutA{}, Int<TmeStages>{}))>);
  using SmemLayoutK = decltype(mate::unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutB{}, Int<TmeStages>{}));
  using SmemLayoutKRestored =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaKU::SmemLayoutA{}, Int<TmeStages>{}));
  using SmemLayoutU = typename CollectiveMmaKU::SmemLayoutB;
  using SmemLayoutV = decltype(mate::unstage_smem_layout(typename CollectiveMmaKU::SmemLayoutB{}, Int<TmeStages>{}));
  using SmemLayoutState = decltype(take<0, 2>(typename CollectiveMmaQS::SmemLayoutB{}));
  static_assert(std::is_same_v<SmemLayoutU, typename CollectiveMmaPU::SmemLayoutB>);
  using SmemLayoutKInverse   = SmemLayoutK;
  using SmemLayoutInverseA   = typename CollectiveMmaPU::SmemLayoutA;
  using SmemLayoutP          = SmemLayoutInverseA;
  using CollectiveInverseNxN = collective::CollectiveLowerTriangularInverseNxN<Element, kChunk>;
  using SmemLayoutInverse    = typename CollectiveInverseNxN::SmemLayout;

  static constexpr int R2SVectorBits    = 128;
  static constexpr int R2SMmaAtomN      = 8;
  static constexpr int R2SFragmentSize  = R2SVectorBits / sizeof_bits_v<Element>;
  static constexpr int R2SGranularity   = R2SMmaAtomN * R2SFragmentSize;
  static constexpr int SmemTileNForR2S  = decltype(size<0>(SmemLayoutU{}))::value;
  static constexpr int R2SPermuteRepeat = SmemTileNForR2S / R2SGranularity;
  static_assert(SmemTileNForR2S % R2SGranularity == 0);
  static constexpr int VPermuteMmaAtomN = 8;
  static constexpr int VPermuteRepeat   = kHeadDim / VPermuteMmaAtomN;
  static_assert(kHeadDim % VPermuteMmaAtomN == 0);
  static constexpr int GmemVPermuteBlocks = VPermuteRepeat / VPermuteMmaAtomN;
  static_assert(VPermuteRepeat % VPermuteMmaAtomN == 0);

  using R2SFragmentType   = mute::uint_bit_t<R2SVectorBits>;
  using UMmaPermuteTile   = decltype(filter(
      make_ordered_layout(Shape<Int<R2SMmaAtomN>, Int<R2SFragmentSize>, Int<R2SPermuteRepeat>>{}, Step<_2, _1, _3>{})));
  using PermuteTileForU   = Tile<Underscore, UMmaPermuteTile, Underscore>;
  using PermuteTiledMmaPU = decltype(mate::convert_to_permuted_sqmma(TiledMmaPU{}, PermuteTileForU{}));
  using PermuteVTile =
      decltype(filter(make_ordered_layout(Shape<Int<VPermuteMmaAtomN>, Int<VPermuteRepeat>>{}, Step<_2, _1>{})));
  // SQMMA fragments enumerate V in 8x8-transposed blocks. Compose this
  // internal-to-natural mapping into gmem views so vectorized copies/casts stay unchanged.
  using InternalToNaturalVLayout = decltype(filter(make_ordered_layout(
      Shape<Int<VPermuteMmaAtomN>, Int<VPermuteMmaAtomN>, Int<GmemVPermuteBlocks>>{}, Step<_2, _1, _3>{})));
  using R2SCopyAtom              = Copy_Atom<UniversalCopy<R2SFragmentType>, Element>;
  using R2STiledCopy             = decltype(make_tiled_copy_C(R2SCopyAtom{}, PermuteTiledMmaPU{}));
  using StateGmemCopyAtom        = Copy_Atom<UniversalCopy<StateElement>, StateElement>;
  using StateGmemTiledCopy       = decltype(make_tiled_copy_C(StateGmemCopyAtom{}, TiledMmaKU{}));

  using SmemLayoutG       = SmemLayoutQ;
  using SmemLayoutGCumsum = decltype(as_position_independent_swizzle_layout(take<0, 2>(SmemLayoutG{})));
  using SmemLayoutDtBias  = decltype(make_layout(make_shape(Int<kHeadDim>{}), make_stride(_1{})));
  using SmemLayoutBeta    = decltype(make_layout(make_shape(Int<kChunk>{}), make_stride(_1{})));

  using SmemLayoutGTotalStaged =
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<TmeStages>{}), make_stride(_1{}, Int<kHeadDim>{})));
  using QKTmeLoadShape = decltype(make_shape(int32_t{}, Int<kHeadDim>{}, make_shape(int32_t{}, int32_t{})));
  using QKTmeStride    = decltype(make_stride(int64_t{}, _1{}, make_stride(int64_t{}, int64_t{})));

  using TME_Q = typename CollectiveMmaQK::Params::TME_A;
  using TME_K = typename CollectiveMmaQK::Params::TME_B;
  using TME_G = TME_Q;
  using TME_V = typename CollectiveMmaKU::Params::TME_B;

  using PipelineQ = mutlass::Mp31PipelineTmeAsync<TmeStages>;
  using PipelineK = mutlass::Mp31PipelineTmeAsync<TmeStages>;
  using PipelineG = mutlass::Mp31PipelineTmeAsync<TmeStages>;
  using PipelineV = mutlass::Mp31PipelineTmeAsync<TmeStages>;
  using LoadQ     = collective::
      CollectiveLoadTme<collective::LoadKind::kQ, PipelineQ, Element, SmemLayoutQ, TME_Q, SmemAlignmentBytes>;
  using LoadK = collective::
      CollectiveLoadTme<collective::LoadKind::kK, PipelineK, Element, SmemLayoutK, TME_K, SmemAlignmentBytes>;
  using LoadV = collective::
      CollectiveLoadTme<collective::LoadKind::kV, PipelineV, Element, SmemLayoutV, TME_V, SmemAlignmentBytes>;
  using LoadG = collective::
      CollectiveLoadTme<collective::LoadKind::kG, PipelineG, Element, SmemLayoutG, TME_G, SmemAlignmentBytes>;
  using PipelineQParams = typename PipelineQ::Params;
  using PipelineKParams = typename PipelineK::Params;
  using PipelineGParams = typename PipelineG::Params;
  using PipelineVParams = typename PipelineV::Params;
  using PipelineQState  = typename PipelineQ::PipelineState;
  using PipelineKState  = typename PipelineK::PipelineState;
  using PipelineGState  = typename PipelineG::PipelineState;
  using PipelineVState  = typename PipelineV::PipelineState;

  template <class StorageElement, class SmemLayout>
  using SmemArray = array_aligned<StorageElement, cosize_v<SmemLayout>, SmemAlignmentBytes>;

  struct SharedStorage {
    typename LoadQ::SharedStorage smem_q_raw;
    union {
      SmemArray<Element, SmemLayoutQ> smem_q_normalized;
      SmemArray<Element, SmemLayoutQ> smem_q_decayed;
    };
    typename LoadK::SharedStorage smem_k_raw;
    union {
      SmemArray<Element, SmemLayoutK> smem_k_normalized;
      SmemArray<Element, SmemLayoutK> smem_k_decayed;
    };
    typename LoadV::SharedStorage                smem_v_raw;
    SmemArray<Element, SmemLayoutKInverse>       smem_k_inverse;
    typename LoadG::SharedStorage                smem_g_raw;
    SmemArray<float, SmemLayoutGCumsum>          smem_g_cumsum;
    SmemArray<Element, SmemLayoutKRestored>      smem_k_restored;
    SmemArray<Element, SmemLayoutU>              smem_u;
    SmemArray<Element, SmemLayoutInverseA>       smem_inverse;
    SmemArray<Element, SmemLayoutP>              smem_p;
    typename CollectiveInverseNxN::SharedStorage smem_inverse_workspace;
    SmemArray<float, SmemLayoutBeta>             smem_beta;
    SmemArray<float, SmemLayoutGTotalStaged>     smem_g_total_unshifted;
    SmemArray<float, SmemLayoutGTotalStaged>     smem_g_total_shifted;
    SmemArray<float, SmemLayoutGTotalStaged>     smem_g_shifted;
    SmemArray<float, SmemLayoutDtBias>           smem_dt_bias;
    SmemArray<Element, SmemLayoutState>          smem_state;
  };

  static constexpr int     SharedStorageSize  = sizeof(SharedStorage);
  static constexpr int64_t KTileBytes         = int64_t(kChunk) * kHeadDim * sizeof(Element);
  static constexpr int64_t KRestoredTileBytes = KTileBytes;
  static constexpr int64_t ChunkMatrixBytes   = int64_t(kChunk) * kChunk * sizeof(Element);

  struct Arguments {
    Element const* ptr_q;
    Element const* ptr_k;
    Element const* ptr_v;
    Element const* ptr_g;
    Element const* ptr_beta;
    void const*    ptr_initial_state;
    Element*       ptr_out;
    void*          ptr_final_state;
    float const*   ptr_A_log;
    float const*   ptr_dt_bias;
    StrideQ        stride_q;
    StrideK        stride_k;
    StrideV        stride_v;
    StrideG        stride_g;
    StrideO        stride_out;
    float          scale;
    float          lower_bound;
  };

  struct Params {
    Element const* ptr_q;
    Element const* ptr_k;
    Element const* ptr_v;
    Element const* ptr_g;
    Element const* ptr_beta;
    void const*    ptr_initial_state;
    Element*       ptr_out;
    void*          ptr_final_state;
    float const*   ptr_A_log;
    float const*   ptr_dt_bias;
    StrideQ        stride_q;
    StrideK        stride_k;
    StrideV        stride_v;
    StrideG        stride_g;
    StrideO        stride_out;
    float          scale;
    float          lower_bound;
    TME_Q          tme_q;
    TME_K          tme_k;
    TME_G          tme_g;
    TME_V          tme_v;
  };

  struct alignas(1) BarrierStorage {
    uint8_t stage[NumStageBarriers];
    uint8_t pipeline_q[PipelineQ::NumBarriers];
    uint8_t pipeline_k[PipelineK::NumBarriers];
    uint8_t pipeline_g[PipelineG::NumBarriers];
    uint8_t pipeline_v[PipelineV::NumBarriers];
  };

  struct Pipeline {
    PipelineQ      q;
    PipelineK      k;
    PipelineG      g;
    PipelineV      v;
    PipelineQState q_read;
    PipelineQState q_write;
    PipelineKState k_read;
    PipelineKState k_write;
    PipelineGState g_read;
    PipelineGState g_write;
    PipelineVState v_read;
    PipelineVState v_write;

    static MUTLASS_DEVICE PipelineQParams make_q_params() {
      PipelineQParams params;
      params.transaction_bytes = uint32_t(KTileBytes);
      params.num_consumers     = NumProducerWarps;
      params.num_producers     = 1;
      return params;
    }

    static MUTLASS_DEVICE PipelineKParams make_k_params() {
      PipelineKParams params = make_q_params();
      params.num_consumers   = NumProducerWarps;
      return params;
    }

    static MUTLASS_DEVICE PipelineGParams make_g_params() {
      PipelineGParams params = make_q_params();
      params.num_consumers   = NumProducerWarps;
      return params;
    }

    static MUTLASS_DEVICE PipelineVParams make_v_params() {
      PipelineVParams params = make_q_params();
      params.num_consumers   = NumOutputWarps;
      return params;
    }

    MUTLASS_DEVICE explicit Pipeline(BarrierStorage* barrier_storage)
        : q(make_q_params(), reinterpret_cast<uint64_t>(&barrier_storage->pipeline_q), 1),
          k(make_k_params(), reinterpret_cast<uint64_t>(&barrier_storage->pipeline_k), 1),
          g(make_g_params(), reinterpret_cast<uint64_t>(&barrier_storage->pipeline_g), 1),
          v(make_v_params(), reinterpret_cast<uint64_t>(&barrier_storage->pipeline_v), 1),
          q_read{},
          q_write(mutlass::make_producer_start_state<PipelineQ>()),
          k_read{},
          k_write(mutlass::make_producer_start_state<PipelineK>()),
          g_read{},
          g_write(mutlass::make_producer_start_state<PipelineG>()),
          v_read{},
          v_write(mutlass::make_producer_start_state<PipelineV>()) {
    }
  };

  static MUTLASS_DEVICE void init_stage_barriers(BarrierStorage* barrier_storage, int tid) {
    (void)barrier_storage;
    static constexpr uint32_t StageBarrierArriveCount[NumStageBarriers] = {
        NumProducerWarps,                // GCumsumReady
        mutlass::NumWarpsPerWarpSquad,   // InverseReady
        NumOutputWarps,                  // KRestoredReady
        NumOutputWarps,                  // VUpdatedReady
        NumStateWarps,                   // StateCommitted
        NumProducerWarps,                // OperandsReady
        NumStateWarps + NumOutputWarps,  // OperandsConsumed
        NumOutputWarps,                  // PReady
        NumProducerWarps,                // DtBiasLoaded
        mutlass::NumWarpsPerWarpSquad,   // BetaLoaded
        mutlass::NumWarpsPerWarpSquad,   // InverseReadySmemReady
        NumStateWarps + NumOutputWarps,  // VUpdatedConsumed
        NumOutputWarps,                  // StateConsumed
    };
    if (tid == 0) {
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < NumStageBarriers; ++i) {
        named_barrier_init(i, StageBarrierArriveCount[i]);
      }
    }
  }

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    auto params_qk = CollectiveMmaQK::to_underlying_arguments(
        make_shape(problem_size.T, problem_size.T, Int<kHeadDim>{}, make_shape(problem_size.Hqk, problem_size.B)),
        typename CollectiveMmaQK::Arguments{
            args.ptr_q,
            args.stride_q,
            args.ptr_k,
            args.stride_k,
        },
        nullptr);
    auto params_v = CollectiveMmaKU::to_underlying_arguments(
        make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}, problem_size.T, make_shape(problem_size.H, problem_size.B)),
        typename CollectiveMmaKU::Arguments{
            args.ptr_v,
            args.stride_v,
            args.ptr_v,
            args.stride_v,
        },
        nullptr);
    auto params_g = CollectiveMmaQK::to_underlying_arguments(
        make_shape(problem_size.T, problem_size.T, Int<kHeadDim>{}, make_shape(problem_size.H, problem_size.B)),
        typename CollectiveMmaQK::Arguments{
            args.ptr_g,
            args.stride_g,
            args.ptr_g,
            args.stride_g,
        },
        nullptr);

    return Params{
        args.ptr_q,           args.ptr_k,           args.ptr_v,
        args.ptr_g,           args.ptr_beta,        args.ptr_initial_state,
        args.ptr_out,         args.ptr_final_state, args.ptr_A_log,
        args.ptr_dt_bias,     args.stride_q,        args.stride_k,
        args.stride_v,        args.stride_g,        args.stride_out,
        args.scale,           args.lower_bound,     params_qk.tme_load_a,
        params_qk.tme_load_b, params_g.tme_load_a,  params_v.tme_load_b,
    };
  }

  static MUTLASS_DEVICE float sigmoid_fast(float x) {
    return 1.0f / (1.0f + ::mate::flat::detail::fast_exp2(-x * kLog2E));
  }

  static MUTLASS_DEVICE Element inactive_gate_raw_element() {
    if constexpr (HasGateParams) {
      return -std::numeric_limits<Element>::infinity();
    } else {
      return Element(0.0f);
    }
  }

  template <class TensorQK>
  static MUTLASS_DEVICE float4 load_qk_vec4(TensorQK&& sQk, int row, int col_base) {
    return simd::load_packed4_as_float4(&sQk(make_coord(row, col_base)));
  }

  template <class TensorQK>
  static MUTLASS_DEVICE void store_qk_vec4(TensorQK&& sQk, int row, int col_base, float4 value) {
    simd::store_float4_to_packed4_rn(&sQk(make_coord(row, col_base)), value);
  }

  static MUTLASS_DEVICE float load_a_log_exp(Params const& params, int head_idx) {
    if constexpr (HasGateParams) {
      return ::mate::flat::detail::fast_exp2(params.ptr_A_log[head_idx] * kLog2E);
    } else {
      return 1.0f;
    }
  }

  static MUTLASS_DEVICE float make_gate(float raw_g, float a_log_exp, float dt_bias, Params const& params) {
    if constexpr (HasGateParams) {
      float scale   = -a_log_exp * kLog2E;
      float exp_arg = (raw_g + dt_bias) * scale;
      float exp_val = ::mate::flat::detail::fast_exp2(exp_arg);
      float coeff   = params.lower_bound * kLog2E;
      return coeff / (exp_val + 1.0f);
    } else {
      return raw_g;
    }
  }

  template <int SubWarpSize>
  static MUTLASS_DEVICE float subwarp_reduce_sum(float value) {
    MUTE_UNROLL
    for (int offset = SubWarpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(uint32_t(-1), value, offset, SubWarpSize);
    }
    return value;
  }

  template <class TensorDtBias>
  static MUTLASS_DEVICE void load_dt_bias(TensorDtBias& sDtBias, Params const& params, int head_idx, int local_tid) {
    if constexpr (HasGateParams) {
      for (int col = local_tid; col < kHeadDim; col += NumProducerThreads) {
        sDtBias(col) = params.ptr_dt_bias[int64_t(head_idx) * kHeadDim + col];
      }
    }
  }

  template <class ProblemSize, class TensorBeta, class WorkDesc>
  static MUTLASS_DEVICE void load_beta(TensorBeta&        sBeta,
                                       Params const&      params,
                                       ProblemSize const& problem_size,
                                       WorkDesc const&    work_desc,
                                       int                chunk_idx,
                                       int                local_tid,
                                       bool               is_not_full_chunk) {
    if (local_tid < kChunk) {
      bool    token_valid = !is_not_full_chunk || local_tid < work_desc.actual_len(chunk_idx);
      int64_t offset =
          (int64_t(work_desc.tme_batch()) * problem_size.T + int64_t(work_desc.tme_token(chunk_idx) + local_tid)) *
              problem_size.H +
          int64_t(work_desc.head_idx);
      float raw_beta =
          token_valid ? static_cast<float>(params.ptr_beta[offset]) : -std::numeric_limits<float>::infinity();
      sBeta(local_tid) = sigmoid_fast(raw_beta);
    }
  }

  template <class TensorK,
            class TensorQ,
            class TensorG,
            class TensorKDecayed,
            class TensorKInverse,
            class TensorQDecayed>
  static MUTLASS_DEVICE void scale_normalized_qk(TensorK&&        sK,
                                                 TensorQ&&        sQ,
                                                 TensorG&&        sGCumsum,
                                                 TensorKDecayed&& sKDecayed,
                                                 TensorKInverse&& sKInverse,
                                                 TensorQDecayed&& sQDecayed,
                                                 float            q_scale,
                                                 int              local_tid,
                                                 int              actual_len,
                                                 bool             is_not_full_chunk) {
    constexpr int ThreadsPerRow     = 8;
    constexpr int RowsPerWarp       = mutlass::NumThreadsPerWarp / ThreadsPerRow;
    constexpr int ElementsPerThread = kHeadDim / ThreadsPerRow;
    constexpr int VectorsPerThread  = ElementsPerThread / VecSize;
    constexpr int RowsPerWave       = mutlass::NumWarpsPerWarpSquad * RowsPerWarp;
    static_assert(kChunk % RowsPerWave == 0);
    int warp_idx      = local_tid / mutlass::NumThreadsPerWarp;
    int lane_idx      = local_tid % mutlass::NumThreadsPerWarp;
    int row_in_warp   = lane_idx / ThreadsPerRow;
    int lane_in_row   = lane_idx % ThreadsPerRow;
    int thread_col_lo = lane_in_row * ElementsPerThread;

    MUTE_UNROLL
    for (int row_base = 0; row_base < kChunk; row_base += RowsPerWave) {
      int    row          = row_base + warp_idx * RowsPerWarp + row_in_warp;
      bool   row_valid    = !is_not_full_chunk || row < actual_len;
      float  thread_sum_q = 0.0f;
      float  thread_sum_k = 0.0f;
      float4 q_decayed_reg[VectorsPerThread];
      float4 k_decayed_reg[VectorsPerThread];
      float4 k_inverse_reg[VectorsPerThread];

      MUTE_UNROLL
      for (int vec_iter = 0; vec_iter < VectorsPerThread; ++vec_iter) {
        int vec_slot = (vec_iter + row_in_warp) % VectorsPerThread;
        int col_base = thread_col_lo + vec_slot * VecSize;
        if (row_valid) {
          float4 gate_arg = simd::load_float4(&sGCumsum(make_coord(row, col_base)));
          float4 gate_vec = simd::fast_exp2(gate_arg);
          float4 inv_gate = simd::fast_exp2(simd::vmul(gate_arg, -1.0f));
          float4 k_vec    = load_qk_vec4(sK, row, col_base);
          float4 q_vec    = load_qk_vec4(sQ, row, col_base);
          if constexpr (NormalizeQK) {
            simd::dot4(k_vec, k_vec, thread_sum_k);
            simd::dot4(q_vec, q_vec, thread_sum_q);
          }
          k_decayed_reg[vec_iter] = simd::vmul(k_vec, gate_vec);
          k_inverse_reg[vec_iter] = simd::vmul(k_vec, inv_gate);
          q_decayed_reg[vec_iter] = simd::vmul(q_vec, gate_vec);
        } else {
          k_decayed_reg[vec_iter] = simd::splat4(0.0f);
          k_inverse_reg[vec_iter] = simd::splat4(0.0f);
          q_decayed_reg[vec_iter] = simd::splat4(0.0f);
        }
      }

      float inv_norm_q = 1.0f;
      float inv_norm_k = 1.0f;
      if constexpr (NormalizeQK) {
        inv_norm_k = rsqrtf(subwarp_reduce_sum<ThreadsPerRow>(thread_sum_k) + 1.0e-6f);
        inv_norm_q = rsqrtf(subwarp_reduce_sum<ThreadsPerRow>(thread_sum_q) + 1.0e-6f);
      }

      MUTE_UNROLL
      for (int vec_iter = 0; vec_iter < VectorsPerThread; ++vec_iter) {
        int vec_slot = (vec_iter + row_in_warp) % VectorsPerThread;
        int col_base = thread_col_lo + vec_slot * VecSize;
        if constexpr (NormalizeQK) {
          k_decayed_reg[vec_iter] = simd::vmul(k_decayed_reg[vec_iter], inv_norm_k);
          k_inverse_reg[vec_iter] = simd::vmul(k_inverse_reg[vec_iter], inv_norm_k);
        }
        q_decayed_reg[vec_iter] = simd::vmul(q_decayed_reg[vec_iter], inv_norm_q * q_scale);
        store_qk_vec4(sKDecayed, row, col_base, k_decayed_reg[vec_iter]);
        store_qk_vec4(sKInverse, row, col_base, k_inverse_reg[vec_iter]);
        store_qk_vec4(sQDecayed, row, col_base, q_decayed_reg[vec_iter]);
      }
    }
  }

  template <class TensorStateSmem, class TensorState>
  static MUTLASS_DEVICE void commit_state(
      TensorStateSmem& sState, TensorState const& rState, SharedStorage& shared_storage, int stage, int local_tid) {
    auto sGShifted =
        make_tensor(make_smem_ptr(shared_storage.smem_g_shifted.data()), SmemLayoutGTotalStaged{})(_, stage);
    TiledMmaKU tiled_mma_ku;
    auto       thr_mma_ku = tiled_mma_ku.get_thread_slice(local_tid);
    auto       cState     = make_identity_tensor(make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}));
    auto       tCcState   = thr_mma_ku.partition_C(cState);
    using StateVec        = mutlass::Array<float, VecSize>;
    using ElementVec      = mutlass::Array<Element, VecSize>;
    auto rStateVec        = recast<StateVec const>(rState);
    auto rStateCvt        = make_fragment_like<Element>(rState);
    auto rStateCvtVec     = recast<ElementVec>(rStateCvt);
    mutlass::NumericArrayConverter<Element, float, VecSize, mutlass::FloatRoundStyle::round_to_nearest> convert_state;

    MUTLASS_PRAGMA_UNROLL
    for (int vec_idx = 0; vec_idx < size(rStateVec); ++vec_idx) {
      int    base           = vec_idx * VecSize;
      int    col_k          = int(get<0>(tCcState(base)));
      float  scale          = sGShifted(col_k);
      float4 values         = simd::vmul(reinterpret_cast<float4 const&>(rStateVec(vec_idx)), scale);
      rStateCvtVec(vec_idx) = convert_state(reinterpret_cast<StateVec const&>(values));
    }

    // KU accumulator coordinates already use the V basis consumed by PU. Keep
    // those coordinates when scattering into the swizzled state tile.
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(rStateCvt); ++i) {
      int col_k                        = int(get<0>(tCcState(i)));
      int col_v                        = int(get<1>(tCcState(i)));
      sState(make_coord(col_v, col_k)) = rStateCvt(i);
    }
    named_barrier_arrive(KdaNamedBarrier::StateCommitted);
  }

  template <class ProblemSize, class WorkDesc, class TensorState>
  static MUTLASS_DEVICE void kv_load(Params const&      params,
                                     ProblemSize const& problem_size,
                                     WorkDesc const&    work_desc,
                                     TensorState&       rState,
                                     int                local_tid) {
    if constexpr (HasStateIn) {
      StateGmemTiledCopy tiled_copy_state;
      auto               thr_copy_state = tiled_copy_state.get_thread_slice(local_tid);
      auto               mState         = make_tensor(
          make_gmem_ptr(static_cast<StateElement const*>(params.ptr_initial_state)),
          make_shape(problem_size.N, problem_size.H, Int<kHeadDim>{}, Int<kHeadDim>{}),
          make_stride(
              int64_t(problem_size.H * kHeadDim * kHeadDim), int64_t(kHeadDim * kHeadDim), _1{}, int64_t(kHeadDim)));
      auto permuted_state =
          make_tensor(mState.data(), composition(mState.layout(), make_tile(_, _, _, InternalToNaturalVLayout{})));
      auto   gState    = permuted_state(work_desc.seq_idx, work_desc.head_idx, _, _);
      Tensor tSgState  = thr_copy_state.partition_S(gState);
      Tensor rStateCvt = make_fragment_like<StateElement>(rState);
      Tensor tSrState  = thr_copy_state.retile_D(rStateCvt);

      copy(tiled_copy_state, tSgState, tSrState);

      using StateVec    = mutlass::Array<StateElement, VecSize>;
      using FloatVec    = mutlass::Array<float, VecSize>;
      auto rStateCvtVec = recast<StateVec>(rStateCvt);
      auto rStateVec    = recast<FloatVec>(rState);

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rStateVec); ++i) {
        rStateVec(i) = mutlass::NumericArrayConverter<float, StateElement, VecSize>{}(rStateCvtVec(i));
      }
    } else {
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rState); ++i) {
        rState(i) = 0.0f;
      }
    }
  }

  template <class ProblemSize, class WorkDesc, class TensorState>
  static MUTLASS_DEVICE void kv_store(Params const&      params,
                                      ProblemSize const& problem_size,
                                      WorkDesc const&    work_desc,
                                      TensorState const& rState,
                                      int                local_tid) {
    if constexpr (HasStateOut) {
      StateGmemTiledCopy tiled_copy_state;
      auto               thr_copy_state = tiled_copy_state.get_thread_slice(local_tid);
      auto               mState         = make_tensor(
          make_gmem_ptr(static_cast<StateElement*>(params.ptr_final_state)),
          make_shape(problem_size.N, problem_size.H, Int<kHeadDim>{}, Int<kHeadDim>{}),
          make_stride(
              int64_t(problem_size.H * kHeadDim * kHeadDim), int64_t(kHeadDim * kHeadDim), _1{}, int64_t(kHeadDim)));
      auto permuted_state =
          make_tensor(mState.data(), composition(mState.layout(), make_tile(_, _, _, InternalToNaturalVLayout{})));
      auto   gState     = permuted_state(work_desc.seq_idx, work_desc.head_idx, _, _);
      Tensor rStateCvt  = make_fragment_like<StateElement>(rState);
      Tensor tSrState   = thr_copy_state.retile_S(rStateCvt);
      Tensor tDgState   = thr_copy_state.partition_D(gState);
      using FloatVec    = mutlass::Array<float, VecSize>;
      using StateVec    = mutlass::Array<StateElement, VecSize>;
      auto rStateVec    = recast<FloatVec const>(rState);
      auto rStateCvtVec = recast<StateVec>(rStateCvt);

      mutlass::NumericArrayConverter<StateElement, float, VecSize, mutlass::FloatRoundStyle::round_to_nearest>
          convert_state;
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rStateCvtVec); ++i) {
        rStateCvtVec(i) = convert_state(rStateVec(i));
      }
      copy(tiled_copy_state, tSrState, tDgState);
    }
  }

  template <class TensorState>
  static MUTLASS_DEVICE void scale_state(TensorState& rState, SharedStorage& shared_storage, int stage, int local_tid) {
    auto sGTotal =
        make_tensor(make_smem_ptr(shared_storage.smem_g_total_unshifted.data()), SmemLayoutGTotalStaged{})(_, stage);

    TiledMmaKU tiled_mma_ku;
    auto       thr_mma_ku = tiled_mma_ku.get_thread_slice(local_tid);
    auto       cState     = make_identity_tensor(make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}));
    auto       tCcState   = thr_mma_ku.partition_C(cState);
    using StateVec        = mutlass::Array<float, VecSize>;
    auto rStateVec        = recast<StateVec>(rState);

    MUTLASS_PRAGMA_UNROLL
    for (int vec_idx = 0; vec_idx < size(rStateVec); ++vec_idx) {
      int   col_k  = int(get<0>(tCcState(vec_idx * VecSize)));
      float scale  = sGTotal(col_k);
      auto& values = reinterpret_cast<float4&>(rStateVec(vec_idx));
      values       = simd::vmul(values, scale);
    }
  }

  static MUTLASS_DEVICE void update_v(SharedStorage& shared_storage, int stage, int local_tid, uint32_t phase) {
    auto sInverse = make_tensor(make_smem_ptr(shared_storage.smem_inverse.data()), SmemLayoutInverseA{})(_, _, stage);
    auto sU       = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);
    auto sUMn     = make_tensor(sU.data(), mate::select_layout<1, 0>(sU.layout()));

    named_barrier_wait(KdaNamedBarrier::InverseReady, phase);

    TiledMmaPU tiled_mma_pu;
    auto       thr_mma_pu = tiled_mma_pu.get_thread_slice(local_tid);
    auto       acc_u      = partition_fragment_C(tiled_mma_pu, make_shape(Int<kChunk>{}, Int<kHeadDim>{}));
    clear(acc_u);
    auto tSsInverse = thr_mma_pu.partition_A(sInverse);
    auto tSsU       = thr_mma_pu.partition_B(sU);
    auto tSrInverse = thr_mma_pu.make_fragment_A(tSsInverse);
    auto tSrU       = thr_mma_pu.make_fragment_B(tSsU);
    gemm(tiled_mma_pu, acc_u, tSrInverse, tSrU, acc_u);
    mate::warpsquad_commit_batch();
    mate::warpsquad_wait();

    R2STiledCopy tiled_copy_r2s;
    auto         thr_copy_r2s = tiled_copy_r2s.get_thread_slice(local_tid);
    Tensor       tUsU         = thr_copy_r2s.partition_D(as_position_independent_swizzle_tensor(sUMn));
    Tensor       accum_cvt    = make_fragment_like<Element>(acc_u);
    Tensor       tUrU         = thr_copy_r2s.retile_S(accum_cvt);
    Tensor       tCvt_frg     = recast<mutlass::Array<Element, R2SFragmentSize>>(accum_cvt);
    Tensor       tAcc_frg     = recast<mutlass::Array<float, R2SFragmentSize>>(acc_u);

    MUTE_UNROLL
    for (int i = 0; i < size(tCvt_frg); ++i) {
      tCvt_frg(i) =
          mutlass::NumericArrayConverter<Element, float, R2SFragmentSize, mutlass::FloatRoundStyle::round_to_nearest>{}(
              tAcc_frg(i));
    }
    copy(tiled_copy_r2s, tUrU, tUsU);
    named_barrier_arrive(KdaNamedBarrier::VUpdatedReady);
  }

  template <class TensorState>
  static MUTLASS_DEVICE void issue_state_update(
      TensorState& rState, SharedStorage& shared_storage, int stage, int local_tid, uint32_t phase) {
    auto sKRestored =
        make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestored{})(_, _, stage);
    auto sU = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);

    TiledMmaKU tiled_mma_ku;
    auto       thr_mma_ku = tiled_mma_ku.get_thread_slice(local_tid);

    auto tSsKRestored = thr_mma_ku.partition_A(sKRestored);
    auto tSsU         = thr_mma_ku.partition_B(sU);
    auto tSrKRestored = thr_mma_ku.make_fragment_A(tSsKRestored);
    auto tSrU         = thr_mma_ku.make_fragment_B(tSsU);

    named_barrier_wait(KdaNamedBarrier::KRestoredReady, phase);
    named_barrier_wait(KdaNamedBarrier::VUpdatedReady, phase);
    gemm(tiled_mma_ku, rState, tSrKRestored, tSrU, rState);
    mate::warpsquad_commit_batch();
  }

  template <class ProblemSize, class WorkDesc, class IsFinalChunk>
  static MUTLASS_DEVICE void prepare_state_operands(SharedStorage&     shared_storage,
                                                    Params const&      params,
                                                    ProblemSize const& problem_size,
                                                    WorkDesc const&    work_desc,
                                                    int                chunk_idx,
                                                    int                local_tid,
                                                    IsFinalChunk       is_final_chunk) {
    auto sKDecayed = make_tensor(make_smem_ptr(shared_storage.smem_k_decayed.data()), SmemLayoutK{})(_, _, 0);
    auto sKInverse = make_tensor(make_smem_ptr(shared_storage.smem_k_inverse.data()), SmemLayoutKInverse{})(_, _, 0);
    auto sBeta     = make_tensor(make_smem_ptr(shared_storage.smem_beta.data()), SmemLayoutBeta{});

    if (local_tid < mutlass::NumThreadsPerWarpSquad) {
      TiledMmaQK tiled_mma_inverse;
      auto       thr_mma_inverse = tiled_mma_inverse.get_thread_slice(local_tid);
      auto       acc_inverse     = partition_fragment_C(tiled_mma_inverse, make_shape(Int<kChunk>{}, Int<kChunk>{}));
      clear(acc_inverse);
      auto tSsKDecayed = thr_mma_inverse.partition_A(sKDecayed);
      auto tSsKInverse = thr_mma_inverse.partition_B(sKInverse);
      auto tSrKDecayed = thr_mma_inverse.make_fragment_A(tSsKDecayed);
      auto tSrKInverse = thr_mma_inverse.make_fragment_B(tSsKInverse);
      gemm(tiled_mma_inverse, acc_inverse, tSrKDecayed, tSrKInverse, acc_inverse);
      mate::warpsquad_commit_batch();
      mate::warpsquad_wait();

      // we put load_beta after warpsquad_wait to avoid the weird compiler MTGPU-DEPENDENCY-GRAPH warning
      bool is_not_full_chunk = IsFinalChunk::value && work_desc.actual_len(chunk_idx) < kChunk;
      load_beta(sBeta, params, problem_size, work_desc, chunk_idx, local_tid, is_not_full_chunk);
      named_barrier_arrive_and_wait(KdaNamedBarrier::BetaLoaded);
      CollectiveInverseNxN inverse_nxn;
      auto sInverseOut = make_tensor(make_smem_ptr(shared_storage.smem_inverse.data()), SmemLayoutInverseA{})(_, _, 0);
      inverse_nxn(
          shared_storage.smem_inverse_workspace,
          acc_inverse,
          thr_mma_inverse,
          sInverseOut,
          sBeta,
          []() { named_barrier_arrive_and_wait(KdaNamedBarrier::InverseSmemReady); },
          local_tid);
      named_barrier_arrive(KdaNamedBarrier::InverseReady);
    } else {
      int  local_p_tid = local_tid - mutlass::NumThreadsPerWarpSquad;
      auto sQDecayed   = make_tensor(make_smem_ptr(shared_storage.smem_q_decayed.data()), SmemLayoutQ{})(_, _, 0);
      auto sP          = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{})(_, _, 0);

      TiledMmaQK tiled_mma_qk_p;
      auto       thr_mma_qk_p = tiled_mma_qk_p.get_thread_slice(local_p_tid);
      auto       acc_p        = partition_fragment_C(tiled_mma_qk_p, make_shape(Int<kChunk>{}, Int<kChunk>{}));
      clear(acc_p);
      auto cP          = make_identity_tensor(make_shape(Int<kChunk>{}, Int<kChunk>{}));
      auto tCcP        = thr_mma_qk_p.partition_C(cP);
      auto tSsQDecayed = thr_mma_qk_p.partition_A(sQDecayed);
      auto tSsKInverse = thr_mma_qk_p.partition_B(sKInverse);
      auto tSrQDecayed = thr_mma_qk_p.make_fragment_A(tSsQDecayed);
      auto tSrKInverse = thr_mma_qk_p.make_fragment_B(tSsKInverse);
      gemm(tiled_mma_qk_p, acc_p, tSrQDecayed, tSrKInverse, acc_p);
      mate::warpsquad_commit_batch();
      mate::warpsquad_wait();

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc_p) / VecSize; ++i) {
        int     base   = i * VecSize;
        float4  values = *reinterpret_cast<float4 const*>(&acc_p(base));
        Element packed[VecSize];
        simd::store_float4_to_packed4_rn(packed, values);

        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < VecSize; ++j) {
          int row      = int(get<0>(tCcP(base + j)));
          int col      = int(get<1>(tCcP(base + j)));
          sP(row, col) = row >= col ? packed[j] : Element(0.0f);
        }
      }
      named_barrier_arrive(KdaNamedBarrier::PReady);
    }
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void load(SharedStorage&     shared_storage,
                                  Params const&      params,
                                  ProblemSize const& problem_size,
                                  WorkDesc const&    work_desc,
                                  Pipeline&          pipeline,
                                  int                local_tid) {
    if (work_desc.n_chunks <= 0) {
      return;
    }

    auto sDtBias = make_tensor(make_smem_ptr(shared_storage.smem_dt_bias.data()), SmemLayoutDtBias{});
    load_dt_bias(sDtBias, params, work_desc.head_idx, local_tid);
    named_barrier_arrive_and_wait(KdaNamedBarrier::DtBiasLoaded);

    auto prefetch_q = [&](int chunk_idx) INLINE_LAMBDA {
      if (local_tid < mutlass::NumThreadsPerWarp) {
        LoadQ load_q(params.tme_q, pipeline.q, shared_storage.smem_q_raw);
        auto  q_src_dst = load_q.partition_SD(problem_size, TileShape{}, work_desc);
        load_q.step(q_src_dst, chunk_idx, pipeline.q_write);
      }
    };
    auto prefetch_k = [&](int chunk_idx) INLINE_LAMBDA {
      if (local_tid < mutlass::NumThreadsPerWarp) {
        LoadK load_k(params.tme_k, pipeline.k, shared_storage.smem_k_raw);
        auto  k_src_dst = load_k.partition_SD(problem_size, TileShape{}, work_desc);
        load_k.step(k_src_dst, chunk_idx, pipeline.k_write);
      }
    };
    auto prefetch_g = [&](int chunk_idx) INLINE_LAMBDA {
      if (local_tid < mutlass::NumThreadsPerWarp) {
        LoadG load_g(params.tme_g, pipeline.g, shared_storage.smem_g_raw);
        auto  g_src_dst = load_g.partition_SD(problem_size, TileShape{}, work_desc);
        load_g.step(g_src_dst, chunk_idx, pipeline.g_write);
      }
    };
    auto compute_gate_cumsum = [&](int chunk_idx, auto is_final_chunk) INLINE_LAMBDA {
      pipeline.g.consumer_wait(pipeline.g_read);
      auto sGRaw =
          make_tensor(make_smem_ptr(shared_storage.smem_g_raw.data()), SmemLayoutG{})(_, _, pipeline.g_read.index());
      auto sGCumsum = make_tensor(make_smem_ptr(shared_storage.smem_g_cumsum.data()), SmemLayoutGCumsum{});

      float a_log_exp  = load_a_log_exp(params, work_desc.head_idx);
      int   actual_len = work_desc.actual_len(chunk_idx);
      int   col        = local_tid;
      int   valid_len  = decltype(is_final_chunk)::value ? actual_len : kChunk;
      float cumsum_reg[kChunk];
      float dt = 0.0f;
      if constexpr (HasGateParams) {
        dt = sDtBias(col);
      }
      auto load_gate = [&](int row) INLINE_LAMBDA {
        float raw_g = static_cast<float>(sGRaw(make_coord(row, col)));
        if constexpr (decltype(is_final_chunk)::value) {
          if (row >= actual_len) {
            raw_g = static_cast<float>(inactive_gate_raw_element());
          }
        }
        return make_gate(raw_g, a_log_exp, dt, params);
      };

      // Keep this column's chunk prefix sums in registers so the average-shift path
      // does a single final shared-memory store instead of read-modify-writing g_cumsum.
      // NOTE: for lower_bound=-5, min(g_cumsum) goes down to -120(chunk=32), so we have to do
      // shifting(renormalization). However, the better way is to increase the lower_bound to -3(maybe), which
      // will also change the write/forget strength.
      cumsum_reg[0] = load_gate(0);
      MUTLASS_PRAGMA_UNROLL
      for (int row = 1; row < kChunk; ++row) {
        cumsum_reg[row] = cumsum_reg[row - 1] + load_gate(row);
      }
      float shift = 0.5f * (cumsum_reg[0] + cumsum_reg[valid_len - 1]);
      MUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < kChunk; ++row) {
        sGCumsum(make_coord(row, col)) = cumsum_reg[row] - shift;
      }
      auto sGShifted = make_tensor(make_smem_ptr(shared_storage.smem_g_shifted.data()), SmemLayoutGTotalStaged{})(_, 0);
      sGShifted(col) = ::mate::flat::detail::fast_exp2(shift);
      pipeline.g.consumer_release(pipeline.g_read);
      ++pipeline.g_read;
      if (chunk_idx + 1 < work_desc.n_chunks) {
        prefetch_g(chunk_idx + 1);
      }
      named_barrier_arrive(KdaNamedBarrier::GCumsumReady);
    };
    auto compute_g_totals = [&]() INLINE_LAMBDA {
      auto sGCumsum = make_tensor(make_smem_ptr(shared_storage.smem_g_cumsum.data()), SmemLayoutGCumsum{});
      auto sGTotalShifted =
          make_tensor(make_smem_ptr(shared_storage.smem_g_total_shifted.data()), SmemLayoutGTotalStaged{})(_, 0);
      auto sGTotalUnshifted =
          make_tensor(make_smem_ptr(shared_storage.smem_g_total_unshifted.data()), SmemLayoutGTotalStaged{})(_, 0);
      auto sGShifted = make_tensor(make_smem_ptr(shared_storage.smem_g_shifted.data()), SmemLayoutGTotalStaged{})(_, 0);
      int  col       = local_tid;
      float shifted_total   = ::mate::flat::detail::fast_exp2(sGCumsum(make_coord(kChunk - 1, col)));
      sGTotalShifted(col)   = shifted_total;
      sGTotalUnshifted(col) = shifted_total * sGShifted(col);
    };
    auto scale_normalized_qk_stage = [&](int chunk_idx, int stage, auto is_final_chunk) INLINE_LAMBDA {
      pipeline.q.consumer_wait(pipeline.q_read);
      pipeline.k.consumer_wait(pipeline.k_read);
      auto sKRaw     = make_tensor(make_smem_ptr(shared_storage.smem_k_raw.data()), SmemLayoutK{})(_, _, stage);
      auto sQRaw     = make_tensor(make_smem_ptr(shared_storage.smem_q_raw.data()), SmemLayoutQ{})(_, _, stage);
      auto sKDecayed = make_tensor(make_smem_ptr(shared_storage.smem_k_decayed.data()), SmemLayoutK{})(_, _, stage);
      auto sKInverse =
          make_tensor(make_smem_ptr(shared_storage.smem_k_inverse.data()), SmemLayoutKInverse{})(_, _, stage);
      auto sQDecayed  = make_tensor(make_smem_ptr(shared_storage.smem_q_decayed.data()), SmemLayoutQ{})(_, _, stage);
      auto sGCumsum   = make_tensor(make_smem_ptr(shared_storage.smem_g_cumsum.data()), SmemLayoutGCumsum{});
      int  actual_len = work_desc.actual_len(chunk_idx);

      bool is_not_full_chunk = decltype(is_final_chunk)::value && actual_len < kChunk;
      scale_normalized_qk(sKRaw,
                          sQRaw,
                          sGCumsum,
                          sKDecayed,
                          sKInverse,
                          sQDecayed,
                          params.scale,
                          local_tid,
                          actual_len,
                          is_not_full_chunk);
      pipeline.q.consumer_release(pipeline.q_read);
      pipeline.k.consumer_release(pipeline.k_read);
      ++pipeline.q_read;
      ++pipeline.k_read;
    };

    prefetch_g(0);
    prefetch_q(0);
    prefetch_k(0);

    auto producer_loop_body = [&](int chunk_idx, auto is_first_chunk, auto is_final_chunk) INLINE_LAMBDA {
      int      stage = chunk_idx % TmeStages;
      uint32_t phase = uint32_t(chunk_idx & 1);

      compute_gate_cumsum(chunk_idx, is_final_chunk);
      if constexpr (!decltype(is_first_chunk)::value) {
        named_barrier_wait(KdaNamedBarrier::OperandsConsumed, uint32_t((chunk_idx - 1) & 1));
      }
      named_barrier_wait(KdaNamedBarrier::GCumsumReady, phase);
      compute_g_totals();
      scale_normalized_qk_stage(chunk_idx, stage, is_final_chunk);
      if constexpr (!decltype(is_final_chunk)::value) {
        prefetch_q(chunk_idx + 1);
        prefetch_k(chunk_idx + 1);
      }
      named_barrier_arrive(KdaNamedBarrier::OperandsReady);
    };

    if (work_desc.n_chunks == 1) {
      producer_loop_body(0, true_type{}, true_type{});
    } else {
      producer_loop_body(0, true_type{}, false_type{});
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 1; chunk_idx + 1 < work_desc.n_chunks; ++chunk_idx) {
        producer_loop_body(chunk_idx, false_type{}, false_type{});
      }
      producer_loop_body(work_desc.n_chunks - 1, false_type{}, true_type{});
    }
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void compute_state(SharedStorage&     shared_storage,
                                           Params const&      params,
                                           ProblemSize const& problem_size,
                                           WorkDesc const&    work_desc,
                                           Pipeline&          pipeline,
                                           int                local_tid) {
    auto sState = make_tensor(make_smem_ptr(shared_storage.smem_state.data()), SmemLayoutState{});
    auto rState = partition_fragment_C(TiledMmaKU{}, make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}));

    kv_load(params, problem_size, work_desc, rState, local_tid);
    if (work_desc.n_chunks <= 0) {
      kv_store(params, problem_size, work_desc, rState, local_tid);
      return;
    }

    auto state_loop_body = [&](int chunk_idx, auto is_first_chunk, auto is_final_chunk) INLINE_LAMBDA {
      int      stage = chunk_idx % TmeStages;
      uint32_t phase = uint32_t(chunk_idx & 1);

      if constexpr (!decltype(is_first_chunk)::value) {
        named_barrier_wait(KdaNamedBarrier::StateConsumed, uint32_t((chunk_idx - 1) & 1));
      }
      named_barrier_wait(KdaNamedBarrier::GCumsumReady, phase);
      commit_state(sState, rState, shared_storage, stage, local_tid);

      named_barrier_wait(KdaNamedBarrier::OperandsReady, phase);
      if constexpr (!decltype(is_first_chunk)::value) {
        // inverse/P share one stage across chunks.  Do not overwrite the
        // previous tiles until both the state and output squads have consumed
        // the previous U/inverse/P epoch.
        named_barrier_wait(KdaNamedBarrier::VUpdatedConsumed, uint32_t((chunk_idx - 1) & 1));
      }
      prepare_state_operands(shared_storage, params, problem_size, work_desc, chunk_idx, local_tid, is_final_chunk);
      scale_state(rState, shared_storage, stage, local_tid);
      if constexpr (!decltype(is_final_chunk)::value) {
        named_barrier_arrive(KdaNamedBarrier::OperandsConsumed);
      }
      if constexpr (!(decltype(is_final_chunk)::value && !HasStateOut)) {
        issue_state_update(rState, shared_storage, stage, local_tid, phase);
        mate::warpsquad_wait();
      }
      if constexpr (!decltype(is_final_chunk)::value) {
        named_barrier_arrive(KdaNamedBarrier::VUpdatedConsumed);
      }
    };

    if (work_desc.n_chunks == 1) {
      state_loop_body(0, true_type{}, true_type{});
    } else {
      state_loop_body(0, true_type{}, false_type{});
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 1; chunk_idx + 1 < work_desc.n_chunks; ++chunk_idx) {
        state_loop_body(chunk_idx, false_type{}, false_type{});
      }
      state_loop_body(work_desc.n_chunks - 1, false_type{}, true_type{});
    }
    if constexpr (HasStateOut) {
      kv_store(params, problem_size, work_desc, rState, local_tid);
    }
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void compute_output(SharedStorage&     shared_storage,
                                            Params const&      params,
                                            ProblemSize const& problem_size,
                                            WorkDesc const&    work_desc,
                                            Pipeline&          pipeline,
                                            int                local_tid) {
    if (work_desc.n_chunks <= 0) {
      return;
    }

    auto prefetch_v = [&](int chunk_idx) INLINE_LAMBDA {
      if (local_tid < mutlass::NumThreadsPerWarp) {
        LoadV load_v(params.tme_v, pipeline.v, shared_storage.smem_v_raw);
        auto  v_src_dst = load_v.partition_SD(problem_size, TileShape{}, work_desc);
        load_v.step(v_src_dst, chunk_idx, pipeline.v_write);
      }
    };

    auto acc_output   = partition_fragment_C(TiledMmaQS{}, make_shape(Int<kChunk>{}, Int<kHeadDim>{}));
    auto issue_w_gemm = [&](int stage, uint32_t phase, auto& acc_w) INLINE_LAMBDA {
      auto sKDecayed = make_tensor(make_smem_ptr(shared_storage.smem_k_decayed.data()), SmemLayoutK{})(_, _, stage);
      auto sState    = make_tensor(make_smem_ptr(shared_storage.smem_state.data()), SmemLayoutState{});

      TiledMmaQS tiled_mma_qs;
      auto       thr_mma_qs = tiled_mma_qs.get_thread_slice(local_tid);
      clear(acc_w);
      auto tSsKDecayed = thr_mma_qs.partition_A(sKDecayed);
      auto tSsState    = thr_mma_qs.partition_B(sState);
      auto tSrKDecayed = thr_mma_qs.make_fragment_A(tSsKDecayed);
      auto tSrState    = thr_mma_qs.make_fragment_B(tSsState);
      named_barrier_wait(KdaNamedBarrier::OperandsReady, phase);
      named_barrier_wait(KdaNamedBarrier::StateCommitted, phase);
      gemm(tiled_mma_qs, acc_w, tSrKDecayed, tSrState, acc_w);
      mate::warpsquad_commit_batch();
    };
    auto issue_output_gemm = [&](int stage, auto& acc_output_ref) INLINE_LAMBDA {
      auto sQDecayed = make_tensor(make_smem_ptr(shared_storage.smem_q_decayed.data()), SmemLayoutQ{})(_, _, stage);
      auto sState    = make_tensor(make_smem_ptr(shared_storage.smem_state.data()), SmemLayoutState{});

      TiledMmaQS tiled_mma_qs;
      auto       thr_mma_qs = tiled_mma_qs.get_thread_slice(local_tid);
      clear(acc_output_ref);
      auto tSsQDecayed = thr_mma_qs.partition_A(sQDecayed);
      auto tSsState    = thr_mma_qs.partition_B(sState);
      auto tSrQDecayed = thr_mma_qs.make_fragment_A(tSsQDecayed);
      auto tSrState    = thr_mma_qs.make_fragment_B(tSsState);
      gemm(tiled_mma_qs, acc_output_ref, tSrQDecayed, tSrState, acc_output_ref);
      mate::warpsquad_commit_batch();
    };
    auto prepare_k_restored_half = [&](int stage, int row_half_base) INLINE_LAMBDA {
      auto sKInverse =
          make_tensor(make_smem_ptr(shared_storage.smem_k_inverse.data()), SmemLayoutKInverse{})(_, _, stage);
      auto sGTotal =
          make_tensor(make_smem_ptr(shared_storage.smem_g_total_shifted.data()), SmemLayoutGTotalStaged{})(_, stage);
      auto sKRestored =
          make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestored{})(_, _, stage);

      constexpr int ThreadsPerRow     = 8;
      constexpr int RowsPerWarp       = mutlass::NumThreadsPerWarp / ThreadsPerRow;
      constexpr int ElementsPerThread = kHeadDim / ThreadsPerRow;
      constexpr int VectorsPerThread  = ElementsPerThread / VecSize;
      int           warp_idx          = local_tid / mutlass::NumThreadsPerWarp;
      int           lane_idx          = local_tid % mutlass::NumThreadsPerWarp;
      int           row_in_warp       = lane_idx / ThreadsPerRow;
      int           lane_in_row       = lane_idx % ThreadsPerRow;
      int           thread_col_lo     = lane_in_row * ElementsPerThread;
      int           row               = row_half_base + warp_idx * RowsPerWarp + row_in_warp;

      MUTLASS_PRAGMA_UNROLL
      for (int vec_iter = 0; vec_iter < VectorsPerThread; ++vec_iter) {
        int    vec_slot = (vec_iter + row_in_warp) % VectorsPerThread;
        int    col_base = thread_col_lo + vec_slot * VecSize;
        float4 values   = load_qk_vec4(sKInverse, row, col_base);
        float4 scales   = simd::load_float4(&sGTotal(col_base));
        simd::store_float4_to_packed4_rn(&sKRestored(make_coord(col_base, row)), simd::vmul(values, scales));
      }
    };
    auto compute_v_residual = [&](int stage, auto& acc_w) INLINE_LAMBDA {
      pipeline.v.consumer_wait(pipeline.v_read);
      auto sV = make_tensor(make_smem_ptr(shared_storage.smem_v_raw.data()), SmemLayoutV{})(_, _, stage);
      auto sU = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);

      using AccumVec = mutlass::Array<float, VecSize>;
      auto w_vec     = recast<AccumVec const>(acc_w);

      TiledMmaQS tiled_mma_qs;
      auto       thr_mma_qs = tiled_mma_qs.get_thread_slice(local_tid);
      auto       cV         = make_identity_tensor(make_shape(Int<kChunk>{}, Int<kHeadDim>{}));
      auto       tCcV       = thr_mma_qs.partition_C(cV);

      MUTLASS_PRAGMA_UNROLL
      for (int vec_idx = 0; vec_idx < size(w_vec); ++vec_idx) {
        auto row = get<0>(tCcV(vec_idx * VecSize));
        auto col = get<1>(tCcV(vec_idx * VecSize));
        // QS C coordinates transpose each 64-element V half independently.
        // A single 8x16 transpose would write the residual into the wrong U rows.
        auto   natural_col = InternalToNaturalVLayout{}(col);
        float4 v_old       = simd::load_packed4_as_float4(&sV(make_coord(natural_col, row)));
        float4 update      = reinterpret_cast<float4 const&>(w_vec(vec_idx));
        simd::store_float4_to_packed4_rn(&sU(make_coord(natural_col, row)), simd::vsub(v_old, update));
      }
      pipeline.v.consumer_release(pipeline.v_read);
      ++pipeline.v_read;
    };
    auto issue_intra_output = [&](int stage, uint32_t phase) INLINE_LAMBDA {
      auto sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{})(_, _, stage);
      auto sU = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);

      TiledMmaPU tiled_mma_pu;
      auto       thr_mma_pu = tiled_mma_pu.get_thread_slice(local_tid);
      auto       tSsP       = thr_mma_pu.partition_A(sP);
      auto       tSsU       = thr_mma_pu.partition_B(sU);
      auto       tSrP       = thr_mma_pu.make_fragment_A(tSsP);
      auto       tSrU       = thr_mma_pu.make_fragment_B(tSsU);
      named_barrier_wait(KdaNamedBarrier::PReady, phase);
      gemm(tiled_mma_pu, acc_output, tSrP, tSrU, acc_output);
      mate::warpsquad_commit_batch();
    };
    auto store_output = [&](int chunk_idx, auto is_final_chunk) INLINE_LAMBDA {
      int  actual_len        = work_desc.actual_len(chunk_idx);
      bool is_not_full_chunk = decltype(is_final_chunk)::value && actual_len < kChunk;
      auto mOut              = make_tensor(make_gmem_ptr(params.ptr_out),
                              make_shape(Int<kHeadDim>{}, problem_size.T, make_shape(problem_size.H, problem_size.B)),
                              params.stride_out);
      auto permuted_out =
          make_tensor(mOut.data(), composition(mOut.layout(), make_tile(InternalToNaturalVLayout{}, _, _)));
      TiledMmaQS tiled_mma_output = TiledMmaQS{};
      auto       thr_mma_output   = tiled_mma_output.get_thread_slice(local_tid);
      auto       cOutput          = make_identity_tensor(make_shape(Int<kChunk>{}, Int<kHeadDim>{}));
      auto       tCcOutput        = thr_mma_output.partition_C(cOutput);
      using AccumVec              = mutlass::Array<float, VecSize>;
      using ElementVec            = mutlass::Array<Element, VecSize>;
      auto output_vec             = recast<AccumVec const>(acc_output);
      mutlass::NumericArrayConverter<Element, float, VecSize, mutlass::FloatRoundStyle::round_to_nearest>
          convert_output;
      MUTLASS_PRAGMA_UNROLL
      for (int vec_idx = 0; vec_idx < size(output_vec); ++vec_idx) {
        int        base   = vec_idx * VecSize;
        ElementVec packed = convert_output(output_vec(vec_idx));

        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < VecSize; ++j) {
          int row = int(get<0>(tCcOutput(base + j)));
          int col = int(get<1>(tCcOutput(base + j)));
          if (!is_not_full_chunk || row < actual_len) {
            auto* ptr = &permuted_out(make_coord(
                col, work_desc.tme_token(chunk_idx) + row, make_coord(work_desc.head_idx, work_desc.tme_batch())));
            *ptr      = packed[j];
          }
        }
      }
    };

    prefetch_v(0);

    auto output_loop_body = [&](int chunk_idx, auto is_first_chunk, auto is_final_chunk) INLINE_LAMBDA {
      int      stage = chunk_idx % TmeStages;
      uint32_t phase = uint32_t(chunk_idx & 1);
      auto     acc_w = partition_fragment_C(TiledMmaQS{}, make_shape(Int<kChunk>{}, Int<kHeadDim>{}));

      issue_w_gemm(stage, phase, acc_w);
      prepare_k_restored_half(stage, 0);
      mate::warpsquad_wait();
      compute_v_residual(stage, acc_w);
      if constexpr (!decltype(is_final_chunk)::value) {
        prefetch_v(chunk_idx + 1);
      }
      issue_output_gemm(stage, acc_output);
      prepare_k_restored_half(stage, 16);
      named_barrier_arrive(KdaNamedBarrier::KRestoredReady);
      mate::warpsquad_wait();
      // The producer-owned q/k/g-total tiles and the committed state tile are
      // no longer read after the output squad's prior MMAs have completed.
      // Release those two dependencies independently; U is still live until
      // the intra-output MMA below finishes.
      if constexpr (!decltype(is_final_chunk)::value) {
        named_barrier_arrive(KdaNamedBarrier::OperandsConsumed);
        named_barrier_arrive(KdaNamedBarrier::StateConsumed);
      }
      update_v(shared_storage, stage, local_tid, phase);
      issue_intra_output(stage, phase);
      mate::warpsquad_wait();
      if constexpr (!decltype(is_final_chunk)::value) {
        // Both state and output have now consumed this scratch epoch.  The next
        // state chunk may reuse inverse/P, and output's sequential execution
        // guarantees U is not reused before the state squad reaches its next
        // StateCommitted handoff.
        named_barrier_arrive(KdaNamedBarrier::VUpdatedConsumed);
      }
      store_output(chunk_idx, is_final_chunk);
    };

    if (work_desc.n_chunks == 1) {
      output_loop_body(0, true_type{}, true_type{});
    } else {
      output_loop_body(0, true_type{}, false_type{});
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 1; chunk_idx + 1 < work_desc.n_chunks; ++chunk_idx) {
        output_loop_body(chunk_idx, false_type{}, false_type{});
      }
      output_loop_body(work_desc.n_chunks - 1, false_type{}, true_type{});
    }
  }
};

}  // namespace mate::flat::kda
