#pragma once

#include "mute/tensor.hpp"

#include "mutlass/mutlass.h"
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/numeric_conversion.h"


#include "collective/fmha_load_primitive_builder.hpp"
#include "collective/fmha_collective_load.hpp"
#include "collective/fmha_collective_pipeline.hpp"
#include "collective/fmha_collective_softmax.hpp"
#include "collective/fmha_common.hpp"
#include "fmha_options.hpp"


namespace mutlass::fmha::collective {

using namespace mute;


template <
  class Element_,
  class ElementAccumulator_,
  class TileShape_,
  class StrideQ_,
  class StrideK_,
  class StrideV_,
  class Fusion,
  class... Options
>
struct FmhaMainloopTmeWarpSpecialized {
  using Element = Element_;
  using ElementAccumulator = ElementAccumulator_;

  using TileShape = TileShape_;
  using StrideQ   = StrideQ_;
  using StrideK   = StrideK_;
  using StrideV   = StrideV_;

  static constexpr int NumLoadWarpSquads = 1;
  static constexpr int NumMmaWarpSquads = find_option_t<Tag::NumMmaWarpSquads, Int<3>, Options...>::value;

  static_assert(get<0>(TileShape{}) % NumMmaWarpSquads == 0, "TileShapeQ must be multiple of NumMmaWarpSquads");

  static constexpr int Alignment = 32 / sizeof_bits_v<Element>;

  static constexpr bool IsVarlen = find_option_t<Tag::Varlen, false_type, Options...>::value;

  using TileShapeQKD = Shape<
    decltype(tuple_element_t<0, TileShape>{} / Int<NumMmaWarpSquads>{}),
    tuple_element_t<1, TileShape>,
    tuple_element_t<2, TileShape>>;

  using TileShapePDV = Shape<
    decltype(tuple_element_t<0, TileShape>{} / Int<NumMmaWarpSquads>{}),
    tuple_element_t<3, TileShape>,
    tuple_element_t<1, TileShape>>;

  using CollectiveMmaQK = typename mutlass::gemm::collective::CollectiveBuilder<
    mutlass::arch::Mp31, mutlass::arch::OpClassTensorOp,
    Element, StrideQ, Alignment,
    Element, StrideK, Alignment,
    ElementAccumulator,
    TileShapeQKD,
    Shape<_1, _1, _1>,
    _2,
    mutlass::gemm::KernelTme>::CollectiveOp;

  using CollectiveMmaPV = typename mutlass::gemm::collective::CollectiveBuilder<
    mutlass::arch::Mp31, mutlass::arch::OpClassTensorOp,
    Element, StrideK, Alignment,
    Element, decltype(select<1, 0, 2>(StrideV{})), Alignment,
    ElementAccumulator,
    TileShapePDV,
    Shape<_1, _1, _1>,
    _2,
    mutlass::gemm::KernelTme>::CollectiveOp;

  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
  using TiledMmaPV = typename CollectiveMmaPV::TiledMma;

  using SmemAtomLayoutQ = typename CollectiveMmaQK::SmemLayoutAtomA;
  using SmemAtomLayoutK = typename CollectiveMmaQK::SmemLayoutAtomB;

  static constexpr int KStage = find_option_t<Tag::KStage, Int<2>, Options...>::value;
  static constexpr int VStage = find_option_t<Tag::VStage, Int<2>, Options...>::value;
  static_assert(KStage >= VStage);

  using SmemLayoutQFull = decltype(tile_to_shape(SmemAtomLayoutQ{}, select<0, 2>(TileShape{})));
  using SmemLayoutQ = decltype(SmemLayoutQFull{}.compose(make_layout(make_shape(_64{}, Int<size<0>(SmemLayoutQFull{}) / 64>{})), Underscore{}));
  using SmemLayoutK = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutB{}, Int<KStage>{}));
  using SmemLayoutP = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutA{}, Int<NumMmaWarpSquads>{}));
  using SmemLayoutV = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutB{}, Int<VStage>{}));

  using TmeLoadKeyBuilder = Mp31FmhaTmeLoadKeyBuilder<Element, SmemLayoutK, StrideK>;
  using LsuLoadKeyBuilder = Mp31FmhaLsuLoadKeyBuilder<Element, TileShapeQKD, SmemAtomLayoutK,
                                                      NumLoadWarpSquads * NumThreadsPerWarpSquad,
                                                      StrideK>;

  static_assert(TmeLoadKeyBuilder::Fragment == LsuLoadKeyBuilder::Fragment);
  // static_assert(std::is_same_v<typename TmeLoadKeyBuilder::PermuteTile,
  //                              typename LsuLoadKeyBuilder::PermuteTile>);

  static constexpr int FragmentSize = TmeLoadKeyBuilder::Fragment;
  using FragmentType = typename TmeLoadKeyBuilder::FragmentType;
  using PermuteTile = typename TmeLoadKeyBuilder::PermuteTile;

  using TmeTileShapeKD = typename TmeLoadKeyBuilder::TmeTileShapeKD;
  using LsuTileShapeKD = typename LsuLoadKeyBuilder::LsuTileShapeKD;

  using TME_Q = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
        make_gmem_ptr(static_cast<Element const*>(nullptr)),
        repeat_like(StrideQ{}, int32_t(0)),
        StrideQ{}
      ),
      SmemLayoutQFull{},
      select<0, 2>(TileShape{})));

  using TME_K = typename TmeLoadKeyBuilder::TME_K;

  using LSU_K = typename LsuLoadKeyBuilder::GmemTiledCopy;

  using TME_V = typename CollectiveMmaPV::Params::TME_B;

  using R2SCopyAtom = Copy_Atom<UniversalCopy<FragmentType>, Element>;
  using R2STiledCopy = decltype(make_tiled_copy_C(R2SCopyAtom{},
                          convert_to_permuted_sqmma(TiledMmaQK{}, PermuteTile{})));

  struct SharedStorage {
    mute::array_aligned<Element, cosize_v<SmemLayoutQ>, 256> smem_q;
    mute::array_aligned<Element, cosize_v<SmemLayoutK>, 256> smem_k;
    mute::array_aligned<Element, cosize_v<SmemLayoutP>, 256> smem_p;
    mute::array_aligned<Element, cosize_v<SmemLayoutV>, 256> smem_v;
  };

  static constexpr bool SIMT_KEY = false;
  using MainloopPipelineK = conditional_t<SIMT_KEY, mutlass::Mp31PipelineAsync<VStage>, mutlass::Mp31PipelineTmeLoadKAsync<KStage>>;
  using MainloopPipelineV = mutlass::Mp31PipelineTmeAsync<VStage>;

  using PipelineKParams = typename MainloopPipelineK::Params;
  using PipelineKState = typename MainloopPipelineK::PipelineState;

  using PipelineVParams = typename MainloopPipelineV::Params;
  using PipelineVState = typename MainloopPipelineV::PipelineState;

  static constexpr int TmeTransactionBytesQ = bits_to_bytes(size(SmemLayoutQ{}) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesK = bits_to_bytes(size(take<0,2>(SmemLayoutK{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesV = bits_to_bytes(size(take<0,2>(SmemLayoutV{})) * sizeof_bits_v<Element>);

  struct Arguments {
    Element const* ptr_Q;
    StrideQ stride_Q;
    Element const* ptr_K;
    StrideK stride_K;
    Element const* ptr_V;
    StrideV stride_V;

    float sm_scale;
  };

  struct Params {
    TME_Q tme_Q;
    TME_K tme_K;
    TME_V tme_V;

    LSU_K lsu_K;
    Element const* ptr_K;
    RobustDescriptor desc_K;
    StrideK stride_K;

    float sm_scale;
    float sm_scale_log2;
    FastDivmod fast_divmod_hr;
  };

  using TmeLoadQ = mutlass::fmha::collective::CollectiveLoad<
    mutlass::fmha::collective::LoadKind::LoadQ,
    MainloopPipelineK, // dummy template
    Element,
    SmemLayoutQFull,
    TME_Q
  >;

  using TmeLoadK = mutlass::fmha::collective::CollectiveLoad<
    mutlass::fmha::collective::LoadKind::LoadKWithTme,
    MainloopPipelineK,
    Element,
    typename TmeLoadKeyBuilder::SmemLayoutK,
    TME_K
  >;

  using LsuLoadK = mutlass::fmha::collective::CollectiveLoad<
    mutlass::fmha::collective::LoadKind::LoadKWithLsu,
    MainloopPipelineK,
    Element,
    SmemLayoutK,
    Params
  >;


  using TmeLoadV = mutlass::fmha::collective::CollectiveLoad<
    mutlass::fmha::collective::LoadKind::LoadV,
    MainloopPipelineV,
    Element,
    SmemLayoutV,
    TME_V
  >;


  template <class ProblemSize>
  static Params
  to_underlying_arguments(ProblemSize problem_size, Arguments const& args, void* workspace = nullptr) {
    auto [Q, K, HeadDimQK, HeadDimVO, H, H_KV, B] = problem_size;

    int total_seq_q = 0;
    if constexpr (is_variable_length_v<decltype(Q)>) {
      total_seq_q = Q.total_length;
    } else {
      total_seq_q = Q;
    }

    int total_seq_k = 0;
    if constexpr (is_variable_length_v<decltype(K)>) {
      total_seq_k = K.total_length;
    } else {
      total_seq_k = K;
    }


    auto problem_size_q = make_shape(total_seq_q, HeadDimQK, make_shape(H, B));
    auto tme_Q = make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
        make_gmem_ptr(args.ptr_Q),
        problem_size_q,
        args.stride_Q
      ),
      SmemLayoutQFull{},
      select<0,2>(TileShape{}));


    auto problem_size_k = make_shape(total_seq_k, HeadDimQK, make_shape(H_KV, B));
    auto tme_K = TmeLoadKeyBuilder::make_tme_copy(
      make_tensor(
        make_gmem_ptr(args.ptr_K),
        problem_size_k,
        args.stride_K
      ));

    auto problem_shape_pv = make_shape(total_seq_q, HeadDimVO, total_seq_k, make_shape(H_KV, B));

    auto params_pv = CollectiveMmaPV::to_underlying_arguments(
        problem_shape_pv,
        typename CollectiveMmaPV::Arguments {
          args.ptr_K, args.stride_K,
          args.ptr_V, select<1, 0, 2>(args.stride_V)
          }, nullptr);

    auto layout_k = make_layout(problem_size_k, args.stride_K);
    RobustDescriptor desc_K = make_robust_desc(args.ptr_K, cosize(layout_k));

    float const log2e = std::log2(std::exp(1.0f));
    int H_R = H / H_KV;

    mutlass::FastDivmod fast_divmod_hr(H_R);

    return {
      tme_Q,
      tme_K,
      params_pv.tme_load_b,
      LSU_K{},
      args.ptr_K,
      desc_K,
      args.stride_K,
      args.sm_scale,
      args.sm_scale * log2e,
      fast_divmod_hr,
    };
  }

  template <
    class BlkCoordQHB,
    class BarrierTuple,
    class ProblemSize,
    class BlkOffset
  >
  MUTE_DEVICE
  void load(Params const& params, SharedStorage& shared_storage, BarrierTuple& barrier_tuple,
            MainloopPipelineK& pipeline_k, PipelineKState& pipe_state_k,
            MainloopPipelineV& pipeline_v, PipelineVState& pipe_state_v,
            BlkCoordQHB const& blk_coord, ProblemSize const& problem_size,
            BlkOffset& blk_offset, int const prev_mask_boundary, int const warp_idx) {

    enum class ProducerWarpRole {
      LoadQKV,
      Warp1,
      Warp2,
      Warp3,
    };

    auto producer_warp_role = ProducerWarpRole(warp_idx);

    Fusion fusion;
    fusion.params.prev_mask_boundary = prev_mask_boundary;

    int fusion_tile_count = fusion.get_trip_count(blk_coord, TileShape{}, problem_size);

    int kv_tile_iter = 0;

    if constexpr (SIMT_KEY) {
      auto [barrier_q, barrier_tail] = barrier_tuple;

      TmeLoadQ load_q {params.tme_Q, /* dummy param */pipeline_k, shared_storage.smem_q};
      auto [tQgQ, tQsQ] = load_q.init_state(problem_size, select<0,1,2>(TileShape{}), blk_coord, blk_offset);

      if (producer_warp_role == ProducerWarpRole::LoadQKV) {
        barrier_q.arrive_and_expect_tx(TmeTransactionBytesQ);
        uint32_t bar_id = barrier_q.get_barrier_id();
        copy(params.tme_Q.with(bar_id), tQgQ, tQsQ);
      }

      LsuLoadK load_k {params, pipeline_k, shared_storage.smem_k};
      auto init_state_k = load_k.init_state(problem_size, LsuTileShapeKD{}, blk_coord, blk_offset);

      TmeLoadV load_v {params.tme_V, pipeline_v, shared_storage.smem_v};
      auto init_state_v = load_v.init_state(problem_size, TileShapePDV{}, blk_coord, blk_offset);

      while (kv_tile_iter < fusion_tile_count) {
        pipeline_k.producer_acquire(pipe_state_k);
        load_k.step(kv_tile_iter, init_state_k, pipe_state_k);
        if (producer_warp_role == ProducerWarpRole::LoadQKV) {
          load_v.step(kv_tile_iter, init_state_v, pipe_state_v);
        }

        mute::ldgsts_wait();
        pipeline_k.producer_commit(pipe_state_k);
        ++pipe_state_k;

        ++kv_tile_iter;
      }
    }
    else {

      auto [barrier_q, barrier_tail] = barrier_tuple;

      TmeLoadV load_v {params.tme_V, pipeline_v, shared_storage.smem_v};
      auto init_state_v = load_v.init_state(problem_size, TileShapePDV{}, blk_coord, blk_offset);

      if (producer_warp_role == ProducerWarpRole::LoadQKV) {
        TmeLoadQ load_q {params.tme_Q, /* dummy param */pipeline_k, shared_storage.smem_q};
        auto [tQgQ, tQsQ] = load_q.init_state(problem_size, select<0,1,2>(TileShape{}), blk_coord, blk_offset);

        barrier_q.arrive_and_expect_tx(TmeTransactionBytesQ);
        uint32_t bar_id = barrier_q.get_barrier_id();
        copy(params.tme_Q.with(bar_id), tQgQ, tQsQ);

        TmeLoadK load_k {params.tme_K, pipeline_k, shared_storage.smem_k};
        auto problem_size_for_key = TmeLoadKeyBuilder::get_problem_size(problem_size);
        // TODO: adapt BN=64 case

        auto blk_offset_for_key = [&] {
          if constexpr (TmeLoadKeyBuilder::BlockFit) {
            return make_coord(
              _0{},
              make_coord(_0{}, get<1>(blk_offset) / TmeLoadKeyBuilder::Fragment)
            );
          } else {
            return make_coord(
              _0{},
              make_coord(_0{}, _0{}, get<1>(blk_offset) / TmeLoadKeyBuilder::Granularity)
            );
          }
        }();

        auto init_state_k = load_k.init_state(
            problem_size_for_key, TmeTileShapeKD{}, blk_coord, blk_offset_for_key);

        while (kv_tile_iter < fusion_tile_count - 1) {
          load_k.step(kv_tile_iter, init_state_k, pipe_state_k);
          load_v.step(kv_tile_iter, init_state_v, pipe_state_v);
          ++kv_tile_iter;
        }

      }

      // SIMT tail
      {
        // alignment
        kv_tile_iter = fusion_tile_count - 1;
        if (producer_warp_role != ProducerWarpRole::LoadQKV) {
          pipe_state_k.advance(fusion_tile_count - 1);
          pipe_state_v.advance(fusion_tile_count - 1);
        }

        LsuLoadK load_k {params, pipeline_k, shared_storage.smem_k};
        auto init_state_k = load_k.init_state(problem_size, LsuTileShapeKD{}, blk_coord, blk_offset);

        // make sure previous loads are issued
        barrier_tail.sync();

        pipeline_k.lsu_producer_acquire(pipe_state_k);

        load_k.step(kv_tile_iter, init_state_k, pipe_state_k);

        mute::ldgsts_wait();

        barrier_tail.sync();

        if (producer_warp_role == ProducerWarpRole::LoadQKV) {
          // Only one warp commit since the pipeline is used for TME
          pipeline_k.lsu_producer_commit(pipe_state_k);
          load_v.step(kv_tile_iter, init_state_v, pipe_state_v);
        }
      }
    }
  }

  template <
    class BlkCoordQHB,
    class ProblemSize,
    class BarrierQ,
    class MathOrderBarrier
  >
  MUTLASS_DEVICE auto
  compute(
      BlkCoordQHB const& blk_coord, Params const& params,
      ProblemSize const& problem_size, int const& prev_mask_boundary,
      MainloopPipelineK& pipeline_k, PipelineKState& smem_pipe_k_read,
      MainloopPipelineV& pipeline_v, PipelineVState& smem_pipe_v_read,
      BarrierQ& barrier_q, MathOrderBarrier& math_order_barrier, SharedStorage& storage,
      int const thread_idx, int const consumer_idx) {

    // Prepare QK MMA
    TiledMmaQK tiled_mma_qk;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});

    Tensor tSsQ = thr_mma_qk.partition_A(sQ(make_coord(_, consumer_idx),_));
    Tensor tSrQ = thr_mma_qk.make_fragment_A(tSsQ);

    Tensor tSsK = thr_mma_qk.partition_B(sK);
    Tensor tSrK = thr_mma_qk.make_fragment_B(tSsK);

    // Prepare PV MMA
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

    Tensor sP = make_tensor(make_smem_ptr(storage.smem_p.data()), SmemLayoutP{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    Tensor tOsP = thr_mma_pv.partition_A(sP(_,_,consumer_idx));
    Tensor tOrP = thr_mma_pv.make_fragment_A(tOsP);

    Tensor tOsV = thr_mma_pv.partition_B(sV);
    Tensor tOrV = thr_mma_pv.make_fragment_B(tOsV);

    // Prepare R2S
    R2STiledCopy tiled_copy_r2s;
    ThrCopy thr_copy_r2s = tiled_copy_r2s.get_thread_slice(thread_idx);
    Tensor tPsP = thr_copy_r2s.partition_D(sP(_,_,consumer_idx));

    // AccQK index
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQKD{}));
    Tensor tPcP = convert_to_permuted_sqmma(tiled_mma_qk, PermuteTile{}).get_thread_slice(thread_idx).partition_C(cP);
    int m_coord = get<0>(blk_coord);

    tPcP.data() = tPcP.data() + E<0>{} * (m_coord * get<0>(TileShape{}) + consumer_idx * get<0>(TileShapeQKD{}));

    auto convert_and_sts = [&] (auto& accum_qk) {
      Tensor accum_cvt = make_fragment_like<Element>(accum_qk);

      Tensor tPrP = thr_copy_r2s.retile_S(accum_cvt);

      Tensor tCvt_frg = recast<mutlass::Array<Element, FragmentSize>>(accum_cvt);
      Tensor tAcc_frg = recast<mutlass::Array<ElementAccumulator, FragmentSize>>(accum_qk);

      MUTE_UNROLL
      for (int i = 0; i < size(tCvt_frg); ++i) {
        tCvt_frg(i) = mutlass::NumericArrayConverter<Element, ElementAccumulator, FragmentSize, FloatRoundStyle::round_to_nearest>{}(tAcc_frg(i));
      }

      copy(tiled_copy_r2s, tPrP, tPsP);
      __syncwarp();
    };

    Fusion fusion;
    fusion.params.prev_mask_boundary = prev_mask_boundary;

    int kv_tile_count = fusion.get_unmasked_trip_count(blk_coord, TileShape{}, problem_size);

    Tensor acc_pv = partition_fragment_C(tiled_mma_pv, take<0,2>(TileShapePDV{}));

    // trigger zero init sqmma
    clear(acc_pv);

    CollectiveSoftmax<Fusion, decltype(params), IsVarlen> softmax(params, fusion);
    auto softmax_state = softmax.init(acc_pv, tiled_mma_pv);

    // wait Q
    barrier_q.wait(0);

    {
      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
      // trigger zero init sqmma
      clear(acc_qk);

      pipeline_k.consumer_wait(smem_pipe_k_read);

      //math_order_barrier.wait();

      // MMA QK
      mute::gemm(tiled_mma_qk, tSrQ, tSrK(_,_,_,smem_pipe_k_read.index()), acc_qk);

      //math_order_barrier.arrive();


      warpsquad_wait();
      pipeline_k.consumer_release(smem_pipe_k_read);
      ++smem_pipe_k_read;

      // Softmax
      softmax.step(acc_qk, tiled_mma_qk, softmax_state, tPcP, problem_size);

      // Sts
      convert_and_sts(acc_qk);

      pipeline_v.consumer_wait(smem_pipe_v_read);

      // MMA PV
      mute::gemm(tiled_mma_pv, tOrP, tOrV(_,_,_,smem_pipe_v_read.index()), acc_pv);

      warpsquad_wait();
      pipeline_v.consumer_release(smem_pipe_v_read);
      ++smem_pipe_v_read;

      --kv_tile_count;
      tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShape{});
    }

    while (kv_tile_count > 0)
    {
      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
      // trigger zero init sqmma
      clear(acc_qk);

      pipeline_k.consumer_wait(smem_pipe_k_read);

      // MMA QK
      mute::gemm(tiled_mma_qk, tSrQ, tSrK(_,_,_,smem_pipe_k_read.index()), acc_qk);


      warpsquad_wait();
      pipeline_k.consumer_release(smem_pipe_k_read);
      ++smem_pipe_k_read;


      // Softmax
      softmax.template step<false>(acc_qk, tiled_mma_qk, softmax_state, acc_pv, tiled_mma_pv, tPcP, problem_size);

      // Sts
      convert_and_sts(acc_qk);

      pipeline_v.consumer_wait(smem_pipe_v_read);

      // MMA PV
      mute::gemm(tiled_mma_pv, tOrP, tOrV(_,_,_,smem_pipe_v_read.index()), acc_pv);


      warpsquad_wait();
      pipeline_v.consumer_release(smem_pipe_v_read);
      ++smem_pipe_v_read;

      --kv_tile_count;
      tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShape{});
    }

    kv_tile_count += fusion.get_masked_trip_count(blk_coord, TileShape{}, problem_size);

    while (kv_tile_count > 0)
    {
      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
      // trigger zero init sqmma
      clear(acc_qk);

      pipeline_k.consumer_wait(smem_pipe_k_read);
          
      // MMA QK
      mute::gemm(tiled_mma_qk, tSrQ, tSrK(_,_,_,smem_pipe_k_read.index()), acc_qk);

      warpsquad_wait();
      pipeline_k.consumer_release(smem_pipe_k_read);
      ++smem_pipe_k_read;

      // Softmax
      softmax.template step<true>(acc_qk, tiled_mma_qk, softmax_state, acc_pv, tiled_mma_pv, tPcP, problem_size);

      // Sts
      convert_and_sts(acc_qk);

      pipeline_v.consumer_wait(smem_pipe_v_read);

      // MMA PV
      mute::gemm(tiled_mma_pv, tOrP, tOrV(_,_,_,smem_pipe_v_read.index()), acc_pv);

      warpsquad_wait();
      pipeline_v.consumer_release(smem_pipe_v_read);
      ++smem_pipe_v_read;

      --kv_tile_count;
      tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShape{});
    }

    Tensor lse = softmax.tail(softmax_state, acc_pv, tiled_mma_pv);

    return make_tuple(acc_pv, lse);
  }
};

} // namespace mutlass::fmha::collective
