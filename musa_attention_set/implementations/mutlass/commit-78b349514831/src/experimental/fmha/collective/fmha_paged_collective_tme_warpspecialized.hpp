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
struct FmhaPagedMainloopTmeWarpSpecialized {
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

  static constexpr bool BarrierSchedule = false;

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

  static constexpr int KStage = find_option_t<Tag::KStage, Int<2>, Options...>::value;
  static constexpr int VStage = find_option_t<Tag::VStage, Int<2>, Options...>::value;
  static_assert(KStage >= VStage);

  using SmemAtomLayoutQ = typename CollectiveMmaQK::SmemLayoutAtomA;
  using SmemAtomLayoutK = typename CollectiveMmaQK::SmemLayoutAtomB;

  using SmemLayoutQFull = decltype(tile_to_shape(SmemAtomLayoutQ{}, select<0, 2>(TileShape{})));
  using SmemLayoutQ = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<NumMmaWarpSquads>{}));
  using SmemLayoutK = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutB{}, Int<KStage>{}));
  using SmemLayoutP = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutA{}, Int<NumMmaWarpSquads>{}));
  using SmemLayoutV = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutB{}, Int<VStage>{}));

  using TmeLoadKeyBuilder = Mp31FmhaTmeLoadKeyBuilder<Element, SmemLayoutK, StrideK>;

  static constexpr int FragmentSize = TmeLoadKeyBuilder::Fragment;
  using FragmentType = typename TmeLoadKeyBuilder::FragmentType;
  using PermuteTile = typename TmeLoadKeyBuilder::PermuteTile;

  using TmeTileShapeKD = typename TmeLoadKeyBuilder::TmeTileShapeKD;

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
  using TME_V = typename CollectiveMmaPV::Params::TME_B;

  using R2SCopyAtom = Copy_Atom<UniversalCopy<FragmentType>, Element>;
  using R2STiledCopy = decltype(make_tiled_copy_C(R2SCopyAtom{},
                          convert_to_permuted_sqmma(TiledMmaQK{}, PermuteTile{})));

  using MainloopPipelineQ = mutlass::Mp31PipelineTmeAsync<1>;
  using PipelineQParams = typename MainloopPipelineQ::Params;
  using PipelineQState = typename MainloopPipelineQ::PipelineState;

  using MainloopPipelineK = mutlass::Mp31PipelineTmeAsync<KStage>;
  using PipelineKParams = typename MainloopPipelineK::Params;
  using PipelineKState = typename MainloopPipelineK::PipelineState;

  using MainloopPipelineV = mutlass::Mp31PipelineTmeAsync<VStage>;
  using PipelineVParams = typename MainloopPipelineV::Params;
  using PipelineVState = typename MainloopPipelineV::PipelineState;

  using StridePageTable = Stride<int, _1>;

  static constexpr int TmeTransactionBytesQ = bits_to_bytes(size(SmemLayoutQ{}) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesK = bits_to_bytes(size(take<0,2>(SmemLayoutK{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesV = bits_to_bytes(size(take<0,2>(SmemLayoutV{})) * sizeof_bits_v<Element>);

  struct SharedStorage {
    mute::array_aligned<Element, cosize_v<SmemLayoutQ>, 256> smem_q;
    mute::array_aligned<Element, cosize_v<SmemLayoutK>, 256> smem_k;
    mute::array_aligned<Element, cosize_v<SmemLayoutP>, 256> smem_p;
    mute::array_aligned<Element, cosize_v<SmemLayoutV>, 256> smem_v;
  };

  struct Arguments {
    Element const* ptr_Q;
    StrideQ stride_Q;
    Element const* ptr_K;
    StrideK stride_K;
    Element const* ptr_V;
    StrideV stride_V;

    int const* ptr_page_table = nullptr;
    StridePageTable stride_page_table;
    int const* ptr_seqlen = nullptr;

    float const sm_scale;

    int page_size;
    int page_count;

    typename Fusion::Arguments fusion;
  };

  struct Params {
    TME_Q tme_Q;
    TME_K tme_K;
    TME_V tme_V;

    int const* ptr_page_table = nullptr;
    StridePageTable stride_page_table;
    int const* ptr_seqlen = nullptr;

    float const sm_scale;
    float const sm_scale_log2;

    int const page_size;
    int const page_count;

    FastDivmod fast_divmod_hr;

    typename Fusion::Params fusion;
  };

  template <class ProblemSize>
  static Params
  to_underlying_arguments(ProblemSize problem_size, Arguments const& args, void* workspace = nullptr) {
    float const log2e = std::log2(std::exp(1.0f));

    auto [Q, K, D_QK, D_VO, H, H_K, B] = problem_size;

    int PageSize  = args.page_size;
    int PageCount = args.page_count;

    auto problem_size_q = make_shape(Q, D_QK, make_shape(H, B));
    auto tme_Q = make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
        make_gmem_ptr(args.ptr_Q),
        problem_size_q,
        args.stride_Q
      ),
      SmemLayoutQFull{},
      select<0,2>(TileShape{}));

    auto problem_size_k = make_shape(PageSize, D_QK, make_shape(H_K, PageCount));
    auto tme_K = TmeLoadKeyBuilder::make_tme_copy(
      make_tensor(
        make_gmem_ptr(args.ptr_K),
        problem_size_k,
        args.stride_K
      ));

    auto problem_shape_pv = make_shape(Q, D_VO, PageSize, make_shape(H_K, PageCount));
    auto params_pv = CollectiveMmaPV::to_underlying_arguments(
        problem_shape_pv,
        typename CollectiveMmaPV::Arguments {
          args.ptr_K, args.stride_K,
          args.ptr_V, select<1, 0, 2>(args.stride_V)
          }, nullptr);

    int H_R = H / H_K;
    mutlass::FastDivmod fast_divmod_hr(H_R);

    return {
      tme_Q,
      tme_K,
      params_pv.tme_load_b,
      args.ptr_page_table,
      args.stride_page_table,
      args.ptr_seqlen,
      args.sm_scale,
      args.sm_scale * log2e,
      args.page_size,
      args.page_count,
      fast_divmod_hr,
      args.fusion
    };
  }

  template <
    class BlkCoord,
    class ProblemSize
  >
  MUTE_DEVICE
  void load(Params const& params, SharedStorage& shared_storage,
            MainloopPipelineQ& pipeline_q, PipelineQState& pipe_state_q,
            MainloopPipelineK& pipeline_k, PipelineKState& pipe_state_k,
            MainloopPipelineV& pipeline_v, PipelineVState& pipe_state_v,
            BlkCoord const& blk_coord, ProblemSize const& problem_size,
            int const warp_idx) {
    enum class ProducerWarpRole {
      LoadQKV,
      Warp1,
      Warp2,
      Warp3,
    };
    auto producer_warp_role = ProducerWarpRole(warp_idx);

    int kv_tile_iter = 0;

    if (producer_warp_role == ProducerWarpRole::LoadQKV) {
      auto [Q, K_, D_QK, D_VO, H, H_K, B] = problem_size;
      int PageSize = params.page_size;
      int PageCount = params.page_count;
      int K = params.ptr_seqlen[get<3>(blk_coord)];
      int H_R = params.fast_divmod_hr.divisor;

      // Init Q
      // TODO: support PackGQA, SIMT or TME
      Tensor mQ = params.tme_Q.get_tme_tensor(make_shape(Q, D_QK, make_shape(H, B)));
      Tensor gQ_full = local_tile(mQ, select<0, 2>(TileShape{}), make_coord(_,_,_));
      Tensor gQ = gQ_full(_, _, get<0>(blk_coord), _0{}, make_coord(get<1>(blk_coord), get<3>(blk_coord)));
      Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQFull{});
      auto cta_tme_q = params.tme_Q.get_slice(0);
      Tensor tQgQ = cta_tme_q.partition_S(gQ);
      Tensor tQsQ = cta_tme_q.partition_D(sQ);

      // Init K
      auto problem_size_for_K = TmeLoadKeyBuilder::get_problem_size(
          make_shape(Q, PageSize, D_QK, D_VO, H, H_K, PageCount));
      Tensor mK = params.tme_K.get_tme_tensor(make_shape(
            get<1>(problem_size_for_K), D_QK, make_shape(H_K, PageCount)));
      Tensor gK_full = local_tile(mK, TmeTileShapeKD{}, make_coord(_,_,_));
      // TODO: adapt if PageSize larger than BN
      Tensor gK = gK_full(_,_,_0{},_0{}, make_coord(get<2>(blk_coord), _));
      Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename TmeLoadKeyBuilder::SmemLayoutK{});
      auto cta_tme_k = params.tme_K.get_slice(0);
      Tensor tKgK = cta_tme_k.partition_S(gK);
      Tensor tKsK = cta_tme_k.partition_D(sK);

      // Init V
      Tensor mV = params.tme_V.get_tme_tensor(make_shape(D_VO, PageSize, make_shape(H_K, PageCount)));
      Tensor gV_full = local_tile(mV, select<1, 2>(TileShapePDV{}), make_coord(_,_,_));
      // TODO: adapt if PageSize larger than BN
      Tensor gV = gV_full(_,_,_0{},_0{}, make_coord(get<2>(blk_coord), _));
      Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
      auto cta_tme_v = params.tme_V.get_slice(0);
      Tensor tVgV = cta_tme_v.partition_S(gV);
      Tensor tVsV = cta_tme_v.partition_D(sV);

      // PageTable
      Tensor mPT = make_tensor(make_gmem_ptr(params.ptr_page_table), make_shape(B, PageCount), params.stride_page_table);
      Tensor gPT = mPT(get<3>(blk_coord), _);


      Fusion fusion {params.fusion};

      auto problem_size_for_fusion = make_shape(Q, K, D_QK, D_VO, H, H_K, B);
      int fusion_tile_count = fusion.get_trip_count(blk_coord, TileShape{}, problem_size_for_fusion);
      int page_index = 0;

      {
        // Load Q
        pipeline_q.producer_acquire(pipe_state_q);
        uint32_t bar_id = pipeline_q.producer_get_barrier_id(pipe_state_q);

        copy(params.tme_Q.with(bar_id), tQgQ, tQsQ);
        ++pipe_state_q;
      }

      while (fusion_tile_count > 0)
      {
        int cur_page = gPT(page_index);

        // load K
        {
          pipeline_k.producer_acquire(pipe_state_k);
          uint32_t bar_id = pipeline_k.producer_get_barrier_id(pipe_state_k);

          copy(params.tme_K.with(bar_id), tKgK(_,_,_,cur_page), tKsK(_,_,_,pipe_state_k.index()));
          ++pipe_state_k;
        }

        // load V
        {
          pipeline_v.producer_acquire(pipe_state_v);
          uint32_t bar_id = pipeline_v.producer_get_barrier_id(pipe_state_v);

          copy(params.tme_V.with(bar_id), tVgV(_,_,_,cur_page), tVsV(_,_,_,pipe_state_v.index()));
          ++pipe_state_v;
        }
        --fusion_tile_count;
        ++page_index;
      }
    }
  }


  template <
    class BlkCoord,
    class ProblemSize,
    class MathOrderBarrier
  >
  MUTLASS_DEVICE auto
  compute(
      BlkCoord const& blk_coord, Params const& params, ProblemSize const& problem_size,
      MainloopPipelineQ& pipeline_q, PipelineQState& smem_pipe_q_read,
      MainloopPipelineK& pipeline_k, PipelineKState& smem_pipe_k_read,
      MainloopPipelineV& pipeline_v, PipelineVState& smem_pipe_v_read,
      MathOrderBarrier& math_order_barrier,
      SharedStorage& storage, int const thread_idx, int const consumer_idx) {

    // Prepare QK MMA
    TiledMmaQK tiled_mma_qk;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});

    Tensor tSsQ = thr_mma_qk.partition_A(sQ(_,_,consumer_idx));
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

    Fusion fusion {params.fusion};

    int kv_tile_count = fusion.get_unmasked_trip_count(blk_coord, TileShape{}, problem_size);

    Tensor acc_pv = partition_fragment_C(tiled_mma_pv, take<0,2>(TileShapePDV{}));

    // trigger zero init sqmma
    clear(acc_pv);

    CollectiveSoftmax<Fusion, decltype(params), false> softmax(params, fusion);
    auto softmax_state = softmax.init(acc_pv, tiled_mma_pv);

    auto warpsquad_lock = [&] () {
      if constexpr (BarrierSchedule) {
        math_order_barrier.wait();
      }
    };

    auto warpsquad_unlock = [&] () {
      if constexpr (BarrierSchedule) {
        math_order_barrier.arrive();
      }
    };

    {
      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
      // trigger zero init sqmma
      clear(acc_qk);

      pipeline_q.consumer_wait(smem_pipe_q_read);
      pipeline_k.consumer_wait(smem_pipe_k_read);

      // MMA QK
      mute::gemm(tiled_mma_qk, tSrQ, tSrK(_,_,_,smem_pipe_k_read.index()), acc_qk);

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

      warpsquad_lock();
      // MMA QK
      mute::gemm(tiled_mma_qk, tSrQ, tSrK(_,_,_,smem_pipe_k_read.index()), acc_qk);

      warpsquad_unlock();

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
