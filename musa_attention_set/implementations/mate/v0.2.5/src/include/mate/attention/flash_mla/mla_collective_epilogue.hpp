#pragma once

#include <mute/tensor.hpp>

namespace mate::flash_mla {

using namespace mute;
using namespace mutlass;
using namespace mutlass::fmha::collective;

template <class Element,
          class ElementAccumulator,
          class EpilogueTileShape,
          class StrideO,
          class StrideLSE,
          bool VarlenQ,
          int  FragmentSize = 4>
struct MlaFwdEpilogue {
  struct Arguments {
    Element* ptr_O;
    StrideO  stride_O;

    ElementAccumulator* ptr_LSE;
    StrideLSE           stride_LSE;

    ElementAccumulator* ptr_O_accum;
    ElementAccumulator* ptr_LSE_accum;
    int                 q_seq_per_hk;
  };

  using Params = Arguments;

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize problem_size, Arguments const& args, void* workspace = nullptr) {
    return args;
  }

  template <bool IsNoSplit, class BlkCoord, class ResultTuple, class TiledMma, class ProblemSize, class BlockOffset>
  MUTLASS_DEVICE void operator()(BlkCoord const&    blk_coord,
                                 ResultTuple const& result,
                                 TiledMma const&    tiled_mma,
                                 ProblemSize const& problem_size,
                                 BlockOffset const& blk_offset,
                                 Params const&      params,
                                 int const          thread_idx,
                                 int const          consumer_qo_coord,
                                 int const          split_idx,
                                 int const          seq_q_begin) {
    auto acc = get<0>(result);
    auto lse = get<1>(result);

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    if constexpr (IsNoSplit) {
      auto [Q_, D_VO_, HQ_, HK_, B_, TotalQ_, MaxQ_] = problem_size;

      int Q      = Q_;
      int D_VO   = D_VO_;
      int HQ     = HQ_;
      int HK     = HK_;
      int B      = B_;
      int TotalQ = TotalQ_;
      int MaxQ   = MaxQ_;
      int h_r    = HQ / HK;

      auto mLSE_in = [&] {
        if constexpr (!VarlenQ) {
          return make_tensor(make_gmem_ptr(params.ptr_LSE),
                             make_shape(Q, D_VO, make_shape(HK, B)),
                             make_stride(_1{}, _0{}, get<1>(params.stride_LSE)));
        } else {  // domain offset for varlen Q; Q is seqlen_q for current batch.
          return domain_offset(make_coord(seq_q_begin, _0{}, make_coord(_0{}, _0{})),
                               make_tensor(make_gmem_ptr(params.ptr_LSE),
                                           make_shape(Q, D_VO, make_shape(HK, 1)),
                                           make_stride(_1{}, _0{}, get<1>(params.stride_LSE))));
        }
      }();
      Tensor mLSE      = domain_offset(make_coord(get<0>(blk_offset), _0{}, make_coord(_0{}, _0{})), mLSE_in);
      Tensor gLSE_full = local_tile(mLSE, EpilogueTileShape{}, make_coord(_, _, _));
      Tensor gLSE      = gLSE_full(_, _, consumer_qo_coord, _0{}, make_coord(get<1>(blk_coord), get<3>(blk_coord)));
      Tensor tOgLSE    = thr_mma.partition_C(gLSE);

      Tensor cO   = make_identity_tensor(EpilogueTileShape{});
      Tensor tOcO = thr_mma.partition_C(cO);

      auto mO_in = [&] {
        if constexpr (!VarlenQ) {
          return make_tensor(make_gmem_ptr(params.ptr_O), make_shape(Q, D_VO, make_shape(HK, B)), params.stride_O);
        } else {  // domain offset for varlen Q; Q is seqlen_q for current batch.
          return domain_offset(
              make_coord(seq_q_begin * HQ, _0{}, make_coord(_0{}, _0{})),
              make_tensor(make_gmem_ptr(params.ptr_O), make_shape(Q, D_VO, make_shape(HK, 1)), params.stride_O));
        }
      }();
      Tensor mO      = domain_offset(make_coord(get<0>(blk_offset), _0{}, make_coord(_0{}, _0{})), mO_in);
      Tensor gO_full = local_tile(mO, EpilogueTileShape{}, make_coord(_, _, _));
      Tensor gO      = gO_full(_, _, consumer_qo_coord, _0{}, make_coord(get<1>(blk_coord), get<3>(blk_coord)));
      Tensor tOgO    = thr_mma.partition_C(gO);

      Tensor tOgLSE_mn = make_tensor(tOgLSE.data(), layout_acc_mn(tiled_mma, tOgLSE.layout()));
      Tensor tOcO_mn   = make_tensor(tOcO.data(), layout_acc_mn(tiled_mma, tOcO.layout()));
      Tensor tOgO_mn   = make_tensor(tOgO.data(), layout_acc_mn(tiled_mma, tOgO.layout()));
      Tensor acc_mn    = make_tensor(acc.data(), layout_acc_mn(tiled_mma, acc.layout()));

      Tensor acc_cvt    = make_fragment_like<Element>(acc);
      Tensor tAcc_frg   = recast<mutlass::Array<ElementAccumulator, FragmentSize>>(acc);
      Tensor tCvt_frg   = recast<mutlass::Array<Element, FragmentSize>>(acc_cvt);
      Tensor acc_cvt_mn = make_tensor(acc_cvt.data(), layout_acc_mn(tiled_mma, acc_cvt.layout()));

      MUTE_UNROLL
      for (int i = 0; i < size(tCvt_frg); ++i) {
        tCvt_frg(i) = mutlass::
            NumericArrayConverter<Element, ElementAccumulator, FragmentSize, FloatRoundStyle::round_to_nearest>{}(
                tAcc_frg(i));
      }

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<0>(tOgO_mn); ++i) {
        int row = consumer_qo_coord * get<0>(EpilogueTileShape{}) + get<0>(tOcO_mn(i, 0));
        if (row < get<0>(problem_size)) {
          if constexpr (!VarlenQ) {
            tOgLSE_mn(i, _0{}) = lse(i);
          } else {
            int h_idx = row % h_r;
            int q_idx = row / h_r;

            mLSE((h_idx)*TotalQ + q_idx, _0{}, make_coord(_0{}, _0{})) = lse(i);
          }

          MUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<1>(tOgO_mn); ++j) {
            tOgO_mn(i, j) = mutlass::NumericConverter<Element, ElementAccumulator, FloatRoundStyle::round_to_nearest>{}(
                acc_mn(i, j));
          }
        }
      }
    } else {
      int const h_k        = 1;
      int const k_head_idx = 0;
      int const batch_idx  = get<3>(blk_coord);

      ElementAccumulator* oaccum_ptr = params.ptr_O_accum + ((split_idx * h_k + k_head_idx) * params.q_seq_per_hk +
                                                             consumer_qo_coord * get<0>(EpilogueTileShape{})) *
                                                                get<1>(EpilogueTileShape{});
      ElementAccumulator* softmax_lseaccum_ptr = params.ptr_LSE_accum +
                                                 (split_idx * h_k + k_head_idx) * params.q_seq_per_hk +
                                                 consumer_qo_coord * get<0>(EpilogueTileShape{});

      Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr),
                                   Layout<EpilogueTileShape, Stride<tuple_element_t<1, EpilogueTileShape>, _1>>{});
      Tensor gSoftmaxAccum =
          make_tensor(make_gmem_ptr(softmax_lseaccum_ptr), Layout<EpilogueTileShape, Stride<_1, _0>>{});

      Tensor cO     = make_identity_tensor(EpilogueTileShape{});
      Tensor tOcO   = thr_mma.partition_C(cO);
      Tensor tOgO   = thr_mma.partition_C(gOAccum);
      Tensor tOgLSE = thr_mma.partition_C(gSoftmaxAccum);

      Tensor tOcO_mn   = make_tensor(tOcO.data(), layout_acc_mn(tiled_mma, tOcO.layout()));
      Tensor tOgO_mn   = make_tensor(tOgO.data(), layout_acc_mn(tiled_mma, tOgO.layout()));
      Tensor tOgLSE_mn = make_tensor(tOgLSE.data(), layout_acc_mn(tiled_mma, tOgLSE.layout()));
      Tensor acc_mn    = make_tensor(acc.data(), layout_acc_mn(tiled_mma, acc.layout()));

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<0>(tOgO_mn); ++i) {
        if (consumer_qo_coord * get<0>(EpilogueTileShape{}) + get<0>(tOcO_mn(i, 0)) < get<0>(problem_size)) {
          tOgLSE_mn(i, _0{}) = lse(i);

          MUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<1>(tOgO_mn); ++j) {
            tOgO_mn(i, j) = acc_mn(i, j);
          }
        }
      }
    }
  }
};

}  // namespace mate::flash_mla
