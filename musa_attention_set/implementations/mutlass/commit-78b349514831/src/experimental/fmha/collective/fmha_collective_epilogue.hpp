#pragma once

#include <mute/tensor.hpp>

#include <mutlass/numeric_conversion.h>

#include "collective/fmha_common.hpp"

namespace mutlass::fmha::collective {

using namespace mute;

template <
  class Element,
  class ElementAccumulator,
  class EpilogueTileShape,
  class StrideO,
  class StrideLSE,
  int FragmentSize = 4
>
struct FmhaFwdEpilogue {

  struct Arguments {
    Element* ptr_O;
    StrideO stride_O;

    ElementAccumulator* ptr_LSE;
    StrideLSE stride_LSE;
    int const splits_kv;
  };

  using Params = Arguments;

  template <class ProblemSize>
  static Params
  to_underlying_arguments(ProblemSize problem_size, Arguments const& args, void* workspace = nullptr) {
    return args;
  }

  template <class BlkCoord, class ResultTuple, class TiledMma, class ProblemSize, class BlockOffset>
  MUTLASS_DEVICE
  void operator()(
    BlkCoord const& blk_coord, ResultTuple const& result,
    TiledMma const& tiled_mma, ProblemSize const& problem_size,
    BlockOffset const& blk_offset, Params const& params,
    int const thread_idx, int const consumer_qo_coord)
  {
    auto acc = get<0>(result);
    auto lse = get<1>(result);

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);


    auto [Q, D_VO, H, B] = problem_size;

    Tensor mLSE_in = make_tensor(make_gmem_ptr(params.ptr_LSE),
      make_shape(Q, D_VO, make_shape(H, B)),
      make_stride(_1{}, _0{}, get<1>(params.stride_LSE)));
    Tensor mLSE = domain_offset(make_coord(get<0>(blk_offset), _0{}, make_coord(_0{}, _0{})), mLSE_in);
    Tensor gLSE_full = local_tile(mLSE, EpilogueTileShape{}, make_coord(_, _, _));
    Tensor gLSE = gLSE_full(_, _, consumer_qo_coord, _0{}, make_coord(get<1>(blk_coord), get<3>(blk_coord)));
    Tensor tOgLSE = thr_mma.partition_C(gLSE);

    Tensor cO = make_identity_tensor(EpilogueTileShape{});
    Tensor tOcO = thr_mma.partition_C(cO);

    Tensor mO_in = make_tensor(make_gmem_ptr(params.ptr_O),
                               make_shape(Q, D_VO, make_shape(H, B)), params.stride_O);
    Tensor mO = domain_offset(make_coord(get<0>(blk_offset), _0{}, make_coord(_0{}, _0{})), mO_in);
    Tensor gO_full = local_tile(mO, EpilogueTileShape{}, make_coord(_, _, _));
    Tensor gO = gO_full(_, _, consumer_qo_coord, _0{}, make_coord(get<1>(blk_coord), get<3>(blk_coord)));
    Tensor tOgO = thr_mma.partition_C(gO);

    Tensor tOgLSE_mn = make_tensor(tOgLSE.data(), layout_acc_mn(tiled_mma, tOgLSE.layout()));
    Tensor tOcO_mn = make_tensor(tOcO.data(), layout_acc_mn(tiled_mma, tOcO.layout()));
    Tensor tOgO_mn = make_tensor(tOgO.data(), layout_acc_mn(tiled_mma, tOgO.layout()));
    Tensor acc_mn = make_tensor(acc.data(), layout_acc_mn(tiled_mma, acc.layout()));

    Tensor acc_cvt = make_fragment_like<Element>(acc);
    Tensor tAcc_frg = recast<mutlass::Array<ElementAccumulator, FragmentSize>>(acc);
    Tensor tCvt_frg = recast<mutlass::Array<Element, FragmentSize>>(acc_cvt);
    Tensor acc_cvt_mn = make_tensor(acc_cvt.data(), layout_acc_mn(tiled_mma, acc_cvt.layout()));

    MUTE_UNROLL
    for (int i = 0; i < size(tCvt_frg); ++i) {
      tCvt_frg(i) = mutlass::NumericArrayConverter<Element, ElementAccumulator, FragmentSize, FloatRoundStyle::round_to_nearest>{}(tAcc_frg(i));
    }


    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(tOgO_mn); ++i) {
      if (consumer_qo_coord * get<0>(EpilogueTileShape{}) + get<0>(tOcO_mn(i, 0)) < get<0>(problem_size)) {
        tOgLSE_mn(i, _0{}) = lse(i);

        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<1>(tOgO_mn); ++j) {
          tOgO_mn(i, j) = acc_cvt_mn(i, j);
        }
      }
    }
  }
};

} // namespace mutlass::fmha::collective
