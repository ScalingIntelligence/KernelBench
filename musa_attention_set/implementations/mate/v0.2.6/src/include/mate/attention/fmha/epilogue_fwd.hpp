#pragma once

#include "seqlen.hpp"

namespace mate::attention::fmha {

using namespace mute;

template <class Element_,
          class TileShapePDV_,
          bool HasCuseqlensQ_,
          bool HasSequsedQ_,
          bool IsPackGQA_,
          int  NumEpilogueThreads_,
          bool IsEvenHeadDim_,
          int  HeadRatio_>
struct CollectiveEpilogueFwd {
  using Element        = Element_;
  using ElementPartial = float;

  using TileShapePDV = TileShapePDV_;

  static constexpr int TileM     = get<0>(TileShapePDV{});
  static constexpr int HeadDimVO = get<1>(TileShapePDV{});

  static constexpr int  NumEpilogueThreads = NumEpilogueThreads_;
  static constexpr bool HasCuseqlensQ      = HasCuseqlensQ_;
  static constexpr bool HasSequsedQ        = HasSequsedQ_;
  static constexpr bool IsPackGQA          = IsPackGQA_;
  static constexpr bool IsEvenHeadDim      = IsEvenHeadDim_;
  static constexpr int  HeadRatio          = HeadRatio_;

  static_assert(sizeof(Element) <= 2);

  static constexpr int FragmentSize = 4;

  // (seqlen_q, d, head, batch, num_splits)
  using ShapeO  = Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;
  using StrideO = Stride<int64_t, _1, int64_t, int64_t, int64_t>;

  // ((qhead_per_khead, seqlen_q), d, nheads_k, batch, num_splits)
  using ShapeOPacked_  = Shape<Shape<int32_t, int32_t>, int32_t, int32_t, int32_t, int32_t>;
  using StrideOPacked_ = Stride<Stride<int64_t, int64_t>, _1, int64_t, int64_t, int64_t>;

  using ShapeOPacked  = std::conditional_t<!IsPackGQA, ShapeO, ShapeOPacked_>;
  using StrideOPacked = std::conditional_t<!IsPackGQA, StrideO, StrideOPacked_>;

  // (seqlen_q, head, batch, num_splits)
  using ShapeLSE  = Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideLSE = Stride<_1, int64_t, int64_t, int64_t>;

  // ((qhead_per_khead, seqlen_q), nheads_k, batch, num_splits)
  using ShapeLSEPacked_  = Shape<Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>;
  using StrideLSEPacked_ = Stride<Stride<int64_t, _1>, int64_t, int64_t, int64_t>;

  using ShapeLSEPacked  = std::conditional_t<!IsPackGQA, ShapeLSE, ShapeLSEPacked_>;
  using StrideLSEPacked = std::conditional_t<!IsPackGQA, StrideLSE, StrideLSEPacked_>;

  using PackGQAManager = PackGQAManager<Element, HeadRatio, TileM, HeadDimVO, NumEpilogueThreads>;

  struct Arguments {
    Element*        ptr_O;
    ShapeO const    shape_O;
    StrideO const   stride_O;
    ElementPartial* ptr_O_partial;
    StrideO const   stride_O_partial;
    float*          ptr_LSE;
    StrideLSE const stride_LSE;
    float*          ptr_LSE_partial;
    StrideLSE const stride_LSE_partial;

    int32_t const   nheads_kv;
    uint32_t const* cu_seqlens = nullptr;
    uint32_t const* seqused    = nullptr;
  };

  struct Params {
    Element*            ptr_O;
    ShapeO const        shape_O;
    StrideO const       stride_O;
    ShapeOPacked const  shape_O_packed;
    StrideOPacked const stride_O_packed;
    ElementPartial*     ptr_O_partial;
    StrideO const       stride_O_partial;
    StrideOPacked const stride_O_partial_packed;

    float*                ptr_LSE;
    StrideLSE const       stride_LSE;
    ShapeLSEPacked const  shape_LSE_packed;
    StrideLSEPacked const stride_LSE_packed;
    float*                ptr_LSE_partial;
    StrideLSE const       stride_LSE_partial;
    StrideLSEPacked const stride_LSE_partial_packed;

    uint32_t const* cu_seqlens = nullptr;
    uint32_t const* seqused    = nullptr;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    int const  qhead_per_khead = !IsPackGQA ? 1 : HeadRatio;
    auto const shape_O_packed_ = make_shape(make_shape(qhead_per_khead, get<0>(args.shape_O)),
                                            get<1>(args.shape_O),
                                            args.nheads_kv,
                                            get<3>(args.shape_O),
                                            get<4>(args.shape_O));
    auto const shape_O_packed  = mute::conditional_return<!IsPackGQA>(args.shape_O, shape_O_packed_);

    auto const stride_O_packed_ = make_stride(make_stride(get<2>(args.stride_O), get<0>(args.stride_O)),
                                              get<1>(args.stride_O),
                                              get<2>(args.stride_O) * qhead_per_khead,
                                              get<3>(args.stride_O),
                                              get<4>(args.stride_O));
    auto const stride_O_packed  = mute::conditional_return<!IsPackGQA>(args.stride_O, stride_O_packed_);

    auto const stride_O_partial_packed_ =
        make_stride(make_stride(get<2>(args.stride_O_partial), get<0>(args.stride_O_partial)),
                    get<1>(args.stride_O_partial),
                    get<2>(args.stride_O_partial) * qhead_per_khead,
                    get<3>(args.stride_O_partial),
                    get<4>(args.stride_O_partial));
    auto const stride_O_partial_packed =
        mute::conditional_return<!IsPackGQA>(args.stride_O_partial, stride_O_partial_packed_);

    auto const shape_LSE_packed_ = make_shape(
        make_shape(qhead_per_khead, get<0>(args.shape_O)), args.nheads_kv, get<3>(args.shape_O), get<4>(args.shape_O));
    auto const shape_LSE_packed =
        mute::conditional_return<!IsPackGQA>(select<0, 2, 3, 4>(args.shape_O), shape_LSE_packed_);

    auto const stride_LSE_packed_ = make_stride(make_stride(get<1>(args.stride_LSE), get<0>(args.stride_LSE)),
                                                get<1>(args.stride_LSE) * qhead_per_khead,
                                                get<2>(args.stride_LSE),
                                                get<3>(args.stride_LSE));
    auto const stride_LSE_packed  = mute::conditional_return<!IsPackGQA>(args.stride_LSE, stride_LSE_packed_);

    auto const stride_LSE_partial_packed_ =
        make_stride(make_stride(get<1>(args.stride_LSE_partial), get<0>(args.stride_LSE_partial)),
                    get<1>(args.stride_LSE_partial) * qhead_per_khead,
                    get<2>(args.stride_LSE_partial),
                    get<3>(args.stride_LSE_partial));
    auto const stride_LSE_partial_packed =
        mute::conditional_return<!IsPackGQA>(args.stride_LSE_partial, stride_LSE_partial_packed_);

    return Params{.ptr_O                   = args.ptr_O,
                  .shape_O                 = args.shape_O,
                  .stride_O                = args.stride_O,
                  .shape_O_packed          = shape_O_packed,
                  .stride_O_packed         = stride_O_packed,
                  .ptr_O_partial           = args.ptr_O_partial,
                  .stride_O_partial        = args.stride_O_partial,
                  .stride_O_partial_packed = stride_O_partial_packed,

                  .ptr_LSE                   = args.ptr_LSE,
                  .stride_LSE                = args.stride_LSE,
                  .shape_LSE_packed          = shape_LSE_packed,
                  .stride_LSE_packed         = stride_LSE_packed,
                  .ptr_LSE_partial           = args.ptr_LSE_partial,
                  .stride_LSE_partial        = args.stride_LSE_partial,
                  .stride_LSE_partial_packed = stride_LSE_partial_packed,

                  .cu_seqlens = args.cu_seqlens,
                  .seqused    = args.seqused};
  }

  template <bool StoreZero,
            bool SplitKV,
            class FrgTensorO,
            class FrgTensorLSE,
            class TiledMma,
            class SeqlenInfo,
            class BlockCoord>
  MUTLASS_DEVICE void store(Params const&       params,
                            FrgTensorO const&   acc_pv,
                            FrgTensorLSE const& lse,
                            TiledMma            tiled_mma,
                            int                 thread_idx,
                            SeqlenInfo const&   seqlen_info,
                            BlockCoord const&   block_coord) {
    static_assert(rank(BlockCoord{}) == 4);
    int m_block   = get<0>(block_coord);
    int bidh      = get<1>(block_coord);
    int bidb      = get<2>(block_coord);
    int split_idx = get<3>(block_coord);

    ThrMMA thr_mma = tiled_mma.get_thread_slice(thread_idx);

    int const offset_o = seqlen_info.offset_q;
    int const seqlen_o = seqlen_info.seqlen_q;

    auto mLSE = [&] {
      if constexpr (!SplitKV) {
        return make_tensor(make_gmem_ptr(params.ptr_LSE + offset_o * get<0>(params.stride_LSE)),
                           params.shape_LSE_packed,
                           params.stride_LSE_packed)(_, bidh, HasCuseqlensQ ? 0 : bidb, _0{});
      } else {
        return make_tensor(make_gmem_ptr(params.ptr_LSE_partial + offset_o * get<0>(params.stride_LSE_partial)),
                           params.shape_LSE_packed,
                           params.stride_LSE_partial_packed)(_, bidh, HasCuseqlensQ ? 0 : bidb, split_idx);
      }
    }();

    auto mO = [&] {
      if constexpr (!SplitKV) {
        return make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)),
                           params.shape_O_packed,
                           params.stride_O_packed)(_, _, bidh, HasCuseqlensQ ? 0 : bidb, _0{});
      } else {
        return make_tensor(make_gmem_ptr(params.ptr_O_partial + offset_o * get<0>(params.stride_O_partial)),
                           params.shape_O_packed,
                           params.stride_O_partial_packed)(_, _, bidh, HasCuseqlensQ ? 0 : bidb, split_idx);
      }
    }();

    Tensor gO = local_tile(mO, take<0, 2>(TileShapePDV{}), make_coord(m_block, _0{}));
    Tensor cO = make_identity_tensor(take<0, 2>(TileShapePDV{}));

    Tensor tOcO = thr_mma.partition_C(cO);
    Tensor tOgO = thr_mma.partition_C(gO);

    Tensor tOcO_mn = make_tensor(tOcO.data(), layout_acc_mn(tiled_mma, tOcO.layout()));
    Tensor tOgO_mn = make_tensor(tOgO.data(), layout_acc_mn(tiled_mma, tOgO.layout()));
    Tensor tOrO_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma, acc_pv.layout()));

    auto assign = [&](auto src, auto zero_val) {
      if constexpr (StoreZero) {
        return decltype(src)(zero_val);
      } else {
        return src;
      }
    };

    if constexpr (!IsPackGQA) {
      // Write LSE from rmem -> gmem
      MUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size(lse); ++mi) {
        int const row = m_block * TileM + get<0>(tOcO_mn(mi, _0{}));
        if (get<1>(tOcO_mn(_0{}, _0{})) == 0 && row < seqlen_o) {
          mLSE(row) = assign(lse(mi), -std::numeric_limits<float>::infinity());
          // mLSE(row) = lse(mi);
        }
      }
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<0>(tOgO_mn); ++i) {
        if (m_block * get<0>(TileShapePDV{}) + get<0>(tOcO_mn(i, _0{})) < seqlen_o) {
          if constexpr (IsEvenHeadDim) {
            MUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<1>(tOgO_mn); ++j) {
              tOgO_mn(i, j) = [&] {
                if constexpr (!SplitKV) {
                  return assign(Element(tOrO_mn(i, j)), 0);
                } else {
                  return assign(ElementPartial(tOrO_mn(i, j)), 0);
                }
              }();
            }
          } else {
            MUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<1>(tOgO_mn); ++j) {
              if (get<1>(tOcO_mn(_0{}, j)) < get<1>(params.shape_O)) {
                tOgO_mn(i, j) = [&] {
                  if constexpr (!SplitKV) {
                    return assign(Element(tOrO_mn(i, j)), 0);
                  } else {
                    return assign(ElementPartial(tOrO_mn(i, j)), 0);
                  }
                }();
              }
            }
          }
        }
      }
    } else {
      PackGQAManager::template store_LSE<StoreZero>(tiled_mma, lse, mLSE, thread_idx, m_block, seqlen_o);
      PackGQAManager::template store_O<StoreZero>(tiled_mma, tOrO_mn, mO, thread_idx, m_block, seqlen_o);
    }
  }
};

}  // namespace mate::attention::fmha
