#pragma once

#include <mute/tensor.hpp>

#include "utils.hpp"

namespace mate::attention::fmha {

using namespace mute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Fragment, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
MUTLASS_DEVICE void apply_rotary_interleaved(Tensor<Engine1, Layout1>       &rK,
                                             Tensor<Engine2, Layout2> const &rCos,
                                             Tensor<Engine2, Layout2> const &rSin) {
  MUTE_STATIC_ASSERT_V(rank(rK) == _1{});
  MUTE_STATIC_ASSERT_V(rank(rCos) == _1{});
  MUTE_STATIC_ASSERT_V(rank(rSin) == _1{});
  MUTE_STATIC_ASSERT_V(size<0>(rCos) == size<0>(rSin));
  static_assert(decltype(size<0>(rK))::value == decltype(size<0>(rCos))::value * 2);
  static_assert(decltype(size<0>(rCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
  Tensor K_fp32 = make_tensor_like<float>(rK);
  convert_type(rK, K_fp32);
  Tensor cos_fp32 = make_tensor_like<float>(rCos);
  convert_type(rCos, cos_fp32);
  Tensor sin_fp32 = make_tensor_like<float>(rSin);
  convert_type(rSin, sin_fp32);

  MUTE_UNROLL
  for (int i = 0; i < size<0>(K_fp32) / 2; ++i) {
    float real        = K_fp32[2 * i] * cos_fp32[i] - K_fp32[2 * i + 1] * sin_fp32[i];
    float imag        = K_fp32[2 * i] * sin_fp32[i] + K_fp32[2 * i + 1] * cos_fp32[i];
    K_fp32[2 * i]     = real;
    K_fp32[2 * i + 1] = imag;
  }
  convert_type(K_fp32, rK);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Fragment, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
MUTLASS_DEVICE void apply_rotary_contiguous(Tensor<Engine1, Layout1>       &rK_left,
                                            Tensor<Engine1, Layout1>       &rK_right,
                                            Tensor<Engine2, Layout2> const &rCos,
                                            Tensor<Engine2, Layout2> const &rSin) {
  MUTE_STATIC_ASSERT_V(rank(rK_left) == _1{});
  MUTE_STATIC_ASSERT_V(rank(rK_right) == _1{});
  MUTE_STATIC_ASSERT_V(rank(rCos) == _1{});
  MUTE_STATIC_ASSERT_V(rank(rSin) == _1{});
  MUTE_STATIC_ASSERT_V(size<0>(rK_left) == size<0>(rK_right));
  MUTE_STATIC_ASSERT_V(size<0>(rK_left) == size<0>(rCos));
  MUTE_STATIC_ASSERT_V(size<0>(rCos) == size<0>(rSin));
  static_assert(decltype(size<0>(rCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
  Tensor K_left_fp32 = make_tensor_like<float>(rK_left);
  convert_type(rK_left, K_left_fp32);
  Tensor K_right_fp32 = make_tensor_like<float>(rK_right);
  convert_type(rK_right, K_right_fp32);
  Tensor cos_fp32 = make_tensor_like<float>(rCos);
  convert_type(rCos, cos_fp32);
  Tensor sin_fp32 = make_tensor_like<float>(rSin);
  convert_type(rSin, sin_fp32);

  MUTE_UNROLL
  for (int i = 0; i < size<0>(K_left_fp32); ++i) {
    float real      = K_left_fp32[i] * cos_fp32[i] - K_right_fp32[i] * sin_fp32[i];
    float imag      = K_left_fp32[i] * sin_fp32[i] + K_right_fp32[i] * cos_fp32[i];
    K_left_fp32[i]  = real;
    K_right_fp32[i] = imag;
  }
  convert_type(K_left_fp32, rK_left);
  convert_type(K_right_fp32, rK_right);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int TileMN,
          int HeadDim,
          int NumThreads,
          typename Element,
          int  Fragment,
          bool FixedPosition = false,
          int  HeadRatio     = 1>
struct Rotary {
  static constexpr int GmemElemsPerLoad = sizeof(mute::uint128_t) / sizeof(Element);
  static_assert(HeadDim % GmemElemsPerLoad == 0, "Headdim must be a multiple of GmemElemsPerLoad");
  // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
  // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
  // We want each thread to have at least 2 loads in the K direction since in the case of non-interleaved
  // rotary (combining elements at indices 0 and rotary_dim/2, 1 and rotary_dim/2+1, etc), each thread will
  // load twice from the same row.
  static constexpr int kBytePerHalfRow = HeadDim / 2 * sizeof(Element);
  static constexpr int kBlockKGmem =
      (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int GmemThreadsPerRow = kBlockKGmem / GmemElemsPerLoad;
  static_assert(NumThreads % GmemThreadsPerRow == 0, "NumThreads must be a multiple of GmemThreadsPerRow");
  // We assume threads loading the same row are in the same warp.
  static_assert(mutlass::NumThreadsPerWarp % GmemThreadsPerRow == 0, "GmemThreadsPerRow must divide NumThreadsPerWarp");

  using LayoutAtom =
      Layout<Shape<Int<NumThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>, Stride<Int<GmemThreadsPerRow>, _1>>;
  using TiledCopyQK =
      decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                               LayoutAtom{},
                               Layout<Shape<_1, Int<GmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store
  using GmemTiledCopyRotary =
      decltype(make_tiled_copy(Copy_Atom<MP31_ROBUST_LOAD<mute::uint64_t>, Element>{},
                               LayoutAtom{},
                               Layout<Shape<_1, Int<GmemElemsPerLoad / 2>>>{}));  // Val layout, 4 or 8 vals per store
  using GmemTiledCopyRotaryCont =
      decltype(make_tiled_copy(Copy_Atom<MP31_ROBUST_LOAD<mute::uint128_t>, Element>{},
                               LayoutAtom{},
                               Layout<Shape<_1, Int<GmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store
  using GmemTiledCopyR2G =
      decltype(make_tiled_copy(Copy_Atom<MP31_ROBUST_STORE<mute::uint128_t>, Element>{},
                               LayoutAtom{},
                               Layout<Shape<_1, Int<GmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

  using ShapeRotary  = mute::Shape<int32_t, int32_t>;  // (seqlen_ro, rotary_dim // 2)
  using StrideRotary = mute::Stride<int64_t, _1>;

  using GmemThrCopyRotary     = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)));
  using GmemThrCopyRotaryCont = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)));
  using TensortRcR            = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_D(
      mute::make_identity_tensor(Shape<Int<TileMN>, Int<HeadDim / 2>>{})));
  using TensortRpR            = decltype(make_tensor<bool>(make_shape(size<2>(TensortRcR{}))));
  using TensortRcRCont        = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)).partition_D(
      mute::make_identity_tensor(Shape<Int<TileMN>, Int<HeadDim / 2>>{})));
  using TensortRpRCont        = decltype(make_tensor<bool>(make_shape(size<2>(TensortRcRCont{}))));
  using TensormR              = decltype(make_tensor(make_gmem_ptr((Element const *)nullptr),
                                        ShapeRotary{},
                                        make_stride(mute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{})));
  using TensortRgR            = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_S(
      make_tensor(make_gmem_ptr((Element const *)nullptr),
                  make_shape(Int<TileMN>{}, Int<HeadDim / 2>{}, int(0)),
                  make_stride(mute::conditional_return<FixedPosition>(_0{}, int64_t(0)),
                              _1{},
                              mute::conditional_return<FixedPosition>(_0{}, int64_t(0))))));
  using TensortRgRCont        = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)).partition_S(
      make_tensor(make_gmem_ptr((Element const *)nullptr),
                  make_shape(Int<TileMN>{}, Int<HeadDim / 2>{}, int(0)),
                  make_stride(mute::conditional_return<FixedPosition>(_0{}, int64_t(0)),
                              _1{},
                              mute::conditional_return<FixedPosition>(_0{}, int64_t(0))))));

  GmemTiledCopyRotary         gmem_tiled_copy_rotary;
  GmemTiledCopyRotaryCont     gmem_tiled_copy_rotary_cont;
  int const                   rotary_dim;
  int const                   thread_idx;
  int const                   max_seqlen;
  GmemThrCopyRotary const     gmem_thr_copy_rotary;
  GmemThrCopyRotaryCont const gmem_thr_copy_rotary_cont;
  TensortRpR                  tRpR;
  TensortRpRCont              tRpRCont;
  TensormR                    mCos, mSin;
  TensortRgR                  tRgCos, tRgSin;
  TensortRgRCont              tRgCosCont, tRgSinCont;
  GmemTiledCopyR2G            gmem_tiled_copy_r2g;

  MUTLASS_DEVICE
  Rotary(Element const *const ptr_rotary_cos,
         ShapeRotary const   &shape_rotary,
         StrideRotary const  &stride_rotary_cos_,
         Element const *const ptr_rotary_sin,
         StrideRotary const  &stride_rotary_sin_,
         int const            thread_idx,
         int const            max_seqlen,
         uint32_t const       start_idx)
      : rotary_dim(get<1>(shape_rotary) * 2),
        thread_idx(thread_idx),
        max_seqlen(max_seqlen),
        gmem_thr_copy_rotary(gmem_tiled_copy_rotary.get_thread_slice(thread_idx)),
        gmem_thr_copy_rotary_cont(gmem_tiled_copy_rotary_cont.get_thread_slice(thread_idx)) {
    auto stride_rotary_cos = make_stride(mute::conditional_return<!FixedPosition>(get<0>(stride_rotary_cos_), _0{}),
                                         get<1>(stride_rotary_cos_));
    auto stride_rotary_sin = make_stride(mute::conditional_return<!FixedPosition>(get<0>(stride_rotary_sin_), _0{}),
                                         get<1>(stride_rotary_sin_));
    mCos                   = make_tensor(
        make_gmem_ptr(ptr_rotary_cos + start_idx * get<0>(stride_rotary_cos_)), shape_rotary, stride_rotary_cos);
    mSin = make_tensor(
        make_gmem_ptr(ptr_rotary_sin + start_idx * get<0>(stride_rotary_sin_)), shape_rotary, stride_rotary_sin);
    Tensor gCos = local_tile(mCos, Shape<Int<TileMN>, Int<HeadDim / 2>>{}, make_coord(_, _0{}));  // (MN, K / 2, _)
    Tensor gSin = local_tile(mSin, Shape<Int<TileMN>, Int<HeadDim / 2>>{}, make_coord(_, _0{}));  // (MN, K / 2, _)
    tRgCos      = gmem_thr_copy_rotary.partition_S(gCos);
    tRgSin      = gmem_thr_copy_rotary.partition_S(gSin);
    tRgCosCont  = gmem_thr_copy_rotary_cont.partition_S(gCos);
    tRgSinCont  = gmem_thr_copy_rotary_cont.partition_S(gSin);
    Tensor cR   = mute::make_identity_tensor(Shape<Int<TileMN>, Int<HeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
    Tensor tRcR = gmem_thr_copy_rotary.partition_D(cR);
    tRpR        = make_tensor<bool>(make_shape(size<2>(tRcR)));
    MUTE_UNROLL
    for (int k = 0; k < size(tRpR); ++k) {
      tRpR(k) = get<1>(tRcR(_0{}, _0{}, k)) < get<1>(shape_rotary);
    }
    Tensor tRcRCont = gmem_thr_copy_rotary_cont.partition_D(cR);
    tRpRCont        = make_tensor<bool>(make_shape(size<2>(tRcRCont)));
    MUTE_UNROLL
    for (int k = 0; k < size(tRpRCont); ++k) {
      tRpRCont(k) = get<1>(tRcRCont(_0{}, _0{}, k)) < get<1>(shape_rotary);
    }
  };

  template <bool IsInterleaved = true, class DescCos, class DescSin>
  MUTLASS_DEVICE auto load_cos_sin(int const block, DescCos const desc_Cos, DescSin const desc_Sin) {
    using GmemTiledCopyRo   = std::conditional_t<IsInterleaved, GmemTiledCopyRotary, GmemTiledCopyRotaryCont>;
    auto   gmem_thr_copy_ro = mute::conditional_return<IsInterleaved>(gmem_thr_copy_rotary, gmem_thr_copy_rotary_cont);
    Tensor tRpRCur          = mute::conditional_return<IsInterleaved>(tRpR, tRpRCont);
    Tensor tRgCosCur        = mute::conditional_return<IsInterleaved>(tRgCos, tRgCosCont)(_, _, _, block);
    Tensor tRgSinCur        = mute::conditional_return<IsInterleaved>(tRgSin, tRgSinCont)(_, _, _, block);
    // make_tensor_like, not make_fragment_like. If the row_stride is _0{} we want to keep it that way
    Tensor tRrCos = make_tensor_like(tRgCosCur);
    Tensor tRrSin = make_tensor_like(tRgSinCur);
    Tensor cR     = mute::make_identity_tensor(Shape<Int<TileMN>, Int<HeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
    Tensor tRcR   = gmem_thr_copy_ro.partition_D(cR);
    // If FixedPosition, only copy the first row as we only need the cos/sin for position cache_seqlens
    MUTE_UNROLL
    for (int m = 0; m < (!FixedPosition ? size<1>(tRrCos) : 1); ++m) {
      bool should_load = get<0>(tRcR(_0{}, m, _0{})) < std::min(max_seqlen - block * TileMN, TileMN);
      MUTE_UNROLL
      for (int k = 0; k < size<2>(tRrCos); ++k) {
        bool pred = should_load && tRpRCur(k);
        mute::copy(GmemTiledCopyRo{}.with(desc_Cos).with(pred), tRgCosCur(_, m, k), tRrCos(_, m, k));
        mute::copy(GmemTiledCopyRo{}.with(desc_Sin).with(pred), tRgSinCur(_, m, k), tRrSin(_, m, k));
      }
    }
    return mute::make_tuple(tRrCos, tRrSin);
    ;
  }

  template <bool IsInterleaved = true, class DescCos, class DescSin>
  MUTLASS_DEVICE auto load_cos_sin_packgqa(int const block, DescCos const desc_Cos, DescSin const desc_Sin) {
    static constexpr int kGmemElemsPerLoadCur = IsInterleaved ? GmemElemsPerLoad / 2 : GmemElemsPerLoad;
    using GmemTiledCopyRo   = std::conditional_t<IsInterleaved, GmemTiledCopyRotary, GmemTiledCopyRotaryCont>;
    auto   gmem_thr_copy_ro = mute::conditional_return<IsInterleaved>(gmem_thr_copy_rotary, gmem_thr_copy_rotary_cont);
    Tensor tRpRCur          = mute::conditional_return<IsInterleaved>(tRpR, tRpRCont);
    // make_tensor_like, not make_fragment_like. If the row_stride is _0{} we want to keep it that way
    Tensor tRrCos = make_tensor_like(mute::conditional_return<IsInterleaved>(tRgCos, tRgCosCont)(_, _, _, _0{}));
    Tensor tRrSin = make_tensor_like(mute::conditional_return<IsInterleaved>(tRgSin, tRgSinCont)(_, _, _, _0{}));
    Tensor cR     = mute::make_identity_tensor(Shape<Int<TileMN>, Int<HeadDim / 2>>{});  // (BLK_N,BLK_K / 2)
    Tensor tRcR   = gmem_thr_copy_ro.partition_D(cR);

    // The main bottleneck here is actually instruction cache misses.

    // Similar to PagedKVNonTMA, it's expensive to compute the pointers.
    // We split the work among threads loading the same row, then __shfl_sync the pointers.
    static constexpr int NumPtrPerThread = mute::ceil_div(size<1>(tRrCos), GmemThreadsPerRow);
    Tensor               tPrCosPtr       = make_tensor<Element const *>(Shape<Int<NumPtrPerThread>>{});
    Tensor               tPrSinPtr       = make_tensor<Element const *>(Shape<Int<NumPtrPerThread>>{});
    MUTE_UNROLL
    for (int i = 0; i < NumPtrPerThread; ++i) {
      int const row        = i * NumThreads + get<0>(tRcR(_0{}, thread_idx % GmemThreadsPerRow, _0{}));
      int const idx        = block * TileMN + row;
      int       row_actual = idx / HeadRatio;
      tPrCosPtr[i]         = &mCos(row_actual, _0{});
      tPrSinPtr[i]         = &mSin(row_actual, _0{});
    }

    MUTE_UNROLL
    for (int m = 0; m < (!FixedPosition ? size<1>(tRgCos) : 1); ++m) {
      int const      idx     = block * TileMN + get<0>(tRcR(_0{}, m, _0{}));
      Element const *cos_ptr = (Element *)__musa_ptr_gen_to_global(
          (void *)(__shfl_sync(0xffffffff,
                               reinterpret_cast<uint64_t>(tPrCosPtr(m / GmemThreadsPerRow)),
                               m % GmemThreadsPerRow,
                               GmemThreadsPerRow)));
      Element const *sin_ptr = (Element *)__musa_ptr_gen_to_global(
          (void *)(__shfl_sync(0xffffffff,
                               reinterpret_cast<uint64_t>(tPrSinPtr(m / GmemThreadsPerRow)),
                               m % GmemThreadsPerRow,
                               GmemThreadsPerRow)));
      bool   should_load = idx < max_seqlen * HeadRatio;
      Tensor mCos_copy   = mute::tiled_divide(make_tensor(make_gmem_ptr(cos_ptr), Shape<Int<HeadDim / 2>>{}),
                                            Shape<Int<kGmemElemsPerLoadCur>>{});
      Tensor mSin_copy   = mute::tiled_divide(make_tensor(make_gmem_ptr(sin_ptr), Shape<Int<HeadDim / 2>>{}),
                                            Shape<Int<kGmemElemsPerLoadCur>>{});
      MUTE_UNROLL
      for (int k = 0; k < size<2>(tRgCos); ++k) {
        bool      pred = should_load && tRpRCur(k);
        int const ki   = get<1>(tRcR(_0{}, _0{}, k)) / (kGmemElemsPerLoadCur);
        mute::copy(GmemTiledCopyRo{}.with(desc_Cos).with(pred), mCos_copy(_, ki), tRrCos(_, m, k));
        mute::copy(GmemTiledCopyRo{}.with(desc_Sin).with(pred), mSin_copy(_, ki), tRrSin(_, m, k));
      }
    }
    return mute::make_tuple(tRrCos, tRrSin);
  }

  template <typename TensorsQ, typename TensortRrR>
  MUTLASS_DEVICE void apply_Q_interleaved(
      TensorsQ         &sQ,      // (TileM, HeadDim)
      TensortRrR const &tRrCos,  // (TileM, HeadDim / 2) split according to GmemThrCopyRotary
      TensortRrR const &tRrSin,  // (TileM, HeadDim / 2) split according to GmemThrCopyRotary
      int const         m_block) {
    TiledCopyQK tiled_copy_q;
    auto        gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
    Tensor      tQsQ            = gmem_thr_copy_q.partition_S(sQ);
    Tensor      tQcQ = gmem_thr_copy_q.partition_S(mute::make_identity_tensor(Shape<Int<TileMN>, Int<HeadDim>>{}));

    MUTE_STATIC_ASSERT_V(rank(tQsQ) == _3{});
    MUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
    MUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
    MUTE_STATIC_ASSERT_V(size<1>(tQsQ) == size<1>(tRrCos));
    MUTE_STATIC_ASSERT_V(size<2>(tQsQ) == size<2>(tRrCos));
    MUTE_STATIC_ASSERT_V(size<1>(tQsQ) == size<1>(tRrSin));
    MUTE_STATIC_ASSERT_V(size<2>(tQsQ) == size<2>(tRrSin));
    MUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
    static_assert(decltype(size<0>(tQsQ))::value == decltype(size<0>(tRrCos))::value * 2);
    static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32

    MUTE_UNROLL
    for (int m = 0; m < size<1>(tQsQ); ++m) {
      if (get<0>(tQcQ(_0{}, m, _0{})) < std::min(max_seqlen * HeadRatio - m_block * TileMN, TileMN)) {
        MUTE_UNROLL
        for (int k = 0; k < size<2>(tQsQ); ++k) {
          if (tRpR(k)) {
            Tensor rQ = make_fragment_like(tQsQ(_, m, k));
            mute::copy(tiled_copy_q, tQsQ(_, m, k), rQ);
            apply_rotary_interleaved<Fragment>(rQ, tRrCos(_, m, k), tRrSin(_, m, k));
            mute::copy(tiled_copy_q, rQ, tQsQ(_, m, k));
          }
        }
      }
    }
  };

  template <typename TensorsQ, typename TensortRrR>
  MUTLASS_DEVICE void apply_Q_contiguous(
      TensorsQ         &sQ,          // (TileM, HeadDim)
      TensortRrR const &tRrCosCont,  // (TileM, HeadDim / 2) split according to GmemThrCopyRotaryCont
      TensortRrR const &tRrSinCont,  // (TileM, HeadDim / 2) split according to GmemThrCopyRotaryCont
      int const         m_block) {
    TiledCopyQK tiled_copy_q;
    auto        gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
    Tensor      sQ_copy         = mute::tiled_divide(sQ, Shape<_1, Int<GmemElemsPerLoad>>{});
    Tensor      tQcQ = gmem_thr_copy_q.partition_S(mute::make_identity_tensor(Shape<Int<TileMN>, Int<HeadDim / 2>>{}));

    MUTE_STATIC_ASSERT_V(rank(tQcQ) == _3{});
    MUTE_STATIC_ASSERT_V(rank(tRrCosCont) == _3{});
    MUTE_STATIC_ASSERT_V(rank(tRrSinCont) == _3{});
    MUTE_STATIC_ASSERT_V(size<1>(tQcQ) == size<1>(tRrCosCont));
    MUTE_STATIC_ASSERT_V(size<2>(tQcQ) == size<2>(tRrCosCont));
    MUTE_STATIC_ASSERT_V(size<1>(tQcQ) == size<1>(tRrSinCont));
    MUTE_STATIC_ASSERT_V(size<2>(tQcQ) == size<2>(tRrSinCont));
    MUTE_STATIC_ASSERT_V(size<0>(tRrCosCont) == size<0>(tRrSinCont));
    MUTE_STATIC_ASSERT_V(size<0>(tQcQ) == size<0>(tRrCosCont));
    static_assert(decltype(size<0>(tRrCosCont))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32

    MUTE_UNROLL
    for (int m = 0; m < size<1>(tQcQ); ++m) {
      int const row = get<0>(tQcQ(_0{}, m, _0{}));
      if (row < std::min(max_seqlen * HeadRatio - m_block * TileMN, TileMN)) {
        MUTE_UNROLL
        for (int k = 0; k < size<2>(tQcQ); ++k) {
          int const col = get<1>(tQcQ(_0{}, _0{}, k));
          if (col < rotary_dim / 2) {
            int const col_idx_left  = col / GmemElemsPerLoad;
            int const col_idx_right = col / GmemElemsPerLoad + rotary_dim / (2 * GmemElemsPerLoad);
            Tensor    rQ_left       = make_fragment_like(sQ_copy(_, row, col_idx_left));
            mute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_left), rQ_left);
            Tensor rQ_right = make_fragment_like(rQ_left);
            mute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_right), rQ_right);
            apply_rotary_contiguous<Fragment>(rQ_left, rQ_right, tRrCosCont(_, m, k), tRrSinCont(_, m, k));
            mute::copy(tiled_copy_q, rQ_left, sQ_copy(_, row, col_idx_left));
            mute::copy(tiled_copy_q, rQ_right, sQ_copy(_, row, col_idx_right));
          }
        }
      }
    }
  };

  template <bool PagedKVNonTMA = false,
            typename TensorsK,
            typename TensorgK,
            typename TensorpK,
            typename TensortRrR,
            typename TensorKPtr,
            class DescK>
  MUTLASS_DEVICE void apply_K_interleaved(
      TensorsK const   &sK,      // (TileN, HeadDim)
      TensorgK         &gK,      // (TileN, HeadDim)
      TensorpK const   &tKpK,    // (TileN, HeadDim) split according to ThrCopyKV
      TensortRrR const &tRrCos,  // (TileN, HeadDim/2) split according to GmemThrCopyRotary
      TensortRrR const &tRrSin,  // (TileN, HeadDim/2) split according to GmemThrCopyRotary
      TensorKPtr const &tPrKPtr,
      int const         n_block,
      int const         max_k,
      DescK const       desc_K) {
    TiledCopyQK tiled_copy_k;
    auto        gmem_thr_copy_q = tiled_copy_k.get_thread_slice(thread_idx);
    // Tensor      sK_copy         = mute::tiled_divide(sK, Shape<_1, Int<GmemElemsPerLoad>>{});
    // Tensor      gK_copy         = mute::tiled_divide(gK, Shape<_1, Int<GmemElemsPerLoad>>{});
    Tensor tKsK = gmem_thr_copy_q.partition_S(sK);
    Tensor tKgK = gmem_thr_copy_q.partition_S(gK);
    Tensor tKcK = gmem_thr_copy_q.partition_S(mute::make_identity_tensor(Shape<Int<TileMN>, Int<HeadDim>>{}));

    MUTE_STATIC_ASSERT_V(rank(tKcK) == _3{});
    MUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
    MUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
    MUTE_STATIC_ASSERT_V(size<1>(tKcK) == size<1>(tRrCos));
    MUTE_STATIC_ASSERT_V(size<2>(tKcK) == size<2>(tRrCos));
    MUTE_STATIC_ASSERT_V(size<1>(tKcK) == size<1>(tRrSin));
    MUTE_STATIC_ASSERT_V(size<2>(tKcK) == size<2>(tRrSin));
    MUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
    // static_assert(decltype(size<1>(sK_copy))::value == TileMN);
    // static_assert(decltype(size<0>(sK_copy))::value == decltype(size<0>(tRrCos))::value * 2);
    static_assert(decltype(size<0>(tKsK))::value == decltype(size<0>(tRrCos))::value * 2);
    static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    if constexpr (PagedKVNonTMA) {
      static_assert(decltype(size(tPrKPtr))::value == mute::ceil_div(size<1>(tKcK), GmemThreadsPerRow));
    }

    MUTE_UNROLL
    for (int m = 0; m < size<1>(tKsK); ++m) {
      int const row         = get<0>(tKcK(_0{}, m, _0{}));
      auto      mK_cur_copy = [&] {
        if constexpr (PagedKVNonTMA) {
          Element *k_ptr = (Element *)__musa_ptr_gen_to_global(
              (void *)(__shfl_sync(0xffffffff,
                                   reinterpret_cast<uint64_t>(tPrKPtr(m / GmemThreadsPerRow)),
                                   (m % GmemThreadsPerRow),
                                   GmemThreadsPerRow)));
          Tensor mK_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<HeadDim>>{});
          return mute::tiled_divide(mK_cur, Shape<Int<GmemElemsPerLoad>>{});
        } else {
          return nullptr;
        }
      }();
      bool should_write = row < std::min(max_seqlen - n_block * TileMN, TileMN);
      MUTE_UNROLL
      for (int k = 0; k < size<2>(tKsK); ++k) {
        Tensor rK   = make_fragment_like(tKsK(_, m, k));
        bool   pred = should_write && tKpK(k);
        mute::copy(tiled_copy_k, tKsK(_, m, k), rK);
        // int const col_idx = get<1>(tKcK(_0{}, _0{}, k)) / GmemElemsPerLoad;
        // Tensor    rK      = make_fragment_like(sK_copy(_, row, col_idx));
        // bool      pred    = should_write && get<1>(tKcK(_0{}, _0{}, k)) < max_k;
        // mute::copy(tiled_copy_k, sK_copy(_, row, col_idx), rK);
        if (tRpR(k)) {
          apply_rotary_interleaved<Fragment>(rK, tRrCos(_, m, k), tRrSin(_, m, k));
        }
        if constexpr (!PagedKVNonTMA) {
          mute::copy(gmem_tiled_copy_r2g.with(desc_K).with(pred), rK, tKgK(_, m, k));
          // mute::copy(gmem_tiled_copy_r2g.with(desc_K).with(pred), rK, gK_copy(_, row, col_idx));
        } else {
          int const ki = get<1>(tKcK(_0{}, _0{}, k)) / GmemElemsPerLoad;
          mute::copy(gmem_tiled_copy_r2g.with(desc_K).with(pred), rK, mK_cur_copy(_, ki));
          // mute::copy(gmem_tiled_copy_r2g.with(desc_K).with(pred), rK, mK_cur_copy(_, col_idx));
        }
      }
    }
  };

  template <bool PagedKVNonTMA = false,
            typename TensorsK,
            typename TensorgK,
            typename TensorpK,
            typename TensortRrR,
            typename TensorKPtr,
            typename DescK>
  MUTLASS_DEVICE void apply_K_contiguous(
      TensorsK const   &sK,          // (TileN, HeadDim)
      TensorgK         &gK,          // (TileN, HeadDim)
      TensorpK const   &tKpK,        // (TileN, HeadDim) split according to ThrCopyKV
      TensortRrR const &tRrCosCont,  // (TileN, HeadDim/2) split according to GmemThrCopyRotaryCont
      TensortRrR const &tRrSinCont,  // (TileN, HeadDim/2) split according to GmemThrCopyRotaryCont
      TensorKPtr const &tPrKPtr,
      int const         n_block,
      int const         max_k,
      DescK const       desc_K) {
    TiledCopyQK tiled_copy_k;
    auto        gmem_thr_copy_q = tiled_copy_k.get_thread_slice(thread_idx);
    Tensor      sK_copy         = mute::tiled_divide(sK, Shape<_1, Int<GmemElemsPerLoad>>{});
    Tensor      gK_copy         = mute::tiled_divide(gK, Shape<_1, Int<GmemElemsPerLoad>>{});
    Tensor      tKcK = gmem_thr_copy_q.partition_S(mute::make_identity_tensor(Shape<Int<TileMN>, Int<HeadDim / 2>>{}));

    MUTE_STATIC_ASSERT_V(rank(tKcK) == _3{});
    MUTE_STATIC_ASSERT_V(rank(tRrCosCont) == _3{});
    MUTE_STATIC_ASSERT_V(rank(tRrSinCont) == _3{});
    MUTE_STATIC_ASSERT_V(size<1>(tKcK) == size<1>(tRrCosCont));
    MUTE_STATIC_ASSERT_V(size<2>(tKcK) == size<2>(tRrCosCont));
    MUTE_STATIC_ASSERT_V(size<1>(tKcK) == size<1>(tRrSinCont));
    MUTE_STATIC_ASSERT_V(size<2>(tKcK) == size<2>(tRrSinCont));
    MUTE_STATIC_ASSERT_V(size<0>(tRrCosCont) == size<0>(tRrSinCont));
    MUTE_STATIC_ASSERT_V(size<0>(tKcK) == size<0>(tRrCosCont));
    static_assert(decltype(size<0>(tRrCosCont))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    if constexpr (PagedKVNonTMA) {
      static_assert(decltype(size(tPrKPtr))::value == mute::ceil_div(size<1>(tKcK), GmemThreadsPerRow));
    }

    const int ro_dim_vec     = rotary_dim / GmemElemsPerLoad;
    const int non_ro_dim_vec = (max_k - rotary_dim) / GmemElemsPerLoad;
    MUTE_UNROLL
    for (int m = 0; m < size<1>(tKcK); ++m) {
      int const row         = get<0>(tKcK(_0{}, m, _0{}));
      Tensor    gK_cur_copy = [&] {
        if constexpr (PagedKVNonTMA) {
          Element *k_ptr = (Element *)__musa_ptr_gen_to_global(
              (void *)(__shfl_sync(0xffffffff,
                                   reinterpret_cast<uint64_t>(tPrKPtr(m / GmemThreadsPerRow)),
                                   (m % GmemThreadsPerRow),
                                   GmemThreadsPerRow)));
          Tensor mK_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<HeadDim>>{});
          return mute::tiled_divide(mK_cur, Shape<Int<GmemElemsPerLoad>>{});
        } else {
          return gK_copy(_, row, _);
        }
      }();
      bool should_write = row < std::min(max_seqlen - n_block * TileMN, TileMN);
      MUTE_UNROLL
      for (int k = 0; k < size<2>(tKcK); ++k) {
        bool      pred          = should_write && tKpK(k);
        int const col           = get<1>(tKcK(_0{}, _0{}, k));
        bool      rotate        = col < rotary_dim / 2;
        int const col_idx_left  = rotate ? col / GmemElemsPerLoad : (col + rotary_dim / 2) / GmemElemsPerLoad;
        int const col_idx_right = col_idx_left + (rotate ? ro_dim_vec / 2 : non_ro_dim_vec / 2);
        Tensor    rK_left       = make_fragment_like(sK_copy(_, row, col_idx_left));
        mute::copy(tiled_copy_k, sK_copy(_, row, col_idx_left), rK_left);
        Tensor rK_right = make_fragment_like(rK_left);
        mute::copy(tiled_copy_k, sK_copy(_, row, col_idx_right), rK_right);
        if (rotate) {
          apply_rotary_contiguous<Fragment>(rK_left, rK_right, tRrCosCont(_, m, k), tRrSinCont(_, m, k));
        }
        mute::copy(gmem_tiled_copy_r2g.with(desc_K).with(pred), rK_left, gK_cur_copy(_, col_idx_left));
        if (col_idx_right * GmemElemsPerLoad < max_k) {
          mute::copy(gmem_tiled_copy_r2g.with(desc_K).with(pred), rK_right, gK_cur_copy(_, col_idx_right));
        }
      }
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace mate::attention::fmha
