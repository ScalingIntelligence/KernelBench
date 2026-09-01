#pragma once

#include <mute/tensor.hpp>
#include <mute/atom/copy_atom.hpp>

namespace mutlass::fmha::collective {

using namespace mute;

template <
  class Element,
  class CanonicalSmemLayout,
  class StrideK_,
  int VectorBits = 128
>
struct Mp31FmhaTmeLoadKeyBuilder {
  static constexpr int MmaAtomN = 8;
  static constexpr int Fragment = VectorBits / sizeof_bits_v<Element>;
  static constexpr int Granularity = MmaAtomN * Fragment;

  static constexpr int SmemN     = size<0>(CanonicalSmemLayout{});
  static constexpr int SmemK     = size<1>(CanonicalSmemLayout{});
  static constexpr int SmemStage = size<2>(CanonicalSmemLayout{});
  static constexpr int Repeats   = SmemN / Granularity;

  static_assert(SmemN % Granularity == 0);

  static constexpr bool BlockFit = Repeats == 1;

  using FragmentType = mute::uint_bit_t<VectorBits>;
  using IndexType = mute::remove_cvref_t<decltype(get<0>(StrideK_{}))>;

  static constexpr int ElementBits = sizeof_bits_v<Element>;

  using PermuteTile = decltype(
      make_tile(Underscore{},
                filter(make_ordered_layout(Shape<Int<MmaAtomN>, Int<Fragment>, Int<Repeats>>{},
                                    Step<_2, _1, _3>{})),
                Underscore{}));

  using SmemLayoutK = decltype(composition(CanonicalSmemLayout{}, select<1, 2, 0>(PermuteTile{})));
  using StrideK = conditional_t<BlockFit, decltype(replace<0>(StrideK_{}, Stride<IndexType, IndexType>{})),
        decltype(replace<2>(replace<0>(StrideK_{}, Stride<IndexType, IndexType, IndexType>{}), IndexType{}))>;

  using TmeTileShapeKD = decltype(make_shape(shape<1>(PermuteTile{}), Int<SmemK>{}));

  using TME_K = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
        make_gmem_ptr(static_cast<Element const*>(nullptr)),
        repeat_like(StrideK{}, int(0)),
        StrideK{}),
      take<0, 2>(SmemLayoutK{}),
      TmeTileShapeKD{}));

  struct Arguments {
    Element const* ptr;
    int64_t page_size_or_seqlen_stride;
    int64_t k_head_stride;
    int64_t num_pages_or_batch_stride;
  };

  template <class ProblemSize>
  MUTE_HOST
  static bool
  can_implement(ProblemSize problem_size, Arguments const& args) {
    bool implementable = true;
    // BNHD layout
    // TODO: support BHND layout
    implementable = implementable && args.k_head_stride == get<2>(problem_size);
    implementable = implementable && get<2>(problem_size) == SmemK;
    return implementable;
  }

  template <class GEngine, class GLayout>
  static TME_K
  make_tme_copy(Tensor<GEngine, GLayout> const& gtensor) {
    MUTE_STATIC_ASSERT_V(rank(gtensor) == Int<3>{});

    if constexpr (BlockFit) {
      auto reshape_tile = make_tile(
        make_layout(make_shape(Int<Fragment>{}, ceil_div(shape<0>(gtensor), Fragment))),
        _,
        _
      );
      auto reshaped_layout = composition(gtensor.layout(), reshape_tile);

      auto reshaped_gtensor = make_tensor(gtensor.data(), reshaped_layout);

      return mute::make_tme_copy(
        MP31_TME_LOAD{},
        reshaped_gtensor,
        take<0, 2>(SmemLayoutK{}),
        TmeTileShapeKD{});
    } else {
      Shape <IndexType, IndexType, IndexType> merged_shape;
      Stride<IndexType,        _1, IndexType> merged_stride;

      auto [K, D, HB] = shape(gtensor);
      auto [dK, dD, dHB] = stride(gtensor);

      auto [H, B] = HB;
      auto [dH, dB] = dHB;

      get<0>(merged_shape) = K;
      get<1>(merged_shape) = D * H;
      get<2>(merged_shape) = B;

      get<0>(merged_stride) = dK;
      get<1>(merged_stride) = dD;
      get<2>(merged_stride) = dB;

      auto merged_layout = make_layout(merged_shape, merged_stride);
      auto reshaped_layout = composition(merged_layout, make_tile(
        make_layout(make_shape(Int<MmaAtomN>{}, Int<Fragment>{}, ceil_div(shape<0>(gtensor), Int<Granularity>{}))), _, _));

      auto reshaped_gtensor = make_tensor(gtensor.data(), reshaped_layout);

      return mute::make_tme_copy(
        MP31_TME_LOAD{},
        reshaped_gtensor,
        take<0, 2>(SmemLayoutK{}),
        TmeTileShapeKD{});
    }
  }

  template <class ProblemSize>
  MUTE_HOST_DEVICE
  static auto
  get_problem_size(ProblemSize problem_size) {
    if constexpr (BlockFit) {
      using ProblemSizeForKey = Shape<int, Shape<Int<Fragment>, int>, int, int, int, int, int>;

      ProblemSizeForKey problem_size_for_key;
      get<0>(problem_size_for_key) = get<0>(problem_size);
      get<1>(problem_size_for_key) = make_shape(Int<Fragment>{}, ceil_div(get<1>(problem_size), Fragment));
      get<2>(problem_size_for_key) = get<2>(problem_size);
      get<3>(problem_size_for_key) = get<3>(problem_size);
      get<4>(problem_size_for_key) = get<4>(problem_size);
      get<5>(problem_size_for_key) = get<5>(problem_size);
      get<6>(problem_size_for_key) = get<6>(problem_size);
      return problem_size_for_key;
    } else {
      using ProblemSizeForKey = Shape<int, Shape<Int<MmaAtomN>, Int<Fragment>, int>, int, int, int, int, int>;

      ProblemSizeForKey problem_size_for_key;
      get<0>(problem_size_for_key) = get<0>(problem_size);
      get<1>(problem_size_for_key) = make_shape(Int<MmaAtomN>{}, Int<Fragment>{}, ceil_div(get<1>(problem_size), Int<Granularity>{}));
      get<2>(problem_size_for_key) = get<2>(problem_size) * get<5>(problem_size);
      get<3>(problem_size_for_key) = get<3>(problem_size);
      get<4>(problem_size_for_key) = get<4>(problem_size);
      get<5>(problem_size_for_key) = 0;
      get<6>(problem_size_for_key) = get<6>(problem_size);
      return problem_size_for_key;
    }
  }
};

template <
  class Element,
  class TileShape,
  class SmemAtomLayout,
  int Threads,
  class StrideK,
  int VectorBits = 128
>
struct Mp31FmhaLsuLoadKeyBuilder {
  static constexpr int MmaAtomN = 8;
  static constexpr int Fragment = VectorBits / sizeof_bits_v<Element>;
  static constexpr int Granularity = MmaAtomN * Fragment;

  static constexpr int SmemAtomN = size<0>(SmemAtomLayout{});
  static constexpr int SmemAtomK = size<1>(SmemAtomLayout{});
  static constexpr int Repeats   = SmemAtomN / Granularity;

  static_assert(SmemAtomN % Granularity == 0);

  using FragmentType = mute::uint_bit_t<VectorBits>;

  using GmemCopyAtom = MP31_ROBUST_LDGSTS<mute::uint_bit_t<sizeof_bits_v<Element> * Fragment>>;
  using GmemTiledCopy = decltype(mutlass::gemm::collective::detail::make_simt_tiled_copy<
                          Threads, Element, Fragment, StrideK,
                          SmemAtomN, SmemAtomK, GmemCopyAtom>());

  using LsuTileShapeKD = decltype(make_tile(
      make_ordered_layout(Shape<Int<MmaAtomN>, Int<Fragment>, Int<Repeats>>{}, Step<_2, _1, _3>{}),
      get<2>(TileShape{})));

  using PermuteTile = decltype(make_tile(Underscore{}, tuple_element_t<0, LsuTileShapeKD>{}, Underscore{}));
};

} // namespace mutlass::fmha::collective
