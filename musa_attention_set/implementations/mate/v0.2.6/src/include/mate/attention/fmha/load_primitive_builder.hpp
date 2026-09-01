#pragma once

#include <mute/atom/copy_atom.hpp>
#include <mute/tensor.hpp>

namespace mate::attention::fmha {

using namespace mute;

template <class Element,
          class CanonicalSmemLayout,
          class StrideK_,
          TME::CacheHint InnerHint  = TME::CacheHint::CACHE_NORMAL,
          TME::CacheHint OuterHint  = TME::CacheHint::CACHE_NORMAL,
          int            VectorBits = 128>
struct Mp31FmhaTmeLoadKeyBuilder {
  static constexpr int MmaAtomN    = 8;
  static constexpr int Fragment    = VectorBits / sizeof_bits_v<Element>;
  static constexpr int Granularity = MmaAtomN * Fragment;

  using FragmentType = mute::uint_bit_t<VectorBits>;
  using IndexType    = mute::remove_cvref_t<decltype(get<0>(StrideK_{}))>;

  static constexpr int SmemTileN  = size<0>(CanonicalSmemLayout{});
  static constexpr int SmemTileK  = size<1>(CanonicalSmemLayout{});
  static constexpr int SmemStages = size<2>(CanonicalSmemLayout{});
  static constexpr int Repeats    = SmemTileN / Granularity;

  static_assert(SmemTileN % Granularity == 0);

  // If BlockFit, we only need 1 extra tme dim
  static constexpr bool BlockFit = Repeats == 1;
  static_assert(BlockFit == true);

  static constexpr int ElementBits = sizeof_bits_v<Element>;

  using MmaPermuteTile =
      decltype(filter(make_ordered_layout(Shape<Int<MmaAtomN>, Int<Fragment>, Int<Repeats>>{}, Step<_2, _1, _3>{})));
  using PermuteTileN =
      decltype(filter(make_ordered_layout(Shape<Int<Fragment>, Int<MmaAtomN>, Int<Repeats>>{}, Step<_2, _1, _3>{})));

  // (int64_t, _1, int64_t, int64_t) -> ((int64_t, int64_t), _1, int64_t, int64_t)
  using StrideK = decltype(replace<0>(StrideK_{}, Stride<IndexType, IndexType>{}));

  using SmemLayoutK = decltype(composition(CanonicalSmemLayout{}, make_tile(PermuteTileN{}, _, _)));

  using TmeKTileShape = decltype(make_shape(shape(PermuteTileN{}), Int<SmemTileK>{}));

  using TME_K = decltype(make_tme_copy<InnerHint, OuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideK{}, int32_t(0)), StrideK{}),
      take<0, 2>(SmemLayoutK{}),
      TmeKTileShape{}));

  template <class GEngine, class GLayout>
  static TME_K make_tme_copy(Tensor<GEngine, GLayout> const& gtensor) {
    MUTE_STATIC_ASSERT_V(rank(gtensor) == _4{});

    auto permute_tile =
        make_tile(make_layout(make_shape(Int<Fragment>{}, ceil_div(shape<0>(gtensor), Fragment))), _, _, _);

    auto permuted_gtensor = make_tensor(gtensor.data(), composition(gtensor.layout(), permute_tile));

    return mute::make_tme_copy<InnerHint, OuterHint>(
        MP31_TME_LOAD{}, permuted_gtensor, take<0, 2>(SmemLayoutK{}), TmeKTileShape{});
  }

  template <class GEngine, class GLayout>
  MUTE_HOST_DEVICE static auto get_permuted_shape(Tensor<GEngine, GLayout> const& gtensor) {
    MUTE_STATIC_ASSERT_V(rank(gtensor) == _4{});

    auto permute_tile =
        make_tile(make_layout(make_shape(Int<Fragment>{}, ceil_div(shape<0>(gtensor), Fragment))), _, _, _);

    return shape(composition(gtensor.layout(), permute_tile));
  }
};

template <class Element, class TileShape, class SmemAtomLayout, int Threads, class StrideK, int VectorBits = 128>
struct Mp31FmhaLsuLoadKeyBuilder {
  static constexpr int MmaAtomN    = 8;
  static constexpr int Fragment    = VectorBits / sizeof_bits_v<Element>;
  static constexpr int Granularity = MmaAtomN * Fragment;

  using FragmentType = mute::uint_bit_t<VectorBits>;

  static constexpr int SmemAtomN = size<0>(SmemAtomLayout{});
  static constexpr int SmemAtomK = size<1>(SmemAtomLayout{});
  static constexpr int Repeats   = SmemAtomN / Granularity;

  static_assert(SmemAtomN % Granularity == 0);

  using GmemCopyAtom = MP31_ROBUST_LDGSTS<mute::uint_bit_t<sizeof_bits_v<Element> * Fragment>>;
  using GmemTiledCopy =
      decltype(mutlass::gemm::collective::detail::
                   make_simt_tiled_copy<Threads, Element, Fragment, StrideK, SmemAtomN, SmemAtomK, GmemCopyAtom>());

  using PermuteTileN =
      decltype(make_ordered_layout(Shape<Int<MmaAtomN>, Int<Fragment>, Int<Repeats>>{}, Step<_2, _1, _3>{}));

  using LsuKTile = decltype(make_tile(PermuteTileN{}, get<2>(TileShape{})));
};

}  // namespace mate::attention::fmha
