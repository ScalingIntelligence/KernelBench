#pragma once

#include <mutlass/fast_math.h>
#include <mutlass/numeric_conversion.h>

#include <mute/atom/mma_atom.hpp>
#include <mute/tensor.hpp>

namespace mate::attention::fmha {

using namespace mute;

template <class Threshold, class Source, class Reference>
MUTE_HOST_DEVICE constexpr auto layout_separate(Threshold const& thr, Source const& src, Reference const& ref) {
  auto lt = filter(transform_layout(src, ref, [&](auto const& s, auto const& r) {
    if constexpr (decltype(r < thr)::value) {
      return s;
    } else {
      return make_layout(_1{}, _0{});
    }
  }));

  auto ge = filter(transform_layout(src, ref, [&](auto const& s, auto const& r) {
    if constexpr (decltype(r >= thr)::value) {
      return s;
    } else {
      return make_layout(_1{}, _0{});
    }
  }));
  return make_tuple(lt, ge);
}

template <class TiledMma, class Acc>
MUTE_HOST_DEVICE constexpr auto layout_acc_mn(TiledMma const& tiled_mma, Acc const& acc) {
  auto [V_M, V_N] =
      layout_separate(get<0>(typename TiledMma::Shape_MNK{}), get<0>(acc), stride<1>(typename TiledMma::LayoutC_TV{}));
  return make_layout(make_layout(V_M, get<1>(acc)), make_layout(V_N, get<2>(acc)));
}

template <class TiledMma>
MUTE_HOST_DEVICE constexpr auto reduction_target_n(TiledMma const& tiled_mma) {
  auto separated = layout_separate(get<0>(typename TiledMma::Shape_MNK{}),
                                   make_layout(shape<0>(typename TiledMma::LayoutC_TV{})),
                                   stride<0>(typename TiledMma::LayoutC_TV{}));
  return get<1>(separated);
}

template <template <class MmaAtom, class AtomMNK, class DefaultPermutation> class MmaPrimtive,
          class MmaAtom,
          class AtomMNK,
          class DefaultPermutation,
          class PermutationMNK>
MUTE_HOST_DEVICE constexpr auto convert_to_permuted_mma(MmaPrimtive<MmaAtom, AtomMNK, DefaultPermutation> const& prim,
                                                        PermutationMNK const&                                    perm) {
  return TiledMMA<MmaAtom, AtomMNK, PermutationMNK>{};
}

template <template <class MmaAtom, class AtomMNK, class DefaultPermutation> class MmaPrimtive,
          class MmaAtom,
          class AtomMNK,
          class Permutation>
MUTE_HOST_DEVICE constexpr auto convert_to_atom_mma(MmaPrimtive<MmaAtom, AtomMNK, Permutation> const& prim) {
  return make_tiled_mma(MmaAtom{});
}

template <class Layout, class Stages = _1>
MUTE_HOST_DEVICE constexpr auto unstageSmemLayout(Layout const& layout, Stages stages = {}) {
  return composition(layout, make_tuple(_, _, make_layout(stages)));
}

template <int FragmentSize_ = -1, class EngineSrc, typename Layout, class EngineDst>
MUTE_DEVICE void convert_type(Tensor<EngineSrc, Layout> const& src, Tensor<EngineDst, Layout>& dst) {
  using SrcType = typename EngineSrc::value_type;
  using DstType = typename EngineDst::value_type;

  static constexpr int FragmentSize = [&] {
    if constexpr (FragmentSize_ > 0) {
      return FragmentSize_;
    } else {
      return sizeof(SrcType) > sizeof(DstType) ? sizeof(SrcType) / sizeof(DstType) : sizeof(DstType) / sizeof(SrcType);
    }
  }();

  static_assert(MUTE_STATIC_V(size(src)) % FragmentSize == 0, "Fragment size does not vectorize properly");

  Tensor src_frg = recast<mutlass::Array<SrcType, FragmentSize> const>(src);
  Tensor dst_frg = recast<mutlass::Array<DstType, FragmentSize>>(dst);
  static_assert(size(src_frg) == size(dst_frg));

  mutlass::NumericArrayConverter<DstType, SrcType, FragmentSize, mutlass::FloatRoundStyle::round_to_nearest> convert_op;

  MUTE_UNROLL
  for (int i = 0; i < size(src_frg); ++i) {
    dst_frg(i) = convert_op(src_frg(i));
  }
}

MUTE_HOST_DEVICE int round_up_headdim(int headdim) {
  if (headdim <= 64) {
    return 64;
  }
  if (headdim <= 96) {
    return 96;
  }
  if (headdim <= 128) {
    return 128;
  }
  if (headdim <= 192) {
    return 192;
  }
  if (headdim <= 256) {
    return 256;
  }
  return 512;
}

MUTE_HOST_DEVICE int div_floor(mutlass::FastDivmod const& divmod, int dividend) {
  int const sign = -static_cast<int>(static_cast<unsigned>(dividend) >> 31);
  return divmod.divide(dividend ^ sign) ^ sign;
}

MUTE_HOST_DEVICE int div_ceil(mutlass::FastDivmod const& divmod, int dividend) {
  return dividend >= 0 ? divmod.divide(dividend + divmod.divisor - 1) : -divmod.divide(-dividend);
}

}  // namespace mate::attention::fmha
