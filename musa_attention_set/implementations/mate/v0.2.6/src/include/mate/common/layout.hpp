#pragma once

#include <mute/atom/mma_atom.hpp>
#include <mute/tensor.hpp>

namespace mate {

using namespace mute;

template <class SmemLayout, class Stages>
MUTE_HOST_DEVICE constexpr auto unstage_smem_layout(SmemLayout const& layout, Stages) {
  auto stage_layout = take<0, 2>(layout);
  return append<3>(stage_layout,
                   make_layout(make_shape(Stages{}), make_stride(Int<cosize_v<decltype(stage_layout)>>{})));
}

template <int... Is, class Layout>
MUTE_HOST_DEVICE constexpr auto select_layout(Layout const& layout) {
  if constexpr (is_composed_layout<Layout>::value) {
    return make_composed_layout(layout.layout_a(), layout.offset(), select<Is...>(layout.layout_b()));
  } else {
    return select<Is...>(layout);
  }
}

template <template <class MmaAtom, class AtomMNK, class DefaultPermutation> class MmaPrimitive,
          class MmaAtom,
          class AtomMNK,
          class DefaultPermutation,
          class PermutationMNK>
MUTE_HOST_DEVICE constexpr auto convert_to_permuted_sqmma(
    MmaPrimitive<MmaAtom, AtomMNK, DefaultPermutation> const& /*mma*/, PermutationMNK const& /*perm*/) {
  return TiledMMA<MmaAtom, AtomMNK, PermutationMNK>{};
}

}  // namespace mate
