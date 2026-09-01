#pragma once

#include "mute/tensor.hpp"
#include "mute/atom/mma_atom.hpp"

namespace mutlass::fmha::collective {

using namespace mute;

template <class Threshold, class Source, class Reference>
MUTE_HOST_DEVICE constexpr
auto
layout_separate(Threshold const& thr, Source const& src, Reference const& ref) {
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
MUTE_HOST_DEVICE constexpr
auto
layout_acc_mn(TiledMma const& tiled_mma, Acc const& acc) {
  auto [V_M, V_N] = layout_separate(get<0>(typename TiledMma::Shape_MNK{}),
                                   get<0>(acc),
                                   stride<1>(typename TiledMma::LayoutC_TV{}));
  return make_layout(make_layout(V_M, get<1>(acc)), make_layout(V_N, get<2>(acc)));
}

template <class TiledMma>
MUTE_HOST_DEVICE constexpr
auto
reduction_target_n(TiledMma const& tiled_mma) {
  auto separated = layout_separate(get<0>(typename TiledMma::Shape_MNK{}),
            make_layout(shape<0>(typename TiledMma::LayoutC_TV{})),
            stride<0>(typename TiledMma::LayoutC_TV{}));
  return get<1>(separated);
}

template <
  template<class MmaAtom, class AtomMNK, class DefaultPermutation> class MmaPrimtive,
  class MmaAtom, class AtomMNK, class DefaultPermutation,
  class PermutationMNK
>
MUTE_HOST_DEVICE constexpr
auto
convert_to_permuted_sqmma(MmaPrimtive<MmaAtom, AtomMNK, DefaultPermutation> const& prim,
                          PermutationMNK const& perm) {
  return TiledMMA<MmaAtom, AtomMNK, PermutationMNK>{};
}

template <
  template<class MmaAtom, class AtomMNK, class DefaultPermutation> class MmaPrimtive,
  class MmaAtom, class AtomMNK, class Permutation
>
MUTE_HOST_DEVICE constexpr
auto
convert_to_atom_sqmma(MmaPrimtive<MmaAtom, AtomMNK, Permutation> const& prim) {
  return make_tiled_mma(MmaAtom{});
}


template <class Layout, class Stages = _1>
MUTE_HOST_DEVICE constexpr
auto
unstageSmemLayout(Layout const& layout, Stages stages = {}) {
  return composition(layout, make_tuple(_, _, make_layout(stages)));
}

struct VariableLength {
  int total_length = -1;
  int max_length;
  int* cumulative_length = nullptr;

  MUTE_HOST_DEVICE operator int() const {
    return max_length;
  }
};


template<class T> struct is_variable_length_impl : std::false_type {};
template<> struct is_variable_length_impl<VariableLength> : std::true_type {};
template<class T> constexpr bool is_variable_length_v = is_variable_length_impl<remove_cvref_t<T>>::value;

template<class Shape, class Idx>
MUTE_HOST_DEVICE
constexpr auto
apply_variable_length(Shape const& shape, Idx const& idx) {
  return transform_leaf(shape, [&](auto const& s) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return s.cumulative_length[idx+1] - s.cumulative_length[idx];
    }
    else {
      return s;
    }
  });
}

template<class Shape, class Coord, class Idx>
MUTE_HOST_DEVICE
constexpr auto
apply_variable_length(Shape const& shape, Coord const& coord, Idx const& idx) {
  auto new_shape = apply_variable_length(shape, idx);
  auto new_coord = transform_leaf(shape, coord, [&](auto const& s, auto const& c) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return mute::make_tuple(c, s.cumulative_length[idx]);
    }
    else {
      return c;
    }
  });
  return mute::make_tuple(new_shape, new_coord);
}

template<class Shape, class Idx>
MUTE_HOST_DEVICE
constexpr auto
apply_variable_length_offset(Shape const& shape, Idx const& idx) {
  auto result_shape = transform_leaf(shape, [&](auto const& s) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return s.cumulative_length[idx+1] - s.cumulative_length[idx];
    }
    else {
      return s;
    }
  });
  auto result_offset = transform_leaf(shape, [&](auto const& s) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return s.cumulative_length[idx];
    }
    else {
      return _0{};
    }
  });
  return mute::make_tuple(result_shape, result_offset);
}

} // namespace mutlass::fmha::collective

namespace mute {

template <>
struct is_integral<mutlass::fmha::collective::VariableLength> : true_type {};

MUTE_HOST_DEVICE
void print(mutlass::fmha::collective::VariableLength _) {
  printf("Varlen<%d(total) o %d(max) o %p>", _.total_length, _.max_length, _.cumulative_length);
}

} // namespace mute
