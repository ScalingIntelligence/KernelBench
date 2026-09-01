#pragma once

#include "mutlass/gemm/gemm.h"
#include "mute/atom/mma_traits_mp31_sqmma.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

//
// Some named constants
//
constexpr int tme_alignment_bytes = 4;
constexpr int mp31_smem_capacity_bytes = 196608;

template <class LayoutA>
constexpr mute::TCE::Major
sqmma_ss_tag_to_major_A() {
  if constexpr (mutlass::gemm::detail::is_mn_major_A<LayoutA>()) {
    return mute::TCE::Major::MN;
  }
  else {
    return mute::TCE::Major::K;
  }
}

template <class LayoutB>
constexpr mute::TCE::Major
sqmma_ss_tag_to_major_B() {
  if constexpr (mutlass::gemm::detail::is_mn_major_B<LayoutB>()) {
    return mute::TCE::Major::MN;
  }
  else {
    return mute::TCE::Major::K;
  }
}

template <mute::TCE::Major major, class ElementType, class SqmmaOp>
MUTE_HOST_DEVICE constexpr
auto
ss_smem_selector_A()
{
  using AtomOpTraits = MMA_Traits<SqmmaOp>;
  using AtomOpShape = typename AtomOpTraits::Shape_MNK;

  using AtomM = decltype(get<0>(AtomOpShape{}));
  using AtomK = decltype(get<2>(AtomOpShape{}));

  return mute::MP31::SQMMA::make_canonical_gemm_smem_atom_layout<ElementType, major, AtomM, AtomK>();
}

template <mute::TCE::Major major, class ElementType, class SqmmaOp>
MUTE_HOST_DEVICE constexpr
auto
ss_smem_selector_B()
{
  using AtomOpTraits = MMA_Traits<SqmmaOp>;
  using AtomOpShape = typename AtomOpTraits::Shape_MNK;

  using AtomN = decltype(get<1>(AtomOpShape{}));
  using AtomK = decltype(get<2>(AtomOpShape{}));

  return mute::MP31::SQMMA::make_canonical_gemm_smem_atom_layout<ElementType, major, AtomN, AtomK>();
}

template <class ElementA, int AlignmentA, class ElementB, int AlignmentB, int RequiredAlignment>
constexpr bool
is_aligned() {
  return ((mute::sizeof_bits_v<ElementA> * AlignmentA / 8) % RequiredAlignment == 0) &&
         ((mute::sizeof_bits_v<ElementB> * AlignmentB / 8) % RequiredAlignment == 0);
}

template <class ElementA, class ElementB>
constexpr bool
is_input_fp8() {
  return ((mute::is_same_v<ElementA, float_e4m3_t> || mute::is_same_v<ElementA, float_e5m2_t>) &&
          (mute::is_same_v<ElementB, float_e4m3_t> || mute::is_same_v<ElementB, float_e5m2_t>));
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
