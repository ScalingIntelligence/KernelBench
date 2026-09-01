#pragma once

#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/arch/mma_mp31.hpp>
#include <mute/atom/mma_atom.hpp>
#include <mutlass/detail/layout.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

namespace mate::flat::collective {

template <class Element,
          class GmemLayoutA,
          class GmemLayoutB,
          class TileShapeMNK,
          class MaxInstructionM,
          class MaxInstructionN,
          class AtomLayout>
struct Mp31TmeSqmmaCollective {
  using ElementAMma = mute::conditional_t<mute::is_same_v<Element, float>, tfloat32_t, Element>;
  using ElementBMma = ElementAMma;

  static constexpr mute::TCE::Major SqmmaMajorA =
      mutlass::gemm::collective::detail::sqmma_ss_tag_to_major_A<GmemLayoutA>();
  static constexpr mute::TCE::Major SqmmaMajorB =
      mutlass::gemm::collective::detail::sqmma_ss_tag_to_major_B<GmemLayoutB>();

  using SqmmaOp  = decltype(mute::MP31::SQMMA::ss_op_selector<Element,
                                                              Element,
                                                              float,
                                                              TileShapeMNK,
                                                              SqmmaMajorA,
                                                              SqmmaMajorB,
                                                              MaxInstructionM,
                                                              MaxInstructionN>());
  using TiledMma = decltype(mute::make_tiled_mma(SqmmaOp{}, AtomLayout{}));

  using SmemLayoutAtomA = decltype(mutlass::gemm::collective::detail::
                                       ss_smem_selector_A<SqmmaMajorA, ElementAMma, SqmmaOp, TileShapeMNK>());
  using SmemLayoutAtomB = decltype(mutlass::gemm::collective::detail::
                                       ss_smem_selector_B<SqmmaMajorB, ElementBMma, SqmmaOp, TileShapeMNK>());

  using CollectiveOp = mutlass::gemm::collective::CollectiveMma<mutlass::gemm::MainloopMp31TmeSqmmaWarpSpecialized<2>,
                                                                TileShapeMNK,
                                                                Element,
                                                                mutlass::detail::TagToStrideA_t<GmemLayoutA>,
                                                                Element,
                                                                mutlass::detail::TagToStrideB_t<GmemLayoutB>,
                                                                TiledMma,
                                                                MP31_TME_LOAD,
                                                                SmemLayoutAtomA,
                                                                void,
                                                                mute::identity,
                                                                MP31_TME_LOAD,
                                                                SmemLayoutAtomB,
                                                                void,
                                                                mute::identity>;
};

}  // namespace mate::flat::collective
