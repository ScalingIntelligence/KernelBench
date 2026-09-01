#pragma once

#include "mutlass/gemm/collective/builders/mp31_sqmma_common.inl"
#include "mutlass/gemm/dispatch_policy.hpp"

#include "mute/arch/mma_mp31.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::collective {

namespace detail {

template <class TileShape_MNK, class SqmmaOp>
constexpr auto
mp31_make_sqmma_tiled_mma() {
  constexpr int TileM = size<0>(TileShape_MNK{});
  constexpr int TileN = size<1>(TileShape_MNK{});

  constexpr int InstM = size<0>(typename MMA_Traits<SqmmaOp>::Shape_MNK{});
  constexpr int InstN = size<1>(typename MMA_Traits<SqmmaOp>::Shape_MNK{});

  constexpr int AtomM = TileM / InstM;
  constexpr int AtomN = TileN / InstN;

  static_assert(TileM % InstM == 0 && TileN % InstN == 0, "Invalid Tile shape");
  using AtomLayout = decltype(make_layout(Shape<Int<AtomM>, Int<AtomN>, Int<1>>{}, LayoutRight{}));

  return mute::make_tiled_mma(SqmmaOp{}, AtomLayout{});
}

}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(StageCount<stages> stage_count) {
  return stages;
}

template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(mute::Int<stages> stage_count) {
  return stages;
}

template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int carveout_bytes>
constexpr int
compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes> stage_count) {
  constexpr auto a_bits = mute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = mute::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes =
    mutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    mutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));

  constexpr int MaximumStages = (CapacityBytes - carveout_bytes) / stage_bytes;

  static_assert(MaximumStages >= 2, "Mainloop need at least 2-stage");
  // Currently we just use 2-stage pipeline
  return 2;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA_TME_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
  arch::Mp31,
  arch::OpClassTensorOp,
  ElementA,
  GmemLayoutATag,
  AlignmentA,
  ElementB,
  GmemLayoutBTag,
  AlignmentB,
  ElementAccumulator,
  TileShape_MNK,
  ClusterShape_MNK,
  StageCountType,
  KernelScheduleType,
  mute::enable_if_t<mute::is_same_v<KernelScheduleType, KernelTme>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);

  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tme_alignment_bytes>(),
      "Should meet TME alignment requirement\n");

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = mute::conditional_t<mute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = mute::conditional_t<mute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr mute::TCE::Major SqmmaMajorA = detail::sqmma_ss_tag_to_major_A<GmemLayoutATag>();
  static constexpr mute::TCE::Major SqmmaMajorB = detail::sqmma_ss_tag_to_major_B<GmemLayoutBTag>();

  using SqmmaOp = decltype(mute::MP31::SQMMA::ss_op_selector<ElementA, ElementB, ElementAccumulator, TileShape_MNK, SqmmaMajorA, SqmmaMajorB>());

  using TiledMma = decltype(detail::mp31_make_sqmma_tiled_mma<TileShape_MNK, SqmmaOp>());

  using GmemTiledCopyA = MP31_TME_LOAD;
  using GmemTiledCopyB = MP31_TME_LOAD;

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector_A<SqmmaMajorA, ElementAMma, SqmmaOp>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector_B<SqmmaMajorB, ElementBMma, SqmmaOp>());

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<detail::mp31_smem_capacity_bytes, ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = MainloopMp31TmeSqmma<PipelineStages>;

  using ClusterShape_MNK_ = Shape<_1, _1, _1>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      mute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      mute::identity
  >;
};

// SQMMA_TME_SS_GROUP_SCALING
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  int ScaleGranularityM,
  int ScaleGranularityN,
  int ScaleGranularityK
>
struct CollectiveBuilder<
  arch::Mp31,
  arch::OpClassTensorOp,
  ElementA,
  GmemLayoutATag,
  AlignmentA,
  ElementB,
  GmemLayoutBTag,
  AlignmentB,
  ElementAccumulator,
  TileShape_MNK,
  ClusterShape_MNK,
  StageCountType,
  KernelTmeGroupScaledAccum<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>
> {
  using KernelScheduleType = KernelTmeGroupScaledAccum<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;

  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);

  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tme_alignment_bytes>(),
      "Should meet TME alignment requirement\n");

  static constexpr bool IsFP8Input = detail::is_input_fp8<ElementA, ElementB>();

  static_assert(IsFP8Input, "KernelTmeGroupScaledAccum is only compatible with FP8 now.");

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = mute::conditional_t<mute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = mute::conditional_t<mute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr mute::TCE::Major SqmmaMajorA = detail::sqmma_ss_tag_to_major_A<GmemLayoutATag>();
  static constexpr mute::TCE::Major SqmmaMajorB = detail::sqmma_ss_tag_to_major_B<GmemLayoutBTag>();

  using MaxInstructionM = Int<64>;
  using SqmmaOp = decltype(mute::MP31::SQMMA::ss_op_selector<ElementA, ElementB, ElementAccumulator, TileShape_MNK, SqmmaMajorA, SqmmaMajorB, MaxInstructionM>());
  using TiledMma = decltype(detail::mp31_make_sqmma_tiled_mma<TileShape_MNK, SqmmaOp>());

  using GmemTiledCopyA = MP31_TME_LOAD;
  using GmemTiledCopyB = MP31_TME_LOAD;

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector_A<SqmmaMajorA, ElementAMma, SqmmaOp>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector_B<SqmmaMajorB, ElementBMma, SqmmaOp>());

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<detail::mp31_smem_capacity_bytes, ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});

  using DispatchPolicy = MainloopMp31TmeSqmmaBlockScalingFP8<PipelineStages, KernelScheduleType, ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      mute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      mute::identity
  >;
};

} // namespace mutlass::gemm::collective
