#pragma once

namespace mate::deep_gemm {

enum class GemmType : int {
  Normal                           = 0,
  MGroupedContiguous               = 1,
  MGroupedMasked                   = 2,
  MGroupedContiguousWithPsumLayout = 3,
  Batched                          = 4,
};

enum class ScaleMode : int {
  Iterative  = 0,
  DualBuffer = 1,
};

namespace detail {

template <GemmType kType>
struct GemmTensorGroupTraits {
  // Whether each logical tensor is physically stored with a leading group
  // dimension. MGroupedContiguous keeps A/D flattened over M while B is grouped.
  static constexpr bool kA      = kType == GemmType::MGroupedMasked || kType == GemmType::Batched;
  static constexpr bool kB      = kType != GemmType::Normal;
  static constexpr bool kD      = kA;
  static constexpr bool kScaleA = kA;
  static constexpr bool kScaleB = kB;
};

}  // namespace detail

}  // namespace mate::deep_gemm
