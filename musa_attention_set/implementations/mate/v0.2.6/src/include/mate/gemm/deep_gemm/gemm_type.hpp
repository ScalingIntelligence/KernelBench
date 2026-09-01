#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>

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

enum class EpilogueType : int {
  Gemm       = 0,
  HeadSplits = 1,
};

struct EpilogueGemm {
  MUTLASS_HOST_DEVICE static uint32_t apply_index_n(uint32_t n_idx) {
    return n_idx;
  }
};

template <uint32_t kLeft, uint32_t kMid, uint32_t kRight>
struct EpilogueHeadSplits : EpilogueGemm {
  static_assert(kLeft > 0, "EpilogueHeadSplits left split must be positive");
  static_assert(kRight > 0, "EpilogueHeadSplits right split must be positive");

  static constexpr uint32_t Left        = kLeft;
  static constexpr uint32_t Mid         = kMid;
  static constexpr uint32_t Right       = kRight;
  static constexpr uint32_t CompactHead = kLeft + kRight;

  MUTLASS_HOST_DEVICE static uint32_t apply_index_n(uint32_t n_idx) {
    return n_idx + (n_idx + kRight) / (kLeft + kRight) * kMid;
  }
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

template <class T>
struct is_epilogue_head_splits : std::false_type {};

template <uint32_t Left, uint32_t Mid, uint32_t Right>
struct is_epilogue_head_splits<EpilogueHeadSplits<Left, Mid, Right>> : std::true_type {};

template <class T>
inline constexpr bool is_epilogue_head_splits_v = is_epilogue_head_splits<T>::value;

}  // namespace detail

}  // namespace mate::deep_gemm
