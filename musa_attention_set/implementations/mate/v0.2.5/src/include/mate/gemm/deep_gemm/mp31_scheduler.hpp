#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <type_traits>

#include "mate/gemm/deep_gemm/gemm_type.hpp"

namespace mate::deep_gemm {

using namespace mute;

namespace detail {

template <GemmType kType>
static constexpr bool kIsNormal = (kType == GemmType::Normal);
template <GemmType kType>
static constexpr bool kIsContig = (kType == GemmType::MGroupedContiguous);
template <GemmType kType>
static constexpr bool kIsMasked = (kType == GemmType::MGroupedMasked);
template <GemmType kType>
static constexpr bool kIsPsum = (kType == GemmType::MGroupedContiguousWithPsumLayout);
template <GemmType kType>
static constexpr bool kIsBatched = (kType == GemmType::Batched);

template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNum1DBlocksPerGroup = 16>
class Mp31PersistentTileScheduler {
 public:
  struct WorkTileInfo {
    int32_t  M_idx         = -1;
    int32_t  N_idx         = -1;
    int32_t  G_idx         = 0;
    uint32_t m_row_offset  = 0;
    bool     is_valid_tile = false;
    bool     compute_valid = false;

    MUTLASS_HOST_DEVICE bool is_valid() const {
      return is_valid_tile;
    }
    MUTLASS_HOST_DEVICE bool is_compute_valid() const {
      return compute_valid;
    }
    MUTLASS_HOST_DEVICE static WorkTileInfo invalid_work_tile() {
      return {-1, -1, 0, 0, false, false};
    }
  };

  MUTLASS_DEVICE
  Mp31PersistentTileScheduler(uint32_t       shape_m,
                              uint32_t       shape_n,
                              uint32_t       num_groups_,
                              int32_t const* grouped_layout_,
                              uint32_t /*expected_m*/ = 0)
      : current_iter_(0),
        shape_m_(shape_m),
        num_m_blocks_(mutlass::ceil_div(shape_m, BLOCK_M)),
        num_n_blocks_(mutlass::ceil_div(shape_n, BLOCK_N)),
        num_blocks_(num_m_blocks_ * num_n_blocks_),
        num_groups_(num_groups_ ? num_groups_ : 1u),
        grouped_layout_(grouped_layout_),
        current_group_idx_(0),
        current_m_cumsum_(0),
        psum_last_m_(0),
        psum_current_m_(0),
        psum_m_block_cumsum_(0) {
    if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
      psum_current_m_ = static_cast<uint32_t>(__ldg(grouped_layout_));
      num_m_blocks_   = mutlass::ceil_div(psum_current_m_, BLOCK_M);
    }
  }

  MUTLASS_DEVICE WorkTileInfo initial_work_tile_info() {
    current_iter_      = 0;
    current_group_idx_ = 0;
    current_m_cumsum_  = 0;

    if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
      psum_last_m_         = 0;
      psum_current_m_      = uint32_t(__ldg(grouped_layout_));
      psum_m_block_cumsum_ = 0;
      num_m_blocks_        = mutlass::ceil_div(psum_current_m_, uint32_t(BLOCK_M));
    }

    return get_work_tile_info();
  }

  MUTLASS_DEVICE WorkTileInfo get_work_tile_info() {
    if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
      return get_psum_work();
    }

    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;
    if (!get_next_block(m_block_idx, n_block_idx)) {
      return WorkTileInfo::invalid_work_tile();
    }

    return make_work_tile(m_block_idx, n_block_idx);
  }

  MUTLASS_DEVICE void advance_to_next_work(uint32_t advance_count = 1) {
    current_iter_ += uint64_t(advance_count);
  }

 private:
  MUTLASS_DEVICE static bool is_power_of_two(uint32_t value) {
    return value != 0u && (value & (value - 1u)) == 0u;
  }

  MUTLASS_DEVICE static uint32_t log2_power_of_two(uint32_t value) {
    return 31u - uint32_t(__clz(value));
  }

  MUTLASS_DEVICE static uint32_t fast_divide(uint32_t dividend, uint32_t divisor) {
    return is_power_of_two(divisor) ? (dividend >> log2_power_of_two(divisor)) : (dividend / divisor);
  }

  MUTLASS_DEVICE static uint32_t fast_mod(uint32_t dividend, uint32_t divisor) {
    return is_power_of_two(divisor) ? (dividend & (divisor - 1u)) : (dividend % divisor);
  }

  MUTLASS_DEVICE static uint32_t valid_m_blocks(int32_t m) {
    return mutlass::ceil_div(m > 0 ? static_cast<uint32_t>(m) : 0u, uint32_t(BLOCK_M));
  }

  MUTLASS_DEVICE static uint32_t align_128(uint32_t value) {
    return (value + 127u) & ~127u;
  }

  MUTLASS_DEVICE void get_swizzled_block_idx(uint32_t  block_idx,
                                             uint32_t& m_block_idx,
                                             uint32_t& n_block_idx,
                                             uint32_t  primary_m_blocks) const {
    static_assert(kNum1DBlocksPerGroup > 0);

    if constexpr (kGemmType == GemmType::Normal || kGemmType == GemmType::MGroupedContiguous) {
      const uint32_t groups             = num_groups_ > 0u ? num_groups_ : 1u;
      const uint32_t m_blocks_per_group = fast_divide(primary_m_blocks, groups);
      if (m_blocks_per_group > 0u && primary_m_blocks == m_blocks_per_group * groups &&
          num_n_blocks_ % kNum1DBlocksPerGroup == 0u) {
        const uint32_t blocks_per_group   = m_blocks_per_group * num_n_blocks_;
        const uint32_t group_idx          = fast_divide(block_idx, blocks_per_group);
        const uint32_t in_group_idx       = block_idx - group_idx * blocks_per_group;
        const uint32_t blocks_per_n_group = m_blocks_per_group * kNum1DBlocksPerGroup;
        const uint32_t n_group_idx        = fast_divide(in_group_idx, blocks_per_n_group);
        const uint32_t in_macro_idx       = in_group_idx - n_group_idx * blocks_per_n_group;

        uint32_t m_in_group = in_macro_idx / kNum1DBlocksPerGroup;
        uint32_t n_offset   = in_macro_idx - m_in_group * kNum1DBlocksPerGroup;
        if (m_in_group % 2u == 1u) {
          n_offset = kNum1DBlocksPerGroup - 1u - n_offset;
        }
        if (n_group_idx % 2u == 1u) {
          m_in_group = m_blocks_per_group - 1u - m_in_group;
        }

        m_block_idx = group_idx * m_blocks_per_group + m_in_group;
        n_block_idx = n_group_idx * kNum1DBlocksPerGroup + n_offset;
        return;
      }
    }

    const uint32_t num_blocks_per_group = num_n_blocks_ * kNum1DBlocksPerGroup;
    const uint32_t group_idx            = fast_divide(block_idx, num_blocks_per_group);
    const uint32_t first_m_block        = group_idx * kNum1DBlocksPerGroup;
    const uint32_t in_group_idx         = block_idx - group_idx * num_blocks_per_group;
    const uint32_t remaining_m_blocks   = primary_m_blocks - first_m_block;
    const uint32_t m_blocks_in_group =
        remaining_m_blocks < kNum1DBlocksPerGroup ? remaining_m_blocks : kNum1DBlocksPerGroup;

    m_block_idx = first_m_block + fast_mod(in_group_idx, m_blocks_in_group);
    n_block_idx = fast_divide(in_group_idx, m_blocks_in_group);
  }

  MUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    const uint64_t next_block_idx = current_iter_ * uint64_t(gridDim.x) + uint64_t(blockIdx.x);

    if constexpr (kGemmType == GemmType::MGroupedMasked) {
      while (true) {
        if (current_group_idx_ >= num_groups_) return false;

        num_m_blocks_             = valid_m_blocks(__ldg(grouped_layout_ + current_group_idx_));
        const uint32_t cumsum_end = current_m_cumsum_ + num_m_blocks_;
        if (next_block_idx < uint64_t(cumsum_end) * uint64_t(num_n_blocks_)) break;

        current_group_idx_++;
        current_m_cumsum_ = cumsum_end;
      }

      const uint32_t in_group_block =
          static_cast<uint32_t>(next_block_idx - uint64_t(current_m_cumsum_) * uint64_t(num_n_blocks_));
      get_swizzled_block_idx(in_group_block, m_block_idx, n_block_idx, num_m_blocks_);
      return true;
    }

    if constexpr (kGemmType == GemmType::Batched) {
      if (next_block_idx >= uint64_t(num_blocks_) * uint64_t(num_groups_)) return false;

      current_group_idx_       = static_cast<uint32_t>(next_block_idx / num_blocks_);
      const uint32_t block_idx = static_cast<uint32_t>(next_block_idx - uint64_t(current_group_idx_) * num_blocks_);
      m_block_idx              = fast_mod(block_idx, num_m_blocks_);
      n_block_idx              = fast_divide(block_idx, num_m_blocks_);
      return true;
    }

    if constexpr (kGemmType == GemmType::Normal || kGemmType == GemmType::MGroupedContiguous) {
      if (next_block_idx >= num_blocks_) return false;

      get_swizzled_block_idx(static_cast<uint32_t>(next_block_idx), m_block_idx, n_block_idx, num_m_blocks_);
      return true;
    }

    return false;
  }

  MUTLASS_DEVICE WorkTileInfo make_work_tile(uint32_t m_block_idx, uint32_t n_block_idx) const {
    if (m_block_idx >= num_m_blocks_ || n_block_idx >= num_n_blocks_) {
      return WorkTileInfo::invalid_work_tile();
    }

    const uint32_t m_row_offset = m_block_idx * uint32_t(BLOCK_M);
    if (m_row_offset >= shape_m_) {
      return WorkTileInfo::invalid_work_tile();
    }

    if constexpr (kGemmType == GemmType::Normal) {
      return {int32_t(m_block_idx), int32_t(n_block_idx), 0, m_row_offset, true, true};
    }

    if constexpr (kGemmType == GemmType::MGroupedContiguous) {
      int32_t raw = __ldg(grouped_layout_ + m_row_offset);
      return {int32_t(m_block_idx), int32_t(n_block_idx), raw < 0 ? 0 : raw, m_row_offset, true, raw >= 0};
    }

    if constexpr (kGemmType == GemmType::MGroupedMasked || kGemmType == GemmType::Batched) {
      return {int32_t(m_block_idx), int32_t(n_block_idx), int32_t(current_group_idx_), m_row_offset, true, true};
    }

    return WorkTileInfo::invalid_work_tile();
  }

  MUTLASS_DEVICE WorkTileInfo get_psum_work() {
    const uint64_t next_block_idx = current_iter_ * uint64_t(gridDim.x) + uint64_t(blockIdx.x);

    while (true) {
      if (current_group_idx_ >= num_groups_) return WorkTileInfo::invalid_work_tile();

      num_m_blocks_             = mutlass::ceil_div(psum_current_m_ - psum_last_m_, uint32_t(BLOCK_M));
      const uint32_t cumsum_end = psum_m_block_cumsum_ + num_m_blocks_;
      if (next_block_idx < uint64_t(cumsum_end) * uint64_t(num_n_blocks_)) break;

      current_group_idx_++;
      if (current_group_idx_ >= num_groups_) return WorkTileInfo::invalid_work_tile();

      psum_m_block_cumsum_ = cumsum_end;
      psum_last_m_         = align_128(psum_current_m_);
      psum_current_m_      = static_cast<uint32_t>(__ldg(grouped_layout_ + current_group_idx_));
    }

    const uint32_t in_group_block =
        static_cast<uint32_t>(next_block_idx - uint64_t(psum_m_block_cumsum_) * uint64_t(num_n_blocks_));
    uint32_t inner_m_block = 0;
    uint32_t n_block_idx   = 0;
    get_swizzled_block_idx(in_group_block, inner_m_block, n_block_idx, num_m_blocks_);

    const uint32_t m_row_offset = psum_last_m_ + inner_m_block * uint32_t(BLOCK_M);
    return {int32_t(inner_m_block), int32_t(n_block_idx), int32_t(current_group_idx_), m_row_offset, true, true};
  }

 private:
  uint64_t current_iter_;
  uint32_t shape_m_;
  uint32_t num_m_blocks_;
  uint32_t num_n_blocks_;
  uint32_t num_blocks_;

  uint32_t       num_groups_;
  int32_t const* grouped_layout_;

  uint32_t current_group_idx_;
  uint32_t current_m_cumsum_;

  // MGroupedContiguousWithPsumLayout
  uint32_t psum_last_m_;
  uint32_t psum_current_m_;
  uint32_t psum_m_block_cumsum_;
};

}  // namespace detail
}  // namespace mate::deep_gemm
