#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>

#include "mate/mega_moe/mega_moe_layout.hpp"
#include "mate/mega_moe/utils.hpp"

namespace mega_moe::mate_stage2 {

template <uint32_t kNumExpertsPerRank,
          uint32_t kNumNBlocks,
          uint32_t kNumMPs,
          uint32_t kNumRanks,
          uint32_t BLOCK_M,
          uint32_t kNumExpertsPerLane =
              math::constexpr_ceil_div(kNumExpertsPerRank, uint32_t(mutlass::NumThreadsPerWarp))>
struct WorkspaceTileScheduler {
  uint32_t block_idx                                        = 0;
  uint32_t current_local_expert_idx                         = 0;
  uint32_t current_num_tokens                               = 0;
  uint32_t current_pool_block_offset                        = 0;
  uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

  struct WorkTileInfo {
    uint32_t expert_idx     = 0;
    uint32_t m_block_idx    = 0;
    uint32_t n_block_idx    = 0;
    uint32_t pool_block_idx = 0;
    uint32_t valid_m        = 0;
    bool     valid          = false;
  };

  __device__ __forceinline__ static uint32_t lane_idx() {
    return static_cast<uint32_t>(mutlass::canonical_lane_idx());
  }

  __device__ __forceinline__ static uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
  }

  __device__ __forceinline__ void fetch_counts(const layout::Workspace& workspace) {
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t expert_idx = i * mutlass::NumThreadsPerWarp + lane_idx();
      uint64_t       value      = 0;
      if constexpr (kNumExpertsPerRank == mutlass::NumThreadsPerWarp) {
        value = ld_volatile_global(workspace.get_expert_recv_count_sum_ptr(expert_idx));
      } else if (expert_idx < kNumExpertsPerRank) {
        value = ld_volatile_global(workspace.get_expert_recv_count_sum_ptr(expert_idx));
      }
      stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
    }
  }

  __device__ __forceinline__ uint32_t get_num_tokens(uint32_t expert_idx) const {
    if constexpr (kNumExpertsPerLane == 1) {
      return __shfl_sync(
          0xffffffffu, stored_num_tokens_per_expert[0], static_cast<int>(expert_idx % mutlass::NumThreadsPerWarp));
    }

    uint32_t value = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t lane_expert_idx = i * mutlass::NumThreadsPerWarp + lane_idx();
      if (expert_idx == lane_expert_idx) {
        value = stored_num_tokens_per_expert[i];
      }
    }
    return __shfl_sync(0xffffffffu, value, static_cast<int>(expert_idx % mutlass::NumThreadsPerWarp));
  }

  __device__ __forceinline__ uint32_t get_current_num_m_blocks() const {
    return ceil_div_u32(current_num_tokens, BLOCK_M);
  }

  __device__ __forceinline__ uint32_t get_pool_block_offset(uint32_t expert_idx) const {
    uint32_t num_blocks = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t lane_expert_idx = i * mutlass::NumThreadsPerWarp + lane_idx();
      if (lane_expert_idx < expert_idx) {
        num_blocks += ceil_div_u32(stored_num_tokens_per_expert[i], BLOCK_M);
      }
    }
    return warp_reduce_add_u32(num_blocks);
  }

  __device__ __forceinline__ void set_expert_idx(uint32_t expert_idx) {
    current_local_expert_idx = expert_idx;
    current_num_tokens       = expert_idx < kNumExpertsPerRank ? get_num_tokens(expert_idx) : 0u;
  }

  __device__ __forceinline__ void advance_expert_idx() {
    current_pool_block_offset += get_current_num_m_blocks();
    set_expert_idx(current_local_expert_idx + 1);
  }

  __device__ __forceinline__ uint32_t get_valid_m(uint32_t m_block_idx) const {
    const uint32_t consumed  = m_block_idx * BLOCK_M;
    const uint32_t remaining = current_num_tokens - consumed;
    return remaining < BLOCK_M ? remaining : BLOCK_M;
  }

  __device__ __forceinline__ WorkTileInfo get_work_tile_info() {
    while (current_local_expert_idx < kNumExpertsPerRank) {
      const uint32_t num_m_blocks = get_current_num_m_blocks();
      const uint32_t m_block_idx  = block_idx / kNumNBlocks;
      if (m_block_idx < num_m_blocks) {
        const uint32_t n_block_idx = block_idx - m_block_idx * kNumNBlocks;
        return {current_local_expert_idx,
                m_block_idx,
                n_block_idx,
                current_pool_block_offset + m_block_idx,
                get_valid_m(m_block_idx),
                true};
      }

      block_idx -= num_m_blocks * kNumNBlocks;
      advance_expert_idx();
    }
    return {};
  }

  __device__ __forceinline__ WorkTileInfo initial_work_tile_info(const layout::Workspace& workspace) {
    block_idx                 = blockIdx.x;
    current_pool_block_offset = 0;
    fetch_counts(workspace);
    set_expert_idx(0);
    return get_work_tile_info();
  }

  __device__ __forceinline__ void advance_to_next_work() {
    block_idx += kNumMPs;
  }
};

}  // namespace mega_moe::mate_stage2
