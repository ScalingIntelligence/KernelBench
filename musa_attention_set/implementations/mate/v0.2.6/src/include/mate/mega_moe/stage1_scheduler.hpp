#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>

#include "mate/mega_moe/mega_moe_layout.hpp"
#include "mate/mega_moe/utils.hpp"

namespace mega_moe::stage1_sched {

struct L1TileDesc {
  enum : uint32_t {
    kValidIdx       = 0,
    kLocalExpertIdx = 1,
    kPoolBlockIdx   = 2,
    kNBlockIdx      = 3,
    kValidMIdx      = 4,
    kNumWords       = 5
  };

  uint32_t valid            = 0;
  uint32_t local_expert_idx = 0;
  uint32_t pool_block_idx   = 0;
  uint32_t n_block_idx      = 0;
  uint32_t valid_m          = 0;

  __device__ __forceinline__ static L1TileDesc make(const uint32_t local_expert_idx,
                                                    const uint32_t pool_block_idx,
                                                    const uint32_t n_block_idx,
                                                    const uint32_t valid_m) {
    L1TileDesc desc;
    desc.valid            = 1;
    desc.local_expert_idx = local_expert_idx;
    desc.pool_block_idx   = pool_block_idx;
    desc.n_block_idx      = n_block_idx;
    desc.valid_m          = valid_m;
    return desc;
  }

  __device__ __forceinline__ static L1TileDesc invalid() {
    return {};
  }

  __device__ __forceinline__ bool is_valid() const {
    return valid != 0;
  }

  __device__ __forceinline__ void store_to_smem(uint32_t* smem_desc) const {
    smem_desc[kLocalExpertIdx] = local_expert_idx;
    smem_desc[kPoolBlockIdx]   = pool_block_idx;
    smem_desc[kNBlockIdx]      = n_block_idx;
    smem_desc[kValidMIdx]      = valid_m;
    smem_desc[kValidIdx]       = valid;
  }

  __device__ __forceinline__ static L1TileDesc load_from_smem(const uint32_t* smem_desc) {
    L1TileDesc desc;
    desc.valid            = smem_desc[kValidIdx];
    desc.local_expert_idx = smem_desc[kLocalExpertIdx];
    desc.pool_block_idx   = smem_desc[kPoolBlockIdx];
    desc.n_block_idx      = smem_desc[kNBlockIdx];
    desc.valid_m          = smem_desc[kValidMIdx];
    return desc;
  }
};

template <uint32_t BLOCK_M,
          uint32_t BLOCK_N,
          uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N,
          uint32_t L1_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumMPs,
          uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane =
              math::constexpr_ceil_div(kNumExpertsPerRank, uint32_t(mutlass::NumThreadsPerWarp)),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K>
struct L1TileScheduler {
  static_assert(L1_SHAPE_N % BLOCK_N == 0, "Invalid L1 N shape");
  static_assert(L1_SHAPE_K % BLOCK_K == 0, "Invalid L1 K shape");
  static_assert(kNumExpertsPerRank % kNumExpertsPerWave == 0, "Invalid L1 expert wave config");
  static_assert(kNumMPs % 2 == 0, "Stage1 scheduler expects even MP count for 2-CTA cluster");
  static_assert(kNumL1BlockNs % 2 == 0, "Stage1 scheduler expects even L1 N block count");

  const layout::Workspace& workspace;
  uint32_t                 current_local_expert_idx                         = 0;
  uint32_t                 current_num_tokens                               = 0;
  uint32_t                 current_pool_block_offset                        = 0;
  uint32_t                 block_idx                                        = 0;
  uint32_t                 m_block_idx                                      = 0;
  uint32_t                 stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

  __device__ explicit L1TileScheduler(const layout::Workspace& workspace) : workspace(workspace) {
    block_idx = blockIdx.x;
  }

  __device__ __forceinline__ static uint32_t lane_idx() {
    return static_cast<uint32_t>(mutlass::canonical_lane_idx());
  }

  __device__ __forceinline__ uint32_t get_wave_expert_end_idx() const {
    return math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
  }

  __device__ __forceinline__ uint32_t get_num_tokens(const uint32_t expert_idx) const {
    if constexpr (kNumExpertsPerLane == 1) {
      return expert_idx < kNumExpertsPerRank ? __shfl_sync(0xffffffffu,
                                                           stored_num_tokens_per_expert[0],
                                                           static_cast<int>(expert_idx % mutlass::NumThreadsPerWarp))
                                             : 0u;
    }

    uint32_t value = 0;
    if (expert_idx < kNumExpertsPerRank) {
#pragma unroll
      for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
        const uint32_t lane_expert_idx = i * mutlass::NumThreadsPerWarp + lane_idx();
        if (expert_idx == lane_expert_idx) {
          value = stored_num_tokens_per_expert[i];
        }
      }
    }
    return __shfl_sync(0xffffffffu, value, static_cast<int>(expert_idx % mutlass::NumThreadsPerWarp));
  }

  __device__ __forceinline__ uint32_t get_pool_block_offset(const uint32_t expert_idx) const {
    uint32_t num_blocks = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t lane_expert_idx = i * mutlass::NumThreadsPerWarp + lane_idx();
      if (lane_expert_idx < expert_idx) {
        num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
      }
    }
    return warp_reduce_add_u32(num_blocks);
  }

  __device__ __forceinline__ uint32_t get_current_num_m_blocks() const {
    return math::ceil_div(current_num_tokens, BLOCK_M);
  }

  __device__ __forceinline__ void advance_expert_idx() {
    current_pool_block_offset += get_current_num_m_blocks();
    current_local_expert_idx += 1;
    current_num_tokens = get_num_tokens(current_local_expert_idx);
  }

  __device__ __forceinline__ void set_expert_idx(const uint32_t expert_idx) {
    current_local_expert_idx  = expert_idx;
    current_num_tokens        = get_num_tokens(expert_idx);
    current_pool_block_offset = get_pool_block_offset(expert_idx);
  }

  __device__ __forceinline__ uint32_t get_current_pool_block_offset() const {
    return current_pool_block_offset;
  }

  __device__ __forceinline__ uint32_t get_valid_m() const {
    const uint32_t consumed  = m_block_idx * BLOCK_M;
    const uint32_t remaining = current_num_tokens - consumed;
    return remaining < BLOCK_M ? remaining : BLOCK_M;
  }

  __device__ __forceinline__ bool fetch_next_l1_block() {
    const uint32_t wave_end_expert_idx = get_wave_expert_end_idx();
    while (current_local_expert_idx < wave_end_expert_idx) {
      const uint32_t num_m_blocks = get_current_num_m_blocks();
      m_block_idx                 = block_idx / kNumL1BlockNs;
      if (m_block_idx < num_m_blocks) {
        return true;
      }

      block_idx -= num_m_blocks * kNumL1BlockNs;
      advance_expert_idx();
    }
    return false;
  }

  __device__ __forceinline__ void fetch_expert_recv_count() {
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t expert_idx = i * mutlass::NumThreadsPerWarp + lane_idx();
      uint64_t       value      = 0;
      if (expert_idx < kNumExpertsPerRank) {
        do {
          value = ld_volatile_global(workspace.get_expert_recv_count_sum_ptr(expert_idx));
        } while (static_cast<uint32_t>(value >> 32) != kNumMPs * kNumRanks);
      }
      stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
    }
    __syncwarp();
  }

  __device__ __forceinline__ void store_expert_recv_count_to_shared(uint32_t* smem_counts) const {
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t expert_idx = i * mutlass::NumThreadsPerWarp + lane_idx();
      if (expert_idx < kNumExpertsPerRank) {
        smem_counts[expert_idx] = stored_num_tokens_per_expert[i];
      }
    }
    __syncwarp();
  }

  __device__ __forceinline__ void load_expert_recv_count_from_shared(uint32_t* smem_counts) {
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t expert_idx       = i * mutlass::NumThreadsPerWarp + lane_idx();
      stored_num_tokens_per_expert[i] = expert_idx < kNumExpertsPerRank ? smem_counts[expert_idx] : 0;
    }
    __syncwarp();
  }

  __device__ __forceinline__ void init_l1_dispatch_count_scheduler() {
    fetch_expert_recv_count();
    set_expert_idx(0);
  }

  __device__ __forceinline__ bool fetch_next_l1_tile_desc(L1TileDesc& desc) {
    while (current_local_expert_idx < kNumExpertsPerRank) {
      if (fetch_next_l1_block()) {
        const uint32_t current_m_block_idx = m_block_idx;
        const uint32_t current_n_block_idx = block_idx - current_m_block_idx * kNumL1BlockNs;
        const uint32_t pool_block_idx      = get_current_pool_block_offset() + current_m_block_idx;
        block_idx += kNumMPs;
        desc = L1TileDesc::make(current_local_expert_idx, pool_block_idx, current_n_block_idx, get_valid_m());
        return true;
      }
    }

    desc = L1TileDesc::invalid();
    return false;
  }

  __device__ __forceinline__ void schedule_next_arrived_l1_tile(const uint32_t compute_lane_idx, uint32_t* smem_desc) {
    L1TileDesc desc;
    const bool has_block = fetch_next_l1_tile_desc(desc);
    if (compute_lane_idx != 0) {
      return;
    }

    if (has_block) {
      const auto arrival_ptr = workspace.get_l1_arrival_count_ptr(desc.pool_block_idx);
      while (ld_volatile_global(arrival_ptr) != desc.valid_m) {
        __nanosleep(500);
      }
    }
    desc.store_to_smem(smem_desc);
  }
};

template <uint32_t BLOCK_M,
          uint32_t BLOCK_N,
          uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N,
          uint32_t L1_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumMPs,
          uint32_t kNumRanks>
using MegaMoEL1Scheduler = L1TileScheduler<BLOCK_M,
                                           BLOCK_N,
                                           BLOCK_K,
                                           L1_SHAPE_N,
                                           L1_SHAPE_K,
                                           kNumExpertsPerRank,
                                           kNumExpertsPerWave,
                                           kNumMPs,
                                           kNumRanks>;

}  // namespace mega_moe::stage1_sched
