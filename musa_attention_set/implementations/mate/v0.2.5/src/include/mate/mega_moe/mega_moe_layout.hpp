#pragma once

#include "mate/mega_moe/utils.hpp"
#include "mate_utils.hpp"

namespace mega_moe::layout {

template <typename T>
constexpr T get_num_max_pool_tokens(
    T num_ranks, T num_max_tokens_per_rank, T num_topk, T num_experts_per_rank, T block_m) {
  const auto num_max_recv_tokens       = num_ranks * num_max_tokens_per_rank;
  const auto num_max_experts_per_token = std::min(num_topk, num_experts_per_rank);
  return math::constexpr_align(num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (block_m - 1),
                               block_m);
}

template <typename T>
constexpr T get_num_padded_sf_pool_tokens(T num_max_pool_tokens, T block_m) {
  return (num_max_pool_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

constexpr uint64_t kBufferAlignmentBytes = 128;

template <typename dtype_t = void>
__host__ __device__ dtype_t* align_ptr(void* ptr, const uint64_t alignment = kBufferAlignmentBytes) {
  return reinterpret_cast<dtype_t*>(math::align(reinterpret_cast<uint64_t>(ptr), alignment));
}

struct TokenSrcMetadata {
  uint32_t rank_idx;
  uint32_t token_idx;
  uint32_t topk_idx;
};

struct Workspace {
  void*    base;
  uint32_t num_ranks, num_experts;
  uint32_t num_experts_per_rank;
  uint32_t num_max_tokens_per_rank;
  uint32_t num_max_recv_tokens_per_expert;

  uint32_t num_max_pool_tokens;
  uint32_t num_max_pool_blocks;

  static constexpr uint64_t kBarrierSlotBytes             = 64;
  static constexpr uint64_t kLogicBufferSlotBytes         = 64;
  static constexpr uint32_t kNumMaxGridSyncCounters       = 4;
  static constexpr uint32_t kNumMtlinkBarrierSignalPhases = 2;
  static constexpr uint64_t kNumBarrierSignalBytes =
      (kNumMaxGridSyncCounters + 1 + kNumMtlinkBarrierSignalPhases) * kBarrierSlotBytes;
  static_assert(sizeof(TokenSrcMetadata) <= kLogicBufferSlotBytes, "Workspace logical slots require 64B padding");

  __host__ __device__ Workspace(void*    base,
                                uint32_t num_ranks,
                                uint32_t num_experts,
                                uint32_t num_max_tokens_per_rank,
                                uint32_t num_topk,
                                uint32_t block_m)
      : base(base), num_ranks(num_ranks), num_experts(num_experts), num_max_tokens_per_rank(num_max_tokens_per_rank) {
    num_experts_per_rank           = num_experts / num_ranks;
    num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
    num_max_pool_tokens =
        get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank, block_m);
    num_max_pool_blocks = num_max_pool_tokens / block_m;
#if !defined(__MUSA_ARCH__)
    TVM_FFI_ICHECK_EQ(num_max_tokens_per_rank % block_m, 0) << "num_max_tokens_per_rank must be aligned to block_m";
#endif
  }

  __host__ __device__ uint64_t get_num_bytes() const {
    uint64_t num_bytes = 0;

    num_bytes += kNumBarrierSignalBytes;
    num_bytes += static_cast<uint64_t>(num_experts) * kLogicBufferSlotBytes * 2;
    num_bytes += static_cast<uint64_t>(num_experts_per_rank) * kLogicBufferSlotBytes;
    num_bytes += static_cast<uint64_t>(math::align(num_max_pool_blocks, 2u)) * kLogicBufferSlotBytes;
    num_bytes += static_cast<uint64_t>(num_max_pool_blocks) * kLogicBufferSlotBytes;
    num_bytes += get_num_src_token_topk_idx_slots() * kLogicBufferSlotBytes;
    num_bytes += static_cast<uint64_t>(num_max_pool_tokens) * kLogicBufferSlotBytes;

    num_bytes = math::align<uint64_t>(num_bytes, kBufferAlignmentBytes);
    return num_bytes;
  }

  __host__ __device__ void* get_end_ptr() const {
    return math::advance_ptr(base, get_num_bytes());
  }

  // Barrier words are separated by 64B slots to keep independent atomic
  // sync variables out of the same cache line.
  // slot [0..3]: 4 x `uint32_t` grid sync counters
  // slot [4]   : `uint32_t` MTLink barrier counter
  // slot [5..6]: 2 x `int` MTLink barrier signals (phase 0 and 1)

  template <uint32_t kIndex = 0>
  __device__ uint32_t* get_grid_sync_count_ptr() const {
    static_assert(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
    return math::advance_ptr<uint32_t>(base, kIndex * kBarrierSlotBytes);
  }

  __device__ uint32_t* get_mtlink_barrier_counter_ptr() const {
    return math::advance_ptr<uint32_t>(base, kNumMaxGridSyncCounters * kBarrierSlotBytes);
  }

  __device__ int* get_mtlink_barrier_signal_ptr(const uint32_t& phase) const {
    return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1 + phase) * kBarrierSlotBytes);
  }

  __device__ uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
    return get_logic_slot_ptr<uint64_t>(get_expert_send_count_offset(), expert_idx);
  }

  __device__ uint64_t* get_expert_recv_count_ptr(const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
    const uint64_t slot_idx = static_cast<uint64_t>(rank_idx) * num_experts_per_rank + expert_idx;
    return get_logic_slot_ptr<uint64_t>(get_expert_recv_count_offset(), slot_idx);
  }

  __device__ uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
    return get_logic_slot_ptr<uint64_t>(get_expert_recv_count_sum_offset(), expert_idx);
  }

  __device__ uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
    return get_logic_slot_ptr<uint32_t>(get_l1_arrival_count_offset(), pool_block_idx);
  }

  __device__ uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
    return get_logic_slot_ptr<uint64_t>(get_l2_arrival_mask_offset(), pool_block_idx);
  }

  // For dispatch pulling
  __device__ uint32_t* get_src_token_topk_idx_ptr(const uint32_t& expert_idx = 0,
                                                  const uint32_t& rank_idx   = 0,
                                                  const uint32_t& token_idx  = 0) const {
    const uint64_t slot_idx = static_cast<uint64_t>(expert_idx) * num_ranks * num_max_recv_tokens_per_expert +
                              static_cast<uint64_t>(rank_idx) * num_max_recv_tokens_per_expert + token_idx;
    return get_logic_slot_ptr<uint32_t>(get_src_token_topk_idx_offset(), slot_idx);
  }

  // For combine usages
  __device__ TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
    return get_logic_slot_ptr<TokenSrcMetadata>(get_token_src_metadata_offset(), pool_token_idx);
  }

 private:
  __host__ __device__ uint64_t get_num_src_token_topk_idx_slots() const {
    return static_cast<uint64_t>(num_experts_per_rank) * num_ranks * num_max_recv_tokens_per_expert;
  }

  __host__ __device__ uint64_t get_expert_send_count_offset() const {
    return kNumBarrierSignalBytes;
  }

  __host__ __device__ uint64_t get_expert_recv_count_offset() const {
    return get_expert_send_count_offset() + static_cast<uint64_t>(num_experts) * kLogicBufferSlotBytes;
  }

  __host__ __device__ uint64_t get_expert_recv_count_sum_offset() const {
    return get_expert_recv_count_offset() + static_cast<uint64_t>(num_experts) * kLogicBufferSlotBytes;
  }

  __host__ __device__ uint64_t get_l1_arrival_count_offset() const {
    return get_expert_recv_count_sum_offset() + static_cast<uint64_t>(num_experts_per_rank) * kLogicBufferSlotBytes;
  }

  __host__ __device__ uint64_t get_l2_arrival_mask_offset() const {
    return get_l1_arrival_count_offset() +
           static_cast<uint64_t>(math::align(num_max_pool_blocks, 2u)) * kLogicBufferSlotBytes;
  }

  __host__ __device__ uint64_t get_src_token_topk_idx_offset() const {
    return get_l2_arrival_mask_offset() + static_cast<uint64_t>(num_max_pool_blocks) * kLogicBufferSlotBytes;
  }

  __host__ __device__ uint64_t get_token_src_metadata_offset() const {
    return get_src_token_topk_idx_offset() + get_num_src_token_topk_idx_slots() * kLogicBufferSlotBytes;
  }

  template <typename T>
  __host__ __device__ T* get_logic_slot_ptr(const uint64_t offset, const uint64_t slot_idx) const {
    static_assert(sizeof(T) <= kLogicBufferSlotBytes, "Workspace logical slots require 64B padding");
    return math::advance_ptr<T>(base, offset + slot_idx * kLogicBufferSlotBytes);
  }
};

struct Data {
  uint32_t num_bytes;
  bool     require_tma_alignment;
  void*    base;

  __host__ __device__ constexpr explicit Data(const uint32_t& num_bytes,
                                              const bool&     require_tma_alignment = true,
                                              void*           base                  = nullptr)
      : num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
  }

  template <typename dtype_t = uint32_t>
  __host__ __device__ constexpr dtype_t get_num_bytes() const {
    return static_cast<dtype_t>(num_bytes);
  }

  template <typename dtype_t = void>
  __host__ __device__ dtype_t* get_base_ptr() const {
    return static_cast<dtype_t*>(base);
  }

  __host__ __device__ void set_base_ptr(void* ptr) {
    base = ptr;
  }
};

struct Buffer {
  Data     data_layout;
  uint32_t num_ranks;
  uint32_t num_max_tokens_per_rank;

  void* base;

  __host__ __device__ Buffer(const Data&     data_layout,
                             const uint32_t& num_ranks,
                             const uint32_t& max_num_tokens_per_rank,
                             void*           base = nullptr)
      : data_layout(data_layout),
        num_ranks(num_ranks),
        num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(align_ptr(base)) {
  }

  __host__ __device__ uint64_t get_num_bytes_per_rank() const {
    return math::align<uint64_t>(num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>(),
                                 kBufferAlignmentBytes);
  }

  __host__ __device__ uint64_t get_num_bytes() const {
    return get_num_bytes_per_rank() * num_ranks;
  }

  template <typename dtype_t = void>
  __host__ __device__ dtype_t* get_base_ptr() const {
    return static_cast<dtype_t*>(base);
  }

  __host__ __device__ void* get_end_ptr() const {
    return math::advance_ptr(base, get_num_bytes());
  }

  __host__ __device__ Buffer get_rank_buffer(const uint32_t& rank_idx) const {
    return {data_layout, 1, num_max_tokens_per_rank, math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)};
  }

  __host__ __device__ Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
    return Data(data_layout.num_bytes,
                data_layout.require_tma_alignment,
                math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx));
  }
};

struct SymmBufferLayoutInfo {
  uint64_t num_bytes;
  uint32_t num_max_pool_tokens;
  uint32_t num_padded_sf_pool_tokens;
  uint64_t input_token_offset;
  uint64_t input_sf_offset;
  uint64_t input_topk_idx_offset;
  uint64_t input_topk_weights_offset;
  uint64_t l1_token_offset;
  uint64_t l1_sf_offset;
  uint64_t l1_topk_weights_offset;
  uint64_t l2_token_offset;
  uint64_t l2_sf_offset;
  uint64_t combine_token_offset;
};

__host__ inline SymmBufferLayoutInfo get_symm_buffer_layout_info(const uint32_t num_ranks,
                                                                 const uint32_t num_experts,
                                                                 const uint32_t num_max_tokens_per_rank,
                                                                 const uint32_t num_topk,
                                                                 const uint32_t hidden,
                                                                 const uint32_t intermediate_hidden,
                                                                 const uint32_t block_m) {
  constexpr uint64_t kSyntheticBase = 1ull << 20;
  auto*              base           = reinterpret_cast<void*>(kSyntheticBase);
  const auto         offset_of      = [](const void* ptr) { return reinterpret_cast<uint64_t>(ptr) - kSyntheticBase; };

  const auto num_experts_per_rank = num_experts / num_ranks;
  const auto num_max_pool_tokens =
      get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank, block_m);
  const auto num_padded_sf_pool_tokens = get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m);

  const auto workspace         = Workspace(base, num_ranks, num_experts, num_max_tokens_per_rank, num_topk, block_m);
  const auto fp8_token_layout  = Data(hidden);
  const auto bf16_token_layout = Data(hidden * 2);
  const auto fp8_intermediate_token_layout = Data(intermediate_hidden);
  const auto fp8_sf_layout                 = Data(hidden / 32, false);
  const auto fp8_intermediate_sf_layout    = Data((intermediate_hidden / 128) * sizeof(float));
  const auto input_topk_idx_layout         = Data(num_topk * sizeof(int64_t), false);
  const auto input_topk_weights_layout     = Data(num_topk * sizeof(float), false);
  const auto l1_topk_weights_layout        = Data(sizeof(float), false);

  const auto input_token_buffer = Buffer(fp8_token_layout, 1, num_max_tokens_per_rank, workspace.get_end_ptr());
  const auto input_sf_buffer    = Buffer(fp8_sf_layout, 1, num_max_tokens_per_rank, input_token_buffer.get_end_ptr());
  const auto input_topk_idx_buffer =
      Buffer(input_topk_idx_layout, 1, num_max_tokens_per_rank, input_sf_buffer.get_end_ptr());
  const auto input_topk_weights_buffer =
      Buffer(input_topk_weights_layout, 1, num_max_tokens_per_rank, input_topk_idx_buffer.get_end_ptr());
  const auto l1_token_buffer =
      Buffer(fp8_token_layout, 1, num_max_pool_tokens, input_topk_weights_buffer.get_end_ptr());
  const auto l1_sf_buffer = Buffer(fp8_sf_layout, 1, num_padded_sf_pool_tokens, l1_token_buffer.get_end_ptr());
  const auto l1_topk_weights_buffer =
      Buffer(l1_topk_weights_layout, 1, num_max_pool_tokens, l1_sf_buffer.get_end_ptr());
  const auto l2_token_buffer =
      Buffer(fp8_intermediate_token_layout, 1, num_max_pool_tokens, l1_topk_weights_buffer.get_end_ptr());
  const auto l2_sf_buffer =
      Buffer(fp8_intermediate_sf_layout, 1, num_padded_sf_pool_tokens, l2_token_buffer.get_end_ptr());
  const auto combine_token_buffer =
      Buffer(bf16_token_layout, num_topk, num_max_tokens_per_rank, l2_sf_buffer.get_end_ptr());

  return {
      offset_of(combine_token_buffer.get_end_ptr()),
      num_max_pool_tokens,
      num_padded_sf_pool_tokens,
      offset_of(input_token_buffer.get_base_ptr()),
      offset_of(input_sf_buffer.get_base_ptr()),
      offset_of(input_topk_idx_buffer.get_base_ptr()),
      offset_of(input_topk_weights_buffer.get_base_ptr()),
      offset_of(l1_token_buffer.get_base_ptr()),
      offset_of(l1_sf_buffer.get_base_ptr()),
      offset_of(l1_topk_weights_buffer.get_base_ptr()),
      offset_of(l2_token_buffer.get_base_ptr()),
      offset_of(l2_sf_buffer.get_base_ptr()),
      offset_of(combine_token_buffer.get_base_ptr()),
  };
}

}  // namespace mega_moe::layout
