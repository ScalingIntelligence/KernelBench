#pragma once

#include "mate/mega_moe/mega_moe_layout.hpp"
#include "mate/mega_moe/mega_moe_sym_buffer.hpp"

namespace mega_moe {

template <uint32_t kNumMPs, uint32_t kGridSyncIndex = 0, typename sync_scope_t>
__device__ void grid_sync(const layout::Workspace& workspace,
                          const uint32_t&          mp_idx,
                          const uint32_t&          thread_idx,
                          const sync_scope_t&      sync_scope) {
  // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
  static constexpr uint32_t kFinishSumTag = 0x80000000u;
  sync_scope();
  if (thread_idx == 0) {
    const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
    const auto old_value = atomic_add_global_u32(count_ptr, mp_idx == 0 ? (kFinishSumTag - (kNumMPs - 1)) : 1);
    uint32_t   new_value;
    do {
      new_value = ld_volatile_global(count_ptr);
    } while (((new_value ^ old_value) & kFinishSumTag) == 0);
  }
  sync_scope();
}

template <uint32_t kNumRanks,
          uint32_t kNumMPs,
          uint32_t kNumThreads,
          uint32_t kGridSyncIndex,
          uint32_t kTag,
          bool     kUseNoflushFence = false,
          typename sync_scope_t>
__device__ void mtlink_barrier(const layout::Workspace&            workspace,
                               const layout::SymBuffer<kNumRanks>& sym_buffer,
                               const uint32_t&                     mp_idx,
                               const uint32_t&                     thread_idx,
                               const sync_scope_t&                 sync_scope,
                               const bool&                         sync_prologue = true,
                               const bool&                         sync_epilogue = true) {
  static_assert(kNumRanks <= kNumThreads, "Insufficient threads");

  if (thread_idx == 0) {
    if constexpr (kUseNoflushFence) {
      __threadfence_system_noflush();
    } else {
      __threadfence_system();
    }
  }

  // Grid sync before MTLink signaling
  if (sync_prologue) grid_sync<kNumMPs, kGridSyncIndex>(workspace, mp_idx, thread_idx, sync_scope);

  // MTLink cross-rank barrier, only MP 0 participates
  if (mp_idx == 0) {
    auto*      counter_ptr  = workspace.get_mtlink_barrier_counter_ptr();
    const auto status       = ld_volatile_global(counter_ptr) & 3;
    const auto signal_phase = status & 1, signal_sign = status >> 1;
    auto*      signal_ptr = workspace.get_mtlink_barrier_signal_ptr(signal_phase);

    // Send signals to remote ranks
    if (thread_idx < kNumRanks) atomicAdd(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
    sync_scope();

    // Update status and wait arrival
    constexpr int64_t kNumTimeoutCycles = 30ll * 2000000000ll;
    if (thread_idx == 0) {
      atomic_add_global_u32(counter_ptr, 1);
      const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
      while (ld_volatile_global(signal_ptr) != target) {
      }
    }
  }

  // Grid sync after MTLink completion
  if (sync_epilogue) grid_sync<kNumMPs, kGridSyncIndex>(workspace, mp_idx, thread_idx, sync_scope);
}

}  // namespace mega_moe
