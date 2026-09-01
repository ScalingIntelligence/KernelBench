#pragma once

#include "mutlass/arch/barrier.hpp"

namespace mate::attention::fmha {

MUTLASS_DEVICE
static void named_barrier_arrive(uint32_t barrier_id_) {
  uint32_t barrier_id = barrier_id_ + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount;
  mutlass::arch::AsyncBarrier::arrive(barrier_id);
}

MUTLASS_DEVICE
static void named_barrier_sync(uint32_t barrier_id_) {
  uint32_t barrier_id = barrier_id_ + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount;
  mutlass::arch::AsyncBarrier::sync(barrier_id);
}

enum class FwdNamedBarriers { QueryEmpty = 0, AppendKV = 1, BarrierKV = 2, RotaryQ = 3, NumFwdNamedBarriers = 4 };

}  // namespace mate::attention::fmha
