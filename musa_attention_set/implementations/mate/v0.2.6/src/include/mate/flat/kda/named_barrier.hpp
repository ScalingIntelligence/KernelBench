#pragma once

#include <cstdint>

#include "mutlass/arch/barrier.hpp"

namespace mate::flat::kda {

enum class KdaNamedBarrier : uint32_t {
  GCumsumReady     = 0,
  InverseReady     = 1,
  KRestoredReady   = 2,
  VUpdatedReady    = 3,
  StateCommitted   = 4,
  OperandsReady    = 5,
  OperandsConsumed = 6,
  PReady           = 7,
  DtBiasLoaded     = 8,
  BetaLoaded       = 9,
  InverseSmemReady = 10,
  VUpdatedConsumed = 11,
  StateConsumed    = 12,
  NumNamedBarriers = 13,
};

MUTLASS_DEVICE
static uint32_t named_barrier_id(uint32_t barrier_id_) {
  return barrier_id_ + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount;
}

MUTLASS_DEVICE
static uint32_t named_barrier_id(KdaNamedBarrier barrier_id_) {
  return named_barrier_id(static_cast<uint32_t>(barrier_id_));
}

MUTLASS_DEVICE
static void named_barrier_init(uint32_t barrier_id_, uint32_t arrive_count, uint32_t init_phase = 0) {
  mutlass::arch::AsyncBarrier::init(named_barrier_id(barrier_id_), arrive_count, init_phase);
}

MUTLASS_DEVICE
static void named_barrier_arrive(uint32_t barrier_id_) {
  mutlass::arch::AsyncBarrier::arrive(named_barrier_id(barrier_id_));
}

MUTLASS_DEVICE
static void named_barrier_arrive(KdaNamedBarrier barrier_id_) {
  named_barrier_arrive(static_cast<uint32_t>(barrier_id_));
}

MUTLASS_DEVICE
static uint32_t named_barrier_arrive_phase(uint32_t barrier_id_) {
  return mutlass::arch::AsyncBarrier::arrive<true>(named_barrier_id(barrier_id_));
}

MUTLASS_DEVICE
static uint32_t named_barrier_arrive_phase(KdaNamedBarrier barrier_id_) {
  return named_barrier_arrive_phase(static_cast<uint32_t>(barrier_id_));
}

MUTLASS_DEVICE
static void named_barrier_wait(uint32_t barrier_id_, uint32_t phase) {
  mutlass::arch::AsyncBarrier::wait(named_barrier_id(barrier_id_), phase);
}

MUTLASS_DEVICE
static void named_barrier_wait(KdaNamedBarrier barrier_id_, uint32_t phase) {
  named_barrier_wait(static_cast<uint32_t>(barrier_id_), phase);
}

MUTLASS_DEVICE
static void named_barrier_arrive_and_wait(KdaNamedBarrier barrier_id_) {
  uint32_t barrier_id = named_barrier_id(barrier_id_);
  uint32_t phase      = mutlass::arch::AsyncBarrier::arrive<true>(barrier_id);
  mutlass::arch::AsyncBarrier::wait(barrier_id, phase);
}

}  // namespace mate::flat::kda
