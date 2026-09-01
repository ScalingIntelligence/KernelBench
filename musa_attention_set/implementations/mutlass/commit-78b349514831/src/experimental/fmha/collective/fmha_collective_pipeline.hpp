#pragma once

#include "mutlass/pipeline/pipeline.hpp"

namespace mutlass {

template <int Stages_>
class Mp31PipelineTmeLoadKAsync {
public:
  using FullBarrier = mutlass::arch::AsyncTransactionBarrier;
  using EmptyBarrier = mutlass::arch::AsyncBarrier;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = mutlass::PipelineState<Stages>;
  static_assert(FullBarrier::ReservedAsyncBarrierCount == EmptyBarrier::ReservedAsyncBarrierCount &&
                FullBarrier::ReservedAsyncBarrierCount == 1);

  static constexpr uint32_t NumBarriers = 2 * Stages;

  struct Params {
    uint32_t transaction_bytes = 0;
    uint32_t num_consumers = 0;
    uint32_t num_producers = 1;
  };

  // Constructor
  MUTLASS_DEVICE
  Mp31PipelineTmeLoadKAsync(Params params, uint32_t barrier_base = 0)
    : params_(params)
    , barrier_base_(barrier_base + FullBarrier::ReservedAsyncBarrierCount)
  {
    int warp_idx = canonical_warp_idx();

    if (warp_idx == 0) {
      // Init full barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        FullBarrier::init(barrier_base_ + i, params_.num_producers, 0);
      }

      // Init empty barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        EmptyBarrier::init(barrier_base_ + i + Stages, params_.num_consumers, 0);
      }
    }
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  MUTLASS_DEVICE
  void producer_acquire(PipelineState state) {
    producer_acquire(state.index(), state.phase());
  }

  MUTLASS_DEVICE
  void lsu_producer_acquire(PipelineState state) {
    lsu_producer_acquire(state.index(), state.phase());
  }

  MUTLASS_DEVICE
  void lsu_producer_commit(PipelineState state) {
    lsu_producer_commit(state.index());
  }

  MUTLASS_DEVICE
  uint32_t producer_get_barrier_id(PipelineState state) {
    return producer_get_barrier_id(state.index());
  }

  ////////////////////
  // Consumers APIs
  ////////////////////
  MUTLASS_DEVICE
  void consumer_wait(PipelineState state) {
    consumer_wait(state.index(), state.phase());
  }

  MUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

private:
  Params params_;
  uint32_t barrier_base_;

  MUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase) {
    uint32_t empty_barrier_id = barrier_base_ + stage + Stages;
    EmptyBarrier::wait(empty_barrier_id, phase);

    uint32_t full_barrier_id = barrier_base_ + stage;
    FullBarrier::arrive_and_expect_tx(full_barrier_id, params_.transaction_bytes);
  }

  MUTLASS_DEVICE
  void lsu_producer_acquire(uint32_t stage, uint32_t phase) {
    uint32_t empty_barrier_id = barrier_base_ + stage + Stages;
    EmptyBarrier::wait(empty_barrier_id, phase);
  }

  MUTLASS_DEVICE
  void lsu_producer_commit(uint32_t stage) {
    uint32_t full_barrier_id = barrier_base_ + stage;
    FullBarrier::arrive(full_barrier_id);
  }

  MUTLASS_DEVICE
  uint32_t producer_get_barrier_id(uint32_t stage) {
    return barrier_base_ + stage;
  }

  MUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    uint32_t full_barrier_id = barrier_base_ + stage;
    FullBarrier::wait(full_barrier_id, phase);
  }

  MUTLASS_DEVICE
  void consumer_release(uint32_t stage) {
    uint32_t empty_barrier_id = barrier_base_ + stage + Stages;
    EmptyBarrier::arrive(empty_barrier_id);
  }
};

} // namespace mutlass
