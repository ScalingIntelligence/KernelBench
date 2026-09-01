/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "mutlass/mutlass.h"
#include "mutlass/detail/dependent_false.hpp"
#include "mutlass/arch/barrier.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Circular Buffer Index + Associated Phase
// Assumes only one operation possible - i.e., ++
template <uint32_t Stages_>
struct PipelineState {

  static constexpr uint32_t Stages = Stages_;

  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;

  MUTLASS_DEVICE
  PipelineState(): index_{}, phase_{}, count_{} {}

  MUTLASS_DEVICE
  PipelineState(int index, uint32_t phase, uint32_t count)
    : index_(index)
    , phase_(phase)
    , count_(count) {}

  MUTLASS_DEVICE
  int index() const {
    return index_;
  }

  MUTLASS_DEVICE
  uint32_t phase() const {
    return phase_;
  }

  MUTLASS_DEVICE
  uint32_t count() const {
    return count_;
  }

  MUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }

  MUTLASS_DEVICE
  PipelineState& operator+=(uint32_t num_iterations) {
    return advance(num_iterations);
  }

  MUTLASS_DEVICE
  PipelineState& operator=(PipelineState const& other) {
    index_ = other.index();
    phase_ = other.phase();
    count_ = other.count();
    return *this;
  }

  MUTLASS_DEVICE
  PipelineState& advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      // Number of iterations cross over the stage boundary => flipped phase
      if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages ) {
        phase_ ^= 1;
      }
      // How many times number of iterations cross over the stage boundary and
      // end up on a odd number => flipped phase
      if ((num_iterations >= Stages) && (((index_ + num_iterations) / Stages) % 2) == 1) {
        phase_ ^= 1;
      }
      index_ = (index_ + num_iterations) % Stages;
      count_ += num_iterations;
    }
    return *this;
  }

  MUTLASS_DEVICE
  static PipelineState make_pipeline_state(PipelineState start_state, uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
};

template <class Pipeline>
MUTLASS_DEVICE
PipelineState<Pipeline::Stages> make_producer_start_state() {
  // Producer starts with an opposite phase as the buffers are initially empty
  constexpr int InitialProducerStage = 0;
  constexpr uint32_t InitialProducerPhase = 1;
  constexpr uint32_t InitialProducerCount = 0;
  return {InitialProducerStage, InitialProducerPhase, InitialProducerCount};
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Mp31 TME load (producer) Async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <int Stages_>
class Mp31PipelineTmeAsync {
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
  Mp31PipelineTmeAsync(Params params, uint32_t barrier_base = 0, uint32_t init_warps = 1)
    : params_(params)
    , barrier_base_(barrier_base + FullBarrier::ReservedAsyncBarrierCount)
  {
    int warp_idx = canonical_warp_idx();

    if (warp_idx < init_warps) {
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
  void producer_expect_transaction(PipelineState state, uint32_t transaction_bytes) {
    producer_expect_transaction(state.index(), transaction_bytes);
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
  void producer_expect_transaction(uint32_t stage, uint32_t transaction_bytes) {
    uint32_t full_barrier_id = barrier_base_ + stage;
    FullBarrier::expect_transaction(full_barrier_id, transaction_bytes);
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

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simple producer-consumer async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages_>
class Mp31PipelineAsync {
public:
  using FullBarrier = mutlass::arch::AsyncBarrier;
  using EmptyBarrier = mutlass::arch::AsyncBarrier;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = mutlass::PipelineState<Stages>;
  static_assert(FullBarrier::ReservedAsyncBarrierCount == EmptyBarrier::ReservedAsyncBarrierCount &&
                FullBarrier::ReservedAsyncBarrierCount == 1);

  static constexpr uint32_t NumBarriers = 2 * Stages;

  struct Params {
    uint32_t producer_arv_count = 1;
    uint32_t consumer_arv_count = 1;
  };

  // Constructor
  MUTLASS_DEVICE
  Mp31PipelineAsync(Params params, uint32_t barrier_base = 0)
    : params_(params)
    , barrier_base_(barrier_base + FullBarrier::ReservedAsyncBarrierCount)
  {
    int warp_idx = canonical_warp_idx();

    if (warp_idx == 0) {
      // Init full barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        FullBarrier::init(barrier_base_ + i, params_.producer_arv_count, 0);
      }

      // Init empty barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        EmptyBarrier::init(barrier_base_ + i + Stages, params_.consumer_arv_count, 0);
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
  void producer_commit(PipelineState state) {
    producer_commit(state.index());
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
  }

  MUTLASS_DEVICE
  void producer_commit(uint32_t stage) {
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

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Barrier to ensure an Ordered Sequence between
// SequenceLength number of groups (each with group_size participants) executing SequenceDepth Stages
// i.e., for all i < j - only after id "i" arrives at a particular stage "m"
// will the wait() for id "j" succeed for the same stage
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int SequenceDepth, int SequenceLength>
class OrderedSequenceBarrier {
public :
  using Barrier = mutlass::arch::AsyncTransactionBarrier;
  static constexpr int NumBarriers = SequenceDepth * SequenceLength;

  struct Params {
    uint32_t group_id;
    uint32_t group_size;
  };

private:
  Params params_;
  PipelineState<SequenceDepth> stage_;
  uint32_t barrier_base_;

  static constexpr int Depth = SequenceDepth;
  static constexpr int Length = SequenceLength;

public:
  OrderedSequenceBarrier() = delete;
  OrderedSequenceBarrier(const OrderedSequenceBarrier&) = delete;
  OrderedSequenceBarrier(OrderedSequenceBarrier&&) = delete;
  OrderedSequenceBarrier& operator=(const OrderedSequenceBarrier&) = delete;
  OrderedSequenceBarrier& operator=(OrderedSequenceBarrier&&) = delete;
  ~OrderedSequenceBarrier() = default;

  MUTLASS_DEVICE
  OrderedSequenceBarrier(Params const& params, uint32_t barrier_base = 0)
      : params_(params)
      , stage_({0, params.group_id == 0, 0})
      , barrier_base_(barrier_base + Barrier::ReservedAsyncBarrierCount)
  {
    int warp_idx = canonical_warp_idx();
    if (warp_idx == 0) {
      MUTLASS_PRAGMA_UNROLL
      for (int d = 0; d < Depth; ++d) {
        MUTLASS_PRAGMA_UNROLL
        for (int l = 0; l < Length; ++l) {
          uint32_t barrier_id = d * Length + l + barrier_base_;
          Barrier::init(barrier_id, params.group_size, 0);
        }
      }
    }
  }

  // Wait on a stage to be unlocked
  MUTLASS_DEVICE
  void wait() {
    uint32_t barrier_id = get_barrier_for_current_stage(params_.group_id);
    Barrier::wait(barrier_id, stage_.phase());
  }

  // Signal completion of Stage and move to the next stage
  // (group_id) signals to (group_id+1)
  MUTLASS_DEVICE
  void arrive() {
    int signalling_id = (params_.group_id + 1) % Length;
    uint32_t barrier_id = get_barrier_for_current_stage(signalling_id);
    Barrier::arrive</* return_phase = */ false>(barrier_id);
    ++stage_;
  }

  MUTLASS_DEVICE
  void advance() {
    ++stage_;
  }

private:
  MUTLASS_DEVICE
  uint32_t get_barrier_for_current_stage(int group_id) {
    return stage_.index() * Length + group_id + barrier_base_;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace mutlass

