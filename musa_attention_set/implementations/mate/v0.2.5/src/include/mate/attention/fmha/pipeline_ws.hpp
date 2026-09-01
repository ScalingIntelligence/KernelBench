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

#include "mutlass/arch/barrier.hpp"
#include "mutlass/detail/dependent_false.hpp"
#include "mutlass/mutlass.h"
#include "mutlass/pipeline/mp31_pipeline.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <uint32_t Stages_, uint32_t BarPerStageRatio_ = 2>
struct Mp31PipelineStateWarpSpecialized {
  static constexpr uint32_t Stages               = Stages_;
  static constexpr uint32_t BarPerStageRatio     = BarPerStageRatio_;
  static constexpr uint32_t BarrierRingSize      = Stages * BarPerStageRatio;
  static constexpr uint32_t InitialProducerPhase = 1;
  static_assert(BarPerStageRatio > 0);

  int      stage_index_   = 0;
  uint32_t barrier_index_ = 0;
  uint32_t phase_         = 0;
  uint32_t count_         = 0;

  MUTLASS_DEVICE
  Mp31PipelineStateWarpSpecialized() : stage_index_{}, barrier_index_{}, phase_{}, count_{} {
  }

  MUTLASS_DEVICE
  Mp31PipelineStateWarpSpecialized(int stage_index, uint32_t barrier_index, uint32_t phase, uint32_t count)
      : stage_index_(stage_index), barrier_index_(barrier_index), phase_(phase), count_(count) {
  }

  MUTLASS_DEVICE
  int index() const {
    return stage_index_;
  }

  MUTLASS_DEVICE
  uint32_t barrier_index() const {
    return barrier_index_;
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
      ++barrier_index_;
      ++count_;
      ++stage_index_;
      if (stage_index_ == static_cast<int>(Stages)) {
        stage_index_ = 0;
      }
      if (barrier_index_ == BarrierRingSize) {
        barrier_index_ = 0;
        phase_ ^= 1;
      }
    }
  }

  MUTLASS_DEVICE
  Mp31PipelineStateWarpSpecialized& operator+=(uint32_t num_iterations) {
    return advance(num_iterations);
  }

  MUTLASS_DEVICE
  Mp31PipelineStateWarpSpecialized& operator=(Mp31PipelineStateWarpSpecialized const& other) {
    stage_index_   = other.index();
    barrier_index_ = other.barrier_index();
    phase_         = other.phase();
    count_         = other.count();
    return *this;
  }

  MUTLASS_DEVICE
  Mp31PipelineStateWarpSpecialized& advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      uint32_t const next_barrier_index = barrier_index_ + num_iterations;
      uint32_t const next_stage_index   = static_cast<uint32_t>(stage_index_) + num_iterations;
      barrier_index_                    = next_barrier_index % BarrierRingSize;
      stage_index_                      = next_stage_index % Stages;
      phase_ ^= (next_barrier_index / BarrierRingSize) & 1;
      count_ += num_iterations;
    }
    return *this;
  }

  MUTLASS_DEVICE
  static Mp31PipelineStateWarpSpecialized make_pipeline_state(Mp31PipelineStateWarpSpecialized start_state,
                                                              uint32_t                         num_iterations) {
    return start_state.advance(num_iterations);
  }
};

template <class Pipeline>
MUTLASS_DEVICE typename Pipeline::PipelineState make_producer_start_state_warpspecialized() {
  constexpr int      InitialProducerStage   = 0;
  constexpr uint32_t InitialProducerBarrier = 0;
  constexpr uint32_t InitialProducerPhase   = 1;
  constexpr uint32_t InitialProducerCount   = 0;
  return {InitialProducerStage, InitialProducerBarrier, InitialProducerPhase, InitialProducerCount};
}

template <int MaxBarPerStageRatio_, int AdditionalBarrier_, int... PipelineStages_>
struct Mp31PipelineWarpSpecializedBarrierRatio {
  static constexpr int MaxBarPerStageRatio = MaxBarPerStageRatio_;
  static constexpr int AdditionalBarrier   = AdditionalBarrier_;
  static_assert(MaxBarPerStageRatio > 0);
  static_assert((MaxBarPerStageRatio & (MaxBarPerStageRatio - 1)) == 0);
  static_assert(AdditionalBarrier >= 0);

  static constexpr int PipelineStages = (PipelineStages_ + ... + 0);
  static constexpr int MaxAsyncBarriers =
      static_cast<int>(mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers);
  static constexpr int AvailablePipelineBarriers = MaxAsyncBarriers - AdditionalBarrier;

  static constexpr int value = [] {
    for (int ratio = MaxBarPerStageRatio; ratio > 0; ratio >>= 1) {
      if (2 * PipelineStages * ratio <= AvailablePipelineBarriers) {
        return ratio;
      }
    }
    return 0;
  }();
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Mp31 TME load (producer) Async Pipeline class FIX
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <int Stages_, int BarPerStageRatio_ = 1>
class Mp31PipelineTmeAsyncWarpSpecialized {
 public:
  using FullBarrier                          = mutlass::arch::AsyncTransactionBarrier;
  using EmptyBarrier                         = mutlass::arch::AsyncBarrier;
  static constexpr uint32_t Stages           = Stages_;
  static constexpr uint32_t BarPerStageRatio = BarPerStageRatio_;
  using PipelineState                        = Mp31PipelineStateWarpSpecialized<Stages, BarPerStageRatio>;
  static_assert(FullBarrier::ReservedAsyncBarrierCount == EmptyBarrier::ReservedAsyncBarrierCount &&
                FullBarrier::ReservedAsyncBarrierCount == 1);
  static_assert(BarPerStageRatio_ > 0);

  static constexpr uint32_t NumBarriersPerStage = 2 * BarPerStageRatio;
  static constexpr uint32_t NumFullBarriers     = Stages * BarPerStageRatio;
  static constexpr uint32_t NumEmptyBarriers    = NumFullBarriers;
  static constexpr uint32_t NumBarriers         = Stages * NumBarriersPerStage;

  struct Params {
    uint32_t transaction_bytes = 0;
    uint32_t num_consumers     = 0;
    uint32_t num_producers     = 1;
  };

  // Constructor
  MUTLASS_DEVICE
  Mp31PipelineTmeAsyncWarpSpecialized(Params params, uint32_t barrier_base = 0, uint32_t init_warps = 1)
      : params_(params), barrier_base_(barrier_base + FullBarrier::ReservedAsyncBarrierCount) {
    int warp_idx = canonical_warp_idx();

    if (warp_idx < init_warps) {
      // Init full barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < BarPerStageRatio; ++j) {
          FullBarrier::init(barrier_base_ + i * BarPerStageRatio + j, params_.num_producers, 0);
        }
      }

      // Init empty barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < BarPerStageRatio; ++j) {
          uint32_t const barrier_index = i * BarPerStageRatio + j;
          uint32_t const init_phase    = barrier_index < Stages ? 0 : 1;
          EmptyBarrier::init(barrier_base_ + NumFullBarriers + barrier_index, params_.num_consumers, init_phase);
        }
      }
    }
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  MUTLASS_DEVICE
  void producer_acquire(PipelineState state) {
    if constexpr (Stages > 0) {
      producer_acquire(state.index(), state.barrier_index(), state.phase());
    }
  }

  MUTLASS_DEVICE
  void producer_expect_transaction(PipelineState state, uint32_t transaction_bytes) {
    if constexpr (Stages > 0) {
      producer_expect_transaction(state.index(), state.barrier_index(), transaction_bytes);
    }
  }

  MUTLASS_DEVICE
  uint32_t producer_get_barrier_id(PipelineState state) {
    if constexpr (Stages > 0) {
      return producer_get_barrier_id(state.index(), state.barrier_index());
    }
    return 0;
  }

  ////////////////////
  // Consumers APIs
  ////////////////////
  MUTLASS_DEVICE
  void consumer_wait(PipelineState state) {
    if constexpr (Stages > 0) {
      consumer_wait(state.index(), state.barrier_index(), state.phase());
    }
  }

  MUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    if constexpr (Stages > 0) {
      consumer_release(state.index(), state.barrier_index());
    }
  }

 private:
  Params   params_;
  uint32_t barrier_base_;

  MUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t barrier_pair, uint32_t phase) {
    uint32_t empty_barrier_id = barrier_base_ + NumFullBarriers + barrier_pair;
    EmptyBarrier::wait(empty_barrier_id, phase);

    uint32_t full_barrier_id = barrier_base_ + barrier_pair;
    FullBarrier::arrive_and_expect_tx(full_barrier_id, params_.transaction_bytes);
  }

  MUTLASS_DEVICE
  void producer_expect_transaction(uint32_t stage, uint32_t barrier_pair, uint32_t transaction_bytes) {
    uint32_t full_barrier_id = barrier_base_ + barrier_pair;
    FullBarrier::expect_transaction(full_barrier_id, transaction_bytes);
  }

  MUTLASS_DEVICE
  uint32_t producer_get_barrier_id(uint32_t stage, uint32_t barrier_pair) {
    return barrier_base_ + barrier_pair;
  }

  MUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t barrier_pair, uint32_t phase) {
    uint32_t full_barrier_id = barrier_base_ + barrier_pair;
    FullBarrier::wait(full_barrier_id, phase);
  }

  MUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t barrier_pair) {
    uint32_t empty_barrier_id = barrier_base_ + NumFullBarriers + (barrier_pair + Stages) % NumFullBarriers;
    EmptyBarrier::arrive(empty_barrier_id);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simple producer-consumer async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages_, int BarPerStageRatio_ = 1>
class Mp31PipelineAsyncWarpSpecialized {
 public:
  using FullBarrier                          = mutlass::arch::AsyncBarrier;
  using EmptyBarrier                         = mutlass::arch::AsyncBarrier;
  static constexpr uint32_t Stages           = Stages_;
  static constexpr uint32_t BarPerStageRatio = BarPerStageRatio_;
  using PipelineState                        = Mp31PipelineStateWarpSpecialized<Stages, BarPerStageRatio>;
  static_assert(FullBarrier::ReservedAsyncBarrierCount == EmptyBarrier::ReservedAsyncBarrierCount &&
                FullBarrier::ReservedAsyncBarrierCount == 1);
  static_assert(BarPerStageRatio_ > 0);

  static constexpr uint32_t NumBarriersPerStage = 2 * BarPerStageRatio;
  static constexpr uint32_t NumFullBarriers     = Stages * BarPerStageRatio;
  static constexpr uint32_t NumEmptyBarriers    = NumFullBarriers;
  static constexpr uint32_t NumBarriers         = Stages * NumBarriersPerStage;

  struct Params {
    uint32_t producer_arv_count = 1;
    uint32_t consumer_arv_count = 1;
  };

  // Constructor
  MUTLASS_DEVICE
  Mp31PipelineAsyncWarpSpecialized(Params params, uint32_t barrier_base = 0)
      : params_(params), barrier_base_(barrier_base + FullBarrier::ReservedAsyncBarrierCount) {
    int warp_idx = canonical_warp_idx();

    if (warp_idx == 0) {
      // Init full barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < BarPerStageRatio; ++j) {
          FullBarrier::init(barrier_base_ + i * BarPerStageRatio + j, params_.producer_arv_count, 0);
        }
      }

      // Init empty barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < BarPerStageRatio; ++j) {
          uint32_t const barrier_index = i * BarPerStageRatio + j;
          uint32_t const init_phase    = barrier_index < Stages ? 0 : 1;
          EmptyBarrier::init(barrier_base_ + NumFullBarriers + barrier_index, params_.consumer_arv_count, init_phase);
        }
      }
    }
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  MUTLASS_DEVICE
  void producer_acquire(PipelineState state) {
    if constexpr (Stages > 0) {
      producer_acquire(state.index(), state.barrier_index(), state.phase());
    }
  }

  MUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    if constexpr (Stages > 0) {
      producer_commit(state.index(), state.barrier_index());
    }
  }

  MUTLASS_DEVICE
  uint32_t producer_get_barrier_id(PipelineState state) {
    if constexpr (Stages > 0) {
      return producer_get_barrier_id(state.index(), state.barrier_index());
    }
    return 0;
  }

  ////////////////////
  // Consumers APIs
  ////////////////////
  MUTLASS_DEVICE
  void consumer_wait(PipelineState state) {
    if constexpr (Stages > 0) {
      consumer_wait(state.index(), state.barrier_index(), state.phase());
    }
  }

  MUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    if constexpr (Stages > 0) {
      consumer_release(state.index(), state.barrier_index());
    }
  }

 private:
  Params   params_;
  uint32_t barrier_base_;

  MUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t barrier_pair, uint32_t phase) {
    uint32_t empty_barrier_id = barrier_base_ + NumFullBarriers + barrier_pair;
    EmptyBarrier::wait(empty_barrier_id, phase);
  }

  MUTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t barrier_pair) {
    uint32_t full_barrier_id = barrier_base_ + barrier_pair;
    FullBarrier::arrive(full_barrier_id);
  }

  MUTLASS_DEVICE
  uint32_t producer_get_barrier_id(uint32_t stage, uint32_t barrier_pair) {
    return barrier_base_ + barrier_pair;
  }

  MUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t barrier_pair, uint32_t phase) {
    uint32_t full_barrier_id = barrier_base_ + barrier_pair;
    FullBarrier::wait(full_barrier_id, phase);
  }

  MUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t barrier_pair) {
    uint32_t empty_barrier_id = barrier_base_ + NumFullBarriers + (barrier_pair + Stages) % NumFullBarriers;
    EmptyBarrier::arrive(empty_barrier_id);
  }
};

}  // end namespace mutlass
