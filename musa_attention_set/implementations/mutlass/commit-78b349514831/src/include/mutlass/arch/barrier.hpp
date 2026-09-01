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

#include <mutlass/mutlass.h>

#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 310
#define MUSA_BARRIER_ENABLED 1
#else
#define MUSA_BARRIER_ENABLED 0
#endif

namespace mutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////


MUTLASS_DEVICE
void allocate_async_barriers(const uint32_t num) {
#if MUSA_BARRIER_ENABLED
  __musa_async_bar_record(num);
#endif
}

struct AsyncBarrier {
  // Range: [0, 63]
  uint32_t const id_;

public:
  static const uint32_t ReservedAsyncBarrierCount = 1;

  static const uint32_t HardwareMaxNumAsyncTransactionBarriers = 64;

  MUTLASS_DEVICE
  AsyncBarrier(uint32_t id = 0)
    : id_(id + ReservedAsyncBarrierCount) {}

  MUTLASS_DEVICE
  void init(uint32_t arrive_count, uint32_t init_phase = 0) const {
    AsyncBarrier::init(id_, arrive_count, init_phase);
  }


  template <bool return_phase = false>
  MUTLASS_DEVICE
  auto arrive() const {
    return AsyncBarrier::arrive<return_phase>(id_);
  }

  MUTLASS_DEVICE
  void wait(uint32_t phase) const {
    AsyncBarrier::wait(id_, phase);
  }

  MUTLASS_DEVICE
  void sync() const {
    AsyncBarrier::sync(id_);
  }

  MUTLASS_DEVICE
  uint32_t get_barrier_id() const {
    return id_;
  }

  //
  //  Static Versions
  //
  MUTLASS_DEVICE
  static void init(uint32_t id, uint32_t arrive_count, uint32_t init_phase) {
#if MUSA_BARRIER_ENABLED
    __musa_async_init_arrival(id, arrive_count, init_phase);
#endif
  }

  template <bool return_phase = false>
  MUTLASS_DEVICE
  static auto arrive(uint32_t id) {
#if MUSA_BARRIER_ENABLED
    if constexpr (return_phase) {
      return __musa_async_arrive(id);
    } else {
      __musa_async_arrive(id);
    }
#else
    if constexpr (return_phase) {
      return -1;
    } else {
      return;
    }
#endif
  }

  MUTLASS_DEVICE
  static void sync(uint32_t id) {
#if MUSA_BARRIER_ENABLED
    auto phase = __musa_async_arrive(id);
    __musa_async_wait(id, phase);
#endif
  }

  MUTLASS_DEVICE
  static void wait(uint32_t id, uint32_t phase) {
#if MUSA_BARRIER_ENABLED
    __musa_async_wait(id, phase);
#endif
  }
};

struct AsyncTransactionBarrier : public AsyncBarrier {
  using AsyncBarrier::AsyncBarrier;

  template <bool return_phase = false>
  MUTLASS_DEVICE
  auto arrive_and_expect_tx(uint32_t transaction_bytes) const {
    return AsyncTransactionBarrier::arrive_and_expect_tx<return_phase>(id_, transaction_bytes);
  }

  MUTLASS_DEVICE
  void expect_transaction(uint32_t transaction_bytes) const {
    AsyncTransactionBarrier::expect_transaction(id_, transaction_bytes);
  }

  MUTLASS_DEVICE
  void complete_transaction(uint32_t transaction_bytes) const {
    AsyncTransactionBarrier::complete_transaction(id_, transaction_bytes);
  }

  //
  //  Static Versions
  //

  template <bool return_phase = false>
  MUTLASS_DEVICE
  static auto arrive_and_expect_tx(uint32_t id, uint32_t transaction_bytes) {
#if MUSA_BARRIER_ENABLED
    __musa_async_add_trans(id, transaction_bytes);
    uint32_t phase = __musa_async_arrive(id);

    if constexpr (return_phase) {
      return phase;
    } else {
      (void)phase;
      return;
    }
#endif
  }

  MUTLASS_DEVICE
  static void expect_transaction(uint32_t id, uint32_t transaction_bytes) {
#if MUSA_BARRIER_ENABLED
    __musa_async_add_trans(id, transaction_bytes);
#endif
  }

  MUTLASS_DEVICE
  static void complete_transaction(uint32_t id, uint32_t transaction_bytes) {
#if MUSA_BARRIER_ENABLED
    __musa_async_decrease_trans(id, transaction_bytes);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // end namespace arch
}  // end namespace mutlass
