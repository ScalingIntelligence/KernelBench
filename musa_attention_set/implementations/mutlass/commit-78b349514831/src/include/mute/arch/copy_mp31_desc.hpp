/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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

#include <mute/arch/copy.hpp>

namespace mute
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// Data Prefetch enum
////////////////////////////////////////////////////////////////////////////////////////////////////
enum class PrefetchSize : uint8_t {
  NONE = 0,
  B64  = 64,
  B128 = 128,
};

MUTE_HOST_DEVICE char const* to_string(PrefetchSize const& t) {
  switch (t) {
    case PrefetchSize::NONE: return "PrefetchNone";
    case PrefetchSize::B64:  return "Prefetch64B";
    case PrefetchSize::B128: return "Prefetch128B";
  }
  return nullptr;
}

#if !defined(__MUSACC_RTC__)
MUTE_HOST std::ostream& operator<<(std::ostream& os, PrefetchSize const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}
#endif // !defined(__MUSACC_RTC__)


//////////////////////////////////////////////////////////////////////////////////////////////////////
// Barriers are 32-bit of user-managed information used in broadly two types syncronization patterns
// 1) arrive/wait on warps
// 2) transaction-based
//////////////////////////////////////////////////////////////////////////////////////////////////////

// Initialize barrier with its barrier id
MUTE_HOST_DEVICE
void
initialize_barrier(uint32_t barrier_id,                // 32 bits user-managed barrier's id
                   uint32_t warp_count = 1,            // Warp count expected to arrive/wait on this barrier
                   uint32_t init_phase = 0)            // Init phase on this barrier
{
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
  __musa_async_init_arrival(barrier_id, warp_count, init_phase);
#endif
}

// Set the number of bytes transfered per transaction
MUTE_HOST_DEVICE
void
set_barrier_transaction_bytes(uint32_t barrier_id,     // 32 bits user-managed barrier's id
                              uint32_t bytes)          // Number of bytes transfered by per TME transaction
{
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
  __musa_async_add_trans(barrier_id, bytes);
#endif
}

// Barrier arrive
template <bool return_phase = true>
MUTE_HOST_DEVICE
auto                                                   // Phase bit of the barrier if return_phase = true
arrive_barrier(uint32_t barrier_id)                    // 32 bits user-managed barrier's id
{
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
  if constexpr (return_phase) {
    return __musa_async_arrive(barrier_id);
  } else {
    __musa_async_arrive(barrier_id);
  }
#else
  if constexpr (return_phase) {
    return -1;
  } else {
    return;
  }
#endif
}

// Barrier wait
MUTE_HOST_DEVICE
void
wait_barrier(uint32_t barrier_id,                      // 32 bits user-managed barrier's id
             uint32_t phase_bit)                       // Current phase bit the barrier waiting to flip
{
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
  __musa_async_wait(barrier_id, phase_bit);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TME Descriptor and utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace TME {

enum class SmemSwizzleGranularity : uint8_t {
  NONE = 0,
  B16  = 1,
  B32  = 2,
  B64  = 3,
};

enum class SmemSwizzleStride : uint8_t {
  B32  = 0,
  B64  = 1,
  B128 = 2,
  B256 = 3,
};

enum class SmemSwizzleLine : uint8_t {
  B128 = 0,
  B256 = 1,
};

enum class CacheHint : uint8_t {
  CACHE_NONE    = 0,
  CACHE_ONCE    = 1,
  CACHE_NORMAL  = 2,
  CACHE_PERSIST = 3,
};


constexpr bool operator<(SmemSwizzleGranularity sg, SmemSwizzleStride ss) {
  return static_cast<uint8_t>(sg) < (static_cast<uint8_t>(ss) + 2);
}

constexpr bool operator<=(SmemSwizzleStride ss, SmemSwizzleLine sl) {
  return static_cast<uint8_t>(ss) <= (static_cast<uint8_t>(sl) + 2);
}

MUTE_HOST_DEVICE char const* to_string(SmemSwizzleGranularity const& t) {
  switch (t) {
    case SmemSwizzleGranularity::NONE: return "NONE";
    case SmemSwizzleGranularity::B16:  return "SG_16B";
    case SmemSwizzleGranularity::B32:  return "SG_32B";
    case SmemSwizzleGranularity::B64:  return "SG_64B";
  }
  return nullptr;
}

MUTE_HOST_DEVICE char const* to_string(SmemSwizzleStride const& t) {
  switch (t) {
    case SmemSwizzleStride::B32:  return "SS_32B";
    case SmemSwizzleStride::B64:  return "SS_64B";
    case SmemSwizzleStride::B128: return "SS_128B";
    case SmemSwizzleStride::B256: return "SS_256B";
  }
  return nullptr;
}

MUTE_HOST_DEVICE char const* to_string(SmemSwizzleLine const& t) {
  switch (t) {
    case SmemSwizzleLine::B128: return "SL_128B";
    case SmemSwizzleLine::B256: return "SL_256B";
  }
  return nullptr;
}


#if !defined(__MUSACC_RTC__)
MUTE_HOST std::ostream& operator<<(std::ostream& os, SmemSwizzleGranularity const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}

MUTE_HOST std::ostream& operator<<(std::ostream& os, SmemSwizzleStride const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}

MUTE_HOST std::ostream& operator<<(std::ostream& os, SmemSwizzleLine const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}

/// @return The TME descriptor datatype enum corresponding to T.
template <typename T>
inline MUtensorDescriptorDataType
to_MUtensorDescriptorDataType() {
  if constexpr (is_same_v<T,       int8_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8;     } else
  if constexpr (is_same_v<T,      uint8_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T, float_e4m3_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T, float_e5m2_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T,     uint16_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;   } else
  if constexpr (is_same_v<T,     uint32_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32;   } else
  if constexpr (is_same_v<T,     uint64_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT64;   } else
  if constexpr (is_same_v<T,      int32_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT32;    } else
  if constexpr (is_same_v<T,      int64_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT64;    } else
  if constexpr (is_same_v<T,       half_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT16;  } else
  if constexpr (is_same_v<T,        float>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32;  } else
  if constexpr (is_same_v<T,       double>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT64;  } else
  if constexpr (is_same_v<T,   bfloat16_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16; } else
  if constexpr (is_same_v<T,   tfloat32_t>) { return MU_TENSOR_DESCRIPTOR_DATA_TYPE_TFLOAT32; } else
  { static_assert(sizeof(T) < 0, "Unknown TME Format!"); }
}
#endif // !defined(__MUSACC_RTC__)

} // namespace TME

using TmeDescriptor = MUtensorDescriptor;

using TmeIm2ColParam = MUconvParamer;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Initiates a TensorDescriptor Prefetch
////////////////////////////////////////////////////////////////////////////////////////////////////

MUTE_HOST_DEVICE
void
prefetch_tme_descriptor(TmeDescriptor const* desc_ptr)
{
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
#if (__MUSA_ARCH__ == 310)
  prefetch(desc_ptr);
#endif
#else
  MUTE_INVALID_CONTROL_PATH("Trying to use TME Descriptor Prefetch on invalid arch");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Robust Buffer Access
////////////////////////////////////////////////////////////////////////////////////////////////////

using RobustReg = uint64_t __attribute__((vector_size(16)));

struct RobustDescriptor {
  RobustReg reg;
  uint64_t oob_addr;
};

template <class T>
MUTE_HOST_DEVICE constexpr
RobustDescriptor
make_robust_desc(T const* ptr, size_t elements) {
  uint64_t base_addr = reinterpret_cast<uint64_t>(ptr);
  uint64_t buff_size = static_cast<uint64_t>(sizeof(T) * elements);
  uint64_t oob_addr  = base_addr + buff_size;

  RobustReg reg {base_addr, buff_size};

  return {reg, oob_addr};
}

} //namespace mute
