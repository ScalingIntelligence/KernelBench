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

#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
#define MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED
#define MUTE_ARCH_TME_MP31_ACTIVATED
#endif

#include <mute/config.hpp>

#include <mute/arch/copy.hpp>
#include <mute/arch/copy_mp31_desc.hpp>


namespace mute {

//
// Robust Buffer Access Load/Store
//

template <class TS, class TD = TS>
struct MP31_ROBUST_LOAD
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD),
                "MP31_ROBUST_LOAD requires sizeof(src_value_type) == sizeof(dst_value_type)");

  MUTE_HOST_DEVICE static void
  copy(TS               const& gmem_src,
       TD                    & rmem_dst,
       bool                    pred,
       RobustDescriptor const& src_desc)
  {
#if defined(MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED)
    void const* gmem_ptr = pred ? reinterpret_cast<void const*>(&gmem_src)
                         : reinterpret_cast<void const*>(src_desc.oob_addr);
    constexpr PrefetchSize prefetch_size = PrefetchSize::B128;
    if constexpr (sizeof(TS) == 1) {
      auto val = __musa_ld_v4_robust_i8(gmem_ptr, src_desc.reg, static_cast<int>(prefetch_size));
      rmem_dst = *reinterpret_cast<TD*>(&val);
    } else if constexpr (sizeof(TS) == 2) {
      auto val = __musa_ld_v4_robust_i16(gmem_ptr, src_desc.reg, static_cast<int>(prefetch_size));
      rmem_dst = *reinterpret_cast<TD*>(&val);
    } else if constexpr (sizeof(TS) == 4) {
      auto val = __musa_ld_v4_robust_i32(gmem_ptr, src_desc.reg, static_cast<int>(prefetch_size));
      rmem_dst = *reinterpret_cast<TD*>(&val);
    } else if constexpr (sizeof(TS) == 8) {
      auto val = __musa_ld_v4_robust_i64(gmem_ptr, src_desc.reg, static_cast<int>(prefetch_size));
      rmem_dst = *reinterpret_cast<TD*>(&val);
    } else if constexpr (sizeof(TS) == 16) {
      auto val = __musa_ld_v4_robust_v4i32(gmem_ptr, src_desc.reg, static_cast<int>(prefetch_size));
      rmem_dst = *reinterpret_cast<TD*>(&val);
    } else if constexpr (sizeof(TS) == 32) {
      auto val = __musa_ld_v4_robust_v8i32(gmem_ptr, src_desc.reg, static_cast<int>(prefetch_size));
      rmem_dst = *reinterpret_cast<TD*>(&val);
    } else if constexpr (sizeof(TS) == 64) {
      auto val = __musa_ld_v4_robust_v16i32(gmem_ptr, src_desc.reg, static_cast<int>(prefetch_size));
      rmem_dst = *reinterpret_cast<TD*>(&val);
    } else if constexpr (sizeof(TS) == 128) {
      auto val = __musa_ld_v4_robust_v32i32(gmem_ptr, src_desc.reg, static_cast<int>(prefetch_size));
      rmem_dst = *reinterpret_cast<TD*>(&val);
    } else {
      static_assert(sizeof(TS) <= 128, "Currently we only support up-to 1024-bit load");
    }
#else
    MUTE_INVALID_CONTROL_PATH(
        "Trying to use MP31_ROBUST_LOAD without MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED.");
#endif
  }
};

template <class TS, class TD = TS>
struct MP31_ROBUST_STORE
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD),
      "MP31_ROBUST_STORE requires sizeof(src_value_type) == sizeof(dst_value_type)");

  MUTE_HOST_DEVICE static void
  copy(TS               const& rmem_src,
       TD                    & gmem_dst,
       bool                    pred,
       RobustDescriptor const& dst_desc)
  {
#if defined(MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED)
    void* gmem_ptr = pred ? reinterpret_cast<void*>(&gmem_dst)
                          : reinterpret_cast<void*>(dst_desc.oob_addr);
    if constexpr (sizeof(TS) == 1) {
      __musa_st_v4_robust_i8(*reinterpret_cast<int8_t const*>(&rmem_src),
                             gmem_ptr, dst_desc.reg);
    } else if constexpr (sizeof(TS) == 2) {
      __musa_st_v4_robust_i16(*reinterpret_cast<mute::int16_t const*>(&rmem_src),
                              gmem_ptr, dst_desc.reg);
    } else if constexpr (sizeof(TS) == 4) {
      __musa_st_v4_robust_i32(*reinterpret_cast<mute::int32_t const*>(&rmem_src),
                              gmem_ptr, dst_desc.reg);
    } else if constexpr (sizeof(TS) == 8) {
      __musa_st_v4_robust_i64(*reinterpret_cast<mute::int64_t const*>(&rmem_src),
                              gmem_ptr, dst_desc.reg);
    } else if constexpr (sizeof(TS) == 16) {
      __musa_st_v4_robust_v4i32(*reinterpret_cast<mute::int128_t const*>(&rmem_src),
                                gmem_ptr, dst_desc.reg);
    } else if constexpr (sizeof(TS) == 32) {
      __musa_st_v4_robust_v8i32(*reinterpret_cast<mute::int256_t const*>(&rmem_src),
                                gmem_ptr, dst_desc.reg);
    } else if constexpr (sizeof(TS) == 64) {
      __musa_st_v4_robust_v16i32(*reinterpret_cast<mute::int512_t const*>(&rmem_src),
                                 gmem_ptr, dst_desc.reg);
    } else if constexpr (sizeof(TS) == 128) {
      __musa_st_v4_robust_v32i32(*reinterpret_cast<mute::int1024_t const*>(&rmem_src),
                                 gmem_ptr, dst_desc.reg);
    } else {
      static_assert(sizeof(TS) <= 128, "Currently we only support up-to 1024-bit store");
    }
#else
    MUTE_INVALID_CONTROL_PATH(
        "Trying to use MP31_ROBUST_STORE without MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED.");
#endif
  }
};


//
// Robust Buffer Access LDGSTS
//

template <class TS, class TD = TS>
struct MP31_ROBUST_LDGSTS
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD),
                "MP31_ROBUST_LDGSTS requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 1  || sizeof(TS) == 2  || sizeof(TS) == 4  || sizeof(TS) == 8 ||
                sizeof(TS) == 16 || sizeof(TS) == 32 || sizeof(TS) == 64 || sizeof(TS) == 128,
                "MP31_ROBUST_LDGSTS sizeof(TS) is not supported");

  MUTE_HOST_DEVICE static void
  copy(TS               const& gmem_src,
       TD                    & smem_dst,
       bool                    pred,
       RobustDescriptor const& src_desc)
  {
#if defined(MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED)
    uint64_t gmem_int_ptr = reinterpret_cast<uint64_t>(&gmem_src);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    auto gmem_ptr = make_ptr_with_address_space<AddressSpace::Global>(
                            pred ? gmem_int_ptr : src_desc.oob_addr);
    auto smem_ptr = make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr);
    constexpr PrefetchSize prefetch_size = PrefetchSize::B128;
    __musa_memcpy_g2s_robust_v4(smem_ptr, gmem_ptr, sizeof(TS), src_desc.reg, static_cast<int>(prefetch_size));
#else
    MUTE_INVALID_CONTROL_PATH(
        "Trying to use MP31_ROBUST_LDGSTS without MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED.");
#endif
  }
};

// LDGSTS
template <class TS, class TD = TS>
struct MP31_LDGSTS
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD),
                "MP31_ROBUST_LDGSTS requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 1  || sizeof(TS) == 2  || sizeof(TS) == 4  || sizeof(TS) == 8 ||
                sizeof(TS) == 16 || sizeof(TS) == 32 || sizeof(TS) == 64 || sizeof(TS) == 128,
                "MP31_ROBUST_LDGSTS sizeof(TS) is not supported");

  MUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst)
  {
#if defined(MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED)
    uint64_t gmem_int_ptr = reinterpret_cast<uint64_t>(&gmem_src);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    auto gmem_ptr = make_ptr_with_address_space<AddressSpace::Global>(gmem_int_ptr);
    auto smem_ptr = make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr);
    constexpr PrefetchSize prefetch_size = PrefetchSize::B128;
    __musa_memcpy_g2s(smem_ptr, gmem_ptr, sizeof(TS), static_cast<int>(prefetch_size));
#else
    MUTE_INVALID_CONTROL_PATH(
        "Trying to use MP31_LDGSTS without MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED.");
#endif
  }
};


MUTE_HOST_DEVICE
void
ldgsts_wait() {
#if defined(MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED)
  __musa_memcpy_g2s_wait();
#else
  MUTE_INVALID_CONTROL_PATH("Trying to use MemcpyG2S Wait without MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED.");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mute

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <mute/arch/copy_mp31_tme.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////

