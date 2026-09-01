/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file
    \brief Architecture-specific operators on memory
*/

#pragma once

#include "mutlass/mutlass.h"
#include "mutlass/arch/cache_operation.h"

namespace mutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// AddressSpace
//
/////////////////////////////////////////////////////////////////////////////////////////////////

enum class AddressSpace {
  Generic = 0,
  Global  = 1,
  Shared  = 3,
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store loaded data
    typename AccessType,
    /// The bytes of loading
    int LoadBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always
    >
struct global_load;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   32
                  > {
  MUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    using Type = uint32_t __attribute__((vector_size(32)));
    Type &data = reinterpret_cast<Type &>(D);
    if (pred_guard) {
      data = *reinterpret_cast<Type const *>(ptr);
    }
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   16
                  > {
  MUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    using Type = uint32_t __attribute__((vector_size(16)));
    Type &data = reinterpret_cast<Type &>(D);
    if (pred_guard) {
      data = *reinterpret_cast<Type const *>(ptr);
    }
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   8
                  > {
  MUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint64_t &data = reinterpret_cast<uint64_t &>(D);
    if (pred_guard) {
      data = *reinterpret_cast<uint64_t const *>(ptr);
    }
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   4
                  > {
  MUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint32_t &data = reinterpret_cast<uint32_t &>(D);
    if (pred_guard) {
      data = *reinterpret_cast<uint32_t const *>(ptr);
    }

  }
};

template <typename AccessType>
struct global_load<AccessType,
                   2
                  > {
  MUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint16_t &data = reinterpret_cast<uint16_t &>(D);
    if (pred_guard) {
      data = *reinterpret_cast<uint16_t const *>(ptr);
    }
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   1
                  > {
  MUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint8_t &data = reinterpret_cast<uint8_t &>(D);
    if (pred_guard) {
      data = *reinterpret_cast<uint8_t const *>(ptr);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store data
    typename AccessType,
    /// The bytes of storing
    int StoreBytes
    >
struct global_store;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////


template <typename AccessType>
struct global_store<AccessType, 64> {
  MUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    using Type = uint32_t __attribute__((vector_size(64)));
    Type const &data = reinterpret_cast<Type const &>(D);
    if (pred_guard) {
      *(reinterpret_cast<Type *>(ptr)) = data;
    }
  }
};


template <typename AccessType>
struct global_store<AccessType, 32> {
  MUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    using Type = uint32_t __attribute__((vector_size(32)));
    Type const &data = reinterpret_cast<Type const &>(D);
    if (pred_guard) {
      *(reinterpret_cast<Type *>(ptr)) = data;
    }
  }
};

template <typename AccessType>
struct global_store<AccessType, 16> {
  MUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    using Type = uint32_t __attribute__((vector_size(16)));
    Type const &data = reinterpret_cast<Type const &>(D);
    if (pred_guard) {
      *(reinterpret_cast<Type *>(ptr)) = data;
    }
  }
};

template <typename AccessType>
struct global_store<AccessType, 8> {
  MUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    uint64_t const &data = reinterpret_cast<uint64_t const &>(D);
    if (pred_guard) {
      *(reinterpret_cast<uint64_t *>(ptr)) = data;
    }
  }
};

template <typename AccessType>
struct global_store<AccessType, 4> {
  MUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
    if (pred_guard) {
      *(reinterpret_cast<uint32_t *>(ptr)) = data;
    }
  }
};

template <typename AccessType>
struct global_store<AccessType, 2> {
  MUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    uint16_t const &data = reinterpret_cast<uint16_t const &>(D);
    if (pred_guard) {
      *(reinterpret_cast<uint16_t *>(ptr)) = data;
    }
  }
};

template <typename AccessType>
struct global_store<AccessType, 1> {
  MUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    uint8_t const &data = reinterpret_cast<uint8_t const &>(D);
    if (pred_guard) {
      *(reinterpret_cast<uint8_t *>(ptr)) = data;
    }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// ld.shared
template <int Bytes>
MUTLASS_DEVICE
void shared_load(void *dst, uint32_t ptr);

/// ld.shared - 16b
template <>
MUTLASS_DEVICE
void shared_load<2>(void *dst, uint32_t ptr) {
  auto smem_ptr = reinterpret_cast<
                    uint16_t __attribute__((address_space(static_cast<int>(AddressSpace::Shared))))*>(ptr);
  uint16_t *dst_u16  = reinterpret_cast<uint16_t*>(dst);
  *dst_u16 = *smem_ptr;
}

/// ld.shared - 32b
template <>
MUTLASS_DEVICE
void shared_load<4>(void *dst, uint32_t ptr) {
  auto smem_ptr = reinterpret_cast<
                    uint32_t __attribute__((address_space(static_cast<int>(AddressSpace::Shared))))*>(ptr);
  uint32_t *dst_u32  = reinterpret_cast<uint32_t*>(dst);
  *dst_u32 = *smem_ptr;
}

/// ld.shared - 64b
template <>
MUTLASS_DEVICE
void shared_load<8>(void *dst, uint32_t ptr) {
  auto smem_ptr = reinterpret_cast<
                    uint64_t __attribute__((address_space(static_cast<int>(AddressSpace::Shared))))*>(ptr);
  uint64_t *dst_u64 = reinterpret_cast<uint64_t *>(dst);
  *dst_u64 = *smem_ptr;
}

/// ld.shared - 128b
template <>
MUTLASS_DEVICE
void shared_load<16>(void *dst, uint32_t ptr) {
  using Type = uint32_t __attribute__((vector_size(16)));
  auto smem_ptr = reinterpret_cast<
                    Type __attribute__((address_space(static_cast<int>(AddressSpace::Shared))))*>(ptr);
  Type *dst_u128 = reinterpret_cast<Type *>(dst);
  *dst_u128 = *smem_ptr;
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// st.shared
template <int Bytes>
MUTLASS_DEVICE
void shared_store(uint32_t ptr, void const *src);

/// st.shared - 16b
template <>
MUTLASS_DEVICE
void shared_store<2>(uint32_t ptr, void const *src) {
  auto smem_ptr = reinterpret_cast<
                    uint16_t __attribute__((address_space(static_cast<int>(AddressSpace::Shared)))*>(ptr);
  *smem_ptr = *reinterpret_cast<uint16_t const*>(src);
}

/// st.shared - 32b
template <>
MUTLASS_DEVICE
void shared_store<4>(uint32_t ptr, void const *src) {
  auto smem_ptr = reinterpret_cast<
                    uint32_t __attribute__((address_space(static_cast<int>(AddressSpace::Shared))))*>(ptr);
  *smem_ptr = *reinterpret_cast<uint32_t const*>(src);
}

/// st.shared - 64b
template <>
MUTLASS_DEVICE
void shared_store<8>(uint32_t ptr, void const *src) {
  auto smem_ptr = reinterpret_cast<
                    uint64_t __attribute__((address_space(static_cast<int>(AddressSpace::Shared))))*>(ptr);
  *smem_ptr = *reinterpret_cast<uint64_t const*>(src);
}

/// st.shared - 128b
template <>
MUTLASS_DEVICE
void shared_store<16>(uint32_t ptr, void const *src) {
  using Type = uint32_t __attribute__((vector_size(16)));
  auto smem_ptr = reinterpret_cast<
                    Type __attribute__((address_space(static_cast<int>(AddressSpace::Shared))))*>(ptr);
  *smem_ptr = *reinterpret_cast<Type const*>(src);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

