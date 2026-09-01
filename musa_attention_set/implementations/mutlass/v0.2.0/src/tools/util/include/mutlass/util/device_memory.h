/******************************************************************************
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
 ******************************************************************************/

#pragma once

/**
 * \file
 * \brief C++ interface to MUSA device memory management functions.
 */

#include <memory>

#include "mutlass/platform/platform.h"
#include "mutlass/numeric_types.h"
#include "exceptions.h"

namespace mutlass {
namespace device_memory {

/******************************************************************************
 * Allocation lifetime
 ******************************************************************************/

/// Allocate a buffer of \p count elements of type \p T on the current MUSA device
template <typename T>
T* allocate(size_t count = 1) {

  T* ptr = 0;
  size_t bytes = 0;

  bytes = count * sizeof(T);

  musaError_t musa_error = musaMalloc((void**)&ptr, bytes);

  if (musa_error != musaSuccess) {
#if (MUTLASS_DEBUG_TRACE_LEVEL > 0)
    std::ostringstream os;
    os << "mutlass::device_memory::allocate: musaMalloc failed: bytes=" << bytes;
    MUTLASS_TRACE_HOST(os.str());
#endif
    throw musa_exception("Failed to allocate memory", musa_error);
  }

#if (MUTLASS_DEBUG_TRACE_LEVEL > 1)
  else {
    std::ostringstream os;
    os << "mutlass::device_memory::allocate: Successful musaMalloc: bytes=" << bytes;
    MUTLASS_TRACE_HOST(os.str());
  }
#endif

  return ptr;
}

/// Free the buffer pointed to by \p ptr
template <typename T>
void free(T* ptr) {
  if (ptr) {
    musaError_t musa_error = (musaFree(ptr));
    if (musa_error != musaSuccess) {
      throw musa_exception("Failed to free device memory", musa_error);
    }
  }
}

/******************************************************************************
 * Data movement
 ******************************************************************************/

template <typename T>
void copy(T* dst, T const* src, size_t count, musaMemcpyKind kind) {
  size_t bytes = count * sizeof_bits<T>::value / 8;
  if (bytes == 0 && count > 0) {
    bytes = 1;
  }
  musaError_t musa_error = (musaMemcpy(dst, src, bytes, kind));
  if (musa_error != musaSuccess) {
    throw musa_exception("musaMemcpy() failed", musa_error);
  }
}

template <typename T>
void copy_to_device(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, musaMemcpyHostToDevice);
}

template <typename T>
void copy_to_host(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, musaMemcpyDeviceToHost);
}

template <typename T>
void copy_device_to_device(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, musaMemcpyDeviceToDevice);
}

template <typename T>
void copy_host_to_host(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, musaMemcpyHostToHost);
}

/// Copies elements from device memory to host-side range
template <typename OutputIterator, typename T>
void insert_to_host(OutputIterator begin, OutputIterator end, T const* device_begin) {
  size_t elements = end - begin;
  copy_to_host(&*begin, device_begin, elements);
}

/// Copies elements to device memory from host-side range
template <typename T, typename InputIterator>
void insert_to_device(T* device_begin, InputIterator begin, InputIterator end) {
  size_t elements = end - begin;
  copy_to_device(device_begin, &*begin, elements);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace device_memory

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DeviceAllocation {
public:

  /// Delete functor for MUSA device memory
  struct deleter {
    void operator()(T* ptr) {
      musaError_t musa_error = (musaFree(ptr));
      if (musa_error != musaSuccess) {
        // noexcept
        //                throw musa_exception("musaFree() failed", musa_error);
        return;
      }
    }
  };

public:
  //
  // Data members
  //

  /// Number of elements of T allocated on the current MUSA device
  size_t capacity;

  /// Smart pointer
  platform::unique_ptr<T, deleter> smart_ptr;

public:

  //
  // Static methods
  //

  /// Static member to compute the number of bytes needed for a given number of elements
  static size_t bytes(size_t elements) {
    if (sizeof_bits<T>::value < 8) {
      size_t const kElementsPerByte = 8 / sizeof_bits<T>::value;
      return elements / kElementsPerByte;
    }
    else {
      size_t const kBytesPerElement = sizeof_bits<T>::value / 8;
      return elements * kBytesPerElement;
    }
  }

public:

  //
  // Methods
  //

  /// Constructor: allocates no memory
  DeviceAllocation() : capacity(0) {}

  /// Constructor: allocates \p capacity elements on the current MUSA device
  DeviceAllocation(size_t _capacity) : 
    smart_ptr(device_memory::allocate<T>(_capacity)), capacity(_capacity) {}

  /// Constructor: allocates \p capacity elements on the current MUSA device taking ownership of the allocation
  DeviceAllocation(T *ptr, size_t _capacity) : smart_ptr(ptr), capacity(_capacity) {}

  /// Copy constructor
  DeviceAllocation(DeviceAllocation const &p): 
    smart_ptr(device_memory::allocate<T>(p.capacity)), capacity(p.capacity) {

    device_memory::copy_device_to_device(smart_ptr.get(), p.get(), capacity);
  }

  /// Move constructor
  DeviceAllocation(DeviceAllocation &&p): capacity(0) {
    std::swap(smart_ptr, p.smart_ptr);
    std::swap(capacity, p.capacity);
  }

  /// Destructor
  ~DeviceAllocation() { reset(); }

  /// Returns a pointer to the managed object
  T* get() const { return smart_ptr.get(); }

  /// Releases the ownership of the managed object (without deleting) and resets capacity to zero
  T* release() {
    capacity = 0;
    return smart_ptr.release();
  }

  /// Deletes the managed object and resets capacity to zero
  void reset() {
    capacity = 0;
    smart_ptr.reset();
  }

  /// Deletes managed object, if owned, and allocates a new object
  void reset(size_t _capacity) {
    reset(device_memory::allocate<T>(_capacity), _capacity);
  }

  /// Deletes managed object, if owned, and replaces its reference with a given pointer and capacity
  void reset(T* _ptr, size_t _capacity) {
    smart_ptr.reset(_ptr);
    capacity = _capacity;
  }

  /// Allocates a new buffer and copies the old buffer into it. The old buffer is then released.
  void reallocate(size_t new_capacity) {
    
    platform::unique_ptr<T, deleter> new_allocation(device_memory::allocate<T>(new_capacity));

    device_memory::copy_device_to_device(
      new_allocation.get(), 
      smart_ptr.get(), 
      std::min(new_capacity, capacity));

    std::swap(smart_ptr, new_allocation);
    std::swap(new_capacity, capacity);
  }

  /// Returns the number of elements
  size_t size() const {
    return capacity;
  }

  /// Returns the number of bytes needed to store the allocation
  size_t bytes() const {
    return bytes(capacity);
  }

  /// Returns a pointer to the object owned by *this
  T* operator->() const { return smart_ptr.get(); }

  /// Returns the deleter object which would be used for destruction of the managed object.
  deleter& get_deleter() { return smart_ptr.get_deleter(); }

  /// Returns the deleter object which would be used for destruction of the managed object (const)
  const deleter& get_deleter() const { return smart_ptr.get_deleter(); }

  /// Copies a device-side memory allocation
  DeviceAllocation & operator=(DeviceAllocation const &p) {
    if (capacity != p.capacity) {
      smart_ptr.reset(device_memory::allocate<T>(p.capacity));
      capacity = p.capacity;
    }
    device_memory::copy_device_to_device(smart_ptr.get(), p.get(), capacity);
    return *this;
  }

  /// Move assignment
  DeviceAllocation & operator=(DeviceAllocation && p) {
    std::swap(smart_ptr, p.smart_ptr);
    std::swap(capacity, p.capacity);
    return *this;
  }

  /// Copies the entire allocation from another location in device memory.
  void copy_from_device(T const *ptr) const {
    copy_from_device(ptr, capacity);
  }

  /// Copies a given number of elements from device memory
  void copy_from_device(T const *ptr, size_t elements) const {
    device_memory::copy_device_to_device(get(), ptr, elements);
  }

  void copy_to_device(T *ptr) const {
    copy_to_device(ptr, capacity);
  }

  void copy_to_device(T *ptr, size_t elements) const {
    device_memory::copy_device_to_device(ptr, get(), elements);
  }

  void copy_from_host(T const *ptr) const {
    copy_from_host(ptr, capacity);
  }

  void copy_from_host(T const *ptr, size_t elements) const {
    device_memory::copy_to_device(get(), ptr, elements);
  }

  void copy_to_host(T *ptr) const {
    copy_to_host(ptr, capacity);
  }

  void copy_to_host(T *ptr, size_t elements) const {
    device_memory::copy_to_host(ptr, get(), elements); 
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace device_memory {

/// Device allocation abstraction that tracks size and capacity
template <typename T>
using allocation = mutlass::DeviceAllocation<T>;

}  // namespace device_memory

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
