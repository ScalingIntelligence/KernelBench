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
    \brief MUTLASS Library handle.
*/
#include <iostream> 
#include <stdexcept>
#include <cstdint>

#include "mutlass/library/handle.h"
#include "mutlass/library/singleton.h"
#include "mutlass/library/util.h"

namespace mutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Constructor
Handle::Handle(
  musaStream_t stream, 
  size_t workspace_size
):
  provider_(Provider::kMUTLASS), 
  stream_(stream), 
  workspace_(nullptr), 
  workspace_size_(0), 
  scalar_pointer_mode_(ScalarPointerMode::kHost), 
  last_operation_(nullptr) {

  int device_idx = -1;

  musaError_t error = musaGetDevice(&device_idx);
  if (error != musaSuccess) {
    throw std::runtime_error("musaGetDevice() failed");
  }

  error = musaGetDeviceProperties(&device_, device_idx);
  if (error != musaSuccess) {
    throw std::runtime_error("musaGetDeviceProperties() failed");
  }

  set_workspace_size(workspace_size);

  Singleton::get();
}

/// Destructor
Handle::~Handle() {
  if (workspace_) {

    if (workspace_) {
      musaFree(workspace_);
    }

    workspace_ = nullptr;
    workspace_size_ = 0;
  }
}

/// Move constructor
Handle::Handle(Handle && handle) {
  device_ = handle.device_;
  workspace_size_ = handle.workspace_size_;
  workspace_ = handle.workspace_;
  stream_ = handle.stream_;
  scalar_pointer_mode_ = handle.scalar_pointer_mode_;
  
  handle.workspace_ = nullptr;
  handle.workspace_size_ = 0;
}

/// Move assignment operator
Handle & Handle::operator=(Handle && handle) {

  provider_ = handle.provider_;
  device_ = handle.device_;
  workspace_size_ = handle.workspace_size_;
  workspace_ = handle.workspace_;
  stream_ = handle.stream_;
  scalar_pointer_mode_ = handle.scalar_pointer_mode_;

  handle.workspace_ = nullptr;
  handle.workspace_size_ = 0;

  return *this;
}

int Handle::compute_capability() const {
  return device_.major * 10 + device_.minor;
}

/// Sets the current MUSA stream
void Handle::set_stream(musaStream_t stream) {
  stream_ = stream;
}

/// Gets the current MUSA stream
musaStream_t Handle::get_stream() const {
  return stream_;
}

/// Gets the current provider
Provider Handle::get_provider() const {
  return provider_;
}

/// Sets the provider of operations
void Handle::set_provider(Provider provider) {
  provider_ = provider;
}

/// Gets the device workspace size
size_t Handle::get_workspace_size() const {
  return workspace_size_;
}

/// Gets a pointer to the device workspace allocation in Global Memory
void *Handle::get_workspace() const {
  return workspace_;
}

/// Sets the size of device workspace, invalidating previous calls to get_device_workspace()
void Handle::set_workspace_size(size_t bytes) {
  if (bytes != workspace_size_) {

    if (workspace_) {
      musaFree(workspace_);
    }
      
    workspace_ = nullptr;
    workspace_size_ = bytes;

    if (workspace_size_) {
  
      musaError_t error = musaMalloc((void **)&workspace_, workspace_size_);
  
      if (error != musaSuccess) {
        throw std::runtime_error("Failed to allocate workspace");
      }
    }
  }

  if (workspace_) {
    musaError_t error = musaMemset(workspace_, 0, workspace_size_);

    if (error != musaSuccess) {
      throw std::runtime_error("Failed to clear workspace");
    }
  }
}

/// Gets the scalar pointer mode
ScalarPointerMode Handle::get_scalar_pointer_mode() const {
  return scalar_pointer_mode_;
}

/// Sets the scalar pointer mode
void Handle::set_scalar_pointer_mode(ScalarPointerMode mode) {
  scalar_pointer_mode_ = mode;
}

/// Gets the last operation
Operation const *Handle::get_last_operation() const {
  return last_operation_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns the maximum required alignment for each operator
static int maximum_alignment_requirement(GemmDescription const &desc) {
  return std::max(
    std::max(desc.A.alignment, desc.B.alignment), desc.C.alignment);
}

/// Returns the largest alignment (in units of elements) the problem satisfies, starting from a
/// given upper limit.
static int gemm_problem_alignment(
  int M,
  int N,
  int K,
  NumericTypeID element_A,
  void const *ptr_A,
  int64_t lda,
  int64_t batch_stride_A,
  NumericTypeID element_B,
  void const *ptr_B,
  int64_t ldb,
  int64_t batch_stride_B,
  NumericTypeID element_C,
  void const * ptr_C,
  int64_t ldc,
  int64_t batch_stride_C,
  void const * ptr_D,
  int64_t ldd,
  int64_t batch_stride_D,
  int max_alignment_in_bytes = 16
) {

  void const *pointers[] = {
    ptr_A, ptr_B, ptr_C, ptr_D
  };

  int64_t extents[] = {
    M, N, K, lda, ldb, ldc, ldd, batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D
  };

  NumericTypeID elements[] = {
    element_A, element_B, element_C
  };

  for (; max_alignment_in_bytes > 0; max_alignment_in_bytes /= 2) {
    
    bool satisfied = true;

    // Can pointers satisfy this?
    for (void const *ptr : pointers) {
      std::uintptr_t int_ptr = reinterpret_cast<std::uintptr_t>(ptr);

      if (int_ptr % max_alignment_in_bytes) {
        satisfied = false;
        break;
      }
    }

    if (!satisfied) {
      continue;
    }

    // Compute the maximum alignment based on element data types
    int max_element_alignment = 0;

    for (NumericTypeID type_id : elements) {
      int element_alignment = max_alignment_in_bytes * 8 / library::sizeof_bits(type_id); 
      max_element_alignment = std::max(max_element_alignment, element_alignment);
    }

    // Can the problem size and leading dimensions satisfy this?
    for (int64_t extent : extents) {
      if (extent % max_element_alignment) {
        satisfied = false;
        break;
      }
    }

    if (!satisfied) {
      continue;
    }

    // Yes
    return max_element_alignment;
  }

  // No alignment satisfies this problem
  return 0;
}

/// Find the best kernel in descending order of preference.
static Operation const * find_gemm_operation(
  GemmOperationFunctionalMap::const_iterator operators_it, 
  GemmPreferenceKey const preference_key) {

  auto cc_it = operators_it->second.upper_bound(preference_key);

  if (cc_it == operators_it->second.begin()) {
    return nullptr;
  }

  Operation const *operation = nullptr;

  // Search in descending order of compute capability
  do {
    --cc_it;

    // Search tile sizes in order, for now.
    for (auto const * op : cc_it->second) {

      GemmDescription const &desc = static_cast<GemmDescription const &>(op->description());

      int min_cc = desc.tile_description.minimum_compute_capability;
      int max_cc = desc.tile_description.maximum_compute_capability;

      int op_alignment = maximum_alignment_requirement(desc);

      if ((min_cc <= preference_key.compute_capability) &&
        (preference_key.compute_capability <= max_cc) &&
        (op_alignment <= preference_key.alignment)) {

        operation = op;
        break;
      }
    }
  } while (!operation && cc_it != operators_it->second.begin());

  return operation;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Executes a GEMM computation: D <= alpha * A*B + beta * C
Status Handle::gemm(

  int M,                                    /// GEMM M dimension
  int N,                                    /// GEMM N dimension
  int K,                                    /// GEMM K dimension

  NumericTypeID element_compute,            /// Data type of internal accumulation

  NumericTypeID element_scalar,             /// Data type of alpha/beta scalars

  void const *alpha,                        /// Pointer to alpha scalar

  NumericTypeID element_A,                  /// Data type of A matrix elements
  LayoutTypeID layout_A,                    /// Layout of A matrix
  ComplexTransform transform_A,             /// Complex transformation applied to A matrix - ignored for real-valued matrices

  void const * ptr_A,                       /// Pointer to A matrix in Global Memory
  int64_t lda,                              /// Leading dimension of A matrix

  NumericTypeID element_B,                  /// Data type of B matrix elements
  LayoutTypeID layout_B,                    /// Layout of B matrix
  ComplexTransform transform_B,             /// Complex transformation applied to B matrix - ignored for real-valued matrices

  void const * ptr_B,                       /// Pointer to B matrix in Global Memory
  int64_t ldb,                              /// Leading dimension of B matrix

  void const * beta,                        /// Pointer to beta scalar

  NumericTypeID element_C,                  /// Data type of C and D matrices

  void const * ptr_C,                       /// Pointer to C matrix
  int64_t ldc,                              /// Leading dimension of C matrix

  void * ptr_D,                             /// Pointer to D matrix
  int64_t ldd                               /// Leading dimension of D matrix
) {
  
  //
  // Find the operation
  //

  GemmFunctionalKey key(
    provider_,
    GemmKind::kGemm,
    element_compute,
    element_scalar,
    element_A,
    layout_A,
    transform_A,
    element_B,
    layout_B,
    transform_B,
    element_C,  // C/D are same type and col major default
    LayoutTypeID::kColumnMajor,
    element_C,
    LayoutTypeID::kColumnMajor
  );

  auto operators_it = Singleton::get().operation_table.gemm_operations.find(key);

  if (operators_it == Singleton::get().operation_table.gemm_operations.end()) {
    return mutlass::Status::kErrorNotSupported;
  }
  
  if (operators_it->second.empty()) {
    return mutlass::Status::kErrorNotSupported;
  }

  //
  // Compute the largest alignment restriction the kernel can satisfy.
  //

  // Maximum alignment expectation among all kernels (in units of bytes)
  int const kMaximumAlignmentSize = 16;

  int alignment = gemm_problem_alignment(
    M, N, K, 
    element_A, ptr_A, lda, 0,
    element_B, ptr_B, ldb, 0,
    element_C, ptr_C, ldc, 0,
    ptr_D, ldd, 0, kMaximumAlignmentSize
  );

  //
  // Find the best kernel in descending order of preference.
  //

  GemmPreferenceKey preference_key(compute_capability(), alignment);

  Operation const *operation = find_gemm_operation(operators_it, preference_key);

  if (!operation) {
    return mutlass::Status::kErrorNotSupported;
  }

  last_operation_ = operation;

  //
  // Configure operation
  //

  GemmConfiguration configuration{
    {M, N, K},
    lda,
    ldb,
    ldc,
    ldd,
    1
  };

  // Query host work space size
  uint64_t host_workspace_size_needed = operation->get_host_workspace_size(&configuration);

  if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
    return mutlass::Status::kErrorNotSupported;
  }

  char host_workspace[kHostWorkspaceSize];

  // Query device workspace size
  uint64_t device_workspace_size_needed = operation->get_device_workspace_size(&configuration);

  if (uint64_t(workspace_size_) < device_workspace_size_needed) {
    return mutlass::Status::kErrorNotSupported;
  }

  // Initialize host and device workspaces
  Status status = operation->initialize(
    &configuration,
    host_workspace,
    workspace_,
    stream_);

  if (status != mutlass::Status::kSuccess) {
    return status;
  }

  // Run the operator
  GemmArguments arguments{
    ptr_A,
    ptr_B,
    ptr_C,
    ptr_D,
    alpha,
    beta,
    scalar_pointer_mode_
  };

  return operation->run(&arguments, host_workspace, workspace_, stream_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Executes a GEMM computation: D <= alpha * A*B + beta * C.
//
// Supports batched-strided, batched array or split-K serial or split-K parallel.
//
Status Handle::gemm_universal(

  GemmUniversalMode mode,                   /// indicates the mode in which the kUniversal GEMM is launched

  int M,                                    /// GEMM M dimension
  int N,                                    /// GEMM N dimension
  int K,                                    /// GEMM K dimension

  NumericTypeID element_compute,            /// Data type of internal accumulation

  NumericTypeID element_scalar,             /// Data type of alpha/beta scalars

  void const *alpha,                        /// Pointer to alpha scalar

  NumericTypeID element_A,                  /// Data type of A matrix elements
  LayoutTypeID layout_A,                    /// Layout of A matrix
  ComplexTransform transform_A,             /// Complex transformation applied to A matrix - ignored for real-valued matrices
  void const * ptr_A,                       /// Pointer to A matrix in Global Memory
  int64_t lda,                              /// Leading dimension of A matrix

  NumericTypeID element_B,                  /// Data type of B matrix elements
  LayoutTypeID layout_B,                    /// Layout of B matrix
  ComplexTransform transform_B,             /// Complex transformation applied to B matrix - ignored for real-valued matrices
  void const * ptr_B,                       /// Pointer to B matrix in Global Memory
  int64_t ldb,                              /// Leading dimension of B matrix

  void const * beta,                        /// Pointer to beta scalar

  NumericTypeID element_C,                  /// Data type of C matrix
  LayoutTypeID layout_C,                    /// Layout of D matrix
  void const * ptr_C,                       /// Pointer to C matrix
  int64_t ldc,                              /// Leading dimension of C matrix

  NumericTypeID element_D,                  /// Data type of D matrix
  LayoutTypeID layout_D,                    /// Layout of D matrix
  void * ptr_D,                             /// Pointer to D matrix
  int64_t ldd,                              /// Leading dimension of D matrix

  int batch_count,                          /// Batch count or number of split-K slices

  int64_t batch_stride_A,                   /// Batch stride of A operand
  int64_t batch_stride_B,                   /// Batch stride of B operand
  int64_t batch_stride_C,                   /// Batch stride of C operand
  int64_t batch_stride_D                    /// Batch stride of D operand
) {
  
  //
  // Find the operation
  //

  GemmFunctionalKey key(
    provider_,
    GemmKind::kUniversal,
    element_compute,
    element_scalar,
    element_A,
    layout_A,
    transform_A,
    element_B,
    layout_B,
    transform_B,
    element_C,
    layout_C,
    element_D,
    layout_D
  );

  auto operators_it = Singleton::get().operation_table.gemm_operations.find(key);

  if (operators_it == Singleton::get().operation_table.gemm_operations.end()) {
    return mutlass::Status::kErrorNotSupported;
  }
  
  if (operators_it->second.empty()) {
    return mutlass::Status::kErrorNotSupported;
  }

  //
  // Compute the largest alignment restriction the kernel can satisfy.
  //

  // Maximum alignment expectation among all kernels (in units of bytes)
  int const kMaximumAlignmentSize = 16;

  void const *ptr_A_check = ptr_A;
  void const *ptr_B_check = ptr_B;
  void const *ptr_C_check = ptr_C;
  void *      ptr_D_check = ptr_D;

  // Ignore alignment of pointers to pointers. We can't check this from the host,
  // as each batch index has its own pointer in device memory.
  if (mode == GemmUniversalMode::kArray) {
    ptr_A_check = nullptr; 
    ptr_B_check = nullptr; 
    ptr_C_check = nullptr; 
    ptr_D_check = nullptr; 
  }

  int alignment = gemm_problem_alignment(
    M, N, K, 
    element_A, ptr_A_check, lda, 0,
    element_B, ptr_B_check, ldb, 0,
    element_C, ptr_C_check, ldc, 0,
    ptr_D_check, ldd, 0, kMaximumAlignmentSize
  );

  //
  // Find the best kernel in descending order of preference.
  //

  GemmPreferenceKey preference_key(compute_capability(), alignment);

  Operation const *operation = find_gemm_operation(operators_it, preference_key);

  if (!operation) {
    return mutlass::Status::kErrorNotSupported;
  }

  last_operation_ = operation;

  //
  // Configure operation
  //

  GemmUniversalConfiguration configuration{
    mode,
    {M, N, K},
    batch_count,
    lda,
    ldb,
    ldc,
    ldd
  };

  // Query host work space size
  uint64_t host_workspace_size_needed = operation->get_host_workspace_size(&configuration);

  if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
    return mutlass::Status::kErrorNotSupported;
  }

  char host_workspace[kHostWorkspaceSize];

  GemmUniversalArguments arguments{
    {M, N, K},
    batch_count,
    ptr_A,
    ptr_B,
    ptr_C,
    ptr_D,
    alpha,
    beta,
    scalar_pointer_mode_,
    lda,
    ldb,
    ldc,
    ldd,
    batch_stride_A,
    batch_stride_B,
    batch_stride_C,
    batch_stride_D
  };

  // Query device workspace size
  uint64_t device_workspace_size_needed = operation->get_device_workspace_size(&configuration, &arguments);

  if (uint64_t(workspace_size_) < device_workspace_size_needed) {
    return mutlass::Status::kErrorNotSupported;
  }

  // Initialize host and device workspaces
  Status status = operation->initialize(
    &configuration,
    host_workspace,
    workspace_,
    stream_);

  if (status != mutlass::Status::kSuccess) {
    return status;
  }

  // Run the operator

  return operation->run(&arguments, host_workspace, workspace_, stream_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
