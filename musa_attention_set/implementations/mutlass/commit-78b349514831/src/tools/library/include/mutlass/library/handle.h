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
    \brief BLAS-like handle used to launch operations on the MUSA device.
*/

#pragma once

#include <memory>
#include "mutlass/library/library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Handle object
class Handle {
private:

  /// Host workspace
  static int const kHostWorkspaceSize = (4 << 10);

  /// Provider of operations
  Provider provider_;

  /// MUSA device properties
  musaDeviceProp device_;

  /// MUSA stream
  musaStream_t stream_;

  /// Device workspace
  void *workspace_;

  /// Size of device workspace in bytes
  size_t workspace_size_;
    
  /// Indicates whether scalars are host or device pointers
  ScalarPointerMode scalar_pointer_mode_;

  /// Pointer to the most recently executed operation
  Operation const *last_operation_;

  int device_idx_;

public:

  /// Constructor
  Handle(musaStream_t stream = nullptr, size_t workspace_size = (4<<20));

  /// Destructor
  ~Handle();

  /// Move constructor
  Handle(Handle && handle);

  /// Move assignment operator
  Handle &operator=(Handle && handle);

  //
  // Persistent state accessors
  //
  
  /// Returns compute capability of the selected device
  int compute_capability() const;

  /// Sets the current MUSA stream
  void set_stream(musaStream_t stream);

  /// Gets the current MUSA stream
  musaStream_t get_stream() const;

  /// Gets the current provider
  Provider get_provider() const;

  /// Sets the provider of operations
  void set_provider(Provider provider);

  /// Gets the device workspace size
  size_t get_workspace_size() const;

  /// Gets a pointer to the device workspace allocation in Global Memory
  void *get_workspace() const;

  /// Sets the size of device workspace, invalidating calls to get_device_workspace()
  void set_workspace_size(size_t bytes);

  /// Gets the scalar pointer mode
  ScalarPointerMode get_scalar_pointer_mode() const;

  /// Sets the scalar pointer mode
  void set_scalar_pointer_mode(ScalarPointerMode mode);

  /// Gets the most recently executed operation
  Operation const *get_last_operation() const;

  //
  // Computations
  //

  /// Executes a GEMM computation: D <= alpha * A*B + beta * C
  Status gemm(

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
  );
  
  /// Executes a GEMM computation: D <= alpha * A*B + beta * C.
  //
  // Supports batched-strided, batched array or split-K serial or split-K parallel.
  //
  Status gemm_universal(

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

    int batch_count = 1,                      /// Batch count or number of split-K slices

    int64_t batch_stride_A = 0,               /// Batch stride of A operand
    int64_t batch_stride_B = 0,               /// Batch stride of B operand
    int64_t batch_stride_C = 0,               /// Batch stride of C operand
    int64_t batch_stride_D = 0                /// Batch stride of D operand
  );
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Unique pointer storing the handle
using HandlePtr = std::unique_ptr<Handle>;
} // namespace library
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

