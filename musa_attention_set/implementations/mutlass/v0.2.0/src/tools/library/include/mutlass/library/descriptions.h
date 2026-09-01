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

#include <mutlass/library/types.h>
#include <mutlass/blas3_types.h>
#include <mutlass/gemm_coord.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

struct MathInstructionDescription {

  /// Shape of the target math instruction
  mutlass::gemm::GemmCoord instruction_shape;

  /// Describes the data type of the internal accumulator
  NumericTypeID element_accumulator;

  /// Classification of math instruction
  OpcodeClassID opcode_class;

  /// Type of math operation performed
  MathOperationID math_operation;

  //
  // Methods
  //

  MathInstructionDescription(
    mutlass::gemm::GemmCoord instruction_shape = mutlass::gemm::GemmCoord(),
    NumericTypeID element_accumulator = NumericTypeID::kInvalid,
    OpcodeClassID opcode_class = OpcodeClassID::kInvalid,
    MathOperationID math_operation = MathOperationID::kMultiplyAdd
  ):
    instruction_shape(instruction_shape), 
    element_accumulator(element_accumulator), 
    opcode_class(opcode_class),
    math_operation(math_operation) {}

  // Equality operator
  inline
  bool operator==(MathInstructionDescription const& rhs) const{
    return (
      (instruction_shape == rhs.instruction_shape) &&
      (element_accumulator == rhs.element_accumulator) &&
      (opcode_class == rhs.opcode_class) &&
      (math_operation == rhs.math_operation));
  }

  // Inequality operator
  inline
  bool operator!=(MathInstructionDescription const& rhs) const {
    return !(*this == rhs);
  }

};

/// Structure describing the tiled structure of a GEMM-like computation
struct TileDescription {

  /// Describes the shape of a threadblock (in elements)
  mutlass::gemm::GemmCoord threadblock_shape;

  /// Describes the number of pipeline stages in the threadblock-scoped mainloop
  int threadblock_stages;

  /// Number of warps in each logical dimension
  mutlass::gemm::GemmCoord warp_count;

  /// Core math instruction
  MathInstructionDescription math_instruction;

  /// Minimum compute capability (e.g. 70, 75) of a device eligible to run the operation.
  int minimum_compute_capability;

  /// Minimum compute capability (e.g. 70, 75) of a device eligible to run the operation.
  int maximum_compute_capability;

  /// Describes the shape of a cluster (in blocks)
  mutlass::gemm::GemmCoord cluster_shape;

  //
  // Methods
  //

  TileDescription(
    mutlass::gemm::GemmCoord threadblock_shape = mutlass::gemm::GemmCoord(),
    int threadblock_stages = 0,
    mutlass::gemm::GemmCoord warp_count = mutlass::gemm::GemmCoord(),
    MathInstructionDescription math_instruction = MathInstructionDescription(),
    int minimum_compute_capability = 0,
    int maximum_compute_capability = 0,
    mutlass::gemm::GemmCoord cluster_shape = mutlass::gemm::GemmCoord(1,1,1)
  ):
    threadblock_shape(threadblock_shape), 
    threadblock_stages(threadblock_stages), 
    warp_count(warp_count),
    math_instruction(math_instruction),
    minimum_compute_capability(minimum_compute_capability),
    maximum_compute_capability(maximum_compute_capability),
    cluster_shape(cluster_shape) { }

  // Equality operator
  inline
  bool operator==(TileDescription const& rhs) const{
    return (
      (threadblock_shape == rhs.threadblock_shape) &&
      (threadblock_stages == rhs.threadblock_stages) &&
      (warp_count == rhs.warp_count) &&
      (math_instruction == rhs.math_instruction) &&
      (minimum_compute_capability == rhs.minimum_compute_capability) &&
      (maximum_compute_capability == rhs.maximum_compute_capability));
  }

  // Inequality operator
  inline
  bool operator!=(TileDescription const& rhs) const {
    return !(*this == rhs);
  }
};

/// High-level description of an operation
struct OperationDescription {

  /// Unique identifier describing the operation
  char const * name;

  /// Operation provider
  Provider provider;

  /// Kind of operation
  OperationKind kind;

  /// Describes the tiled structure of a GEMM-like computation
  TileDescription tile_description;

  //
  // Methods
  //
  OperationDescription(
    char const * name = "unknown",
    Provider provider = Provider::kInvalid,
    OperationKind kind = OperationKind::kInvalid, 
    TileDescription const&  tile_description = TileDescription()
  ):
    name(name), provider(provider), kind(kind), tile_description(tile_description) { }
};

/// Structure describing the properties of a tensor
struct TensorDescription {

  /// Numeric type of an individual element
  NumericTypeID element;

  /// Enumerant identifying the layout function for the tensor
  LayoutTypeID layout;

  /// Alignment restriction on pointers, strides, and extents
  int alignment;

  /// log2() of the maximum extent of each dimension
  int log_extent_range;

  /// log2() of the maximum value each relevant stride may have
  int log_stride_range;
  
  //
  // Methods
  //

  TensorDescription(
    NumericTypeID element = NumericTypeID::kInvalid,
    LayoutTypeID layout = LayoutTypeID::kInvalid,
    int alignment = 1,
    int log_extent_range = 24,
    int log_stride_range = 24
  ):
    element(element), 
    layout(layout), 
    alignment(alignment), 
    log_extent_range(log_extent_range), 
    log_stride_range(log_stride_range)  { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description of all GEMM computations
struct GemmDescription : public OperationDescription {

  /// Indicates the kind of GEMM performed
  GemmKind gemm_kind;
  
  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand
  TensorDescription B;

  /// Describes the source matrix
  TensorDescription C;

  /// Describes the destination matrix
  TensorDescription D;

  /// Describes the sparse meta matrices
  TensorDescription E;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  /// Transformation on B operand
  ComplexTransform transform_B;

  //
  // Methods
  //

  GemmDescription(
    GemmKind gemm_kind = GemmKind::kGemm,
    TensorDescription const& A = TensorDescription(),
    TensorDescription const& B = TensorDescription(),
    TensorDescription const& C = TensorDescription(),
    TensorDescription const& D = TensorDescription(),
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    D(D),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {} 

  GemmDescription(
    OperationDescription op_desc,
    GemmKind gemm_kind,
    TensorDescription const& A,
    TensorDescription const& B,
    TensorDescription const& C,
    TensorDescription const& D,
    NumericTypeID element_epilogue,
    SplitKMode split_k_mode,
    ComplexTransform transform_A,
    ComplexTransform transform_B
  ):
    OperationDescription(op_desc),
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    D(D),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
