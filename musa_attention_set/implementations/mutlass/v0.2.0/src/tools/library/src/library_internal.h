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
/*! 
  \file

  \brief MUTLASS Library is an object-oriented approach to managing operations implemented by MUTLASS.

  Generally,
    
    description   - compile-time constant parameters used to instantiate an operation

    configuration - runtime parameters with computationally expensive initialization 
    
    arguments     - runtime parameters that may be passed to an initialized operation with low
                    computational overhead
*/

#pragma once

#include "mutlass/mutlass.h"
#include "mutlass/complex.h"
#include "mutlass/numeric_types.h"
#include "mutlass/arch/arch.h"
#include "mutlass/arch/mma.h"
#include "mutlass/layout/matrix.h"

#include "mutlass/library/library.h"
#include "mutlass/library/arch_mappings.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct NumericTypeMap;

template <> struct NumericTypeMap<void> {
  static NumericTypeID const kId = NumericTypeID::kVoid;
};

template <> struct NumericTypeMap<mutlass::uint1b_t> {
  static NumericTypeID const kId = NumericTypeID::kB1;
};

template <> struct NumericTypeMap<mutlass::int4b_t> {
  static NumericTypeID const kId = NumericTypeID::kS4;
};

template <> struct NumericTypeMap<int8_t> {
  static NumericTypeID const kId = NumericTypeID::kS8;
};

template <> struct NumericTypeMap<int16_t> {
  static NumericTypeID const kId = NumericTypeID::kS16;
};

template <> struct NumericTypeMap<int32_t> {
  static NumericTypeID const kId = NumericTypeID::kS32;
};

template <> struct NumericTypeMap<int64_t> {
  static NumericTypeID const kId = NumericTypeID::kS64;
};

template <> struct NumericTypeMap<mutlass::uint4b_t> {
  static NumericTypeID const kId = NumericTypeID::kU4;
};

template <> struct NumericTypeMap<uint8_t> {
  static NumericTypeID const kId = NumericTypeID::kU8;
};

template <> struct NumericTypeMap<mutlass::float_e4m3_t> {
  static NumericTypeID const kId = NumericTypeID::kFE4M3;
};

template <> struct NumericTypeMap<mutlass::float_e5m2_t> {
  static NumericTypeID const kId = NumericTypeID::kFE5M2;
};

template <> struct NumericTypeMap<uint16_t> {
  static NumericTypeID const kId = NumericTypeID::kU16;
};

template <> struct NumericTypeMap<uint32_t> {
  static NumericTypeID const kId = NumericTypeID::kU32;
};

template <> struct NumericTypeMap<uint64_t> {
  static NumericTypeID const kId = NumericTypeID::kU64;
};

template <> struct NumericTypeMap<mutlass::half_t> {
  static NumericTypeID const kId = NumericTypeID::kF16;
};

template <> struct NumericTypeMap<float> {
  static NumericTypeID const kId = NumericTypeID::kF32;
};

template <> struct NumericTypeMap<double> {
  static NumericTypeID const kId = NumericTypeID::kF64;
};

template <> struct NumericTypeMap<mutlass::complex<mutlass::half_t> > {
  static NumericTypeID const kId = NumericTypeID::kCF16;
};

template <> struct NumericTypeMap<mutlass::complex<float> > {
  static NumericTypeID const kId = NumericTypeID::kCF32;
};

template <> struct NumericTypeMap<mutlass::complex<double> > {
  static NumericTypeID const kId = NumericTypeID::kCF64;
};

template <> struct NumericTypeMap<mutlass::bfloat16_t> {
  static NumericTypeID const kId = NumericTypeID::kBF16;
};

template <> struct NumericTypeMap<mutlass::tfloat32_t> {
  static NumericTypeID const kId = NumericTypeID::kTF32;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct MathOperationMap {
  static MathOperationID const kId = MathOperationID::kInvalid;
};

template <> struct MathOperationMap<mutlass::arch::OpMultiplyAdd> {
  static MathOperationID const kId = MathOperationID::kMultiplyAdd;
};

template <> struct MathOperationMap<mutlass::arch::OpMultiplyAddFastBF16> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddFastBF16;
};

template <> struct MathOperationMap<mutlass::arch::OpMultiplyAddFastF16> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddFastF16;
};

template <> struct MathOperationMap<mutlass::arch::OpMultiplyAddSaturate> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddSaturate;
};

template <> struct MathOperationMap<mutlass::arch::OpMultiplyAddMixedInputUpcast> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddMixedInputUpcast;
};

template <> struct MathOperationMap<mutlass::arch::OpMultiplyAddComplex> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddComplex;
};

template <> struct MathOperationMap<mutlass::arch::OpMultiplyAddGaussianComplex> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddGaussianComplex;
};

template <> struct MathOperationMap<mutlass::arch::OpXorPopc> {
  static MathOperationID const kId = MathOperationID::kXorPopc;
};


template <> struct MathOperationMap<mutlass::arch::OpMultiplyAddFastF32> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddFastF32;
};

template <> struct MathOperationMap<mutlass::arch::OpMultiplyAddComplexFastF32> {
  static MathOperationID const kId = MathOperationID::kMultiplyAddComplexFastF32;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct LayoutMap;

template <> struct LayoutMap<mutlass::layout::ColumnMajor> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajor;
};

template <> struct LayoutMap<mutlass::layout::RowMajor> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajor;
};

template <> struct LayoutMap<mutlass::layout::ColumnMajorInterleaved<2>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK2;
};

template <> struct LayoutMap<mutlass::layout::RowMajorInterleaved<2>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK2;
};

template <> struct LayoutMap<mutlass::layout::ColumnMajorInterleaved<4>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK4;
};

template <> struct LayoutMap<mutlass::layout::RowMajorInterleaved<4>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK4;
};

template <> struct LayoutMap<mutlass::layout::ColumnMajorInterleaved<16>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK16;
};

template <> struct LayoutMap<mutlass::layout::RowMajorInterleaved<16>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK16;
};

template <> struct LayoutMap<mutlass::layout::ColumnMajorInterleaved<32>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK32;
};

template <> struct LayoutMap<mutlass::layout::RowMajorInterleaved<32>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK32;
};

template <> struct LayoutMap<mutlass::layout::ColumnMajorInterleaved<64>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK64;
};

template <> struct LayoutMap<mutlass::layout::RowMajorInterleaved<64>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK64;
};

template <> struct LayoutMap<mutlass::layout::TensorNHWC> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorNHWC;
};

template <> struct LayoutMap<mutlass::layout::TensorNDHWC> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorNDHWC;
};

template <> struct LayoutMap<mutlass::layout::TensorNCxHWx<32>> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorNC32HW32;
};

template <> struct LayoutMap<mutlass::layout::TensorNCxHWx<64>> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorNC64HW64;
};

template <> struct LayoutMap<mutlass::layout::TensorCxRSKx<32>> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorC32RSK32;
};

template <> struct LayoutMap<mutlass::layout::TensorCxRSKx<64>> {
  static LayoutTypeID const kId = LayoutTypeID::kTensorC64RSK64;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct OpcodeClassMap;

template <> struct OpcodeClassMap<arch::OpClassSimt> {
  static OpcodeClassID const kId = OpcodeClassID::kSimt;
};

template <> struct OpcodeClassMap<arch::OpClassTensorOp> {
  static OpcodeClassID const kId = OpcodeClassID::kTensorOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <mutlass::ComplexTransform Transform> struct ComplexTransformMap;

template <> struct ComplexTransformMap<mutlass::ComplexTransform::kNone> {
  static mutlass::library::ComplexTransform const kId = mutlass::library::ComplexTransform::kNone;
};

template <> struct ComplexTransformMap<mutlass::ComplexTransform::kConjugate> {
  static mutlass::library::ComplexTransform const kId = mutlass::library::ComplexTransform::kConjugate;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <mutlass::conv::Mode  T> struct ConvModeMap;

template <> struct ConvModeMap<conv::Mode::kCrossCorrelation> {
  static ConvModeID const kId = ConvModeID::kCrossCorrelation;
};

template <> struct ConvModeMap<conv::Mode::kConvolution> {
  static ConvModeID const kId = ConvModeID::kConvolution;
};


template <mutlass::conv::Operator  T> struct ConvKindMap;

template <> struct ConvKindMap<conv::Operator::kFprop> {
  static ConvKind const kId = ConvKind::kFprop;
};

template <> struct ConvKindMap<conv::Operator::kDgrad> {
  static ConvKind const kId = ConvKind::kDgrad;
};

template <> struct ConvKindMap<conv::Operator::kWgrad> {
  static ConvKind const kId = ConvKind::kWgrad;
};


template <mutlass::conv::IteratorAlgorithm  T> struct IteratorAlgorithmMap;

template <> struct IteratorAlgorithmMap<conv::IteratorAlgorithm::kAnalytic> {
  static IteratorAlgorithmID const kId = IteratorAlgorithmID::kAnalytic;
};

template <> struct IteratorAlgorithmMap<conv::IteratorAlgorithm::kOptimized> {
  static IteratorAlgorithmID const kId = IteratorAlgorithmID::kOptimized;
};

template <> struct IteratorAlgorithmMap<conv::IteratorAlgorithm::kFixedChannels> {
  static IteratorAlgorithmID const kId = IteratorAlgorithmID::kFixedChannels;
};

template <> struct IteratorAlgorithmMap<conv::IteratorAlgorithm::kFewChannels> {
  static IteratorAlgorithmID const kId = IteratorAlgorithmID::kFewChannels;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename Layout>
TensorDescription make_TensorDescription(int alignment = 1) {
  TensorDescription desc;

  desc.element = NumericTypeMap<Element>::kId;
  desc.layout = LayoutMap<Layout>::kId;
  desc.alignment = alignment;
  desc.log_extent_range = int(sizeof(typename Layout::TensorCoord::Index) - 1) * 8;
  desc.log_stride_range = int(sizeof(typename Layout::Stride::Index) - 1) * 8;

  return desc;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
