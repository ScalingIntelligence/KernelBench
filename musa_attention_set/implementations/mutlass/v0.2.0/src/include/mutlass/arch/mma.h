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
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once

#include "mutlass/array.h"
#include "mutlass/numeric_types.h"
#include "mutlass/functional.h"

#include "mutlass/gemm/gemm.h"
#include "mutlass/arch/arch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the operation implied by MMA.
struct OpMultiplyAdd {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the result is saturated to MAX_FLOAT|MIN_FLOAT or MAX_INT|MIN_INT
struct OpMultiplyAddSaturate {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the input is converted to a narrower type (BF16)
struct OpMultiplyAddFastBF16 {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the input is converted to a narrower type (F16)
struct OpMultiplyAddFastF16 {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the input data types are mixed and the narrower type is 
/// upcasted to the wider type
struct OpMultiplyAddMixedInputUpcast {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the input is converted to 2 (big and small) TF32 components
//  Perform 3xTF32 or 4xTF32 for every F32 output element
struct OpMultiplyAddFastF32 {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the input is converted to 2 (big and small) TF32 components
//  Perform 3xTF32 or 4xTF32 for every complex<F32> output element
struct OpMultiplyAddComplexFastF32 {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper for determining whether staged accumulation should be used for a given operator
template <typename Operator>
struct UseStagedAccumulation {
  static bool const value = platform::is_same<Operator, OpMultiplyAddFastF32>::value ||
                            platform::is_same<Operator, OpMultiplyAddComplexFastF32>::value;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the complex multiply-add operation
struct OpMultiplyAddComplex {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the gaussian complex multiply-add operation
struct OpMultiplyAddGaussianComplex {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the inner product is defined by (XOR, POPC)
struct OpXorPopc {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the inner product is defined by (AND, POPC)
struct OpAndPopc {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag classifying math operators as thread-level operations.
struct OpClassSimt {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag classifying operators as Tensor Core operations.
struct OpClassTensorOp {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag classifying operators as Tensor Core with structure sparse operations.
struct OpClassSparseTensorOp {};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

