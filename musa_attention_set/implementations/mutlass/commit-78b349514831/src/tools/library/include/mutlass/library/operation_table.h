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
/*
  \file
  \brief Defines a data structure in which a set of functionally equivalent library::Operation
        instances may be queried.
*/

#pragma once
#include <fstream>
#include <iosfwd>
#include <unordered_map>
#include <algorithm>

#include "mutlass/library/library.h"
#include "mutlass/library/manifest.h"
#include "mutlass/library/util.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////
//                          Data Structures for Gemm Functional Maps
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tuple uniquely identifying Gemm functional behavior
struct GemmFunctionalKey {

  Provider provider;
  GemmKind gemm_kind;
  NumericTypeID element_compute;
  NumericTypeID element_scalar;
  NumericTypeID element_A;
  LayoutTypeID layout_A;
  ComplexTransform transform_A;
  NumericTypeID element_B;
  LayoutTypeID layout_B;
  ComplexTransform transform_B;
  NumericTypeID element_C;
  LayoutTypeID layout_C;
  NumericTypeID element_D;
  LayoutTypeID layout_D;

  //
  // Methods
  //

  inline
  GemmFunctionalKey(
    Provider provider,
    GemmKind gemm_kind = GemmKind::kGemm,
    NumericTypeID element_compute = NumericTypeID::kF32,
    NumericTypeID element_scalar = NumericTypeID::kF32,
    NumericTypeID element_A = NumericTypeID::kF16,
    LayoutTypeID layout_A = LayoutTypeID::kColumnMajor,
    ComplexTransform transform_A = ComplexTransform::kNone,
    NumericTypeID element_B = NumericTypeID::kF16,
    LayoutTypeID layout_B = LayoutTypeID::kColumnMajor,
    ComplexTransform transform_B = ComplexTransform::kNone,
    NumericTypeID element_C = NumericTypeID::kF16,
    LayoutTypeID layout_C = LayoutTypeID::kColumnMajor,
    NumericTypeID element_D = NumericTypeID::kF16,
    LayoutTypeID layout_D = LayoutTypeID::kColumnMajor
  ):
    provider(provider),
    gemm_kind(gemm_kind),
    element_compute(element_compute),
    element_scalar(element_scalar),
    element_A(element_A),
    layout_A(layout_A),
    transform_A(transform_A),
    element_B(element_B),
    layout_B(layout_B),
    transform_B(transform_B),
    element_C(element_C),
    layout_C(layout_C),
    element_D(element_D),
    layout_D(layout_D)
  { }

  inline
  bool operator==(GemmFunctionalKey const &rhs) const {
    return 
      (provider == rhs.provider) &&
      (gemm_kind == rhs.gemm_kind) &&
      (element_compute == rhs.element_compute) &&
      (element_scalar == rhs.element_scalar) &&
      (element_A == rhs.element_A) &&
      (layout_A == rhs.layout_A) &&
      (transform_A == rhs.transform_A) &&
      (element_B == rhs.element_B) &&
      (layout_B == rhs.layout_B) &&
      (transform_B == rhs.transform_B) &&
      (element_C == rhs.element_C) &&
      (layout_C == rhs.layout_C) &&
      (element_D == rhs.element_D) &&
      (layout_D == rhs.layout_D);
  }

  inline
  bool operator!=(GemmFunctionalKey const &rhs) const {
    return !(*this == rhs);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
inline
std::ostream & operator<<(std::ostream &out, mutlass::library::GemmFunctionalKey const &k) {

  out << "{\n"
    << "         provider: " << to_string(k.provider) << "\n"
    << "        gemm_kind: " << to_string(k.gemm_kind) << "\n"
    << "  element_compute: " << to_string(k.element_compute) << "\n"
    << "   element_scalar: " << to_string(k.element_scalar) << "\n"
    << "        element_A: " << to_string(k.element_A) << "\n"
    << "         layout_A: " << to_string(k.layout_A) << "\n"
    << "      transform_A: " << to_string(k.transform_A) << "\n"
    << "        element_B: " << to_string(k.element_B) << "\n"
    << "         layout_B: " << to_string(k.layout_B) << "\n"
    << "      transform_B: " << to_string(k.transform_B) << "\n"
    << "        element_C: " << to_string(k.element_C) << "\n"
    << "         layout_C: " << to_string(k.layout_C) << "\n"
    << "        element_D: " << to_string(k.element_D) << "\n"
    << "         layout_D: " << to_string(k.layout_D) << "\n"
    << "}";

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Hash function for GemmFunctionalKey
struct GemmFunctionalKeyHasher {
  using IntHash = std::hash<int>;

  inline
  static size_t rotl(size_t key, int shl) {
    return (key << shl) | (key >> (sizeof(key)*8 - shl));
  }

  inline
  size_t operator()(GemmFunctionalKey const &key) const {
    IntHash hash;

    return
      rotl(hash(int(key.provider)),        1) ^ 
      rotl(hash(int(key.gemm_kind)),       2) ^ 
      rotl(hash(int(key.element_compute)), 3) ^
      rotl(hash(int(key.element_scalar)),  4) ^
      rotl(hash(int(key.element_A)),       5) ^
      rotl(hash(int(key.layout_A)),        6) ^
      rotl(hash(int(key.transform_A)),     7) ^
      rotl(hash(int(key.element_B)),       8) ^
      rotl(hash(int(key.layout_B)),        9) ^
      rotl(hash(int(key.transform_B)),    10) ^
      rotl(hash(int(key.element_C)),      11) ^
      rotl(hash(int(key.layout_C)),       12) ^
      rotl(hash(int(key.element_D)),      13) ^
      rotl(hash(int(key.layout_D)),       14);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Establishes a partial ordering to search for GEMM operators
struct GemmPreferenceKey {

  int compute_capability;
  int alignment;

  //
  // Methods
  //

  GemmPreferenceKey(): compute_capability(), alignment() { }

  GemmPreferenceKey(int cc, int alignment): compute_capability(cc), alignment(alignment) { }

  bool operator<(GemmPreferenceKey const &rhs) const {
    return (compute_capability < rhs.compute_capability) || 
      ((compute_capability == rhs.compute_capability) && (alignment < rhs.alignment));
  }

  bool operator==(GemmPreferenceKey const &rhs) const {
    return compute_capability == rhs.compute_capability;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

inline
std::ostream& operator<< (std::ostream& out, const mutlass::library::GemmPreferenceKey& key) {
    out << "{\n"
      << "compute_capability : " << key.compute_capability << std::endl
      << "alignment          : " << key.alignment << std::endl
      << "}";

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Maps minimum compute capability onto a vector of possible operations
using GemmOperationVectorMap = std::map<
  GemmPreferenceKey,
  std::vector<Operation const *>
>;

/// Maps a GemmFunctionalKey onto a vector of Operation * objects expected to be of kind kGemm
using GemmOperationFunctionalMap = std::unordered_map<
  GemmFunctionalKey,
  GemmOperationVectorMap,
  GemmFunctionalKeyHasher
>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Table of mutlass::library::Operation instances
class OperationTable {
public:

  /// Map of all operations of type kGemm 
  // provider (kMUTLASS)
  GemmOperationFunctionalMap gemm_operations;
  
public:

  void append(Manifest const &manifest);

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream & operator<<(std::ostream &out, mutlass::library::GemmFunctionalKey const &k);
