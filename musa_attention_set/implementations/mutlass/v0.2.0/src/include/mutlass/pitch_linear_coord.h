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
    \brief Defines layout functions used by TensorRef and derived classes for pitch-linear memory.
*/
#pragma once

#include "mutlass/mutlass.h"
#include "mutlass/coord.h"

namespace mutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template defining a shape used by pitch-linear operators
template <
  int Contiguous,
  int Strided
>
struct PitchLinearShape {
  static int const kContiguous = Contiguous;
  static int const kStrided = Strided;
  static int const kCount = Contiguous * Strided;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Coordinate in pitch-linear space
struct PitchLinearCoord : public Coord<2, int> {
public:

  /// Integer-valued index
  using Index = int;

  /// Base type is a Coord of rank=2
  using Base = Coord<2, Index>;

  /// Long integer type
  using LongIndex = typename Base::LongIndex;

private:

  /// Rows dimension
  static int const kContiguous = 0;

  /// Columns dimension
  static int const kStrided = 1;

public:

  //
  // Methods
  //

  /// Default ctor
  MUTLASS_HOST_DEVICE
  PitchLinearCoord() { }

  /// Constructs from Coord<2>
  MUTLASS_HOST_DEVICE
  PitchLinearCoord(Coord<2, Index> const &coord): Base(coord) { }

  /// Helper to construct from a row and column
  MUTLASS_HOST_DEVICE
  PitchLinearCoord(Index contiguous_, Index strided_): Base(make_Coord(contiguous_, strided_)) { }

  /// Helper to construct from a row and column based on LongIndex
  MUTLASS_HOST_DEVICE
  PitchLinearCoord(LongIndex contiguous_, LongIndex strided_)
    : Base(make_Coord(Index(contiguous_), Index(strided_))) { }

  /// Returns the contiguous dimension
  MUTLASS_HOST_DEVICE
  Index const & contiguous() const { return this->at(kContiguous); }

  /// Returns the contiguous dimension
  MUTLASS_HOST_DEVICE
  Index & contiguous() { return this->at(kContiguous); }

  /// Returns the column of the coordinate
  MUTLASS_HOST_DEVICE
  Index const & strided() const { return this->at(kStrided); }

  /// Returns the column of the coordinate
  MUTLASS_HOST_DEVICE
  Index & strided() { return this->at(kStrided); }

  //
  // Coord operators
  //

  /// Element-wise addition
  MUTLASS_HOST_DEVICE
  PitchLinearCoord operator+(Base const& b) const {
    return PitchLinearCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  MUTLASS_HOST_DEVICE
  PitchLinearCoord operator-(Base const& b) const {
    return PitchLinearCoord(Base::operator-(b));
  }

  MUTLASS_HOST_DEVICE
  PitchLinearCoord operator-() const {
    return PitchLinearCoord(-at(0), -at(1));
  }

  /// Element-wise multiplication
  MUTLASS_HOST_DEVICE
  PitchLinearCoord operator*(Base const& b) const {
    return PitchLinearCoord(Base::operator*(b));
  }

  /// Element-wise division
  MUTLASS_HOST_DEVICE
  PitchLinearCoord operator/(Base const& b) const {
    return PitchLinearCoord(Base::operator/(b));
  }

  /// In-place addition
  MUTLASS_HOST_DEVICE
  PitchLinearCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  MUTLASS_HOST_DEVICE
  PitchLinearCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  MUTLASS_HOST_DEVICE
  PitchLinearCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  MUTLASS_HOST_DEVICE
  PitchLinearCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass

