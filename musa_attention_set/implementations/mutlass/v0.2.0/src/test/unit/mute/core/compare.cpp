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

#include "mutlass_unit_test.h"

#include <mutlass/trace.h>
#include <mute/tensor.hpp>

TEST(MuTe_core, Compare_simple_2d_GenColMajor)
{
  using namespace mute;

  // Simple 2D layout
  auto layout = make_layout(make_shape(Int<3>{}, Int<5>{}), GenColMajor{});
  MUTLASS_TRACE_HOST("Layout: " << layout);

  for (int i = 0; i < size(layout); ++i) {
    auto coord_i = layout.get_hier_coord(i);

    MUTLASS_TRACE_HOST(i << ": " << coord_i);

    EXPECT_TRUE(elem_less(coord_i, shape(layout)));

    for (int j = 0; j < size(layout); ++j) {
      auto coord_j = layout.get_hier_coord(j);
      MUTLASS_TRACE_HOST("  " << j << ": " << coord_j);
      EXPECT_TRUE(elem_less(coord_j, shape(layout)));

      EXPECT_EQ((i < j), colex_less(coord_i,coord_j));
    }
  }
}


TEST(MuTe_core, Compare_simple_2d_GenRowMajor)
{
  using namespace mute;

  auto layout = make_layout(make_shape(Int<3>{}, Int<5>{}), GenRowMajor{});
  MUTLASS_TRACE_HOST("Layout: " << layout);

  for (int i = 0; i < size(layout); ++i) {
    auto coord_i = layout.get_hier_coord(i);
    MUTLASS_TRACE_HOST(i << ": " << coord_i);
    EXPECT_TRUE(elem_less(coord_i, shape(layout)));

    for (int j = 0; j < size(layout); ++j) {
      auto coord_j = layout.get_hier_coord(j);
      EXPECT_TRUE(elem_less(coord_j, shape(layout)));

      EXPECT_EQ((i < j), lex_less(coord_i,coord_j));
    }
  }
}


TEST(MuTe_core, Compare_simple_3d_GenColMajor)
{
  using namespace mute;

  auto layout = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<5>{}), GenColMajor{});
  MUTLASS_TRACE_HOST("Layout: " << layout);

  for (int i = 0; i < size(layout); ++i) {
    auto coord_i = layout.get_hier_coord(i);
    MUTLASS_TRACE_HOST(i << ": " << coord_i);
    EXPECT_TRUE(elem_less(coord_i, shape(layout)));

    for (int j = 0; j < size(layout); ++j) {
      auto coord_j = layout.get_hier_coord(j);
      EXPECT_TRUE(elem_less(coord_j, shape(layout)));

      EXPECT_EQ((i < j), colex_less(coord_i,coord_j));
    }
  }
}


TEST(MuTe_core, Compare_simple_3d_GenRowMajor)
{
  using namespace mute;

  auto layout = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<5>{}), GenRowMajor{});
  MUTLASS_TRACE_HOST("Layout: " << layout);

  for (int i = 0; i < size(layout); ++i) {
    auto coord_i = layout.get_hier_coord(i);
    MUTLASS_TRACE_HOST(i << ": " << coord_i);
    EXPECT_TRUE(elem_less(coord_i, shape(layout)));

    for (int j = 0; j < size(layout); ++j) {
      auto coord_j = layout.get_hier_coord(j);
      EXPECT_TRUE(elem_less(coord_j, shape(layout)));

      EXPECT_EQ((i < j), lex_less(coord_i,coord_j));
    }
  }
}


TEST(MuTe_core, Compare_hierarchical_3d_GenColMajor)
{
  using namespace mute;

  auto layout = make_layout(Shape<Shape<_3,_2>,Shape<_5,_2,_2>>{}, GenColMajor{});
  MUTLASS_TRACE_HOST("Layout: " << layout);

  for (int i = 0; i < size(layout); ++i) {
    auto coord_i = layout.get_hier_coord(i);
    MUTLASS_TRACE_HOST(i << ": " << coord_i);
    EXPECT_TRUE(elem_less(coord_i, shape(layout)));

    for (int j = 0; j < size(layout); ++j) {
      auto coord_j = layout.get_hier_coord(j);
      EXPECT_TRUE(elem_less(coord_j, shape(layout)));

      EXPECT_EQ((i < j), colex_less(coord_i,coord_j));
    }
  }
}

TEST(MuTe_core, Compare_hierarchical_3d_GenRowMajor)
{
  using namespace mute;
  auto layout = make_layout(Shape<Shape<_3,_2>,Shape<_5,_2,_2>>{}, GenRowMajor{});
  MUTLASS_TRACE_HOST("Layout: " << layout);

  for (int i = 0; i < size(layout); ++i) {
    auto coord_i = layout.get_hier_coord(i);
    MUTLASS_TRACE_HOST(i << ": " << coord_i);
    EXPECT_TRUE(elem_less(coord_i, shape(layout)));

    for (int j = 0; j < size(layout); ++j) {
      auto coord_j = layout.get_hier_coord(j);
      EXPECT_TRUE(elem_less(coord_j, shape(layout)));

      EXPECT_EQ((i < j), lex_less(coord_i,coord_j));
    }
  }
}
