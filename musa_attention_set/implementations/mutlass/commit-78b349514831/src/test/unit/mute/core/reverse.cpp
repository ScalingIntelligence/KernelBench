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
#include "mutlass/trace.h"

#include "mute/algorithm/tuple_algorithms.hpp"
#include "mute/container/array.hpp"
#include "mute/container/tuple.hpp"

TEST(MuTe_core, Reverse_Tuple)
{
  using mute::get;

  {
    const auto t = mute::make_tuple();
    [[maybe_unused]] auto t_r = mute::reverse(t);
    static_assert(mute::tuple_size_v<decltype(t_r)> == 0);
  }

  {
    const auto t = mute::make_tuple(123);
    [[maybe_unused]] auto t_r = mute::reverse(t);
    static_assert(mute::tuple_size_v<decltype(t_r)> == 1);
    EXPECT_EQ(get<0>(t_r), 123);
  }

  {
    const auto t = mute::make_tuple(123, 456);
    [[maybe_unused]] auto t_r = mute::reverse(t);
    static_assert(mute::tuple_size_v<decltype(t_r)> == 2);
    EXPECT_EQ(get<0>(t_r), 456);
    EXPECT_EQ(get<1>(t_r), 123);
  }

  {
    const auto t = mute::make_tuple(1, 2, 3, 4, 5);
    auto t_r = mute::reverse(t);
    static_assert(mute::tuple_size_v<decltype(t_r)> == 5);

    EXPECT_EQ(get<0>(t_r), 5);
    EXPECT_EQ(get<1>(t_r), 4);
    EXPECT_EQ(get<2>(t_r), 3);
    EXPECT_EQ(get<3>(t_r), 2);
    EXPECT_EQ(get<4>(t_r), 1);
  }

  {
    const auto t = mute::make_tuple(mute::Int<1>{}, mute::Int<2>{}, 3);
    auto t_r = mute::reverse(t);
    static_assert(mute::tuple_size_v<decltype(t_r)> == 3);
    static_assert(mute::is_same_v<mute::remove_cvref_t<decltype(get<0>(t_r))>, int>);
    static_assert(mute::is_same_v<mute::remove_cvref_t<decltype(get<1>(t_r))>, mute::Int<2>>);
    static_assert(mute::is_same_v<mute::remove_cvref_t<decltype(get<2>(t_r))>, mute::Int<1>>);

    EXPECT_EQ(get<0>(t_r), 3);
    EXPECT_EQ(get<1>(t_r), mute::Int<2>{});
    EXPECT_EQ(get<2>(t_r), mute::Int<1>{});
  }
}

TEST(MuTe_core, Reverse_Array)
{
  using mute::get;

  {
    const auto t = mute::array<int, 0>{};
    [[maybe_unused]] auto t_r = mute::reverse(t);
    static_assert(mute::tuple_size_v<decltype(t_r)> == 0);

    using reverse_type = mute::array<int, 0>;
    static_assert(mute::is_same_v<decltype(t_r), reverse_type>);
  }

  {
    const auto t = mute::array<int, 1>{123};
    [[maybe_unused]] auto t_r = mute::reverse(t);
    static_assert(mute::tuple_size_v<decltype(t_r)> == 1);
    EXPECT_EQ(get<0>(t_r), 123);

    using reverse_type = mute::array<int, 1>;
    static_assert(mute::is_same_v<decltype(t_r), reverse_type>);
  }

  {
    const auto t = mute::array<int, 2>{123, 456};
    [[maybe_unused]] auto t_r = mute::reverse(t);
    static_assert(mute::tuple_size_v<decltype(t_r)> == 2);
    EXPECT_EQ(get<0>(t_r), 456);
    EXPECT_EQ(get<1>(t_r), 123);

    using reverse_type = mute::array<int, 2>;
    static_assert(mute::is_same_v<decltype(t_r), reverse_type>);
  }

  {
    const auto t = mute::array<float, 5>{1.125f, 2.25f, 3.5f, 4.625f, 5.75f};
    auto t_r = mute::reverse(t);
    static_assert(mute::tuple_size_v<decltype(t_r)> == 5);
    EXPECT_EQ(get<0>(t_r), 5.75f);
    EXPECT_EQ(get<1>(t_r), 4.625f);
    EXPECT_EQ(get<2>(t_r), 3.5f);
    EXPECT_EQ(get<3>(t_r), 2.25f);
    EXPECT_EQ(get<4>(t_r), 1.125f);

    using reverse_type = mute::array<float, 5>;
    static_assert(mute::is_same_v<decltype(t_r), reverse_type>);
  }
}
