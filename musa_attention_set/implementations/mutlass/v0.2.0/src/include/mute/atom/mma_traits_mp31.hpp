/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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

#include <mute/arch/mma_mp31.hpp>
#include <mute/atom/mma_traits.hpp>
#include <mute/tensor.hpp>

namespace mute {

namespace {
template <int Bits, int M, int K, TCE::Major tnsp>
using MP31_MxK = mute::conditional_t<tnsp == TCE::Major::K,
  // Row layout
  Layout<Shape <Shape <                _4, _8>, Shape <Int<32 / Bits>, Int<K * Bits / 128>, Int<M / 8>>>,
         Stride<Stride<Int<M * 32 / Bits>, _1>, Stride<        Int<M>, Int<128 * M / Bits>,        _8>>>,
  // Column layout
  Layout<Shape <Shape <Int< Bits / 4>, Int<128 / Bits>>,Shape <Int<32 / Bits>, Int<M / 8>, Int<K * Bits / 128>>>,
         Stride<Stride<Int<32 / Bits>, Int<         M>>,Stride<            _1,         _8, Int<128 * M / Bits>>>>
>;

template <int M, int N>
using MP31_MxN = Layout<Shape <Shape <    _8, _4>, Shape <Int<N/8>, Int<M/4>>>,
                        Stride<Stride<Int<M>, _1>, Stride<Int<M*8>,      _4>>>;
}

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x8_F32F16F16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _8>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 8, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 8, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x8_F32BF16BF16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _8>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 8, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 8, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_F32F16F16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_F32BF16BF16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_F32F16F16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_F32BF16BF16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32F16F16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32BF16BF16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32F16F16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32BF16BF16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_S32S8S8S32<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_U32U8U8U32<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_F32E5M2E5M2F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_F32E4M3E4M3F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_F32E4M3E5M2F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_F32E5M2E4M3F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_S32S8S8S32<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_U32U8U8U32<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_F32E5M2E5M2F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_F32E4M3E4M3F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_F32E4M3E5M2F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_F32E5M2E4M3F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_S32S8S8S32<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_U32U8U8U32<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32E5M2E5M2F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32E4M3E4M3F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32E4M3E5M2F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32E5M2E4M3F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_S32S8S8S32<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_U32U8U8U32<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32E5M2E5M2F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32E4M3E4M3F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32E4M3E5M2F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32E5M2E4M3F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x64_S32S8S8S32<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16, _16, _64>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 64, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 64, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x64_U32U8U8U32<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using Shape_MNK = Shape<_16, _16, _64>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 64, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 64, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x64_F32E5M2E5M2F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _64>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 64, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 64, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x64_F32E4M3E4M3F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _64>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 64, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 64, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x64_F32E4M3E5M2F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _64>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 64, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 64, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x64_F32E5M2E4M3F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _64>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 64, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 64, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x4_F32TF32TF32F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _4>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 4, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 4, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x8_F32TF32TF32F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _8>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 8, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 8, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32TF32TF32F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32F16S4F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32S4F16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32BF16S4F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32S4BF16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_F32F16S8F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_F32S8F16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_F32BF16S8F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x8x16_F32S8BF16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>,  8, 16, tnspB>;
  using CLayout = MP31_MxN<16, 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_F32F16S8F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_F32S8F16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_F32BF16S8F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_8x16x16_F32S8BF16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>,  8, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<8, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32F16S8F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32S8F16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32BF16S8F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x16_F32S8BF16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _16>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 16, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 16, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32F16S8F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32S8F16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32BF16S8F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x16x32_F32S8BF16F32<tnspA, tnspB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_32>;
  using ALayout = MP31_MxK<sizeof_bits_v<ValTypeA>, 16, 32, tnspA>;
  using BLayout = MP31_MxK<sizeof_bits_v<ValTypeB>, 16, 32, tnspB>;
  using CLayout = MP31_MxN<16, 16>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mute
