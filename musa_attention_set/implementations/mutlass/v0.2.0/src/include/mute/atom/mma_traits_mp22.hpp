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

#include <mute/arch/mma_mp22.hpp>
#include <mute/atom/mma_traits.hpp>

#include <mute/layout.hpp>

namespace mute {

namespace {
  using MP22_32x32_32b = Layout<Shape <Shape < _8, _4,   _2, _2>, Shape <  _2, _4>>,
                                Stride<Stride<_32, _1, _256, _4>, Stride<_512, _8>>>;

  using MP22_32x16_Row = Layout<Shape <Shape < _4, _16,   _2>, Shape < _2,  _2>>,
                                Stride<Stride<_64,  _1, _256>, Stride<_32, _16>>>;

  using MP22_32x16_Col = Layout<Shape <Shape <_8, _16>, Shape <_2,  _2>>,
                                Stride<Stride<_2, _32>, Stride<_1, _16>>>;

  using MP22_32x8_Row  = Layout<Shape <Shape < _4, _16,   _2>,  _2>,
                                Stride<Stride<_32,  _1, _128>, _16>>;

  using MP22_32x8_Col  = Layout<Shape <Shape <_16,  _8>,  _2>,
                                Stride<Stride< _1, _32>, _16>>;

  using MP22_32x32_Row = Layout<Shape <Shape <  _4, _16,   _2>, Shape < _4,  _2>>,
                                Stride<Stride<_128,  _1, _512>, Stride<_32, _16>>>;

  using MP22_32x32_Col = Layout<Shape <Shape <_4, _32>, Shape <_4,  _2>>,
                                Stride<Stride<_4, _32>, Stride<_1, _16>>>;
}

/////////////////////////////////////////////////////////////////////////////////
////////////////////////// fp32 = fp16 * fp16 + fp32 ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<MP22_32x32x16_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32, _32, _16>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x16_Row;
  using BLayout = MP22_32x16_Col;
  using CLayout = MP22_32x32_32b;
};

template <>
struct MMA_Traits<MP22_32x32x16_F32F16F16F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32, _32, _16>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x16_Row;
  using BLayout = MP22_32x16_Row;
  using CLayout = MP22_32x32_32b;
};

template <>
struct MMA_Traits<MP22_32x32x16_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32, _32, _16>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x16_Col;
  using BLayout = MP22_32x16_Col;
  using CLayout = MP22_32x32_32b;
};

template <>
struct MMA_Traits<MP22_32x32x16_F32F16F16F32_NN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32, _32, _16>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x16_Col;
  using BLayout = MP22_32x16_Row;
  using CLayout = MP22_32x32_32b;
};

/////////////////////////////////////////////////////////////////////////////////
////////////////////////// fp32 = bf16 * bf16 + fp32 ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<MP22_32x32x16_F32BF16BF16F32_TT>
     : MMA_Traits<MP22_32x32x16_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<MP22_32x32x16_F32BF16BF16F32_TN>
     : MMA_Traits<MP22_32x32x16_F32F16F16F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<MP22_32x32x16_F32BF16BF16F32_NT>
     : MMA_Traits<MP22_32x32x16_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<MP22_32x32x16_F32BF16BF16F32_NN>
     : MMA_Traits<MP22_32x32x16_F32F16F16F32_NN>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

/////////////////////////////////////////////////////////////////////////////////
////////////////////////// fp32 = tf32 * tf32 + fp32 ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<MP22_32x32x8_F32TF32TF32F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32, _32, _8>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x8_Row;
  using BLayout = MP22_32x8_Col;
  using CLayout = MP22_32x32_32b;
};

template <>
struct MMA_Traits<MP22_32x32x8_F32TF32TF32F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32, _32, _8>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x8_Row;
  using BLayout = MP22_32x8_Row;
  using CLayout = MP22_32x32_32b;
};

template <>
struct MMA_Traits<MP22_32x32x8_F32TF32TF32F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32, _32, _8>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x8_Col;
  using BLayout = MP22_32x8_Col;
  using CLayout = MP22_32x32_32b;
};

template <>
struct MMA_Traits<MP22_32x32x8_F32TF32TF32F32_NN>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32, _32, _8>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x8_Col;
  using BLayout = MP22_32x8_Row;
  using CLayout = MP22_32x32_32b;
};


/////////////////////////////////////////////////////////////////////////////////
////////////////////////// s32 = s8 * s8 + s32 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<MP22_32x32x32_S32S8S8S32_TT>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x32_Row;
  using BLayout = MP22_32x32_Col;
  using CLayout = MP22_32x32_32b;
};

template <>
struct MMA_Traits<MP22_32x32x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x32_Row;
  using BLayout = MP22_32x32_Row;
  using CLayout = MP22_32x32_32b;
};

template <>
struct MMA_Traits<MP22_32x32x32_S32S8S8S32_NT>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x32_Col;
  using BLayout = MP22_32x32_Col;
  using CLayout = MP22_32x32_32b;
};

template <>
struct MMA_Traits<MP22_32x32x32_S32S8S8S32_NN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID = Layout<_128>;
  using ALayout = MP22_32x32_Col;
  using BLayout = MP22_32x32_Row;
  using CLayout = MP22_32x32_32b;
};

} // namespace mute
