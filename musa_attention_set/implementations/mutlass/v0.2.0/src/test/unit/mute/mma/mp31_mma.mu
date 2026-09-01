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

#include "mutlass_unit_test.h"
#include "mma_inst_rr_test.hpp"
#include <mute/atom/mma_atom.hpp>

using namespace mute;

TEST(MP31_MuTe_MMA, f16f16_16x8x8_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x8_F32F16F16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x8_F32F16F16F32_TT SUCCESS\n");
  }

  {
    using Traits = TN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x8_F32F16F16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x8_F32F16F16F32_TN SUCCESS\n");
  }

}

TEST(MP31_MuTe_MMA, f16f16_8x16x16_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32F16F16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32F16F16F32_TT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32F16F16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32F16F16F32_NN SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, bf16bf16_8x16x16_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32BF16BF16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32BF16BF16F32_TT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32BF16BF16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32BF16BF16F32_NN SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, f16f16_16x16x32_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x32_F32F16F16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x32_F32F16F16F32_TT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x32_F32F16F16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x32_F32F16F16F32_NN SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, bf16bf16_16x16x32_mma_inst_test) {

  {
    using Traits = NT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x32_F32BF16BF16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x32_F32BF16BF16F32_NT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x32_F32BF16BF16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x32_F32BF16BF16F32_NN SUCCESS\n");
  }
}



TEST(MP31_MuTe_MMA, u8u8_16x8x16_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x16_U32U8U8U32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x16_U32U8U8U32_TT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x16_U32U8U8U32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x16_U32U8U8U32_NN SUCCESS\n");
  }
}



TEST(MP31_MuTe_MMA, s8s8_8x16x16_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_S32S8S8S32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_S32S8S8S32_TT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_S32S8S8S32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_S32S8S8S32_NN SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, u8u8_8x16x16_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_U32U8U8U32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_U32U8U8U32_TT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_U32U8U8U32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_U32U8U8U32_NN SUCCESS\n");
  }
}


TEST(MP31_MuTe_MMA, e4m3e4m3_8x16x16_mma_inst_test) {

  {
    using Traits = TN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32E4M3E4M3F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32E4M3E4M3F32_TN SUCCESS\n");
  }

  {
    using Traits = NT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32E4M3E4M3F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32E4M3E4M3F32_NT SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, s8s8_16x16x16_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x16_S32S8S8S32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x16_S32S8S8S32_TT SUCCESS\n");
  }

  {
    using Traits = TN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x16_S32S8S8S32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x16_S32S8S8S32_TN SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, u8u8_16x16x16_mma_inst_test) {

  {
    using Traits = NT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x16_U32U8U8U32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x16_U32U8U8U32_NT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x16_U32U8U8U32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x16_U32U8U8U32_NN SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, tf32tf32_16x8x4_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x4_F32TF32TF32F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x4_F32TF32TF32F32_TT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x4_F32TF32TF32F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x4_F32TF32TF32F32_NN SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, tf32tf32_16x8x8_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x8_F32TF32TF32F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x8_F32TF32TF32F32_TT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x8_F32TF32TF32F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x8_F32TF32TF32F32_NN SUCCESS\n");
  }
}


TEST(MP31_MuTe_MMA, f16s8_16x8x16_mma_inst_test) {

  {
    using Traits = TN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x16_F32F16S8F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x16_F32F16S8F32_TN SUCCESS\n");
  }

  {
    using Traits = NT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x16_F32F16S8F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x16_F32F16S8F32_NT SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, s8bf16_16x8x16_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x16_F32S8BF16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x16_F32S8BF16F32_TT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x8x16_F32S8BF16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x8x16_F32S8BF16F32_NN SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, s8f16_8x16x16_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32S8F16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32S8F16F32_TT SUCCESS\n");
  }

  {
    using Traits = NT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32S8F16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32S8F16F32_NT SUCCESS\n");
  }

}

TEST(MP31_MuTe_MMA, bf16s8_8x16x16_mma_inst_test) {

  {
    using Traits = NT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32BF16S8F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32BF16S8F32_NT SUCCESS\n");
  }

  {
    using Traits = NN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32BF16S8F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32BF16S8F32_NN SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, s8bf16_8x16x16_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32S8BF16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32S8BF16F32_TT SUCCESS\n");
  }

  {
    using Traits = TN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_8x16x16_F32S8BF16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_8x16x16_F32S8BF16F32_TN SUCCESS\n");
  }
}


TEST(MP31_MuTe_MMA, s8f16_16x16x16_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x16_F32S8F16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x16_F32S8F16F32_TT SUCCESS\n");
  }

  {
    using Traits = TN_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x16_F32S8F16F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x16_F32S8F16F32_TN SUCCESS\n");
  }
}


TEST(MP31_MuTe_MMA, f16s8_16x16x32_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x32_F32F16S8F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x32_F32F16S8F32_TT SUCCESS\n");
  }

  {
    using Traits = NT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x32_F32F16S8F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x32_F32F16S8F32_NT SUCCESS\n");
  }
}

TEST(MP31_MuTe_MMA, bf16s8_16x16x32_mma_inst_test) {

  {
    using Traits = TT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x32_F32BF16S8F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x32_F32BF16S8F32_TT SUCCESS\n");
  }

  {
    using Traits = NT_Traits;
    EXPECT_TRUE((mma_test_body<MP31_16x16x32_F32BF16S8F32<Traits::AMajor, Traits::BMajor>, Traits>()));
    MUTLASS_TRACE_HOST("MuTe MMA MP31_16x16x32_F32BF16S8F32_NT SUCCESS\n");
  }
}

