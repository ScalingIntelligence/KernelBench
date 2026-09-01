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

#include "tme_load_testbed.hpp"

using namespace mute;

TEST(MP31_MuTe_TME_Load, TME_Load_1D) {

  {
    Layout smem_layout = Layout<Int<256>, Int<1>>{};
    {
      Layout gmem_layout = smem_layout;
      run_test_tme_load<int8_t>(gmem_layout, smem_layout);
      run_test_tme_load<half_t>(gmem_layout, smem_layout);
      run_test_tme_load< float>(gmem_layout, smem_layout);
      run_test_tme_load<double>(gmem_layout, smem_layout);
    }

    {
      Layout gmem_layout = make_layout(Int<128>{}, LayoutLeft{});
      run_test_tme_load<int8_t>(gmem_layout, smem_layout);
      run_test_tme_load<half_t>(gmem_layout, smem_layout);
      run_test_tme_load< float>(gmem_layout, smem_layout);
      run_test_tme_load<double>(gmem_layout, smem_layout);
    }

    {
      Layout gmem_layout = make_layout(Int<384>{}, LayoutLeft{});
      run_test_tme_load<int8_t>(gmem_layout, smem_layout);
      run_test_tme_load<half_t>(gmem_layout, smem_layout);
      run_test_tme_load< float>(gmem_layout, smem_layout);
      run_test_tme_load<double>(gmem_layout, smem_layout);
    }
  }
}

TEST(MP31_MuTe_TME_Load, TME_Load_32x32_Col) {

  {
    Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
    {
      Layout gmem_layout = smem_layout;
      run_test_tme_load<int8_t>(gmem_layout, smem_layout);
      run_test_tme_load<half_t>(gmem_layout, smem_layout);
      run_test_tme_load< float>(gmem_layout, smem_layout);
      run_test_tme_load<double>(gmem_layout, smem_layout);
    }

    {
      Layout gmem_layout = make_layout(make_shape(Int<32>{}, Int<32>{}), LayoutLeft{});
      run_test_tme_load<int8_t>(gmem_layout, smem_layout);
      run_test_tme_load<half_t>(gmem_layout, smem_layout);
      run_test_tme_load< float>(gmem_layout, smem_layout);
      run_test_tme_load<double>(gmem_layout, smem_layout);
    }

    {
      Layout gmem_layout = make_layout(make_shape(Int<32>{},Int<32>{}), make_stride(Int<1>{}, Int<1024>{}));
      run_test_tme_load<int8_t>(gmem_layout, smem_layout);
      run_test_tme_load<half_t>(gmem_layout, smem_layout);
      run_test_tme_load< float>(gmem_layout, smem_layout);
      run_test_tme_load<double>(gmem_layout, smem_layout);
    }
  }
}

TEST(MP31_MuTe_TME_Load, TME_Load_32x32_Row) {

  {
    Layout smem_layout = Layout<Shape<_32,_32>, Stride<_32,_1>>{};
    {
      Layout gmem_layout = smem_layout;
      run_test_tme_load<int8_t>(gmem_layout, smem_layout);
      run_test_tme_load<half_t>(gmem_layout, smem_layout);
      run_test_tme_load< float>(gmem_layout, smem_layout);
      run_test_tme_load<double>(gmem_layout, smem_layout);
    }

    {
      Layout gmem_layout = make_layout(make_shape(Int<32>{}, Int<32>{}), LayoutRight{});
      run_test_tme_load<int8_t>(gmem_layout, smem_layout);
      run_test_tme_load<half_t>(gmem_layout, smem_layout);
      run_test_tme_load< float>(gmem_layout, smem_layout);
      run_test_tme_load<double>(gmem_layout, smem_layout);
    }

    {
      Layout gmem_layout = make_layout(make_shape(Int<32>{},Int<32>{}), make_stride(Int<1024>{}, Int<1>{}));
      run_test_tme_load<int8_t>(gmem_layout, smem_layout);
      run_test_tme_load<half_t>(gmem_layout, smem_layout);
      run_test_tme_load< float>(gmem_layout, smem_layout);
      run_test_tme_load<double>(gmem_layout, smem_layout);
    }
  }
}

TEST(MP31_MuTe_TME_Load, TME_Load_Tensor)
{
  // 3-mode
  {
    Layout gmem_layout = make_layout(make_shape(Int<128>{}, Int<64>{}, Int<5>{}));
    auto cta_tile      = Shape<_64, _32>{};                                     // GMEM Tiling:
                                                                                //   Take 64-elem from m
                                                                                //   Take 32-elem from k
    auto smem_layout = make_layout(Shape<_64,_32>{});
    run_test_tme_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  // 4-mode
  {
    Layout gmem_layout = make_layout(make_shape(make_shape(Int<80>{},Int<40>{}),
                                     make_shape(Int<32>{},Int<12>{})));
    auto cta_tile      = Shape<Shape<_16,_8>,Shape<_32,_2>>{};                  // GMEM Tiling:
                                                                                //   Take 16-elem from m0, 8-elem from m1,
                                                                                //   Take 32-elem from k0, 2-elem from k1
    auto smem_layout = make_layout(Shape<_128,_64>{});
    run_test_tme_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  // 5-mode
  {
    Layout gmem_layout = make_layout(make_shape(make_shape(Int<32>{},Int<32>{},Int<32>{}),
                                     make_shape(Int<32>{},Int<12>{})));
    auto cta_tile      = Shape<Shape<_16,_4,_2>,Shape<_16,_2>>{};               // GMEM Tiling:
                                                                                //   Take 4-elem from m0, 4-elem from m1, 5-elem from m2
                                                                                //   Take 32-elem from k0, 2-elem from k1
    auto smem_layout = make_layout(Shape<_128,_32>{});
    run_test_tme_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }
}

TEST(MP31_MuTe_TME_Load, TME_Load_Tensor_Multimode)
{
  {
    Layout gmem_layout = make_layout(make_shape(make_shape(_32{}, _3{}, _2{}, _2{}),
                                     make_shape(_32{}, _4{}, _2{})));
    auto cta_tile      = Shape<Shape<_32>, Shape<_32,_2>>{};                    // GMEM Tiling:
                                                                                //  Take 32-elem from m0
                                                                                //  Take 32-elem from k0, 2-elem from k1
    auto smem_layout = make_layout(Shape<_32,_64>{});
    run_test_tme_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  {
    Layout gmem_layout = make_layout(make_shape(make_shape(_64{}, _3{}, _2{}, _2{}),
                                     make_shape(_32{}, _4{}, _2{})));
    auto cta_tile      = Shape<Shape<_32,_3>, Shape<_32,_2>>{};                 // GMEM Tiling:
                                                                                //  Take 32-elem from m0, 3-elem from m1
                                                                                //  Take 32-elem from k0, 2-elem from k1
    auto smem_layout = make_layout(Shape<_96,_64>{});
    run_test_tme_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  {
    Layout gmem_layout = make_layout(make_shape(make_shape(_64{}, _3{}, _2{}, _3{}, _2{}),
                                     make_shape(_32{}, _4{}, _2{}, _2{})));
    auto cta_tile      = Shape<Shape<_32>, Shape<_16,_2>>{};                    // GMEM Tiling:
                                                                                //  Take 32-elem from m0
                                                                                //  Take 16-elem from k0, 2-elem from k1
    auto smem_layout = make_layout(Shape<_32,_32>{});
    run_test_tme_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }
}

TEST(MP31_MuTe_TME_Load, TME_LOAD_Coalesce)
{
  // Interleaved ColMajor
  {
    Layout gmem_layout = make_layout(make_shape ( _128{}, make_shape (_4{},  _128{})),
                                     make_stride(   _4{}, make_stride(_1{},  _512{})));
    auto   smem_layout = make_layout(make_shape (  _32{}, make_shape (_4{},  _32{})),
                                     make_stride(   _4{}, make_stride(_1{},  _128{})));

    // By default, uses cta_tile = Shape<_32,_128>
    auto tme = run_test_tme_load<int8_t>(gmem_layout, smem_layout);
    // Check the TME rank
    EXPECT_EQ(rank(tme.get_tme_tensor(shape(gmem_layout))(0)), 2);
  }

  // Interleaved RowMajor
  {
    Layout gmem_layout = make_layout(make_shape (make_shape (_4{},   128),   128),
                                     make_stride(make_stride(_1{},   512),   _4{}));
    auto   smem_layout = make_layout(make_shape (make_shape (_4{},  _32{}), _32{}),
                                     make_stride(make_stride(_1{}, _128{}),  _4{}));

    // By default, uses cta_tile = Shape<_128,_32>
    auto tme = run_test_tme_load<int8_t>(gmem_layout, smem_layout);
    // Check the TME rank
    EXPECT_EQ(rank(tme.get_tme_tensor(shape(gmem_layout))(0)), 2);
  }

  // Account for stride-0 modes within the TME tile
  {
    Layout gmem_layout = make_layout(make_shape (  128, make_shape (_32{},   4)),
                                     make_stride( _1{}, make_stride( _0{}, 128)));
    auto   smem_layout = make_layout(make_shape (_64{}, make_shape (_32{}     )),
                                     make_stride( _1{}, make_stride( _0{}     )));

    // By default, uses cta_tile = Shape<_64,_32>
    auto tme = run_test_tme_load<uint16_t>(gmem_layout, smem_layout);
    // Check the TME rank
    EXPECT_EQ(rank(tme.get_tme_tensor(shape(gmem_layout))(0)), 2);
  }

  // Coalesce many modes and account for stride-0 modes within the TME tile
  {
    Layout gmem_layout = make_layout(make_shape (make_shape (_32{},_4{},     4), _32{}, make_shape (_4{},      4)),
                                     make_stride(make_stride(_16{},_4{},  2048),  _0{}, make_stride(_1{}, _512{})));
    auto   smem_layout = make_layout(make_shape (make_shape (_32{},_4{}       ), _32{}, make_shape (_4{}        )),
                                     make_stride(make_stride(_16{},_4{}       ),  _0{}, make_stride(_1{}        )));
    // By default, uses cta_tile = Shape<_128,_32,_4>
    auto tme = run_test_tme_load<int8_t>(gmem_layout, smem_layout);
    // Check the TME rank (Could be 3 instead of 4 with even better coalescing...?)
    EXPECT_EQ(rank(tme.get_tme_tensor(shape(gmem_layout))(0)), 4);
  }
}
