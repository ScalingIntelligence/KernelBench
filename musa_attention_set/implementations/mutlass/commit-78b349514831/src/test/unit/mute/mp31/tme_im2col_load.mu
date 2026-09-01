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

#include "tme_im2col_load_testbed.hpp"

/*

-------------------------------- 1D Cases --------------------------------

*/
TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_1D) {

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<128>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam =TmeIm2ColTestParam<1, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1},                                                  // [s]
                    {0},                                            // [l_pad_w]
                    {0},                                            // [u_pad_w]
                    {1},                                           // [stride_w]
                    {1},                                         // [dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 320 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<320>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam =TmeIm2ColTestParam<1, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1},                                                  // [s]
                    {0},                                            // [l_pad_w]
                    {0},                                            // [u_pad_w]
                    {1},                                           // [stride_w]
                    {1},                                         // [dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 128 C = 64
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<128>{}, Int<64>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam =TmeIm2ColTestParam<1, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1},                                                  // [s]
                    {0},                                            // [l_pad_w]
                    {0},                                            // [u_pad_w]
                    {1},                                           // [stride_w]
                    {1},                                         // [dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_1D_Kernel) {

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<130>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam =TmeIm2ColTestParam<1, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{3},                                                  // [s]
                    {0},                                            // [l_pad_w]
                    {0},                                            // [u_pad_w]
                    {1},                                           // [stride_w]
                    {1},                                         // [dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_1D_Padding) {

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<127>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam =TmeIm2ColTestParam<1, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1},                                                  // [s]
                    {1},                                            // [l_pad_w]
                    {0},                                            // [u_pad_w]
                    {1},                                           // [stride_w]
                    {1},                                         // [dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<127>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam =TmeIm2ColTestParam<1, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1},                                                  // [s]
                    {0},                                            // [l_pad_w]
                    {1},                                            // [u_pad_w]
                    {1},                                           // [stride_w]
                    {1},                                         // [dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<126>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam =TmeIm2ColTestParam<1, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1},                                                  // [s]
                    {1},                                            // [l_pad_w]
                    {1},                                            // [u_pad_w]
                    {1},                                           // [stride_w]
                    {1},                                         // [dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }
}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_1D_Stride) {

  {
    // M = 64 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<128>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<64>{}, Int<16>{}), LayoutRight{});
    using TestParam =TmeIm2ColTestParam<1, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1},                                                  // [s]
                    {0},                                            // [l_pad_w]
                    {0},                                            // [u_pad_w]
                    {2},                                           // [stride_w]
                    {1},                                         // [dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_1D_Dilation) {

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<132>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam =TmeIm2ColTestParam<1, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{3},                                                  // [s]
                    {0},                                            // [l_pad_w]
                    {0},                                            // [u_pad_w]
                    {1},                                           // [stride_w]
                    {2},                                         // [dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

}

/*

-------------------------------- 2D Cases --------------------------------

*/

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_2D) {

  {
    // M = 256 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<16>{}, Int<16>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<256>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<2, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1},                                            // [r, s]
                    {0, 0},                                // [l_pad_h, l_pad_w]
                    {0, 0},                                // [u_pad_h, u_pad_w]
                    {1, 1},                              // [stride_h, stride_w]
                    {1, 1},                          // [dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 640 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<32>{}, Int<20>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<256>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<2, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1},                                            // [r, s]
                    {0, 0},                                // [l_pad_h, l_pad_w]
                    {0, 0},                                // [u_pad_h, u_pad_w]
                    {1, 1},                              // [stride_h, stride_w]
                    {1, 1},                          // [dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 256 C = 64
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<16>{}, Int<16>{}, Int<64>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<256>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<2, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1},                                            // [r, s]
                    {0, 0},                                // [l_pad_h, l_pad_w]
                    {0, 0},                                // [u_pad_h, u_pad_w]
                    {1, 1},                              // [stride_h, stride_w]
                    {1, 1},                          // [dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }
}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_2D_Kernel) {

  {
    // M = 256 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<18>{}, Int<16>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<256>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<2, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{3, 1},                                            // [r, s]
                    {0, 0},                                // [l_pad_h, l_pad_w]
                    {0, 0},                                // [u_pad_h, u_pad_w]
                    {1, 1},                              // [stride_h, stride_w]
                    {1, 1},                          // [dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 256 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<18>{}, Int<18>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<256>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<2, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{3, 3},                                            // [r, s]
                    {0, 0},                                // [l_pad_h, l_pad_w]
                    {0, 0},                                // [u_pad_h, u_pad_w]
                    {1, 1},                              // [stride_h, stride_w]
                    {1, 1},                          // [dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }
}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_2D_Padding) {

  {
    // M = 256 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<15>{}, Int<16>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<256>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<2, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1},                                            // [r, s]
                    {1, 0},                                // [l_pad_h, l_pad_w]
                    {0, 0},                                // [u_pad_h, u_pad_w]
                    {1, 1},                              // [stride_h, stride_w]
                    {1, 1},                          // [dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 256 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<15>{}, Int<16>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<256>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<2, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1},                                            // [r, s]
                    {0, 0},                                // [l_pad_h, l_pad_w]
                    {1, 0},                                // [u_pad_h, u_pad_w]
                    {1, 1},                              // [stride_h, stride_w]
                    {1, 1},                          // [dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 256 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<14>{}, Int<16>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<256>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<2, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1},                                            // [r, s]
                    {1, 0},                                // [l_pad_h, l_pad_w]
                    {1, 0},                                // [u_pad_h, u_pad_w]
                    {1, 1},                              // [stride_h, stride_w]
                    {1, 1},                          // [dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_2D_Stride) {

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<16>{}, Int<16>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<2, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1},                                            // [r, s]
                    {0, 0},                                // [l_pad_h, l_pad_w]
                    {0, 0},                                // [u_pad_h, u_pad_w]
                    {2, 1},                              // [stride_h, stride_w]
                    {1, 1},                          // [dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }
}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_2D_Dilation) {

  {
    // M = 256 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<20>{}, Int<16>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<256>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<2, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{3, 1},                                            // [r, s]
                    {0, 0},                                // [l_pad_h, l_pad_w]
                    {0, 0},                                // [u_pad_h, u_pad_w]
                    {1, 1},                              // [stride_h, stride_w]
                    {2, 1},                          // [dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }
}

/*

-------------------------------- 3D Cases --------------------------------

*/

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_3D) {

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<8>{}, Int<4>{}, Int<4>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<3, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1, 1},                                      // [t, r, s]
                    {0, 0, 0},                    // [l_pad_d, l_pad_h, l_pad_w]
                    {0, 0, 0},                    // [u_pad_d, u_pad_h, u_pad_w]
                    {1, 1, 1},                 // [stride_d, stride_h, stride_w]
                    {1, 1, 1},           // [dilation_d, dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 256 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<8>{}, Int<8>{}, Int<4>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<3, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1, 1},                                      // [t, r, s]
                    {0, 0, 0},                    // [l_pad_d, l_pad_h, l_pad_w]
                    {0, 0, 0},                    // [u_pad_d, u_pad_h, u_pad_w]
                    {1, 1, 1},                 // [stride_d, stride_h, stride_w]
                    {1, 1, 1},           // [dilation_d, dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 128 C = 64
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<8>{}, Int<4>{}, Int<4>{}, Int<64>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<64>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<3, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1, 1},                                      // [t, r, s]
                    {0, 0, 0},                    // [l_pad_d, l_pad_h, l_pad_w]
                    {0, 0, 0},                    // [u_pad_d, u_pad_h, u_pad_w]
                    {1, 1, 1},                 // [stride_d, stride_h, stride_w]
                    {1, 1, 1},           // [dilation_d, dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }
}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_3D_Kernel) {

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<10>{}, Int<4>{}, Int<4>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<3, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{3, 1, 1},                                      // [t, r, s]
                    {0, 0, 0},                    // [l_pad_d, l_pad_h, l_pad_w]
                    {0, 0, 0},                    // [u_pad_d, u_pad_h, u_pad_w]
                    {1, 1, 1},                 // [stride_d, stride_h, stride_w]
                    {1, 1, 1},           // [dilation_d, dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_3D_Padding) {

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<7>{}, Int<4>{}, Int<4>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<3, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1, 1},                                      // [t, r, s]
                    {1, 0, 0},                    // [l_pad_d, l_pad_h, l_pad_w]
                    {0, 0, 0},                    // [u_pad_d, u_pad_h, u_pad_w]
                    {1, 1, 1},                 // [stride_d, stride_h, stride_w]
                    {1, 1, 1},           // [dilation_d, dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<7>{}, Int<4>{}, Int<4>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<3, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1, 1},                                      // [t, r, s]
                    {0, 0, 0},                    // [l_pad_d, l_pad_h, l_pad_w]
                    {1, 0, 0},                    // [u_pad_d, u_pad_h, u_pad_w]
                    {1, 1, 1},                 // [stride_d, stride_h, stride_w]
                    {1, 1, 1},           // [dilation_d, dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

  {
    // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<6>{}, Int<4>{}, Int<4>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<3, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1, 1},                                      // [t, r, s]
                    {1, 0, 0},                    // [l_pad_d, l_pad_h, l_pad_w]
                    {1, 0, 0},                    // [u_pad_d, u_pad_h, u_pad_w]
                    {1, 1, 1},                 // [stride_d, stride_h, stride_w]
                    {1, 1, 1},           // [dilation_d, dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }
}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_3D_Stride) {

  {
     // M = 128 C = 16
    auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<8>{}, Int<4>{}, Int<4>{}, Int<16>{}), LayoutRight{});
    auto smem_layout = make_layout(make_shape(Int<64>{}, Int<16>{}), LayoutRight{});
    using TestParam = TmeIm2ColTestParam<3, decltype(smem_layout), decltype(gmem_layout)>;

    TestParam param{{1, 1, 1},                                      // [t, r, s]
                    {0, 0, 0},                    // [l_pad_d, l_pad_h, l_pad_w]
                    {0, 0, 0},                    // [u_pad_d, u_pad_h, u_pad_w]
                    {2, 1, 1},                 // [stride_d, stride_h, stride_w]
                    {1, 1, 1},           // [dilation_d, dilation_h, dilation_w]
                    smem_layout,
                    gmem_layout};

    run_test_tme_im2col_load<int8_t>(param);
    run_test_tme_im2col_load<half_t>(param);
    run_test_tme_im2col_load< float>(param);
    run_test_tme_im2col_load<double>(param);
  }

}

TEST(MP31_MuTe_TME_LOAD_IM2COl, Tme_Load_Im2Col_3D_Dilation) {

  {
     // M = 128 C = 16
     auto gmem_layout = make_layout(make_shape(Int<1>{}, Int<12>{}, Int<4>{}, Int<4>{}, Int<16>{}), LayoutRight{});
     auto smem_layout = make_layout(make_shape(Int<128>{}, Int<16>{}), LayoutRight{});
     using TestParam = TmeIm2ColTestParam<3, decltype(smem_layout), decltype(gmem_layout)>;

     TestParam param{{3, 1, 1},                                     // [t, r, s]
                     {0, 0, 0},                   // [l_pad_d, l_pad_h, l_pad_w]
                     {0, 0, 0},                   // [u_pad_d, u_pad_h, u_pad_w]
                     {1, 1, 1},                // [stride_d, stride_h, stride_w]
                     {2, 1, 1},          // [dilation_d, dilation_h, dilation_w]
                     smem_layout,
                     gmem_layout};

     run_test_tme_im2col_load<int8_t>(param);
     run_test_tme_im2col_load<half_t>(param);
     run_test_tme_im2col_load< float>(param);
     run_test_tme_im2col_load<double>(param);
  }

}