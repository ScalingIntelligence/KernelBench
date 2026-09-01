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

#include "mute/tensor.hpp"

#include <vector>

using namespace mute;

template <
  class InputEngine, class InputLayout,
  class KernelPosTRS,
  class LowerPadding,
  class UpperPadding,
  class KernelStride,
  class Dilation
>
auto im2col_host(Tensor<InputEngine, InputLayout> const& input_tensor,
                 KernelPosTRS                     const& trs,
                 LowerPadding                     const& lower_pad_dhw,
                 UpperPadding                     const& upper_pad_dhw,
                 KernelStride                     const& stride_dhw,
                 Dilation                         const& dilation_dhw) {
  constexpr int num_total_modes = 5;
  constexpr int num_spatial_modes = num_total_modes - 2;
  using Element = typename InputEngine::value_type;

  const auto shape_ndhwc = shape(input_tensor);
  const int N = get<0>(shape_ndhwc);
  const int D = get<1>(shape_ndhwc);
  const int H = get<2>(shape_ndhwc);
  const int W = get<3>(shape_ndhwc);
  const int C = get<4>(shape_ndhwc);

  const int lower_pad_d = get<0>(lower_pad_dhw);
  const int lower_pad_h = get<1>(lower_pad_dhw);
  const int lower_pad_w = get<2>(lower_pad_dhw);

  const int upper_pad_d = get<0>(upper_pad_dhw);
  const int upper_pad_h = get<1>(upper_pad_dhw);
  const int upper_pad_w = get<2>(upper_pad_dhw);

  const int stride_d = get<0>(stride_dhw);
  const int stride_h = get<1>(stride_dhw);
  const int stride_w = get<2>(stride_dhw);

  const int dilation_d = get<0>(dilation_dhw);
  const int dilation_h = get<1>(dilation_dhw);
  const int dilation_w = get<2>(dilation_dhw);

  const auto shape_zpq = mute::transform(mute::make_seq<num_spatial_modes>{}, [&](auto i){
      return (get<i+1>(shape_ndhwc) + get<i>(lower_pad_dhw) + get<i>(upper_pad_dhw) - get<i>(dilation_dhw) * (get<i>(trs) - Int<1>{}) - Int<1>{}) / get<i>(stride_dhw) + Int<1>{};
  });
  const int Z = get<0>(shape_zpq);
  const int P = get<1>(shape_zpq);
  const int Q = get<2>(shape_zpq);

  const int T = get<0>(trs);
  const int R = get<1>(trs);
  const int S = get<2>(trs);

  const int kernel_size = C * S * R * T;
  const int rows = N * Z * P * Q;

  static auto get_val = [&](int n, int d, int h, int w, int c) {
    if (d < 0 || d >= D || h < 0 || h >= H || w < 0 || w >= W) {
      return Element{};
    }
    return input_tensor(n, d, h, w, c);
  };

  std::vector<Element> output(kernel_size * rows, Element{});

  for (int n = 0; n < N; ++n) {
    for (int z = 0; z < Z; ++z) {
      for (int p = 0; p < P; ++p) {
        for (int q = 0; q < Q; ++q) {
          int col = ((n * Z + z) * P + p) * Q + q;
          int patch_idx = 0;
          for (int t = 0; t < T; ++t) {
            int d = z * stride_d + t * dilation_d - lower_pad_d;
            for (int r = 0; r < R; ++r) {
              int h = p * stride_h + r * dilation_h - lower_pad_h;
              for (int s = 0; s < S; ++s) {
                int w = q * stride_w + s * dilation_w - lower_pad_w;
                for (int c = 0; c < C; ++c) {
                  auto val = get_val(n, d, h, w, c);
                  output[col * kernel_size + patch_idx] = val;
                  ++patch_idx;
                }
              }
            }
          }
        }
      }
    }
  }
  return output;
}



template <
  class InputEngine, class InputLayout,
  class KernelPosTRS,
  class LowerPadding,
  class UpperPadding,
  class KernelStride,
  class Dilation
>
auto run_im2col_cpu(Tensor<InputEngine, InputLayout> const& input_tensor,
                    KernelPosTRS                     const& trs,
                    LowerPadding                     const& lower_pad_dhw,
                    UpperPadding                     const& upper_pad_dhw,
                    KernelStride                     const& stride_dhw,
                    Dilation                         const& dilation_dhw)
{
  constexpr int total_modes = rank(InputLayout{});
  constexpr int spatial_modes = total_modes - 2;

  if constexpr (spatial_modes == 1) {
    auto input_tensor_shape_full = insert<1>(insert<1>(shape(input_tensor), _1{}), _1{});
    auto input_tensor_full = make_tensor(input_tensor.data(),
                                         make_layout(input_tensor_shape_full, LayoutRight{}));
    auto trs_full = insert<0>(insert<0>(trs, _1{}), _1{});
    auto lower_pad_dhw_full = insert<0>(insert<0>(lower_pad_dhw, _0{}), _0{});
    auto upper_pad_dhw_full = insert<0>(insert<0>(upper_pad_dhw, _0{}), _0{});
    auto stride_dhw_full = insert<0>(insert<0>(stride_dhw, _1{}), _1{});
    auto dilation_dhw_full = insert<0>(insert<0>(dilation_dhw, _1{}), _1{});

    return im2col_host(input_tensor_full,
                       trs_full,
                       lower_pad_dhw_full,
                       upper_pad_dhw_full,
                       stride_dhw_full,
                       dilation_dhw_full);

  } else if constexpr (spatial_modes == 2) {
    auto input_tensor_shape_full = insert<1>(shape(input_tensor), _1{});
    auto input_tensor_full = make_tensor(input_tensor.data(),
                                         make_layout(input_tensor_shape_full, LayoutRight{}));
    auto trs_full = insert<0>(trs, _1{});
    auto lower_pad_dhw_full = insert<0>(lower_pad_dhw, _0{});
    auto upper_pad_dhw_full = insert<0>(upper_pad_dhw, _0{});
    auto stride_dhw_full = insert<0>(stride_dhw, _1{});
    auto dilation_dhw_full = insert<0>(dilation_dhw, _1{});

    return im2col_host(input_tensor_full,
                       trs_full,
                       lower_pad_dhw_full,
                       upper_pad_dhw_full,
                       stride_dhw_full,
                       dilation_dhw_full);
  } else {
    return im2col_host(input_tensor,
                       trs,
                       lower_pad_dhw,
                       upper_pad_dhw,
                       stride_dhw,
                       dilation_dhw);
  }
}
