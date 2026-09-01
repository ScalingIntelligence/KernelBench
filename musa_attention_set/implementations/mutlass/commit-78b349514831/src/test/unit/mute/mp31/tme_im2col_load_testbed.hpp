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

#include <vector>
#include <iostream>

#include "mutlass_unit_test.h"
#include "mutlass/arch/barrier.hpp"
#include "mutlass/util/device_memory.h"
#include "mutlass/util/reference/host/tensor_fill.h"

#include <mute/tensor.hpp>
#include <mute/atom/copy_atom.hpp>

#include "tme_im2col_host.hpp"

using namespace mute;

template <
  int   SpatialRank_,
  class SmemLayout_,
  class GmemLayout_
>
struct TmeIm2ColTestParam
{
  static constexpr int SpatialRank = SpatialRank_;
  static constexpr int TotalRank   = SpatialRank + 2;

  using TotalExtent   = mute::array<int, TotalRank>;
  using SpatialExtent = mute::array<int, SpatialRank>;
  using SmemLayout    = SmemLayout_;
  using GmemLayout    = GmemLayout_;

  TotalExtent shape_ndwhc;
  SpatialExtent trs;
  SpatialExtent lower_padding_dhw;
  SpatialExtent upper_padding_dhw;
  SpatialExtent stride_dhw;
  SpatialExtent dilation_dhw;

  SmemLayout smem_layout;
  GmemLayout gmem_layout;

  TmeIm2ColTestParam(
    SpatialExtent const& weight_pos,
    SpatialExtent const& lower_padding,
    SpatialExtent const& upper_padding,
    SpatialExtent const& stride,
    SpatialExtent const& dilation,
    SmemLayout    const& slayout,
    GmemLayout    const& glayout
  ): trs{weight_pos},
     lower_padding_dhw{lower_padding},
     upper_padding_dhw{upper_padding},
     stride_dhw{stride},
     dilation_dhw{dilation},
     smem_layout{slayout},
     gmem_layout{glayout} {}

  auto get_test_params() const
  {
    return make_tuple(shape(trs),
                      shape(lower_padding_dhw),
                      shape(upper_padding_dhw),
                      shape(stride_dhw),
                      shape(dilation_dhw),
                      smem_layout,
                      gmem_layout);
  }
};

template <class Element>
bool im2col_initialize_block(Element* block,
                             size_t capacity,
                             uint64_t seed=2025)
{
  Element scope_max, scope_min;
  int bits_input = mutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  mutlass::reference::host::BlockFillRandomUniform(block, capacity, seed, scope_max, scope_min, 0);

  return true;
}

template <
  class Element,
  class TiledCopy,
  class GmemLayout,
  class SmemLayout,
  class CTA_Tiler
>
__global__ __launch_bounds__(32) void
tme_im2col_test_device_mute(Element    const* g_in,
                            Element         * g_out,
                            TiledCopy  const  im2col,
                            GmemLayout const  gmem_layout,
                            SmemLayout const  smem_layout,
                            CTA_Tiler  const  cta_tiler)
{
  __shared__ __align__(256) Element smem[cosize_v<SmemLayout>];

  mutlass::arch::allocate_async_barriers(1);
  mutlass::arch::AsyncTransactionBarrier bar{};
  if(thread0()) {
    bar.init(/* arrive_count */ 1);
  }
  __syncthreads();

  Tensor sA = make_tensor(make_smem_ptr(smem), smem_layout);

  Tensor mA = im2col.get_tme_tensor(shape(_1{}));
  Tensor mB = make_tensor(make_gmem_ptr<Element>(g_out), gmem_layout);

  Tensor gA = local_tile(mA, cta_tiler, make_coord(_, _));                                       // (BLK_M, BLK_N, m, n)
  Tensor gB = local_tile(mB, cta_tiler, make_coord(_, _));                                       // (BLK_M, BLK_N, m, n)

  auto cta_im2col = im2col.get_slice(Int<0>{});
  Tensor tAgA = cta_im2col.partition_S(gA);                                                 // (TME, TME_M, TME_N, m, n)
  Tensor tAsA = cta_im2col.partition_D(sA);                                                       // (TME, TME_M, TME_N)

  auto tiled_copy_b = make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                                      make_layout(make_shape(Int<2>{}, Int<16>{}), LayoutRight{}),
                                      Layout<Shape<_1, _1>>{});
  auto thr_tiled_copy_b = tiled_copy_b.get_slice(threadIdx.x);
  Tensor tBsA = thr_tiled_copy_b.partition_S(sA);                                                 // (CPY, CPY_M, CPY_N)
  Tensor tBgB = thr_tiled_copy_b.partition_D(gB);                                           // (CPY, CPY_M, CPY_N, M, N)

  for(int stage = 0; stage < size<2>(gA); ++stage) {
    auto tile_iter = mute::make_coord_iterator(shape<3>(gA));
    int tile_count = size<3>(gA);
    for(int tc = 0; tc < tile_count; ++tc) {
      if(threadIdx.x < 32) {
        int TransactionBytes = mutlass::bits_to_bytes(cosize(smem_layout) * sizeof_bits_v<Element>);
        bar.expect_transaction(TransactionBytes);
        copy(im2col.with(bar.get_barrier_id()), tAgA(_,_,_,stage,*tile_iter), tAsA(_,_,_));
      }

      ++tile_iter;

      auto phase = bar.arrive<true>();
      bar.wait(phase);

      copy(tBsA(_,_,_), tBgB(_,_,_, stage, tc));
    }
  }
}

template <
  class Element,
  class CopyOp,
  class GmemLayout,
  class SmemLayout,
  class CTA_Tiler,
  class WeightPosSRT,
  class LowerPadding,
  class UpperPadding,
  class Stride,
  class Dilation
>
auto
test_tme_im2col_load(CopyOp        const& copy_op,
                     GmemLayout    const& gmem_layout,
                     SmemLayout    const& smem_layout,
                     CTA_Tiler     const& cta_tiler,
                     WeightPosSRT  const& trs,
                     LowerPadding  const& lower_pad_dhw,
                     UpperPadding  const& upper_pad_dhw,
                     Stride        const& stride_dhw,
                     Dilation      const& dilation_dhw)
{
  constexpr int total_modes = rank(flatten(shape(GmemLayout{})));
  constexpr int num_spatial_modes = total_modes - 2;

  const auto shape_ndhwc = shape(gmem_layout);
  const auto shape_zpq = mute::transform(mute::make_seq<num_spatial_modes>{}, [&](auto i){
      return (get<i+1>(shape_ndhwc) + get<i>(lower_pad_dhw) + get<i>(upper_pad_dhw)
              - get<i>(dilation_dhw) * (get<i>(trs) - Int<1>{})
              - Int<1>{}) / get<i>(stride_dhw) + Int<1>{};
  });
  const int NDHWC = size(shape(gmem_layout));
  const int NZPQ = size(shape_zpq) * get<0>(shape(gmem_layout));
  const int CSRT = get<total_modes - 1>(shape_ndhwc) * size(trs);
  const int NZPQ_CSRT = NZPQ * CSRT;

  std::vector<Element> vec_in(NDHWC, Element{});
  std::vector<Element> vec_out(NZPQ * CSRT, Element{});

  mutlass::DeviceAllocation<Element> g_in;
  mutlass::DeviceAllocation<Element> g_out;
  g_in.reset(NDHWC);
  g_out.reset(NZPQ * CSRT);

  im2col_initialize_block(vec_in.data(), NDHWC);

  g_in.copy_from_host(vec_in.data());
  g_out.copy_from_host(vec_out.data());

  auto tensor_d_shape = make_shape(reverse(take<0, total_modes - 1>(shape_ndhwc)),
                                   get<total_modes - 1>(shape_ndhwc));
  auto tensor_d_stride = make_stride(reverse(take<0, total_modes - 1>(stride(gmem_layout))),
                                     stride<total_modes - 1>(gmem_layout));
  auto tensor_d_layout = make_layout(tensor_d_shape, tensor_d_stride);
  Tensor tensor_d_in = make_tensor(g_in.get(), tensor_d_layout);

  auto tme_copy = make_im2col_tme_copy(copy_op,
                                       tensor_d_in,
                                       SmemLayout{},
                                       product_each(shape(SmemLayout{})),
                                       mute::reverse(shape(lower_pad_dhw)),
                                       mute::reverse(shape(upper_pad_dhw)),
                                       mute::reverse(shape(stride_dhw)),
                                       mute::reverse(shape(dilation_dhw)),
                                       mute::reverse(shape(trs)));

  auto layout_out = make_layout(make_shape(NZPQ, CSRT), LayoutRight{});

  tme_im2col_test_device_mute<<<1, 32>>>(g_in.get(), g_out.get(), tme_copy, layout_out, smem_layout, cta_tiler);

  g_out.copy_to_host(vec_out.data());

  Tensor tensor_h_in = make_tensor(vec_in.data(), gmem_layout);
  auto vec_out_ref = run_im2col_cpu(tensor_h_in, trs, lower_pad_dhw, upper_pad_dhw, stride_dhw, dilation_dhw);

  int count_fail = 3;
  for(int i = 0; i < vec_out_ref.size() && count_fail; ++i) {
    EXPECT_EQ(vec_out_ref.at(i), vec_out.at(i));
    if(vec_out_ref.at(i) != vec_out.at(i)) {
      --count_fail;
    }
  }
}


template <class TestDType, class TestParam>
void
run_test_tme_im2col_load(TestParam const& param)
{

  auto [trs,
        lower_padding_dhw,
        upper_padding_dhw,
        stride_dhw,
        dilation_dhw,
        smem_layout,
        gmem_layout] = param.get_test_params();

  auto tiler = flatten(shape(smem_layout));
  auto cta_tiler = make_shape(get<0>(tiler), make_shape(get<1>(tiler)));

  test_tme_im2col_load<TestDType>(MP31_TME_LOAD_IM2COL{},
                                  gmem_layout,
                                  smem_layout,
                                  cta_tiler,
                                  trs,
                                  lower_padding_dhw,
                                  upper_padding_dhw,
                                  stride_dhw,
                                  dilation_dhw);
}