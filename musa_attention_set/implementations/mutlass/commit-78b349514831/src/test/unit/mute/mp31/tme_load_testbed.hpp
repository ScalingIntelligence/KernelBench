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

#pragma once

#include "mutlass_unit_test.h"
#include "mutlass/arch/barrier.hpp"
#include "mutlass/util/device_memory.h"
#include "mutlass/util/reference/host/tensor_fill.h"

#include <mute/tensor.hpp>
#include <mute/atom/copy_atom.hpp>

using namespace mute;

template <class Element>
bool tme_load_initialize_block(Element* block,
                          size_t capacity,
                          uint64_t seed=2025) {

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

  // mutlass::reference::host::BlockFillRandomUniform(
  //   block, capacity, seed, scope_max, scope_min, 0);

  for(int i = 0; i < capacity; ++i) {
    block[i] = i % 256;
  }

  return true;
}

template<
  class Element,
  class TiledCopy,
  class GmemLayout,
  class SmemLayout,
  class CTA_Tiler
>
__global__ __launch_bounds__(32) void
tme_load_test_device_mute(Element    const* g_in,
                          Element         * g_out,
                          TiledCopy  const  tme,
                          GmemLayout const  gmem_layout,
                          SmemLayout const  smem_layout,
                          CTA_Tiler  const  cta_tiler)
{
  __shared__ __align__(256) Element smem[cosize(SmemLayout{})];

  mutlass::arch::allocate_async_barriers(1);
  mutlass::arch::AsyncTransactionBarrier bar{};
  if(thread0()) {
    bar.init(/* arrive_count */ 1);
  }
  __syncthreads();

  Tensor sA = make_tensor(make_smem_ptr(smem), smem_layout);

  Tensor mA = tme.get_tme_tensor(shape(gmem_layout));
  Tensor mB = make_tensor(make_gmem_ptr<Element>(g_out), gmem_layout);

  Tensor gA = flat_divide(mA, cta_tiler);                                                        // (BLK_M, BLK_N, m, n)
  Tensor gB = flat_divide(mB, cta_tiler);                                                        // (BLK_M, BLK_N, m, n)

  auto cta_tme = tme.get_slice(Int<0>{});
  auto tAgA_ = cta_tme.partition_S(gA);                                                     // (TME, TME_M, TME_N, m, n)
  auto tAsA_ = cta_tme.partition_D(sA);                                                           // (TME, TME_M, TME_N)

#if 0
  if (thread0()) {
    // print(tme);
    // print("TILE  :  "); print(cta_tiler); print("\n");
    // print("  mA  :  "); print(  mA);   print("\n");
    // print("  mB  :  "); print(  mB);   print("\n");
    // print("  gA  :  "); print(  gA);   print("\n");
    // print("  gB  :  "); print(  gB);   print("\n");
    // print("  sA  :  "); print(  sA);   print("\n");
    // print("tAgA_:  "); print(tAgA_); print("\n");
    // print("tAsA_:  "); print(tAsA_); print("\n");
  }
#endif

  auto tAgA = group_modes<1, rank(tAgA_)>(tAgA_);
  auto tAsA = group_modes<1, rank(tAsA_)>(tAsA_);

  constexpr int R = rank_v<CTA_Tiler>;
  Tensor tBgB = group_modes<0,R>(group_modes<R,rank(gB)>(gB));

#if 0
  if (thread0()) {
    print("tAgA  :  "); print(tAgA); print("\n");
    print("tAsA  :  "); print(tAsA); print("\n");
    print("tBgB  :  "); print(tBgB); print("\n");
  }
#endif

  for(int stage = 0; stage < size<1>(tAgA); ++stage) {

    if(threadIdx.x < 32) {
      int TransactionBytes = mutlass::bits_to_bytes(cosize(smem_layout) * sizeof_bits_v<Element>);
      bar.expect_transaction(TransactionBytes);
      copy(tme.with(bar.get_barrier_id()), tAgA(_,stage), tAsA(_,Int<0>{}));
    }

    auto phase = bar.arrive<true>();
    bar.wait(phase);

    // if(thread0()) {
    //   copy(sA, tBgB(_,stage));
    // }
    Tensor stage_tBgB = tBgB(_,stage);
    for(int i = threadIdx.x; i < size(sA); i += blockDim.x) {
      stage_tBgB(i) = sA(i);
    }
  }
}

template<
  class Element,
  class CopyOp,
  class GmemLayout,
  class SmemLayout,
  class CTA_Tiler
>
auto
test_tme_load(CopyOp        const& copy_op,
              GmemLayout    const& gmem_layout,
              SmemLayout    const& smem_layout,
              CTA_Tiler     const& cta_tiler)
{
  const int NUM_ELEM = cosize(gmem_layout);

  std::vector<Element> vec_in(NUM_ELEM, Element{});
  std::vector<Element> vec_out(NUM_ELEM, Element{});

  mutlass::DeviceAllocation<Element> g_in;
  mutlass::DeviceAllocation<Element> g_out;
  g_in.reset(NUM_ELEM);
  g_out.reset(NUM_ELEM);

  tme_load_initialize_block(vec_in.data(), NUM_ELEM);

  g_in.copy_from_host(vec_in.data());
  g_out.copy_from_host(vec_out.data());

  Tensor tensor_d_in = make_tensor(g_in.get(), gmem_layout);

  auto tme_copy = make_tme_copy(copy_op,
                                tensor_d_in,
                                smem_layout,
                                cta_tiler);


  tme_load_test_device_mute<<<1, 32>>>(g_in.get(), g_out.get(), tme_copy, gmem_layout, smem_layout, cta_tiler);

  g_out.copy_to_host(vec_out.data());

  Tensor h_in = make_tensor(recast_ptr<Element>(vec_in.data()), gmem_layout);
  Tensor h_out = make_tensor(recast_ptr<Element>(vec_out.data()), gmem_layout);

  int count_fail = 3;
  for(int i = 0; i < size(h_out) && count_fail; ++i) {
    EXPECT_EQ(h_in(i), h_out(i));
    if(h_in(i) != h_out(i)) {
      --count_fail;
    }
  }

  return tme_copy;
}

template<
  class TestDType,
  class GmemLayout,
  class SmemLayout,
  class CTA_Tiler
>
auto
run_test_tme_load(GmemLayout  const& gmem_layout,
                  SmemLayout  const& smem_layout,
                  CTA_Tiler   const& cta_tiler)
{
  return test_tme_load<TestDType>(mute::MP31_TME_LOAD{}, gmem_layout, smem_layout, cta_tiler);
}

template<
  class TestDType,
  class GmemLayout,
  class SmemLayout
>
auto
run_test_tme_load(GmemLayout  const& gmem_layout,
                  SmemLayout  const& smem_layout)
{
  return run_test_tme_load<TestDType>(gmem_layout, smem_layout, product_each(shape(smem_layout)));
}