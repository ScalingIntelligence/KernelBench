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

#include <mutlass/mutlass.h>
#include <mutlass/fast_math.h>

#include <mute/tensor.hpp>

using namespace mute;

template <class KernelTraits>
struct CollectiveFwdParams {
  using Element = typename KernelTraits::Element;
  using TileShapeQK = typename KernelTraits::TileShapeQK;
  using TileShapePV = typename KernelTraits::TileShapePV;

  using GmemTiledCopyQ = typename KernelTraits::GmemTiledCopyQ;
  using GmemTiledCopyV = typename KernelTraits::GmemTiledCopyV;

  using SmemLayoutQ = typename KernelTraits::SmemLayoutQ;
  using SmemLayoutV = typename KernelTraits::SmemLayoutV;

  using IndexT = int64_t;
  using ShapeT  = Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideT = Stride<IndexT, _1, IndexT, IndexT>;
  using LayoutT = Layout<ShapeT, StrideT>;

  using TME_Q = decltype(make_tme_copy(
      GmemTiledCopyQ{},
      make_tensor(
        make_gmem_ptr(static_cast<Element const*>(nullptr)),
        repeat_like(StrideT{}, int32_t(0)),
        StrideT{}
      ),
      SmemLayoutQ{},
      select<0, 2>(TileShapeQK{})));

  using TME_V = decltype(make_tme_copy(
      GmemTiledCopyV{},
      make_tensor(
        make_gmem_ptr(static_cast<Element const*>(nullptr)),
        repeat_like(StrideT{}, int32_t(0)),
        StrideT{}
      ),
      take<0, 2>(SmemLayoutV{}),
      select<2, 1>(TileShapePV{})));

  // Host side kernel arguments
  struct Arguments {
    // Mainloop
    Element const* ptr_Q;
    LayoutT layout_Q;
    Element const* ptr_K;
    LayoutT layout_K;
    Element const* ptr_V;
    LayoutT layout_V;
    float const rln2_scale;
    int n_tiles;

    // Epilogue
    Element* ptr_O;
    LayoutT layout_O;

    // Debug
    Element* ptr_S;
    LayoutT layout_S;
  };

  // Device side kernel arguments
  struct Params {
    // mainloop
    TME_Q tme_load_Q;
    LayoutT layout_Q;
    Element const* ptr_K;
    LayoutT layout_K;
    RobustDescriptor key_desc;
    TME_V tme_load_V;
    LayoutT layout_V;
    float const rln2_scale;
    int const n_tiles;

    // Epilogue
    Element* ptr_O;
    LayoutT layout_O;

    // Debug
    Element* ptr_S;
    LayoutT layout_S;
  };

  MUTLASS_HOST_DEVICE
  static auto
  get_gmem_layout(
      int m, int k, int h, int b,
      IndexT m_stride, IndexT h_stride, IndexT b_stride) {
    return make_layout(make_shape(m, k, h, b),
                       make_stride(m_stride, _1{}, h_stride, b_stride));
  }

  static Params
  to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.layout_Q);
    TME_Q tme_load_q = make_tme_copy(
        GmemTiledCopyQ{},
        mQ,
        SmemLayoutQ{},
        select<0, 2>(TileShapeQK{}));

    auto element_key = cosize(args.layout_K);

    auto key_desc = make_robust_desc(args.ptr_K, element_key);

    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.layout_V);
    TME_V tme_load_v = make_tme_copy(
        GmemTiledCopyV{},
        mV,
        take<0,2>(SmemLayoutV{}),
        select<2, 1>(TileShapePV{}));

    return {tme_load_q, args.layout_Q, args.ptr_K, args.layout_K, key_desc, tme_load_v, args.layout_V,
            args.rln2_scale, args.n_tiles,
            args.ptr_O, args.layout_O, args.ptr_S, args.layout_S};
  }
};
