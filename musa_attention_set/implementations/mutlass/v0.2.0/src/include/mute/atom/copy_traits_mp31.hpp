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

#include <mute/arch/copy_mp31.hpp>
#include <mute/atom/copy_traits.hpp>


namespace mute {

//
// ROBUST LOAD
//

template <class S, class D>
struct MP31_ROBUST_LOAD_OP : MP31_ROBUST_LOAD<S, D> {};

template <class S, class D>
struct MP31_ROBUST_PRED_LOAD_OP : MP31_ROBUST_LOAD<S, D> {};

template <class S, class D>
struct Copy_Traits<MP31_ROBUST_PRED_LOAD_OP<S, D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID     = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Robust Buffer Access Descriptor
  const RobustDescriptor src_desc;

  // Predicate value that determines whether to load or return zero
  bool pred;

  // Overload copy_unpack to pass robust buffer access descriptor
  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits         const& traits,
              Tensor<TS, SLayout> const& src,
              Tensor<TD, DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem src for MP31_ROBUST_LOAD");
    static_assert(is_rmem<TD>::value, "Expected rmem dst for MP31_ROBUST_LOAD");

    Tensor rS = recast<S>(src);
    Tensor rD = recast<D>(dst);

    MUTE_STATIC_ASSERT_V(size(rS) == Int<1>{},
        "This src layout is incompatible with this tiled copy.");
    MUTE_STATIC_ASSERT_V(size(rD) == Int<1>{},
        "This dst layout is incompatible with this tiled copy.");

    MP31_ROBUST_LOAD<S, D>::copy(rS[0], rD[0], traits.pred, traits.src_desc);
  }
};

template <class S, class D>
struct Copy_Traits<MP31_ROBUST_LOAD_OP<S, D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID     = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Robust Buffer Access Descriptor
  const RobustDescriptor src_desc;

  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP31_ROBUST_PRED_LOAD_OP<S, D>>
  with(bool pred) const {
    return {src_desc, pred};
  }
};


template <class S, class D>
struct Copy_Traits<MP31_ROBUST_LOAD<S, D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID     = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Construct an executable MP31_ROBUST_LOAD with robust buffer access descriptor
  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP31_ROBUST_LOAD_OP<S, D>>
  with(RobustDescriptor desc) const {
    return {desc};
  }

  // Don't try to execute a copy with MP31_ROBUST_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits         const& traits,
              Tensor<TS, SLayout> const& src,
              Tensor<TD, DLayout>      & dst) = delete;

};

//
// ROBUST STORE
//

template <class S, class D>
struct MP31_ROBUST_STORE_OP : MP31_ROBUST_STORE<S, D> {};

template <class S, class D>
struct MP31_ROBUST_PRED_STORE_OP : MP31_ROBUST_STORE<S, D> {};

template <class S, class D>
struct Copy_Traits<MP31_ROBUST_PRED_STORE_OP<S, D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID     = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Robust Buffer Access Descriptor
  const RobustDescriptor dst_desc;

  // Predicate value that determines whether to store or discard
  bool pred;

  // Overload copy_unpack to pass robust buffer access descriptor
  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits         const& traits,
              Tensor<TS, SLayout> const& src,
              Tensor<TD, DLayout>      & dst)
  {
    static_assert(is_rmem<TS>::value, "Expected rmem src for MP31_ROBUST_STORE");
    static_assert(is_gmem<TD>::value, "Expected gmem dst for MP31_ROBUST_STORE");

    Tensor rS = recast<S>(src);
    Tensor rD = recast<D>(dst);

    MUTE_STATIC_ASSERT_V(size(rS) == Int<1>{},
        "This src layout is incompatible with this tiled copy.");
    MUTE_STATIC_ASSERT_V(size(rD) == Int<1>{},
        "This dst layout is incompatible with this tiled copy.");

    MP31_ROBUST_STORE<S, D>::copy(rS[0], rD[0], traits.pred, traits.dst_desc);
  }
};

template <class S, class D>
struct Copy_Traits<MP31_ROBUST_STORE_OP<S, D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID     = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Robust Buffer Access Descriptor
  const RobustDescriptor dst_desc;

  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP31_ROBUST_PRED_STORE_OP<S, D>>
  with(bool pred) const {
    return {dst_desc, pred};
  }
};

template <class S, class D>
struct Copy_Traits<MP31_ROBUST_STORE<S, D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID     = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Construct an executable MP31_ROBUST_STORE with robust buffer access descriptor
  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP31_ROBUST_STORE_OP<S, D>>
  with(RobustDescriptor desc) const {
    return {desc};
  }

  // Don't try to execute a copy with MP31_ROBUST_STORE before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits         const& traits,
              Tensor<TS, SLayout> const& src,
              Tensor<TD, DLayout>      & dst) = delete;

};

//
// ROBUST LDGSTS
//

template <class S, class D>
struct MP31_ROBUST_LDGSTS_OP : MP31_ROBUST_LDGSTS<S, D> {};

template <class S, class D>
struct MP31_ROBUST_PRED_LDGSTS_OP : MP31_ROBUST_LDGSTS<S, D> {};


template <class S, class D>
struct Copy_Traits<MP31_ROBUST_PRED_LDGSTS_OP<S, D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID     = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Robust Buffer Access Descriptor
  const RobustDescriptor src_desc;

  // Predicate value that determines whether to load or return zero
  bool pred;

  // Overload copy_unpack to pass robust buffer access descriptor
  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits         const& traits,
              Tensor<TS, SLayout> const& src,
              Tensor<TD, DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem src for MP31_ROBUST_LDGSTS");
    static_assert(is_smem<TD>::value, "Expected smem dst for MP31_ROBUST_LDGSTS");

    Tensor rS = recast<S>(src);
    Tensor rD = recast<D>(dst);

    MUTE_STATIC_ASSERT_V(size(rS) == Int<1>{},
        "This src layout is incompatible with this tiled copy.");
    MUTE_STATIC_ASSERT_V(size(rD) == Int<1>{},
        "This dst layout is incompatible with this tiled copy.");

    MP31_ROBUST_LDGSTS<S, D>::copy(rS[0], rD[0], traits.pred, traits.src_desc);
  }
};

template <class S, class D>
struct Copy_Traits<MP31_ROBUST_LDGSTS_OP<S, D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID     = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Robust Buffer Access Descriptor
  const RobustDescriptor src_desc;

  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP31_ROBUST_PRED_LDGSTS_OP<S, D>>
  with(bool pred) const {
    return {src_desc, pred};
  }
};

template <class S, class D>
struct Copy_Traits<MP31_ROBUST_LDGSTS<S, D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID     = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Construct an executable MP31_ROBUST_LDGSTS with robust buffer access descriptor
  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP31_ROBUST_LDGSTS_OP<S, D>>
  with(RobustDescriptor desc) const {
    return {desc};
  }

  // Don't try to execute a copy with MP31_ROBUST_LDGSTS before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits         const& traits,
              Tensor<TS, SLayout> const& src,
              Tensor<TD, DLayout>      & dst) = delete;

};

template <class S, class D>
struct Copy_Traits<MP31_LDGSTS<S, D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID     = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

} // namespace mute
