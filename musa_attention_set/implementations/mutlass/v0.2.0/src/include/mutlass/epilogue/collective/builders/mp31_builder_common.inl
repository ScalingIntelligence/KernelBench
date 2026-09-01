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
#include "mute/atom/copy_traits_mp31_tme_swizzle.hpp"
#include "mute/atom/mma_atom.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace mute;

template <
  int MN,
  int K,
  class Type,
  TCE::Major tnsp,
  class SmemAtomLayout, 
  TME::SmemSwizzleGranularity sg = TME::SmemSwizzleGranularity::NONE,
  TME::SmemSwizzleStride      ss = TME::SmemSwizzleStride::B128,
  TME::SmemSwizzleLine        sl = TME::SmemSwizzleLine::B128
>
MUTE_HOST_DEVICE constexpr
auto
get_layout_swizzle_Atom() {
  constexpr int BITS = sizeof_bits_v<Type>;
  using swizzle = decltype(mute::detail::get_tme_swizzle<sl, ss, sg>());
  using layout_atom_bits = mute::conditional_t<tnsp == TCE::Major::MN,
                            decltype(make_layout(Shape<Int<MN * sizeof_bits_v<Type>>, Int<K>>{}, LayoutLeft{})),
                            decltype(make_layout(Shape<Int<MN>, Int<K * sizeof_bits_v<Type>>>{}, LayoutRight{}))>;
  return upcast<BITS>(ComposedLayout<
              swizzle,
              smem_ptr_flag,
              decltype(blocked_product(
              layout_atom_bits{},
              SmemAtomLayout{}))>{});
};

template <class Type, TCE::Major tnsp, class AtomM, class AtomN, class SmemAtomLayout>
MUTE_HOST_DEVICE constexpr
auto
make_tme_smem_atom_layout_C() {
  constexpr int BITS = sizeof_bits_v<Type>;
  constexpr int ATOM_M = size(AtomM{});
  constexpr int ATOM_N = size(AtomN{});
  constexpr int BYTE = sizeof(Type);
  static_assert(BITS <= 32, "Unsupported Tme Type");
  if constexpr (tnsp == TCE::Major::MN) {
    constexpr auto SL = ATOM_M * BYTE <= 128 ? TME::SmemSwizzleLine::B128 : TME::SmemSwizzleLine::B256;
    if constexpr (shape<0>(SmemAtomLayout{}) > 1 && shape<1>(SmemAtomLayout{}) == 1) {
      return get_layout_swizzle_Atom<ATOM_M, ATOM_N, Type, tnsp, SmemAtomLayout, 
        TME::SmemSwizzleGranularity::B16, TME::SmemSwizzleStride::B128, SL>();
    } else {
      // S4/S8/U8/FP8
      if constexpr (BITS <= 8) {
        return get_layout_swizzle_Atom<ATOM_M, ATOM_N, Type, tnsp, SmemAtomLayout, 
          TME::SmemSwizzleGranularity::NONE, TME::SmemSwizzleStride::B128, TME::SmemSwizzleLine::B256>();
      }
      // FP16/BF16
      else if constexpr (BITS == 16) {
        return get_layout_swizzle_Atom<ATOM_M, ATOM_N, Type, tnsp, SmemAtomLayout, 
            TME::SmemSwizzleGranularity::B16, TME::SmemSwizzleStride::B32, TME::SmemSwizzleLine::B128>();
      }
      // TF32
      else {
        return get_layout_swizzle_Atom<ATOM_M, ATOM_N, Type, tnsp, SmemAtomLayout, 
            TME::SmemSwizzleGranularity::B16, TME::SmemSwizzleStride::B64, TME::SmemSwizzleLine::B128>();
      }
    }
  }
  else if constexpr (tnsp == TCE::Major::K) {
    constexpr auto SL = ATOM_N * BYTE <= 128 ? TME::SmemSwizzleLine::B128 : TME::SmemSwizzleLine::B256;
    if constexpr (shape<0>(SmemAtomLayout{}) > 1 && shape<1>(SmemAtomLayout{}) == 1) {
      return get_layout_swizzle_Atom<ATOM_M, ATOM_N, Type, tnsp, SmemAtomLayout, 
        TME::SmemSwizzleGranularity::NONE, TME::SmemSwizzleStride::B128, TME::SmemSwizzleLine::B256>();
    } else {
      return get_layout_swizzle_Atom<ATOM_M, ATOM_N, Type, tnsp, SmemAtomLayout, 
        TME::SmemSwizzleGranularity::B32, TME::SmemSwizzleStride::B128, SL>();
    }
  }
  else {
    static_assert(tnsp != TCE::Major::MN && tnsp != TCE::Major::K, "Unrecognized MajorMode!");
  }
}

namespace mutlass::epilogue::collective::detail {
template <TCE::Major major, class ElementType, class EpilogueTileMN>
MUTE_HOST_DEVICE constexpr
auto
ss_smem_selector_C()
{
  using AtomN = decltype(get<0>(shape<1>(EpilogueTileMN{})));
  using AtomM = decltype(get<0>(shape<0>(EpilogueTileMN{})));

  constexpr int smem_atom_m = size<1>(shape<0>(EpilogueTileMN{}));
  constexpr int smem_atom_n = size<1>(shape<1>(EpilogueTileMN{}));

  using SmemAtomLayout = Layout<Shape <Int<smem_atom_m>, Int<smem_atom_n>>,
                                Stride<Int<smem_atom_n>,               _1>>;

  return make_tme_smem_atom_layout_C<ElementType, major, AtomM, AtomN, SmemAtomLayout>();
}

template<class CopyOp, class CopyV, class EpilogueTile, class... MArgs>
MUTLASS_HOST_DEVICE constexpr
auto make_tiled_copy_C_sqmma(gemm::TiledMMA<MArgs...> const& mma){
  auto layoutC_TV = mma.get_layoutC_TV();
  MUTE_STATIC_ASSERT_V(CopyV{} <= size<1>(layoutC_TV));
  constexpr int AtomM = get<1>(shape<0>(EpilogueTile{}));
  constexpr int AtomN = get<1>(shape<1>(EpilogueTile{}));
  constexpr int stride_v = (AtomM > 1 && AtomN == 1) ? get<2>(get<0>(stride<1>(layoutC_TV))) : get<0>(get<0>(stride<1>(layoutC_TV)));
  auto layout_TV = make_layout(make_shape(shape<0>(layoutC_TV), CopyV{}),  make_stride(stride<0>(layoutC_TV), Int<stride_v>{}));

  auto mma_tiler = make_shape(tile_size<0>(mma), tile_size<1>(mma));
  auto mma_zeros = repeat_like(mma_tiler, Int<0>{});

  auto orig_tiler = mute::transform(make_seq<rank(mma_tiler)>{}, [&](auto i) {
    return filter(composition(make_layout(mma_tiler, replace<i>(mma_zeros, Int<1>{})), layout_TV));
  });

  auto orig_tiler_m = get<0>(EpilogueTile{});
  auto orig_tiler_n = get<1>(EpilogueTile{});

  auto tiler_n = append<3>(orig_tiler_n, Layout<_1, _0>{});
  auto permute_tiler_n = make_layout(get<0>(tiler_n), get<2>(tiler_n), get<1>(tiler_n));

  auto tiler = make_tile(orig_tiler_m, permute_tiler_n);
  auto tile2mma = composition(make_layout(mma_tiler), tiler);

  auto tiler_mn = make_tile(make_layout(size<0>(tiler)),
                            make_layout(size<1>(tiler)));

  auto layout_tv = composition(left_inverse(tile2mma), layout_TV);
  return make_tiled_copy_impl(CopyOp{}, layout_tv, tiler_mn);
}

///////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::epilogue::collective::detail