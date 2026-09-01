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

#if !defined(__MUSACC_RTC__)
#include <musa.h>
#endif

#include <mute/atom/copy_traits_mp31_tme_swizzle.hpp>
#include <mute/atom/copy_traits.hpp>
#include <mute/atom/copy_atom.hpp>

#include <mute/numeric/integral_ratio.hpp>

namespace mute
{

template <class GmemTmeBasisStrides_, class TmeGmemBasis_, class TmeSwizzle_, class TmeBlockDim_>
struct AuxTmeParams {
  using GmemStrides  = GmemTmeBasisStrides_;    // Strides for Gmem mode -> Tme coord mode, may be dynamic
  GmemStrides g_stride_;
  using TmeGmemBasis = TmeGmemBasis_;
  static_assert(is_static<TmeGmemBasis>::value);
  using TmeSwizzle   = TmeSwizzle_;
  static_assert(is_static<TmeSwizzle>::value);
  using TmeBlockDim  = TmeBlockDim_;
  static_assert(is_static<TmeBlockDim>::value);
  TmeBlockDim block_dim_;
};

// Utility for unpacking TME_LOAD arguments into a CopyOp
template <class CopyOp>
struct TME_LOAD_Unpack
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    auto src_coord = src.data().coord_;
    static_assert(is_smem<TD>::value, "MP31_TME_LOAD requires the destination be shared memory.");
    void* dst_ptr = mute::raw_pointer_cast(dst.data());

    auto aux_params = get<2>(traits.opargs_);
    auto block_dim = mute::flatten(aux_params.block_dim_);

    return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                 traits.opargs_, make_seq<2>{},
                                 make_tuple(dst_ptr), seq<0>{},
                                 src_coord, tuple_seq<decltype(src_coord)>{},
                                 block_dim, tuple_seq<decltype(block_dim)>{});
  }
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TME_LOAD ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_OP : MP31_TME_LOAD_ENTRY<sg, ss, sl, prefetch> {};


// The non-executable MP31_TME_LOAD_ENTRY with tme_desc and no tme_bar
// Use .with(tme_bar) to construct an executable version
template <
  class NumBitsPerTME,
  class AuxParams_,
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct Copy_Traits<MP31_TME_LOAD_ENTRY<sg, ss, sl, prefetch>, NumBitsPerTME, AuxParams_>
{
  using ThrID    = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // MP31_TME_LOAD arguments
  TmeDescriptor tme_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmeDescriptor
  MUTE_HOST_DEVICE constexpr
  TmeDescriptor const*
  get_tme_descriptor() const {
    return &tme_desc_;
  }

  // Construct an executable MP31_TME_LOAD with tme_bar
  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP31_TME_LOAD_OP<sg, ss, sl, prefetch>, NumBitsPerTME, AuxParams>
  with(uint32_t tme_bar) const {
    return {{}, {&tme_desc_, tme_bar, aux_params_}};
  }

  // Generate the TME coord tensor
  template <class GShape>
  MUTE_HOST_DEVICE constexpr
  auto
  get_tme_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_counting_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // Don't try to execute a copy with SM90_TME_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

// The executable MP31_TME_LOAD with tme_desc, tme_bar & tme_params
template <
  class NumBitsPerTME,
  class AuxParams,
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct Copy_Traits<MP31_TME_LOAD_OP<sg, ss, sl, prefetch>, NumBitsPerTME, AuxParams>
     : TME_LOAD_Unpack<MP31_TME_LOAD_OP<sg, ss, sl, prefetch>>
{
  using ThrID    = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // MP31_TME_LOAD arguments
  tuple<
    TmeDescriptor const*,
    uint32_t const&,
    AuxParams const&
  > const opargs_;
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TME_STORE //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// The executable MP31_TME_STORE_ENTRY with tme_desc
template <
  class NumBitsPerTME,
  class AuxParams_,
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl
>
struct Copy_Traits<MP31_TME_STORE_ENTRY<sg, ss, sl>, NumBitsPerTME, AuxParams_>
{
  using ThrID    = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // MP31_TME_STORE arguments
  TmeDescriptor tme_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmeDescriptor
  MUTE_HOST_DEVICE constexpr
  TmeDescriptor const*
  get_tme_descriptor() const {
    return &tme_desc_;
  }

  // Generate the TME coord tensor
  template <class GShape>
  MUTE_HOST_DEVICE constexpr
  auto
  get_tme_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_counting_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for MP31_TME_STORE");
    //static_assert(is_gmem<TD>::value, "Expected gmem dst for MP31_TME_STORE");

    void const* const desc_ptr = &(traits.tme_desc_);
    void const* const smem_ptr = mute::raw_pointer_cast(src.data());
    auto dst_coord = dst.data().coord_;
    auto block_dim = mute::flatten(traits.aux_params_.block_dim_);

#if 0
    auto [c0,c1,c2,c3,c4] = append<5>(dst_coord, 0);
    if (thread0()) {
      printf("TME CRD (%d,%d,%d,%d,%d) SMEM_ADDR (%p)\n",
             int32_t(c0), int32_t(c1), int32_t(c2), int32_t(c3), int32_t(c4), smem_ptr);
    }
#endif

    return detail::explode_tuple(detail::CallCOPY<MP31_TME_STORE_ENTRY<sg, ss, sl>>{},
                                 make_tuple(desc_ptr, smem_ptr), seq<0,1>{},
                                 dst_coord, tuple_seq<decltype(dst_coord)>{},
                                 block_dim, tuple_seq<decltype(block_dim)>{});
  }
};


//////////////////////////////////////////////////////////////////////////////
///////////////////////////// BLK_COPY ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <class NumBitsPerTME, class... OpArgs>
struct Copy_Traits<MP31_BLK_COPY_G2S, NumBitsPerTME, OpArgs...>
{

  using ThrID    = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // MP31_BLK_COPY_G2S arguments
  // 0: uint32_t& barrier_id
  mute::tuple<OpArgs...> blk_load_bar_;

  // Record the async barrier for the instruction
  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP31_BLK_COPY_G2S, NumBitsPerTME, uint32_t const&>
  with(uint32_t tme_bar) const {
    return {{tme_bar}};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_same<mute::tuple<OpArgs...>, mute::tuple<uint32_t const&>>::value,
                  "Extra arguments not set. Set .with() before use.");
    static_assert(is_gmem<TS>::value, "Expected gmem src for MP31_BLK_COPY_G2S");
    static_assert(is_smem<TS>::value, "Expected smem dst for MP31_BLK_COPY_G2S");
    MP31_BLK_COPY_G2S::copy(raw_pointer_cast(src.data()), get<0>(traits.blk_load_bar_),
                            raw_pointer_cast(dst.data()), int32_t(NumBitsPerTME::value / 8));

  }

};

//
// MAKE_TME_COPY and related
//

namespace detail {

// Custom version of coalesce that greedily combines modes only up to size-65536
// Look at each element and the back of the stack (in order of priority)
// back(NewLayout)  get<I>(OldLayout)
//      s0:d0           _1:d1     =>  continue
//      _1:d0           s1:d1     =>  replace_back     s1:d1
//      s0:d0           s1:s0*d0  =>  replace_back  s0*s1:d0   if s0*s1 <= 65536
//      s0:d0           s1:d1     =>  append           s1:d1
//
// @pre OldShape and OldStride are flat
template <int I, class OldShape, class OldStride, class NewShape, class NewStride>
MUTE_HOST_DEVICE constexpr
auto
coalesce_65536_impl(OldShape const& old_shape, OldStride const& old_stride,
                    NewShape const& new_shape, NewStride const& new_stride)
{
  if constexpr (I == rank_v<OldShape>) {
    // Base case, we're done
    if constexpr (is_constant<1, NewShape>::value) {
      return Layout<_1,_0>{};
    } else {
      return Layout<NewShape,NewStride>{new_shape,new_stride};
    }
  } else if constexpr (is_constant<1, decltype(get<I>(old_shape))>::value) {
    // shape<I>(layout) == _1, skip it and continue
    return coalesce_65536_impl<I+1>(old_shape, old_stride, new_shape, new_stride);
  } else if constexpr (is_constant<1, NewShape>::value) {
    // Replace our shape-1 with anything (Can only happen on input new_shape/new_stride)
    return coalesce_65536_impl<I+1>(old_shape, old_stride, get<I>(old_shape), get<I>(old_stride));
  } else if constexpr (is_constant<true, decltype(back(new_shape) * back(new_stride) == get<I>(old_stride) &&
                                                  get<I>(old_shape) * back(new_shape) <= Int<65536>{})>::value) {
    // Merge modes because the shapes and strides match and the merge is 65536 or less
    return coalesce_65536_impl<I+1>(old_shape, old_stride,
                                  replace_back(new_shape, get<I>(old_shape) * back(new_shape)),
                                  new_stride);
  } else {
    // Can't replace or merge, so append a new mode
    return coalesce_65536_impl<I+1>(old_shape, old_stride,
                                  append(new_shape,  get<I>(old_shape)),
                                  append(new_stride, get<I>(old_stride)));
  }

  MUTE_GCC_UNREACHABLE;
}

// Combine all the modes that are possible to combine
// Does not respect the profile of the layout, but does preserve total size
template <class Shape, class Stride>
MUTE_HOST_DEVICE constexpr
auto
coalesce_65536(Layout<Shape,Stride> const& layout)
{
  auto flat_shape  = flatten(layout.shape());
  auto flat_stride = flatten(layout.stride());
  return coalesce_65536_impl<1>(flat_shape, flat_stride, get<0>(flat_shape), get<0>(flat_stride));
}


template <class TmeInternalType,
          class GEngine, class GLayout,
          class SShape, class SStride,
          class VShape, class VStride>
MUTE_HOST_DEVICE constexpr
auto
construct_tme_gbasis(Tensor<GEngine,GLayout> const& gtensor,       // The original GMEM Tensor
                     Layout<SShape,SStride>  const& slayout,       // The layout of SMEM
                     Layout<VShape,VStride>  const& cta_v_map)     // smem_coord to hier gmode
{
  //
  // TME parameter checking
  //
  MUTE_STATIC_ASSERT_V(product_each(shape(slayout)) == product_each(shape(cta_v_map)),
                       "TME requires CTA_Tile and SLayout top-level shape equivalence.");

#if 0
  print("gtensor         : "); print(gtensor); print("\n");
  print("slayout         : "); print(slayout); print("\n");
  print("cta_v_map       : "); print(cta_v_map); print("\n");
#endif

  //
  // TME slayout manipulation
  //

  // Invert the smem to get the largest contiguous vector in the smem layout
  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));

  // Compose with the V-Map to convert smem coord (CTA val idx) to gmem mode
  // smem idx -> gmem mode
  auto sidx2gmode_full = coalesce(composition(cta_v_map, inv_smem_layout));

#if 0
  print("inv_smem_layout : "); print(inv_smem_layout); print("\n");
  print("sidx2gmode_full : "); print(sidx2gmode_full); print("\n");
#endif

  //
  // TME gtensor truncation
  //

  // Truncate any incompatibilities -- no starting in the middle of gmodes
  auto smem_rank = find_if(stride(sidx2gmode_full), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1,decltype(v)>{};
  });
  static_assert(smem_rank > 0, "Could not find a common tile-gmem vectorization. Does the Tile select out major GMEM modes?");

  // Keep only the static-1 basis modes into gmem
  auto sidx2gmode = take<0,smem_rank>(sidx2gmode_full);

#if 0
  print("smem_rank  : "); print(smem_rank); print("\n");
  print("sidx2gmode : "); print(sidx2gmode); print("\n");
#endif

  //
  // TME gtensor manipulation
  //

  // The smem vector is the same units as gtensor, so compose first and then recast
  // tme_val_idx:gmem_strides
  auto tile_gstride = recast<TmeInternalType>(gtensor.compose(sidx2gmode)).layout();
  // Coalesce modes up to size-65536
  // tme_box_shape:gmem_strides
  auto tme_gstride  = coalesce_65536(tile_gstride);

  // Perform the tiling, recast, and coalesce to the gmem vector again, but with indirections to the gtensor modes
  auto gbasis = make_identity_layout(shape(gtensor));
  auto tile_gbasis_tmp = gbasis.compose(sidx2gmode);

  // Instead of the recast (gbasis doesn't have type info), replace the shape with the already-recasted shape
  // tme_box_shape:gmem_mode
  auto tile_gbasis = make_layout(shape(tile_gstride), stride(tile_gbasis_tmp));

  // "Coalesce" the tile basis into a compatible shape with the tme_gstride
  auto tme_gbasis_tile = tile_gbasis.compose(make_layout(wrap(shape(tme_gstride))));

  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<TmeInternalType>(gtensor);

  // Find missing bases that don't appear in tile_gbasis
  auto tile_gbasis_remaining_stride = filter_tuple(flatten(shape (gtensor_T)), flatten(stride(gtensor_T)),
                                                   flatten(stride(gbasis)),
                                                   [&](auto s, auto d, auto e)
  {
    if constexpr (is_constant<1, decltype(s)>::value || is_constant<0, decltype(d)>::value) {
      return mute::tuple<>{};          // If size-1 or stride-0, then don't append
    } else {
      using E = decltype(e);
      auto has_e = any_of(flatten(stride(tme_gbasis_tile)), [] (auto tb) { return tb == E{}; });
      if constexpr (decltype(has_e)::value) {
        return mute::tuple<>{};        // If d was found, then don't append
      } else {
        return mute::tuple<E>(e);      // Else, this is missing so append
      }
    }
  });

  // Append the remaining basis modes that contribute to the TME with size-1
  auto tile_gbasis_remaining_shape = repeat<rank(tile_gbasis_remaining_stride)>(Int<1>{});
  auto tme_gbasis_full = make_layout(tuple_cat(wrap( shape(tme_gbasis_tile)), wrap(tile_gbasis_remaining_shape )),
                                     tuple_cat(wrap(stride(tme_gbasis_tile)), wrap(tile_gbasis_remaining_stride)));

  // Group the trailing modes to make this max rank-5 -- TME rank limitation
  // tme_box_shape:gmem_mode
  auto tme_gbasis = group<mute::min(rank(tme_gbasis_full),4),-1>(tme_gbasis_full);

#if 0
  print("tile_gstride : "); print(tile_gstride); print("\n");
  print("tme_gstride  : "); print(tme_gstride); print("\n");
  print("gbasis       : "); print(gbasis); print("\n");
  print("tile_gbasis  : "); print(tme_gbasis_tile); print("\n");
  print("tme_gbasis   : "); print(tme_gbasis); print("\n");
#endif

  return tme_gbasis;
}

template <class GEngine, class GLayout,
          class TmeGmemBasisStride,
          class ShapeT, size_t TmeRank>
MUTE_HOST_DEVICE constexpr
void
fill_tme_gmem_shape_stride(Tensor<GEngine,GLayout>   const& gtensor,           // Gmem Shapes and Strides, in units of TmeInternalType
                           TmeGmemBasisStride        const& tme_gbasis_stride, // Map Tme mode idx -> Gmem mode(s)
                           mute::array<ShapeT,   TmeRank> & gmem_prob_shape,   // Tme Shapes, uint32_t or uin64_t
                           mute::array<uint64_t, TmeRank> & gmem_prob_stride)  // Tme Strides
{
  static_assert(is_tuple<TmeGmemBasisStride>::value);
  static_assert(is_same<uint32_t, ShapeT>::value || is_same<uint64_t, ShapeT>::value);

  using TmeInternalType = typename GEngine::value_type;
  constexpr int tme_rank = decltype(rank(tme_gbasis_stride))::value;
  static_assert(TmeRank >= tme_rank);

  auto gmem_shape  =  shape(gtensor);
  auto gmem_stride = stride(gtensor);
  // Use the indirections in tme_gbasis_stride into gtensor to construct the tme gmem shapes/strides
  for_each(make_seq<tme_rank>{}, [&](auto i) {
    constexpr int tme_i_rank = decltype(rank<i>(tme_gbasis_stride))::value;
    if constexpr (tme_i_rank == 1) {
      // Trivial contribution of this gmem mode to this tme mode
      auto ej = unwrap(get<i>(tme_gbasis_stride));
      gmem_prob_shape[i]  = basis_get(ej, gmem_shape);
      gmem_prob_stride[i] = basis_get(ej, gmem_stride);
    } else {
      // Apply a recurrence to each gmem mode that contributes to this tme mode
      for_each(get<i>(tme_gbasis_stride), [&](auto ej) {
        // Problem shape
        uint64_t shape_j  = basis_get(ej, gmem_shape);
        // Problem stride (in bytes)
        uint64_t stride_j = basis_get(ej, gmem_stride);
        uint64_t old_stride = gmem_prob_stride[i];
        gmem_prob_stride[i] = gcd(gmem_prob_stride[i], stride_j);

        if (gmem_prob_stride[i] != 0) {
          // Recurrence: g_shape = (s_i - 1) * (d_i / gcd_j d_j) + 1
          gmem_prob_shape[i] = (gmem_prob_shape[i]-1) * (old_stride / gmem_prob_stride[i])
                             +            (shape_j-1) * (stride_j   / gmem_prob_stride[i])
                             + 1;
        } else {
          gmem_prob_shape[i] = shape_j;
        }
      });
    }
  });
}


template <class TmeInternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class TShape,  class TStride,
          int B, int M, int S>
MUTE_HOST_RTC
auto
make_tme_copy_desc(Tensor<GEngine,GLayout> const& gtensor,         // The original GMEM Tensor
                   Layout<TShape,TStride>  const& tme_gbasis,      // TME mode -> GMEM mode mapping
                   Swizzle<B,M,S>          const& swizzle)         // Swizzle fn on smem_idx
{
  //
  // TME desc creation
  //

  constexpr int tme_dim = decltype(rank(tme_gbasis))::value;

  //
  // TME gmem desc info
  //

  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<TmeInternalType>(gtensor);

  void* gmem_address = (void*)raw_pointer_cast(gtensor_T.data());
  auto  gmem_layout  = gtensor_T.layout();

  mute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
  mute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};

  fill_tme_gmem_shape_stride(gtensor_T, stride(tme_gbasis), gmem_prob_shape, gmem_prob_stride);

  // TME descriptor does not store the zeroth stride and assumes it is 1 (TmaInternalType element).
  assert(gmem_prob_stride[0] == 1 && "Majorness of smem doesn't match majorness of gmem");

  // convert strides to byte strides
  for(uint64_t& stride : gmem_prob_stride) {
    stride = (stride * sizeof_bits_v<TmeInternalType>) / 8;
  }


  //
  // Construct the descriptor
  //

  TmeDescriptor tme_desc{};

  //
  // TME general info
  //

  MUtensorDescriptorDataType   tme_format     = TME::to_MUtensorDescriptorDataType<TmeInternalType>();
  MUtensorDescriptorInterleave tme_interleave = MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE;
  uint64_t                     tme_oobFill    = 0;

  MUresult result = muTensorDescriptorEncode(
      &tme_desc,
      tme_format,
      tme_dim,
      gmem_address,
      gmem_prob_shape.data(),
      gmem_prob_stride.data() + 1, // gmem_prob_stride[0] implicitly 1
      tme_interleave,
      tme_oobFill);

  if (result != MUSA_SUCCESS) {
    std::cerr << "TME Desc Addr:    " << &tme_desc
              << "\nformat          " << tme_format
              << "\ndim             " << tme_dim
              << "\ngmem_address    " << gmem_address
              << "\nglobalDim       " << gmem_prob_shape
              << "\nglobalStrides   " << gmem_prob_stride
              << "\ninterleave      " << tme_interleave
              << "\noobFill         " << tme_oobFill << std::endl;
    std::cerr << "Error: Failed to initialize the TME descriptor: "
              << result << std::endl;
    //assert(false);
  }

  //
  // TME auxiliary params
  //
  auto recast_ratio = mute::trait_ratio(sizeof_bits<typename GEngine::value_type>{},
                                        sizeof_bits<             TmeInternalType>{});

  auto gbasis = make_basis_like(shape(gtensor));

  // Finally, get the inverse permutation of the E<i> bases for the mocked gmem stride
  auto gmem_tme_basis_stride = transform_leaf(gbasis, [&](auto ei) {
    auto si = basis_get(ei,  shape(gmem_layout));
    auto di = basis_get(ei, stride(gmem_layout));
    if constexpr (is_constant<1, decltype(si)>::value || is_constant<0, decltype(di)>::value) {
      return Int<0>{};                  // If size-1 or stride-0, return arithmetic identity -- no contribution to the TME
    } else {
      auto tme_gmem_basis_stride = stride(tme_gbasis);
      // Find j such that E<i> is in stride<j>(tme_gbasis)
      using EI = decltype(ei);
      [[maybe_unused]] auto j = find_if(tme_gmem_basis_stride, [&](auto tme_stride_j) { return any_of(tme_stride_j, [&](auto dj) { return dj == EI{}; }); });
      if constexpr (decltype(j == rank(tme_gmem_basis_stride))::value) {
        return Int<0>{};               // If not-found, return arithmetic identity -- no contribution to the TME
      } else
      if constexpr (decltype(j == Int<0>{})::value) {
        auto scale = recast_ratio * basis_get(ei, stride(gtensor));
        return E<j>{} * scale;         // Return TME Coord basis -- with a recast scale factor
      } else
      if constexpr (decltype(rank<j>(tme_gmem_basis_stride) == Int<1>{})::value) {
        return E<j>{};                 // Return TME Coord basis -- known scale of Int<1>{}
      } else {
        int32_t scale = ceil_div(int32_t(di * sizeof_bits_v<TmeInternalType> / mute::max(gmem_prob_stride[j], uint64_t{16})), 8);
        return E<j>{} * scale;         // Return TME Coord basis -- with a dynamic scale factor
      }
    }
  });

  constexpr auto attributes             = get_tme_swizzle_attributes(Swizzle<B, M, S>{});
  constexpr auto tme_SwizzleGranularity = get<0>(attributes);
  constexpr auto tme_SwizzleStride      = get<1>(attributes);
  constexpr auto tme_SwizzleLine        = get<2>(attributes);
  constexpr auto tme_l2Prefetch         = PrefetchSize::B128;


  static_assert(is_same_v<CopyOp, MP31_TME_LOAD> || is_same_v<CopyOp, MP31_TME_STORE>, "Unsupported CopyOp");

  using CopyOpEntry = mute::conditional_t<is_same_v<CopyOp, MP31_TME_LOAD>,
                                          MP31_TME_LOAD_ENTRY<tme_SwizzleGranularity, tme_SwizzleStride, tme_SwizzleLine, tme_l2Prefetch>,
                                          MP31_TME_STORE_ENTRY<tme_SwizzleGranularity, tme_SwizzleStride, tme_SwizzleLine>>;

  auto block_dim = product_each(take<0, tme_dim>(tme_gbasis).shape());

#if 0
  print("block dim                :"); print(block_dim); print("\n");
  print("gmem_tme_basis_stride   : "); print(gmem_tme_basis_stride); print("\n");
  print("tme swizzle line        : "); print(to_string(tme_SwizzleLine)); print("\n");
  print("tme swizzle stride      : "); print(to_string(tme_SwizzleStride)); print("\n");
  print("tme swizzle granularity : "); print(to_string(tme_SwizzleGranularity)); print("\n");
  print("tme prefetch size       : "); print(to_string(tme_l2Prefetch)); print("\n");
#endif

  using AuxParams = AuxTmeParams<decltype(gmem_tme_basis_stride),
                                 decltype(tme_gbasis),
                                 decltype(swizzle),
                                 decltype(block_dim)>;
  return mute::make_tuple(tme_desc,
                          AuxParams{gmem_tme_basis_stride, block_dim},
                          CopyOpEntry{});
}

template <class TmeInternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride>
MUTE_HOST_RTC
auto
make_tme_copy_atom(CopyOp,
                   Tensor<GEngine,GLayout> const& gtensor,       // Full GMEM Tensor
                   SLayout                 const& slayout,       // CTA Tile of SMEM, potentially swizzled
                   Layout<VShape,VStride>  const& cta_v_map)     // V: CTA val idx -> gmem mode
{
  //
  // TME truncated layout
  //

  auto smem_swizzle = get_swizzle_portion(slayout);
  auto smem_layout  = get_nonswizzle_portion(slayout);

  auto tme_gbasis = detail::construct_tme_gbasis<TmeInternalType>(gtensor, smem_layout, cta_v_map);

  auto [tme_desc, aux_params, copy_op] = detail::make_tme_copy_desc<TmeInternalType, CopyOp>(
                                                                                     gtensor,
                                                                                     tme_gbasis,
                                                                                     smem_swizzle);
  //
  // Construct the Copy_Traits
  //

  constexpr int num_bits_per_tme = size(tme_gbasis) * sizeof_bits_v<TmeInternalType>;
  using Traits = Copy_Traits<decltype(copy_op), mute::C<num_bits_per_tme>, decltype(aux_params)>;
  using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;


  Traits tme_traits{tme_desc, aux_params};

#if 0
  print("num_bits_per_tme    : "); print(num_bits_per_tme); print("\n");
  print("g_stride_bases      : "); print(tme_traits.aux_params_.g_stride_); print("\n");
  print("smem block dim      : "); print(tme_traits.aux_params_.block_dim_); print("\n");
#endif

  // Return the Copy_Atom
  return Atom{tme_traits};

}

template <class TmeInternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride>
MUTE_HOST_RTC
auto
make_tme_copy_tiled(CopyOp                   const& copy_op,
                    Tensor<GEngine, GLayout> const& gtensor,    // Full GMEM Tensor
                    SLayout                  const& slayout,    // CTA Tile of SMEM
                    Layout<VShape, VStride>  const& cta_v_map)
{
  Copy_Atom atom = make_tme_copy_atom<TmeInternalType>(copy_op, gtensor, slayout, cta_v_map);

  //
  // Construct the TiledCopy
  //

  [[maybe_unused]] auto cta_tiler = product_each(shape(cta_v_map));

  auto num_elems_per_tme = size<1>(typename decltype(atom)::RefLayout{}) / static_value<sizeof_bits<typename GEngine::value_type>>();

  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  // CTA V -> smem_coord
  auto layout_v = composition(inv_smem_layout, num_elems_per_tme);
  // Scale that up to cover all of the smem_coords
  auto layout_V = tile_to_shape(make_layout(layout_v), size(cta_v_map));

  // Simplely construct thread layout.
  // TME is a warp-level instruction and only the first active lane will issue.
  auto layout_T = Layout<_1, _0>{};
  // Combine with the T mapping
  [[maybe_unused]] auto layout_TV = make_layout(layout_T, layout_V);

#if 0
  print("cta_tiler : "); print(cta_tiler); print("\n");
  print("layout_v  : "); print(layout_v); print("\n");
  print("layout_V  : "); print(layout_V); print("\n");
  print("layout_T  : "); print(layout_T); print("\n");
  print("layout_TV : "); print(layout_TV); print("\n");
#endif

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};

}

} // namespace detail


template <class TmeInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler>
MUTE_HOST_RTC
auto
make_tme_copy(CopyOp                   const& copy_op,
              Tensor<GEngine, GLayout> const& gtensor,
              SLayout                  const& slayout,
              CTA_Tiler                const& cta_tiler)
{
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);

  using TmeType = conditional_t<is_same<void, TmeInternalType>::value, typename GEngine::value_type, TmeInternalType>;
  return detail::make_tme_copy_tiled<TmeType>(copy_op,
                                              gtensor, slayout,
                                              cta_v_tile);
}

// Explicit defaulting
template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout>
MUTE_HOST_RTC
auto
make_tme_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout)
{
  return make_tme_copy(copy_op, gtensor, slayout, product_each(shape(slayout)));
}

//////////////////////////////////////////////////
// Experimental make TME Atom and TME Partitioner
//////////////////////////////////////////////////


template <class TmeInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler>
MUTE_HOST_RTC
auto
make_tme_atom(CopyOp                   const& copy_op,
              Tensor<GEngine, GLayout> const& gtensor,
              SLayout                  const& slayout,
              CTA_Tiler                const& cta_tiler)
{
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);

  using TmeType = conditional_t<is_same<void, TmeInternalType>::value, typename GEngine::value_type, TmeInternalType>;
  return detail::make_tme_copy_atom<TmeType>(copy_op,
                                             gtensor, slayout,
                                             cta_v_tile);
}

template <class... Args,
          class WarpCoord,
          class WarpLayout,
          class SEngine, class SLayout,
          class GEngine, class GLayout>
MUTE_HOST_DEVICE
auto
tme_partition(Copy_Atom<Args...>       const& copy_atom,
              WarpCoord                const& warp_coord,
              WarpLayout               const& warp_layout,  // warp id
              Tensor<SEngine, SLayout> const& stensor,      // SMEM Tensor (TMETile, Rest...)
              Tensor<GEngine, GLayout> const& gtensor)      // GMEM Tensor (TMETile, Rest...)
{
  MUTE_STATIC_ASSERT_V(size<0>(stensor) == size<0>(gtensor));

  // Invert the smem to get the largest contiguous vector in the smem layout
  Layout inv_smem_layout = right_inverse(get_nonswizzle_portion(layout<0>(stensor)));

  // Scale that up to conver all of the smem_coords
  Layout layout_v = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor));

  // Factor out the single-instruction portion
  Layout tme_layout_v = make_layout(Int<Copy_Atom<Args...>::NumValSrc>{});
  auto layout_V = logical_divide(layout_v, tme_layout_v);

  // TME Iters must be divided by issue_warps
  static_assert(size<1>(layout_V) % cosize_v<WarpLayout> == 0, "TME Iters must be divided by issue warps");

  Layout tme_layout_v_ext = make_layout(Shape<Int<Copy_Atom<Args...>::NumValSrc>, Int<cosize_v<WarpLayout>>>{});
  auto layout_V_ext = make_tile(flat_divide(layout_v, tme_layout_v_ext));

  // Append with _ until we conver all Rest... modes
  auto glayout_V = append<GLayout::rank>(layout_V_ext, _);
  auto slayout_V = append<SLayout::rank>(layout_V_ext, _);
  // Transform tile mode and coalesce
  Tensor gtensor_v = coalesce(gtensor.compose(glayout_V), Shape<Shape<_1, _1>>{}); // ((TME, IssueWarps, TME Iter/IssueWarps), Rest...)
  Tensor stensor_v = coalesce(stensor.compose(slayout_V), Shape<Shape<_1, _1>>{}); // ((TME, IssueWarps, TME Iter/IssueWarps), Rest...)

  // Offset inside the TME-mode for the warp slice
  auto warp_offset = warp_layout(warp_coord);
  auto slice_coord = make_coord(make_coord(Underscore{}, warp_offset, Underscore{}));

  auto scoord = append<SLayout::rank>(slice_coord, Underscore{});
  auto gcoord = append<GLayout::rank>(slice_coord, Underscore{});

  Tensor sresult = stensor_v(scoord);
  Tensor gresult = gtensor_v(gcoord);

  return mute::make_tuple(gresult, sresult);
}

} // namespace mute
