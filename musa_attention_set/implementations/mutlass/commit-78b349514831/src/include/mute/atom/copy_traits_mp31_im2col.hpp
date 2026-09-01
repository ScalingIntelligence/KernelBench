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

#if !defined(__MUSACC_RTC__)
#include <musa.h>
#endif

#include <mute/numeric/integral_constant.hpp>
#include <mute/atom/copy_traits.hpp>
#include <mute/atom/copy_traits_mp31_tme.hpp>
#include <mute/atom/copy_atom.hpp>

namespace mute
{

template <
  class TmeSwizzle_,
  class TmeIm2ColBlockDim_,
  class TmeIm2ColConvParam_,
  class TmeIm2ColOutputDim_
>
struct AuxTmeIm2ColParams
{
  using TmeSwizzle = TmeSwizzle_;
  static_assert(is_static<TmeSwizzle>::value);

  using TmeIm2ColBlockDim = TmeIm2ColBlockDim_;
  TmeIm2ColBlockDim block_dim_;

  using TmeIm2ColConvParam = TmeIm2ColConvParam_;
  TmeIm2ColConvParam conv_param_;

  using TmeIm2ColOutputDim = TmeIm2ColOutputDim_;
  TmeIm2ColOutputDim output_dim_;
};

template <class CopyOp>
struct TME_LOAD_IM2COL_Unpack
{
  template <
    class... Args,
    class TS, class SLayout,
    class TD, class DLayout
  >
  MUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    static_assert(is_smem<TD>::value, "MP31_TME_LOAD_IM2COL requires the destination be shared memory.");

    // For Fprop: (c, q, p, z, n, s, r, t)
    auto src_coord = flatten(src(Int<0>{}));

    void* dst_ptr = mute::raw_pointer_cast(dst.data());

    const auto &aux_param  = get<2>(traits.opargs_);
    const auto &block_dim  = aux_param.block_dim_;                                                  // (dim_c, dim_nzpq)
    const auto &conv_param = aux_param.conv_param_;                                       // (padding, stride, dilation)
    const auto &output_dim = aux_param.output_dim_;                                                         // (Q, P, Z)

    return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                 traits.opargs_, make_seq<2>{},
                                 make_tuple(dst_ptr), seq<0>{},
                                 src_coord, tuple_seq<decltype(src_coord)>{},
                                 block_dim, seq<0, 1>{},
                                 make_tuple(conv_param), seq<0>{},
                                 make_tuple(output_dim), seq<0>{});
  }
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TME_IM2COl_LOAD ////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_IM2COL_OP : MP31_TME_LOAD_IM2COL_ENTRY<sg, ss, sl, prefetch> {};


// The non-executable MP31_TME_IM2COL_LOAD_ENTRY with tme_desc and no tme_bar
// Use .with(tme_bar) to construct an executable version
template <
  class NumBitsPerTME,
  class AuxParams,
  class TMETensor,
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct Copy_Traits<MP31_TME_LOAD_IM2COL_ENTRY<sg, ss, sl, prefetch>,
                   NumBitsPerTME,
                   AuxParams,
                   TMETensor>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // MP31_TME_IM2COL_LOAD arguments
  TmeDescriptor tme_desc_;

  AuxParams aux_params_;

  TMETensor tme_tensor_;

  // Return TmeDescriptor
  MUTE_HOST_DEVICE constexpr
  TmeDescriptor const*
  get_tme_descriptor() const {
    return &tme_desc_;
  }

  // Construct an executable MP31_TME_LOAD_IM2COL with tme_bar
  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP31_TME_LOAD_IM2COL_OP<sg, ss, sl, prefetch>, NumBitsPerTME, AuxParams>
  with(uint32_t tme_bar) const {
    return {{}, {&tme_desc_, tme_bar, aux_params_}};
  }

  // Generate the TME coord tensor
  template <class GShape>
  MUTE_HOST_DEVICE constexpr
  auto
  get_tme_tensor(GShape const&) const {
    return tme_tensor_;
  }

  // Don't try to execute a copy with MP31_TME_LOAD_IM2COL before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

// The executable MP31_TME_IM2COL_LOAD with tme_desc, tme_im2col_desc, tme_bar & tme_params
template <
  class NumBitsPerTME,
  class AuxParams,
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct Copy_Traits<MP31_TME_LOAD_IM2COL_OP<sg, ss, sl, prefetch>, NumBitsPerTME, AuxParams>
     : TME_LOAD_IM2COL_Unpack<MP31_TME_LOAD_IM2COL_OP<sg, ss, sl, prefetch>>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // MP31_TME_IM2COL_LOAD arguments
  tuple<
    TmeDescriptor const*,
    uint32_t const&,
    AuxParams const&
  > const opargs_;
};

namespace detail {

template <class Shape>
auto
make_trait_vec(Shape const& s)
{
  if constexpr (rank(Shape{}) == 3) {
    return mute::v3i32_t{size<0>(s), size<1>(s), size<2>(s)};
  } else if constexpr (rank(Shape{}) == 2) {
    return mute::v2i32_t{size<0>(s), size<1>(s)};
  } else {
    return int32_t{size<0>(s)};
  }
}

template <
  class TmeInternalType,
  class EngineA, class LayoutA,
  class LowerPadding,
  class Stride,
  class Dilation,
  class WeightPosSRT
>
MUTE_HOST_RTC
auto
make_tme_im2col_copy_desc(Tensor<EngineA, LayoutA> const& tensor,
                          LowerPadding             const& lower_padding_whd,
                          Stride                   const& stride_whd,
                          Dilation                 const& dilation_whd,
                          WeightPosSRT             const& srt)
{
  Tensor tensor_cwhdn = recast<TmeInternalType>(tensor);
  void* gmem_address = (void*)raw_pointer_cast(tensor_cwhdn.data());

  constexpr uint32_t num_modes = LayoutA::rank;

  mute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
  for_each(make_seq<num_modes>{}, [&](auto i) {
    gmem_prob_shape[i] = static_cast<uint64_t>(shape<i>(tensor_cwhdn));
  });

  constexpr int elem_byte = mutlass::bits_to_bytes(sizeof_bits_v<TmeInternalType>);
  mute::array<uint64_t, 5> gmem_prob_stride = {elem_byte,0,0,0,0};
  for_each(make_seq<num_modes - 1>{}, [&](auto i) {
    gmem_prob_stride[i + 1] = gmem_prob_stride[i] * gmem_prob_shape[i];
  });

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

  MUresult result = MUTLASS_MUSA_DRIVER_WRAPPER_CALL(muTensorDescriptorEncode)(
      &tme_desc,
      tme_format,
      num_modes,
      gmem_address,
      gmem_prob_shape.data(),
      gmem_prob_stride.data() + 1,
      tme_interleave,
      tme_oobFill);

  if (result != MUSA_SUCCESS) {
    std::cerr << "TME Desc Addr:    " << &tme_desc
              << "\nformat          " << tme_format
              << "\nnum_modes       " << num_modes
              << "\ngmem_address    " << gmem_address
              << "\nglobalDim       " << gmem_prob_shape
              << "\nglobalStrides   " << gmem_prob_stride
              << "\ninterleave      " << tme_interleave
              << "\noobFill         " << tme_oobFill << std::endl;

    std::cerr << "Error: Failed to initialize the TME descriptor: "
              << result << std::endl;
    assert(false);
  }

  //
  // Construct the IM2COL Parameter
  //

  TmeIm2ColParam tme_im2col_param{};

  //
  // TME IM2COL general info
  //

  constexpr int            num_spatial_modes    = num_modes - 2;
  mute::array<uint32_t, 3> im2col_lower_padding = {0, 0, 0};
  mute::array<uint32_t, 3> im2col_stride        = {0, 0, 0};
  mute::array<uint32_t, 3> im2col_dilation      = {0, 0, 0};

  for_each(make_seq<num_spatial_modes>{}, [&](auto i){
    im2col_lower_padding[i] = static_cast<uint32_t>(get<i>(lower_padding_whd));
    im2col_stride[i] = static_cast<uint32_t>(get<i>(stride_whd));
    im2col_dilation[i] = static_cast<uint32_t>(get<i>(dilation_whd));
  });

  result = MUTLASS_MUSA_DRIVER_WRAPPER_CALL(muTensorIm2colConvParamEncode)(
      &tme_im2col_param,
      num_spatial_modes,
      im2col_lower_padding.data(),
      im2col_stride.data(),
      im2col_dilation.data());


  if (result != MUSA_SUCCESS) {
    std::cerr << "TME im2col Param Addr:    " << &tme_im2col_param
              << "\nnum_spatial_modes       " << num_spatial_modes
              << "\nlower_padding           " << im2col_lower_padding
              << "\nstride                  " << im2col_stride
              << "\ndilation                " << im2col_dilation << std::endl;
    std::cerr << "Error: Failed to initialize the TME Im2Col Parameters: "
              << result << std::endl;
    assert(false);
  }

  mute::v3i32_t conv_param = {static_cast<int32_t>(tme_im2col_param.params[0]),
                              static_cast<int32_t>(tme_im2col_param.params[1]),
                              static_cast<int32_t>(tme_im2col_param.params[2])};

  return mute::make_tuple(tme_desc, conv_param);
}

template <
  class TmeInternalType,
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout,
  class VShape, class VStride,
  class LowerPadding,
  class UpperPadding,
  class Stride,
  class Dilation,
  class WeightPosSRT
>
MUTE_HOST_RTC
auto
make_tme_atom_im2col(CopyOp,
                     Tensor<GEngine,GLayout> const& gtensor,
                     SLayout                 const& slayout,
                     Layout<VShape,VStride>  const& cta_v_map,
                     LowerPadding            const& lower_padding_whd,
                     UpperPadding            const& upper_padding_whd,
                     Stride                  const& stride_whd,
                     Dilation                const& dilation_whd,
                     WeightPosSRT            const& srt)
{
  // FProp gtensor shape ((w, h, d, n), c)
 constexpr int num_spatial_modes = rank<0>(GLayout{}) - 1;
 constexpr int num_total_modes = num_spatial_modes + 2;

  // Invert the smem to get the largest contiguous vector in the smem layout
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));

  // Map from smem idx to a gmem mode
  auto sidx_to_gmode = coalesce(composition(cta_v_map, inv_smem_layout));

  // Generate a TupleBasis for the gtensor
  auto glayout_basis = make_identity_layout(product_each(shape(gtensor)));

  // Tile the modes of gtensor with the truncated cta_v_map o inv_smem_layout_trunc
  auto tme_layout_full = flatten(composition(glayout_basis, sidx_to_gmode));

  // Truncate any incompatibilities -- no starting in the middle of gmodes
  auto smem_rank = find_if(stride(tme_layout_full), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1,decltype(v)>{};
  });

  static_assert(smem_rank >= 2, "IM2COL expects at least 2 modes of the smem to vectorize with gmem.");

  // IM2COL uses a maximum of 2 modes
  constexpr int smem_tme_rank = mute::min(int(smem_rank), 2);

  // Keep only the static-1 basis modes into gmem
  auto tme_layout_trunc = take<0,smem_tme_rank>(tme_layout_full);

  #if 0
    print("gtensor shape      : "); print(shape(gtensor)); print("\n");
    print("num_total_modes    : "); print(num_total_modes); print("\n");
    print("num_spatial_modes  : "); print(num_spatial_modes); print("\n");
    print("cta_v_map          : "); print(cta_v_map); print("\n");
    print("inv_smem_layout    : "); print(inv_smem_layout); print("\n");
    print("sidx_to_gmode      : "); print(sidx_to_gmode); print("\n");
    print("glayout_basis      : "); print(glayout_basis); print("\n");
    print("tme_layout_full    : "); print(tme_layout_full); print("\n");
    print("smem_rank          : "); print(smem_rank); print("\n");
    print("tme_layout_trunc   : "); print(tme_layout_trunc); print("\n");
  #endif

  auto shape_cwhdn = flatten(make_shape(basis_get(stride<0>(tme_layout_trunc), gtensor.shape()),
                                        basis_get(stride<1>(tme_layout_trunc), gtensor.shape())));
  auto layout_cwhdn = make_layout(shape_cwhdn);
  Tensor tensor_cwhdn = make_tensor(gtensor.data(), layout_cwhdn);

  // compute tme im2col desc & conv param
  auto [tme_desc, tme_im2col_param] = make_tme_im2col_copy_desc<TmeInternalType>(tensor_cwhdn,
                                                                                 lower_padding_whd,
                                                                                 stride_whd,
                                                                                 dilation_whd,
                                                                                 srt);

  // Fprop: convert (w, h, d, n) to (q, p, z, n)
  auto gemm_mn_ = mute::transform(mute::make_seq<num_spatial_modes>{}, [&](auto i) {
    return (get<i+1>(shape_cwhdn)
            + (get<i>(lower_padding_whd) + get<i>(upper_padding_whd))
            - get<i>(dilation_whd) * (get<i>(srt) - Int<1>{}) - Int<1>{})
            / get<i>(stride_whd) + Int<1>{};
  });
  auto gemm_mn = append(gemm_mn_, get<num_spatial_modes+1>(shape_cwhdn));

  // Fprop: compute (c, s, r, t)
  auto gemm_k_ = srt;
  auto gemm_k = prepend(gemm_k_, get<0>(shape_cwhdn));

  // Fprop: ((q, p, z, n), (c, s, r, t))
  auto gemm_shapes_ = make_shape(gemm_mn, gemm_k);
  auto gemm_shape = make_shape(basis_get(stride<1>(tme_layout_trunc), gemm_shapes_),
                               basis_get(stride<0>(tme_layout_trunc), gemm_shapes_));

  // compute (qpzn, (c, s, r, t))
  auto linear_shape_common = make_shape(size(gemm_mn), gemm_k);
  auto linear_shape = make_shape(basis_get(stride<1>(tme_layout_trunc), linear_shape_common),
                                 basis_get(stride<0>(tme_layout_trunc), linear_shape_common));

  #if 0
    print("layout_cwhdn            : "); print(layout_cwhdn); print("\n");
    print("gemm_mn                 : "); print(gemm_mn); print("\n");
    print("gemm_k                  : "); print(gemm_k); print("\n");
    print("gemm_shape              : "); print(gemm_shape); print("\n");
    print("linear_shape            : "); print(linear_shape); print("\n");
  #endif

  // compute tme basis
  auto tme_basis_scale = make_shape(_1{},
                                    repeat_like(gemm_mn_, _1{}),
                                    _1{},
                                    repeat_like(gemm_k_, _1{}));
  auto tme_basis = elem_scale(tme_basis_scale, make_basis_like(tme_basis_scale));

  auto gbasis_strides_common = make_stride(append(get<1>(tme_basis), get<2>(tme_basis)),
                                           prepend(get<3>(tme_basis), get<0>(tme_basis)));
  auto gbasis_strides = make_stride(basis_get(stride<1>(tme_layout_trunc), gbasis_strides_common),
                                    basis_get(stride<0>(tme_layout_trunc), gbasis_strides_common));

  #if 0
    print("tme_basis_scale         : "); print(tme_basis_scale); print("\n");
    print("tme_basis               : "); print(tme_basis); print("\n");
    print("gbasis_strides_common   : "); print(gbasis_strides_common); print("\n");
    print("gbasis_strides          : "); print(gbasis_strides); print("\n");
  #endif

  auto lower_corner = make_arithmetic_tuple(_0{},
                                            repeat_like(gemm_mn_, _0{}),
                                            _0{},
                                            repeat_like(gemm_k_, _0{}));
  auto tensor_multimode = make_tensor(ArithmeticTupleIterator(lower_corner),
                                      gemm_shape,
                                      gbasis_strides);

  auto tensor_linear = make_identity_tensor(linear_shape);
  auto tme_tensor = make_tensor(tensor_multimode.data(), composition(tensor_multimode.layout(),
                                                                     tensor_linear(_0{}),
                                                                     tensor_linear.layout()));

  // build traits
  constexpr auto range_c    = size<0>(tme_layout_trunc);
  constexpr auto range_qpzn = size<1>(tme_layout_trunc);

  constexpr uint64_t num_bits_per_tme = size(tme_layout_trunc) * sizeof_bits_v<TmeInternalType>;
  auto blk_dim = make_shape(range_c, range_qpzn);

  #if 0
    print("tensor_multimode   : "); print(tensor_multimode); print("\n");
    print("tensor_linear0     : "); print(tensor_linear(_0{})); print("\n");
    print("tensor_linear      : "); print(tensor_linear); print("\n");
    print("tme_tensor         : "); print(tme_tensor); print("\n");
  #endif

  constexpr auto smem_swizzle_attr      = get_tme_swizzle_attributes(decltype(get_swizzle_portion(slayout)){});
  constexpr auto tme_SwizzleGranularity = get<0>(smem_swizzle_attr);
  constexpr auto tme_SwizzleStride      = get<1>(smem_swizzle_attr);
  constexpr auto tme_SwizzleSwizzleLine = get<2>(smem_swizzle_attr);
  constexpr auto tme_L2Prefetch         = PrefetchSize::B128;

  auto trait_shape_qpz = make_trait_vec(gemm_mn_);

  using CopyOpEntry = MP31_TME_LOAD_IM2COL_ENTRY<tme_SwizzleGranularity,
                                                 tme_SwizzleStride,
                                                 tme_SwizzleSwizzleLine,
                                                 tme_L2Prefetch>;
  using AuxParam = AuxTmeIm2ColParams<decltype(get_swizzle_portion(slayout)),
                                      decltype(blk_dim),
                                      decltype(tme_im2col_param),
                                      decltype(trait_shape_qpz)>;
  using Traits = Copy_Traits<CopyOpEntry,
                             mute::C<num_bits_per_tme>,
                             AuxParam,
                             decltype(tme_tensor)>;
  using Atom = Copy_Atom<Traits, typename GEngine::value_type>;

  AuxParam aux_param{blk_dim, tme_im2col_param, trait_shape_qpz};
  Traits tme_traits{tme_desc, aux_param, tme_tensor};

  return Atom{tme_traits};
}

template <
  class TmeInternalType,
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout,
  class VShape, class VStride,
  class LowerPadding,
  class UpperPadding,
  class Stride,
  class Dilation,
  class WeightPosSRT
>
MUTE_HOST_RTC
auto
make_tme_copy_im2col(CopyOp                    const& copy_op,
                     Tensor<GEngine, GLayout>  const& tensor_whdn_c,
                     SLayout                   const& slayout,
                     Layout<VShape, VStride>   const& cta_v_map,
                     LowerPadding              const& lower_padding_whd,
                     UpperPadding              const& upper_padding_whd,
                     Stride                    const& stride_whd,
                     Dilation                  const& dilation_whd,
                     WeightPosSRT              const& srt)
{
  Copy_Atom atom = make_tme_atom_im2col<TmeInternalType>(copy_op,
                                                         tensor_whdn_c,
                                                         slayout,
                                                         cta_v_map,
                                                         lower_padding_whd,
                                                         upper_padding_whd,
                                                         stride_whd,
                                                         dilation_whd,
                                                         srt);

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
  print("num_elem_per_tme : "); print(num_elems_per_tme); print("\n");
  print("slayout          : "); print(slayout); print("\n");
  print("inv_smem_layout  : "); print(inv_smem_layout); print("\n");
  print("cta_v_map        : "); print(cta_v_map); print("\n");
  print("cta_tiler        : "); print(cta_tiler); print("\n");
  print("layout_v         : "); print(layout_v); print("\n");
  print("layout_V         : "); print(layout_V); print("\n");
  print("layout_T         : "); print(layout_T); print("\n");
  print("layout_TV        : "); print(layout_TV); print("\n");
#endif

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}

template <
  class TmeInternalType,
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout,
  class VShape, class VStride
>
MUTE_HOST_RTC
auto
make_tme_copy_im2col(CopyOp                    const& copy_op,
                     Tensor<GEngine, GLayout>  const& tensor_whdn_c,
                     SLayout                   const& slayout,
                     Layout<VShape, VStride>   const& cta_v_map)
{
  constexpr int num_spatial_modes = rank<0>(tensor_whdn_c) - 1;

  return make_tme_copy_im2col(copy_op,
                              tensor_whdn_c,
                              slayout,
                              cta_v_map,
                              append<num_spatial_modes>(Stride<_0>{}, Int<0>{}),   // LowerPadding
                              append<num_spatial_modes>(Stride<_0>{}, Int<0>{}),   // UpperPadding
                              append<num_spatial_modes>(Stride<_1>{}, Int<1>{}),   // Stride
                              append<num_spatial_modes>(Stride<_1>{}, Int<1>{}),   // Dilation
                              append<num_spatial_modes>(Stride<_1>{}, Int<1>{}));  // WeightPos
}

} // namespace detail

template <
  class TmeInternalType = void,
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout,
  class CTATiler,
  class LowerPadding,
  class UpperPadding,
  class Stride,
  class Dilation,
  class WeightPosSRT
>
MUTE_HOST_RTC
auto
make_im2col_tme_copy(CopyOp                   const& copy_op,
                     Tensor<GEngine, GLayout> const& gtensor,
                     SLayout                  const& slayout,
                     CTATiler                 const& cta_tiler,
                     LowerPadding             const& lower_padding_whd,
                     UpperPadding             const& upper_padding_whd,
                     Stride                   const& stride_whd,
                     Dilation                 const& dilation_whd,
                     WeightPosSRT             const& srt)
{
  // FProp: gtensor shape ((w, h, d, n), c)

  auto cta_v_tile = make_identity_layout(product_each(shape(gtensor))).compose(cta_tiler);

  using TMEType = conditional_t<is_same<void, TmeInternalType>::value,
                                typename GEngine::value_type,
                                TmeInternalType>;

  return detail::make_tme_copy_im2col<TMEType>(copy_op,
                                               gtensor,
                                               slayout,
                                               cta_v_tile,
                                               lower_padding_whd,
                                               upper_padding_whd,
                                               stride_whd,
                                               dilation_whd,
                                               srt);
}

template <
  class TmeInternalType = void,
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout,
  class LowerPadding,
  class UpperPadding,
  class Stride,
  class Dilation,
  class WeightPosSRT
>
MUTE_HOST_RTC
auto
make_im2col_tme_copy(CopyOp                   const& copy_op,
                     Tensor<GEngine, GLayout> const& gtensor,
                     SLayout                  const& slayout,
                     LowerPadding             const& lower_padding_whd,
                     UpperPadding             const& upper_padding_whd,
                     Stride                   const& stride_whd,
                     Dilation                 const& dilation_whd,
                     WeightPosSRT             const& srt)
{
  return make_im2col_tme_copy(copy_op,
                              gtensor,
                              slayout,
                              product_each(shape(slayout)),
                              lower_padding_whd,
                              upper_padding_whd,
                              stride_whd,
                              dilation_whd,
                              srt);
}

template <
  class TmeInternalType = void,
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout,
  class CTATiler
>
MUTE_HOST_RTC
auto
make_im2col_tme_copy(CopyOp                   const& copy_op,
                     Tensor<GEngine, GLayout> const& gtensor,
                     SLayout                  const& slayout,
                     CTATiler                 const& cta_tiler)
{
  // FProp: gtensor shape ((w, h, d, n), c)
  auto cta_v_tile = make_identity_layout(product_each(shape(gtensor))).compose(cta_tiler);

  using TMEType = conditional_t<is_same<void, TmeInternalType>::value,
                                typename GEngine::value_type,
                                TmeInternalType>;
  return detail::make_tme_copy_im2col(copy_op, gtensor, slayout, cta_tiler);
}

template <
  class TmeInternalType = void,
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout
>
MUTE_HOST_RTC
auto
make_im2col_tme_copy(CopyOp                   const& copy_op,
                     Tensor<GEngine, GLayout> const& gtensor,
                     SLayout                  const& slayout)
{
  return make_im2col_tme_copy(copy_op, gtensor, slayout, product_each(shape(slayout)));
}

} // namespace mute
